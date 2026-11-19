"""Flask backend for real-time sitting posture monitoring and voice feedback."""

import base64
import json
import logging
import os
import queue
import threading
import time

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, redirect, request, send_from_directory, url_for
from flask_cors import CORS

from anomaly_detector import AnomalyDetector
from pose_estimator import PoseEstimator
from speaker.ip_speaker import send_tts
from yolo_detector import YOLODetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

MODEL_PATH = os.environ.get("YOLO_POSE_MODEL", "yolov8n-pose.pt")
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://192.168.3.242:8554/live")

detector = YOLODetector(
    model_path=os.environ.get("YOLO_DET_MODEL", "yolov8n.pt"),
    device="auto",
    conf_thres=0.7,
    iou_thres=0.45,
    gamma=1.0,
    advanced_box_filter=True,
)
pose_estimator = PoseEstimator(model_path=MODEL_PATH)
anomaly_detector = AnomalyDetector()

rtsp_frame_queue = queue.Queue(maxsize=1)
rtsp_result_queue = queue.Queue(maxsize=1)
rtsp_thread_running = False
rtsp_thread = None


def scale_keypoints(kp_normalized, shape):
    """Convert normalized (0..1) keypoints to pixel coordinates."""
    h, w, _ = shape
    return [(x * w, y * h) for x, y in kp_normalized]


def parse_flat_keypoints(flat_list, min_conf=0.5):
    """Parse COCO 17-keypoint flat list [x1, y1, c1, ...] into (x, y) tuples."""
    keypoints = []
    for i in range(0, len(flat_list), 3):
        x = flat_list[i]
        y = flat_list[i + 1]
        conf = flat_list[i + 2]
        if conf >= min_conf:
            keypoints.append((x, y))
        else:
            keypoints.append((None, None))
    return keypoints


def evaluate_keypoints(kp_abs):
    """Compute posture angles and classify posture status against baseline or defaults."""
    angles = anomaly_detector._compute_angles(kp_abs)
    if anomaly_detector.has_baseline():
        is_bad = anomaly_detector.is_deviated_from_baseline(kp_abs)
    else:
        is_bad = anomaly_detector.is_fall_like(kp_abs)
    return ("bad" if is_bad else "good"), angles


def run_posture_detection(frame):
    """Run person detection, pose estimation, and posture evaluation on a BGR frame."""
    boxes, _ = detector.detect(frame)
    if not boxes:
        return frame, "no_person", {}

    kp_norm, annotated_frame = pose_estimator.get_pose(frame)
    if not kp_norm:
        return annotated_frame, "no_person", {}

    kp_abs = scale_keypoints(kp_norm, frame.shape)
    posture_status, angles = evaluate_keypoints(kp_abs)
    return annotated_frame, posture_status, angles


def get_access_token(api_key, secret_key):
    """Fetch Baidu OAuth access token."""
    if not api_key or not secret_key:
        return None
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    }
    try:
        response = requests.post(
            url, params=params, proxies={"http": None, "https": None}, timeout=5.0
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as exc:
        logger.warning("Failed to fetch Baidu access token: %s", exc)
        return None


def fallback_advice(posture_status):
    """Provide deterministic local advice when the LLM endpoint is unavailable."""
    if posture_status == "bad":
        return "检测到不良坐姿，请抬头挺胸、放松双肩并保持背部直立。"
    return "当前坐姿良好，请继续保持并注意定时起身活动。"


def generate_posture_advice(token, posture_status, angles):
    """Generate personalized posture correction advice via Baidu Wenxin API with fallback."""
    if token is None:
        return fallback_advice(posture_status)

    llm_url = (
        "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/"
        f"ernie-4.0-turbo-128k?access_token={token}"
    )
    user_prompt = (
        "你是一位关心办公人员和学生脊柱健康的助手。请根据以下坐姿数据，"
        "给出一条简短实用的改进建议（30字以内，不要播报具体度数）：\n"
        f"姿势状态：{posture_status}\n"
        f"颈部角度：{angles.get('neck', '--')}度\n"
        f"左侧身体角度：{angles.get('left_body', '--')}度\n"
        f"右侧身体角度：{angles.get('right_body', '--')}度\n"
        f"平均身体角度：{angles.get('avg_body', '--')}度"
    )
    payload = json.dumps(
        {
            "messages": [{"role": "user", "content": user_prompt}],
            "penalty_score": 1,
            "enable_system_memory": False,
            "disable_search": True,
            "enable_citation": False,
            "enable_trace": False,
        },
        ensure_ascii=False,
    )
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            llm_url,
            headers=headers,
            data=payload.encode("utf-8"),
            proxies={"http": None, "https": None},
            timeout=5.0,
        )
        result = response.json()
        if "result" in result:
            return result["result"].strip()
    except Exception as exc:
        logger.warning("LLM request failed, using fallback advice: %s", exc)

    return fallback_advice(posture_status)


def get_posture_report(angles, posture_status, speak=True):
    """Generate posture advice text and optionally trigger LAN IP speaker TTS."""
    api_key = os.environ.get("BAIDU_API_KEY", "")
    secret_key = os.environ.get("BAIDU_SECRET_KEY", "")
    token = get_access_token(api_key, secret_key)
    advice = generate_posture_advice(token, posture_status, angles)

    if speak and advice:
        send_tts(advice)

    return {"text": advice, "ai_advice": advice}


@app.route("/capture_baseline", methods=["POST"])
def capture_baseline():
    """Capture a frame from the active RTSP queue and store it as the ideal baseline."""
    if not rtsp_thread_running:
        return jsonify({"success": False, "error": "RTSP stream not running"}), 400

    for _ in range(5):
        if rtsp_frame_queue.empty():
            time.sleep(0.5)
            continue

        frame = rtsp_frame_queue.queue[0]
        kp_norm, annotated_frame = pose_estimator.get_pose(frame)
        kp_abs = scale_keypoints(kp_norm, frame.shape)
        if len(kp_abs) < 15:
            time.sleep(0.5)
            continue

        anomaly_detector.set_baseline(kp_abs)
        baseline_angles = anomaly_detector._compute_angles(kp_abs)
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        return jsonify(
            {
                "success": True,
                "baseline": kp_abs,
                "angles": baseline_angles,
                "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            }
        )

    return jsonify({"success": False, "error": "Failed to capture valid baseline frame"}), 400


@app.route("/detect_posture", methods=["POST"])
def detect_posture():
    """Evaluate a base64-encoded image submitted via HTTP POST."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No 'image' field in request"}), 400

    try:
        _, b64_data = data["image"].split(",", 1)
    except ValueError:
        return jsonify({"error": "Invalid image format"}), 400

    img_bytes = base64.b64decode(b64_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Invalid image data"}), 400

    annotated_frame, posture_status, angles = run_posture_detection(frame)
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    report_data = {"text": ""}
    if posture_status == "bad":
        report_data = get_posture_report(angles, posture_status)

    return jsonify(
        {
            "posture": posture_status,
            "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
            "angles": angles,
            "report": report_data.get("text", ""),
        }
    )


@app.route("/detect_keypoints", methods=["POST"])
def detect_keypoints():
    """Evaluate 17 COCO keypoints JSON produced by the AirBox edge inference node."""
    data = request.get_json()
    if not data or "keypoints" not in data:
        return jsonify({"error": "Missing 'keypoints' field"}), 400

    raw_kpts = data["keypoints"]
    if len(raw_kpts) < 51 or len(raw_kpts) % 3 != 0:
        return jsonify({"error": "Expected flat list of 17x3 COCO keypoints"}), 400

    parsed = parse_flat_keypoints(raw_kpts, min_conf=0.3)
    valid_pts = [(x or 0.0, y or 0.0) for x, y in parsed]
    posture_status, angles = evaluate_keypoints(valid_pts)
    report_data = get_posture_report(
        angles, posture_status, speak=(posture_status == "bad")
    )

    return jsonify(
        {
            "posture": posture_status,
            "angles": angles,
            "advice": report_data.get("text", ""),
            "timestamp": time.time(),
        }
    )


def rtsp_detection_thread():
    """Background worker pulling RTSP frames via dual-queue sliding-window buffering."""
    global rtsp_thread_running
    last_posture_status = None
    last_advice_time = 0.0
    continuous_bad_time = 0.0

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|timeout;15000000"
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        logger.error("Unable to open RTSP stream: %s", RTSP_URL)
        rtsp_thread_running = False
        return

    try:
        while rtsp_thread_running:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.release()
                time.sleep(1.0)
                cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                last_posture_status = None
                continuous_bad_time = 0.0
                continue

            annotated_frame, posture_status, angles = run_posture_detection(frame)
            advice_text = None

            if posture_status == "bad":
                if last_posture_status == "bad":
                    continuous_bad_time += time.time() - loop_start
                else:
                    continuous_bad_time = 0.0

                if (
                    last_posture_status != "bad"
                    or continuous_bad_time > 60
                    or (loop_start - last_advice_time > 180)
                ):
                    advice_text = get_posture_report(angles, posture_status).get("text")
                    last_advice_time = loop_start
                    continuous_bad_time = 0.0
            else:
                continuous_bad_time = 0.0

            last_posture_status = posture_status

            _, buffer = cv2.imencode(".jpg", annotated_frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")
            result_dict = {
                "image": f"data:image/jpeg;base64,{img_b64}",
                "posture_status": posture_status,
                "angles": angles,
            }
            if advice_text:
                result_dict["advice"] = advice_text

            if rtsp_result_queue.full():
                rtsp_result_queue.get()
            rtsp_result_queue.put(result_dict)

            if rtsp_frame_queue.full():
                rtsp_frame_queue.get()
            rtsp_frame_queue.put(frame)

            time.sleep(0.01)
    finally:
        cap.release()


@app.route("/rtsp/start", methods=["POST"])
def start_rtsp():
    """Start RTSP background monitoring thread."""
    global rtsp_thread, rtsp_thread_running, RTSP_URL
    if rtsp_thread_running and rtsp_thread and rtsp_thread.is_alive():
        return jsonify({"success": True, "message": "RTSP detection already running"})

    data = request.get_json(silent=True) or {}
    if "rtsp_url" in data:
        RTSP_URL = data["rtsp_url"]

    rtsp_thread_running = True
    rtsp_thread = threading.Thread(target=rtsp_detection_thread, daemon=True)
    rtsp_thread.start()
    return jsonify({"success": True, "message": "RTSP detection started"})


@app.route("/rtsp/stop", methods=["POST"])
def stop_rtsp():
    """Stop RTSP monitoring thread and clear frame/result queues."""
    global rtsp_thread_running
    if rtsp_thread_running:
        rtsp_thread_running = False
        if rtsp_thread:
            rtsp_thread.join(timeout=5.0)
        while not rtsp_frame_queue.empty():
            rtsp_frame_queue.get()
        while not rtsp_result_queue.empty():
            rtsp_result_queue.get()
        return jsonify({"success": True, "message": "RTSP detection stopped"})
    return jsonify({"success": True, "message": "RTSP detection not running"})


@app.route("/rtsp/status", methods=["GET"])
def rtsp_status():
    """Return current RTSP worker status."""
    return jsonify(
        {"running": bool(rtsp_thread_running and rtsp_thread and rtsp_thread.is_alive())}
    )


@app.route("/rtsp/latest", methods=["GET"])
def rtsp_latest():
    """Return latest processed frame and posture metrics."""
    if not rtsp_thread_running:
        return jsonify(
            {
                "success": False,
                "message": "RTSP detection is not running",
                "error_code": "not_running",
            }
        )
    if rtsp_result_queue.empty():
        return jsonify(
            {
                "success": False,
                "message": "No RTSP detection results available",
                "error_code": "no_results",
            }
        )
    return jsonify({"success": True, "result": rtsp_result_queue.queue[0]})


@app.route("/rtsp/report", methods=["GET"])
def rtsp_report():
    """Generate a posture report for the latest RTSP frame."""
    if rtsp_result_queue.empty():
        return jsonify({"success": False, "message": "No RTSP detection results available"})

    result = dict(rtsp_result_queue.queue[0])
    posture_status = result.get("posture_status")
    angles = result.get("angles", {})

    if posture_status == "bad":
        result["advice"] = get_posture_report(angles, posture_status).get("text", "")
    elif posture_status in ("unknown", "no_person"):
        result["advice"] = "未检测到人物，无法生成姿势报告"

    return jsonify({"success": True, "result": result})


@app.route("/get_advice", methods=["GET"])
def get_advice():
    """Return latest posture advice (from RTSP stream if active, or latest fallback)."""
    if not rtsp_result_queue.empty():
        latest = rtsp_result_queue.queue[0]
        angles = latest.get("angles", {})
        posture_status = latest.get("posture_status", "good")
    else:
        angles = {"neck": 90.0, "left_body": 95.0, "right_body": 95.0, "avg_body": 95.0}
        posture_status = "good"

    report_data = get_posture_report(angles, posture_status, speak=(posture_status == "bad"))
    advice = report_data.get("text", "")

    if request.args.get("format", "json") == "text":
        return advice

    return jsonify(
        {
            "advice": advice,
            "posture": posture_status,
            "angles": angles,
            "timestamp": time.time(),
        }
    )


@app.route("/")
def serve_index():
    return redirect(url_for("serve_advice"))


@app.route("/advice")
def serve_advice():
    return send_from_directory(app.static_folder, "advice.html")


@app.route("/static/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    os.makedirs(os.path.join(app.static_folder, "audio"), exist_ok=True)
    app.run(host=host, port=port, debug=False)
