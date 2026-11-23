"""Experiment 1: Front-end pose estimation & quantization benchmark.

Evaluates YOLOv8n-pose (PyTorch FP32, ONNX FP32, ONNX Static INT8) and MediaPipe Pose
across front-view (4,813 images) and side-view (655 images / 273 unique sources) datasets.
"""

import argparse
import collections
import glob
import json
import os
import time

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from ultralytics import YOLO

from common_features import FRONT_CLASSES, imread_unicode


DEFAULT_FRONT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "front_view_classification")
DEFAULT_SIDE_DIR = os.path.join(os.path.dirname(__file__), "datasets", "side_view_keypoints")
DEFAULT_PT_MODEL = "yolov8n-pose.pt"


def scan_front_dataset(front_dir: str):
    """Merge original broken Roboflow splits into a unified list of records."""
    records = []
    for split in ("train", "valid", "test"):
        sp = os.path.join(front_dir, split)
        if not os.path.isdir(sp):
            continue
        for cls_name in sorted(os.listdir(sp)):
            cp = os.path.join(sp, cls_name)
            if not os.path.isdir(cp) or cls_name not in FRONT_CLASSES:
                continue
            for fn in sorted(os.listdir(cp)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    records.append(
                        {
                            "path": os.path.join(cp, fn),
                            "label": cls_name,
                            "label_id": FRONT_CLASSES.index(cls_name),
                            "orig_split": split,
                            "filename": fn,
                        }
                    )
    return records


def scan_side_dataset(side_dir: str):
    """Scan YOLOv8-pose formatted side-view dataset (4 keypoints: hip, shoulder, ear, mid-back)."""
    records = []
    seen_bases = set()
    for split in ("train", "valid", "test"):
        img_dir = os.path.join(side_dir, split, "images")
        lbl_dir = os.path.join(side_dir, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        for fn in sorted(os.listdir(img_dir)):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            ip = os.path.join(img_dir, fn)
            lp = os.path.join(lbl_dir, os.path.splitext(fn)[0] + ".txt")
            if not os.path.isfile(lp):
                continue
            parts = open(lp, "r", encoding="utf-8").read().strip().split()
            if len(parts) < 5 + 4 * 3:
                continue
            cls_id = int(parts[0])
            kpts_norm = np.array([float(x) for x in parts[5 : 5 + 12]], dtype=np.float32).reshape(4, 3)
            base_id = fn.split(".rf.")[0]
            is_first_unique = base_id not in seen_bases
            seen_bases.add(base_id)
            records.append(
                {
                    "path": ip,
                    "label_id": cls_id,  # 0=Bad, 1=Good
                    "label": "Good" if cls_id == 1 else "Bad",
                    "orig_split": split,
                    "filename": fn,
                    "base_id": base_id,
                    "is_unique_source": is_first_unique,
                    "gt_kpts_norm": kpts_norm,
                }
            )
    return records


def letterbox_640(img_bgr: np.ndarray, new_shape=(640, 640)):
    """Resize and pad image with stride=32 letterbox matching Ultralytics YOLOv8."""
    h, w = img_bgr.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    if (w, h) != new_unpad:
        img_bgr = cv2.resize(img_bgr, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(blob[None, ...]), r, (dw, dh)


def decode_yolov8_pose_output(
    out: np.ndarray, ratio: float, pad: tuple, orig_w: int, orig_h: int, conf_thresh: float = 0.25
):
    """Decode [1, 56, 8400] YOLOv8-pose ONNX output into highest-confidence person (17, 3)."""
    preds = out[0].T  # (8400, 56): cx, cy, w, h, conf, 17*3
    mask = preds[:, 4] > conf_thresh
    preds = preds[mask]
    if len(preds) == 0:
        return None

    best_idx = int(np.argmax(preds[:, 4]))
    best = preds[best_idx]
    kpts = best[5:].reshape(17, 3).copy()
    dw, dh = pad
    kpts[:, 0] = np.clip((kpts[:, 0] - dw) / ratio, 0, orig_w - 1)
    kpts[:, 1] = np.clip((kpts[:, 1] - dh) / ratio, 0, orig_h - 1)
    return kpts.astype(np.float32)


class PostureCalibrationReader(CalibrationDataReader):
    """Calibration data reader using real posture images for static INT8 quantization."""

    def __init__(self, image_paths, input_name: str):
        self.image_paths = list(image_paths)
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        while self.idx < len(self.image_paths):
            path = self.image_paths[self.idx]
            self.idx += 1
            img = imread_unicode(path)
            if img is None:
                continue
            blob, _, _ = letterbox_640(img)
            return {self.input_name: blob}
        return None


def prepare_onnx_models(pt_path: str, cache_dir: str, calib_paths):
    """Export YOLOv8n-pose ONNX FP32 and create static INT8 QDQ quantized ONNX model."""
    os.makedirs(cache_dir, exist_ok=True)
    fp32_onnx = os.path.join(cache_dir, "yolov8n-pose.onnx")
    int8_onnx = os.path.join(cache_dir, "yolov8n-pose-int8-static.onnx")

    if not os.path.isfile(fp32_onnx):
        model = YOLO(pt_path)
        exported = model.export(format="onnx", opset=17, imgsz=640, simplify=True)
        if os.path.abspath(exported) != os.path.abspath(fp32_onnx):
            import shutil

            shutil.move(exported, fp32_onnx)

    if not os.path.isfile(int8_onnx):
        sess = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        reader = PostureCalibrationReader(calib_paths, input_name)
        quantize_static(
            model_input=fp32_onnx,
            model_output=int8_onnx,
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    return fp32_onnx, int8_onnx


def run_yolo_pt(model: YOLO, records, device: str):
    """Extract 17x3 keypoints using Ultralytics PyTorch YOLOv8n-pose."""
    kpts_all = []
    img_sizes = []
    latencies = []
    for rec in records:
        img = imread_unicode(rec["path"])
        h, w = img.shape[:2]
        img_sizes.append((w, h))
        t0 = time.perf_counter()
        res = model.predict(img, device=device, imgsz=640, conf=0.25, verbose=False)[0]
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if res.keypoints is None or len(res.keypoints) == 0:
            kpts_all.append(np.zeros((17, 3), dtype=np.float32))
            continue
        boxes_conf = res.boxes.conf.cpu().numpy()
        best_i = int(np.argmax(boxes_conf))
        xy = res.keypoints.xy[best_i].cpu().numpy()
        kc = res.keypoints.conf[best_i].cpu().numpy() if res.keypoints.conf is not None else np.ones(17)
        kpts_all.append(np.column_stack([xy, kc]).astype(np.float32))
    return np.array(kpts_all, dtype=np.float32), np.array(img_sizes, dtype=np.int32), latencies


def run_onnx_cpu(onnx_path: str, records):
    """Extract 17x3 keypoints using ONNX Runtime CPU execution provider."""
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_opts, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    kpts_all = []
    latencies = []
    for rec in records:
        img = imread_unicode(rec["path"])
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        blob, ratio, pad = letterbox_640(img)
        out = sess.run(None, {input_name: blob})[0]
        kpts = decode_yolov8_pose_output(out, ratio, pad, w, h, conf_thresh=0.25)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        if kpts is None:
            kpts_all.append(np.zeros((17, 3), dtype=np.float32))
        else:
            kpts_all.append(kpts)
    return np.array(kpts_all, dtype=np.float32), latencies


def run_mediapipe(records):
    """Extract COCO-mapped 17x3 keypoints using MediaPipe Pose on CPU."""
    # Map MediaPipe 33 landmarks -> COCO 17 indices
    mp_to_coco = {
        0: 0,   # nose
        1: 2,   # left_eye
        2: 5,   # right_eye
        3: 7,   # left_ear
        4: 8,   # right_ear
        5: 11,  # left_shoulder
        6: 12,  # right_shoulder
        7: 13,  # left_elbow
        8: 14,  # right_elbow
        9: 15,  # left_wrist
        10: 16, # right_wrist
        11: 23, # left_hip
        12: 24, # right_hip
        13: 25, # left_knee
        14: 26, # right_knee
        15: 27, # left_ankle
        16: 28, # right_ankle
    }
    kpts_all = []
    latencies = []
    with mp.solutions.pose.Pose(
        static_image_mode=True, model_complexity=1, min_detection_confidence=0.25
    ) as pose:
        for rec in records:
            img = imread_unicode(rec["path"])
            h, w = img.shape[:2]
            t0 = time.perf_counter()
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            arr = np.zeros((17, 3), dtype=np.float32)
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                for coco_idx, mp_idx in mp_to_coco.items():
                    lm = lms[mp_idx]
                    arr[coco_idx] = [lm.x * w, lm.y * h, lm.visibility]
            kpts_all.append(arr)
    return np.array(kpts_all, dtype=np.float32), latencies


def evaluate_side_localization(kpts_all: np.ndarray, side_records, unique_only=True):
    """
    Evaluate predicted keypoints (hip, shoulder, ear) against Roboflow ground-truth
    4-keypoint labels (k0=hip, k1=shoulder, k2=ear). Normalized by torso length.
    """
    errors_hip, errors_sh, errors_ear = [], [], []
    detected_cnt = 0
    total_cnt = 0

    for idx, rec in enumerate(side_records):
        if unique_only and not rec["is_unique_source"]:
            continue
        total_cnt += 1
        img = imread_unicode(rec["path"])
        h, w = img.shape[:2]
        gt = rec["gt_kpts_norm"]  # (4, 3) normalized [0..1]
        gt_hip = np.array([gt[0, 0] * w, gt[0, 1] * h])
        gt_sh = np.array([gt[1, 0] * w, gt[1, 1] * h])
        gt_ear = np.array([gt[2, 0] * w, gt[2, 1] * h])
        torso = max(float(np.linalg.norm(gt_sh - gt_hip)), 20.0)

        pred = kpts_all[idx]
        if np.max(pred[:, 2]) < 0.15:
            continue
        detected_cnt += 1

        # Match either left or right side whichever is closer to GT shoulder+ear
        l_err = np.linalg.norm(pred[5, :2] - gt_sh) + np.linalg.norm(pred[3, :2] - gt_ear)
        r_err = np.linalg.norm(pred[6, :2] - gt_sh) + np.linalg.norm(pred[4, :2] - gt_ear)
        use_left = (pred[5, 2] > 0.1 and pred[6, 2] <= 0.1) or (
            not (pred[6, 2] > 0.1 and pred[5, 2] <= 0.1) and l_err <= r_err
        )
        p_ear = pred[3] if use_left else pred[4]
        p_sh = pred[5] if use_left else pred[6]
        p_hip = pred[11] if use_left else pred[12]

        if p_hip[2] > 0.15:
            errors_hip.append(float(np.linalg.norm(p_hip[:2] - gt_hip) / torso))
        if p_sh[2] > 0.15:
            errors_sh.append(float(np.linalg.norm(p_sh[:2] - gt_sh) / torso))
        if p_ear[2] > 0.15:
            errors_ear.append(float(np.linalg.norm(p_ear[:2] - gt_ear) / torso))

    all_errs = errors_hip + errors_sh + errors_ear
    return {
        "total_images": total_cnt,
        "detection_rate": round(detected_cnt / max(total_cnt, 1) * 100.0, 2),
        "nme_hip": round(float(np.mean(errors_hip)) if errors_hip else 1.0, 4),
        "nme_shoulder": round(float(np.mean(errors_sh)) if errors_sh else 1.0, 4),
        "nme_ear": round(float(np.mean(errors_ear)) if errors_ear else 1.0, 4),
        "nme_mean": round(float(np.mean(all_errs)) if all_errs else 1.0, 4),
        "pck_015": round(float(np.mean(np.array(all_errs) <= 0.15) * 100.0) if all_errs else 0.0, 2),
        "pck_020": round(float(np.mean(np.array(all_errs) <= 0.20) * 100.0) if all_errs else 0.0, 2),
    }


def summarize_front_detection(kpts_all: np.ndarray):
    """Compute detection rate and upper-body/lower-body visibility statistics."""
    n = len(kpts_all)
    det = int(np.sum(np.max(kpts_all[:, :, 2], axis=1) > 0.15))
    upper_ok = int(
        np.sum(
            (kpts_all[:, 0, 2] > 0.15)
            & (kpts_all[:, 5, 2] > 0.15)
            & (kpts_all[:, 6, 2] > 0.15)
        )
    )
    hip_ok = int(np.sum((kpts_all[:, 11, 2] > 0.3) & (kpts_all[:, 12, 2] > 0.3)))
    knee_ok = int(np.sum((kpts_all[:, 13, 2] > 0.3) & (kpts_all[:, 14, 2] > 0.3)))
    return {
        "total_images": n,
        "person_detection_rate_pct": round(det / max(n, 1) * 100.0, 2),
        "upper_body_complete_pct": round(upper_ok / max(n, 1) * 100.0, 2),
        "hip_visible_pct": round(hip_ok / max(n, 1) * 100.0, 2),
        "knee_visible_pct": round(knee_ok / max(n, 1) * 100.0, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-dir", default=DEFAULT_FRONT_DIR)
    parser.add_argument("--side-dir", default=DEFAULT_SIDE_DIR)
    parser.add_argument("--pt-model", default=DEFAULT_PT_MODEL)
    parser.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "cache"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    parser.add_argument("--cpu-bench-n", type=int, default=300, help="Images for CPU FPS/INT8 benchmark on front view")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    print("[1/5] Scanning datasets...")
    front_records = scan_front_dataset(args.front_dir)
    side_records = scan_side_dataset(args.side_dir)
    print(f"Front-view total: {len(front_records)} images")
    print(
        f"Side-view total: {len(side_records)} images "
        f"({sum(1 for r in side_records if r['is_unique_source'])} unique sources)"
    )

    # Save dataset metadata for Exp 2
    with open(os.path.join(args.cache_dir, "front_records.json"), "w", encoding="utf-8") as f:
        json.dump(front_records, f, ensure_ascii=False)
    side_meta = [
        {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
        for r in side_records
    ]
    with open(os.path.join(args.cache_dir, "side_records.json"), "w", encoding="utf-8") as f:
        json.dump(side_meta, f, ensure_ascii=False)

    # Calibration subset: 20 per class from front + 40 from side = 140 real images
    rng = np.random.default_rng(42)
    calib_paths = []
    by_cls = collections.defaultdict(list)
    for r in front_records:
        by_cls[r["label"]].append(r["path"])
    for cls_name in FRONT_CLASSES:
        chosen = rng.choice(by_cls[cls_name], size=min(20, len(by_cls[cls_name])), replace=False)
        calib_paths.extend(chosen.tolist())
    side_unique_paths = [r["path"] for r in side_records if r["is_unique_source"]]
    calib_paths.extend(rng.choice(side_unique_paths, size=min(40, len(side_unique_paths)), replace=False).tolist())

    print("[2/5] Preparing ONNX FP32 & Static INT8 models...")
    fp32_onnx, int8_onnx = prepare_onnx_models(args.pt_model, args.cache_dir, calib_paths)
    print(
        f"Model sizes: PT={os.path.getsize(args.pt_model)/1e6:.2f} MB, "
        f"ONNX-FP32={os.path.getsize(fp32_onnx)/1e6:.2f} MB, "
        f"ONNX-INT8={os.path.getsize(int8_onnx)/1e6:.2f} MB"
    )

    # Stratified subset for CPU speed / INT8 comparison on front view
    front_sub_indices = []
    per_cls_n = max(1, args.cpu_bench_n // len(FRONT_CLASSES))
    for cls_name in FRONT_CLASSES:
        idxs = [i for i, r in enumerate(front_records) if r["label"] == cls_name]
        pick = rng.choice(idxs, size=min(per_cls_n, len(idxs)), replace=False)
        front_sub_indices.extend(sorted(pick.tolist()))
    front_sub_records = [front_records[i] for i in front_sub_indices]

    print("[3/5] Running YOLOv8n-pose (PyTorch FP32) on full Front & Side datasets...")
    yolo_model = YOLO(args.pt_model)
    import torch
    dev = "0" if torch.cuda.is_available() else "cpu"
    f_kpts_yolo, f_sizes, f_lat_gpu = run_yolo_pt(yolo_model, front_records, dev)
    s_kpts_yolo, s_sizes, s_lat_gpu = run_yolo_pt(yolo_model, side_records, dev)
    np.savez_compressed(
        os.path.join(args.cache_dir, "kpts_yolo_fp32.npz"),
        front_kpts=f_kpts_yolo,
        front_sizes=f_sizes,
        side_kpts=s_kpts_yolo,
        side_sizes=s_sizes,
        front_sub_indices=np.array(front_sub_indices, dtype=np.int32),
    )

    print("[4/5] Running MediaPipe Pose on full Front & Side datasets...")
    f_kpts_mp, f_lat_mp = run_mediapipe(front_records)
    s_kpts_mp, s_lat_mp = run_mediapipe(side_records)
    np.savez_compressed(
        os.path.join(args.cache_dir, "kpts_mediapipe.npz"),
        front_kpts=f_kpts_mp,
        side_kpts=s_kpts_mp,
    )

    print("[5/5] Running ONNX FP32 & ONNX Static INT8 on CPU...")
    fsub_kpts_onnx_fp32, fsub_lat_fp32 = run_onnx_cpu(fp32_onnx, front_sub_records)
    s_kpts_onnx_fp32, s_lat_fp32 = run_onnx_cpu(fp32_onnx, side_records)

    fsub_kpts_onnx_int8, fsub_lat_int8 = run_onnx_cpu(int8_onnx, front_sub_records)
    s_kpts_onnx_int8, s_lat_int8 = run_onnx_cpu(int8_onnx, side_records)

    np.savez_compressed(
        os.path.join(args.cache_dir, "kpts_onnx_bench.npz"),
        front_sub_indices=np.array(front_sub_indices, dtype=np.int32),
        front_sub_fp32=fsub_kpts_onnx_fp32,
        front_sub_int8=fsub_kpts_onnx_int8,
        side_fp32=s_kpts_onnx_fp32,
        side_int8=s_kpts_onnx_int8,
    )

    summary = {
        "model_sizes_mb": {
            "yolov8n_pt_fp32": round(os.path.getsize(args.pt_model) / 1e6, 2),
            "yolov8n_onnx_fp32": round(os.path.getsize(fp32_onnx) / 1e6, 2),
            "yolov8n_onnx_int8_static": round(os.path.getsize(int8_onnx) / 1e6, 2),
        },
        "front_detection": {
            "yolov8n_pt_fp32": summarize_front_detection(f_kpts_yolo),
            "mediapipe_pose": summarize_front_detection(f_kpts_mp),
            "yolov8n_onnx_fp32_sub": summarize_front_detection(fsub_kpts_onnx_fp32),
            "yolov8n_onnx_int8_sub": summarize_front_detection(fsub_kpts_onnx_int8),
        },
        "side_localization_unique": {
            "yolov8n_pt_fp32": evaluate_side_localization(s_kpts_yolo, side_records, unique_only=True),
            "yolov8n_onnx_fp32": evaluate_side_localization(s_kpts_onnx_fp32, side_records, unique_only=True),
            "yolov8n_onnx_int8": evaluate_side_localization(s_kpts_onnx_int8, side_records, unique_only=True),
            "mediapipe_pose": evaluate_side_localization(s_kpts_mp, side_records, unique_only=True),
        },
        "side_localization_noisy_all655": {
            "yolov8n_pt_fp32": evaluate_side_localization(s_kpts_yolo, side_records, unique_only=False),
            "yolov8n_onnx_int8": evaluate_side_localization(s_kpts_onnx_int8, side_records, unique_only=False),
            "mediapipe_pose": evaluate_side_localization(s_kpts_mp, side_records, unique_only=False),
        },
        "latency_ms_cpu_warmup_skipped": {
            "mediapipe_pose_front_ms": round(float(np.mean(f_lat_mp[20:])), 2),
            "onnx_fp32_front_ms": round(float(np.mean(fsub_lat_fp32[20:])), 2),
            "onnx_int8_front_ms": round(float(np.mean(fsub_lat_int8[20:])), 2),
            "mediapipe_pose_side_ms": round(float(np.mean(s_lat_mp[20:])), 2),
            "onnx_fp32_side_ms": round(float(np.mean(s_lat_fp32[20:])), 2),
            "onnx_int8_side_ms": round(float(np.mean(s_lat_int8[20:])), 2),
            "pytorch_gpu_front_ms": round(float(np.mean(f_lat_gpu[20:])), 2),
        },
    }

    out_json = os.path.join(args.out_dir, "exp1_frontend_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 1 Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved results to {out_json}")


if __name__ == "__main__":
    main()
