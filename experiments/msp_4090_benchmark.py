"""RTX 4090 Benchmark Suite for MSP Multi-View 7-Class Sitting Posture Dataset.

Evaluates:
1. Full-body YOLOv8n-pose detection & completeness across all 5,600 frames (8 views x 7 classes).
2. ONNX Static INT8 Collapse vs Mixed_DecodeFP32 (3.93 MB) & Mixed_HeadFP32 (6.96 MB)
   on a balanced 1,120-frame multi-view subset (detection, keypoint drift vs FP32, latency).
3. 7-class classification (upright, back, cheek, down, left, right, leg) under:
   (a) Block-Temporal split (with buffer gap, 3,360 -> 1,120)
   (b) Pure Cross-View Leave-One-View-Out on view1..view7 (4,900 frames)
   (c) Cross-Subject & Cross-Room (view1..view7 <-> view8)
   comparing: End-to-end RGB YOLOv8n-cls vs RawXY vs FullBody_NormGeo vs GT 3D Hkeypoints.
4. Downstream 7-class accuracy preservation under Mixed_DecodeFP32 (3.93 MB) INT8.
"""

import argparse
import json
import math
import os
import shutil
import time

import cv2
import numpy as np
import onnxruntime as ort
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from ultralytics import YOLO


def letterbox_pad(img_bgr: np.ndarray, new_shape=(640, 640)):
    h, w = img_bgr.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (new_shape[1] - nw) / 2.0, (new_shape[0] - nh) / 2.0
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return padded, r, (dw, dh)


def decode_yolov8_pose_raw(preds: np.ndarray, ratio: float, pad, orig_shape, conf_thres=0.25):
    p = np.squeeze(preds, axis=0).T
    scores = p[:, 4]
    mask = scores >= conf_thres
    if not np.any(mask):
        return None
    best = p[mask][int(np.argmax(scores[mask]))]
    dw, dh = pad
    oh, ow = orig_shape
    kpts_raw = best[5:].reshape(17, 3).copy()
    kpts_raw[:, 0] = np.clip((kpts_raw[:, 0] - dw) / ratio, 0, ow)
    kpts_raw[:, 1] = np.clip((kpts_raw[:, 1] - dh) / ratio, 0, oh)
    return kpts_raw.astype(np.float32)


def run_onnx_pose(onnx_path: str, img_paths, conf_thres=0.25):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(onnx_path, sess_options=so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    n = len(img_paths)
    out = np.zeros((n, 17, 3), dtype=np.float32)
    t_list = []
    for i, path in enumerate(img_paths):
        img = cv2.imread(path)
        padded, ratio, pad = letterbox_pad(img, (640, 640))
        blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        t0 = time.perf_counter()
        preds = sess.run(None, {in_name: blob})[0]
        t_list.append((time.perf_counter() - t0) * 1000.0)
        kpts = decode_yolov8_pose_raw(preds, ratio, pad, img.shape[:2], conf_thres=conf_thres)
        if kpts is not None:
            out[i] = kpts
    return out, float(np.mean(t_list)), float(np.median(t_list))


def run_yolo_gpu_pose(pt_path: str, img_paths, batch=64, device=0):
    model = YOLO(pt_path)
    n = len(img_paths)
    out = np.zeros((n, 17, 3), dtype=np.float32)
    for start in range(0, n, batch):
        chunk = img_paths[start : start + batch]
        results = model.predict(chunk, verbose=False, conf=0.25, imgsz=640, device=device)
        for j, res in enumerate(results):
            if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
                continue
            boxes = res.boxes
            best = int(np.argmax(boxes.conf.cpu().numpy())) if (boxes is not None and len(boxes) > 1) else 0
            kp = res.keypoints.data[best].cpu().numpy()
            if kp.shape == (17, 3):
                out[start + j] = kp
    return out


def detection_stats(kpts_arr: np.ndarray):
    n = len(kpts_arr)
    det = np.any(kpts_arr[:, :, 2] > 0.15, axis=1)
    upper = (kpts_arr[:, 5, 2] > 0.15) & (kpts_arr[:, 6, 2] > 0.15) & (kpts_arr[:, 0, 2] > 0.15)
    hips = (kpts_arr[:, 11, 2] > 0.15) & (kpts_arr[:, 12, 2] > 0.15)
    knees = (kpts_arr[:, 13, 2] > 0.15) & (kpts_arr[:, 14, 2] > 0.15)
    ankles = (kpts_arr[:, 15, 2] > 0.15) & (kpts_arr[:, 16, 2] > 0.15)
    return {
        "n": int(n),
        "person_det_pct": round(float(np.mean(det) * 100), 2),
        "upper_ok_pct": round(float(np.mean(upper) * 100), 2),
        "torso_hip_ok_pct": round(float(np.mean(upper & hips) * 100), 2),
        "fullbody_ok_pct": round(float(np.mean(upper & hips & knees & ankles) * 100), 2),
        "mean_conf": round(float(np.mean(kpts_arr[det, :, 2])) if np.any(det) else 0.0, 4),
    }


def pose_agreement_vs_fp32(pred_kpts: np.ndarray, ref_kpts: np.ndarray):
    ref_ok = (ref_kpts[:, 5, 2] > 0.15) & (ref_kpts[:, 6, 2] > 0.15) & (ref_kpts[:, 11, 2] > 0.15) & (ref_kpts[:, 12, 2] > 0.15)
    pred_det = np.any(pred_kpts[:, :, 2] > 0.15, axis=1)
    valid = ref_ok & pred_det
    if not np.any(valid):
        return {"matched_frames": 0, "nme_vs_fp32": 1.0, "pck_005_pct": 0.0, "pck_010_pct": 0.0}
    sh_c = 0.5 * (ref_kpts[valid, 5, :2] + ref_kpts[valid, 6, :2])
    hip_c = 0.5 * (ref_kpts[valid, 11, :2] + ref_kpts[valid, 12, :2])
    torso = np.maximum(np.linalg.norm(sh_c - hip_c, axis=1), 16.0)
    dists = np.linalg.norm(pred_kpts[valid, :, :2] - ref_kpts[valid, :, :2], axis=2) / torso[:, None]
    joint_mask = ref_kpts[valid, :, 2] > 0.15
    vals = dists[joint_mask]
    return {
        "matched_frames": int(np.sum(valid)),
        "nme_vs_fp32": round(float(np.mean(vals)), 4),
        "pck_005_pct": round(float(np.mean(vals <= 0.05) * 100), 2),
        "pck_010_pct": round(float(np.mean(vals <= 0.10) * 100), 2),
    }


def extract_fullbody_normgeo(kpts_arr: np.ndarray, img_w=400.0, img_h=400.0):
    """Extract 24 view-robust, scale- & translation-invariant full-body features."""
    n = len(kpts_arr)
    feats = np.full((n, 24), np.nan, dtype=np.float32)
    for i in range(n):
        k = kpts_arr[i]
        if not np.any(k[:, 2] > 0.15):
            continue
        nose = k[0]
        l_eye, r_eye = k[1], k[2]
        l_ear, r_ear = k[3], k[4]
        l_sh, r_sh = k[5], k[6]
        l_elb, r_elb = k[7], k[8]
        l_wri, r_wri = k[9], k[10]
        l_hip, r_hip = k[11], k[12]
        l_kne, r_kne = k[13], k[14]
        l_ank, r_ank = k[15], k[16]

        sh_ok = (l_sh[2] > 0.15) and (r_sh[2] > 0.15)
        hip_ok = (l_hip[2] > 0.15) and (r_hip[2] > 0.15)
        eye_ok = (l_eye[2] > 0.15) and (r_eye[2] > 0.15)

        if sh_ok:
            sh_cx, sh_cy = 0.5 * (l_sh[0] + r_sh[0]), 0.5 * (l_sh[1] + r_sh[1])
            sh_w = float(np.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1]))
            sh_tilt = math.degrees(math.atan2(l_sh[1] - r_sh[1], l_sh[0] - r_sh[0]))
            if sh_tilt > 90:
                sh_tilt -= 180
            elif sh_tilt < -90:
                sh_tilt += 180
        elif l_sh[2] > 0.15 or r_sh[2] > 0.15:
            pt = l_sh if l_sh[2] >= r_sh[2] else r_sh
            sh_cx, sh_cy = float(pt[0]), float(pt[1])
            sh_w, sh_tilt = 0.0, 0.0
        else:
            continue

        if hip_ok:
            hip_cx, hip_cy = 0.5 * (l_hip[0] + r_hip[0]), 0.5 * (l_hip[1] + r_hip[1])
            torso_len = float(np.hypot(sh_cx - hip_cx, sh_cy - hip_cy))
        elif l_hip[2] > 0.15 or r_hip[2] > 0.15:
            pt = l_hip if l_hip[2] >= r_hip[2] else r_hip
            hip_cx, hip_cy = float(pt[0]), float(pt[1])
            torso_len = float(np.hypot(sh_cx - hip_cx, sh_cy - hip_cy))
        else:
            hip_cx, hip_cy = sh_cx, min(img_h, sh_cy + max(sh_w * 1.4, 60.0))
            torso_len = max(sh_w * 1.4, 60.0)

        # Torso height remains stable even in 90-deg side views where sh_w collapses
        scale = max(torso_len, sh_w * 1.35, 16.0)

        torso_dx = (sh_cx - hip_cx) / scale
        torso_dy = (sh_cy - hip_cy) / scale
        torso_pitch = math.degrees(math.atan2(sh_cx - hip_cx, max(1e-3, hip_cy - sh_cy)))

        nose_ok = nose[2] > 0.15
        nose_sh_dx = (nose[0] - sh_cx) / scale if nose_ok else 0.0
        nose_sh_dy = (nose[1] - sh_cy) / scale if nose_ok else -0.45
        nose_hip_dx = (nose[0] - hip_cx) / scale if nose_ok else torso_dx
        nose_hip_dy = (nose[1] - hip_cy) / scale if nose_ok else -1.45

        head_roll = 0.0
        if eye_ok:
            hr = math.degrees(math.atan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0]))
            head_roll = hr - 180 if hr > 90 else (hr + 180 if hr < -90 else hr)

        if eye_ok and nose_ok:
            eye_cx = 0.5 * (l_eye[0] + r_eye[0])
            eye_d = max(float(np.hypot(r_eye[0] - l_eye[0], r_eye[1] - l_eye[1])), 3.0)
            head_yaw = (nose[0] - eye_cx) / eye_d
        else:
            head_yaw = 0.0

        ears_valid = [p for p in (l_ear, r_ear) if p[2] > 0.15]
        if ears_valid:
            ear_dy = np.mean([(p[1] - sh_cy) / scale for p in ears_valid])
            ear_dx_signed = np.mean([(p[0] - sh_cx) / scale for p in ears_valid])
        else:
            ear_dy, ear_dx_signed = nose_sh_dy, nose_sh_dx

        # Cheek-resting signature: wrist proximity to ear / nose / face height
        head_pts = [p for p in (nose, l_ear, r_ear, l_eye, r_eye) if p[2] > 0.15]
        wrists = [p for p in (l_wri, r_wri) if p[2] > 0.15]
        if wrists and head_pts:
            min_wh = min(float(np.hypot(w[0] - h[0], w[1] - h[1])) / scale for w in wrists for h in head_pts)
        else:
            min_wh = 2.0
        if wrists and nose_ok:
            min_wn = min(float(np.hypot(w[0] - nose[0], w[1] - nose[1])) / scale for w in wrists)
        else:
            min_wn = 2.0

        lw_sh_dy = (l_wri[1] - sh_cy) / scale if l_wri[2] > 0.15 else 0.8
        rw_sh_dy = (r_wri[1] - sh_cy) / scale if r_wri[2] > 0.15 else 0.8
        min_w_sh_dy = min(lw_sh_dy, rw_sh_dy)
        w_asym_dy = abs(lw_sh_dy - rw_sh_dy) if (l_wri[2] > 0.15 and r_wri[2] > 0.15) else 0.0

        # Elbow spread/elevation
        elbs = [p for p in (l_elb, r_elb) if p[2] > 0.15]
        elb_sh_dy = float(np.mean([(e[1] - sh_cy) / scale for e in elbs])) if elbs else 0.6

        # Crossed-leg signature
        knees_ok = (l_kne[2] > 0.15) and (r_kne[2] > 0.15)
        ankles_ok = (l_ank[2] > 0.15) and (r_ank[2] > 0.15)
        knee_dx = abs(l_kne[0] - r_kne[0]) / scale if knees_ok else 0.4
        knee_dy = abs(l_kne[1] - r_kne[1]) / scale if knees_ok else 0.0
        ankle_dx = abs(l_ank[0] - r_ank[0]) / scale if ankles_ok else 0.4
        ankle_dy = abs(l_ank[1] - r_ank[1]) / scale if ankles_ok else 0.0

        cross_dists = []
        if l_ank[2] > 0.15 and r_kne[2] > 0.15:
            cross_dists.append(float(np.hypot(l_ank[0] - r_kne[0], l_ank[1] - r_kne[1])) / scale)
        if r_ank[2] > 0.15 and l_kne[2] > 0.15:
            cross_dists.append(float(np.hypot(r_ank[0] - l_kne[0], r_ank[1] - l_kne[1])) / scale)
        if l_ank[2] > 0.15 and r_ank[2] > 0.15:
            cross_dists.append(float(np.hypot(l_ank[0] - r_ank[0], l_ank[1] - r_ank[1])) / scale)
        min_leg_cross = min(cross_dists) if cross_dists else 1.0

        mean_upper_conf = float(np.mean(k[:11, 2]))
        mean_lower_conf = float(np.mean(k[11:, 2]))

        feats[i] = [
            torso_dx, torso_dy, torso_pitch,
            nose_sh_dx, nose_sh_dy, nose_hip_dx, nose_hip_dy,
            sh_w / scale, sh_tilt, head_roll, head_yaw,
            ear_dy, ear_dx_signed,
            min_wh, min_wn, min_w_sh_dy, w_asym_dy, elb_sh_dy,
            knee_dx, knee_dy, ankle_dx, ankle_dy, min_leg_cross,
            mean_upper_conf - mean_lower_conf,
        ]
    return feats


def extract_rawxy(kpts_arr: np.ndarray, img_w=400.0, img_h=400.0):
    raw = kpts_arr.copy()
    raw[:, :, 0] /= img_w
    raw[:, :, 1] /= img_h
    return raw.reshape(len(kpts_arr), -1).astype(np.float32)


def extract_gt_h3d(h_kpts: np.ndarray):
    """Pelvis-centered, torso-length-normalized 3D ground-truth Hkeypoints (51 dims)."""
    hk = np.nan_to_num(h_kpts.copy(), nan=0.0)
    pelvis = hk[:, 0:1, :]
    centered = hk - pelvis
    neck = hk[:, 8, :]
    torso = np.maximum(np.linalg.norm(neck - hk[:, 0, :], axis=1), 1e-3)
    normed = centered / torso[:, None, None]
    return normed.reshape(len(hk), -1).astype(np.float32)


def make_models():
    return {
        "LogReg": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1500, C=1.0, random_state=42)),
        ]),
        "SVM-RBF": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", SVC(C=4.0, kernel="rbf", gamma="scale", random_state=42)),
        ]),
        "RandomForest": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=18, random_state=42, n_jobs=-1)),
        ]),
        "MLP": Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42)),
        ]),
    }


def eval_splits_for_features(X, y, view_ids, seq_pos):
    res = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    tr_blk = seq_pos <= 59
    te_blk = seq_pos >= 80
    v1_7_mask = view_ids <= 7
    v8_mask = view_ids == 8

    for mname, factory in [
        ("LogReg", lambda: make_models()["LogReg"]),
        ("SVM-RBF", lambda: make_models()["SVM-RBF"]),
        ("RandomForest", lambda: make_models()["RandomForest"]),
        ("MLP", lambda: make_models()["MLP"]),
    ]:
        # 1) 5-fold CV
        accs, f1s = [], []
        for tr, va in skf.split(X, y):
            m = factory()
            m.fit(X[tr], y[tr])
            p = m.predict(X[va])
            accs.append(accuracy_score(y[va], p) * 100)
            f1s.append(f1_score(y[va], p, average="macro") * 100)

        # 2) Block-Temporal (0..59 -> 80..99, 20% buffer gap)
        m_blk = factory()
        m_blk.fit(X[tr_blk], y[tr_blk])
        p_blk = m_blk.predict(X[te_blk])

        # 3) Pure Cross-View LOVO across view1..view7 (same subject/env, unseen angle)
        lovo_accs, lovo_f1s = [], []
        per_view = {}
        for hv in range(1, 8):
            tr_v = (view_ids <= 7) & (view_ids != hv)
            te_v = view_ids == hv
            mv = factory()
            mv.fit(X[tr_v], y[tr_v])
            pv = mv.predict(X[te_v])
            va_acc = round(accuracy_score(y[te_v], pv) * 100, 2)
            va_f1 = round(f1_score(y[te_v], pv, average="macro") * 100, 2)
            per_view[f"view{hv}"] = va_acc
            lovo_accs.append(va_acc)
            lovo_f1s.append(va_f1)

        # 4) Cross-Subject & Room: v1..7 -> v8 and v8 -> v1..7
        m_ab = factory()
        m_ab.fit(X[v1_7_mask], y[v1_7_mask])
        p_ab = m_ab.predict(X[v8_mask])

        m_ba = factory()
        m_ba.fit(X[v8_mask], y[v8_mask])
        p_ba = m_ba.predict(X[v1_7_mask])

        res[mname] = {
            "cv5_acc_mean": round(float(np.mean(accs)), 2),
            "cv5_f1_mean": round(float(np.mean(f1s)), 2),
            "block_temporal_acc": round(accuracy_score(y[te_blk], p_blk) * 100, 2),
            "block_temporal_f1": round(f1_score(y[te_blk], p_blk, average="macro") * 100, 2),
            "cross_view_lovo_acc_mean": round(float(np.mean(lovo_accs)), 2),
            "cross_view_lovo_acc_std": round(float(np.std(lovo_accs)), 2),
            "cross_view_lovo_f1_mean": round(float(np.mean(lovo_f1s)), 2),
            "cross_view_per_view": per_view,
            "v1_7_to_v8_acc": round(accuracy_score(y[v8_mask], p_ab) * 100, 2),
            "v1_7_to_v8_f1": round(f1_score(y[v8_mask], p_ab, average="macro") * 100, 2),
            "v8_to_v1_7_acc": round(accuracy_score(y[v1_7_mask], p_ba) * 100, 2),
            "v8_to_v1_7_f1": round(f1_score(y[v1_7_mask], p_ba, average="macro") * 100, 2),
        }
    return res


def run_yolo_cls_split(base_pt: str, work_dir: str, split_name: str, tr_records, val_records, data_dir: str, classes, epochs=20):
    ds_dir = os.path.join(work_dir, split_name)
    if os.path.isdir(ds_dir):
        shutil.rmtree(ds_dir)
    for sp_name, recs in (("train", tr_records), ("val", val_records)):
        for c in classes:
            os.makedirs(os.path.join(ds_dir, sp_name, c), exist_ok=True)
        for r in recs:
            src = os.path.abspath(os.path.join(data_dir, r["rel_path"]))
            dst_name = f"{r['view']}_{r['frame_idx']:05d}.jpg"
            os.symlink(src, os.path.join(ds_dir, sp_name, r["class"], dst_name))

    model = YOLO(base_pt)
    model.train(
        data=ds_dir,
        epochs=epochs,
        imgsz=224,
        batch=64,
        device=0,
        workers=8,
        project=os.path.join(work_dir, "runs"),
        name=split_name,
        exist_ok=True,
        verbose=False,
    )
    metrics = model.val(data=ds_dir, split="val", imgsz=224, device=0, verbose=False)
    shutil.rmtree(ds_dir, ignore_errors=True)
    return round(float(metrics.top1) * 100, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.path.join(os.path.dirname(__file__), "cache", "msp_subset"))
    parser.add_argument("--weights-dir", default=os.path.join(os.path.dirname(__file__), "cache"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    parser.add_argument("--cls-epochs", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = json.load(open(os.path.join(args.data_dir, "msp_manifest.json"), "r", encoding="utf-8"))
    classes = manifest["classes"]
    records = manifest["records"]
    gt_npz = np.load(os.path.join(args.data_dir, "msp_subset_gt.npz"))
    h_kpts = gt_npz["h_kpts"]
    y = gt_npz["class_ids"]
    view_ids = gt_npz["view_ids"]
    seq_pos = gt_npz["seq_pos"]

    img_paths = [os.path.join(args.data_dir, r["rel_path"]) for r in records]
    print(f"Loaded {len(records)} MSP samples across 8 views and {len(classes)} classes.")

    # 1. GPU YOLOv8n-pose over all 5,600 samples
    kpts_cache = os.path.join(args.out_dir, "msp_yolo_fp32_5600.npz")
    if os.path.isfile(kpts_cache):
        kpts_gpu_5600 = np.load(kpts_cache)["kpts"]
    else:
        print("Running YOLOv8n-pose GPU FP32 on all 5,600 frames...")
        kpts_gpu_5600 = run_yolo_gpu_pose(os.path.join(args.weights_dir, "yolov8n-pose.pt"), img_paths)
        np.savez_compressed(kpts_cache, kpts=kpts_gpu_5600)
    det_all = detection_stats(kpts_gpu_5600)
    per_view_det = {f"view{v}": detection_stats(kpts_gpu_5600[view_ids == v]) for v in range(1, 9)}

    # 2. Balanced 1,120-frame subset (20 per view x class: seq_pos % 5 == 0) for ONNX INT8 ablation
    sub_mask = (seq_pos % 5) == 0
    sub_idx = np.where(sub_mask)[0]
    sub_paths = [img_paths[i] for i in sub_idx]
    sub_y = y[sub_idx]
    sub_views = view_ids[sub_idx]
    print(f"Running ONNX quantization ablation on {len(sub_paths)} multi-view frames...")

    onnx_models = [
        ("ONNX_FP32", os.path.join(args.weights_dir, "yolov8n-pose.onnx")),
        ("ONNX_INT8_Full", os.path.join(args.weights_dir, "yolov8n-pose-int8-static.onnx")),
        ("ONNX_INT8_Mixed_DecodeFP32", os.path.join(args.weights_dir, "yolov8n-pose-int8-mixed-decode.onnx")),
        ("ONNX_INT8_Mixed_HeadFP32", os.path.join(args.weights_dir, "yolov8n-pose-int8-mixed-head.onnx")),
    ]
    onnx_results = {}
    onnx_kpts_map = {}
    for tag, mpath in onnx_models:
        kpts_o, lat_mean, lat_med = run_onnx_pose(mpath, sub_paths)
        onnx_kpts_map[tag] = kpts_o
        onnx_results[tag] = {
            "size_mb": round(os.path.getsize(mpath) / 1e6, 2),
            "cpu_mean_ms": round(lat_mean, 2),
            "cpu_median_ms": round(lat_med, 2),
            "detection": detection_stats(kpts_o),
        }

    ref_onnx = onnx_kpts_map["ONNX_FP32"]
    for tag in onnx_results:
        if tag == "ONNX_FP32":
            continue
        onnx_results[tag]["agreement_vs_fp32"] = pose_agreement_vs_fp32(onnx_kpts_map[tag], ref_onnx)

    # 3. 7-Class Evaluation across Feature Sets & Splits
    print("Evaluating 7-class splits across RawXY, FullBody_NormGeo, and GT 3D Hkeypoints...")
    X_raw = extract_rawxy(kpts_gpu_5600)
    X_geo = extract_fullbody_normgeo(kpts_gpu_5600)
    X_gt3d = extract_gt_h3d(h_kpts)

    cls_eval = {
        "YOLO_RawXY_51d": eval_splits_for_features(X_raw, y, view_ids, seq_pos),
        "YOLO_FullBody_NormGeo_24d": eval_splits_for_features(X_geo, y, view_ids, seq_pos),
        "GT_Hkeypoints_3D_Oracle": eval_splits_for_features(X_gt3d, y, view_ids, seq_pos),
    }

    # 4. Downstream 7-class preservation on 1,120-frame ONNX subset
    quant_downstream = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tag in ("ONNX_FP32", "ONNX_INT8_Mixed_DecodeFP32", "ONNX_INT8_Mixed_HeadFP32"):
        Xq = extract_fullbody_normgeo(onnx_kpts_map[tag])
        quant_downstream[tag] = {}
        for mname in ("SVM-RBF", "RandomForest", "MLP"):
            accs, f1s = [], []
            for tr, va in skf.split(Xq, sub_y):
                clf = make_models()[mname]
                clf.fit(Xq[tr], sub_y[tr])
                p = clf.predict(Xq[va])
                accs.append(accuracy_score(sub_y[va], p) * 100)
                f1s.append(f1_score(sub_y[va], p, average="macro") * 100)
            # Also LOVO on view1..7 inside 1,120 subset
            lovo = []
            for hv in range(1, 8):
                tr_v = (sub_views <= 7) & (sub_views != hv)
                te_v = sub_views == hv
                clf_v = make_models()[mname]
                clf_v.fit(Xq[tr_v], sub_y[tr_v])
                lovo.append(accuracy_score(sub_y[te_v], clf_v.predict(Xq[te_v])) * 100)
            quant_downstream[tag][mname] = {
                "cv5_acc": round(float(np.mean(accs)), 2),
                "cv5_f1": round(float(np.mean(f1s)), 2),
                "cross_view_lovo_acc": round(float(np.mean(lovo)), 2),
            }

    # 5. End-to-End RGB YOLOv8n-cls GPU baselines on 4090
    cls_pt = os.path.join(args.weights_dir, "yolov8n-cls.pt")
    work_dir = os.path.join(args.out_dir, "cls_work")
    os.makedirs(work_dir, exist_ok=True)
    print("Training RTX 4090 YOLOv8n-cls RGB baselines...")

    tr_blk_recs = [r for r in records if r["seq_pos"] <= 59]
    te_blk_recs = [r for r in records if r["seq_pos"] >= 80]
    v1_7_recs = [r for r in records if r["view_id"] <= 7]
    v8_recs = [r for r in records if r["view_id"] == 8]
    cv_tr_recs = [r for r in records if r["view_id"] in (1, 2, 4, 5, 6)]
    cv_v3_recs = [r for r in records if r["view_id"] == 3]
    cv_v7_recs = [r for r in records if r["view_id"] == 7]

    rgb_cls_results = {
        "block_temporal_acc": run_yolo_cls_split(cls_pt, work_dir, "msp_cls_block", tr_blk_recs, te_blk_recs, args.data_dir, classes, args.cls_epochs),
        "cross_view_train12456_test_v3_acc": run_yolo_cls_split(cls_pt, work_dir, "msp_cls_cv_v3", cv_tr_recs, cv_v3_recs, args.data_dir, classes, args.cls_epochs),
        "cross_view_train12456_test_v7_acc": run_yolo_cls_split(cls_pt, work_dir, "msp_cls_cv_v7", cv_tr_recs, cv_v7_recs, args.data_dir, classes, args.cls_epochs),
        "v1_7_to_v8_acc": run_yolo_cls_split(cls_pt, work_dir, "msp_cls_v1_7_to_v8", v1_7_recs, v8_recs, args.data_dir, classes, args.cls_epochs),
        "v8_to_v1_7_acc": run_yolo_cls_split(cls_pt, work_dir, "msp_cls_v8_to_v1_7", v8_recs, v1_7_recs, args.data_dir, classes, args.cls_epochs),
    }

    summary = {
        "dataset": {
            "total_frames": len(records),
            "views": 8,
            "classes": classes,
            "per_view_detection": per_view_det,
            "overall_detection": det_all,
        },
        "onnx_quant_1120": onnx_results,
        "classification_5600": cls_eval,
        "quant_downstream_1120": quant_downstream,
        "rgb_yolov8n_cls_4090": rgb_cls_results,
    }
    out_json = os.path.join(args.out_dir, "msp_benchmark_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved summary to", out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
