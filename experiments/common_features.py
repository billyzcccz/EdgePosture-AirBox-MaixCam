"""Scale-invariant geometric feature extraction and baseline posture rules."""

import math
import cv2
import numpy as np


FRONT_CLASSES = [
    "upright",
    "leaning_forward",
    "leaning_backward",
    "leaning_left",
    "leaning_right",
]

FEATURE_NAMES = [
    "nose_sh_dx_norm",
    "nose_sh_dy_norm",
    "eye_sh_dy_norm",
    "ear_sh_dy_norm",
    "ear_sh_dx_asym",
    "shoulder_tilt_deg",
    "head_roll_deg",
    "head_yaw_asym",
    "sh_width_img_ratio",
    "eye_sh_ratio",
    "nose_eye_dy_norm",
    "left_ear_sh_dist_norm",
    "right_ear_sh_dist_norm",
    "ear_dist_sh_ratio",
    "mean_upper_conf",
    "hip_visible",
]


def imread_unicode(path: str):
    """Read image safely on Windows paths containing non-ASCII characters."""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _angle_deg_3pt(a, b, c):
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float64)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float64)
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
    cos_val = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cos_val))


def extract_front_features(kpts_17x3: np.ndarray, img_w: int, img_h: int):
    """
    Extract 16 scale- and translation-invariant geometric features from COCO 17x3 keypoints.

    Parameters
    ----------
    kpts_17x3 : np.ndarray of shape (17, 3) -> (x, y, conf) in pixel coordinates.
    img_w, img_h : image dimensions.

    Returns
    -------
    features : np.ndarray of shape (16,), or None if core upper-body joints are missing.
    """
    if kpts_17x3 is None or kpts_17x3.shape[0] < 13:
        return None

    nose = kpts_17x3[0]
    l_eye, r_eye = kpts_17x3[1], kpts_17x3[2]
    l_ear, r_ear = kpts_17x3[3], kpts_17x3[4]
    l_sh, r_sh = kpts_17x3[5], kpts_17x3[6]
    l_hip, r_hip = kpts_17x3[11], kpts_17x3[12]

    sh_ok = (l_sh[2] > 0.15) and (r_sh[2] > 0.15)
    eye_ok = (l_eye[2] > 0.15) and (r_eye[2] > 0.15)
    nose_ok = nose[2] > 0.15

    if not (sh_ok or eye_ok):
        return None

    sh_width = float(np.hypot(r_sh[0] - l_sh[0], r_sh[1] - l_sh[1])) if sh_ok else 0.0
    eye_dist = float(np.hypot(r_eye[0] - l_eye[0], r_eye[1] - l_eye[1])) if eye_ok else 0.0

    # Fallback scale estimation when shoulders are cropped out in close-up frames
    scale = sh_width if sh_width > 12.0 else (eye_dist * 3.2 if eye_dist > 5.0 else img_w * 0.35)
    scale = max(scale, 12.0)

    if sh_ok:
        sh_cx = 0.5 * (l_sh[0] + r_sh[0])
        sh_cy = 0.5 * (l_sh[1] + r_sh[1])
        sh_tilt = math.degrees(math.atan2(l_sh[1] - r_sh[1], l_sh[0] - r_sh[0]))
        # Normalize tilt around horizontal (0 deg)
        if sh_tilt > 90:
            sh_tilt -= 180
        elif sh_tilt < -90:
            sh_tilt += 180
    else:
        sh_cx = float(nose[0]) if nose_ok else 0.5 * (l_eye[0] + r_eye[0])
        sh_cy = float(img_h * 0.92)
        sh_tilt = 0.0

    if eye_ok:
        eye_cx = 0.5 * (l_eye[0] + r_eye[0])
        eye_cy = 0.5 * (l_eye[1] + r_eye[1])
        head_roll = math.degrees(math.atan2(l_eye[1] - r_eye[1], l_eye[0] - r_eye[0]))
        if head_roll > 90:
            head_roll -= 180
        elif head_roll < -90:
            head_roll += 180
    else:
        eye_cx, eye_cy = float(nose[0]), float(nose[1])
        head_roll = 0.0

    ref_x = float(nose[0]) if nose_ok else eye_cx
    ref_y = float(nose[1]) if nose_ok else eye_cy

    nose_sh_dx = (ref_x - sh_cx) / scale
    nose_sh_dy = (sh_cy - ref_y) / scale
    eye_sh_dy = (sh_cy - eye_cy) / scale

    ear_pts = [p for p in (l_ear, r_ear) if p[2] > 0.15]
    if ear_pts:
        ear_cy = float(np.mean([p[1] for p in ear_pts]))
        ear_sh_dy = (sh_cy - ear_cy) / scale
    else:
        ear_sh_dy = eye_sh_dy

    if l_ear[2] > 0.15 and r_ear[2] > 0.15:
        ear_dist = float(np.hypot(r_ear[0] - l_ear[0], r_ear[1] - l_ear[1]))
        ear_dist_ratio = ear_dist / scale
        ear_sh_dx_asym = ((l_ear[0] - l_sh[0]) + (r_ear[0] - r_sh[0])) / (2.0 * scale)
    else:
        ear_dist_ratio = (eye_dist * 1.8) / scale if eye_ok else 0.4
        ear_sh_dx_asym = nose_sh_dx

    if nose_ok and eye_ok and eye_dist > 2.0:
        d_le = float(np.hypot(nose[0] - l_eye[0], nose[1] - l_eye[1]))
        d_re = float(np.hypot(nose[0] - r_eye[0], nose[1] - r_eye[1]))
        head_yaw_asym = (d_le - d_re) / eye_dist
    else:
        head_yaw_asym = 0.0

    sh_width_img_ratio = sh_width / max(float(img_w), 1.0)
    eye_sh_ratio = eye_dist / scale if eye_ok else 0.25
    nose_eye_dy = ((ref_y - eye_cy) / scale) if (nose_ok and eye_ok) else 0.05

    l_ear_sh = (
        float(np.hypot(l_ear[0] - l_sh[0], l_ear[1] - l_sh[1])) / scale
        if (l_ear[2] > 0.15 and l_sh[2] > 0.15)
        else ear_sh_dy
    )
    r_ear_sh = (
        float(np.hypot(r_ear[0] - r_sh[0], r_ear[1] - r_sh[1])) / scale
        if (r_ear[2] > 0.15 and r_sh[2] > 0.15)
        else ear_sh_dy
    )

    mean_upper_conf = float(np.mean(kpts_17x3[:7, 2]))
    hip_visible = 1.0 if (l_hip[2] > 0.3 or r_hip[2] > 0.3) else 0.0

    return np.array(
        [
            nose_sh_dx,
            nose_sh_dy,
            eye_sh_dy,
            ear_sh_dy,
            ear_sh_dx_asym,
            sh_tilt,
            head_roll,
            head_yaw_asym,
            sh_width_img_ratio,
            eye_sh_ratio,
            nose_eye_dy,
            l_ear_sh,
            r_ear_sh,
            ear_dist_ratio,
            mean_upper_conf,
            hip_visible,
        ],
        dtype=np.float32,
    )


def baseline_rule_predict_front(kpts_17x3: np.ndarray) -> str:
    """
    Legacy baseline posture rule from the initial competition prototype
    (unnormalized nose-shoulder angles + hip-knee fallback).
    Returns 'upright' or 'bad' (binary; cannot reliably separate 5 classes).
    """
    if kpts_17x3 is None or kpts_17x3.shape[0] < 15:
        return "bad"

    l_sh, r_sh = kpts_17x3[5], kpts_17x3[6]
    l_hip, r_hip = kpts_17x3[11], kpts_17x3[12]
    l_knee, r_knee = kpts_17x3[13], kpts_17x3[14]
    nose = kpts_17x3[0]

    # Original AnomalyDetector.is_fall_like required hips and knees visible
    if min(l_hip[2], r_hip[2], l_knee[2], r_knee[2]) < 0.3:
        return "bad"

    left_body = _angle_deg_3pt(l_sh[:2], l_hip[:2], l_knee[:2])
    right_body = _angle_deg_3pt(r_sh[:2], r_hip[:2], r_knee[:2])
    avg_body = 0.5 * (left_body + right_body)
    neck_angle = _angle_deg_3pt(nose[:2], l_sh[:2], r_sh[:2])

    if avg_body < 55.0 or neck_angle < 50.0:
        return "bad"
    return "upright"


def extract_side_features(kpts_17x3: np.ndarray, img_w: int, img_h: int):
    """
    Extract lateral sitting posture geometric features (ear, shoulder, hip).
    Automatically picks left or right body side with higher joint confidence.
    """
    if kpts_17x3 is None or kpts_17x3.shape[0] < 13:
        return None

    left_conf = float(kpts_17x3[3, 2] + kpts_17x3[5, 2] + kpts_17x3[11, 2])
    right_conf = float(kpts_17x3[4, 2] + kpts_17x3[6, 2] + kpts_17x3[12, 2])

    if left_conf >= right_conf:
        ear, sh, hip = kpts_17x3[3], kpts_17x3[5], kpts_17x3[11]
    else:
        ear, sh, hip = kpts_17x3[4], kpts_17x3[6], kpts_17x3[12]

    if ear[2] < 0.1 or sh[2] < 0.1 or hip[2] < 0.1:
        return None

    torso_len = float(np.hypot(sh[0] - hip[0], sh[1] - hip[1]))
    torso_len = max(torso_len, 15.0)

    # Ear-Shoulder-Hip angle (180 deg = straight neck/torso alignment)
    esh_angle = _angle_deg_3pt(ear[:2], sh[:2], hip[:2])

    # Craniovertebral proxy: angle of vector (shoulder -> ear) from vertical
    dx_se = abs(float(ear[0] - sh[0]))
    dy_se = float(sh[1] - ear[1])
    neck_fwd_deg = math.degrees(math.atan2(dx_se, max(dy_se, 1e-3)))

    # Trunk inclination from vertical
    dx_hs = abs(float(sh[0] - hip[0]))
    dy_hs = float(hip[1] - sh[1])
    trunk_tilt_deg = math.degrees(math.atan2(dx_hs, max(dy_hs, 1e-3)))

    ear_sh_len_norm = float(np.hypot(ear[0] - sh[0], ear[1] - sh[1])) / torso_len
    ear_sh_dx_norm = dx_se / torso_len
    sh_hip_dx_norm = dx_hs / torso_len
    mean_side_conf = float((ear[2] + sh[2] + hip[2]) / 3.0)

    return np.array(
        [
            esh_angle,
            neck_fwd_deg,
            trunk_tilt_deg,
            ear_sh_len_norm,
            ear_sh_dx_norm,
            sh_hip_dx_norm,
            torso_len / max(float(img_h), 1.0),
            mean_side_conf,
        ],
        dtype=np.float32,
    )
