"""RTX 4090 Full Training & Ablation Suite.

Runs GPU training and evaluation for:
1. Side-view 4-keypoint spine/posture fine-tuning (YOLOv8n-pose & YOLOv8s-pose)
   on deduplicated unique source frames.
2. Front-view 5-class end-to-end RGB baseline (YOLOv8n-cls) on:
   (a) Stratified 80/20 split across all 4,813 images, and
   (b) Cross-subject split (train on 'extract' 3,780 -> test on unseen 'mb' 899).
"""

import argparse
import json
import os
import shutil

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from ultralytics import YOLO


FRONT_CLASSES = [
    "upright",
    "leaning_forward",
    "leaning_backward",
    "leaning_left",
    "leaning_right",
]


def scan_front_dataset(front_dir: str):
    """Merge Roboflow splits into a unified list of front-view records."""
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
    """Scan YOLOv8-pose side-view dataset and mark unique source frames."""
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
            base_id = fn.split(".rf.")[0]
            is_first_unique = base_id not in seen_bases
            seen_bases.add(base_id)
            records.append(
                {
                    "path": ip,
                    "label_id": cls_id,
                    "label": "Good" if cls_id == 1 else "Bad",
                    "orig_split": split,
                    "filename": fn,
                    "base_id": base_id,
                    "is_unique_source": is_first_unique,
                }
            )
    return records


def prepare_dedup_side_dataset(side_dir: str, work_dir: str):
    """Build a deduplicated (unique source images) YOLO-pose dataset for clean training."""
    records = scan_side_dataset(side_dir)
    unique_recs = [r for r in records if r["is_unique_source"]]
    y = np.array([r["label_id"] for r in unique_recs])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))

    ds_root = os.path.join(work_dir, "side_dedup_pose")
    if os.path.isdir(ds_root):
        shutil.rmtree(ds_root)
    for split, idxs in [("train", tr_idx), ("val", va_idx)]:
        img_out = os.path.join(ds_root, split, "images")
        lbl_out = os.path.join(ds_root, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)
        for i in idxs:
            rec = unique_recs[i]
            src_img = rec["path"]
            src_lbl = (
                src_img.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}")
                .rsplit(".", 1)[0]
                + ".txt"
            )
            shutil.copy2(src_img, os.path.join(img_out, rec["filename"]))
            shutil.copy2(src_lbl, os.path.join(lbl_out, os.path.splitext(rec["filename"])[0] + ".txt"))

    yaml_path = os.path.join(ds_root, "data.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {os.path.abspath(ds_root)}\n"
            "train: train/images\n"
            "val: val/images\n"
            "kpt_shape: [4, 3]\n"
            "flip_idx: [0, 1, 2, 3]\n"
            "nc: 2\n"
            "names: ['Bad', 'Good']\n"
        )
    return yaml_path, len(tr_idx), len(va_idx)


def prepare_stratified_front_cls(front_dir: str, work_dir: str):
    """Merge broken Roboflow splits and create a clean 80/20 stratified classification dataset."""
    records = scan_front_dataset(front_dir)
    y = np.array([r["label_id"] for r in records])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y)), y))

    ds_root = os.path.join(work_dir, "front_stratified_cls")
    if os.path.isdir(ds_root):
        shutil.rmtree(ds_root)
    for split, idxs in [("train", tr_idx), ("val", va_idx)]:
        for i in idxs:
            rec = records[i]
            cls_dir = os.path.join(ds_root, split, rec["label"])
            os.makedirs(cls_dir, exist_ok=True)
            shutil.copy2(rec["path"], os.path.join(cls_dir, rec["filename"]))
    return ds_root, len(tr_idx), len(va_idx)


def prepare_cross_subject_front_cls(front_dir: str, work_dir: str):
    """Create cross-subject split: train on 'extract' (3,780) -> val on 'mb' (899)."""
    records = scan_front_dataset(front_dir)
    ds_root = os.path.join(work_dir, "front_cross_subject_cls")
    if os.path.isdir(ds_root):
        shutil.rmtree(ds_root)

    n_tr, n_va = 0, 0
    for rec in records:
        if rec["filename"].startswith("extract"):
            split = "train"
            n_tr += 1
        elif rec["filename"].startswith("mb"):
            split = "val"
            n_va += 1
        else:
            continue
        cls_dir = os.path.join(ds_root, split, rec["label"])
        os.makedirs(cls_dir, exist_ok=True)
        shutil.copy2(rec["path"], os.path.join(cls_dir, rec["filename"]))
    return ds_root, n_tr, n_va


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-dir", required=True)
    parser.add_argument("--side-dir", required=True)
    parser.add_argument("--pose-weights", default="yolov8n-pose.pt")
    parser.add_argument("--pose-s-weights", default="yolov8s-pose.pt")
    parser.add_argument("--cls-weights", default="yolov8n-cls.pt")
    parser.add_argument("--epochs-pose", type=int, default=120)
    parser.add_argument("--epochs-cls", type=int, default=40)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action="store_true", help="Enable Ultralytics AMP check")
    parser.add_argument("--work-dir", default=os.path.join(os.path.dirname(__file__), "runs"))
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    print("[1/5] Preparing clean deduplicated side-view pose dataset...")
    side_yaml, n_tr_s, n_va_s = prepare_dedup_side_dataset(args.side_dir, args.work_dir)
    print(f"Side-view split: train={n_tr_s}, val={n_va_s}")

    print("[2/5] Fine-tuning YOLOv8n-pose (4-keypoint spine/posture)...")
    pose_n = YOLO(args.pose_weights)
    pose_n.train(
        data=side_yaml,
        epochs=args.epochs_pose,
        imgsz=640,
        batch=args.batch,
        device=args.device,
        amp=args.amp,
        project=args.work_dir,
        name="yolov8n_pose_side4",
        exist_ok=True,
    )
    pose_n_best = YOLO(os.path.join(args.work_dir, "yolov8n_pose_side4", "weights", "best.pt"))
    pose_n_val = pose_n_best.val(data=side_yaml, device=args.device)

    print("[3/5] Fine-tuning YOLOv8s-pose (4-keypoint spine/posture)...")
    pose_s = YOLO(args.pose_s_weights)
    pose_s.train(
        data=side_yaml,
        epochs=args.epochs_pose,
        imgsz=640,
        batch=max(8, args.batch // 2),
        device=args.device,
        amp=args.amp,
        project=args.work_dir,
        name="yolov8s_pose_side4",
        exist_ok=True,
    )
    pose_s_best = YOLO(os.path.join(args.work_dir, "yolov8s_pose_side4", "weights", "best.pt"))
    pose_s_val = pose_s_best.val(data=side_yaml, device=args.device)

    print("[4/5] Training end-to-end RGB baseline (YOLOv8n-cls) on stratified 80/20 split...")
    front_cls_dir, n_tr_f, n_va_f = prepare_stratified_front_cls(args.front_dir, args.work_dir)
    cls_model = YOLO(args.cls_weights)
    cls_model.train(
        data=front_cls_dir,
        epochs=args.epochs_cls,
        imgsz=224,
        batch=args.batch * 2,
        device=args.device,
        amp=args.amp,
        project=args.work_dir,
        name="yolov8n_cls_front5",
        exist_ok=True,
    )
    cls_best = YOLO(os.path.join(args.work_dir, "yolov8n_cls_front5", "weights", "best.pt"))
    cls_val = cls_best.val(data=front_cls_dir, device=args.device)

    print("[5/5] Training YOLOv8n-cls on cross-subject split (extract -> mb)...")
    cross_cls_dir, n_tr_c, n_va_c = prepare_cross_subject_front_cls(args.front_dir, args.work_dir)
    cls_cross = YOLO(args.cls_weights)
    cls_cross.train(
        data=cross_cls_dir,
        epochs=args.epochs_cls,
        imgsz=224,
        batch=args.batch * 2,
        device=args.device,
        amp=args.amp,
        project=args.work_dir,
        name="yolov8n_cls_cross_subject",
        exist_ok=True,
    )
    cls_cross_best = YOLO(os.path.join(args.work_dir, "yolov8n_cls_cross_subject", "weights", "best.pt"))
    cls_cross_val = cls_cross_best.val(data=cross_cls_dir, device=args.device)

    summary = {
        "side_pose_finetune_yolov8n": {
            "train_images": n_tr_s,
            "val_images": n_va_s,
            "box_map50": round(float(pose_n_val.box.map50), 4),
            "box_map50_95": round(float(pose_n_val.box.map), 4),
            "pose_map50": round(float(pose_n_val.pose.map50), 4),
            "pose_map50_95": round(float(pose_n_val.pose.map), 4),
        },
        "side_pose_finetune_yolov8s": {
            "train_images": n_tr_s,
            "val_images": n_va_s,
            "box_map50": round(float(pose_s_val.box.map50), 4),
            "box_map50_95": round(float(pose_s_val.box.map), 4),
            "pose_map50": round(float(pose_s_val.pose.map50), 4),
            "pose_map50_95": round(float(pose_s_val.pose.map), 4),
        },
        "front_rgb_cls_stratified": {
            "train_images": n_tr_f,
            "val_images": n_va_f,
            "top1_acc": round(float(cls_val.top1), 4),
            "top5_acc": round(float(cls_val.top5), 4),
        },
        "front_rgb_cls_cross_subject": {
            "train_images": n_tr_c,
            "val_images": n_va_c,
            "top1_acc": round(float(cls_cross_val.top1), 4),
            "top5_acc": round(float(cls_cross_val.top5), 4),
        },
    }
    out_path = os.path.join(args.work_dir, "train_4090_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== 4090 Training Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
