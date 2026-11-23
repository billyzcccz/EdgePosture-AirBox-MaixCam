"""Experiment 2: Scale-invariant posture classification & ablation study.

Reads cached keypoints from Experiment 1 and evaluates:
1. Legacy rule baseline vs. raw coordinates vs. scale-normalized geometric features
2. MediaPipe Pose vs. YOLOv8n-pose (PyTorch FP32 / ONNX FP32 / ONNX Static INT8)
3. 5-fold Stratified Cross-Validation on 4,813 front-view (5-class) and
   273 unique side-view (2-class) sitting posture samples.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common_features import (
    FRONT_CLASSES,
    baseline_rule_predict_front,
    extract_front_features,
    extract_side_features,
)


def build_front_matrix(kpts_all: np.ndarray, sizes_all: np.ndarray, records, raw_mode: bool = False):
    """Build feature matrix (X, y, valid_mask) for front-view posture samples."""
    feats = []
    labels = []
    valid_indices = []
    for i, rec in enumerate(records):
        k = kpts_all[i]
        w, h = int(sizes_all[i][0]), int(sizes_all[i][1])
        if raw_mode:
            if np.max(k[:7, 2]) < 0.15:
                continue
            # Unnormalized upper-body + hip coordinates (ablation baseline)
            f = k[:13, :2].flatten().astype(np.float32)
        else:
            f = extract_front_features(k, w, h)
            if f is None:
                continue
        feats.append(f)
        labels.append(rec["label_id"])
        valid_indices.append(i)
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(valid_indices)


def build_side_matrix(kpts_all: np.ndarray, sizes_all: np.ndarray, records, unique_only: bool = True):
    """Build feature matrix (X, y, valid_indices) for side-view posture samples."""
    feats = []
    labels = []
    valid_indices = []
    for i, rec in enumerate(records):
        if unique_only and not rec["is_unique_source"]:
            continue
        w, h = int(sizes_all[i][0]), int(sizes_all[i][1])
        f = extract_side_features(kpts_all[i], w, h)
        if f is None:
            continue
        feats.append(f)
        labels.append(rec["label_id"])
        valid_indices.append(i)
    return np.array(feats, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(valid_indices)


def make_classifiers():
    """Return dictionary of lightweight classifiers suitable for edge/host deployment."""
    return {
        "LogReg": Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))]
        ),
        "SVM-RBF": Pipeline(
            [("scaler", StandardScaler()), ("clf", SVC(C=5.0, kernel="rbf", gamma="scale", random_state=42))]
        ),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=600,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def eval_cv(X: np.ndarray, y: np.ndarray, clf_factory, n_splits: int = 5):
    """Run stratified k-fold CV and return overall metrics and out-of-fold predictions."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_pred = np.zeros_like(y)
    fold_accs = []
    fold_f1s = []
    for tr_idx, va_idx in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X[tr_idx], y[tr_idx])
        pred = clf.predict(X[va_idx])
        oof_pred[va_idx] = pred
        fold_accs.append(accuracy_score(y[va_idx], pred))
        fold_f1s.append(f1_score(y[va_idx], pred, average="macro"))

    return {
        "samples": int(len(y)),
        "accuracy_mean": round(float(np.mean(fold_accs) * 100.0), 2),
        "accuracy_std": round(float(np.std(fold_accs) * 100.0), 2),
        "macro_f1_mean": round(float(np.mean(fold_f1s) * 100.0), 2),
        "macro_f1_std": round(float(np.std(fold_f1s) * 100.0), 2),
        "oof_pred": oof_pred,
    }


def eval_legacy_binary_baseline(kpts_all: np.ndarray, records):
    """Evaluate legacy AnomalyDetector binary rule (upright vs non-upright) on front view."""
    y_true = [1 if r["label"] == "upright" else 0 for r in records]
    y_pred = [1 if baseline_rule_predict_front(kpts_all[i]) == "upright" else 0 for i in range(len(records))]
    acc = accuracy_score(y_true, y_pred) * 100.0
    f1 = f1_score(y_true, y_pred, average="macro") * 100.0
    upright_recall = (
        sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1) / max(sum(y_true), 1) * 100.0
    )
    return {
        "samples": len(records),
        "binary_accuracy_pct": round(acc, 2),
        "binary_macro_f1_pct": round(f1, 2),
        "upright_recall_pct": round(upright_recall, 2),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, out_path: str, title: str):
    """Save publication-grade 300 DPI normalized confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=300)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    short_labels = [c.replace("leaning_", "lean_") for c in class_names]
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=short_labels,
        yticklabels=short_labels,
        ylabel="Ground Truth",
        xlabel="Predicted Label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right", rotation_mode="anchor")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm_norm[i, j] * 100.0
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{val:.1f}%\n({cm[i, j]})", ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_ablation_bar(results_dict, out_path: str):
    """Save bar chart comparing feature and front-end combinations."""
    labels = list(results_dict.keys())
    accs = [results_dict[k]["accuracy_mean"] for k in labels]
    f1s = [results_dict[k]["macro_f1_mean"] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=300)
    b1 = ax.bar(x - width / 2, accs, width, label="Accuracy (%)", color="#2b6cb0")
    b2 = ax.bar(x + width / 2, f1s, width, label="Macro-F1 (%)", color="#ed8936")

    ax.set_ylabel("Score (%)")
    ax.set_title("Front-View 5-Class Posture Classification Ablation (5-Fold CV)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=8.5)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "cache"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(os.path.join(args.cache_dir, "front_records.json"), "r", encoding="utf-8") as f:
        front_records = json.load(f)
    with open(os.path.join(args.cache_dir, "side_records.json"), "r", encoding="utf-8") as f:
        side_records = json.load(f)

    yolo_npz = np.load(os.path.join(args.cache_dir, "kpts_yolo_fp32.npz"))
    mp_npz = np.load(os.path.join(args.cache_dir, "kpts_mediapipe.npz"))
    onnx_npz = np.load(os.path.join(args.cache_dir, "kpts_onnx_bench.npz"))

    f_kpts_yolo, f_sizes = yolo_npz["front_kpts"], yolo_npz["front_sizes"]
    s_kpts_yolo, s_sizes = yolo_npz["side_kpts"], yolo_npz["side_sizes"]
    f_kpts_mp, s_kpts_mp = mp_npz["front_kpts"], mp_npz["side_kpts"]

    # 1. Legacy rule baseline (shows hip/knee occlusion failure on upper-body webcam)
    legacy_res = eval_legacy_binary_baseline(f_kpts_yolo, front_records)

    # 2. Front-view 5-class evaluation across front-ends & feature sets
    X_y_raw, y_y_raw, _ = build_front_matrix(f_kpts_yolo, f_sizes, front_records, raw_mode=True)
    X_y_norm, y_y_norm, _ = build_front_matrix(f_kpts_yolo, f_sizes, front_records, raw_mode=False)
    X_mp_norm, y_mp_norm, _ = build_front_matrix(f_kpts_mp, f_sizes, front_records, raw_mode=False)

    front_results = {}
    best_name, best_acc, best_oof, best_y = None, -1.0, None, None

    # Raw pixel ablation (MLP & RF)
    for clf_name in ("SVM-RBF", "RandomForest", "MLP"):
        res = eval_cv(X_y_raw, y_y_raw, lambda n=clf_name: make_classifiers()[n])
        front_results[f"YOLOv8n-FP32_RawXY_{clf_name}"] = {k: v for k, v in res.items() if k != "oof_pred"}

    # Normalized geometric features on MediaPipe
    for clf_name in ("SVM-RBF", "RandomForest", "MLP"):
        res = eval_cv(X_mp_norm, y_mp_norm, lambda n=clf_name: make_classifiers()[n])
        front_results[f"MediaPipe_NormGeo_{clf_name}"] = {k: v for k, v in res.items() if k != "oof_pred"}

    # Normalized geometric features on YOLOv8n-pose FP32
    for clf_name in ("LogReg", "SVM-RBF", "RandomForest", "MLP"):
        res = eval_cv(X_y_norm, y_y_norm, lambda n=clf_name: make_classifiers()[n])
        clean_res = {k: v for k, v in res.items() if k != "oof_pred"}
        front_results[f"YOLOv8n-FP32_NormGeo_{clf_name}"] = clean_res
        if clean_res["accuracy_mean"] > best_acc:
            best_acc = clean_res["accuracy_mean"]
            best_name = f"YOLOv8n-FP32_NormGeo_{clf_name}"
            best_oof = res["oof_pred"]
            best_y = y_y_norm

    # 3. Quantization impact on Front subset (ONNX FP32 vs ONNX INT8)
    sub_idx = onnx_npz["front_sub_indices"].tolist()
    sub_records = [front_records[i] for i in sub_idx]
    sub_sizes = f_sizes[sub_idx]
    X_onnx_fp32, y_onnx_fp32, _ = build_front_matrix(onnx_npz["front_sub_fp32"], sub_sizes, sub_records, raw_mode=False)
    X_onnx_int8, y_onnx_int8, _ = build_front_matrix(onnx_npz["front_sub_int8"], sub_sizes, sub_records, raw_mode=False)

    quant_front_results = {}
    for tag, Xq, yq in [("ONNX_FP32", X_onnx_fp32, y_onnx_fp32), ("ONNX_Static_INT8", X_onnx_int8, y_onnx_int8)]:
        if len(yq) >= 25:
            res = eval_cv(Xq, yq, lambda: make_classifiers()["RandomForest"])
            quant_front_results[f"{tag}_RF"] = {k: v for k, v in res.items() if k != "oof_pred"}

    # 4. Side-view 2-class evaluation (273 deduplicated unique sources)
    side_results = {}
    for tag, s_kpts, uniq in [
        ("YOLOv8n-FP32_Unique273", s_kpts_yolo, True),
        ("MediaPipe_Unique273", s_kpts_mp, True),
        ("ONNX_FP32_Unique273", onnx_npz["side_fp32"], True),
        ("ONNX_Static_INT8_Unique273", onnx_npz["side_int8"], True),
        ("YOLOv8n-FP32_NoisyAll655", s_kpts_yolo, False),
    ]:
        Xs, ys, _ = build_side_matrix(s_kpts, s_sizes, side_records, unique_only=uniq)
        if len(ys) >= 25:
            for clf_name in ("SVM-RBF", "RandomForest"):
                res = eval_cv(Xs, ys, lambda n=clf_name: make_classifiers()[n])
                side_results[f"{tag}_{clf_name}"] = {k: v for k, v in res.items() if k != "oof_pred"}

    # Save Figures
    if best_oof is not None:
        plot_confusion_matrix(
            best_y,
            best_oof,
            FRONT_CLASSES,
            os.path.join(args.out_dir, "fig1_confusion_matrix_front5.png"),
            f"Confusion Matrix ({best_name})",
        )

    bar_subset = {
        "YOLO Raw XY\n(MLP)": front_results["YOLOv8n-FP32_RawXY_MLP"],
        "YOLO Raw XY\n(RF)": front_results["YOLOv8n-FP32_RawXY_RandomForest"],
        "MediaPipe Norm\n(SVM-RBF)": front_results["MediaPipe_NormGeo_SVM-RBF"],
        "MediaPipe Norm\n(RF)": front_results["MediaPipe_NormGeo_RandomForest"],
        "YOLO Norm\n(SVM-RBF)": front_results["YOLOv8n-FP32_NormGeo_SVM-RBF"],
        "YOLO Norm\n(MLP)": front_results["YOLOv8n-FP32_NormGeo_MLP"],
        "YOLO Norm\n(RF)": front_results["YOLOv8n-FP32_NormGeo_RandomForest"],
    }
    plot_ablation_bar(bar_subset, os.path.join(args.out_dir, "fig2_ablation_comparison.png"))

    cls_report = classification_report(best_y, best_oof, target_names=FRONT_CLASSES, output_dict=True) if best_oof is not None else {}
    summary = {
        "legacy_rule_baseline": legacy_res,
        "front_5class_cv": front_results,
        "front_quant_subset_cv": quant_front_results,
        "side_2class_cv": side_results,
        "best_front_model": best_name,
        "best_front_per_class_report": cls_report,
    }
    out_json = os.path.join(args.out_dir, "exp2_classification_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Experiment 2 Summary ===")
    print(json.dumps(summary, indent=2))
    print(f"Saved results and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
