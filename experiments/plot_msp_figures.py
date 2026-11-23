"""Generate publication-quality ablation figures from experiment JSON summaries."""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_fig3_quantization(out_dir: str):
    msp = json.load(open(os.path.join(out_dir, "msp_benchmark_summary.json"), encoding="utf-8"))
    exp3 = json.load(open(os.path.join(out_dir, "exp3_quant_ablation.json"), encoding="utf-8"))

    q_msp = msp["onnx_quant_1120"]
    q_down = msp["quant_downstream_1120"]

    models = ["ONNX_FP32", "ONNX_INT8_Full", "ONNX_INT8_Mixed_DecodeFP32", "ONNX_INT8_Mixed_HeadFP32"]
    labels = ["ONNX FP32\n(13.51 MB)", "INT8 Full\n(3.82 MB)", "INT8 + Decode FP32\n(3.93 MB)", "INT8 + Head FP32\n(6.96 MB)"]

    det_msp = [q_msp[m]["detection"]["person_det_pct"] for m in models]
    det_side = [exp3[m]["side_unique_273"]["detection_rate"] for m in models]
    lat_msp = [q_msp[m]["cpu_mean_ms"] for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8), dpi=300)

    x = np.arange(len(models))
    w = 0.34
    b1 = ax1.bar(x - w / 2, det_msp, w, label="MSP Multi-View (n=1,120)", color="#1f6feb")
    b2 = ax1.bar(x + w / 2, det_side, w, label="Side-View Unique (n=273)", color="#2da44e")
    ax1.set_ylim(0, 126)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Person Detection Rate (%)", fontsize=10)
    ax1.set_title("(a) INT8 Collapse & Decode-FP32 Recovery", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper center", ncol=2, fontsize=8.3, framealpha=0.95)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)

    # Right panel: trade-off on MSP 1,120 (CPU latency + 7-class downstream acc)
    sub_models = ["ONNX_FP32", "ONNX_INT8_Mixed_DecodeFP32", "ONNX_INT8_Mixed_HeadFP32"]
    sub_labels = ["ONNX FP32\n(13.51 MB)", "INT8 + Decode FP32\n(3.93 MB)", "INT8 + Head FP32\n(6.96 MB)"]
    svm_accs = [q_down[m]["SVM-RBF"]["cv5_acc"] for m in sub_models]
    rf_accs = [q_down[m]["RandomForest"]["cv5_acc"] for m in sub_models]
    sub_lats = [q_msp[m]["cpu_mean_ms"] for m in sub_models]

    x2 = np.arange(len(sub_models))
    w2 = 0.25
    b3 = ax2.bar(x2 - w2, svm_accs, w2, label="7-Class SVM-RBF Acc (%)", color="#1f6feb")
    b4 = ax2.bar(x2, rf_accs, w2, label="7-Class RF Acc (%)", color="#8250df")
    ax2_t = ax2.twinx()
    b5 = ax2_t.bar(x2 + w2, sub_lats, w2, label="CPU Latency (ms/img)", color="#f97316", alpha=0.88)

    ax2.set_ylim(0, 115)
    ax2_t.set_ylim(0, 115)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sub_labels, fontsize=8.8)
    ax2.set_ylabel("5-Fold 7-Class Accuracy (%)", fontsize=10)
    ax2_t.set_ylabel("Mean CPU Latency (ms)", fontsize=10, color="#c2410c")
    ax2.set_title("(b) Accuracy Preservation & CPU Speedup (n=1,120)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    for bar in list(b3) + list(b4):
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5)
    for bar in b5:
        h = bar.get_height()
        ax2_t.text(bar.get_x() + bar.get_width() / 2, h + 1.8, f"{h:.1f}ms", ha="center", va="bottom", fontsize=7.5, color="#9a3412")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_t.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.92)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig3_quantization_recovery.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved", out_path)


def plot_fig4_msp_generalization(out_dir: str):
    msp = json.load(open(os.path.join(out_dir, "msp_benchmark_summary.json"), encoding="utf-8"))
    cls56 = msp["classification_5600"]
    rgb = msp["rgb_yolov8n_cls_4090"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=300)

    regimes = [
        "Block-Temporal\n(Same Subj/Env)",
        "Cross-Subject/Env\n(view1-7 -> view8)",
        "Reverse Cross-Subj\n(view8 -> view1-7)",
    ]
    rgb_vals = [rgb["block_temporal_acc"], rgb["v1_7_to_v8_acc"], rgb["v8_to_v1_7_acc"]]
    raw_vals = [
        cls56["YOLO_RawXY_51d"]["SVM-RBF"]["block_temporal_acc"],
        cls56["YOLO_RawXY_51d"]["SVM-RBF"]["v1_7_to_v8_acc"],
        cls56["YOLO_RawXY_51d"]["SVM-RBF"]["v8_to_v1_7_acc"],
    ]
    geo_vals = [
        cls56["YOLO_FullBody_NormGeo_24d"]["SVM-RBF"]["block_temporal_acc"],
        cls56["YOLO_FullBody_NormGeo_24d"]["SVM-RBF"]["v1_7_to_v8_acc"],
        cls56["YOLO_FullBody_NormGeo_24d"]["SVM-RBF"]["v8_to_v1_7_acc"],
    ]
    gt3d_vals = [
        cls56["GT_Hkeypoints_3D_Oracle"]["SVM-RBF"]["block_temporal_acc"],
        cls56["GT_Hkeypoints_3D_Oracle"]["SVM-RBF"]["v1_7_to_v8_acc"],
        cls56["GT_Hkeypoints_3D_Oracle"]["SVM-RBF"]["v8_to_v1_7_acc"],
    ]

    x = np.arange(len(regimes))
    w = 0.19
    bars = [
        ax1.bar(x - 1.5 * w, rgb_vals, w, label="YOLOv8n-cls (RGB)", color="#ef4444"),
        ax1.bar(x - 0.5 * w, raw_vals, w, label="RawXY-51d (SVM-RBF)", color="#f59e0b"),
        ax1.bar(x + 0.5 * w, geo_vals, w, label="FullBody NormGeo-24d (SVM-RBF)", color="#1f6feb"),
        ax1.bar(x + 1.5 * w, gt3d_vals, w, label="GT 3D Skeleton Oracle", color="#2da44e"),
    ]
    ax1.set_ylim(0, 126)
    ax1.set_xticks(x)
    ax1.set_xticklabels(regimes, fontsize=8.8)
    ax1.set_ylabel("7-Class Accuracy (%)", fontsize=10)
    ax1.set_title("(a) MSP 7-Class Cross-Subject & Room Generalization", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.legend(loc="upper center", ncol=2, fontsize=7.8, framealpha=0.95)
    for grp in bars:
        for bar in grp:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.4, f"{h:.1f}", ha="center", va="bottom", fontsize=7.0)

    # Right panel: Per-view LOVO on view1..view7 (exposing pure cross-view bottleneck)
    views = [f"view{i}" for i in range(1, 8)]
    v_raw = [cls56["YOLO_RawXY_51d"]["RandomForest"]["cross_view_per_view"][v] for v in views]
    v_geo = [cls56["YOLO_FullBody_NormGeo_24d"]["RandomForest"]["cross_view_per_view"][v] for v in views]
    v_gt3d = [cls56["GT_Hkeypoints_3D_Oracle"]["LogReg"]["cross_view_per_view"][v] for v in views]

    xv = np.arange(len(views))
    wv = 0.25
    b_r = ax2.bar(xv - wv, v_raw, wv, label=f"RawXY RF (mean {cls56['YOLO_RawXY_51d']['RandomForest']['cross_view_lovo_acc_mean']:.1f}%)", color="#f59e0b")
    b_g = ax2.bar(xv, v_geo, wv, label=f"NormGeo-24d RF (mean {cls56['YOLO_FullBody_NormGeo_24d']['RandomForest']['cross_view_lovo_acc_mean']:.1f}%)", color="#1f6feb")
    b_3 = ax2.bar(xv + wv, v_gt3d, wv, label=f"GT 3D LogReg (mean {cls56['GT_Hkeypoints_3D_Oracle']['LogReg']['cross_view_lovo_acc_mean']:.1f}%)", color="#2da44e")

    ax2.set_ylim(0, 116)
    ax2.set_xticks(xv)
    ax2.set_xticklabels(["View 1", "View 2", "View 3", "View 4", "View 5", "View 6", "View 7"], fontsize=8.5)
    ax2.set_ylabel("Held-Out View Accuracy (%)", fontsize=10)
    ax2.set_title("(b) Pure Cross-View LOVO (Unseen Camera Angles)", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.0, framealpha=0.95)
    for bar in b_g:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2, color="#0d419d", fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig4_msp_generalization_gap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved", out_path)


if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    plot_fig3_quantization(out_dir)
    plot_fig4_msp_generalization(out_dir)
