# Reproducible Experiments & Ablation Suite

Benchmark, quantization diagnosis, ablation, and cross-domain evaluation scripts for **EdgePosture**.

## 1. Datasets Evaluated

1. **Front-View Desktop Sitting Posture** (`front_view_classification`, **4,813** unique images across 5 upper-body classes: `upright`, `leaning_forward`, `leaning_backward`, `leaning_left`, `leaning_right`; `CC BY 4.0`).
   - *Data hygiene note*: The original Roboflow split folders omitted `leaning_forward` from `train/` and `upright` from `valid/`/`test/`. All scripts here merge all splits and perform 5-fold stratified CV, 80/20 stratified splits, or cross-subject (`extract*` 3,780 <-> `mb*` 899) evaluation.
2. **Side-View 4-Keypoint Posture** (`side_view_keypoints`, 655 augmented images deduplicated to **273** unique source frames with `[hip, shoulder, ear, mid-back]` keypoints and `Good`/`Bad` posture labels; `CC BY 4.0`).
3. **Multi-View Full-Body 7-Class Sitting Posture (`MSP`)** (`49,798` total frames across 8 camera views/environments `view1`–`view8` and 7 ergonomic classes: `upright`, `back`, `cheek`, `down`, `left`, `right`, `leg`; uniformly subsampled to **5,600** balanced frames = 100 per view x class with channel-order `BGR->RGB` fix and paired 3D `Hkeypoints`/`Mkeypoints`).

## 2. Script Inventory

| Script | Purpose | Key Output |
| :--- | :--- | :--- |
| `common_features.py` | Scale- & translation-invariant geometric feature extractor + legacy threshold baseline | Shared feature helpers |
| `exp1_frontend_benchmark.py` | `YOLOv8n-pose` vs `MediaPipe` & ONNX FP32/INT8 front-end keypoint localization (`n=4,813` + `n=273`) | `outputs/exp1_frontend_summary.json` |
| `exp2_posture_classification.py` | 5-fold posture classification & rule-baseline ablation (`front_5class` + `side_2class`) | `outputs/exp2_classification_summary.json`, `fig1`, `fig2` |
| `exp3_quant_ablation.py` | Static INT8 collapse reproduction & selective `/model.22/` FP32 exclusion ablation | `outputs/exp3_quant_ablation.json` |
| `exp4_cross_domain_and_quant_cls.py` | Cross-subject (`extract <-> mb`) & mixed-INT8 downstream classification preservation | `outputs/exp4_cross_domain_and_quant.json` |
| `train_4090.py` | RTX 4090 GPU fine-tuning (`YOLOv8n/s-pose` 4-kpt) & end-to-end `YOLOv8n-cls` baselines | `outputs/train_4090_summary.json` |
| `msp_prepare_subset.py` | Uniform subsampling (5,600 frames), `BGR->RGB` fix, and 3D keypoint pairing from `MSP` archives | `cache/msp_subset/` |
| `msp_4090_benchmark.py` | Multi-view full-body benchmark: ONNX INT8 (`n=1,120`), Block-Temporal, Cross-View LOVO, Cross-Subject (`view1-7 <-> view8`), `YOLOv8n-cls`, & 3D GT oracle | `outputs/msp_benchmark_summary.json` |
| `plot_msp_figures.py` | Generates publication figures from JSON outputs | `fig3_quantization_recovery.png`, `fig4_msp_generalization_gap.png` |

## 3. Reproducing the Pipeline

```bash
# Step 1-4: Front-end pose, classification, ONNX quantization, and cross-domain checks
python exp1_frontend_benchmark.py \
  --front-dir /path/to/front_view_classification \
  --side-dir /path/to/side_view_keypoints \
  --pt-model yolov8n-pose.pt
python exp2_posture_classification.py
python exp3_quant_ablation.py
python exp4_cross_domain_and_quant_cls.py

# Step 5: RTX 4090 pose fine-tuning & end-to-end RGB baseline
python train_4090.py \
  --front-dir /path/to/front_view_classification \
  --side-dir /path/to/side_view_keypoints \
  --epochs-pose 120 --epochs-cls 40 --batch 32 --device 0

# Step 6: MSP 5,600-frame preparation, 4090 multi-view benchmark, and figures
python msp_prepare_subset.py --msp-dir /path/to/MSP --per-cell 100
python msp_4090_benchmark.py \
  --data-dir ./cache/msp_subset \
  --weights-dir ./cache \
  --out-dir ./outputs \
  --cls-epochs 15
python plot_msp_figures.py
```

## 4. Key Quantitative Findings

- **INT8 Quantization Collapse & Mixed-Precision Fix**: Full static INT8 (`3.82 MB`) collapses `YOLOv8n-pose` person detection to `0.0%` on both desktop side-view (`n=273`) and MSP multi-view (`n=1,120`). Isolating only the 72 post-convolution decode/DFL/Sigmoid nodes inside `/model.22/` as FP32 (`Mixed_DecodeFP32`, **`3.93 MB`**, `-70.9%` vs FP32) restores **`100.0%`** full-body detection, cuts mean CPU inference latency from `85.21 ms` to **`43.44 ms` (`1.96x` speedup)**, achieves `97.73%` `PCK@0.10` keypoint agreement with FP32, and preserves **`87.14%`** vs `87.86%` 5-fold 7-class SVM-RBF accuracy.
- **Cross-Subject / Environment Generalization vs End-to-End RGB CNN**: Under same-subject temporal splitting on MSP, `YOLOv8n-cls` (RGB) achieves `95.45%`, but collapses to **`31.00%`** (`view1-7 -> view8`) and **`30.86%`** (`view8 -> view1-7`) under unseen subject & room conditions. Our 24-dim torso- & view-normalized geometric representation (`FullBody_NormGeo`) achieves **`80.71%`** (`SVM-RBF`, Macro-F1 `80.83%`) on `view1-7 -> view8` and **`70.22%`** on `view8 -> view1-7` (`+49.71` and `+39.36` pp over RGB CNN).
- **Documented Failure Boundary (Pure Unseen-Angle LOVO)**: Leave-One-View-Out evaluation across unseen camera azimuths (`view1`–`view7`) yields `63.31%` mean accuracy (`RandomForest` on `NormGeo_24d`, with `view3` 90° side at `52.14%` and elevated `view6` at `48.14%`), and even 3D GT skeletons drop to `57%–67%` without camera-azimuth canonicalization — identifying explicit view-canonicalization / multi-frame temporal filtering as the key open challenge beyond single-frame 2D geometry.
