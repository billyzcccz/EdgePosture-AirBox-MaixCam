# EdgePosture: Edge-Collaborative Sitting Posture Monitoring & Voice Feedback System

**Project Origin**: 9th National College Integrated Circuit Innovation & Entrepreneurship Competition (CICC 2025, Sophgo Cup) — **National Third Prize & South China Regional First Prize** (Team: Xiaoxiaobei, Shenzhen MSU-BIT University).

## Overview

EdgePosture is a low-cost, privacy-preserving sitting posture monitoring and voice correction system built on heterogeneous edge hardware:

- **Front-End Perception**: Sipeed **MaixCAM** (RISC-V SG2002) mounted on a custom 3D-printed 2-DOF dual-SG90 servo gimbal for image capture / face tracking.
- **Edge AI Inference Server**: Sophgo **SOPHON AirBox** (BM1684X TPU) executing quantized `YOLOv8-Pose` (`BModel`) inference locally via `sophon-sail` and `BMCV` / `OpenCV` pipelines without uploading raw video frames to the cloud.
- **Posture Analysis & Multimodal Feedback**: A Flask-based posture evaluation service computing 17-keypoint neck/torso geometry, generating personalized posture advice (Baidu Wenxin API with local deterministic fallback), displaying live metrics on a web dashboard, and broadcasting audio reminders over a LAN IP speaker.

## System Architecture

```text
+-------------------+       HTTP / SCP / RTSP        +----------------------------+
|  Sipeed MaixCAM   | -----------------------------> |   SOPHON AirBox (BM1684X)  |
|  (SG2002 + SG90   |                                |  YOLOv8-Pose BModel (SAIL) |
|   2-DOF Gimbal)   |                                |  -> 17 Keypoints JSON      |
+-------------------+                                +----------------------------+
                                                                   |
                                                                   v  JSON Keypoints
+-------------------+         LAN HTTP TTS           +----------------------------+
|   LAN IP Speaker  | <----------------------------- |   Flask Analysis Backend   |
|  (Voice Alerts)   |                                |  - Angle & Baseline Check  |
+-------------------+                                |  - Web UI (/advice)        |
                                                     |  - LLM Advice + Fallback   |
                                                     +----------------------------+
```

## Repository Structure

```text
EdgePosture-AirBox-MaixCam/
├── backend/                  # Posture evaluation server, web UI, LLM & TTS integration
│   ├── app.py                # Flask server (RTSP stream + /detect_keypoints + /advice UI)
│   ├── anomaly_detector.py   # Baseline deviation & angle-threshold posture classifier
│   ├── pose_estimator.py     # Ultralytics YOLOv8-Pose wrapper (host fallback/dev)
│   ├── yolo_detector.py      # Person bounding-box pre-detector with gamma correction
│   ├── speaker/
│   │   └── ip_speaker.py     # GB2312-Base64 HTTP TTS client for LAN IP speaker
│   └── static/               # Web dashboard (advice.html, rtsp-manager.js, styles.css)
├── edge_inference/           # SOPHON BM1684X deployment (Python & C++ BMCV pipelines)
│   ├── python/
│   │   ├── yolov8_opencv.py  # Multi-threaded image decode + SAIL BModel inference
│   │   ├── yolov8_bmcv.py    # Hardware JPU decode + BMCV preprocess pipeline
│   │   └── utils.py
│   ├── cpp/                  # Native C++ BMCV + FFmpeg inference implementation
│   ├── docs/                 # ONNX export guide for TPU-MLIR compilation
│   └── tools/                # COCO keypoint evaluation & benchmark comparison tools
├── hardware/                 # 3D CAD models (.step, .stl, SolidWorks parts)
│   ├── README.md
│   └── cad_parts/
├── .env.example
├── .gitignore
├── requirements.txt
└── LICENSE
```

## Quick Start

### 1. Backend & Web Dashboard (Host / Edge Gateway)

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env to configure BAIDU_API_KEY, BAIDU_SECRET_KEY, RTSP_URL, and IP_SPEAKER_HOST
```

Run the Flask server:

```bash
cd backend
python app.py
```

Open `http://localhost:5000/advice` in your browser to view live posture status, neck/body angles, and posture feedback.

### 2. SOPHON AirBox (BM1684X) Edge Inference

On the AirBox (Ubuntu 20.04 LTS with Sophon SDK 23.09 LTS SP4 installed):

```bash
cd edge_inference/python
python3 yolov8_opencv.py \
  --input ../datasets/test \
  --bmodel ../models/BM1684X/yolov8n_pose_1684x_f32.bmodel \
  --dev_id 0 \
  --conf_thresh 0.25 \
  --nms_thresh 0.7
```

To benchmark `BModel` latency directly on the BM1684X TPU:

```bash
bmrt_test --bmodel /path/to/yolov8n_pose_1684x_f32.bmodel
bmrt_test --bmodel /path/to/yolov8n_pose_1684x_int8_sym.bmodel
```

See [edge_inference/docs/YOLOv8_Export_Guide.md](edge_inference/docs/YOLOv8_Export_Guide.md) and [edge_inference/cpp/README.md](edge_inference/cpp/README.md) for ONNX export, TPU-MLIR compilation, and C++ BMCV build instructions.

## Measured Hardware Benchmarks (BM1684X)

Observed `bmrt_test` TPU kernel measurements on our AirBox unit (`yolov8n_pose`, input `1x3x640x640`):

| Model Precision | Model Size | NPU Launch Total Time | `calculate` Time | Notes |
| :--- | :--- | :--- | :--- | :--- |
| `yolov8n_pose_1684x_f32.bmodel` | 16 MB | 13.34 ms | 13.34 ms | Reliable keypoint detection on test images |
| `yolov8n_pose_1684x_int8_sym.bmodel` | 6.3 MB | 2.23 ms | 2.24 ms | 6x faster kernel, but symmetric INT8 requires calibration tuning / mixed-precision head to avoid recall drop |

## Hardware BOM (Prototype)

| Component | Qty | Approx. Cost (CNY) |
| :--- | :---: | :---: |
| Radxa / SOPHON Fogwise AirBox (BM1684X) | 1 | 2699 |
| Sipeed MaixCAM RISC-V Dev Board (with 32GB SD) | 1 | 320 |
| SG90 Micro Servos | 2 | 10 |
| 3D-Printed PETG Desk Clamp & PLA/PETG Gimbal | 1 set | ~10 |

## Team & Acknowledgments

- **Team**: HAN Yun, DONG Jie, ZHANG Chengzhen (Shenzhen MSU-BIT University)
- **Advisors**: DONG Yanjie, WANG Wei, ZHENG Jianbo
- Built upon [Sophgo SOPHON-DEMO](https://github.com/sophgo/sophon-demo) (`2-Clause BSD`) and [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
