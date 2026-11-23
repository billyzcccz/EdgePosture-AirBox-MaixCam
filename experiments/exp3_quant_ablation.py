"""Experiment 3: Mixed-precision INT8 quantization ablation on YOLOv8n-pose.

Demonstrates why full-graph static INT8 PTQ collapses detection rate to 0.0%
(reproducing the BM1684X hardware observation) and compares selective FP32
exclusion tiers for the Pose/Detect head (/model.22/):
  1) INT8-Full: All 267 ONNX nodes quantized to INT8.
  2) INT8-Mixed-Decode: Only post-Conv decode/DFL/Sigmoid/coord nodes kept FP32.
  3) INT8-Mixed-Head: Entire /model.22/ Pose head kept FP32 (backbone+neck INT8).
"""

import argparse
import json
import os
import time

import numpy as np
import onnx
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from exp1_frontend_benchmark import (
    PostureCalibrationReader,
    evaluate_side_localization,
    run_onnx_cpu,
    summarize_front_detection,
)


def build_mixed_onnx(fp32_onnx: str, out_onnx: str, exclude_nodes, calib_paths):
    if os.path.isfile(out_onnx):
        return out_onnx
    import onnxruntime as ort

    sess = ort.InferenceSession(fp32_onnx, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    reader = PostureCalibrationReader(calib_paths, input_name)
    quantize_static(
        model_input=fp32_onnx,
        model_output=out_onnx,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_exclude=exclude_nodes,
    )
    return out_onnx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=os.path.join(os.path.dirname(__file__), "cache"))
    parser.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "outputs"))
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    fp32_onnx = os.path.join(args.cache_dir, "yolov8n-pose.onnx")
    full_int8_onnx = os.path.join(args.cache_dir, "yolov8n-pose-int8-static.onnx")
    mixed_decode_onnx = os.path.join(args.cache_dir, "yolov8n-pose-int8-mixed-decode.onnx")
    mixed_head_onnx = os.path.join(args.cache_dir, "yolov8n-pose-int8-mixed-head.onnx")

    with open(os.path.join(args.cache_dir, "front_records.json"), "r", encoding="utf-8") as f:
        front_records = json.load(f)
    with open(os.path.join(args.cache_dir, "side_records.json"), "r", encoding="utf-8") as f:
        side_records = json.load(f)
    for r in side_records:
        r["gt_kpts_norm"] = np.array(r["gt_kpts_norm"], dtype=np.float32)

    yolo_npz = np.load(os.path.join(args.cache_dir, "kpts_yolo_fp32.npz"))
    sub_idx = yolo_npz["front_sub_indices"].tolist()
    front_sub_records = [front_records[i] for i in sub_idx]

    # Same deterministic 140-image calibration set
    rng = np.random.default_rng(42)
    calib_paths = [front_records[i]["path"] for i in sub_idx[:100]]
    side_unique = [r["path"] for r in side_records if r["is_unique_source"]]
    calib_paths.extend(rng.choice(side_unique, size=min(40, len(side_unique)), replace=False).tolist())

    model_onnx = onnx.load(fp32_onnx)
    decode_exclude = [
        n.name
        for n in model_onnx.graph.node
        if "/model.22/" in n.name
        and n.op_type in ("Sigmoid", "Softmax", "Mul", "Add", "Sub", "Div", "Reshape", "Concat", "Transpose", "Slice")
    ]
    head_exclude = [n.name for n in model_onnx.graph.node if "/model.22/" in n.name]

    build_mixed_onnx(fp32_onnx, mixed_decode_onnx, decode_exclude, calib_paths)
    build_mixed_onnx(fp32_onnx, mixed_head_onnx, head_exclude, calib_paths)

    side_unique_records = [r for r in side_records if r["is_unique_source"]]

    results = {}
    cached_kpts = {}
    for name, path in [
        ("ONNX_FP32", fp32_onnx),
        ("ONNX_INT8_Full", full_int8_onnx),
        ("ONNX_INT8_Mixed_DecodeFP32", mixed_decode_onnx),
        ("ONNX_INT8_Mixed_HeadFP32", mixed_head_onnx),
    ]:
        print(f"Evaluating {name} ({os.path.getsize(path)/1e6:.2f} MB)...")
        f_kpts, f_lat = run_onnx_cpu(path, front_sub_records)
        s_kpts, s_lat = run_onnx_cpu(path, side_unique_records)
        cached_kpts[name] = (f_kpts, s_kpts)
        results[name] = {
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "front_sub_300": summarize_front_detection(f_kpts),
            "side_unique_273": evaluate_side_localization(s_kpts, side_unique_records, unique_only=True),
            "cpu_latency_front_ms": round(float(np.mean(f_lat[20:])), 2),
            "cpu_latency_side_ms": round(float(np.mean(s_lat[20:])), 2),
        }

    np.savez_compressed(
        os.path.join(args.cache_dir, "kpts_quant_ablation.npz"),
        front_sub_indices=np.array(sub_idx, dtype=np.int32),
        front_decode=cached_kpts["ONNX_INT8_Mixed_DecodeFP32"][0],
        side_decode=cached_kpts["ONNX_INT8_Mixed_DecodeFP32"][1],
        front_head=cached_kpts["ONNX_INT8_Mixed_HeadFP32"][0],
        side_head=cached_kpts["ONNX_INT8_Mixed_HeadFP32"][1],
    )

    out_json = os.path.join(args.out_dir, "exp3_quant_ablation.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\n=== Experiment 3 Mixed-Precision Quantization Ablation ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
