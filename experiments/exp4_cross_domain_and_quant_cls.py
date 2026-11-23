"""Experiment 4: Cross-subject generalization & mixed-INT8 downstream accuracy."""

import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from exp2_posture_classification import (
    build_front_matrix,
    build_side_matrix,
    eval_cv,
    make_classifiers,
)


def main():
    base = os.path.dirname(__file__)
    cache = os.path.join(base, "cache")
    out_dir = os.path.join(base, "outputs")
    os.makedirs(out_dir, exist_ok=True)

    front_records = json.load(open(os.path.join(cache, "front_records.json"), encoding="utf-8"))
    side_records = json.load(open(os.path.join(cache, "side_records.json"), encoding="utf-8"))
    yolo_npz = np.load(os.path.join(cache, "kpts_yolo_fp32.npz"))
    mp_npz = np.load(os.path.join(cache, "kpts_mediapipe.npz"))
    quant_npz = np.load(os.path.join(cache, "kpts_quant_ablation.npz"))

    f_kpts, f_sizes = yolo_npz["front_kpts"], yolo_npz["front_sizes"]
    f_mp = mp_npz["front_kpts"]

    ext_idx = [i for i, r in enumerate(front_records) if r["filename"].startswith("extract")]
    mb_idx = [i for i, r in enumerate(front_records) if r["filename"].startswith("mb")]

    cross_domain = {}
    for feat_name, kpts_src, raw_mode in [
        ("YOLO_RawXY", f_kpts, True),
        ("YOLO_NormGeo", f_kpts, False),
        ("MediaPipe_NormGeo", f_mp, False),
    ]:
        X_ext, y_ext, _ = build_front_matrix(
            kpts_src[ext_idx], f_sizes[ext_idx], [front_records[i] for i in ext_idx], raw_mode=raw_mode
        )
        X_mb, y_mb, _ = build_front_matrix(
            kpts_src[mb_idx], f_sizes[mb_idx], [front_records[i] for i in mb_idx], raw_mode=raw_mode
        )
        for clf_name in ("LogReg", "SVM-RBF", "RandomForest", "MLP"):
            clf1 = make_classifiers()[clf_name]
            clf1.fit(X_ext, y_ext)
            p_mb = clf1.predict(X_mb)

            clf2 = make_classifiers()[clf_name]
            clf2.fit(X_mb, y_mb)
            p_ext = clf2.predict(X_ext)

            key = f"{feat_name}_{clf_name}"
            cross_domain[key] = {
                "extract_to_mb_acc": round(accuracy_score(y_mb, p_mb) * 100, 2),
                "extract_to_mb_f1": round(f1_score(y_mb, p_mb, average="macro") * 100, 2),
                "mb_to_extract_acc": round(accuracy_score(y_ext, p_ext) * 100, 2),
                "mb_to_extract_f1": round(f1_score(y_ext, p_ext, average="macro") * 100, 2),
            }

    sub_idx = quant_npz["front_sub_indices"].tolist()
    sub_records = [front_records[i] for i in sub_idx]
    sub_sizes = f_sizes[sub_idx]

    quant_cls_front = {}
    for tag, arr in [
        ("Mixed_DecodeFP32", quant_npz["front_decode"]),
        ("Mixed_HeadFP32", quant_npz["front_head"]),
    ]:
        Xq, yq, _ = build_front_matrix(arr, sub_sizes, sub_records, raw_mode=False)
        res = eval_cv(Xq, yq, lambda: make_classifiers()["RandomForest"])
        quant_cls_front[tag] = {k: v for k, v in res.items() if k != "oof_pred"}

    side_unique_mask = [i for i, r in enumerate(side_records) if r["is_unique_source"]]
    side_unique_recs = [side_records[i] for i in side_unique_mask]
    s_sizes_u = yolo_npz["side_sizes"][side_unique_mask]

    quant_cls_side = {}
    for tag, arr in [
        ("Mixed_DecodeFP32", quant_npz["side_decode"]),
        ("Mixed_HeadFP32", quant_npz["side_head"]),
    ]:
        Xs, ys, _ = build_side_matrix(arr, s_sizes_u, side_unique_recs, unique_only=False)
        for cname in ("SVM-RBF", "RandomForest"):
            res = eval_cv(Xs, ys, lambda n=cname: make_classifiers()[n])
            quant_cls_side[f"{tag}_{cname}"] = {k: v for k, v in res.items() if k != "oof_pred"}

    summary = {
        "cross_domain_generalization": cross_domain,
        "quant_front_subset_300_rf": quant_cls_front,
        "quant_side_unique_273": quant_cls_side,
    }
    out_json = os.path.join(out_dir, "exp4_cross_domain_and_quant.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
