"""Preprocess and uniformly subsample the MSP (Multi-View Sitting Posture) dataset.

Reads view1.zip..view8.zip directly without extracting all 14.87 GB, applies:
1) Uniform index-based temporal subsampling (default: 100 frames per (view, class)
   = 5,600 balanced frames across 8 views x 7 posture classes).
2) Channel fix (origin PNGs in MSP archives were stored with B/R channels swapped).
3) Extraction of paired GT 3D keypoints (Hkeypoints 17x3, Mkeypoints 33x4).
4) Clean manifest + tar archive ready for upload to RTX 4090.
"""

import argparse
import io
import json
import os
import tarfile
import zipfile

import numpy as np
from PIL import Image

MSP_CLASSES = ["upright", "back", "cheek", "down", "left", "right", "leg"]


def list_class_indices(zf: zipfile.ZipFile, view: str, cls_name: str):
    prefix = f"{view}/{cls_name}/origin/"
    indices = []
    for name in zf.namelist():
        if name.startswith(prefix) and name.endswith(".png"):
            stem = os.path.splitext(os.path.basename(name))[0]
            if stem.isdigit():
                indices.append(int(stem))
    indices.sort()
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--msp-dir",
        default=os.path.join(os.path.dirname(__file__), "datasets", "msp_raw"),
        help="Directory containing MSP view1.zip..view8.zip",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "cache", "msp_subset"),
    )
    parser.add_argument("--per-cell", type=int, default=100)
    args = parser.parse_args()

    img_root = os.path.join(args.out_dir, "images")
    os.makedirs(img_root, exist_ok=True)

    records = []
    h_kpts_list = []
    m_kpts_list = []

    for v_idx in range(1, 9):
        view = f"view{v_idx}"
        zp = os.path.join(args.msp_dir, f"{view}.zip")
        if not os.path.isfile(zp):
            raise FileNotFoundError(zp)
        print(f"Processing {zp}...")
        with zipfile.ZipFile(zp, "r") as zf:
            names_set = set(zf.namelist())
            for cls_id, cls_name in enumerate(MSP_CLASSES):
                all_indices = list_class_indices(zf, view, cls_name)
                if not all_indices:
                    raise RuntimeError(f"Empty class {view}/{cls_name}")
                n_take = min(args.per_cell, len(all_indices))
                sel_pos = np.linspace(0, len(all_indices) - 1, n_take, dtype=int)
                sel_indices = [all_indices[p] for p in sel_pos]

                out_cls_dir = os.path.join(img_root, view, cls_name)
                os.makedirs(out_cls_dir, exist_ok=True)

                for seq_pos, frame_idx in enumerate(sel_indices):
                    png_entry = f"{view}/{cls_name}/origin/{frame_idx}.png"
                    hk_entry = f"{view}/{cls_name}/Hkeypoints/{frame_idx}.txt"
                    mk_entry = f"{view}/{cls_name}/Mkeypoints/{frame_idx}.txt"

                    raw_bytes = zf.read(png_entry)
                    arr = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                    # Original dataset PNGs stored OpenCV BGR channels in RGB order.
                    arr_fixed = arr[:, :, ::-1].copy()

                    rel_img = f"images/{view}/{cls_name}/{frame_idx:05d}.jpg"
                    out_img_path = os.path.join(args.out_dir, rel_img.replace("/", os.sep))
                    Image.fromarray(arr_fixed).save(out_img_path, format="JPEG", quality=95)

                    if hk_entry in names_set:
                        hk = np.loadtxt(
                            io.StringIO(zf.read(hk_entry).decode("utf-8", errors="replace")),
                            delimiter=",",
                            dtype=np.float32,
                        )
                    else:
                        hk = np.full((17, 3), np.nan, dtype=np.float32)

                    if mk_entry in names_set:
                        mk = np.loadtxt(
                            io.StringIO(zf.read(mk_entry).decode("utf-8", errors="replace")),
                            delimiter=",",
                            dtype=np.float32,
                        )
                    else:
                        mk = np.full((33, 4), np.nan, dtype=np.float32)

                    h_kpts_list.append(hk.reshape(17, 3))
                    m_kpts_list.append(mk.reshape(33, 4))
                    records.append(
                        {
                            "sample_id": len(records),
                            "view": view,
                            "view_id": v_idx,
                            "subject_group": "subject_B_env2" if v_idx == 8 else "subject_A_env1",
                            "class": cls_name,
                            "class_id": cls_id,
                            "frame_idx": int(frame_idx),
                            "seq_pos": int(seq_pos),
                            "total_view_cls_frames": int(len(all_indices)),
                            "rel_path": rel_img,
                        }
                    )

    manifest_path = os.path.join(args.out_dir, "msp_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes": MSP_CLASSES,
                "per_cell": args.per_cell,
                "total_samples": len(records),
                "records": records,
            },
            f,
            indent=2,
        )

    gt_npz_path = os.path.join(args.out_dir, "msp_subset_gt.npz")
    np.savez_compressed(
        gt_npz_path,
        h_kpts=np.stack(h_kpts_list, axis=0),
        m_kpts=np.stack(m_kpts_list, axis=0),
        class_ids=np.array([r["class_id"] for r in records], dtype=np.int64),
        view_ids=np.array([r["view_id"] for r in records], dtype=np.int64),
        seq_pos=np.array([r["seq_pos"] for r in records], dtype=np.int64),
    )

    tar_path = os.path.join(os.path.dirname(args.out_dir), "msp_subset.tar.gz")
    print(f"Packing {len(records)} samples into {tar_path}...")
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(args.out_dir, arcname="msp_subset")

    tar_mb = os.path.getsize(tar_path) / 1e6
    print(f"Done! total_samples={len(records)}, tar_size_mb={tar_mb:.2f}")


if __name__ == "__main__":
    main()
