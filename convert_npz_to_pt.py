"""
One-time conversion: .npz -> .pt
Saves the packed (4, H, W) Bayer HDR array directly as a float32 tensor.
Channel order: B, G1, G2, R (CVPR2023 spec).

Run once before training:
    python convert_npz_to_pt.py --dataset_dir /scratch/gilbreth/chen4848/datasets/Mobile-HDR
"""
import os
import glob
import argparse
import numpy as np
import torch


def convert(dataset_dir: str):
    patterns = [
        os.path.join(dataset_dir, "train", "tensors", "**", "*.npz"),
        os.path.join(dataset_dir, "test", "tensors", "with_gt", "*.npz"),
    ]

    all_files = []
    for pat in patterns:
        all_files.extend(glob.glob(pat, recursive=True))

    if not all_files:
        print(f"No .npz files found under {dataset_dir}")
        return

    print(f"Converting {len(all_files)} files...")
    failed = []

    for i, path in enumerate(all_files):
        out_path = path.replace(".npz", ".pt")
        if os.path.exists(out_path):
            continue

        try:
            with np.load(path) as data:
                hdr = data['hdr'].astype(np.float32)  # (4, H, W) packed Bayer

            assert hdr.shape[0] == 4, f"Expected 4-channel Bayer, got shape {hdr.shape}"
            torch.save(torch.from_numpy(hdr.copy()), out_path)

        except Exception as e:
            print(f"  FAILED: {path} — {e}")
            failed.append(path)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(all_files)}")

    print(f"\nDone. {len(all_files) - len(failed)} converted, {len(failed)} failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()
    convert(args.dataset_dir)