"""
Download a small subset of Cap3D for Stage 1 training.

Cap3D has 660K ShapeNet objects with GPT-4 captions.
We grab just 5000 objects (≈2.5GB) which is enough for Stage 1.

Full Cap3D download would be ~80GB — not needed for our purposes.

Usage:
    conda activate utonia
    export PYTHONPATH=./
    python utonia_cap/download_cap3d.py

Output: ~/.cache/utonia/cap3d/
"""

import os
import sys
import json
import requests
import numpy as np
from pathlib import Path

SAVE_DIR = os.path.expanduser("~/.cache/utonia/cap3d")


def download_captions():
    """Download the Cap3D caption CSV from HuggingFace."""
    os.makedirs(SAVE_DIR, exist_ok=True)

    caption_url = (
        "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/"
        "Cap3D_automated_Objaverse_no3Dword_train.csv"
    )
    save_path = os.path.join(SAVE_DIR, "Cap3D_automated_Objaverse_no3Dword_train.csv")

    if os.path.exists(save_path):
        print(f"  Captions already downloaded: {save_path}")
        return save_path

    print(f"Downloading Cap3D captions...")
    r = requests.get(caption_url, stream=True)
    r.raise_for_status()

    total = int(r.headers.get("content-length", 0))
    downloaded = 0
    with open(save_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            pct = downloaded / total * 100 if total else 0
            print(f"\r  {downloaded/1024**2:.1f}MB / {total/1024**2:.1f}MB  ({pct:.0f}%)", end="")

    print(f"\n  ✓ Captions saved to {save_path}")
    return save_path


def parse_captions(csv_path, limit=5000):
    """Read first N uid,caption pairs from the CSV."""
    items = []
    with open(csv_path, "r") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                uid, caption = parts
                caption = caption.strip('"').strip()
                if caption:
                    items.append((uid.strip(), caption))
            if len(items) >= limit:
                break
    print(f"  Parsed {len(items)} caption entries")
    return items


def download_pointclouds(items):
    """
    Download pre-computed .pt point cloud files from Cap3D HuggingFace repo.
    Each file is a dict with keys: coord [N,3], color [N,3].
    """
    pc_dir = os.path.join(SAVE_DIR, "pointclouds")
    os.makedirs(pc_dir, exist_ok=True)

    base_url = (
        "https://huggingface.co/datasets/tiange/Cap3D/resolve/main/"
        "RenderedImage_perobj_zips/pcs_pt/"
    )

    ok, skip, fail = 0, 0, 0

    for i, (uid, caption) in enumerate(items):
        save_path = os.path.join(pc_dir, f"{uid}.pt")

        if os.path.exists(save_path):
            skip += 1
            continue

        url = f"{base_url}{uid}.pt"
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                ok += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1

        if i % 100 == 0:
            print(f"  [{i+1:5d}/{len(items)}] downloaded={ok} skipped={skip} failed={fail}")

    print(f"\n  ✓ Download complete: {ok} new, {skip} cached, {fail} failed")
    return ok + skip


def create_synthetic_from_scannet():
    """
    Fallback: create a larger synthetic dataset by augmenting our 3 existing
    scenes with random crops, rotations and paraphrased captions.
    This gives ~300 training pairs without any download.
    """
    import torch
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import utonia
    from utonia_cap.dataset import SCANNET_CLASSES, CAPTION_TEMPLATES, labels_to_caption
    import random

    source_dir = os.path.expanduser("~/.cache/utonia/data")
    save_dir   = os.path.expanduser("~/.cache/utonia/augmented")
    os.makedirs(save_dir, exist_ok=True)

    indoor_files = [
        f for f in os.listdir(source_dir)
        if f.endswith(".npz") and "outdoor" not in f and "object" not in f
    ]
    if not indoor_files:
        print("No indoor .npz files found. Skipping augmentation.")
        return

    print(f"\nCreating augmented dataset from {len(indoor_files)} indoor scene(s)...")
    count = 0

    for fname in indoor_files:
        raw = dict(np.load(os.path.join(source_dir, fname)))
        raw.pop("segment200", None)
        raw.pop("instance", None)
        segment = raw.pop("segment20", None)
        if segment is not None:
            raw["segment"] = segment
        if "normal" not in raw:
            raw["normal"] = np.zeros_like(raw["coord"])

        coord  = raw["coord"].astype(np.float32)
        color  = raw["color"].astype(np.float32) / 255.0
        normal = raw["normal"].astype(np.float32)
        seg    = raw.get("segment", np.zeros(len(coord), dtype=np.int8))
        N      = len(coord)

        # Generate 100 augmented crops per scene
        for aug_idx in range(100):
            # Random rotation around Z axis
            angle = random.uniform(0, 2 * np.pi)
            c, s  = np.cos(angle), np.sin(angle)
            R     = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            coord_aug = coord @ R.T

            # Random crop: take 50-80% of points from a spatial region
            frac = random.uniform(0.5, 0.8)
            n_crop = int(N * frac)
            center = coord_aug[random.randint(0, N-1)]
            dists  = np.linalg.norm(coord_aug - center, axis=1)
            idx    = np.argsort(dists)[:n_crop]

            aug_data = {
                "coord":   coord_aug[idx],
                "color":   color[idx],
                "normal":  normal[idx],
                "segment": seg[idx],
            }

            # Different caption template each time
            caption = labels_to_caption(seg[idx])

            out_path = os.path.join(save_dir, f"{fname[:-4]}_aug{aug_idx:03d}.npz")
            np.savez_compressed(out_path, **aug_data,
                                caption=np.array([caption]))
            count += 1

    print(f"  ✓ Created {count} augmented scenes → {save_dir}")
    return save_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="augment",
                        choices=["augment", "cap3d"],
                        help="augment=local data augmentation (instant), cap3d=download Cap3D")
    parser.add_argument("--limit", type=int, default=5000,
                        help="Number of Cap3D objects to download (cap3d mode only)")
    args = parser.parse_args()

    if args.mode == "augment":
        print("Creating augmented synthetic dataset (no download needed)...")
        save_dir = create_synthetic_from_scannet()
        print(f"\n✓ Done! Now run:")
        print(f"  python utonia_cap/train.py --stage 1 --data augmented --epochs 100")

    elif args.mode == "cap3d":
        print(f"Downloading Cap3D (first {args.limit} objects)...")
        csv_path = download_captions()
        items    = parse_captions(csv_path, limit=args.limit)
        n_ok     = download_pointclouds(items)
        print(f"\n✓ Cap3D ready ({n_ok} objects) at {SAVE_DIR}")
        print(f"  Now run: python utonia_cap/train.py --stage 1 --data cap3d")
