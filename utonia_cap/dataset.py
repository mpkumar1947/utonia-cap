"""
Utonia-Cap Dataset: Loads point clouds paired with text captions.

Three data sources supported:
  1. Synthetic:  Auto-generated captions from segment labels in .npz files.
                 No download needed — works with sample1.npz right away.
                 Use this FIRST to verify the pipeline end-to-end.

  2. Cap3D:      660K ShapeNet object captions, free via HuggingFace.
                 Download: python utonia_cap/dataset.py --download cap3d

  3. ScanRefer:  51K indoor scene descriptions (ScanNet license required).
                 Download: python utonia_cap/dataset.py --download scanrefer

Usage:
    conda activate utonia
    export PYTHONPATH=./
    python utonia_cap/dataset.py          # test with synthetic data
    python utonia_cap/dataset.py --download cap3d
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import utonia


# ScanNet class names (same 20 classes as sem_seg demo)
SCANNET_CLASSES = [
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "desk",
    "curtain", "refrigerator", "shower curtain", "toilet", "sink",
    "bathtub", "other furniture",
]

# Caption templates for synthetic data generation
CAPTION_TEMPLATES = [
    "A 3D scene containing {objects}.",
    "This indoor scene features {objects}.",
    "The point cloud shows a room with {objects}.",
    "A room that includes {objects}.",
    "An indoor environment with {objects} visible in the scene.",
]


def labels_to_caption(segment_labels: np.ndarray) -> str:
    """
    Convert segment label array to a natural language caption.

    Given an array like [1, 1, 2, 5, 5, 5, ...] (class indices),
    finds which classes appear and writes a sentence like:
    "A 3D scene containing a wall, floor, sofa, and chair."

    This is our bootstrap strategy — no human annotations needed!
    """
    # Count each class
    class_counts = {}
    for label in segment_labels:
        label = int(label)
        if 0 <= label < len(SCANNET_CLASSES):
            class_counts[SCANNET_CLASSES[label]] = class_counts.get(SCANNET_CLASSES[label], 0) + 1

    if not class_counts:
        return "A 3D scene."

    # Sort by prevalence (most common objects first)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    objects = [cls for cls, _ in sorted_classes]

    # Format as natural language list
    if len(objects) == 1:
        object_str = f"a {objects[0]}"
    elif len(objects) == 2:
        object_str = f"a {objects[0]} and a {objects[1]}"
    else:
        object_str = ", ".join(f"a {o}" for o in objects[:-1]) + f", and a {objects[-1]}"

    template = random.choice(CAPTION_TEMPLATES)
    return template.format(objects=object_str)


class SyntheticPointCloudDataset(Dataset):
    """
    Dataset using locally available .npz files with auto-generated captions.
    Perfect for testing the pipeline before real datasets arrive.

    Each item:
        point_dict: transformed point cloud dict (ready for Utonia)
        caption:    auto-generated sentence from segment labels
    """

    def __init__(self, npz_files: list, grid_size: float = 0.5, seed: int = 42):
        self.npz_files = npz_files
        self.transform = utonia.transform.default(grid_size)
        random.seed(seed)

        print(f"SyntheticPointCloudDataset: {len(npz_files)} scenes")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx: int):
        data_path = self.npz_files[idx]
        point = dict(np.load(data_path))

        # Extract caption embedded in augmented files (from download_cap3d.py)
        embedded_caption = None
        if "caption" in point:
            cap_arr = point.pop("caption")
            # numpy array of shape (1,) with dtype U* — use .item() to get str
            try:
                embedded_caption = cap_arr.item() if hasattr(cap_arr, "item") else str(cap_arr)
            except Exception:
                embedded_caption = None

        # Extract and remove segment labels before feeding to model
        segment = None
        if "segment200" in point:
            point.pop("segment200")
        if "segment20" in point:
            segment = point.pop("segment20")
            point["segment"] = segment
        if "instance" in point:
            point.pop("instance", None)

        # Add zero normals if missing
        if "normal" not in point:
            n = point["coord"].shape[0]
            point["normal"] = np.zeros((n, 3), dtype=np.float32)

        # Use embedded caption if present, else auto-generate from labels
        if embedded_caption and embedded_caption != "None":
            caption = embedded_caption
        else:
            caption = labels_to_caption(segment) if segment is not None else "A 3D scene."

        # Apply Utonia transforms (voxelization, normalization, etc.)
        point = self.transform(point)

        return point, caption


class Cap3DDataset(Dataset):
    """
    Cap3D: 660K ShapeNet object captions.
    Download: huggingface.co/datasets/tiange/Cap3D

    Each item:
        point_dict:  ShapeNet object point cloud (coord, color, normal)
        caption:     Human-written + GPT-refined description

    Example captions:
        "A wooden armchair with blue cushions and carved armrests."
        "A modern desk lamp with an adjustable neck and circular base."
    """

    def __init__(self, data_dir: str, split: str = "train", max_points: int = 8192):
        self.data_dir = data_dir
        self.max_points = max_points
        self.transform = utonia.transform.default(4.0)  # Object-scale grid size

        # Load captions CSV
        caption_file = os.path.join(data_dir, f"Cap3D_automated_Objaverse_no3Dword_{split}.csv")
        if not os.path.exists(caption_file):
            raise FileNotFoundError(
                f"Cap3D captions not found at {caption_file}\n"
                f"Run: python utonia_cap/dataset.py --download cap3d"
            )

        self.items = []
        with open(caption_file) as f:
            for line in f:
                parts = line.strip().split(",", 1)
                if len(parts) == 2:
                    uid, caption = parts
                    ply_path = os.path.join(data_dir, "pointclouds", f"{uid}.pt")
                    if os.path.exists(ply_path):
                        self.items.append((ply_path, caption.strip('"')))

        print(f"Cap3DDataset ({split}): {len(self.items):,} objects loaded")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        ply_path, caption = self.items[idx]
        data = torch.load(ply_path, weights_only=False)

        point = {
            "coord": data["coord"].numpy() if isinstance(data["coord"], torch.Tensor) else data["coord"],
            "color": data.get("color", np.ones((len(data["coord"]), 3), dtype=np.float32)),
            "normal": data.get("normal", np.zeros((len(data["coord"]), 3), dtype=np.float32)),
        }

        # Subsample if too many points
        N = point["coord"].shape[0]
        if N > self.max_points:
            idx_sample = np.random.choice(N, self.max_points, replace=False)
            point = {k: v[idx_sample] for k, v in point.items()}

        point = self.transform(point)
        return point, caption


def collate_fn(batch):
    """
    Custom collate: handles variable-length point clouds.
    Unlike images (all same HxW), point clouds have different sizes.
    We concatenate them and track boundaries using offsets.
    """
    points_list, captions = zip(*batch)

    # Concatenate all point features along the point dimension
    # e.g., scene 1 has 800 points, scene 2 has 1200 → cat to [2000, ...]
    merged = {}
    for key in points_list[0].keys():
        if isinstance(points_list[0][key], torch.Tensor):
            merged[key] = torch.cat([p[key] for p in points_list], dim=0)

    # Create offset: cumulative point counts for splitting later
    counts = torch.tensor([p["coord"].shape[0] for p in points_list])
    merged["offset"] = torch.cumsum(counts, dim=0).int()

    # Create batch index: which scene each point belongs to
    merged["batch"] = torch.cat([
        torch.full((p["coord"].shape[0],), i, dtype=torch.long)
        for i, p in enumerate(points_list)
    ])

    return merged, list(captions)


def download_cap3d(save_dir: str):
    """Download Cap3D dataset from HuggingFace."""
    from huggingface_hub import snapshot_download
    print("Downloading Cap3D from HuggingFace (this is large, ~10GB)...")
    snapshot_download(
        repo_id="tiange/Cap3D",
        repo_type="dataset",
        local_dir=save_dir,
        ignore_patterns=["*.zip"],  # Skip raw zips, use extracted files
    )
    print(f"✓ Cap3D downloaded to {save_dir}")


# ── Self-Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", choices=["cap3d", "scanrefer"],
                        help="Download a dataset")
    args = parser.parse_args()

    if args.download == "cap3d":
        save_dir = os.path.expanduser("~/.cache/utonia/cap3d")
        download_cap3d(save_dir)
        sys.exit(0)

    # Test with synthetic data (always available)
    print("\nTesting SyntheticPointCloudDataset...")
    sample_files = [os.path.expanduser("~/.cache/utonia/data/sample1.npz")]
    dataset = SyntheticPointCloudDataset(sample_files)

    point, caption = dataset[0]
    print(f"\n  Point keys: {list(point.keys())}")
    print(f"  Coord shape: {point['coord'].shape}")
    print(f"  Generated caption: '{caption}'")
    print()

    # Test collate_fn
    loader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    batch_point, batch_caps = next(iter(loader))
    print(f"  Batched coord shape: {batch_point['coord'].shape}")
    print(f"  Offset: {batch_point['offset']}")
    print(f"  Caption: '{batch_caps[0]}'")
    print("\n✓ Dataset OK!")
    print("\n→ Next step: run  python utonia_cap/train.py --stage 1 --debug")
