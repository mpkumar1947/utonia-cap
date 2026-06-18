"""
Cap3D Caption Injection Dataset.

The core insight: Cap3D has 660K beautifully written, diverse captions.
We don't have Cap3D point clouds (they need 80GB of downloads).
BUT — we CAN use Cap3D captions as style references for our indoor scenes.

Strategy:
  1. Take our 100 augmented indoor scenes (we have these)
  2. For each scene, randomly sample a Cap3D caption style as prefix
  3. Append the actual object list from segment labels
  Result: "A detailed indoor space featuring a wall, floor, and sofa." 
          vs just: "A 3D scene containing a wall, floor, sofa."

This teaches the LLM to write in a richer, more descriptive style
while still grounding descriptions in actual scene content.

Usage:
    conda activate utonia
    export PYTHONPATH=./
    python utonia_cap/dataset_cap3d_inject.py   # self-test
"""

import os, sys, random, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import Dataset

import utonia
from utonia_cap.dataset import SCANNET_CLASSES, labels_to_caption


# Rich caption templates informed by Cap3D style
CAP3D_STYLE_TEMPLATES = [
    "A detailed {room_type} featuring {objects}.",
    "An indoor space with {objects} arranged throughout the room.",
    "A {room_type} scene containing {objects}, typical of a residential interior.",
    "The 3D scan captures a {room_type} with {objects} visible.",
    "A point cloud reconstruction of a {room_type}, showing {objects}.",
    "An interior scene depicting {objects} within a {room_type} environment.",
    "A realistic 3D scan of a {room_type} containing {objects}.",
    "The scene shows a {room_type} with {objects} placed throughout.",
]

ROOM_TYPES = {
    "bed":      ["bedroom", "sleeping quarters"],
    "sofa":     ["living room", "lounge"],
    "toilet":   ["bathroom", "restroom"],
    "bathtub":  ["bathroom", "washroom"],
    "sink":     ["kitchen", "bathroom"],
    "counter":  ["kitchen", "kitchenette"],
    "bookshelf":["study", "library", "office"],
    "desk":     ["office", "study room"],
    "default":  ["indoor space", "room", "interior"],
}


def infer_room_type(class_counts: dict) -> str:
    """Guess room type from which objects are present."""
    for key, names in ROOM_TYPES.items():
        if key in class_counts:
            return random.choice(names)
    return random.choice(ROOM_TYPES["default"])


def rich_caption(segment_labels: np.ndarray) -> str:
    """Generate a Cap3D-style rich caption from segment labels."""
    class_counts = {}
    for label in segment_labels:
        label = int(label)
        if 0 <= label < len(SCANNET_CLASSES):
            name = SCANNET_CLASSES[label]
            class_counts[name] = class_counts.get(name, 0) + 1

    if not class_counts:
        return "An indoor 3D scene."

    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    top_objects = [cls for cls, _ in sorted_classes[:6]]  # top 6 most frequent

    # Format as natural language
    if len(top_objects) == 1:
        obj_str = f"a {top_objects[0]}"
    elif len(top_objects) == 2:
        obj_str = f"a {top_objects[0]} and a {top_objects[1]}"
    else:
        obj_str = (
            ", ".join(f"a {o}" for o in top_objects[:-1])
            + f", and a {top_objects[-1]}"
        )

    room_type = infer_room_type(class_counts)
    template  = random.choice(CAP3D_STYLE_TEMPLATES)
    return template.format(room_type=room_type, objects=obj_str)


def load_cap3d_captions(csv_path: str, limit: int = 10000) -> list:
    """
    Load Cap3D captions. We use them purely for style diversity —
    the actual UIDs don't matter since we're pairing with our own point clouds.
    """
    captions = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split(",", 1)
            if len(parts) == 2:
                caption = parts[1].strip().strip('"')
                # Filter: only keep indoor/room-related captions
                indoor_keywords = ["room", "chair", "table", "sofa", "desk",
                                   "shelf", "cabinet", "wall", "floor", "furniture",
                                   "interior", "indoor", "bedroom", "kitchen"]
                if any(kw in caption.lower() for kw in indoor_keywords):
                    captions.append(caption)
            if len(captions) >= limit:
                break
    return captions


class RichSyntheticDataset(Dataset):
    """
    Augmented indoor scenes with Cap3D-style captions.

    If Cap3D CSV is available: randomly replaces 50% of captions with
    real Cap3D indoor captions for style diversity.
    If not available: uses rich templates from this file.
    """

    def __init__(
        self,
        npz_files: list,
        cap3d_csv: str = None,
        grid_size: float = 0.5,
        seed: int = 42,
    ):
        self.npz_files = npz_files
        self.transform = utonia.transform.default(grid_size)
        random.seed(seed)

        # Load Cap3D captions for style mixing
        self.cap3d_captions = []
        if cap3d_csv and os.path.exists(cap3d_csv):
            print("  Loading Cap3D indoor captions for style injection...")
            self.cap3d_captions = load_cap3d_captions(cap3d_csv, limit=5000)
            print(f"  Loaded {len(self.cap3d_captions)} Cap3D indoor captions")
        else:
            print("  No Cap3D CSV found — using rich templates only")

        print(f"RichSyntheticDataset: {len(npz_files)} scenes")

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx: int):
        data_path = self.npz_files[idx]
        point = dict(np.load(data_path, allow_pickle=True))

        # Extract caption embedded in augmented files
        embedded_caption = None
        if "caption" in point:
            cap_arr = point.pop("caption")
            try:
                embedded_caption = cap_arr.item() if hasattr(cap_arr, "item") else str(cap_arr)
            except Exception:
                embedded_caption = None

        # Clean up non-feature keys
        for key in ["segment200", "instance"]:
            point.pop(key, None)
        segment = point.pop("segment20", None) or point.pop("segment", None)
        if segment is not None and "segment" not in point:
            point["segment"] = segment

        if "normal" not in point:
            point["normal"] = np.zeros((point["coord"].shape[0], 3), dtype=np.float32)

        # Caption selection strategy:
        # 30% → Cap3D real indoor caption
        # 70% → rich template from segment labels
        r = random.random()
        if r < 0.3 and self.cap3d_captions:
            caption = random.choice(self.cap3d_captions)
        elif segment is not None:
            caption = rich_caption(segment)
        elif embedded_caption and embedded_caption != "None":
            caption = embedded_caption
        else:
            caption = "An indoor 3D scene."

        point = self.transform(point)
        return point, caption


# ── Self-Test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    aug_dir  = os.path.expanduser("~/.cache/utonia/augmented")
    cap3d_csv = os.path.expanduser("~/.cache/utonia/cap3d/Cap3D_automated_Objaverse_full.csv")

    files = [os.path.join(aug_dir, f) for f in os.listdir(aug_dir) if f.endswith(".npz")][:5]
    ds = RichSyntheticDataset(files, cap3d_csv=cap3d_csv)

    print("\nSample captions:")
    for i in range(5):
        _, cap = ds[i % len(ds)]
        print(f"  [{i}] {cap[:120]}")

    print("\n✓ Rich dataset OK")
    print("→ Run training with: python utonia_cap/train.py --data rich")
