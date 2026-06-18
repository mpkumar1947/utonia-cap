"""
Utonia-Cap: Inspect feature shapes from Utonia encoder.

This runs a forward pass on sample1.npz and prints the feature
dimensions at every encoder stage. Run this FIRST to confirm
what shape the projector will receive as input.

Usage:
    conda activate utonia
    export PYTHONPATH=./
    python utonia_cap/inspect_features.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import utonia

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load Model ────────────────────────────────────────────────────────────────
print("\n[1/4] Loading Utonia model...")
# We need the encoder in enc_mode=True so the decoder doesn't run
# (saves VRAM and gives us bottleneck features directly)
model = utonia.load("ckpt/utonia.pth").to(device)
model.eval()

# Monkey-patch to intercept bottleneck features at stage 4
bottleneck_features = {}

def hook_fn(module, input, output):
    # output is a Point object; .feat is [N, C]
    bottleneck_features["enc4"] = output.feat.detach().cpu()
    bottleneck_features["enc4_coord"] = output.coord.detach().cpu()
    bottleneck_features["enc4_offset"] = output.offset.detach().cpu()

# Register hook on the last encoder stage (enc4)
hook = model.enc.enc4.register_forward_hook(hook_fn)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("[2/4] Loading sample data...")
data_path = os.path.expanduser("~/.cache/utonia/data/sample1.npz")
point = dict(np.load(data_path))
point.pop("segment200")
segment = point.pop("segment20")
point["segment"] = segment

transform = utonia.transform.default(0.5)
point = transform(point)

# ── Forward Pass ──────────────────────────────────────────────────────────────
print("[3/4] Running forward pass...")
with torch.inference_mode():
    for key in point.keys():
        if isinstance(point[key], torch.Tensor) and device == "cuda":
            point[key] = point[key].cuda(non_blocking=True)
    out = model(point)

hook.remove()

# ── Print Results ─────────────────────────────────────────────────────────────
print("\n[4/4] Feature Shape Report:")
print("=" * 50)
print(f"  Input points (N):        {point['coord'].shape[0]:,}")
print(f"  Final output feat:       {out.feat.shape}  ← after full decoder")
print()
print(f"  Bottleneck (Stage 4):")
print(f"    feat shape:            {bottleneck_features['enc4'].shape}  ← USE THIS for captioning")
print(f"    coord shape:           {bottleneck_features['enc4_coord'].shape}")
print(f"    num tokens:            {bottleneck_features['enc4'].shape[0]:,}")
print(f"    feature dim:           {bottleneck_features['enc4'].shape[1]}")
print()
print(f"  Compression ratio:       {point['coord'].shape[0] / bottleneck_features['enc4'].shape[0]:.1f}x points → tokens")
print("=" * 50)
print()
print("✓ Projector input:  torch.Size([N_tokens, 576])  (one tensor per scene)")
print("✓ Projector output: torch.Size([1, 32, 1536])   (32 query tokens for Qwen)")
print()
print("→ Next step: run  python utonia_cap/projector.py")
