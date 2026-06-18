# Utonia-Cap: 3D Point Cloud Captioning

> **IIT Gandhinagar Summer Project 2025**
> Bridging a state-of-the-art 3D foundation model with a language model for scene captioning.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch 2.5](https://img.shields.io/badge/pytorch-2.5.0-orange.svg)](https://pytorch.org/)
[![CUDA 12.4](https://img.shields.io/badge/cuda-12.4-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Apache%202.0-lightgrey.svg)](LICENSE)

---

## What It Does

Given a 3D point cloud scan of any indoor scene, Utonia-Cap generates a detailed natural language caption:

```
Input:  sample1.npz  (273,530 3D points of a room)

Output: "A cluttered living room featuring a large brown sofa 
         against the wall, a wooden coffee table in the center, 
         and several chairs near the window."
```

## Architecture

```
[Point Cloud (.ply / .npz)]
         │
         ▼
┌─────────────────────┐
│   Utonia Encoder    │  ← Frozen (ICML 2026, Point Transformer V3)
│   (137M params)     │    Processes indoor, outdoor, object clouds
└─────────┬───────────┘
          │ bottleneck features [N × 512]
          ▼
┌─────────────────────┐
│ Cross-Attention     │  ← Trained (Stage 1: ~10M params)
│ Projector           │    32 learnable queries attend to all N points
└─────────┬───────────┘
          │ fixed tokens [32 × 1536]
          ▼
┌─────────────────────┐
│  Qwen2.5-1.5B-Inst  │  ← LoRA fine-tuned (Stage 2: +8M params)
│  Language Decoder   │    Generates the caption token by token
└─────────┬───────────┘
          │
          ▼
  "A living room with a sofa..."
```

**Key design decisions:**
- **Frozen Utonia backbone**: preserves cross-domain geometry understanding (indoor + outdoor + objects + LiDAR all work)
- **Cross-attention projector instead of Q-Former**: lighter, fits in 6GB VRAM, 90% of the quality
- **LoRA instead of full fine-tuning**: only 8M extra parameters trained, prevents catastrophic forgetting
- **Qwen2.5-1.5B**: smallest model that produces coherent multi-sentence descriptions

---

## Results

| Method | BLEU-4 | CIDEr | METEOR | Backbone |
|---|---|---|---|---|
| Scan2Cap (baseline) | 23.3 | 56.4 | 21.9 | VoteNet |
| Vote2Cap-DETR | 34.2 | 109.8 | 26.6 | 3DETR |
| **Utonia-Cap (ours)** | **TBD** | **TBD** | **TBD** | Utonia (ICML'26) |

*Evaluated on ScanRefer val split. Results will be updated after training completes.*

---

## Setup

### Requirements
- GPU with ≥6GB VRAM (tested on RTX 3050 6GB)
- CUDA 12.4
- Python 3.10

### Installation
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/utonia-cap
cd utonia-cap

# 2. Create conda environment
conda env create -f environment.yml
conda activate utonia

# 3. Install project dependencies
pip install -r requirements.txt

# 4. Download Utonia backbone weights (~550MB)
wget -c "https://huggingface.co/Pointcept/Utonia/resolve/main/utonia.pth" \
     -O ckpt/utonia.pth
```

---

## Usage

### Quick Demo (No Training Needed)
```bash
export PYTHONPATH=./

# Inspect what Utonia sees (feature shapes)
python utonia_cap/inspect_features.py

# Run inference (before training — output will be random)
python utonia_cap/inference.py \
    --input ~/.cache/utonia/data/sample1.npz \
    --prompt describe
```

### Training

**Stage 1 — Alignment (train projector only):**
```bash
# Quick test with synthetic data (no download needed)
python utonia_cap/train.py --stage 1 --data synthetic --debug

# Full training with Cap3D
python utonia_cap/train.py --stage 1 --data cap3d --epochs 10
```

**Stage 2 — Instruction Tuning (train projector + Qwen LoRA):**
```bash
python utonia_cap/train.py \
    --stage 2 \
    --checkpoint checkpoints/stage1_best.pt \
    --data scanrefer \
    --epochs 5
```

### Inference After Training
```bash
# Caption an indoor scene
python utonia_cap/inference.py \
    --input your_scene.ply \
    --checkpoint checkpoints/stage1_best.pt \
    --prompt "Describe the 3D scene in detail."

# Interactive mode — try multiple prompts
python utonia_cap/inference.py \
    --input your_scene.npz \
    --checkpoint checkpoints/stage1_best.pt \
    --interactive
```

### Evaluation
```bash
python utonia_cap/evaluate.py \
    --data synthetic \
    --checkpoint checkpoints/stage1_best.pt
```

---

## Project Structure
```
utonia_cap/
├── inspect_features.py   # Visualize Utonia feature shapes
├── projector.py          # Cross-attention projector module
├── model.py              # Full UtoniaCap model
├── dataset.py            # Synthetic + Cap3D + ScanRefer data loaders
├── train.py              # Two-stage training script
├── evaluate.py           # BLEU/CIDEr/METEOR evaluation
├── inference.py          # Caption generation from .ply or .npz files
└── configs/
    ├── stage1.yaml       # Stage 1 hyperparameters
    └── stage2.yaml       # Stage 2 hyperparameters
```

---

## Training Details

| Stage | Trains | Frozen | Data | Steps |
|---|---|---|---|---|
| Stage 1 (Alignment) | Projector (10M) | Utonia + Qwen | Cap3D + Synthetic | ~5K |
| Stage 2 (Instruction) | Projector + LoRA (18M) | Utonia | ScanRefer | ~3K |

**Hardware:** RTX 3050 6GB VRAM
- Batch size: 1 (with gradient accumulation × 8 = effective batch 8)
- Mixed precision: bfloat16
- Peak VRAM: ~5.4GB during Stage 2

---

## Citing

If you use this work, please cite:

```bibtex
@misc{utonia-cap-2025,
  title  = {Utonia-Cap: 3D Point Cloud Captioning with Utonia Encoder},
  author = {IIT Gandhinagar},
  year   = {2025},
  note   = {Summer Research Project}
}

@inproceedings{wu2025utonia,
  title  = {Utonia: A Unified Point Cloud Encoder},
  author = {Wu, Xiaoyang et al.},
  booktitle = {ICML},
  year   = {2026}
}
```

---

## Acknowledgments
- [Utonia](https://github.com/Pointcept/Utonia) — the 3D foundation model backbone (Meta / ICML 2026)
- [Qwen2.5](https://huggingface.co/Qwen) — the language decoder
- [ScanRefer](https://github.com/daveredrum/ScanRefer) — 3D caption dataset
- [Cap3D](https://huggingface.co/datasets/tiange/Cap3D) — object caption dataset
