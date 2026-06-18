"""
Utonia-Cap Training Script: Two-stage training for 3D captioning.

VRAM Strategy for RTX 3050 6GB:
    Utonia encoder:    ~1.5 GB on GPU
    Qwen2.5-1.5B:      ~3.2 GB on GPU (bfloat16)
    Projector:         ~0.02 GB
    Activations:       ~0.5 GB
    Total headroom:    5.7 GB (tight if both models on GPU simultaneously)

    Solution: Pre-computation strategy
      Phase A: Load Utonia → encode all training scenes → save features to CPU RAM → unload Utonia
      Phase B: Load Qwen → train projector on saved features → gradient checkpointing

    This keeps peak VRAM at ~3.7 GB, well within budget.

Usage:
    conda activate utonia
    export PYTHONPATH=./

    # Quick debug (20 steps, synthetic data)
    python utonia_cap/train.py --stage 1 --debug

    # Full Stage 1 training
    python utonia_cap/train.py --stage 1 --epochs 10 --data synthetic

    # Stage 2 (run after Stage 1 checkpoint exists)
    python utonia_cap/train.py --stage 2 --checkpoint checkpoints/stage1_best.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import gc
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import utonia
from utonia_cap.projector import UtoniaCrossAttentionProjector
from utonia_cap.dataset import SyntheticPointCloudDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train Utonia-Cap")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2])
    parser.add_argument("--data", type=str, default="synthetic",
                        choices=["synthetic", "augmented", "cap3d", "scanrefer"])
    parser.add_argument("--data-dir", type=str,
                        default=os.path.expanduser("~/.cache/utonia"))
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="Gradient accumulation (effective batch = accum × 1)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--debug", action="store_true",
                        help="Stop after 20 training steps for pipeline check")
    return parser.parse_args()


def get_dataset(args):
    if args.data == "synthetic":
        data_dir = os.path.join(args.data_dir, "data")
        sample_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir) if f.endswith(".npz")
        ]
        if not sample_files:
            raise FileNotFoundError(f"No .npz files in {data_dir}")
        print(f"Synthetic: {len(sample_files)} scene file(s) on disk")
        if args.debug:
            sample_files = sample_files * 20  # repeat to get enough steps
        return SyntheticPointCloudDataset(sample_files)
    elif args.data == "augmented":
        aug_dir = os.path.join(args.data_dir, "augmented")
        if not os.path.exists(aug_dir):
            raise FileNotFoundError(
                f"Augmented data not found at {aug_dir}\n"
                "Run: python utonia_cap/download_cap3d.py --mode augment"
            )
        aug_files = [os.path.join(aug_dir, f) for f in os.listdir(aug_dir) if f.endswith(".npz")]
        print(f"Augmented: {len(aug_files)} scenes")
        return SyntheticPointCloudDataset(aug_files)
    elif args.data == "cap3d":
        from utonia_cap.dataset import Cap3DDataset
        return Cap3DDataset(os.path.join(args.data_dir, "cap3d"), split="train")
    else:
        raise NotImplementedError("ScanRefer loader coming in Week 3")


# ── Phase A: Pre-compute Utonia features ─────────────────────────────────────

def precompute_features(dataset, device: str, debug_limit: int = None):
    """
    Run all scenes through frozen Utonia encoder.
    Stores bottleneck features in CPU RAM (not GPU) so Utonia can be unloaded.

    Returns:
        feature_cache: list of {"feat": Tensor[N,576], "offset": Tensor[1], "caption": str}
    """
    print("\n" + "="*60)
    print("  Phase A: Pre-computing Utonia features")
    print("  (Utonia will be unloaded from GPU after this)")
    print("="*60)

    utonia_model = utonia.load("ckpt/utonia.pth").to(device)
    utonia_model.eval()
    for p in utonia_model.parameters():
        p.requires_grad = False

    # Hook captures bottleneck at Stage 4
    captured = {}
    def hook(m, i, o):
        # Store on CPU immediately to avoid filling GPU VRAM
        captured["feat"] = o.feat.detach().cpu()
        captured["offset"] = o.offset.detach().cpu()
    utonia_model.enc.enc4.register_forward_hook(hook)

    loader = DataLoader(dataset, batch_size=1, shuffle=True,
                        collate_fn=collate_fn, num_workers=0)

    feature_cache = []
    transform_ok = 0

    for idx, (point_dict, captions) in enumerate(loader):
        if debug_limit and idx >= debug_limit:
            break

        # Move to GPU only for Utonia forward pass
        for k in point_dict:
            if isinstance(point_dict[k], torch.Tensor):
                point_dict[k] = point_dict[k].to(device)

        try:
            with torch.inference_mode():
                utonia_model(point_dict)

            feature_cache.append({
                "feat":    captured["feat"].clone(),    # [N, 576] on CPU
                "offset":  captured["offset"].clone(),  # [1] on CPU
                "caption": captions[0],
            })
            transform_ok += 1

            if idx % 10 == 0:
                n_tok = captured["feat"].shape[0]
                print(f"  [{idx+1:4d}] tokens={n_tok}, "
                      f"feat_dim={captured['feat'].shape[1]}, "
                      f"caption='{captions[0][:60]}...'")

        except Exception as e:
            print(f"  [{idx+1}] SKIP — {e}")
            continue

    # Unload Utonia from GPU completely
    del utonia_model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"\n  ✓ Pre-computed {len(feature_cache)} scenes")
    print(f"  ✓ Utonia unloaded from GPU")
    vram_free = torch.cuda.memory_reserved() / 1024**3
    print(f"  GPU VRAM reserved after unload: {vram_free:.2f} GB\n")

    return feature_cache


# ── Cached Feature Dataset ────────────────────────────────────────────────────

class CachedFeatureDataset(Dataset):
    """Wraps pre-computed features — no GPU needed to iterate."""
    def __init__(self, cache):
        self.cache = cache

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        item = self.cache[idx]
        return item["feat"], item["offset"], item["caption"]


def cached_collate(batch):
    feats, offsets, captions = zip(*batch)
    # Stack features and compute new cumulative offsets
    n = [f.shape[0] for f in feats]
    merged_feat = torch.cat(feats, dim=0)
    merged_offset = torch.cumsum(torch.tensor(n), dim=0).int()
    return merged_feat, merged_offset, list(captions)


# ── Phase B: Train projector with Qwen frozen ─────────────────────────────────

class ProjectorTrainer:
    """
    Trains only the cross-attention projector.
    Qwen acts as a frozen loss function — its weights never change.
    Peak VRAM: ~3.7 GB (projector activations + Qwen forward pass).
    """

    def __init__(self, args):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("="*60)
        print("  Phase B: Loading Qwen2.5-1.5B (Utonia already unloaded)")
        print("="*60)

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-1.5B-Instruct",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=self.device,
        )
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()

        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"  ✓ Qwen loaded | VRAM used: {vram:.2f} GB")

        # Projector: the only thing that trains
        self.projector = UtoniaCrossAttentionProjector(
            utonia_dim=576,
            num_queries=32,
            llm_dim=1536,
        ).to(self.device)
        print(f"  ✓ Projector: {self.projector.num_parameters()/1e6:.2f}M params (trainable)")

        self.optimizer = torch.optim.AdamW(
            self.projector.parameters(),
            lr=args.lr,
            weight_decay=0.01,
        )

        def lr_lambda(step):
            if step < args.warmup_steps:
                return step / max(1, args.warmup_steps)
            return 1.0

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        os.makedirs(args.save_dir, exist_ok=True)

    def compute_loss(self, feat_cpu, offset_cpu, caption):
        """
        Loss = cross-entropy on caption tokens only.
        Input pipeline: cached feat (CPU) → projector (GPU) → Qwen (GPU)
        """
        # Move pre-computed features to GPU just-in-time
        feat   = feat_cpu.to(self.device)
        offset = offset_cpu.to(self.device)

        # Projector forward (trainable, gradients flow here)
        geo_tokens = self.projector(feat, offset).to(torch.bfloat16)  # [B, 32, 1536]

        # Tokenize prompt and caption
        prompt_text = (
            "<|im_start|>system\nYou are a 3D scene understanding assistant."
            "<|im_end|>\n<|im_start|>user\nDescribe the 3D scene."
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        caption_text = caption + "<|im_end|>"

        prompt_ids = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        caption_ids = self.tokenizer(
            caption_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)

        # Build embeddings: [prompt | geo_tokens | caption]
        embed_fn = self.llm.get_input_embeddings()
        prompt_emb  = embed_fn(prompt_ids)    # [1, P, 1536]
        caption_emb = embed_fn(caption_ids)   # [1, C, 1536]

        input_embeds = torch.cat([prompt_emb, geo_tokens, caption_emb], dim=1)

        # Labels: -100 = ignore loss; only predict caption tokens
        prefix_len = prompt_emb.shape[1] + 32
        labels = torch.full(
            (1, input_embeds.shape[1]), -100, dtype=torch.long, device=self.device
        )
        labels[:, prefix_len:] = caption_ids

        # Qwen forward (frozen, but gradients flow back through input_embeds)
        outputs = self.llm(inputs_embeds=input_embeds, labels=labels)
        return outputs.loss

    def train(self, feature_cache):
        dataset = CachedFeatureDataset(feature_cache)
        loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                             collate_fn=cached_collate, num_workers=0)

        global_step = 0
        best_loss = float("inf")
        log = []

        print(f"\n{'='*60}")
        print(f"  Stage 1 Training — Projector Only")
        print(f"  Training samples: {len(dataset)}")
        print(f"  Grad accumulation: {self.args.grad_accum}x  (effective batch = {self.args.grad_accum})")
        print(f"  Debug mode: {self.args.debug}")
        print(f"{'='*60}\n")

        for epoch in range(self.args.epochs):
            self.projector.train()
            epoch_loss = 0.0
            n_steps = 0
            self.optimizer.zero_grad()

            for step, (feat, offset, captions) in enumerate(loader):
                if self.args.debug and step >= 20:
                    print("  [DEBUG] Stopping after 20 steps.")
                    break

                try:
                    loss = self.compute_loss(feat, offset, captions[0])
                    (loss / self.args.grad_accum).backward()

                except torch.cuda.OutOfMemoryError:
                    print(f"  ⚠ OOM at step {step} — skipping batch.")
                    torch.cuda.empty_cache()
                    self.optimizer.zero_grad()
                    continue

                epoch_loss += loss.item()
                n_steps    += 1

                if (step + 1) % self.args.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.projector.parameters(), max_norm=1.0
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                if step % self.args.log_every == 0:
                    lr  = self.scheduler.get_last_lr()[0]
                    vram = torch.cuda.memory_allocated() / 1024**3
                    print(f"  Epoch {epoch+1} | Step {step:4d} | "
                          f"Loss: {loss.item():.4f} | "
                          f"LR: {lr:.2e} | VRAM: {vram:.2f}GB")

            avg_loss = epoch_loss / max(1, n_steps)
            log.append({"epoch": epoch + 1, "loss": avg_loss})
            print(f"\n  ✓ Epoch {epoch+1} done | Avg Loss: {avg_loss:.4f}\n")

            if avg_loss < best_loss:
                best_loss = avg_loss
                ckpt_path = os.path.join(self.args.save_dir, "stage1_best.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "projector_state": self.projector.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": {"utonia_dim": 576, "num_queries": 32, "llm_dim": 1536},
                }, ckpt_path)
                print(f"  💾 Best checkpoint → {ckpt_path}")

            if self.args.debug:
                break

        with open(os.path.join(self.args.save_dir, "stage1_log.json"), "w") as f:
            json.dump(log, f, indent=2)

        print(f"\n{'='*60}")
        print(f"  Training complete! Best loss: {best_loss:.4f}")
        print(f"  Checkpoint: {self.args.save_dir}/stage1_best.pt")
        print(f"  Next → python utonia_cap/inference.py \\")
        print(f"         --input ~/.cache/utonia/data/sample1.npz \\")
        print(f"         --checkpoint {self.args.save_dir}/stage1_best.pt")
        print(f"{'='*60}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nUtonia-Cap Training")
    print(f"Stage: {args.stage} | Data: {args.data} | "
          f"Debug: {args.debug} | Device: {device}")

    if args.stage == 1:
        # Step 1: Get raw dataset
        raw_dataset = get_dataset(args)

        # Step 2: Pre-compute features with Utonia (Phase A)
        debug_limit = 25 if args.debug else None
        feature_cache = precompute_features(raw_dataset, device, debug_limit)

        # Step 3: Train projector with Qwen frozen (Phase B)
        trainer = ProjectorTrainer(args)
        trainer.train(feature_cache)

    else:
        raise NotImplementedError(
            "Stage 2 LoRA coming next. Run Stage 1 first."
        )
