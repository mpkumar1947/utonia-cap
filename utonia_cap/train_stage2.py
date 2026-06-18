"""
Utonia-Cap Stage 2: LoRA fine-tuning of Qwen2.5-1.5B + projector.

Stage 1 taught the projector to map geometry → language tokens.
Stage 2 teaches the LLM to *use* those tokens to write good captions.

What changes vs Stage 1:
  - Qwen gets LoRA adapters on q_proj, k_proj, v_proj, o_proj
  - Both projector AND LoRA adapters receive gradients
  - Trainable params: ~5.1M (projector) + ~8M (LoRA) = ~13M total
  - VRAM stays at ~4.2GB (LoRA is parameter-efficient)

Usage:
    conda activate utonia
    export PYTHONPATH=./

    # Start Stage 2 from Stage 1 checkpoint
    python utonia_cap/train_stage2.py \\
        --stage1-checkpoint checkpoints/stage1_best.pt \\
        --data cap3d \\
        --epochs 5

    # Debug run (10 steps)
    python utonia_cap/train_stage2.py --debug
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse, json, gc
import numpy as np
import torch
from torch.utils.data import DataLoader

import utonia
from utonia_cap.projector import UtoniaCrossAttentionProjector
from utonia_cap.dataset import collate_fn
from utonia_cap.train import (
    get_dataset, precompute_features,
    CachedFeatureDataset, cached_collate,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Utonia-Cap Stage 2 Training")
    parser.add_argument("--stage1-checkpoint", default="checkpoints/stage1_best.pt",
                        help="Path to Stage 1 projector checkpoint")
    parser.add_argument("--data", default="augmented",
                        choices=["augmented", "cap3d", "scanrefer"])
    parser.add_argument("--data-dir", default=os.path.expanduser("~/.cache/utonia"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4,
                        help="Learning rate for LoRA adapters")
    parser.add_argument("--proj-lr", type=float, default=5e-5,
                        help="Learning rate for projector (usually lower than LoRA LR)")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def apply_lora(llm, rank: int, alpha: int):
    """Add LoRA adapters to Qwen attention layers."""
    from peft import get_peft_model, LoraConfig, TaskType

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    llm = get_peft_model(llm, config)
    trainable = sum(p.numel() for p in llm.parameters() if p.requires_grad)
    print(f"  LoRA applied: {trainable/1e6:.1f}M trainable LLM params (rank={rank})")
    return llm


def compute_loss(feat_cpu, offset_cpu, caption, projector, llm, tokenizer, device):
    """Same loss as Stage 1 but now both projector and LoRA adapters have gradients."""
    feat   = feat_cpu.to(device)
    offset = offset_cpu.to(device)

    geo_tokens = projector(feat, offset).to(torch.bfloat16)

    prompt_text = (
        "<|im_start|>system\nYou are a 3D scene understanding assistant.<|im_end|>\n"
        "<|im_start|>user\nDescribe the 3D scene in detail.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    caption_text = caption + "<|im_end|>"

    prompt_ids  = tokenizer(prompt_text,  return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    caption_ids = tokenizer(caption_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

    embed_fn    = llm.get_input_embeddings()
    prompt_emb  = embed_fn(prompt_ids)
    caption_emb = embed_fn(caption_ids)
    input_embeds = torch.cat([prompt_emb, geo_tokens, caption_emb], dim=1)

    prefix_len = prompt_emb.shape[1] + 32
    labels = torch.full(
        (1, input_embeds.shape[1]), -100, dtype=torch.long, device=device
    )
    labels[:, prefix_len:] = caption_ids

    outputs = llm(inputs_embeds=input_embeds, labels=labels)
    return outputs.loss


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Utonia-Cap Stage 2: LoRA Fine-Tuning")
    print(f"  Data: {args.data} | Epochs: {args.epochs} | Device: {device}")
    print(f"{'='*60}\n")

    # ── Phase A: Pre-compute Utonia features ──────────────────────────────
    # (same as Stage 1 — Utonia is always frozen)
    class FakeArgs:
        data = args.data
        data_dir = args.data_dir
        debug = args.debug

    fake_args = FakeArgs()
    raw_dataset = get_dataset(fake_args)
    debug_limit = 15 if args.debug else None
    feature_cache = precompute_features(raw_dataset, device, debug_limit)

    # ── Phase B: Load Qwen + apply LoRA ───────────────────────────────────
    print("="*60)
    print("  Phase B: Loading Qwen + applying LoRA")
    print("="*60)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )
    llm = apply_lora(llm, args.lora_rank, args.lora_alpha)
    llm.train()

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM after Qwen+LoRA: {vram:.2f} GB")

    # ── Load projector from Stage 1 ───────────────────────────────────────
    projector = UtoniaCrossAttentionProjector(
        utonia_dim=576, num_queries=32, llm_dim=1536
    ).to(device)

    if os.path.exists(args.stage1_checkpoint):
        ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt["projector_state"])
        print(f"  ✓ Stage 1 projector loaded (epoch {ckpt.get('epoch','?')}, loss {ckpt.get('loss',0):.4f})")
    else:
        print("  ⚠ No Stage 1 checkpoint found — projector randomly initialized")

    # ── Optimizer: separate LRs for projector vs LoRA ─────────────────────
    optimizer = torch.optim.AdamW([
        {"params": projector.parameters(),                                  "lr": args.proj_lr},
        {"params": [p for p in llm.parameters() if p.requires_grad],       "lr": args.lr},
    ], weight_decay=0.01)

    def lr_lambda(step):
        return step / max(1, args.warmup_steps) if step < args.warmup_steps else 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training Loop ─────────────────────────────────────────────────────
    dataset = CachedFeatureDataset(feature_cache)
    loader  = DataLoader(dataset, batch_size=1, shuffle=True,
                         collate_fn=cached_collate, num_workers=0)

    best_loss = float("inf")
    log = []

    print(f"\n{'='*60}")
    print(f"  Stage 2 Training Loop")
    print(f"  Samples: {len(dataset)} | Grad accum: {args.grad_accum}x")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        projector.train()
        llm.train()
        epoch_loss, n_steps = 0.0, 0
        optimizer.zero_grad()

        for step, (feat, offset, captions) in enumerate(loader):
            if args.debug and step >= 10:
                print("  [DEBUG] Stopping at 10 steps.")
                break

            try:
                loss = compute_loss(
                    feat, offset, captions[0],
                    projector, llm, tokenizer, device
                )
                (loss / args.grad_accum).backward()

            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠ OOM at step {step} — skipping")
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                continue

            epoch_loss += loss.item()
            n_steps += 1

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(projector.parameters()) +
                    [p for p in llm.parameters() if p.requires_grad],
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % args.log_every == 0:
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"  Epoch {epoch+1} | Step {step:4d} | "
                      f"Loss: {loss.item():.4f} | VRAM: {vram:.2f}GB")

        avg_loss = epoch_loss / max(1, n_steps)
        log.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"\n  ✓ Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save projector + LoRA adapters together
            ckpt_path = os.path.join(args.save_dir, "stage2_best.pt")
            llm.save_pretrained(os.path.join(args.save_dir, "lora_adapter"))
            torch.save({
                "epoch": epoch + 1,
                "projector_state": projector.state_dict(),
                "loss": avg_loss,
                "config": {"utonia_dim": 576, "num_queries": 32, "llm_dim": 1536},
            }, ckpt_path)
            print(f"  💾 Best checkpoint → {ckpt_path}")
            print(f"     LoRA adapter → {args.save_dir}/lora_adapter/\n")

        if args.debug:
            break

    with open(os.path.join(args.save_dir, "stage2_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Stage 2 complete! Best loss: {best_loss:.4f}")
    print(f"  Run inference: python utonia_cap/inference.py \\")
    print(f"    --checkpoint checkpoints/stage2_best.pt \\")
    print(f"    --lora-adapter checkpoints/lora_adapter \\")
    print(f"    --input ~/.cache/utonia/data/sample1.npz")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
