"""
Unified End-to-End Training for the Remote Server.
Uses UtoniaCap (the actual model class from model.py) with LoRA enabled.
Trains on full Cap3D dataset (660K objects) streaming from zip files.
Runs on GPU 0 with gradient accumulation to simulate large batches.
"""

import os
import sys
import argparse
import gc
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure PYTHONPATH is set correctly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import utonia
from utonia_cap.dataset import Cap3DDataset, collate_fn
from utonia_cap.model import UtoniaCap


def train_server(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 1. Load Dataset
    print(f"\nLoading Cap3D dataset from: {args.data_dir}")
    dataset = Cap3DDataset(args.data_dir, split="train", max_points=args.max_points)

    if len(dataset) == 0:
        print("ERROR: Dataset is empty! Check the data_dir path and CSV file.")
        return

    # Optionally cap the dataset size for practical training time
    if args.max_samples and args.max_samples < len(dataset):
        import random
        indices = random.sample(range(len(dataset)), args.max_samples)
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Capped dataset to {args.max_samples:,} random samples")

    # num_workers=0 avoids multiprocessing issues with zip file handles
    # pin_memory=False reduces RAM/swap pressure over long runs
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    print(f"Dataloader ready: {len(dataset):,} objects, {len(dataloader):,} batches/epoch")

    # 2. Build the Full Model (UtoniaCap with LoRA enabled)
    print("\nBuilding UtoniaCap model with LoRA...")
    model = UtoniaCap(
        utonia_ckpt=args.utonia_ckpt,
        llm_name="Qwen/Qwen2.5-1.5B-Instruct",
        num_queries=32,
        freeze_utonia=True,
        use_lora=True,          # Enable LoRA adapters for LLM
        lora_rank=8,
        lora_alpha=16,
        device=device,
    )
    model.projector.train()
    model.llm.train()

    trainable = model.trainable_param_count()
    print(f"Trainable parameters: {trainable / 1e6:.1f}M")

    # 3. Optimizer (only trainable params — projector + LoRA)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr)

    # 4. Training Loop
    os.makedirs(args.out_dir, exist_ok=True)
    accumulation_steps = args.accumulate
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"grad_accum={accumulation_steps} (effective batch={args.batch_size * accumulation_steps})")

    for epoch in range(args.epochs):
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (point_dict, captions) in enumerate(pbar):
            # Move point cloud tensors to GPU
            for k, v in point_dict.items():
                if isinstance(v, torch.Tensor):
                    point_dict[k] = v.to(device)

            # Tokenize captions
            caption_enc = model.tokenizer(
                captions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            caption_ids = caption_enc.input_ids.to(device)

            try:
                # End-to-end forward pass → returns loss
                loss = model(point_dict, caption_ids=caption_ids)

                # Normalize for gradient accumulation
                loss = loss / accumulation_steps
                loss.backward()

                if (step + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                total_loss += loss.item() * accumulation_steps
                n_batches += 1
                pbar.set_postfix(
                    loss=f"{loss.item() * accumulation_steps:.4f}",
                    gpu=f"{torch.cuda.memory_allocated() / 1024**3:.1f}GB"
                )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nOOM on step {step}. Skipping batch and clearing cache.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                else:
                    raise e

            # Periodic full garbage collection every 500 steps
            # Prevents PyTorch allocator cache from slowly filling swap
            if step % 500 == 0 and step > 0:
                gc.collect()
                torch.cuda.empty_cache()

        # Filter out any NaN values before averaging
        avg_loss = total_loss / max(n_batches, 1) if n_batches > 0 else float('nan')
        avg_str = f"{avg_loss:.4f}" if avg_loss == avg_loss else "nan (check for OOM cascades)"
        print(f"\n✓ Epoch {epoch+1} complete. Avg Loss: {avg_str}")

        # Save checkpoint
        proj_path = os.path.join(args.out_dir, f"projector_server_epoch{epoch+1}.pt")
        lora_path = os.path.join(args.out_dir, "lora_adapter_server")
        torch.save(model.projector.state_dict(), proj_path)
        model.llm.save_pretrained(lora_path)
        print(f"  Saved → {proj_path}")
        print(f"  Saved → {lora_path}/")

    print("\n✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str,   default="/mnt/zone_b/Datasets/utonia_cap/PointCloud_pt_zips")
    parser.add_argument("--utonia_ckpt", type=str,   default="./ckpt/utonia.pth")
    parser.add_argument("--out_dir",     type=str,   default="./checkpoints")
    parser.add_argument("--epochs",      type=int,   default=5)
    parser.add_argument("--batch_size",  type=int,   default=1)
    parser.add_argument("--accumulate",  type=int,   default=16)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--max_points",  type=int,   default=60000)
    parser.add_argument("--max_samples", type=int, default=100000, help="Max training samples per epoch (default 100K)")
    args = parser.parse_args()
    train_server(args)
