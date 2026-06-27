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
import json
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

    # Resume from a saved checkpoint if specified
    start_epoch = 0
    start_step = 0
    
    # Check for a mid-epoch checkpoint first (highest priority)
    mid_proj_path = os.path.join(args.out_dir, "projector_server_latest.pt")
    mid_lora_path = os.path.join(args.out_dir, "lora_adapter_latest")
    step_file = os.path.join(args.out_dir, "latest_step.json")
    
    if os.path.exists(step_file) and os.path.exists(mid_proj_path):
        with open(step_file, 'r') as f:
            state = json.load(f)
            start_epoch = state['epoch']
            start_step = state['step']
            
        print(f"\nFound MID-EPOCH checkpoint! Resuming Epoch {start_epoch+1} at Step {start_step}...")
        model.projector.load_state_dict(torch.load(mid_proj_path, map_location=device))
        
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm.base_model.model, mid_lora_path, is_trainable=True)
        print(f"  ✓ Loaded projector & LoRA from mid-epoch backup")
        
    elif args.resume_epoch > 0:
        proj_path = os.path.join(args.out_dir, f"projector_server_epoch{args.resume_epoch}.pt")
        lora_path = os.path.join(args.out_dir, "lora_adapter_server")
        if os.path.exists(proj_path):
            model.projector.load_state_dict(torch.load(proj_path, map_location=device))
            print(f"  ✓ Loaded projector from {proj_path}")
        else:
            print(f"  WARNING: Projector checkpoint not found at {proj_path}")
        if os.path.exists(lora_path):
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(model.llm.base_model.model, lora_path, is_trainable=True)
            print(f"  ✓ Loaded LoRA adapter from {lora_path}")
        else:
            print(f"  WARNING: LoRA adapter not found at {lora_path}")
        start_epoch = args.resume_epoch
        print(f"  Resuming from the beginning of epoch {start_epoch + 1}")

    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"grad_accum={accumulation_steps} (effective batch={args.batch_size * accumulation_steps})")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{start_epoch + args.epochs}")
        for step, (point_dict, captions) in enumerate(pbar):
            # Fast-forward dataloader if we are resuming mid-epoch
            if step < start_step:
                if step % 1000 == 0:
                    pbar.set_postfix(skip=f"Skipping to step {start_step}...")
                continue
            
            # Reset start_step so future epochs don't skip
            if step == start_step:
                start_step = 0
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
                
                # Aggressively free memory at the end of every step
                del loss
                del point_dict
                del caption_enc
                del caption_ids

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\nOOM on step {step}. Skipping batch and clearing cache.")
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.zero_grad()
                else:
                    raise e

            # Periodic full garbage collection every 500 steps
            if step % 500 == 0 and step > 0:
                gc.collect()
                torch.cuda.empty_cache()
                
            # MID-EPOCH CHECKPOINT: Save progress every 5,000 steps so we don't lose work if killed
            if step % 5000 == 0 and step > 0:
                torch.save(model.projector.state_dict(), mid_proj_path)
                model.llm.save_pretrained(mid_lora_path)
                with open(step_file, 'w') as f:
                    json.dump({'epoch': epoch, 'step': step + 1}, f)
                pbar.set_postfix(saved=f"Backup step {step}")

        # Filter out any NaN values before averaging
        avg_loss = total_loss / max(n_batches, 1) if n_batches > 0 else float('nan')
        avg_str = f"{avg_loss:.4f}" if avg_loss == avg_loss else "nan (check for OOM cascades)"
        print(f"\n✓ Epoch {epoch+1} complete. Avg Loss: {avg_str}")

        # Save checkpoint with absolute epoch number
        proj_path = os.path.join(args.out_dir, f"projector_server_epoch{epoch+1}.pt")
        lora_path = os.path.join(args.out_dir, "lora_adapter_server")
        torch.save(model.projector.state_dict(), proj_path)
        model.llm.save_pretrained(lora_path)
        print(f"  Saved → {proj_path}")
        print(f"  Saved → {lora_path}/")
        
        # Clear mid-epoch backup since we successfully completed the epoch
        if os.path.exists(step_file):
            os.remove(step_file)

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
    parser.add_argument("--max_points",  type=int,   default=20000)
    parser.add_argument("--max_samples", type=int,   default=100000, help="Max training samples per epoch (default 100K)")
    parser.add_argument("--resume_epoch",type=int,   default=0,      help="Resume from this epoch number (e.g. 2 loads epoch2 checkpoint)")
    args = parser.parse_args()
    train_server(args)
