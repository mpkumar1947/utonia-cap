"""
Unified End-to-End Training for the Remote Server.
Loads Utonia (frozen) and Qwen (LoRA) simultaneously on GPU 0.
Uses gradient accumulation to simulate large batch sizes while fitting in 8GB VRAM.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import utonia
from utonia_cap.dataset import Cap3DDataset, collate_fn
from utonia_cap.model import UtoniaCapPipeline
from utonia_cap.projector import SimpleCrossAttentionProjector

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def train_server(args):
    # Set device to GPU 0 (the free 8GB GPU on the server)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    print(f"Loading Cap3D dataset from zip files...")
    dataset = Cap3DDataset(args.data_dir, split="train", max_points=args.max_points)
    # Batch size 2 is safe for 8GB VRAM when both models are loaded
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)

    # 2. Utonia Backbone (Frozen)
    print("Loading Utonia backbone...")
    encoder = utonia.model.default(pretrained=args.utonia_ckpt)
    encoder = encoder.to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # 3. Projector (Trainable)
    print("Initializing Projector...")
    projector = SimpleCrossAttentionProjector(
        in_dim=576,   # Utonia features
        out_dim=1536, # Qwen2.5-1.5B embed_dim
        num_heads=8,
        num_query_tokens=64
    )
    projector = projector.to(device)
    projector.train()

    # 4. LLM with LoRA (Trainable)
    print("Loading LLM (Qwen2.5-1.5B) and injecting LoRA adapters...")
    llm_name = "Qwen/Qwen2.5-1.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(llm_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load in bfloat16 to save memory
    llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.bfloat16)
    
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
    )
    llm = get_peft_model(llm, lora_config)
    llm = llm.to(device)
    llm.train()
    llm.print_trainable_parameters()

    # 5. Pipeline
    pipeline = UtoniaCapPipeline(encoder, projector, llm, tokenizer)
    pipeline.train()

    # Optimizer (only train projector and LoRA parameters)
    trainable_params = list(projector.parameters()) + list(llm.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    print(f"Starting End-to-End Server Training for {args.epochs} epochs!")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    accumulation_steps = args.accumulate
    
    for epoch in range(args.epochs):
        total_loss = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for step, (point_dict, captions) in enumerate(pbar):
            # Move points to device
            for k, v in point_dict.items():
                if isinstance(v, torch.Tensor):
                    point_dict[k] = v.to(device)

            # E2E Forward Pass
            try:
                loss = pipeline(point_dict, captions)
                
                # Normalize loss to account for accumulation
                loss = loss / accumulation_steps
                loss.backward()
                
                if (step + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                total_loss += loss.item() * accumulation_steps
                pbar.set_postfix(loss=f"{loss.item() * accumulation_steps:.4f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"OOM on step {step}. Emptying cache and skipping batch.")
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                else:
                    raise e
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Complete. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        torch.save(projector.state_dict(), os.path.join(args.out_dir, f"projector_e2e_epoch{epoch+1}.pt"))
        llm.save_pretrained(os.path.join(args.out_dir, "lora_adapter_e2e"))
        print(f"Saved checkpoint for Epoch {epoch+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/mnt/zone_b/Datasets/utonia_cap")
    parser.add_argument("--utonia_ckpt", type=str, default="./ckpt/utonia.pth")
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_points", type=int, default=60000, help="Cap points to save VRAM")
    args = parser.parse_args()
    
    train_server(args)
