"""
Utonia-Cap: Full model assembling Utonia encoder + Projector + Qwen2.5-1.5B.

The complete forward pass:
    [Point Cloud] → Utonia (frozen) → Projector → [32 geometry tokens]
    [32 geometry tokens] + [text prompt tokens] → Qwen2.5-1.5B → [caption]

Training strategy:
    Stage 1:  Only projector parameters train. ~10M params updated.
    Stage 2:  Projector + LoRA adapters in Qwen train. ~18M params updated.
    (Utonia backbone is always frozen.)

Usage:
    conda activate utonia
    export PYTHONPATH=./
    python utonia_cap/model.py          # runs self-test, generates random caption
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, List

import utonia
from utonia_cap.projector import UtoniaCrossAttentionProjector


class UtoniaCap(nn.Module):
    """
    3D Point Cloud Captioning Model.

    Combines:
    - Utonia (Point Transformer V3) as the frozen 3D backbone
    - A cross-attention projector to bridge geometry → language
    - Qwen2.5-1.5B-Instruct as the language decoder
    """

    # Qwen2.5-1.5B hidden dim is 1536
    LLM_DIM = 1536

    def __init__(
        self,
        utonia_ckpt: str = "ckpt/utonia.pth",
        llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        num_queries: int = 32,
        utonia_bottleneck_dim: int = 576,
        freeze_utonia: bool = True,
        use_lora: bool = False,          # True only during Stage 2
        lora_rank: int = 16,
        lora_alpha: int = 32,
        device: str = "cuda",
    ):
        super().__init__()
        self.device_str = device
        self.num_queries = num_queries

        # ── 1. Load Utonia Encoder ─────────────────────────────────────────
        print(f"Loading Utonia from {utonia_ckpt}...")
        self.utonia = utonia.load(utonia_ckpt).to(device)

        if freeze_utonia:
            for param in self.utonia.parameters():
                param.requires_grad = False
            self.utonia.eval()
            print("  ✓ Utonia frozen (no gradients computed through backbone)")

        # Register hook to capture bottleneck features
        self._bottleneck_feats = None
        self._bottleneck_offset = None
        self._hook = self.utonia.enc.enc4.register_forward_hook(
            self._capture_bottleneck
        )

        # ── 2. Projector ───────────────────────────────────────────────────
        print(f"Building projector ({utonia_bottleneck_dim} → {num_queries} queries → {self.LLM_DIM})...")
        self.projector = UtoniaCrossAttentionProjector(
            utonia_dim=utonia_bottleneck_dim,
            num_queries=num_queries,
            llm_dim=self.LLM_DIM,
        ).to(device)
        print(f"  ✓ Projector: {self.projector.num_parameters()/1e6:.1f}M parameters")

        # ── 3. Load Qwen2.5-1.5B ──────────────────────────────────────────
        print(f"Loading LLM: {llm_name}")
        print("  (This downloads ~3GB on first run — please wait...)")

        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_name,
            trust_remote_code=True,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load in bfloat16 to save VRAM (~3GB instead of ~6GB)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map=device,
        )

        # Freeze LLM initially (Stage 1 only trains projector)
        for param in self.llm.parameters():
            param.requires_grad = False
        print(f"  ✓ LLM frozen (Stage 1 mode)")

        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)

    def _capture_bottleneck(self, module, input, output):
        """Hook: saves Utonia's Stage 4 encoder features during forward."""
        self._bottleneck_feats = output.feat
        self._bottleneck_offset = output.offset

    def _apply_lora(self, rank: int, alpha: int):
        """Adds LoRA adapters to Qwen's attention layers (Stage 2 only)."""
        from peft import get_peft_model, LoraConfig, TaskType

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        self.llm = get_peft_model(self.llm, lora_config)
        trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        print(f"  ✓ LoRA applied: {trainable/1e6:.1f}M trainable LLM params")

    def encode_point_cloud(self, point_dict: dict) -> torch.Tensor:
        """
        Run a point cloud through Utonia and project to LLM token space.

        Returns:
            geo_tokens: [1, num_queries, LLM_DIM]
        """
        with torch.no_grad() if not self.projector.training else torch.enable_grad():
            # Forward through Utonia — hook captures bottleneck automatically
            with torch.inference_mode():
                self.utonia(point_dict)

        # Projector is always trainable (has gradients)
        geo_tokens = self.projector(
            self._bottleneck_feats,
            self._bottleneck_offset,
        )
        return geo_tokens  # [1, 32, 1536]

    def forward(
        self,
        point_dict: dict,
        caption_ids: Optional[torch.Tensor] = None,
        prompt: str = "Describe the 3D scene in detail.",
    ):
        """
        Full forward pass.

        Training mode: pass caption_ids, returns cross-entropy loss.
        Inference mode: don't pass caption_ids, returns generated text.
        """
        # Step 1: Encode point cloud into geometry tokens
        geo_tokens = self.encode_point_cloud(point_dict)  # [B, 32, 1536]

        # Step 2: Build prompt tokens
        # Format: <system> + prompt text + [GEOMETRY TOKENS] + caption
        prompt_text = (
            "<|im_start|>system\nYou are a helpful 3D scene understanding assistant."
            "<|im_end|>\n<|im_start|>user\n"
            f"{prompt}\n"
        )
        prompt_ids = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(self.device_str)

        # Get text embeddings from LLM's embedding layer
        llm_embed = self.llm.get_input_embeddings()
        prompt_embeds = llm_embed(prompt_ids)  # [1, prompt_len, 1536]

        # Step 3: Concatenate [prompt] + [geometry tokens] as input
        # The LLM "reads" the geometry tokens as if they were words
        input_embeds = torch.cat([prompt_embeds, geo_tokens], dim=1)
        # Shape: [1, prompt_len + 32, 1536]

        if caption_ids is not None:
            # ── Training Mode ─────────────────────────────────────────────
            # Append caption tokens after geometry
            caption_embeds = llm_embed(caption_ids.to(self.device_str))
            input_embeds = torch.cat([input_embeds, caption_embeds], dim=1)

            # Labels: -100 means "don't compute loss here"
            # We only want loss on the caption tokens, not the prompt/geometry
            prefix_len = prompt_embeds.shape[1] + self.num_queries
            labels = torch.full(
                (1, input_embeds.shape[1]),
                fill_value=-100,
                dtype=torch.long,
                device=self.device_str,
            )
            labels[:, prefix_len:] = caption_ids

            outputs = self.llm(
                inputs_embeds=input_embeds,
                labels=labels,
            )
            return outputs.loss

        else:
            # ── Inference Mode ────────────────────────────────────────────
            # Add end of user turn
            end_token_ids = self.tokenizer(
                "<|im_end|>\n<|im_start|>assistant\n",
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(self.device_str)
            end_embeds = llm_embed(end_token_ids)
            input_embeds = torch.cat([input_embeds, end_embeds], dim=1)

            with torch.no_grad():
                generated = self.llm.generate(
                    inputs_embeds=input_embeds,
                    max_new_tokens=150,
                    do_sample=False,        # Greedy decoding (deterministic)
                    temperature=1.0,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the newly generated tokens
            caption = self.tokenizer.decode(
                generated[0],
                skip_special_tokens=True,
            )
            return caption

    def trainable_parameters(self):
        """Returns only the parameters that should receive gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def trainable_param_count(self):
        return sum(p.numel() for p in self.trainable_parameters())


# ── Quick Self-Test (runs when you execute this file directly) ────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM loading (just test projector shape)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Utonia-Cap Model Self-Test")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    if args.skip_llm:
        # Lightweight test: just verify projector
        from utonia_cap.projector import UtoniaCrossAttentionProjector
        proj = UtoniaCrossAttentionProjector().to(device)
        fake = torch.randn(800, 512).to(device)
        off = torch.tensor([800])
        out = proj(fake, off)
        print(f"Projector output: {out.shape}")
        print("✓ Projector OK — use --skip-llm=False to test full model")
    else:
        print("Loading full model (this will download Qwen2.5-1.5B on first run)...")
        model = UtoniaCap(device=device)
        print(f"\nTrainable parameters: {model.trainable_param_count()/1e6:.1f}M")

        # Load sample data and run inference
        print("\nLoading sample point cloud...")
        data_path = os.path.expanduser("~/.cache/utonia/data/sample1.npz")
        point = dict(np.load(data_path))
        point.pop("segment200")
        segment = point.pop("segment20")
        point["segment"] = segment

        transform = utonia.transform.default(0.5)
        point = transform(point)
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].to(device)

        print("\nGenerating caption (Stage 1 — model not yet trained, expect random output)...")
        caption = model(point, prompt="Describe the 3D scene in detail.")
        print(f"\n{'='*60}")
        print(f"  Generated Caption:")
        print(f"  {caption}")
        print(f"{'='*60}")
        print("\n✓ Full pipeline working! Caption will improve after training.")
        print("→ Next step: run  python utonia_cap/dataset.py")
