"""
Utonia-Cap Inference: Generate a text caption for any 3D point cloud.

This is the "show me it works" script — the one you'll demo in interviews.

Usage:
    conda activate utonia
    export PYTHONPATH=./

    # Caption the indoor scene we already have
    python utonia_cap/inference.py \
        --input ~/.cache/utonia/data/sample1.npz \
        --checkpoint checkpoints/stage1_best.pt

    # Caption a custom PLY file (e.g. from MeshLab)
    python utonia_cap/inference.py --input my_room.ply

    # Run interactively — type different prompts
    python utonia_cap/inference.py --input sample1.npz --interactive
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time
import numpy as np
import torch
import open3d as o3d

import utonia
from utonia_cap.projector import UtoniaCrossAttentionProjector


PROMPT_PRESETS = {
    "describe": "Describe the 3D scene in detail.",
    "objects": "What objects are visible in this scene?",
    "spatial": "Describe the spatial layout of this 3D scene.",
    "short": "Give a short one-sentence description of this scene.",
    "room": "What type of room is this and what is it used for?",
}


def load_point_cloud(path: str) -> dict:
    """
    Load a point cloud from .npz (Utonia format) or .ply (MeshLab/Open3D).
    Returns a dict with coord, color, normal keys.
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        data = dict(np.load(path))
        # Remove keys that aren't needed for inference
        for key in ["segment20", "segment200", "instance"]:
            data.pop(key, None)
        return data

    elif ext == ".ply":
        pcd = o3d.io.read_point_cloud(path)
        coord = np.asarray(pcd.points, dtype=np.float32)
        color = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.ones_like(coord)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        normal = np.asarray(pcd.normals, dtype=np.float32)
        return {"coord": coord, "color": color, "normal": normal}

    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .npz or .ply")


def run_inference(
    input_path: str,
    checkpoint_path: str = None,
    prompt: str = "Describe the 3D scene in detail.",
    device: str = None,
    grid_size: float = 0.5,
    lora_adapter: str = None,
) -> str:
    import gc
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"  Utonia-Cap Inference")
    print(f"  Input:   {input_path}")
    print(f"  Prompt:  '{prompt}'")
    print(f"  Device:  {device}")
    print(f"{'='*60}\n")

    # ── 1. Load Point Cloud ───────────────────────────────────────────────
    print("[1/4] Loading point cloud...")
    raw_point = load_point_cloud(input_path)
    # Ensure normal exists
    if "normal" not in raw_point:
        n = raw_point["coord"].shape[0]
        raw_point["normal"] = np.zeros((n, 3), dtype=np.float32)
    n_points = raw_point["coord"].shape[0]
    print(f"  {n_points:,} points loaded")

    # ── 2. Transform + Encode with Utonia ─────────────────────────────────
    print("[2/4] Encoding with Utonia backbone...")
    utonia_model = utonia.load("ckpt/utonia.pth").to(device)
    utonia_model.eval()
    for p in utonia_model.parameters():
        p.requires_grad = False

    bottleneck_feats = {}
    def hook_fn(module, input, output):
        # Store on CPU immediately to free GPU VRAM before Qwen loads
        bottleneck_feats["feat"]   = output.feat.detach().cpu()
        bottleneck_feats["offset"] = output.offset.detach().cpu()
    utonia_model.enc.enc4.register_forward_hook(hook_fn)

    transform = utonia.transform.default(grid_size)
    point = transform(dict(raw_point))
    for key in point.keys():
        if isinstance(point[key], torch.Tensor):
            point[key] = point[key].to(device)

    with torch.inference_mode():
        utonia_model(point)

    print(f"  Bottleneck tokens: {bottleneck_feats['feat'].shape}")

    # Unload Utonia to free ~1.5GB VRAM before loading Qwen
    del utonia_model, point
    gc.collect()
    torch.cuda.empty_cache()

    # ── 3. Project to LLM Space ───────────────────────────────────────────
    print("[3/4] Running projector...")
    projector = UtoniaCrossAttentionProjector(
        utonia_dim=576, num_queries=32, llm_dim=1536
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt["projector_state"])
        epoch = ckpt.get("epoch", "?")
        loss  = ckpt.get("loss", float("nan"))
        print(f"  ✓ Checkpoint loaded (epoch {epoch}, loss {loss:.4f})")
    else:
        print("  ⚠ No checkpoint — projector randomly initialized (results will be poor)")

    projector.eval()
    with torch.no_grad():
        geo_tokens = projector(
            bottleneck_feats["feat"].to(device),
            bottleneck_feats["offset"].to(device),
        ).to(torch.bfloat16)   # [1, 32, 1536]

    print(f"  Geometry tokens: {geo_tokens.shape}")

    # ── 4. Generate Caption with Qwen ─────────────────────────────────────
    print("[4/4] Generating caption...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=device,
    )
    llm.eval()

    # Load LoRA adapters if Stage 2 checkpoint provided
    if lora_adapter and os.path.exists(lora_adapter):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, lora_adapter)
        llm.eval()
        print(f"  ✓ LoRA adapter loaded from {lora_adapter}")

    prompt_text = (
        "<|im_start|>system\nYou are a helpful 3D scene understanding assistant. "
        "Geometric tokens have been embedded into your input representing a real 3D point cloud. "
        "Use them to generate an accurate, specific description.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(device)

    llm_embed     = llm.get_input_embeddings()
    prompt_embeds = llm_embed(prompt_ids)                          # [1, P, 1536]
    input_embeds  = torch.cat([prompt_embeds, geo_tokens], dim=1) # [1, P+32, 1536]

    # Build attention mask (all ones — attend to everything)
    attention_mask = torch.ones(
        input_embeds.shape[:2], dtype=torch.long, device=device
    )

    t_start = time.time()
    with torch.no_grad():
        generated_ids = llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            max_new_tokens=200,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    t_elapsed = time.time() - t_start

    caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    # Strip any system/user prompt leakage
    for marker in ["assistant\n", "assistant:"]:
        if marker in caption:
            caption = caption.split(marker)[-1].strip()

    print(f"\n{'='*60}")
    print(f"  Generated Caption:")
    print(f"  {caption}")
    print(f"{'='*60}")
    print(f"\n  Generation time: {t_elapsed:.1f}s")

    return caption



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utonia-Cap Inference")
    parser.add_argument("--input", required=True,
                        help="Path to .npz or .ply point cloud file")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to trained projector checkpoint")
    parser.add_argument("--prompt", default="describe",
                        help=f"Prompt preset or custom text. Presets: {list(PROMPT_PRESETS.keys())}")
    parser.add_argument("--lora-adapter", default=None,
                        help="Path to LoRA adapter dir from Stage 2 (optional)")
    parser.add_argument("--grid-size", type=float, default=0.5)
    parser.add_argument("--interactive", action="store_true",
                        help="Run multiple prompts interactively")
    args = parser.parse_args()

    # Resolve prompt preset
    prompt = PROMPT_PRESETS.get(args.prompt, args.prompt)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.interactive:
        print("\nInteractive mode — type prompts, 'quit' to exit, 'list' to see presets")
        while True:
            user_prompt = input("\nPrompt > ").strip()
            if user_prompt.lower() == "quit":
                break
            if user_prompt.lower() == "list":
                for k, v in PROMPT_PRESETS.items():
                    print(f"  {k}: {v}")
                continue
            resolved = PROMPT_PRESETS.get(user_prompt, user_prompt)
            run_inference(args.input, args.checkpoint, resolved, device,
                         args.grid_size, args.lora_adapter)
    else:
        run_inference(args.input, args.checkpoint, prompt, device,
                     args.grid_size, args.lora_adapter)
