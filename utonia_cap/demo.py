"""
Utonia-Cap Gradio Demo
======================
Upload a 3D point cloud (.npz or .ply) → get a natural language caption.

Run:
    conda activate utonia
    export PYTHONPATH=~/Desktop/utonia/repo
    cd ~/Desktop/utonia/repo
    python utonia_cap/demo.py
"""

import sys, os, gc, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import gradio as gr
import plotly.graph_objects as go

import utonia
from utonia_cap.projector import UtoniaCrossAttentionProjector

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_S2     = os.path.join(REPO_DIR, "checkpoints", "stage2_best.pt")
CKPT_S1     = os.path.join(REPO_DIR, "checkpoints", "stage1_best.pt")
LORA_DIR    = os.path.join(REPO_DIR, "checkpoints", "lora_adapter")
UTONIA_CKPT = os.path.join(REPO_DIR, "ckpt", "utonia.pth")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_PRESETS = {
    "Describe the scene in detail":   "Describe the 3D scene in detail.",
    "List all visible objects":        "What objects are visible in this scene?",
    "Describe the spatial layout":     "Describe the spatial layout and arrangement of this 3D scene.",
    "Identify the room type":          "What type of room is this and what is it used for?",
    "One-sentence summary":            "Give a short one-sentence description of this 3D scene.",
}

EXAMPLE_FILES = {
    "Indoor Room (sample1)": os.path.expanduser("~/.cache/utonia/data/sample1.npz"),
    "Outdoor LiDAR (sample2)": os.path.expanduser("~/.cache/utonia/data/sample2_outdoor_multiframe.npz"),
    "Object (sample3)": os.path.expanduser("~/.cache/utonia/data/sample3_object.npz"),
}

# ── Model loading (cached globally so we don't reload every click) ─────────────

_qwen_cache = {}   # holds tokenizer + llm after first load

def load_qwen_with_lora():
    if "llm" in _qwen_cache:
        return _qwen_cache["tokenizer"], _qwen_cache["llm"]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=DEVICE,
    )

    if os.path.exists(LORA_DIR):
        from peft import PeftModel
        llm = PeftModel.from_pretrained(llm, LORA_DIR)

    llm.eval()
    _qwen_cache["tokenizer"] = tokenizer
    _qwen_cache["llm"] = llm
    return tokenizer, llm


# ── Core inference function ────────────────────────────────────────────────────

def caption_pointcloud(file_obj, prompt_choice, custom_prompt):
    """Main inference function called by Gradio."""

    if file_obj is None:
        return "⚠ Please upload a .npz or .ply file.", None, ""

    # Resolve prompt
    prompt = custom_prompt.strip() if custom_prompt.strip() else PROMPT_PRESETS.get(prompt_choice, prompt_choice)

    try:
        # ── 1. Load point cloud ────────────────────────────────────────────
        path = file_obj.name if hasattr(file_obj, "name") else file_obj
        ext  = os.path.splitext(path)[1].lower()

        if ext == ".npz":
            raw = dict(np.load(path, allow_pickle=True))
            for k in ["segment20", "segment200", "instance", "caption"]:
                raw.pop(k, None)
        elif ext == ".ply":
            import open3d as o3d
            pcd    = o3d.io.read_point_cloud(path)
            coord  = np.asarray(pcd.points, dtype=np.float32)
            color  = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else np.ones_like(coord) * 0.5
            pcd.estimate_normals()
            normal = np.asarray(pcd.normals, dtype=np.float32)
            raw    = {"coord": coord, "color": color, "normal": normal}
        else:
            return f"⚠ Unsupported format: {ext}. Use .npz or .ply", None, ""

        if "normal" not in raw:
            raw["normal"] = np.zeros((raw["coord"].shape[0], 3), dtype=np.float32)

        n_points = raw["coord"].shape[0]

        # ── 2. Build point cloud visualisation (subsample for speed) ──────
        viz_fig = build_pointcloud_viz(raw, max_pts=20000)

        # ── 3. Encode with Utonia ──────────────────────────────────────────
        utonia_model = utonia.load(UTONIA_CKPT).to(DEVICE)
        utonia_model.eval()
        for p in utonia_model.parameters():
            p.requires_grad = False

        captured = {}
        def hook(m, i, o):
            captured["feat"]   = o.feat.detach().cpu()
            captured["offset"] = o.offset.detach().cpu()
        utonia_model.enc.enc4.register_forward_hook(hook)

        transform = utonia.transform.default(0.5)
        point = transform(dict(raw))
        for k in point:
            if isinstance(point[k], torch.Tensor):
                point[k] = point[k].to(DEVICE)

        with torch.inference_mode():
            utonia_model(point)

        del utonia_model, point
        gc.collect()
        torch.cuda.empty_cache()

        # ── 4. Project to LLM space ────────────────────────────────────────
        ckpt_path = CKPT_S2 if os.path.exists(CKPT_S2) else CKPT_S1
        projector = UtoniaCrossAttentionProjector(
            utonia_dim=576, num_queries=32, llm_dim=1536
        ).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        projector.load_state_dict(ckpt["projector_state"])
        projector.eval()

        with torch.no_grad():
            geo_tokens = projector(
                captured["feat"].to(DEVICE),
                captured["offset"].to(DEVICE),
            ).to(torch.bfloat16)

        # ── 5. Generate caption with Qwen ──────────────────────────────────
        tokenizer, llm = load_qwen_with_lora()

        prompt_text = (
            "<|im_start|>system\nYou are a 3D scene understanding assistant. "
            "Geometric tokens representing a real 3D point cloud are embedded in your input. "
            "Use them to generate an accurate, specific description.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        prompt_ids = tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(DEVICE)

        embed_fn     = llm.get_input_embeddings()
        prompt_emb   = embed_fn(prompt_ids)
        input_embeds = torch.cat([prompt_emb, geo_tokens], dim=1)
        attn_mask    = torch.ones(input_embeds.shape[:2], dtype=torch.long, device=DEVICE)

        t0 = time.time()
        with torch.no_grad():
            out_ids = llm.generate(
                inputs_embeds=input_embeds,
                attention_mask=attn_mask,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0

        caption = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        for marker in ["assistant\n", "assistant:"]:
            if marker in caption:
                caption = caption.split(marker)[-1].strip()

        stats = (
            f"**Points:** {n_points:,}  |  "
            f"**Tokens:** {captured['feat'].shape[0]:,}  |  "
            f"**Generated in:** {elapsed:.1f}s  |  "
            f"**VRAM:** {torch.cuda.memory_allocated()/1024**3:.2f} GB"
        )

        return caption, viz_fig, stats

    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n\n```\n{traceback.format_exc()}\n```", None, ""


def build_pointcloud_viz(raw: dict, max_pts: int = 20000) -> go.Figure:
    """Build a Plotly 3D scatter plot of the point cloud."""
    coord = raw["coord"].astype(np.float32)
    color = raw.get("color", None)

    # Subsample for browser performance
    N = coord.shape[0]
    if N > max_pts:
        idx   = np.random.choice(N, max_pts, replace=False)
        coord = coord[idx]
        if color is not None:
            color = color[idx]

    # Normalise colour to [0,1]
    if color is not None:
        c = color.astype(np.float32)
        if c.max() > 1.0:
            c = c / 255.0
        rgb_str = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})" for r, g, b in c]
    else:
        rgb_str = "#60a5fa"   # blue fallback

    fig = go.Figure(data=[go.Scatter3d(
        x=coord[:, 0], y=coord[:, 1], z=coord[:, 2],
        mode="markers",
        marker=dict(size=1.2, color=rgb_str, opacity=0.85),
    )])
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
    )
    return fig


def load_example(name):
    path = EXAMPLE_FILES.get(name)
    if path and os.path.exists(path):
        return path
    return None


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --surface: #0f0f1a;
    --surface-2: #1a1a2e;
    --surface-3: #16213e;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --border: rgba(99,102,241,0.25);
    --glow: rgba(99,102,241,0.15);
}

body { background: var(--surface) !important; color: var(--text) !important; }

.gradio-container {
    max-width: 1200px !important;
    background: var(--surface) !important;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border-radius: 16px;
    border: 1px solid var(--border);
    margin-bottom: 1.5rem;
    box-shadow: 0 0 40px var(--glow);
}

.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #818cf8, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.5rem;
    letter-spacing: -0.5px;
}

.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin: 0;
}

.badge {
    display: inline-block;
    background: rgba(99,102,241,0.2);
    border: 1px solid var(--border);
    color: #818cf8;
    border-radius: 999px;
    padding: 0.25rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin: 0.6rem 0.2rem 0;
}

/* Panel cards */
.panel {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* Caption output */
.caption-box {
    background: linear-gradient(135deg, rgba(99,102,241,0.08), rgba(167,139,250,0.08));
    border: 1px solid rgba(99,102,241,0.35);
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    font-size: 1.15rem;
    line-height: 1.7;
    color: #e2e8f0;
    min-height: 80px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.05), 0 0 20px rgba(99,102,241,0.1);
}

/* Buttons */
.generate-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
    transition: all 0.2s !important;
}

.generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.55) !important;
}

/* Stats bar */
.stats { color: var(--text-muted); font-size: 0.82rem; padding: 0.3rem 0; }

/* File upload */
.upload-area { border-radius: 10px !important; border: 1px dashed var(--border) !important; }
"""

HEADER_HTML = """
<div class="hero">
  <h1>Utonia-Cap</h1>
  <p>3D Point Cloud → Natural Language Captioning</p>
  <span class="badge">Utonia Encoder</span>
  <span class="badge">Cross-Attention Projector</span>
  <span class="badge">Qwen2.5-1.5B + LoRA</span>
  <span class="badge">RTX 3050 · 6 GB VRAM</span>
</div>
"""

with gr.Blocks(title="Utonia-Cap Demo") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Row(equal_height=False):

        # ── Left: inputs ──────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=340):
            gr.Markdown("### Upload Point Cloud")
            file_input = gr.File(
                label="Drop .npz or .ply here",
                file_types=[".npz", ".ply"],
                elem_classes=["upload-area"],
            )

            gr.Markdown("### Quick Load Example")
            with gr.Row():
                for name in EXAMPLE_FILES:
                    path = EXAMPLE_FILES[name]
                    if os.path.exists(path):
                        btn = gr.Button(name.split("(")[0].strip(), size="sm")
                        btn.click(fn=lambda n=name: EXAMPLE_FILES[n],
                                  outputs=file_input)

            gr.Markdown("### Prompt")
            prompt_choice = gr.Dropdown(
                choices=list(PROMPT_PRESETS.keys()),
                value=list(PROMPT_PRESETS.keys())[0],
                label="Preset prompt",
                show_label=False,
            )
            custom_prompt = gr.Textbox(
                placeholder="Or type your own question...",
                label="Custom prompt (overrides preset)",
                show_label=False,
                lines=2,
            )

            gen_btn = gr.Button(
                "▶  Generate Caption",
                variant="primary",
                elem_classes=["generate-btn"],
            )

            gr.Markdown(
                "<small style='color:#64748b'>**Model:** Utonia (137M) + projector (5.1M) + Qwen2.5-1.5B LoRA  \n"
                "**Training:** Stage 1 (150 ep) + Stage 2 LoRA (10 ep)  \n"
                "**Dataset:** 100 augmented indoor scenes + Cap3D style injection</small>"
            )

        # ── Right: outputs ────────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### Generated Caption")
            caption_out = gr.Textbox(
                label="",
                lines=4,
                show_label=False,
                placeholder="Caption will appear here after you click Generate...",
                elem_classes=["caption-box"],
            )
            stats_out = gr.Markdown(elem_classes=["stats"])

            gr.Markdown("### 3D Point Cloud Preview")
            viz_out = gr.Plot(label="", show_label=False)

    # ── Wire up ───────────────────────────────────────────────────────────
    gen_btn.click(
        fn=caption_pointcloud,
        inputs=[file_input, prompt_choice, custom_prompt],
        outputs=[caption_out, viz_out, stats_out],
    )

    # Also trigger on file upload for instant preview
    file_input.change(
        fn=lambda f: build_pointcloud_viz(
            {k: v for k, v in dict(
                np.load(f.name if hasattr(f, "name") else f, allow_pickle=True)
            ).items() if k in ["coord", "color"]}
        ) if f else None,
        inputs=[file_input],
        outputs=[viz_out],
    )


if __name__ == "__main__":
    print("\nUtonia-Cap Demo")
    print(f"  Device: {DEVICE}")
    print(f"  Stage 2 checkpoint: {'✓' if os.path.exists(CKPT_S2) else '✗'}")
    print(f"  LoRA adapter:       {'✓' if os.path.exists(LORA_DIR) else '✗'}")
    print("\nStarting Gradio server...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        css=CSS,
        theme=gr.themes.Base(),
    )
