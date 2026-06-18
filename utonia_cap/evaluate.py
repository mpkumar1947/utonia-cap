"""
Utonia-Cap Evaluation: Computes BLEU-4, CIDEr, METEOR against ground truth.

These are the standard metrics used in all 3D captioning papers.
You NEED these numbers for your intern drive — they prove your model works.

Usage:
    conda activate utonia
    export PYTHONPATH=./

    # Evaluate on synthetic data (instant, no extra download)
    python utonia_cap/evaluate.py --data synthetic

    # Evaluate on ScanRefer validation split (after training)
    python utonia_cap/evaluate.py --data scanrefer \\
        --checkpoint checkpoints/stage1_best.pt

What good scores look like (from published papers):
    Method               BLEU-4   CIDEr   METEOR
    Scan2Cap (baseline)   23.3     56.4     21.9
    Vote2Cap-DETR         34.2    109.8     26.6
    Our Utonia-Cap (aim)  ~28-32   ~70-90   ~22-25
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

import utonia


def load_projector(checkpoint_path: str, device: str):
    """Load trained projector from checkpoint."""
    from utonia_cap.projector import UtoniaCrossAttentionProjector

    projector = UtoniaCrossAttentionProjector(
        utonia_dim=576, num_queries=32, llm_dim=1536
    ).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        projector.load_state_dict(ckpt["projector_state"])
        print(f"  Loaded projector from epoch {ckpt.get('epoch', '?')}, loss={ckpt.get('loss', '?'):.4f}")
    else:
        print("  WARNING: No checkpoint loaded — evaluating random projector (baseline)")

    projector.eval()
    return projector


def generate_caption(
    point_dict: dict,
    utonia_model,
    projector,
    llm,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int = 150,
) -> str:
    """Generate a caption for one point cloud. Returns the caption string."""

    # Utonia encode
    bottleneck = {}
    def hook(m, i, o):
        bottleneck["feat"] = o.feat
        bottleneck["offset"] = o.offset
    h = utonia_model.enc.enc4.register_forward_hook(hook)

    for key in point_dict:
        if isinstance(point_dict[key], torch.Tensor):
            point_dict[key] = point_dict[key].to(device)

    with torch.inference_mode():
        utonia_model(point_dict)
    h.remove()

    # Project
    with torch.no_grad():
        geo_tokens = projector(
            bottleneck["feat"], bottleneck["offset"]
        ).to(torch.bfloat16)

    # Build prompt embeds
    prompt_text = (
        "<|im_start|>system\nYou are a 3D scene understanding assistant.<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt",
                           add_special_tokens=False).input_ids.to(device)
    llm_embed = llm.get_input_embeddings()
    prompt_embeds = llm_embed(prompt_ids)
    input_embeds = torch.cat([prompt_embeds, geo_tokens], dim=1)

    with torch.no_grad():
        gen_ids = llm.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    caption = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    # Clean assistant prefix if leaked
    if "assistant" in caption.lower():
        caption = caption.split("assistant")[-1].strip().lstrip("\n").strip()

    return caption


def compute_metrics(predictions: dict, references: dict) -> dict:
    """
    Compute BLEU-4, CIDEr, METEOR using pycocoevalcap.

    Args:
        predictions: {scene_id: "generated caption string"}
        references:  {scene_id: ["ref caption 1", "ref caption 2", ...]}

    Returns:
        dict with metric names and scores
    """
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor

        # Format for pycocoevalcap
        gts = {k: [{"caption": c} for c in v] for k, v in references.items()}
        res = {k: [{"caption": v}] for k, v in predictions.items()}

        scorers = [
            (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
            (Cider(), "CIDEr"),
            (Meteor(), "METEOR"),
        ]

        results = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(method, list):
                for m, s in zip(method, score):
                    results[m] = round(s * 100, 2)
            else:
                results[method] = round(score * 100, 2)

        return results

    except ImportError:
        # Fallback: compute BLEU manually (no external dependencies)
        print("  pycocoevalcap not available, computing simple BLEU-4 only...")
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

        all_refs, all_hyps = [], []
        for k in predictions:
            if k in references:
                hyp = predictions[k].split()
                refs = [r.split() for r in references[k]]
                all_hyps.append(hyp)
                all_refs.append(refs)

        smoothie = SmoothingFunction().method4
        bleu4 = corpus_bleu(all_refs, all_hyps, smoothing_function=smoothie)
        return {"BLEU_4": round(bleu4 * 100, 2)}


def evaluate_synthetic(args, utonia_model, projector, llm, tokenizer):
    """
    Evaluate on synthetic data — uses auto-generated captions as ground truth.
    Tests that the model can at least learn to predict the objects present.
    """
    from utonia_cap.dataset import SyntheticPointCloudDataset, labels_to_caption

    data_dir = os.path.join(args.data_dir, "data")
    npz_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir) if f.endswith(".npz")
    ]
    if not npz_files:
        print("No .npz files found. Run demos first to download sample data.")
        return

    transform = utonia.transform.default(0.5)
    predictions, references = {}, {}
    prompt = "Describe the 3D scene in detail."

    for i, path in enumerate(tqdm(npz_files, desc="Evaluating")):
        # Load and prepare data
        raw = dict(np.load(path))
        raw.pop("segment200", None)
        segment = raw.pop("segment20", None)
        if segment is not None:
            raw["segment"] = segment

        # Ground truth = auto-generated caption
        gt_caption = labels_to_caption(segment) if segment is not None else "A 3D scene."

        point = transform(dict(raw))
        pred = generate_caption(
            point, utonia_model, projector, llm, tokenizer, prompt,
            device=args.device
        )

        scene_id = os.path.basename(path).replace(".npz", "")
        predictions[scene_id] = pred
        references[scene_id] = [gt_caption]

        print(f"\n  [{scene_id}]")
        print(f"  GT:   {gt_caption}")
        print(f"  Pred: {pred}")

    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, references)

    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    for name, score in metrics.items():
        print(f"  {name:<15} {score:>8.2f}")
    print(f"{'='*50}")

    # Save results
    results = {"metrics": metrics, "predictions": predictions, "references": references}
    out_path = os.path.join(args.save_dir, "eval_results.json")
    os.makedirs(args.save_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Utonia-Cap")
    parser.add_argument("--data", default="synthetic",
                        choices=["synthetic", "scanrefer"])
    parser.add_argument("--data-dir", default=os.path.expanduser("~/.cache/utonia"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save-dir", default="eval_results")
    args = parser.parse_args()

    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nEvaluating on {args.data} | device: {args.device}\n")

    # Load models
    print("Loading Utonia...")
    utonia_model = utonia.load("ckpt/utonia.pth").to(args.device)
    utonia_model.eval()
    for p in utonia_model.parameters():
        p.requires_grad = False

    projector = load_projector(args.checkpoint, args.device)

    print("Loading Qwen2.5-1.5B...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct", trust_remote_code=True
    )
    llm = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map=args.device,
    )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    if args.data == "synthetic":
        evaluate_synthetic(args, utonia_model, projector, llm, tokenizer)
    else:
        print("ScanRefer evaluation coming in Week 3.")
