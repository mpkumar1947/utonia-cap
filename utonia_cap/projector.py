"""
Utonia-Cap Projector: Bridges Utonia geometry features → LLM token space.

Architecture:
    Utonia bottleneck: [N_tokens, utonia_dim=512]  (variable N per scene)
                ↓
    Cross-Attention: 32 learnable queries attend to all N point tokens
                ↓
    MLP: projects from utonia_dim → llm_dim (1536 for Qwen2.5-1.5B)
                ↓
    Output: [batch, 32, llm_dim]  ← exactly what the LLM reads

Why cross-attention instead of mean pooling?
    Mean pooling loses spatial structure — the model can't tell "there's
    a sofa on the LEFT and a table on the RIGHT". Learnable queries let
    different queries specialize: one might learn to attend to furniture,
    another to the walls, another to the floor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class UtoniaCrossAttentionProjector(nn.Module):
    """
    Pools variable-length Utonia point features into a fixed set of
    query tokens that can be directly prepended to an LLM's input.

    Args:
        utonia_dim:  Feature dimension from Utonia's encoder (512).
        num_queries: Number of fixed output tokens (32 is a good default;
                     more = more context for LLM, more VRAM).
        llm_dim:     Hidden dim of the target LLM (1536 for Qwen2.5-1.5B).
        num_heads:   Number of attention heads in cross-attention.
        dropout:     Dropout for regularization during training.
    """

    def __init__(
        self,
        utonia_dim: int = 576,
        num_queries: int = 32,
        llm_dim: int = 1536,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.utonia_dim = utonia_dim
        self.num_queries = num_queries
        self.llm_dim = llm_dim

        # Learnable query tokens — these are the "questions" the projector
        # asks of the point cloud. Initialized randomly, learned during Stage 1.
        self.queries = nn.Parameter(torch.randn(1, num_queries, utonia_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Cross-attention: queries (Q) attend to point features (K, V)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=utonia_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,   # [batch, seq, dim] convention
        )
        self.norm1 = nn.LayerNorm(utonia_dim)

        # Self-attention among queries (lets queries share information)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=utonia_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(utonia_dim)

        # MLP: maps from Utonia's geometry space → LLM's language space
        # We use a 2-layer MLP with GELU (same as the original LLaVA projector)
        self.mlp = nn.Sequential(
            nn.Linear(utonia_dim, utonia_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(utonia_dim * 2, llm_dim),
        )
        self.norm3 = nn.LayerNorm(llm_dim)

    def forward(
        self,
        point_feats: torch.Tensor,      # [N_total_points, utonia_dim]
        offsets: torch.Tensor,          # [batch_size] cumulative point counts
    ) -> torch.Tensor:
        """
        Args:
            point_feats: Concatenated point features from all scenes in batch.
                         Shape: [N_total, utonia_dim]
            offsets:     Cumulative point counts per scene.
                         e.g. for 2 scenes with 800 and 900 points: [800, 1700]

        Returns:
            tokens: [batch_size, num_queries, llm_dim]
                    Ready to be concatenated with text tokens and fed to LLM.
        """
        batch_size = offsets.shape[0]
        all_tokens = []

        # Split concatenated features back into per-scene tensors
        start = 0
        for i in range(batch_size):
            end = offsets[i].item()
            scene_feats = point_feats[start:end]  # [N_i, utonia_dim]
            start = end

            # Add batch dimension: [1, N_i, utonia_dim]
            scene_feats = scene_feats.unsqueeze(0)

            # Expand queries for this scene
            queries = self.queries.to(scene_feats.dtype)  # [1, num_queries, utonia_dim]

            # Cross-attention: queries ask the point cloud for information
            attn_out, _ = self.cross_attn(
                query=queries,
                key=scene_feats,
                value=scene_feats,
            )
            queries = self.norm1(queries + attn_out)

            # Self-attention: queries communicate with each other
            self_out, _ = self.self_attn(queries, queries, queries)
            queries = self.norm2(queries + self_out)

            all_tokens.append(queries)

        # Stack all scenes: [batch, num_queries, utonia_dim]
        tokens = torch.cat(all_tokens, dim=0)

        # Project to LLM dimension: [batch, num_queries, llm_dim]
        tokens = self.norm3(self.mlp(tokens))

        return tokens

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


# ── Quick Self-Test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing UtoniaCrossAttentionProjector...")
    print()

    projector = UtoniaCrossAttentionProjector(
        utonia_dim=576,
        num_queries=32,
        llm_dim=1536,
        num_heads=8,
    )
    projector.eval()

    # Simulate a batch of 2 scenes with different numbers of tokens
    # (like you'd get from 2 different point clouds)
    N1, N2 = 1133, 987   # typical Utonia stage-4 token counts
    fake_feats = torch.randn(N1 + N2, 576)
    fake_offsets = torch.tensor([N1, N1 + N2])

    with torch.no_grad():
        out = projector(fake_feats, fake_offsets)

    print(f"  Input:   point_feats = {fake_feats.shape}")
    print(f"           offsets     = {fake_offsets.tolist()}")
    print()
    print(f"  Output:  tokens = {out.shape}")
    print(f"           → Ready to feed into Qwen2.5-1.5B as prefix!")
    print()
    print(f"  Projector parameters: {projector.num_parameters():,} "
          f"({projector.num_parameters()/1e6:.1f}M)")
    print()

    expected = torch.Size([2, 32, 1536])
    assert out.shape == expected, f"Expected {expected}, got {out.shape}"
    print("✓ Shape check passed!")
    print()
    print("→ Next step: run  python utonia_cap/model.py")
