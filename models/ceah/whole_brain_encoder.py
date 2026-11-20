"""
models/ceah/whole_brain_encoder.py
------------------------------------
Level 3 of CE-AH: Whole-brain encoder.

Takes the 90 region embeddings from level 2 and compresses them
into a single global brain representation using a CLS token
Transformer — identical in spirit to BERT's [CLS] token.

This level sees everything simultaneously: patterns that span
multiple lobes, bilateral differences (left vs right hippocampus),
and global atrophy signatures.
"""

import torch
import torch.nn as nn
from utils.brain_atlas import NUM_REGIONS


class WholeBrainTransformerBlock(nn.Module):
    """Transformer block at the whole-brain level."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class WholeBrainEncoder(nn.Module):
    """
    Full whole-brain encoder.

    1. Project region_embed_dim → global_embed_dim
    2. Prepend a learnable [CLS] token
    3. Run `depth` global Transformer blocks
    4. Return [CLS] token embedding as the brain summary vector

    Also returns per-region attention weights for explainability (Grad-CAM target).
    """

    def __init__(
        self,
        region_embed_dim: int = 256,
        global_embed_dim: int = 512,
        num_heads: int = 8,
        depth: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(region_embed_dim, global_embed_dim)

        # CLS token — summarises the whole brain
        self.cls_token = nn.Parameter(torch.zeros(1, 1, global_embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, NUM_REGIONS + 1, global_embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            WholeBrainTransformerBlock(global_embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(global_embed_dim)

    def forward(self, region_feats: torch.Tensor) -> torch.Tensor:
        """
        region_feats : [B, 90, region_embed_dim]
        Returns      : [B, global_embed_dim]   (CLS token embedding)
        """
        B = region_feats.shape[0]

        x = self.input_proj(region_feats)               # [B, 90, global_embed_dim]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, G]
        x   = torch.cat([cls, x], dim=1)                # [B, 91, G]
        x   = x + self.pos_embed[:, :x.shape[1], :]

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0, :]   # [B, global_embed_dim]  — CLS token only
