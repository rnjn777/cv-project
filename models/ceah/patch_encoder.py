"""
models/ceah/patch_encoder.py
----------------------------
Level 1 of CE-AH: Patch-level encoder.

Splits the 3D MRI volume into non-overlapping 3D patches and
encodes each patch into a fixed-dimensional embedding via a
3D convolutional stem followed by a multi-head self-attention
Transformer block.

This is the finest granularity level — it sees local texture,
fine structural details, and small lesions.
"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchEmbed3D(nn.Module):
    """
    3D Patch Embedding — splits a 3D volume into patches and projects
    each patch to `embed_dim` using a Conv3D with stride=patch_size.

    Input:  [B, 1,    D,  H,  W]
    Output: [B, N,    embed_dim]   where N = (D/P)*(H/P)*(W/P)
    """

    def __init__(self, patch_size: int = 8, embed_dim: int = 128):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        self.proj = nn.Conv3d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        x: [B, 1, D, H, W]
        Returns:
            patches : [B, N, embed_dim]
            grid    : (Gd, Gh, Gw) — number of patches per dimension
        """
        x = self.proj(x)                        # [B, E, Gd, Gh, Gw]
        B, E, Gd, Gh, Gw = x.shape
        x = x.flatten(2).transpose(1, 2)        # [B, N, E]  where N=Gd*Gh*Gw
        x = self.norm(x)
        return x, (Gd, Gh, Gw)


class PatchTransformerBlock(nn.Module):
    """Single Transformer encoder block with pre-norm."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, N, D]"""
        # Self-attention with residual
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # Feed-forward with residual
        x = x + self.ffn(self.norm2(x))
        return x


class PatchEncoder(nn.Module):
    """
    Full patch-level encoder.

    1. Splits MRI into 3D patches
    2. Projects each patch to embed_dim
    3. Adds learnable positional encoding
    4. Runs `depth` Transformer blocks
    5. Returns patch embeddings + grid info for the region-level encoder
    """

    def __init__(
        self,
        patch_size:  int = 8,
        embed_dim:   int = 128,
        num_heads:   int = 4,
        depth:       int = 2,
        dropout:     float = 0.1,
        mri_shape:   Tuple[int, int, int] = (91, 109, 91),
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, embed_dim)

        # Compute max number of patches for positional encoding
        Gd = mri_shape[0] // patch_size
        Gh = mri_shape[1] // patch_size
        Gw = mri_shape[2] // patch_size
        max_patches = Gd * Gh * Gw

        self.pos_embed = nn.Parameter(torch.zeros(1, max_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            PatchTransformerBlock(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        x : [B, 1, D, H, W]
        Returns:
            patch_feats : [B, N, embed_dim]
            grid        : (Gd, Gh, Gw)
        """
        patches, grid = self.patch_embed(x)  # [B, N, E]

        N = patches.shape[1]
        patches = patches + self.pos_embed[:, :N, :]

        for blk in self.blocks:
            patches = blk(patches)

        return self.norm(patches), grid
