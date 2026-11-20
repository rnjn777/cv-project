"""
models/ceah/region_encoder.py
------------------------------
Level 2 of CE-AH: Region-level encoder.

Takes the patch embeddings from level 1 and pools them into
anatomical region representations using the AAL atlas.

Each of the 90 AAL brain regions gets a fixed set of patches
assigned to it (based on spatial overlap). A cross-attention
Transformer then refines each region embedding by attending
over all its constituent patches.

This level captures region-specific features: hippocampus atrophy,
entorhinal cortex thinning, ventricle expansion, etc.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from utils.brain_atlas import NUM_REGIONS, _CENTROIDS


class RegionPooling(nn.Module):
    """
    Assigns each patch to its nearest brain region (centroid-based)
    and mean-pools all patch embeddings per region.

    Input:
        patch_feats : [B, N, patch_embed_dim]
        grid        : (Gd, Gh, Gw) — spatial layout of patches
        mri_shape   : (D, H, W) — original volume size (for voxel→MNI mapping)

    Output:
        region_feats : [B, 90, patch_embed_dim]
    """

    def __init__(self, patch_size: int = 8, mri_shape=(91, 109, 91)):
        super().__init__()
        self.patch_size = patch_size
        self.mri_shape  = mri_shape
        # Pre-compute patch-to-region assignments (fixed at init)
        self._assignment = self._build_assignment()

    def _build_assignment(self) -> torch.Tensor:
        """
        For each patch (by grid position), find the nearest AAL centroid.
        Returns: LongTensor [Gd, Gh, Gw]
        """
        D, H, W = self.mri_shape
        P = self.patch_size
        Gd, Gh, Gw = D // P, H // P, W // P

        centroids = _CENTROIDS   # [90, 3]  in mm (approximate MNI)

        assignment = np.zeros((Gd, Gh, Gw), dtype=np.int64)
        for pd in range(Gd):
            for ph in range(Gh):
                for pw in range(Gw):
                    # Centre voxel of this patch
                    cx = (pw + 0.5) * P / W * 160 - 80   # rough MNI-like x
                    cy = (ph + 0.5) * P / H * 220 - 110  # rough MNI-like y
                    cz = (pd + 0.5) * P / D * 160 - 80   # rough MNI-like z
                    patch_coord = np.array([cx, cy, cz])
                    dists = np.linalg.norm(centroids - patch_coord, axis=1)
                    assignment[pd, ph, pw] = int(np.argmin(dists))

        return torch.tensor(assignment.flatten(), dtype=torch.long)  # [N]

    def forward(
        self,
        patch_feats: torch.Tensor,
        grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        patch_feats : [B, N, E]
        Returns     : [B, 90, E]
        """
        B, N, E = patch_feats.shape
        assignment = self._assignment[:N].to(patch_feats.device)  # [N]

        # Mean pool patches per region
        region_feats = torch.zeros(B, NUM_REGIONS, E, device=patch_feats.device)
        counts       = torch.zeros(NUM_REGIONS, device=patch_feats.device)

        region_feats.scatter_add_(
            1,
            assignment.unsqueeze(0).unsqueeze(-1).expand(B, N, E),
            patch_feats,
        )
        counts.scatter_add_(0, assignment, torch.ones(N, device=patch_feats.device))
        counts = counts.clamp(min=1.0)

        region_feats = region_feats / counts.unsqueeze(0).unsqueeze(-1)
        return region_feats  # [B, 90, E]


class RegionTransformerBlock(nn.Module):
    """Transformer block at the region level."""

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


class RegionEncoder(nn.Module):
    """
    Full region-level encoder.

    1. Projects patch_embed_dim → region_embed_dim
    2. Adds learnable region positional encoding (one per AAL region)
    3. Runs `depth` region-level Transformer blocks
    4. Returns region embeddings
    """

    def __init__(
        self,
        patch_embed_dim: int = 128,
        region_embed_dim: int = 256,
        num_heads: int = 4,
        depth: int = 2,
        dropout: float = 0.1,
        patch_size: int = 8,
        mri_shape: Tuple[int, int, int] = (91, 109, 91),
    ):
        super().__init__()

        self.pooling = RegionPooling(patch_size=patch_size, mri_shape=mri_shape)
        self.input_proj = nn.Linear(patch_embed_dim, region_embed_dim)

        # Learnable anatomical positional encoding (one per region)
        self.region_pos_embed = nn.Parameter(
            torch.zeros(1, NUM_REGIONS, region_embed_dim)
        )
        nn.init.trunc_normal_(self.region_pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            RegionTransformerBlock(region_embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(region_embed_dim)

    def forward(
        self,
        patch_feats: torch.Tensor,
        grid: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        patch_feats : [B, N, patch_embed_dim]
        Returns     : [B, 90, region_embed_dim]
        """
        region_feats = self.pooling(patch_feats, grid)   # [B, 90, patch_embed_dim]
        region_feats = self.input_proj(region_feats)      # [B, 90, region_embed_dim]
        region_feats = region_feats + self.region_pos_embed

        for blk in self.blocks:
            region_feats = blk(region_feats)

        return self.norm(region_feats)   # [B, 90, region_embed_dim]
