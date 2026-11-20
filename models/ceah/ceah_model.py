"""
models/ceah/ceah_model.py
--------------------------
Full CE-AH backbone: assembles the three hierarchical levels.

    MRI → PatchEncoder → RegionEncoder → WholeBrainEncoder → feature vector

The feature vector is then:
  1. Passed to a ClassificationHead → logits (for classification loss)
  2. Returned as-is as the attribute_vector (for cross-assistant comparison)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
from omegaconf import DictConfig

from models.ceah.patch_encoder      import PatchEncoder
from models.ceah.region_encoder     import RegionEncoder
from models.ceah.whole_brain_encoder import WholeBrainEncoder
from models.heads.classification_head import ClassificationHead


class CEAHModel(nn.Module):
    """
    Contrast-Enhanced Attention Hierarchical backbone.

    Forward pass returns both:
      - logits         : [B, num_classes]
      - feature_vector : [B, global_embed_dim]   ← the attribute vector
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__()
        c = cfg.model.ceah
        mri_shape = tuple(cfg.data.mri_shape)

        self.patch_encoder = PatchEncoder(
            patch_size  = c.patch_size,
            embed_dim   = c.patch_embed_dim,
            num_heads   = c.num_heads_patch,
            depth       = c.depth_patch,
            dropout     = c.dropout,
            mri_shape   = mri_shape,
        )
        self.region_encoder = RegionEncoder(
            patch_embed_dim  = c.patch_embed_dim,
            region_embed_dim = c.region_embed_dim,
            num_heads        = c.num_heads_region,
            depth            = c.depth_region,
            dropout          = c.dropout,
            patch_size       = c.patch_size,
            mri_shape        = mri_shape,
        )
        self.whole_brain_encoder = WholeBrainEncoder(
            region_embed_dim = c.region_embed_dim,
            global_embed_dim = c.global_embed_dim,
            num_heads        = c.num_heads_global,
            depth            = c.depth_global,
            dropout          = c.dropout,
        )
        self.head = ClassificationHead(
            input_dim   = c.global_embed_dim,
            hidden_dim  = c.global_embed_dim // 2,
            num_classes = num_classes,
            dropout     = c.dropout,
        )

    def forward(self, mri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        mri : [B, 1, D, H, W]

        Returns
        -------
        logits         : [B, num_classes]
        feature_vector : [B, global_embed_dim]   ← attribute vector for arbitration
        """
        patch_feats, grid  = self.patch_encoder(mri)        # [B, N, E_p]
        region_feats       = self.region_encoder(patch_feats, grid)  # [B, 90, E_r]
        global_feat        = self.whole_brain_encoder(region_feats)  # [B, E_g]
        logits             = self.head(global_feat)          # [B, C]
        return logits, global_feat
