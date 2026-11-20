"""
models/fusion/cross_attention.py
--------------------------------
Cross-Attention Multimodal Feature Fusion.

Transforms the multi-agent system from naive late-probability averaging
into deep cross-modal reasoning.

Mechanism:
  1. Each assistant's attribute/feature vector (CE-AH: 512, GNN: 256,
     Clinical: 64, Cognitive: 64) is projected into a shared embedding space (dim=128).
  2. Modality-type embeddings are added (identifying which token belongs to which agent).
  3. Multi-Head Cross/Self Attention allows agents to directly query one another:
       - CE-AH queries Cognitive to verify if hippocampal atrophy matches memory drop.
       - GNN queries Clinical to verify if connectivity breakdown aligns with patient risk.
  4. Yields fused classification logits [B, num_classes] + explainability attention matrix [B, 4, 4].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Any, Optional


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Multimodal Feature Fusion.

    Parameters
    ----------
    dim_ceah      : int (default: 512, whole_brain_encoder output)
    dim_gnn       : int (default: 256, gnn graph embedding)
    dim_clinical  : int (default: 64, clinical MLP embedding)
    dim_cognitive : int (default: 64, cognitive MLP embedding)
    shared_dim    : int (default: 128, common latent space)
    num_heads     : int (default: 4, attention heads)
    num_classes   : int (default: 3 — CN / MCI / AD)
    dropout       : float (default: 0.2)
    """

    MODALITY_NAMES = ["CE-AH", "GNN", "Clinical", "Cognitive"]

    def __init__(
        self,
        dim_ceah: int = 512,
        dim_gnn: int = 256,
        dim_clinical: int = 64,
        dim_cognitive: int = 64,
        shared_dim: int = 128,
        num_heads: int = 4,
        num_classes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared_dim = shared_dim
        self.num_classes = num_classes

        # ── 1. Linear Projection Heads to Shared Latent Space ─────────
        self.proj_ceah = nn.Sequential(
            nn.Linear(dim_ceah, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_gnn = nn.Sequential(
            nn.Linear(dim_gnn, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_clinical = nn.Sequential(
            nn.Linear(dim_clinical, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.proj_cognitive = nn.Sequential(
            nn.Linear(dim_cognitive, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── 2. Modality Type Embeddings (Token ID 0..3) ──────────────
        self.modality_embed = nn.Parameter(torch.randn(1, 4, shared_dim) * 0.02)

        # ── 3. Multi-Head Cross/Self Attention ────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=shared_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(shared_dim)

        # ── 4. Feed-Forward Network (FFN) ─────────────────────────────
        self.ffn = nn.Sequential(
            nn.Linear(shared_dim, shared_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim * 2, shared_dim),
        )
        self.norm2 = nn.LayerNorm(shared_dim)

        # ── 5. Classification Output Head ────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(shared_dim * 4, shared_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, num_classes),
        )

    def forward(
        self,
        feat_ceah: torch.Tensor,
        feat_gnn: torch.Tensor,
        feat_clinical: torch.Tensor,
        feat_cognitive: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the cross-attention fusion network.

        Parameters
        ----------
        feat_ceah      : FloatTensor [B, dim_ceah]
        feat_gnn       : FloatTensor [B, dim_gnn]
        feat_clinical  : FloatTensor [B, dim_clinical]
        feat_cognitive : FloatTensor [B, dim_cognitive]

        Returns
        -------
        logits       : FloatTensor [B, num_classes]
        attn_weights : FloatTensor [B, 4, 4]  (inter-agent attention matrix)
        """
        B = feat_ceah.shape[0]

        # Project all modalities into the shared token space [B, 1, shared_dim]
        t_ceah = self.proj_ceah(feat_ceah).unsqueeze(1)
        t_gnn  = self.proj_gnn(feat_gnn).unsqueeze(1)
        t_clin = self.proj_clinical(feat_clinical).unsqueeze(1)
        t_cogn = self.proj_cognitive(feat_cognitive).unsqueeze(1)

        # Stack into sequence of 4 agent tokens: [B, 4, shared_dim]
        tokens = torch.cat([t_ceah, t_gnn, t_clin, t_cogn], dim=1)
        tokens = tokens + self.modality_embed

        # Multi-Head Attention: all agents attend across each other
        attn_out, attn_weights = self.cross_attn(
            query=tokens, key=tokens, value=tokens, need_weights=True
        )
        tokens = self.norm1(tokens + attn_out)

        # Position-wise feed-forward
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Flatten all 4 enhanced agent tokens: [B, 4 * shared_dim]
        flat = tokens.reshape(B, -1)
        logits = self.classifier(flat)

        return logits, attn_weights
