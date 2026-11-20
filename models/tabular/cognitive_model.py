"""
models/tabular/cognitive_model.py
----------------------------------
Cognitive scores assistant backbone.

Input features (8 total, configurable):
  - Memory composite score
  - Executive function score
  - Language score
  - Visuospatial score
  - Attention score
  - Processing speed
  - ADAS-Cog 11 (Alzheimer's Disease Assessment Scale)
  - Digit span (working memory)

Same MLP-with-residuals architecture as clinical_model.py,
but semantically different — these are neuropsychological test
scores that directly reflect cognitive decline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from omegaconf import DictConfig

from models.heads.classification_head import ClassificationHead
from models.tabular.clinical_model import ResidualBlock  # reuse

# Named cognitive features (for explainability / SHAP)
COGNITIVE_FEATURE_NAMES = [
    "Memory_composite", "Executive_function", "Language",
    "Visuospatial", "Attention", "Processing_speed",
    "ADAS_Cog_11", "Digit_span",
]


class CognitiveModel(nn.Module):
    """
    MLP backbone for neuropsychological test scores.
    Identical structure to ClinicalModel — different semantics.
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__()
        c = cfg.model.cognitive
        in_dim = getattr(cfg.data, "num_cognitive_features", c.input_dim) if hasattr(cfg, "data") else c.input_dim
        self.in_dim = in_dim
        hidden_dims: List[int] = list(c.hidden_dims)
        dropout     = c.dropout

        layers = [
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for i in range(len(hidden_dims) - 1):
            if hidden_dims[i] == hidden_dims[i + 1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])

        self.backbone = nn.Sequential(*layers)
        feature_dim   = hidden_dims[-1]

        self.head = ClassificationHead(
            input_dim   = feature_dim,
            hidden_dim  = feature_dim,
            num_classes = num_classes,
            dropout     = dropout,
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x       : [B, num_cognitive_features]
        Returns : (logits [B, C], feature_vec [B, hidden_dim])
        """
        if x.shape[-1] != self.in_dim:
            if x.shape[-1] > self.in_dim:
                x = x[:, :self.in_dim]
            else:
                x = F.pad(x, (0, self.in_dim - x.shape[-1]))
        feat   = self.backbone(x)
        logits = self.head(feat)
        return logits, feat
