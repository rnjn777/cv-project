"""
models/tabular/clinical_model.py
---------------------------------
Clinical features assistant backbone.

Input features (12 total, configurable):
  - MMSE score (Mini-Mental State Examination)
  - CDR score  (Clinical Dementia Rating)
  - Age
  - Sex (binary)
  - Years of education
  - Handedness
  - BMI
  - Systolic blood pressure
  - Family history (binary)
  - Depression history (binary)
  - Number of medications
  - Comorbidity index

Architecture: MLP with residual connections and BatchNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from omegaconf import DictConfig

from models.heads.classification_head import ClassificationHead

# Named clinical features (for explainability / SHAP)
CLINICAL_FEATURE_NAMES = [
    "MMSE", "CDR", "Age", "Sex", "Education_years",
    "Handedness", "BMI", "Systolic_BP", "Family_history",
    "Depression_history", "Num_medications", "Comorbidity_index",
]


class ResidualBlock(nn.Module):
    """MLP block with residual skip connection."""

    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class ClinicalModel(nn.Module):
    """
    MLP backbone for clinical tabular features.

    Returns (logits, feature_embedding) same interface as image models.
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__()
        c = cfg.model.clinical
        in_dim = getattr(cfg.data, "num_clinical_features", c.input_dim) if hasattr(cfg, "data") else c.input_dim
        self.in_dim = in_dim
        hidden_dims: List[int] = list(c.hidden_dims)
        dropout     = c.dropout

        # Input projection
        layers = [
            nn.Linear(in_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        # Hidden layers with residual blocks
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
        x       : [B, num_clinical_features]
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
