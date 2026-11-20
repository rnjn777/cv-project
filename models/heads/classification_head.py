"""
models/heads/classification_head.py
-------------------------------------
Shared classification head used by all assistants.

Takes a feature vector of any size and maps it to class logits
via a 2-layer MLP with dropout and batch normalisation.
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    input_dim  → hidden_dim → num_classes (logits)

    Used by every assistant. The head is always trained alongside
    its backbone — they share a single forward pass.
    """

    def __init__(
        self,
        input_dim:  int,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, input_dim]  →  logits: [B, num_classes]"""
        return self.head(x)
