"""
agents/base_assistant.py
--------------------------
Abstract base class for all assistants.

Every assistant — whether it wraps a 3D MRI model or a tabular MLP —
must conform to this interface:

    result = assistant.predict(inputs)

The result is a standardised AssistantOutput dict containing:
  - class_probs    : mean probability distribution (from MC Dropout)
  - confidence     : stability score (1 - std_dev)
  - feature_vector : attribute vector from the backbone (for arbitration)
  - std_probs      : raw uncertainty per class
"""

from __future__ import annotations

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List

from utils.mc_dropout import mc_dropout_inference


@dataclass
class AssistantOutput:
    """
    Standardised output from every assistant.

    Attributes
    ----------
    name           : Human-readable name ("CE-AH", "GNN", etc.)
    class_probs    : FloatTensor [B, num_classes]  — mean over MC Dropout passes
    confidence     : float  — scalar in [0, 1]
    std_probs      : FloatTensor [B, num_classes]  — uncertainty per class
    feature_vector : FloatTensor [B, feature_dim]  — attribute vector for arbitration
    metadata       : Optional dict for assistant-specific extras
    """
    name:           str
    class_probs:    torch.Tensor
    confidence:     float
    std_probs:      torch.Tensor
    feature_vector: torch.Tensor
    metadata:       Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "name":           self.name,
            "class_probs":    self.class_probs,
            "confidence":     self.confidence,
            "std_probs":      self.std_probs,
            "feature_vector": self.feature_vector,
            "metadata":       self.metadata,
        }


class BaseAssistant(ABC, nn.Module):
    """
    Abstract base for all assistants.

    Subclasses must implement:
        _forward(batch) → (logits, feature_vector)

    The base class handles MC Dropout and output packaging.
    """

    def __init__(self, name: str, mc_samples: int = 30):
        super().__init__()
        self.name       = name
        self.mc_samples = mc_samples

    @abstractmethod
    def _forward(self, batch: Dict[str, torch.Tensor]):
        """
        Raw forward pass through the backbone + head.
        Returns: (logits [B, C], feature_vector [B, D])
        """
        ...

    def predict(self, batch: Dict[str, torch.Tensor]) -> AssistantOutput:
        """
        Run MC Dropout inference and return a standardised AssistantOutput.
        """
        device = next(self.parameters()).device

        # Move batch tensors to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # --- MC Dropout: run N stochastic forward passes ---
        self.eval()
        # enable_dropout called inside mc_dropout_inference

        # We need feature_vector too — run one clean pass to get it
        with torch.no_grad():
            logits, feature_vector = self._forward(batch)

        # Run MC Dropout on the model's forward (logits only)
        class_probs, std_probs, confidence = mc_dropout_inference(
            model    = self,
            x        = batch,
            n_samples = self.mc_samples,
            device   = device,
        )

        return AssistantOutput(
            name           = self.name,
            class_probs    = class_probs,       # [B, C]
            confidence     = confidence,
            std_probs      = std_probs,         # [B, C]
            feature_vector = feature_vector,    # [B, D]
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Standard nn.Module forward — returns logits only.
        Used by MC Dropout runner which expects a single tensor out.
        """
        logits, _ = self._forward(batch)
        return logits
