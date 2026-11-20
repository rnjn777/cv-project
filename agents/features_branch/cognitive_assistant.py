"""agents/features_branch/cognitive_assistant.py"""
import torch
from typing import Dict, Tuple
from omegaconf import DictConfig
from agents.base_assistant import BaseAssistant
from models.tabular.cognitive_model import CognitiveModel


class CognitiveAssistant(BaseAssistant):
    """
    Assistant D — Cognitive test scores (Features branch).
    Reads: Memory, Executive function, ADAS-Cog, etc.
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__(name="Cognitive", mc_samples=cfg.model.cognitive.mc_dropout_samples)
        self.model = CognitiveModel(cfg, num_classes=num_classes)

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(batch["cognitive"])
