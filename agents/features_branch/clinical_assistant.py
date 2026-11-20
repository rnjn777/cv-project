"""agents/features_branch/clinical_assistant.py"""
import torch
from typing import Dict, Tuple
from omegaconf import DictConfig
from agents.base_assistant import BaseAssistant
from models.tabular.clinical_model import ClinicalModel


class ClinicalAssistant(BaseAssistant):
    """
    Assistant C — Clinical features (Features branch).
    Reads: MMSE, CDR, age, sex, education, etc.
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__(name="Clinical", mc_samples=cfg.model.clinical.mc_dropout_samples)
        self.model = ClinicalModel(cfg, num_classes=num_classes)

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model(batch["clinical"])
