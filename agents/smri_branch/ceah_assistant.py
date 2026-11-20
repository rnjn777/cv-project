"""agents/smri_branch/ceah_assistant.py"""
import torch
from typing import Dict, Tuple
from omegaconf import DictConfig
from agents.base_assistant import BaseAssistant
from models.ceah.ceah_model import CEAHModel


class CEAHAssistant(BaseAssistant):
    """
    Assistant A — CE-AH Hierarchical Attention (sMRI branch).

    Unique angle: WHERE in the brain are structures abnormal?
    (Patch → Region → Whole Brain spatial hierarchy)
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__(name="CE-AH", mc_samples=cfg.model.ceah.mc_dropout_samples)
        self.model = CEAHModel(cfg, num_classes=num_classes)

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        mri = batch["mri"]
        return self.model(mri)   # (logits, feature_vector)
