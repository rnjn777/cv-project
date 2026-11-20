"""agents/smri_branch/gnn_assistant.py"""
import torch
from typing import Dict, Tuple
from omegaconf import DictConfig
from agents.base_assistant import BaseAssistant
from models.gnn.brain_graph import BrainGraphBuilder
from models.gnn.gnn_model import GNNModel


class GNNAssistant(BaseAssistant):
    """
    Assistant B — Graph Neural Network (sMRI branch).

    Unique angle: HOW do brain regions relate to each other?
    (Inter-region co-atrophy, connectivity breakdown)
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__(name="GNN", mc_samples=cfg.model.gnn.mc_dropout_samples)
        mri_shape = tuple(cfg.data.mri_shape)
        self.graph_builder = BrainGraphBuilder(
            mri_shape        = mri_shape,
            node_feature_dim = cfg.model.gnn.node_feature_dim,
        )
        self.model = GNNModel(cfg, num_classes=num_classes)

    def _forward(self, batch: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        mri = batch["mri"]
        node_features, edge_index, _ = self.graph_builder(mri)
        return self.model(node_features, edge_index)   # (logits, graph_embedding)
