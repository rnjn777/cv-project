"""
models/gnn/gnn_model.py
------------------------
GNN backbone for the brain graph assistant.

Uses Graph Attention Networks (GAT) — each region attends over
its neighbours, weighting their contributions by learned attention.
This captures co-atrophy patterns: if hippocampus is atrophying,
do entorhinal cortex and amygdala show correlated changes?

Architecture:
    Node features [90, node_dim]
        ↓  GAT layer × num_layers
    Enhanced node features [90, gnn_out_dim]
        ↓  Global mean pool
    Graph embedding [gnn_out_dim]
        ↓  ClassificationHead
    Logits [num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from omegaconf import DictConfig

from models.heads.classification_head import ClassificationHead


# ── Simple GAT layer (no PyG dependency needed for pure-PyTorch version) ──────

class GATLayer(nn.Module):
    """
    Single Graph Attention layer.
    Implements Velickovic et al. (2018) attention mechanism.

    Operates on dense batched graphs: [B, N, F] + edge_index [2, E]
    """

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        num_heads:    int = 4,
        dropout:      float = 0.1,
        concat:       bool = True,
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.num_heads    = num_heads
        self.concat       = concat
        self.head_dim     = out_features

        # Linear transformation for each attention head
        self.W  = nn.Linear(in_features, num_heads * out_features, bias=False)
        # Attention coefficients: [head, 2*out_features] → scalar
        self.a  = nn.Parameter(torch.zeros(num_heads, 2 * out_features))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout    = nn.Dropout(dropout)
        self.norm       = nn.LayerNorm(num_heads * out_features if concat else out_features)

    def forward(
        self,
        x:          torch.Tensor,   # [B, N, in_features]
        edge_index: torch.Tensor,   # [2, E]
    ) -> torch.Tensor:
        """Returns [B, N, num_heads*out_features] or [B, N, out_features]"""
        B, N, _ = x.shape
        E = edge_index.shape[1]
        H = self.num_heads
        D = self.out_features

        Wh = self.W(x)                          # [B, N, H*D]
        Wh = Wh.view(B, N, H, D)               # [B, N, H, D]

        src, dst = edge_index[0], edge_index[1] # [E]

        # Gather source and destination features for each edge
        Wh_src = Wh[:, src, :, :]              # [B, E, H, D]
        Wh_dst = Wh[:, dst, :, :]              # [B, E, H, D]

        # Attention scores: concat(Wh_src, Wh_dst) dot a
        e = torch.cat([Wh_src, Wh_dst], dim=-1)     # [B, E, H, 2D]
        e = (e * self.a).sum(dim=-1)                 # [B, E, H]
        e = self.leaky_relu(e)

        # Softmax per destination node, per head
        alpha = torch.full((B, N, H), float("-inf"), device=x.device)
        alpha[:, dst, :] = torch.maximum(alpha[:, dst, :], e)

        # Proper softmax normalisation per target node
        alpha_exp = torch.zeros(B, E, H, device=x.device)
        alpha_max = alpha[:, dst, :]                  # [B, E, H]
        alpha_exp = torch.exp(e - alpha_max)

        denom = torch.zeros(B, N, H, device=x.device)
        denom.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).expand(B, E, H),
            alpha_exp,
        )
        alpha_norm = alpha_exp / (denom[:, dst, :] + 1e-8)  # [B, E, H]
        alpha_norm = self.dropout(alpha_norm)

        # Weighted sum of source features
        out = torch.zeros(B, N, H, D, device=x.device)
        out.scatter_add_(
            1,
            dst.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(B, E, H, D),
            (alpha_norm.unsqueeze(-1) * Wh_src),
        )

        if self.concat:
            out = out.view(B, N, H * D)       # [B, N, H*D]
        else:
            out = out.mean(dim=2)             # [B, N, D]

        return F.elu(self.norm(out))


class GNNModel(nn.Module):
    """
    Multi-layer GAT brain graph network.

    Input : node_features [B, 90, node_feature_dim]  +  edge_index
    Output: (logits [B, C], graph_embedding [B, gnn_out_dim])
    """

    def __init__(self, cfg: DictConfig, num_classes: int = 3):
        super().__init__()
        c = cfg.model.gnn

        in_dim  = c.node_feature_dim
        hid_dim = c.gnn_hidden_dim
        out_dim = c.gnn_out_dim
        heads   = c.gat_heads
        n_layers = c.num_gnn_layers
        dropout  = c.dropout

        layers = []
        for i in range(n_layers):
            _in  = in_dim if i == 0 else hid_dim * heads
            _out = hid_dim
            _concat = (i < n_layers - 1)   # last layer: average heads
            layers.append(GATLayer(_in, _out, num_heads=heads, dropout=dropout, concat=_concat))

        self.gat_layers = nn.ModuleList(layers)
        self.dropout    = nn.Dropout(dropout)

        # After last layer: hid_dim (averaged heads)
        self.output_proj = nn.Linear(hid_dim, out_dim)

        self.head = ClassificationHead(
            input_dim   = out_dim,
            hidden_dim  = out_dim // 2,
            num_classes = num_classes,
            dropout     = dropout,
        )

    def forward(
        self,
        node_features: torch.Tensor,   # [B, 90, node_feature_dim]
        edge_index:    torch.Tensor,   # [2, E]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        logits           : [B, num_classes]
        graph_embedding  : [B, gnn_out_dim]   ← attribute vector
        """
        x = node_features
        for layer in self.gat_layers:
            x = self.dropout(x)
            x = layer(x, edge_index)             # [B, 90, H*D] or [B, 90, D]

        x = self.output_proj(x)                  # [B, 90, gnn_out_dim]

        # Global mean pooling: summarise all regions into one vector
        graph_embedding = x.mean(dim=1)          # [B, gnn_out_dim]

        logits = self.head(graph_embedding)      # [B, num_classes]
        return logits, graph_embedding
