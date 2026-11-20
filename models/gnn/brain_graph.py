"""
models/gnn/brain_graph.py
--------------------------
Converts a 3D MRI volume into a brain graph for the GNN assistant.

Each node = one AAL brain region (90 nodes total).
Node features = mean intensity statistics of the MRI voxels
                falling within that region's spatial extent.
Edges = anatomical adjacency from utils/brain_atlas.py
        (connect regions whose centroids are within 40mm).

This graph representation captures something CE-AH cannot:
the *relationships* between regions — which regions are atrophying
together, co-degeneration patterns, bilateral symmetry breakdown.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

from utils.brain_atlas import NUM_REGIONS, _CENTROIDS, EDGE_INDEX, EDGE_ATTR


def extract_roi_features(
    mri: torch.Tensor,          # [B, 1, D, H, W]
    assignment: torch.Tensor,   # [D*H*W] — each voxel → region index
) -> torch.Tensor:
    """
    For each of the 90 AAL regions, compute:
      - mean intensity
      - std intensity
      - max intensity
      - min intensity
      (4 statistics → node_feature_dim = 4 if no projection)

    Returns: [B, 90, 4]
    """
    B = mri.shape[0]
    flat = mri.view(B, -1)   # [B, D*H*W]
    node_feats = torch.zeros(B, NUM_REGIONS, 4, device=mri.device)

    for r in range(NUM_REGIONS):
        mask = (assignment == r)          # [D*H*W]
        if mask.sum() == 0:
            continue
        voxels = flat[:, mask]            # [B, n_voxels_in_region]
        node_feats[:, r, 0] = voxels.mean(dim=1)
        node_feats[:, r, 1] = voxels.std(dim=1)
        node_feats[:, r, 2] = voxels.max(dim=1).values
        node_feats[:, r, 3] = voxels.min(dim=1).values

    return node_feats   # [B, 90, 4]


def build_voxel_assignment(
    mri_shape: Tuple[int, int, int] = (91, 109, 91),
) -> torch.Tensor:
    """
    Pre-compute a flattened tensor mapping each voxel to its nearest AAL region.
    Shape: [D*H*W]

    In production, use a real atlas NIfTI file (each voxel labelled with region ID).
    Here we assign each voxel to the spatially nearest region centroid.
    """
    D, H, W = mri_shape
    centroids = _CENTROIDS   # [90, 3] MNI coords

    voxel_indices = np.zeros(D * H * W, dtype=np.int64)
    k = 0
    for d in range(D):
        for h in range(H):
            for w in range(W):
                cx = w / W * 160 - 80
                cy = h / H * 220 - 110
                cz = d / D * 160 - 80
                coord = np.array([cx, cy, cz])
                dists = np.linalg.norm(centroids - coord, axis=1)
                voxel_indices[k] = np.argmin(dists)
                k += 1

    return torch.tensor(voxel_indices, dtype=torch.long)


class BrainGraphBuilder(nn.Module):
    """
    Wraps voxel assignment + ROI feature extraction into a single module.
    Pre-builds the voxel assignment at init so __call__ is fast.

    Also projects raw 4-dim ROI features → node_feature_dim.
    """

    def __init__(
        self,
        mri_shape: Tuple[int, int, int] = (91, 109, 91),
        node_feature_dim: int = 64,
    ):
        super().__init__()
        self.mri_shape = mri_shape
        # Pre-compute assignment (slow, done once)
        # For large volumes this is the bottleneck — cache it
        self._assignment = build_voxel_assignment(mri_shape)

        # Project 4 raw stats → node_feature_dim
        self.node_proj = nn.Sequential(
            nn.Linear(4, node_feature_dim),
            nn.LayerNorm(node_feature_dim),
            nn.GELU(),
        )

        # Register edges as buffers (move to device automatically)
        self.register_buffer("edge_index", EDGE_INDEX)
        self.register_buffer("edge_attr",  EDGE_ATTR)

    def forward(self, mri: torch.Tensor):
        """
        mri : [B, 1, D, H, W]

        Returns
        -------
        node_features : [B, 90, node_feature_dim]
        edge_index    : [2, num_edges]
        edge_attr     : [num_edges, 1]
        """
        assignment = self._assignment.to(mri.device)
        raw_feats  = extract_roi_features(mri, assignment)   # [B, 90, 4]
        node_feats = self.node_proj(raw_feats)               # [B, 90, node_feature_dim]
        return node_feats, self.edge_index, self.edge_attr
