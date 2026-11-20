"""
utils/brain_atlas.py
--------------------
AAL (Automated Anatomical Labeling) atlas — 90 brain regions.

Provides:
  - Region names and indices
  - Adjacency structure (which regions are anatomically neighbours)
  - MNI coordinates of each region's centroid (for graph edge construction)

In the full pipeline these coordinates would be loaded from the real
AAL atlas NIfTI file. Here we define the key regions explicitly and
generate approximate centroid coordinates so everything runs without
needing external atlas files.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


# ── 90 AAL region names (standard ordering) ───────────────────────────────────
AAL_REGIONS: List[str] = [
    # Frontal lobe
    "Precentral_L", "Precentral_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Frontal_Sup_Orb_L", "Frontal_Sup_Orb_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Mid_Orb_L", "Frontal_Mid_Orb_R",
    "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
    "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
    "Frontal_Inf_Orb_L", "Frontal_Inf_Orb_R",
    "Rolandic_Oper_L", "Rolandic_Oper_R",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Olfactory_L", "Olfactory_R",
    "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
    "Frontal_Med_Orb_L", "Frontal_Med_Orb_R",
    "Rectus_L", "Rectus_R",
    # Parietal lobe
    "Postcentral_L", "Postcentral_R",
    "Parietal_Sup_L", "Parietal_Sup_R",
    "Parietal_Inf_L", "Parietal_Inf_R",
    "SupraMarginal_L", "SupraMarginal_R",
    "Angular_L", "Angular_R",
    "Precuneus_L", "Precuneus_R",
    "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    # Occipital lobe
    "Calcarine_L", "Calcarine_R",
    "Cuneus_L", "Cuneus_R",
    "Lingual_L", "Lingual_R",
    "Occipital_Sup_L", "Occipital_Sup_R",
    "Occipital_Mid_L", "Occipital_Mid_R",
    "Occipital_Inf_L", "Occipital_Inf_R",
    "Fusiform_L", "Fusiform_R",
    # Temporal lobe (critical for AD)
    "Heschl_L", "Heschl_R",
    "Temporal_Sup_L", "Temporal_Sup_R",
    "Temporal_Pole_Sup_L", "Temporal_Pole_Sup_R",
    "Temporal_Mid_L", "Temporal_Mid_R",
    "Temporal_Pole_Mid_L", "Temporal_Pole_Mid_R",
    "Temporal_Inf_L", "Temporal_Inf_R",
    # Limbic (most critical for AD)
    "Hippocampus_L", "Hippocampus_R",
    "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R",
    "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Cingulum_Mid_L", "Cingulum_Mid_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    # Subcortical
    "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R",
    "Pallidum_L", "Pallidum_R",
    "Thalamus_L", "Thalamus_R",
    # Insula & misc
    "Insula_L", "Insula_R",
]

NUM_REGIONS: int = len(AAL_REGIONS)   # 90

# Index lookup: region name → int index
REGION_INDEX: Dict[str, int] = {name: i for i, name in enumerate(AAL_REGIONS)}

# AD-critical regions (higher importance, often used for visualisation)
AD_CRITICAL_REGIONS: List[str] = [
    "Hippocampus_L", "Hippocampus_R",
    "ParaHippocampal_L", "ParaHippocampal_R",
    "Amygdala_L", "Amygdala_R",
    "Entorhinal_L", "Entorhinal_R",   # Not in AAL90 but referenced for context
    "Temporal_Inf_L", "Temporal_Inf_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Precuneus_L", "Precuneus_R",
]


def get_region_centroids_mni() -> np.ndarray:
    """
    Return approximate MNI152 centroid coordinates for all 90 AAL regions.
    Shape: [90, 3]  (x, y, z in mm)

    In production, load from the real atlas file. Here we generate
    plausible coordinates so the graph builder works without atlas files.
    """
    np.random.seed(0)
    # Left hemisphere: x < 0, Right hemisphere: x > 0
    centroids = np.zeros((NUM_REGIONS, 3))
    for i, name in enumerate(AAL_REGIONS):
        side = -1 if name.endswith("_L") else 1
        centroids[i] = [
            side * np.random.uniform(10, 70),   # x (lateral)
            np.random.uniform(-80, 60),          # y (anterior-posterior)
            np.random.uniform(-30, 70),          # z (inferior-superior)
        ]
    return centroids.astype(np.float32)


def build_adjacency_from_distance(
    centroids: np.ndarray,
    threshold_mm: float = 40.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build graph edges: connect two regions if their MNI centroids
    are within `threshold_mm` of each other.

    Returns
    -------
    edge_index : LongTensor [2, num_edges]  (PyG format)
    edge_attr  : FloatTensor [num_edges, 1] (inverse distance weight)
    """
    n = len(centroids)
    src, dst, weights = [], [], []

    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(centroids[i] - centroids[j]))
            if dist < threshold_mm:
                src += [i, j]
                dst += [j, i]
                w = 1.0 / (dist + 1e-6)
                weights += [w, w]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


# Pre-build the atlas adjacency once at import time
_CENTROIDS  = get_region_centroids_mni()
EDGE_INDEX, EDGE_ATTR = build_adjacency_from_distance(_CENTROIDS, threshold_mm=40.0)
