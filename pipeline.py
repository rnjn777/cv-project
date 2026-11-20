"""
pipeline.py
-----------
End-to-end inference pipeline for the multi-agent Alzheimer's detector.

Assembles the full hierarchy:

    batch
      ├── sMRI Team
      │     ├── CEAHAssistant
      │     ├── GNNAssistant
      │     └── IntraArbitrator  →  sMRI BranchResult
      ├── Features Team
      │     ├── ClinicalAssistant
      │     ├── CognitiveAssistant
      │     └── IntraArbitrator  →  Features BranchResult
      ├── CrossAttentionFusion   →  Inter-agent attention & cross-modal interaction
      └── ChiefPhysician         →  FinalResult (Dynamic Meta-Arbitration)

Usage
-----
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/config.yaml")

    pipeline = AlzheimerPipeline(cfg)
    result   = pipeline.predict(batch)
    print(result)

The pipeline also exposes helpers for moving to device, saving /
loading checkpoints, and listing all trainable parameters so a
single optimiser can cover the whole system.
"""

from __future__ import annotations

import os
from typing import Dict, Optional
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from agents.smri_branch.smri_supervisor         import SMRISupervisor
from agents.features_branch.features_supervisor import FeaturesSupervisor
from agents.arbitration.chief_physician         import ChiefPhysician, FinalResult
from agents.arbitration.intra_arbitrator        import BranchResult
from models.fusion.cross_attention             import CrossAttentionFusion


class AlzheimerPipeline:
    """
    Full multi-agent Alzheimer's detection pipeline.

    Parameters
    ----------
    cfg : Hydra DictConfig  (merged from configs/config.yaml + model YAMLs)
    device : str | torch.device  (default: "cuda" if available, else "cpu")
    conflict_amplification : float
        Passed to ChiefPhysician — how aggressively to favour the more
        trusted team when the two branches disagree.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: Optional[str] = None,
        conflict_amplification: float = 2.5,
    ):
        self.cfg    = cfg
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        num_classes = cfg.get("num_classes", 3)

        # Build teams
        self.smri_supervisor     = SMRISupervisor(cfg, num_classes=num_classes)
        self.features_supervisor = FeaturesSupervisor(cfg, num_classes=num_classes)

        # Cross-Attention Multimodal Feature Fusion
        dim_clin = cfg.model.clinical.hidden_dims[-1] if hasattr(cfg.model.clinical, "hidden_dims") else 64
        dim_cogn = cfg.model.cognitive.hidden_dims[-1] if hasattr(cfg.model.cognitive, "hidden_dims") else 64
        self.fusion = CrossAttentionFusion(
            dim_ceah      = getattr(cfg.model.ceah, "global_embed_dim", 512),
            dim_gnn       = getattr(cfg.model.gnn, "gnn_out_dim", 256),
            dim_clinical  = dim_clin,
            dim_cognitive = dim_cogn,
            shared_dim    = 128,
            num_classes   = num_classes,
        )

        # Chief Physician is stateless (pure function) — no weights
        self.chief_physician = ChiefPhysician(
            conflict_amplification = conflict_amplification,
        )

        # Move models to device
        self.to(self.device)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, batch: Dict[str, torch.Tensor], use_tta: bool = False) -> FinalResult:
        """
        Run the full pipeline on a single batch.

        Parameters
        ----------
        batch : dict with keys
            "mri"       : FloatTensor [B, 1, D, H, W]
            "clinical"  : FloatTensor [B, num_clinical_features]
            "cognitive" : FloatTensor [B, num_cognitive_features]
            "label"     : LongTensor  [B]  (optional — for logging)
        use_tta : bool (default False)
            Test-Time Augmentation: averages original scan with sagittal plane mirror

        Returns
        -------
        FinalResult
        """
        # Set everything to eval before running
        self.smri_supervisor.eval()
        self.features_supervisor.eval()
        self.fusion.eval()

        mri = batch.get("mri", None)
        has_valid_mri = (
            mri is not None and isinstance(mri, torch.Tensor) and
            mri.abs().sum().item() > 0 and mri.std().item() >= 1e-4
        )

        smri_result: BranchResult = self.smri_supervisor.run(batch)

        if use_tta and has_valid_mri:
            batch_flip = dict(batch)
            batch_flip["mri"] = torch.flip(mri, dims=[-1])
            smri_flip: BranchResult = self.smri_supervisor.run(batch_flip)
            smri_result.class_probs = 0.5 * (smri_result.class_probs + smri_flip.class_probs)
            smri_result.confidence = 0.5 * (smri_result.confidence + smri_flip.confidence)

        features_result: BranchResult = self.features_supervisor.run(batch)

        # Extract specialist feature vectors for neural cross-modal interaction
        fusion_probs = None
        attn_matrix = None
        try:
            ceah_out = next((o for o in smri_result.assistant_outputs if o.name == "CE-AH"), None)
            gnn_out  = next((o for o in smri_result.assistant_outputs if o.name == "GNN"), None)
            clin_out = next((o for o in features_result.assistant_outputs if o.name == "Clinical"), None)
            cogn_out = next((o for o in features_result.assistant_outputs if o.name == "Cognitive"), None)

            if (ceah_out is not None and gnn_out is not None and
                clin_out is not None and cogn_out is not None and
                ceah_out.feature_vector is not None and gnn_out.feature_vector is not None and
                clin_out.feature_vector is not None and cogn_out.feature_vector is not None):

                f_ceah = ceah_out.feature_vector.to(self.device)
                f_gnn  = gnn_out.feature_vector.to(self.device)
                f_clin = clin_out.feature_vector.to(self.device)
                f_cogn = cogn_out.feature_vector.to(self.device)

                logits_fusion, attn_matrix = self.fusion(f_ceah, f_gnn, f_clin, f_cogn)
                fusion_probs = F.softmax(logits_fusion, dim=-1)
        except Exception:
            fusion_probs = None
            attn_matrix = None

        # Fuse with Chief Physician (combines dynamic intra-arbitration consensus + neural cross-attention)
        final: FinalResult = self.chief_physician.fuse(
            smri_result,
            features_result,
            fusion_probs=fusion_probs,
            attn_matrix=attn_matrix,
        )

        return final

    # ------------------------------------------------------------------
    # Training utilities
    # ------------------------------------------------------------------

    def train_mode(self):
        """Set all assistants and fusion module to training mode."""
        self.smri_supervisor.train()
        self.features_supervisor.train()
        self.fusion.train()

    def eval_mode(self):
        """Set all assistants and fusion module to eval mode."""
        self.smri_supervisor.eval()
        self.features_supervisor.eval()
        self.fusion.eval()

    def smri_parameters(self):
        """Yield parameters belonging to the sMRI branch only."""
        yield from self.smri_supervisor.parameters()

    def features_parameters(self):
        """Yield parameters belonging to the Features branch only."""
        yield from self.features_supervisor.parameters()

    def fusion_parameters(self):
        """Yield parameters belonging to the CrossAttentionFusion network."""
        yield from self.fusion.parameters()

    def all_parameters(self):
        """Yield all trainable parameters across both branches and fusion module."""
        yield from self.smri_parameters()
        yield from self.features_parameters()
        yield from self.fusion_parameters()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    def to(self, device):
        """Move all assistant models and fusion module to the given device."""
        self.device = torch.device(device)
        self.smri_supervisor.to(self.device)
        self.features_supervisor.to(self.device)
        self.fusion.to(self.device)
        return self

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra: Optional[dict] = None):
        """
        Save the full pipeline state.

        Parameters
        ----------
        path  : file path for the .pt checkpoint
        extra : optional dict of extra metadata to embed (e.g. epoch, metrics)
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "smri_branch":     self.smri_supervisor.state_dict(),
            "features_branch": self.features_supervisor.state_dict(),
            "fusion":          self.fusion.state_dict(),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[Pipeline] Checkpoint saved -> {path}")

    def load_checkpoint(self, path: str, strict: bool = False):
        """
        Load pipeline weights from a checkpoint.

        Parameters
        ----------
        path   : path to the .pt checkpoint file
        strict : if True, all keys must match exactly
        """
        sd = torch.load(path, map_location=self.device)
        if "smri_branch" in sd:
            self.smri_supervisor.load_state_dict(sd["smri_branch"], strict=strict)
        if "features_branch" in sd:
            self.features_supervisor.load_state_dict(sd["features_branch"], strict=strict)
        if "fusion" in sd:
            self.fusion.load_state_dict(sd["fusion"], strict=strict)
        print(f"[Pipeline] Checkpoint loaded <- {path}")
        return sd

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_smri = sum(p.numel() for p in self.smri_supervisor.parameters())
        n_feat = sum(p.numel() for p in self.features_supervisor.parameters())
        n_fuse = sum(p.numel() for p in self.fusion.parameters())
        return (
            f"AlzheimerPipeline(\n"
            f"  device          = {self.device}\n"
            f"  sMRI params     = {n_smri:,}\n"
            f"  Features params = {n_feat:,}\n"
            f"  Fusion params   = {n_fuse:,}\n"
            f"  Total params    = {n_smri + n_feat + n_fuse:,}\n"
            f")"
        )
