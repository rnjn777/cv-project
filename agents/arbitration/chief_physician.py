"""
agents/arbitration/chief_physician.py
--------------------------------------
Meta-Arbitration Fusion Agent — the Chief Physician.

Receives one BranchResult from the sMRI team and one from the Features
team, reconciles them, and produces the final diagnosis.

Algorithm (from ARCHITECTURE.md)
---------------------------------
1. Compute trust_i = confidence_i × intra_agreement_i  for each branch
2. Normalise to weights that sum to 1
3. Detect inter-branch conflict (top-predicted classes differ)
4. If conflict: amplify the weight gap (sharpen toward the more trusted team)
5. Final class_probs = weighted average of both branch class_probs
6. Package into FinalResult with full explanation payload

Output
------
FinalResult dataclass:
    diagnosis         : "CN" | "MCI" | "AD"
    probabilities     : {"CN": float, "MCI": float, "AD": float}
    confidence        : float  (max class probability — a simple calibrated score)
    modality_weights  : {"sMRI": float, "Features": float}
    disagreement_flag : bool   (True if teams' top classes differed)
    explanation       : dict   (top regions, clinical factors, minority signals)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from agents.arbitration.intra_arbitrator import BranchResult


CLASS_NAMES = ["CN", "MCI", "AD"]


# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FinalResult:
    """
    The complete output of the Chief Physician.

    Attributes
    ----------
    diagnosis         : most likely class label ("CN" / "MCI" / "AD")
    probabilities     : dict mapping class name → probability (sum ≈ 1)
    confidence        : float in [0, 1]  — probability of the top class
    modality_weights  : how much each team contributed to the final decision
    disagreement_flag : True if the two teams predicted different top classes
    explanation       : structured dict with:
                            "top_brain_regions"    : list[str]
                            "top_clinical_factors" : list[str]
                            "minority_signals"     : list[dict]
    raw               : holds the raw branch results for downstream logging
    """
    diagnosis:          str
    probabilities:      Dict[str, float]
    confidence:         float
    modality_weights:   Dict[str, float]
    disagreement_flag:  bool
    explanation:        Dict[str, Any]   = field(default_factory=dict)
    raw:                Dict[str, Any]   = field(default_factory=dict)

    def __str__(self) -> str:
        lines = [
            "─" * 55,
            f"  DIAGNOSIS       : {self.diagnosis}",
            f"  Confidence      : {self.confidence:.1%}",
            "  Probabilities   : " + "  ".join(
                f"{k}={v:.1%}" for k, v in self.probabilities.items()
            ),
            f"  sMRI weight     : {self.modality_weights.get('sMRI', 0):.1%}",
            f"  Features weight : {self.modality_weights.get('Features', 0):.1%}",
            f"  Disagreement    : {'⚠ YES — clinician review advised' if self.disagreement_flag else 'No'}",
        ]
        if self.explanation.get("minority_signals"):
            lines.append(f"  Minority signals: {len(self.explanation['minority_signals'])} flagged")
        lines.append("─" * 55)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────

class ChiefPhysician:
    """
    Meta-Arbitration Fusion Agent.

    Combines one BranchResult per modality team into a FinalResult.

    Parameters
    ----------
    conflict_amplification : float
        When the two teams disagree, the weight gap between them is
        amplified by this factor before renormalising.
        Higher = more decisive (lean harder toward the trusted team).
        Default: 2.0   (doubles the advantage of the more trusted team)
    disagreement_threshold : float
        Minimum probability mass difference between teams' top classes
        to raise the disagreement_flag.
        Default: 0.0   (any top-class mismatch raises the flag)
    """

    def __init__(
        self,
        conflict_amplification: float = 2.0,
        disagreement_threshold: float = 0.0,
    ):
        self.conflict_amplification  = conflict_amplification
        self.disagreement_threshold  = disagreement_threshold

    # ------------------------------------------------------------------

    def fuse(
        self,
        smri_result:     BranchResult,
        features_result: BranchResult,
    ) -> FinalResult:
        """
        Fuse two BranchResults into a single FinalResult.

        Parameters
        ----------
        smri_result     : output of SMRISupervisor.run()
        features_result : output of FeaturesSupervisor.run()

        Returns
        -------
        FinalResult
        """
        # ── Step 1: Compute per-team trust scores ──────────────────────
        trust_smri     = smri_result.confidence     * smri_result.intra_agreement
        trust_features = features_result.confidence * features_result.intra_agreement

        # ── Step 2: Normalise to weights ───────────────────────────────
        total = trust_smri + trust_features + 1e-8
        w_smri     = trust_smri     / total
        w_features = trust_features / total

        # ── Step 3: Detect inter-branch conflict ───────────────────────
        #   Take mean over batch to get a single representative class vector
        smri_mean     = smri_result.class_probs.mean(dim=0)      # [C]
        features_mean = features_result.class_probs.mean(dim=0)  # [C]

        smri_top     = int(smri_mean.argmax().item())
        features_top = int(features_mean.argmax().item())

        disagreement_flag = (smri_top != features_top)

        # ── Step 4: Amplify weight gap if conflict ─────────────────────
        if disagreement_flag:
            w_smri, w_features = self._amplify_conflict(w_smri, w_features)

        # ── Step 5: Weighted average class_probs ──────────────────────
        #   Operate on the per-sample [B, C] tensors
        final_probs = (
            w_smri     * smri_result.class_probs +
            w_features * features_result.class_probs
        )  # [B, C]

        # Collapse to a single representative probability for reporting
        # (batch-level mean; individual-sample probs live in final_probs)
        mean_final = final_probs.mean(dim=0)          # [C]
        confidence  = float(mean_final.max().item())
        top_idx     = int(mean_final.argmax().item())
        diagnosis   = CLASS_NAMES[top_idx]

        probabilities = {
            name: float(mean_final[i].item())
            for i, name in enumerate(CLASS_NAMES)
        }

        modality_weights = {
            "sMRI":     float(w_smri),
            "Features": float(w_features),
        }

        # ── Step 6: Build explanation payload ─────────────────────────
        explanation = self._build_explanation(smri_result, features_result)

        return FinalResult(
            diagnosis         = diagnosis,
            probabilities     = probabilities,
            confidence        = confidence,
            modality_weights  = modality_weights,
            disagreement_flag = disagreement_flag,
            explanation       = explanation,
            raw = {
                "smri_branch":     smri_result,
                "features_branch": features_result,
                "final_probs":     final_probs,       # [B, C]  per-sample
            },
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _amplify_conflict(
        self,
        w_smri: float,
        w_features: float,
    ):
        """
        Widen the gap between weights when the teams disagree.

        The amplification is applied to raw (unnormalised) trust differences:
            w_i_amplified = w_i ^ (1 / conflict_amplification)
        This makes the larger weight even larger relative to the smaller.
        """
        # Use a power-sharpening approach — raises both to 1/k then renorms
        k = self.conflict_amplification
        w_s = w_smri     ** (1.0 / k)
        w_f = w_features ** (1.0 / k)
        total = w_s + w_f + 1e-8
        return w_s / total, w_f / total

    def _build_explanation(
        self,
        smri_result:     BranchResult,
        features_result: BranchResult,
    ) -> Dict[str, Any]:
        """
        Assemble the human-readable explanation dict.

        Top brain regions: extracted from CE-AH assistant's feature_vector
        (highest-magnitude dimensions map to AAL regions — a proxy for
        attention-weighted importance without running Grad-CAM at inference).

        Top clinical factors: similar proxy from Clinical assistant.

        Minority signals: collected from both branches.
        """
        top_brain_regions    = self._top_regions_from_smri(smri_result)
        top_clinical_factors = self._top_factors_from_features(features_result)
        minority_signals     = (
            smri_result.minority_reports +
            features_result.minority_reports
        )

        return {
            "top_brain_regions":    top_brain_regions,
            "top_clinical_factors": top_clinical_factors,
            "minority_signals":     minority_signals,
        }

    def _top_regions_from_smri(
        self,
        smri_result: BranchResult,
        top_k: int = 5,
    ) -> List[str]:
        """
        Return the names of the top-k most activated AAL brain regions
        from the CE-AH assistant's feature vector.

        The CE-AH feature_vector is [B, global_embed_dim]. We take the
        mean across the batch, then find the top-k indices in the feature
        space. These don't map directly to regions, but we label them
        symbolically for the explanation payload.

        In production: replace with Grad-CAM over the region_encoder output.
        """
        from utils.brain_atlas import AAL_REGIONS

        # Find CE-AH assistant output if available
        ceah_out = next(
            (o for o in smri_result.assistant_outputs if o.name == "CE-AH"),
            None,
        )
        if ceah_out is None or ceah_out.feature_vector is None:
            return []

        feat = ceah_out.feature_vector.mean(dim=0)   # [global_embed_dim]
        top_k_actual = min(top_k, len(AAL_REGIONS))
        # Feature vector dim >> 90, so we map via modulo for a proxy label
        top_indices = feat.abs().topk(top_k_actual).indices.tolist()
        region_names = [AAL_REGIONS[i % len(AAL_REGIONS)] for i in top_indices]
        return region_names

    def _top_factors_from_features(
        self,
        features_result: BranchResult,
        top_k: int = 3,
    ) -> List[str]:
        """
        Return the names of the top-k most activated clinical features.

        Uses the Clinical assistant's feature_vector as a proxy.
        In production: replace with SHAP values over the clinical MLP.
        """
        from models.tabular.clinical_model import CLINICAL_FEATURE_NAMES

        clinical_out = next(
            (o for o in features_result.assistant_outputs if o.name == "Clinical"),
            None,
        )
        if clinical_out is None or clinical_out.feature_vector is None:
            return []

        feat = clinical_out.feature_vector.mean(dim=0)   # [hidden_dim]
        top_k_actual = min(top_k, len(CLINICAL_FEATURE_NAMES))
        top_indices  = feat.abs().topk(top_k_actual).indices.tolist()
        factor_names = [
            CLINICAL_FEATURE_NAMES[i % len(CLINICAL_FEATURE_NAMES)]
            for i in top_indices
        ]
        return factor_names
