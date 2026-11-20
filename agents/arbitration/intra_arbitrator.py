"""
agents/arbitration/intra_arbitrator.py
----------------------------------------
Intra-Branch Arbitration Agent.

One instance per branch (sMRI branch, Features branch).
Receives all assistant outputs from its branch and produces
a single trusted BranchResult.

Algorithm
---------
1. Build N×N cosine agreement matrix between all assistant class_probs
2. Peer validation weight: weight_i = mean agreement with all j ≠ i
3. Minority report detection: low weight but high confidence → specialist signal
4. Weighted consensus: branch_probs = Σ(weight_i × probs_i) / Σ weight_i
5. Report overall intra_agreement score for the Fusion Agent
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from agents.base_assistant import AssistantOutput


@dataclass
class BranchResult:
    """
    Output of an Intra-Branch Arbitration Agent.
    This is what gets passed to the Meta-Arbitration Fusion Agent.
    """
    branch_name:      str
    class_probs:      torch.Tensor          # [B, num_classes]
    confidence:       float                  # weighted avg confidence
    intra_agreement:  float                  # overall branch agreement score
    weights:          Dict[str, float]       # assistant_name → weight
    minority_reports: List[Dict[str, Any]]   # list of flagged outlier outputs
    assistant_outputs: List[AssistantOutput] # raw outputs (for explainability)


class IntraArbitrator:
    """
    Intra-Branch Arbitration Agent.

    Parameters
    ----------
    branch_name           : "sMRI" or "Features"
    minority_weight_thresh : assistants below this weight are potential outliers
    minority_conf_thresh   : only flag as minority report if confidence also high
    """

    def __init__(
        self,
        branch_name:            str,
        minority_weight_thresh: float = 0.3,
        minority_conf_thresh:   float = 0.7,
    ):
        self.branch_name            = branch_name
        self.minority_weight_thresh = minority_weight_thresh
        self.minority_conf_thresh   = minority_conf_thresh

    def arbitrate(self, outputs: List[AssistantOutput]) -> BranchResult:
        """
        Run the full arbitration pipeline on a list of assistant outputs.

        Parameters
        ----------
        outputs : list of AssistantOutput (one per assistant in this branch)

        Returns
        -------
        BranchResult
        """
        if len(outputs) == 1:
            # Only one assistant — trivial case, no arbitration needed
            o = outputs[0]
            return BranchResult(
                branch_name       = self.branch_name,
                class_probs       = o.class_probs,
                confidence        = o.confidence,
                intra_agreement   = 1.0,
                weights           = {o.name: 1.0},
                minority_reports  = [],
                assistant_outputs = outputs,
            )

        # Step 1 — Cosine agreement matrix [N, N]
        agreement_matrix = self._build_agreement_matrix(outputs)

        # Step 2 — Peer validation weights [N]
        raw_weights = self._compute_peer_weights(agreement_matrix)

        # Step 3 — Normalise weights to sum to 1
        names   = [o.name for o in outputs]
        weights = {name: float(w) for name, w in zip(names, raw_weights)}

        # Step 4 — Detect minority reports
        minority_reports = self._detect_minority_reports(outputs, weights)

        # Step 5 — Weighted consensus of class_probs
        branch_probs = self._weighted_consensus(outputs, raw_weights)

        # Step 6 — Overall agreement = mean of off-diagonal agreement matrix entries
        n = len(outputs)
        off_diag = agreement_matrix[~torch.eye(n, dtype=torch.bool)].mean()
        intra_agreement = float(off_diag.item())

        # Weighted avg confidence
        w_tensor = raw_weights
        confs    = torch.tensor([o.confidence for o in outputs])
        conf     = float((w_tensor * confs).sum().item())

        return BranchResult(
            branch_name       = self.branch_name,
            class_probs       = branch_probs,
            confidence        = conf,
            intra_agreement   = intra_agreement,
            weights           = weights,
            minority_reports  = minority_reports,
            assistant_outputs = outputs,
        )

    # ── Internal methods ────────────────────────────────────────────────────

    def _build_agreement_matrix(
        self, outputs: List[AssistantOutput]
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarities between class_probs vectors.
        Returns: FloatTensor [N, N]

        We take the mean over the batch dimension so we get a single
        scalar agreement per assistant pair.
        """
        probs = [o.class_probs for o in outputs]   # each [B, C]
        n = len(probs)
        matrix = torch.zeros(n, n)

        for i in range(n):
            for j in range(n):
                # Average cosine similarity over batch
                sim = F.cosine_similarity(
                    probs[i], probs[j], dim=-1
                ).mean()   # scalar
                matrix[i, j] = sim

        return matrix   # [N, N]

    def _compute_peer_weights(self, agreement_matrix: torch.Tensor) -> torch.Tensor:
        """
        For each assistant i:
            weight_i = mean agreement with all j ≠ i

        Then normalise so weights sum to 1.
        Returns: FloatTensor [N]
        """
        n = agreement_matrix.shape[0]
        weights = torch.zeros(n)

        for i in range(n):
            others = [agreement_matrix[i, j] for j in range(n) if j != i]
            weights[i] = torch.stack(others).mean()

        # Softmax to get clean probabilities that sum to 1
        weights = torch.softmax(weights, dim=0)
        return weights

    def _detect_minority_reports(
        self,
        outputs: List[AssistantOutput],
        weights: Dict[str, float],
    ) -> List[Dict]:
        """
        Flag any assistant that has:
          - Low peer weight (< minority_weight_thresh)
          - High confidence (> minority_conf_thresh)

        These are potential specialist signals — suppress them in the
        weighted average but pass them up to the Fusion Agent as flags.
        """
        reports = []
        for o in outputs:
            w = weights[o.name]
            if w < self.minority_weight_thresh and o.confidence > self.minority_conf_thresh:
                prediction = int(o.class_probs.mean(0).argmax().item())
                reports.append({
                    "assistant": o.name,
                    "weight":    w,
                    "confidence": o.confidence,
                    "prediction": prediction,
                    "class_probs": o.class_probs,
                    "message": (
                        f"{o.name} strongly disagrees (conf={o.confidence:.2f}) "
                        f"but has low peer agreement (weight={w:.2f}). "
                        f"Predicts class {prediction}. Possible specialist signal."
                    ),
                })
        return reports

    def _weighted_consensus(
        self,
        outputs: List[AssistantOutput],
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Weighted average of class_probs.
        weights: [N] (normalised, sum to 1)
        Returns: [B, num_classes]
        """
        result = None
        for o, w in zip(outputs, weights):
            weighted = o.class_probs * float(w.item())
            result = weighted if result is None else result + weighted
        return result
