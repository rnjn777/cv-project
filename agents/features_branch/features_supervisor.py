"""
agents/features_branch/features_supervisor.py
----------------------------------------------
Features Branch Supervisor.

Runs both features assistants (Clinical and Cognitive) on the same
tabular batch, then hands their outputs to the IntraArbitrator to
produce a single trusted branch verdict.

Usage
-----
    supervisor = FeaturesSupervisor(cfg)
    branch_result = supervisor.run(batch)
    # batch must contain "clinical" and "cognitive" keys
"""

from __future__ import annotations

from omegaconf import DictConfig

from agents.base_assistant import AssistantOutput
from agents.features_branch.clinical_assistant import ClinicalAssistant
from agents.features_branch.cognitive_assistant import CognitiveAssistant
from agents.arbitration.intra_arbitrator import IntraArbitrator, BranchResult


class FeaturesSupervisor:
    """
    Orchestrates the clinical features branch.

    Steps
    -----
    1. Run Clinical assistant   → AssistantOutput
    2. Run Cognitive assistant  → AssistantOutput
    3. IntraArbitrator          → BranchResult  (weighted consensus + minority reports)

    Parameters
    ----------
    cfg : Hydra DictConfig
    num_classes : int (default 3 — CN / MCI / AD)
    minority_weight_thresh : float
    minority_conf_thresh : float
    """

    BRANCH_NAME = "Features"

    def __init__(
        self,
        cfg: DictConfig,
        num_classes: int = 3,
        minority_weight_thresh: float = 0.3,
        minority_conf_thresh: float = 0.7,
    ):
        self.clinical_assistant  = ClinicalAssistant(cfg, num_classes=num_classes)
        self.cognitive_assistant = CognitiveAssistant(cfg, num_classes=num_classes)

        self.arbitrator = IntraArbitrator(
            branch_name            = self.BRANCH_NAME,
            minority_weight_thresh = minority_weight_thresh,
            minority_conf_thresh   = minority_conf_thresh,
        )

    # ------------------------------------------------------------------

    def run(self, batch: dict) -> BranchResult:
        """
        Run the full features branch for a batch.

        Parameters
        ----------
        batch : dict with keys:
            "clinical"  : FloatTensor [B, num_clinical_features]
            "cognitive" : FloatTensor [B, num_cognitive_features]

        Returns
        -------
        BranchResult — single trusted verdict from the features team
        """
        clinical_out:  AssistantOutput = self.clinical_assistant.predict(batch)
        cognitive_out: AssistantOutput = self.cognitive_assistant.predict(batch)

        return self.arbitrator.arbitrate([clinical_out, cognitive_out])

    # ------------------------------------------------------------------

    def to(self, device):
        """Move both assistants to the given device."""
        self.clinical_assistant.to(device)
        self.cognitive_assistant.to(device)
        return self

    def train(self, mode: bool = True):
        """Propagate training mode to assistants."""
        self.clinical_assistant.train(mode)
        self.cognitive_assistant.train(mode)
        return self

    def eval(self):
        """Set both assistants to eval mode."""
        self.clinical_assistant.eval()
        self.cognitive_assistant.eval()
        return self

    def parameters(self):
        """Yield all parameters (for passing to an optimiser)."""
        yield from self.clinical_assistant.parameters()
        yield from self.cognitive_assistant.parameters()

    def state_dict(self) -> dict:
        return {
            "clinical":  self.clinical_assistant.state_dict(),
            "cognitive": self.cognitive_assistant.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        self.clinical_assistant.load_state_dict(sd["clinical"])
        self.cognitive_assistant.load_state_dict(sd["cognitive"])
