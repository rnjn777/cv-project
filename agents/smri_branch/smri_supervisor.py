"""
agents/smri_branch/smri_supervisor.py
--------------------------------------
sMRI Branch Supervisor.

Runs both sMRI assistants (CE-AH and GNN) on the same MRI batch,
then hands their outputs to the IntraArbitrator to produce a single
trusted branch verdict.

Usage
-----
    supervisor = SMRISupervisor(cfg)
    branch_result = supervisor.run(batch)  # batch must contain "mri" key
"""

from __future__ import annotations

from omegaconf import DictConfig

from agents.base_assistant import AssistantOutput
from agents.smri_branch.ceah_assistant import CEAHAssistant
from agents.smri_branch.gnn_assistant import GNNAssistant
from agents.arbitration.intra_arbitrator import IntraArbitrator, BranchResult


class SMRISupervisor:
    """
    Orchestrates the sMRI branch.

    Steps
    -----
    1. Run CE-AH assistant  → AssistantOutput
    2. Run GNN assistant    → AssistantOutput
    3. IntraArbitrator      → BranchResult  (weighted consensus + minority reports)

    Parameters
    ----------
    cfg : Hydra DictConfig
    num_classes : int (default 3 — CN / MCI / AD)
    minority_weight_thresh : float
        Assistants below this peer weight are candidate minority reporters.
    minority_conf_thresh : float
        Only flag as minority report if confidence is also this high.
    """

    BRANCH_NAME = "sMRI"

    def __init__(
        self,
        cfg: DictConfig,
        num_classes: int = 3,
        minority_weight_thresh: float = 0.3,
        minority_conf_thresh: float = 0.7,
    ):
        self.ceah_assistant = CEAHAssistant(cfg, num_classes=num_classes)
        self.gnn_assistant  = GNNAssistant(cfg, num_classes=num_classes)

        self.arbitrator = IntraArbitrator(
            branch_name            = self.BRANCH_NAME,
            minority_weight_thresh = minority_weight_thresh,
            minority_conf_thresh   = minority_conf_thresh,
        )

    # ------------------------------------------------------------------

    def run(self, batch: dict) -> BranchResult:
        """
        Run the full sMRI branch for a batch.

        Parameters
        ----------
        batch : dict with at least key "mri" : FloatTensor [B, 1, D, H, W]

        Returns
        -------
        BranchResult — single trusted verdict from the sMRI team
        """
        ceah_out: AssistantOutput = self.ceah_assistant.predict(batch)
        gnn_out:  AssistantOutput = self.gnn_assistant.predict(batch)

        return self.arbitrator.arbitrate([ceah_out, gnn_out])

    # ------------------------------------------------------------------

    def to(self, device):
        """Move both assistants to the given device."""
        self.ceah_assistant.to(device)
        self.gnn_assistant.to(device)
        return self

    def train(self, mode: bool = True):
        """Propagate training mode to assistants (for joint fine-tuning)."""
        self.ceah_assistant.train(mode)
        self.gnn_assistant.train(mode)
        return self

    def eval(self):
        """Set both assistants to eval mode."""
        self.ceah_assistant.eval()
        self.gnn_assistant.eval()
        return self

    def parameters(self):
        """Yield all parameters (for passing to an optimiser)."""
        yield from self.ceah_assistant.parameters()
        yield from self.gnn_assistant.parameters()

    def state_dict(self) -> dict:
        return {
            "ceah": self.ceah_assistant.state_dict(),
            "gnn":  self.gnn_assistant.state_dict(),
        }

    def load_state_dict(self, sd: dict):
        self.ceah_assistant.load_state_dict(sd["ceah"])
        self.gnn_assistant.load_state_dict(sd["gnn"])
        self.gnn_assistant.load_state_dict(sd["gnn"])
