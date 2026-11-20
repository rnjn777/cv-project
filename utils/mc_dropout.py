"""
utils/mc_dropout.py
-------------------
Monte Carlo Dropout inference helper.

During inference, we keep dropout ON and run N forward passes.
The spread (std dev) of the resulting probability distributions
gives us an honest measure of the model's uncertainty.
"""

import torch
import torch.nn as nn
from typing import Tuple


def enable_dropout(model: nn.Module) -> None:
    """Switch all Dropout layers to training mode so they stay active at inference."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_inference(
    model: nn.Module,
    x,                          # input — can be a tensor or tuple of tensors
    n_samples: int = 30,
    device: torch.device = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Run MC Dropout inference.

    Returns
    -------
    mean_probs : Tensor [num_classes]
        Mean probability distribution across all stochastic passes.
    std_probs  : Tensor [num_classes]
        Std dev across passes — higher = more uncertain.
    confidence : float
        Scalar in [0, 1]. High = stable & confident, low = unstable.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    enable_dropout(model)   # keep dropout live

    all_probs = []
    for _ in range(n_samples):
        if isinstance(x, dict):
            # Batch dict — move tensor values to device, pass dict to model
            x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in x.items()}
            logits = model(x_dev)
        elif isinstance(x, (list, tuple)):
            logits = model(*[t.to(device) for t in x])
        else:
            logits = model(x.to(device))

        probs = torch.softmax(logits, dim=-1)   # [batch, num_classes]
        all_probs.append(probs)

    # Stack: [n_samples, batch, num_classes]
    stacked = torch.stack(all_probs, dim=0)

    mean_probs = stacked.mean(dim=0)    # [batch, num_classes]
    std_probs  = stacked.std(dim=0)     # [batch, num_classes]

    # Confidence = 1 - mean std across classes
    # A high std means the model is uncertain; we invert to get confidence.
    confidence = float(1.0 - std_probs.mean().item())
    confidence = max(0.0, min(1.0, confidence))   # clamp to [0, 1]

    return mean_probs, std_probs, confidence
