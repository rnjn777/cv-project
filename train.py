"""
train.py
--------
Three-phase training script for the multi-agent Alzheimer's detector.

Phases (as per ARCHITECTURE.md)
--------------------------------
Phase 1 — sMRI Branch
    Train CE-AH and GNN assistants independently (separate optimisers),
    then run one epoch of joint sMRI branch training.

Phase 2 — Features Branch
    Train Clinical and Cognitive assistants independently,
    then run one epoch of joint features branch training.

Phase 3 — Joint Fine-tune (optional, controlled by cfg.training.joint_finetune)
    Fine-tune all four assistants simultaneously with a reduced LR.

Each assistant is trained with cross-entropy + label smoothing.
Val metrics: accuracy, macro-F1, per-class AUC.

Usage
-----
    python train.py                         # uses configs/config.yaml
    python train.py data.synthetic=true     # force synthetic data
    python train.py training.epochs=5       # quick debug run
"""

from __future__ import annotations

import os
import sys
import math
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import hydra
from omegaconf import DictConfig, OmegaConf

from data.dataset import build_dataloaders
from pipeline import AlzheimerPipeline

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(params, cfg_training) -> AdamW:
    return AdamW(
        params,
        lr           = cfg_training.lr,
        weight_decay = cfg_training.weight_decay,
    )


def make_scheduler(optimizer, cfg_training, n_steps_per_epoch: int):
    """
    Cosine schedule with linear warm-up.
    Total steps = epochs × steps_per_epoch.
    """
    warmup_steps = cfg_training.warmup_epochs * n_steps_per_epoch
    total_steps  = cfg_training.epochs        * n_steps_per_epoch

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=max(warmup_steps, 1))
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))

    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


def smooth_ce_loss(logits: torch.Tensor, labels: torch.Tensor,
                   smoothing: float = 0.1,
                   weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross-entropy with label smoothing and optional class weights."""
    return F.cross_entropy(logits, labels, label_smoothing=smoothing, weight=weight)


@torch.no_grad()
def evaluate(
    model_forward,          # callable: batch → logits [B, C]
    loader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1, and per-class AUC on a DataLoader.

    model_forward must return logits (not softmax probabilities).
    """
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        logits = model_forward(batch)           # [B, C]
        probs  = torch.softmax(logits, dim=-1)  # [B, C]
        preds  = logits.argmax(dim=-1)          # [B]

        all_preds.append(preds.cpu())
        all_labels.append(batch["label"].cpu())
        all_probs.append(probs.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    probs  = torch.cat(all_probs).numpy()

    acc    = float(accuracy_score(labels, preds))
    f1     = float(f1_score(labels, preds, average="macro", zero_division=0))

    # AUC: one-vs-rest, handle edge cases with fewer than 2 classes present
    try:
        auc = float(roc_auc_score(
            labels, probs, multi_class="ovr", average="macro"
        ))
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "macro_f1": f1, "macro_auc": auc}


# ─────────────────────────────────────────────────────────────────────────────
# Single-assistant training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_assistant(
    assistant: nn.Module,
    input_key: str,           # "mri" | "clinical" | "cognitive"
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg_training,
    device: torch.device,
    checkpoint_dir: str,
    tag: str,                 # used for log prefix and checkpoint filename
    num_classes: int = 3,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """
    Train a single assistant end-to-end.

    The assistant is an nn.Module whose forward(batch) → logits.
    We use the BaseAssistant.forward() which calls _forward and returns
    logits only (no MC Dropout) — that is appropriate for training.
    """
    log.info(f"[{tag}] Starting training for {cfg_training.epochs} epochs")
    assistant = assistant.to(device)
    assistant.train()

    optimizer = make_optimizer(assistant.parameters(), cfg_training)
    scheduler = make_scheduler(optimizer, cfg_training, len(train_loader))

    best_f1   = -1.0
    patience  = cfg_training.patience
    no_improve = 0

    for epoch in range(cfg_training.epochs):
        assistant.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch["label"]

            logits, _ = assistant._forward(batch)
            loss = smooth_ce_loss(logits, labels, cfg_training.label_smoothing, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(assistant.parameters(),
                                     cfg_training.gradient_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if (step + 1) % cfg_training.log_every_n_steps == 0:
                log.info(
                    f"[{tag}] Epoch {epoch+1} step {step+1}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)

        # ── Validation ─────────────────────────────────────────────────
        def _fwd(b):
            logits, _ = assistant._forward(b)
            return logits

        metrics = evaluate(_fwd, val_loader, device, num_classes)
        elapsed = time.time() - t0

        log.info(
            f"[{tag}] Epoch {epoch+1}/{cfg_training.epochs} "
            f"loss={avg_loss:.4f} "
            f"val_acc={metrics['accuracy']:.3f} "
            f"val_f1={metrics['macro_f1']:.3f} "
            f"val_auc={metrics['macro_auc']:.3f} "
            f"({elapsed:.1f}s)"
        )

        # ── Early stopping & checkpointing ─────────────────────────────
        if metrics["macro_f1"] > best_f1:
            best_f1    = metrics["macro_f1"]
            no_improve = 0
            ckpt_path  = os.path.join(checkpoint_dir, f"{tag}_best.pt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(assistant.state_dict(), ckpt_path)
            log.info(f"[{tag}] New best val F1={best_f1:.3f} -> {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"[{tag}] Early stop after {epoch+1} epochs (patience={patience})")
                break

    # Load best weights back
    ckpt_path = os.path.join(checkpoint_dir, f"{tag}_best.pt")
    if os.path.exists(ckpt_path):
        assistant.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        log.info(f"[{tag}] Restored best weights <- {ckpt_path}")

    return assistant


# ─────────────────────────────────────────────────────────────────────────────
# Joint branch training (post-assistant training)
# ─────────────────────────────────────────────────────────────────────────────

def train_branch_joint(
    branch,                   # SMRISupervisor or FeaturesSupervisor
    branch_name: str,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg_training,
    device: torch.device,
    checkpoint_dir: str,
    num_classes: int = 3,
    n_epochs: int = 5,
    class_weights: Optional[torch.Tensor] = None,
):
    """
    Fine-tune the two assistants in a branch jointly for a few epochs.
    Loss = average CE of both assistants' logits against the same label.
    """
    log.info(f"[{branch_name} joint] Starting {n_epochs}-epoch joint fine-tune")

    # Get all branch parameters into one optimiser
    all_params = list(branch.parameters())
    optimizer  = AdamW(all_params, lr=cfg_training.lr * 0.5,
                       weight_decay=cfg_training.weight_decay)

    best_loss  = float("inf")

    for epoch in range(n_epochs):
        branch.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch["label"]

            # Forward both assistants manually (branch.run() uses no_grad)
            if branch_name.lower() == "smri":
                logits_a, _ = branch.ceah_assistant._forward(batch)
                logits_b, _ = branch.gnn_assistant._forward(batch)
            else:
                logits_a, _ = branch.clinical_assistant._forward(batch)
                logits_b, _ = branch.cognitive_assistant._forward(batch)

            loss = (
                smooth_ce_loss(logits_a, labels, cfg_training.label_smoothing, weight=class_weights) +
                smooth_ce_loss(logits_b, labels, cfg_training.label_smoothing, weight=class_weights)
            ) / 2.0

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(all_params, cfg_training.gradient_clip)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        log.info(f"[{branch_name} joint] Epoch {epoch+1}/{n_epochs} loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(checkpoint_dir, f"{branch_name}_joint_best.pt")
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(branch.state_dict(), ckpt_path)

    # Restore best
    ckpt_path = os.path.join(checkpoint_dir, f"{branch_name}_joint_best.pt")
    if os.path.exists(ckpt_path):
        branch.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        log.info(f"[{branch_name} joint] Restored best joint weights")


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    log.info("Config:\n" + OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ── Data ─────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    log.info(
        f"Dataset: {len(train_loader.dataset)} train  "
        f"{len(val_loader.dataset)} val  "
        f"{len(test_loader.dataset)} test"
    )

    ckpt_dir    = cfg.paths.checkpoints
    num_classes = cfg.num_classes

    # Class-balancing weights to counter MCI dominance in ADNI
    ds = train_loader.dataset
    if hasattr(ds, "labels"):
        train_labels = [int(l) for l in ds.labels]
    elif hasattr(ds, "dataset") and hasattr(ds.dataset, "labels"):
        train_labels = [int(ds.dataset.labels[i]) for i in ds.indices]
    else:
        train_labels = [int(ds[i]["label"].item()) for i in range(len(ds))]
    counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = torch.tensor(
        len(train_labels) / (num_classes * np.maximum(counts, 1).astype(np.float32)),
        dtype=torch.float32,
        device=device,
    )
    log.info(f"Class-balanced inverse weights: {class_weights.cpu().numpy().tolist()}")

    # ── Build full pipeline ───────────────────────────────────────────────
    pipeline = AlzheimerPipeline(cfg, device=str(device))

    # ── Phase 1: sMRI Branch ─────────────────────────────────────────────
    if cfg.training.train_smri_branch:
        log.info("=" * 55)
        log.info("PHASE 1 — sMRI Branch")
        log.info("=" * 55)

        train_assistant(
            pipeline.smri_supervisor.ceah_assistant,
            input_key     = "mri",
            train_loader  = train_loader,
            val_loader    = val_loader,
            cfg_training  = cfg.training,
            device        = device,
            checkpoint_dir = ckpt_dir,
            tag           = "ceah",
            num_classes   = num_classes,
            class_weights = class_weights,
        )

        train_assistant(
            pipeline.smri_supervisor.gnn_assistant,
            input_key     = "mri",
            train_loader  = train_loader,
            val_loader    = val_loader,
            cfg_training  = cfg.training,
            device        = device,
            checkpoint_dir = ckpt_dir,
            tag           = "gnn",
            num_classes   = num_classes,
            class_weights = class_weights,
        )

        # Short joint pass
        train_branch_joint(
            pipeline.smri_supervisor,
            branch_name   = "smri",
            train_loader  = train_loader,
            val_loader    = val_loader,
            cfg_training  = cfg.training,
            device        = device,
            checkpoint_dir = ckpt_dir,
            num_classes   = num_classes,
            n_epochs      = 3,
            class_weights = class_weights,
        )

    # ── Phase 2: Features Branch ─────────────────────────────────────────
    if cfg.training.train_features_branch:
        log.info("=" * 55)
        log.info("PHASE 2 — Features Branch")
        log.info("=" * 55)

        train_assistant(
            pipeline.features_supervisor.clinical_assistant,
            input_key     = "clinical",
            train_loader  = train_loader,
            val_loader    = val_loader,
            cfg_training  = cfg.training,
            device        = device,
            checkpoint_dir = ckpt_dir,
            tag           = "clinical",
            num_classes   = num_classes,
            class_weights = class_weights,
        )

        train_assistant(
            pipeline.features_supervisor.cognitive_assistant,
            input_key     = "cognitive",
            train_loader  = train_loader,
            val_loader    = val_loader,
            cfg_training  = cfg.training,
            device        = device,
            checkpoint_dir = ckpt_dir,
            tag           = "cognitive",
            num_classes   = num_classes,
            class_weights = class_weights,
        )

        train_branch_joint(
            pipeline.features_supervisor,
            branch_name   = "features",
            train_loader  = train_loader,
            val_loader    = val_loader,
            cfg_training  = cfg.training,
            device        = device,
            checkpoint_dir = ckpt_dir,
            num_classes   = num_classes,
            n_epochs      = 3,
            class_weights = class_weights,
        )

    # ── Phase 3: Full joint fine-tune ────────────────────────────────────
    if cfg.training.get("joint_finetune", False):
        log.info("=" * 55)
        log.info("PHASE 3 — Full Joint Fine-Tune")
        log.info("=" * 55)

        all_params = list(pipeline.all_parameters())
        optimizer  = AdamW(
            all_params,
            lr           = cfg.training.joint_finetune_lr,
            weight_decay = cfg.training.weight_decay,
        )
        n_epochs = cfg.training.joint_finetune_epochs

        for epoch in range(n_epochs):
            pipeline.train_mode()
            epoch_loss = 0.0

            for batch in train_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                labels = batch["label"]

                # Sum losses from all four assistants and extract feature vectors
                logits_ceah, f_ceah = pipeline.smri_supervisor.ceah_assistant._forward(batch)
                logits_gnn,  f_gnn  = pipeline.smri_supervisor.gnn_assistant._forward(batch)
                logits_clin, f_clin = pipeline.features_supervisor.clinical_assistant._forward(batch)
                logits_cogn, f_cogn = pipeline.features_supervisor.cognitive_assistant._forward(batch)

                # Train Cross-Attention multimodal feature fusion module end-to-end!
                logits_fuse, _ = pipeline.fusion(f_ceah, f_gnn, f_clin, f_cogn)
                loss_fuse = smooth_ce_loss(logits_fuse, labels, cfg.training.label_smoothing, weight=class_weights)

                loss_assistants = (
                    smooth_ce_loss(logits_ceah, labels, cfg.training.label_smoothing, weight=class_weights) +
                    smooth_ce_loss(logits_gnn,  labels, cfg.training.label_smoothing, weight=class_weights) +
                    smooth_ce_loss(logits_clin, labels, cfg.training.label_smoothing, weight=class_weights) +
                    smooth_ce_loss(logits_cogn, labels, cfg.training.label_smoothing, weight=class_weights)
                ) / 4.0

                loss = loss_assistants + 1.2 * loss_fuse

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, cfg.training.gradient_clip)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            log.info(f"[joint] Epoch {epoch+1}/{n_epochs}  loss={avg_loss:.4f}")

    # ── Final evaluation on test set ──────────────────────────────────────
    log.info("=" * 55)
    log.info("TEST SET EVALUATION")
    log.info("=" * 55)

    test_metrics = {}
    for tag, assistant in [
        ("CE-AH",    pipeline.smri_supervisor.ceah_assistant),
        ("GNN",      pipeline.smri_supervisor.gnn_assistant),
        ("Clinical", pipeline.features_supervisor.clinical_assistant),
        ("Cognitive",pipeline.features_supervisor.cognitive_assistant),
    ]:
        assistant.eval()
        def _fwd(b, a=assistant):
            logits, _ = a._forward(b)
            return logits
        m = evaluate(_fwd, test_loader, device, num_classes)
        test_metrics[tag] = m
        log.info(
            f"[{tag}] test acc={m['accuracy']:.3f}  "
            f"f1={m['macro_f1']:.3f}  auc={m['macro_auc']:.3f}"
        )

    # Evaluate full Chief Physician Multi-Agent System (with Cross-Attention)
    pipeline.eval_mode()
    all_chief_preds, all_chief_labels, all_chief_probs = [], [], []
    all_fuse_preds, all_fuse_probs = [], []

    for batch in test_loader:
        batch = {k: (torch.nan_to_num(v.to(device)) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad():
            final_res = pipeline.predict(batch)
            probs = final_res.raw["final_probs"].cpu()
            preds = probs.argmax(dim=-1)
            all_chief_preds.append(preds)
            all_chief_labels.append(batch["label"].cpu())
            all_chief_probs.append(probs)

            if final_res.raw.get("cross_attention_probs") is not None:
                f_probs = final_res.raw["cross_attention_probs"].cpu()
                all_fuse_probs.append(f_probs)
                all_fuse_preds.append(f_probs.argmax(dim=-1))

    c_preds  = torch.cat(all_chief_preds).numpy()
    c_labels = torch.cat(all_chief_labels).numpy()
    c_probs  = torch.cat(all_chief_probs).numpy()

    c_acc = float(accuracy_score(c_labels, c_preds))
    c_f1  = float(f1_score(c_labels, c_preds, average="macro", zero_division=0))
    try:
        c_auc = float(roc_auc_score(c_labels, c_probs, multi_class="ovr", average="macro"))
    except ValueError:
        c_auc = float("nan")

    test_metrics["Chief_Physician"] = {"accuracy": c_acc, "macro_f1": c_f1, "macro_auc": c_auc}
    log.info("-" * 55)
    log.info(
        f"[CHIEF PHYSICIAN (Multi-Agent)] test acc={c_acc:.3f}  "
        f"f1={c_f1:.3f}  auc={c_auc:.3f}"
    )

    if len(all_fuse_preds) > 0:
        f_preds = torch.cat(all_fuse_preds).numpy()
        f_probs = torch.cat(all_fuse_probs).numpy()
        f_acc = float(accuracy_score(c_labels, f_preds))
        f_f1  = float(f1_score(c_labels, f_preds, average="macro", zero_division=0))
        try:
            f_auc = float(roc_auc_score(c_labels, f_probs, multi_class="ovr", average="macro"))
        except ValueError:
            f_auc = float("nan")
        test_metrics["Cross_Attention_Head"] = {"accuracy": f_acc, "macro_f1": f_f1, "macro_auc": f_auc}
        log.info(
            f"[CROSS-ATTENTION HEAD]          test acc={f_acc:.3f}  "
            f"f1={f_f1:.3f}  auc={f_auc:.3f}"
        )
    log.info("-" * 55)

    # Save metrics JSON
    import json
    os.makedirs(cfg.paths.outputs, exist_ok=True)
    metrics_path = os.path.join(cfg.paths.outputs, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    log.info(f"Test metrics saved to {metrics_path}")

    # Save final checkpoint
    pipeline.save_checkpoint(
        os.path.join(ckpt_dir, "pipeline_final.pt"),
        extra={"cfg": OmegaConf.to_container(cfg, resolve=True), "metrics": test_metrics},
    )

    log.info("Training complete.")


if __name__ == "__main__":
    main()
