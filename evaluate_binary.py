"""
evaluate_binary.py
------------------
Binary Clinical Evaluation Benchmark: Cognitively Normal (CN) vs Alzheimer's Disease (AD).

Direct apples-to-apples comparison against the CE-AH paper and binary literature baselines.
Filters the unseen ADNI test cohort for CN (0) and AD (2) subjects, renormalizing multi-agent
probabilities to compute binary Accuracy, Sensitivity, Specificity, F1-Score, and ROC-AUC.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple

import torch
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize, compose

from pipeline import AlzheimerPipeline
from data.dataset import build_dataloaders


def compute_binary_roc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute ROC curve and AUC using pure NumPy numerical integration."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1]), 0.5

    desc_indices = np.argsort(y_score)[::-1]
    y_sorted = y_true[desc_indices]
    scores_sorted = y_score[desc_indices]

    distinct_mask = np.diff(scores_sorted, append=scores_sorted[-1] - 1) != 0
    tps = np.cumsum(y_sorted)[distinct_mask]
    fps = np.cumsum(1 - y_sorted)[distinct_mask]

    tpr = np.concatenate([[0.0], tps / n_pos, [1.0]])
    fpr = np.concatenate([[0.0], fps / n_neg, [1.0]])
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))
    return fpr, tpr, auc


def generate_binary_plots(
    cm: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc: float,
    output_dir: str = "outputs",
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[Notice] matplotlib is not installed. Skipping PNG chart export.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # 1. Binary Confusion Matrix
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["CN (Healthy)", "AD (Alzheimer's)"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Diagnosis", fontsize=12, labelpad=10)
    ax.set_ylabel("True Ground Truth Label", fontsize=12, labelpad=10)
    ax.set_title("Binary Clinical Confusion Matrix (CN vs AD)\n(Direct Literature Benchmark)", fontsize=13, pad=15)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            total_true = np.sum(cm[i, :])
            pct = cm[i, j] / total_true * 100 if total_true > 0 else 0
            text_str = f"{cm[i, j]}\n({pct:.1f}%)"
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, text_str, ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "binary_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"  ✓ Exported: {cm_path}")

    # 2. Binary ROC Curve
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    ax.plot(fpr, tpr, color="#0284c7", lw=2.5, label=f"Multi-Agent ROC (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Chance (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    ax.set_title("Binary ROC Curve: CN vs AD\n(ADNI Unseen Test Cohort)", fontsize=12, pad=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    roc_path = os.path.join(output_dir, "binary_roc_curve.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"  ✓ Exported: {roc_path}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Binary Clinical Benchmark (CN vs AD)")
    parser.add_argument("--fast", action="store_true", help="Fast mode (5 MC-Dropout passes per agent)")
    parser.add_argument("--instant", action="store_true", help="Ultra-fast deterministic screening (1 pass)")
    parser.add_argument("--tta", action="store_true", help="Enable Test-Time Augmentation (TTA)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("  🧠 ADNI MULTI-AGENT AI SYSTEM — BINARY CLINICAL BENCHMARK (CN vs AD)  ")
    print("=" * 80 + "\n")

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    _, _, test_loader = build_dataloaders(cfg)
    test_ds = test_loader.dataset
    n_total = len(test_ds)

    # Filter for CN (0) and AD (2) only
    if hasattr(test_ds, "labels"):
        binary_indices = [i for i in range(n_total) if int(test_ds.labels[i]) in [0, 2]]
    else:
        binary_indices = [i for i in range(n_total) if test_ds[i]["label"].item() in [0, 2]]
    n_binary = len(binary_indices)

    print(f"📂 Extracted {n_binary} binary patients (CN vs AD) from {n_total} total test subjects.")
    pipeline = AlzheimerPipeline(cfg, device=str(device))
    pipeline.load_checkpoint("checkpoints/pipeline_final.pt")
    pipeline.eval_mode()

    mc_passes = 1 if args.instant else (5 if args.fast else 30)
    for asst in [
        pipeline.smri_supervisor.ceah_assistant,
        pipeline.smri_supervisor.gnn_assistant,
        pipeline.features_supervisor.clinical_assistant,
        pipeline.features_supervisor.cognitive_assistant,
    ]:
        asst.mc_samples = mc_passes

    print(f"⚡ Mode: {'Instant' if args.instant else ('Fast' if args.fast else 'High-Assurance')} ({mc_passes} MC passes)")
    if args.tta:
        print("🧠 Test-Time Augmentation (TTA) Active")

    print("-" * 80)
    print(f"Evaluating {n_binary} binary patients through Multi-Agent Pipeline...")

    y_true_binary = []  # 0 for CN, 1 for AD
    y_pred_binary = []
    y_scores_ad = []    # Renormalized P(AD) / (P(CN) + P(AD))
    latencies = []

    t_start_all = time.perf_counter()

    for count, idx in enumerate(binary_indices):
        patient_data = test_ds[idx]
        batch = {
            k: (torch.nan_to_num(v.unsqueeze(0).to(device)) if isinstance(v, torch.Tensor) else v)
            for k, v in patient_data.items()
        }
        raw_label = patient_data["label"].item()
        b_label = 0 if raw_label == 0 else 1  # 0: CN, 1: AD
        y_true_binary.append(b_label)

        t_0 = time.perf_counter()
        with torch.no_grad():
            res = pipeline.predict(batch, use_tta=args.tta)
        t_1 = time.perf_counter()
        latencies.append((t_1 - t_0) * 1000.0)

        probs_3class = res.raw["final_probs"].cpu().numpy().squeeze(0)
        p_cn = float(probs_3class[0])
        p_ad = float(probs_3class[2])

        # Renormalize between CN and AD (standard literature binary protocol)
        total_p = p_cn + p_ad + 1e-8
        score_ad = p_ad / total_p
        y_scores_ad.append(score_ad)

        pred_b = 1 if score_ad >= 0.50 else 0
        y_pred_binary.append(pred_b)

        if (count + 1) % 10 == 0 or (count + 1) == n_binary:
            print(f"  [{count + 1:>2}/{n_binary}] patients evaluated ({(count + 1)/n_binary*100:5.1f}%)...")

    t_end_all = time.perf_counter()
    y_true = np.array(y_true_binary)
    y_pred = np.array(y_pred_binary)
    y_scores = np.array(y_scores_ad)

    # 2x2 Confusion Matrix: [[TN, FP], [FN, TP]]
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))

    cm = np.array([[tn, fp], [fn, tp]])
    accuracy = (tp + tn) / n_binary * 100
    sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0.0  # AD Recall
    specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0.0  # CN Recall
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity) / 100 if (precision + sensitivity) > 0 else 0.0

    fpr, tpr, auc = compute_binary_roc(y_true, y_scores)

    print("\n" + "=" * 80)
    print("  📊 BINARY (CN vs AD) BENCHMARK RESULTS")
    print("=" * 80)
    print(f"   • Binary Accuracy   : {accuracy:.2f}% ({tp + tn} / {n_binary} correct)")
    print(f"   • AD Sensitivity    : {sensitivity:.2f}% ({tp} / {tp + fn} Alzheimer's detected)")
    print(f"   • CN Specificity    : {specificity:.2f}% ({tn} / {tn + fp} Normal Controls verified)")
    print(f"   • Precision (PPV)   : {precision:.2f}%")
    print(f"   • Binary F1-Score   : {f1:.4f}")
    print(f"   • Binary ROC-AUC    : {auc:.4f}")
    print(f"   • Mean Latency      : {np.mean(latencies):.1f} ms/patient")
    print(f"   • Total Time        : {t_end_all - t_start_all:.2f} seconds")

    print("\n" + "─" * 80)
    print("  🎯 2×2 BINARY CONFUSION MATRIX (CN vs AD)")
    print("─" * 80)
    print(f"                     ┌── Predicted: CN ──┬── Predicted: AD ──┐")
    print(f"  True: CN (Healthy) │     {tn:>4} ({tn/(tn+fp)*100:>5.1f}%)   │     {fp:>4} ({fp/(tn+fp)*100:>5.1f}%)   │ (Total: {tn+fp})")
    print(f"  True: AD (Disease) │     {fn:>4} ({fn/(fn+tp)*100:>5.1f}%)   │     {tp:>4} ({tp/(fn+tp)*100:>5.1f}%)   │ (Total: {fn+tp})")
    print(f"                     └───────────────────┴───────────────────┘")

    print("\n" + "─" * 80)
    print("  🏆 COMPARISON WITH PUBLISHED LITERATURE (BINARY CN vs AD)")
    print("─" * 80)
    print(f"   • CE-AH Paper Backbone Baseline : ~88.0% – 92.0% Accuracy")
    print(f"   • Standard 3D-ResNet Baseline   : ~86.5% – 90.0% Accuracy")
    print(f"   • ⭐ OUR MULTI-AGENT SYSTEM     : {accuracy:.1f}% Accuracy | AUC: {auc:.3f}")
    print("─" * 80)

    # Export plots and JSON
    generate_binary_plots(cm, fpr, tpr, auc, output_dir="outputs")

    report = {
        "task": "Binary (CN vs AD)",
        "num_patients": n_binary,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "roc_auc": auc,
        "confusion_matrix": cm.tolist(),
    }
    json_path = os.path.join("outputs", "binary_evaluation.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Exported Binary Audit: {json_path}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
