from __future__ import annotations

# Windows cp1252 fix: force UTF-8 output
import os as _os, sys as _sys
_os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(_sys.stdout, "reconfigure"):
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

"""
smoke_test.py
-------------
End-to-end sanity check for the full multi-agent pipeline.

Runs the complete stack with a tiny synthetic batch (B=2) and verifies:
  - All models build without errors
  - All tensor shapes are correct at every stage
  - The FinalResult has the right keys and sensible values
  - No NaNs or Infs in any tensor

NO real data, GPU, or trained weights are needed.

Usage
-----
    python smoke_test.py
"""

import sys
import torch
from omegaconf import OmegaConf, DictConfig


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ok(msg: str):
    print(f"  \033[92m✓\033[0m  {msg}")

def _fail(msg: str):
    print(f"  \033[91m✗\033[0m  {msg}")
    sys.exit(1)

def assert_shape(tensor: torch.Tensor, expected: tuple, label: str):
    if tuple(tensor.shape) != expected:
        _fail(f"{label}: expected shape {expected}, got {tuple(tensor.shape)}")
    _ok(f"{label} shape {tuple(tensor.shape)}")

def assert_no_nan(tensor: torch.Tensor, label: str):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        _fail(f"{label}: contains NaN or Inf")
    _ok(f"{label} is finite")

def assert_sums_to_one(probs: dict, label: str):
    total = sum(probs.values())
    if not (0.99 < total < 1.01):
        _fail(f"{label}: probabilities sum to {total:.4f} (expected ~1.0)")
    _ok(f"{label} sums to {total:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Config builder — compose without @hydra.main
# ─────────────────────────────────────────────────────────────────────────────

def build_test_cfg() -> DictConfig:
    """
    Assemble a minimal valid DictConfig without Hydra's runtime.
    We inline just enough to initialise all models.
    """
    cfg = OmegaConf.create({
        "num_classes": 3,
        "class_names": ["CN", "MCI", "AD"],
        "seed": 42,
        "data": {
            "synthetic":              True,
            "num_synthetic_samples":  20,
            "mri_shape":              [91, 109, 91],
            "num_clinical_features":  11,
            "num_cognitive_features": 10,
            "batch_size":             2,
            "num_workers":            0,
            "val_split":              0.15,
            "test_split":             0.15,
            "root":                   "./data/raw",
        },
        "model": {
            "ceah": {
                "patch_size":        8,
                "patch_embed_dim":   128,
                "num_heads_patch":   4,
                "num_heads_region":  4,
                "num_heads_global":  8,
                "num_regions":       90,
                "region_embed_dim":  256,
                "global_embed_dim":  512,
                "dropout":           0.3,
                "depth_patch":       2,
                "depth_region":      2,
                "depth_global":      2,
                "mc_dropout_samples": 5,  # small for speed in smoke test
            },
            "gnn": {
                "num_regions":       90,
                "node_feature_dim":  64,
                "gnn_hidden_dim":    128,
                "gnn_out_dim":       256,
                "num_gnn_layers":    3,
                "gnn_type":          "GAT",
                "gat_heads":         4,
                "dropout":           0.3,
                "mc_dropout_samples": 5,
            },
            "clinical": {
                "input_dim":         11,
                "hidden_dims":       [64, 128, 64],
                "dropout":           0.3,
                "mc_dropout_samples": 5,
            },
            "cognitive": {
                "input_dim":         10,
                "hidden_dims":       [64, 128, 64],
                "dropout":           0.3,
                "mc_dropout_samples": 5,
            },
        },
        "paths": {
            "checkpoints": "./checkpoints",
            "logs":        "./logs",
            "outputs":     "./outputs",
        },
        "training": {
            "lr":                     1e-4,
            "weight_decay":           1e-4,
            "epochs":                 2,
            "patience":               5,
            "scheduler":              "cosine",
            "warmup_epochs":          1,
            "gradient_clip":          1.0,
            "label_smoothing":        0.1,
            "train_smri_branch":      True,
            "train_features_branch":  True,
            "joint_finetune":         False,
            "joint_finetune_epochs":  5,
            "joint_finetune_lr":      1e-5,
            "use_wandb":              False,
            "log_every_n_steps":      10,
            "save_top_k":             1,
        },
    })
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main smoke test
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CV Lab -- Multi-Agent Pipeline Smoke Test")
    print("=" * 60 + "\n")

    B   = 2   # batch size
    C   = 3   # num_classes
    D, H, W = 91, 109, 91

    # -- 1. Build config --------------------------------------------------
    print("\n>> Building config...")
    cfg = build_test_cfg()
    _ok("Config assembled")

    # -- 2. Build pipeline ------------------------------------------------
    print("\n>> Building pipeline...")
    from pipeline import AlzheimerPipeline
    pipeline = AlzheimerPipeline(cfg, device="cpu")
    _ok("AlzheimerPipeline instantiated")
    print(pipeline)

    # -- 3. Build synthetic batch -----------------------------------------
    print("\n>> Building synthetic batch (B=2)...")
    batch = {
        "mri":        torch.randn(B, 1, D, H, W),
        "clinical":   torch.randn(B, cfg.data.num_clinical_features),
        "cognitive":  torch.randn(B, cfg.data.num_cognitive_features),
        "label":      torch.randint(0, C, (B,)),
        "subject_id": ["SYN_0000", "SYN_0001"],
    }
    _ok(f"Batch created: mri={tuple(batch['mri'].shape)}  "
        f"clinical={tuple(batch['clinical'].shape)}  "
        f"cognitive={tuple(batch['cognitive'].shape)}")

    # -- 4. Run sMRI branch -----------------------------------------------
    print("\n>> Running sMRI branch...")
    smri_result = pipeline.smri_supervisor.run(batch)
    _ok(f"sMRI branch ran: branch_name='{smri_result.branch_name}'")
    assert_shape(smri_result.class_probs, (B, C), "smri_result.class_probs")
    assert_no_nan(smri_result.class_probs, "smri_result.class_probs")
    _ok(f"sMRI intra_agreement = {smri_result.intra_agreement:.4f}")
    _ok(f"sMRI confidence      = {smri_result.confidence:.4f}")
    _ok(f"sMRI weights         = {smri_result.weights}")
    _ok(f"sMRI minority reports: {len(smri_result.minority_reports)}")

    # Check CE-AH and GNN feature vectors
    ceah_out = smri_result.assistant_outputs[0]
    gnn_out  = smri_result.assistant_outputs[1]
    assert_shape(ceah_out.feature_vector, (B, 512), "CE-AH feature_vector")
    assert_shape(gnn_out.feature_vector,  (B, 256), "GNN feature_vector")

    # -- 5. Run Features branch -------------------------------------------
    print("\n>> Running Features branch...")
    features_result = pipeline.features_supervisor.run(batch)
    _ok(f"Features branch ran: branch_name='{features_result.branch_name}'")
    assert_shape(features_result.class_probs, (B, C), "features_result.class_probs")
    assert_no_nan(features_result.class_probs, "features_result.class_probs")
    _ok(f"Features intra_agreement = {features_result.intra_agreement:.4f}")
    _ok(f"Features confidence      = {features_result.confidence:.4f}")

    # -- 6. Run Chief Physician -------------------------------------------
    print("\n>> Running Chief Physician fusion...")
    from agents.arbitration.chief_physician import ChiefPhysician
    chief = ChiefPhysician()
    final = chief.fuse(smri_result, features_result)
    _ok(f"FinalResult: diagnosis='{final.diagnosis}'")
    assert final.diagnosis in {"CN", "MCI", "AD"}, f"Bad diagnosis: {final.diagnosis}"
    _ok(f"Diagnosis is valid: '{final.diagnosis}'")
    assert_sums_to_one(final.probabilities, "final.probabilities")
    assert 0.0 <= final.confidence <= 1.0, f"Confidence out of range: {final.confidence}"
    _ok(f"Confidence in [0,1]: {final.confidence:.4f}")
    assert set(final.modality_weights.keys()) == {"sMRI", "Features"}
    _ok(f"Modality weights: sMRI={final.modality_weights['sMRI']:.3f}  "
        f"Features={final.modality_weights['Features']:.3f}")
    _ok(f"Disagreement flag: {final.disagreement_flag}")
    assert isinstance(final.disagreement_flag, bool)
    _ok(f"Top brain regions: {final.explanation.get('top_brain_regions', [])[:3]}")
    _ok(f"Top clinical factors: {final.explanation.get('top_clinical_factors', [])}")

    # -- 7. Full pipeline.predict() ---------------------------------------
    print("\n>> Running full pipeline.predict()...")
    result = pipeline.predict(batch)
    _ok(f"pipeline.predict() succeeded")
    assert result.diagnosis in {"CN", "MCI", "AD"}
    _ok(f"End-to-end diagnosis: '{result.diagnosis}'")

    print("\n>> Checking final_probs shape...")
    final_probs = result.raw["final_probs"]
    assert_shape(final_probs, (B, C), "final_probs [B, C]")
    assert_no_nan(final_probs, "final_probs")

    # -- 9. __str__ output ------------------------------------------------
    print("\n>> FinalResult pretty-print:")
    print(result)

    # ── 10. Parameter count ──────────────────────────────────────────────
    print("\n>> Parameter counts:")
    n_smri  = sum(p.numel() for p in pipeline.smri_parameters())
    n_feat  = sum(p.numel() for p in pipeline.features_parameters())
    n_total = n_smri + n_feat
    _ok(f"sMRI branch:     {n_smri:>12,} parameters")
    _ok(f"Features branch: {n_feat:>12,} parameters")
    _ok(f"Total:           {n_total:>12,} parameters")

    # ── Done ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ALL CHECKS PASSED -- pipeline is healthy!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
