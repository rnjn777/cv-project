"""
data/generate_dataset.py
-------------------------
Generates a realistic synthetic ADNI-style dataset on disk.

Creates the exact directory structure expected by AlzheimerDataset
when cfg.data.synthetic = false:

    data/raw/
      metadata.csv            subject_id, label (0/1/2), split (train/val/test)
      clinical.csv            12 clinical features, indexed by subject_id
      cognitive.csv           8 cognitive score features, indexed by subject_id
      mri/
        <subject_id>.nii.gz   91x109x91 structural MRI NIfTI file

Class-specific signals
----------------------
The generator adds realistic, learnable class-specific patterns:

  CN  (label=0): normal brain volumes, good cognition, low clinical risk
  MCI (label=1): mild hippocampal atrophy, slightly reduced scores
  AD  (label=2): pronounced hippocampal/entorhinal atrophy, low MMSE/CDR

Usage
-----
    python data/generate_dataset.py                      # 300 subjects
    python data/generate_dataset.py --n 500              # 500 subjects
    python data/generate_dataset.py --n 100 --seed 7     # reproducible
    python data/generate_dataset.py --no-mri             # CSV only, fast
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# NIfTI support — gracefully degrade if nibabel not installed
try:
    import nibabel as nib
    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False

# ─────────────────────────────────────────────────────────────────────────────
# Class-specific distribution parameters
# (calibrated to roughly match ADNI population statistics)
# ─────────────────────────────────────────────────────────────────────────────

# Clinical feature distributions: (mean_CN, mean_MCI, mean_AD, std)
CLINICAL_PARAMS = {
    # name                   CN      MCI     AD      std
    "MMSE":               (29.0,  26.5,   18.0,   2.5),
    "CDR":                ( 0.0,   0.5,    1.5,   0.3),
    "Age":                (72.0,  75.0,   78.0,   6.0),
    "Sex":                ( 0.5,   0.5,    0.5,   0.5),   # binary, thresholded later
    "Education_years":    (15.0,  13.5,   12.0,   3.0),
    "Handedness":         ( 0.9,   0.9,    0.9,   0.1),   # ~90% right-handed
    "BMI":                (26.0,  26.5,   27.0,   4.0),
    "Systolic_BP":        (128.0, 132.0,  136.0,  15.0),
    "Family_history":     ( 0.3,   0.4,    0.5,   0.4),   # binary
    "Depression_history": ( 0.15,  0.25,   0.40,  0.4),   # binary
    "Num_medications":    ( 2.0,   4.0,    6.0,   2.0),
    "Comorbidity_index":  ( 1.0,   2.0,    3.5,   1.5),
}

# Cognitive score distributions: (mean_CN, mean_MCI, mean_AD, std)
# Higher = better (except ADAS-Cog where lower = better, handled below)
COGNITIVE_PARAMS = {
    # name                    CN      MCI     AD     std
    "Memory_composite":    ( 0.8,   0.4,  -0.6,   0.4),
    "Executive_function":  ( 0.7,   0.3,  -0.4,   0.4),
    "Language":            ( 0.6,   0.2,  -0.5,   0.4),
    "Visuospatial":        ( 0.5,   0.1,  -0.6,   0.4),
    "Attention":           ( 0.7,   0.3,  -0.3,   0.4),
    "Processing_speed":    ( 0.6,   0.2,  -0.5,   0.4),
    "ADAS_Cog_11":         ( 8.0,  18.0,  35.0,   5.0),   # lower = better
    "Digit_span":          (10.0,   8.0,   5.5,   2.0),
}

MRI_SHAPE = (91, 109, 91)  # MNI152


# ─────────────────────────────────────────────────────────────────────────────
# Feature generators
# ─────────────────────────────────────────────────────────────────────────────

def _sample_clinical(label: int, rng: np.random.Generator) -> dict:
    """Generate one subject's clinical features based on their class label."""
    row = {}
    means = {k: v[label] for k, v in CLINICAL_PARAMS.items()}
    stds  = {k: v[3]     for k, v in CLINICAL_PARAMS.items()}

    for feat, mean in means.items():
        val = rng.normal(mean, stds[feat])
        # Clip/binarise specific features
        if feat == "MMSE":
            val = float(np.clip(val, 0, 30))
        elif feat == "CDR":
            val = float(np.clip(np.round(val * 2) / 2, 0, 3))
        elif feat in ("Sex", "Family_history", "Depression_history", "Handedness"):
            val = float(1 if val > 0.5 else 0)
        elif feat == "Age":
            val = float(np.clip(val, 50, 95))
        elif feat == "Num_medications":
            val = float(max(0, round(val)))
        else:
            val = float(val)
        row[feat] = val
    return row


def _sample_cognitive(label: int, rng: np.random.Generator) -> dict:
    """Generate one subject's cognitive scores based on their class label."""
    row = {}
    means = {k: v[label] for k, v in COGNITIVE_PARAMS.items()}
    stds  = {k: v[3]     for k, v in COGNITIVE_PARAMS.items()}

    for feat, mean in means.items():
        val = rng.normal(mean, stds[feat])
        if feat == "ADAS_Cog_11":
            val = float(np.clip(val, 0, 70))
        elif feat == "Digit_span":
            val = float(np.clip(val, 0, 20))
        else:
            val = float(val)
        row[feat] = val
    return row


def _generate_mri(label: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate a synthetic 91x109x91 brain MRI volume.

    The volume is mostly Gaussian noise (simulating grey/white matter)
    with class-specific atrophy signals in anatomically relevant regions:

      Hippocampus (bilateral): voxels [35:55, 45:65, 35:55]
      Entorhinal cortex:       voxels [38:48, 48:60, 40:52]
      Ventricles:              voxels [38:53, 52:70, 40:51]

    CN:  no atrophy, normal ventricles
    MCI: mild hippocampal atrophy (~10% intensity drop)
    AD:  strong hippocampal + entorhinal atrophy, enlarged ventricles
    """
    D, H, W = MRI_SHAPE

    # Base brain tissue: slightly positive background
    vol = rng.normal(0.5, 0.15, size=(D, H, W)).astype(np.float32)

    # Brain mask: rough ellipsoidal skull-strip
    zz, yy, xx = np.mgrid[0:D, 0:H, 0:W]
    ellipse = (
        ((zz - D/2) / (D/2 * 0.85))**2 +
        ((yy - H/2) / (H/2 * 0.90))**2 +
        ((xx - W/2) / (W/2 * 0.85))**2
    )
    outside_brain = ellipse > 1.0
    vol[outside_brain] = 0.0

    # Hippocampal region (bilateral)
    hippo_L = np.s_[35:52, 45:62, 28:45]
    hippo_R = np.s_[35:52, 45:62, 46:63]

    # Entorhinal cortex (near hippocampus)
    entorh  = np.s_[38:50, 48:62, 32:58]

    # Ventricles (central)
    ventr   = np.s_[38:53, 52:70, 38:53]

    if label == 0:     # CN — healthy
        # Hippocampus has higher density (normal grey matter)
        vol[hippo_L] += rng.normal(0.15, 0.05, vol[hippo_L].shape)
        vol[hippo_R] += rng.normal(0.15, 0.05, vol[hippo_R].shape)

    elif label == 1:   # MCI — mild atrophy
        # Slightly reduced hippocampal density
        vol[hippo_L] += rng.normal(0.05, 0.06, vol[hippo_L].shape)
        vol[hippo_R] += rng.normal(0.05, 0.06, vol[hippo_R].shape)
        # Mild entorhinal thinning
        vol[entorh]  -= rng.normal(0.05, 0.03, vol[entorh].shape)

    else:              # AD — pronounced atrophy
        # Hippocampal atrophy (intensity drop = volume loss proxy)
        vol[hippo_L] -= rng.normal(0.25, 0.08, vol[hippo_L].shape)
        vol[hippo_R] -= rng.normal(0.25, 0.08, vol[hippo_R].shape)
        # Entorhinal atrophy
        vol[entorh]  -= rng.normal(0.20, 0.07, vol[entorh].shape)
        # Ventricle expansion (CSF signal = near-zero in sMRI)
        vol[ventr]   -= rng.normal(0.30, 0.10, vol[ventr].shape)

    # Add per-subject anatomical variance (makes each scan unique)
    noise = rng.normal(0.0, 0.03, size=vol.shape).astype(np.float32)
    vol   = np.clip(vol + noise, 0.0, 1.5)

    # Normalise to zero mean, unit std (same as _load_nifti in dataset.py)
    vol = (vol - vol.mean()) / (vol.std() + 1e-8)

    return vol.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    output_dir: str,
    n_subjects: int = 300,
    seed: int       = 42,
    generate_mri: bool = True,
    class_balance: tuple = (0.4, 0.3, 0.3),   # CN, MCI, AD proportions
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    # test = 1 - train - val
):
    """
    Generate a complete synthetic ADNI-like dataset.

    Parameters
    ----------
    output_dir   : where to write data/raw/
    n_subjects   : total number of synthetic subjects
    seed         : random seed for reproducibility
    generate_mri : if False, skip NIfTI generation (much faster)
    class_balance: fraction of subjects per class (CN, MCI, AD)
    train_frac   : fraction of subjects for training split
    val_frac     : fraction for validation split
    """
    rng = np.random.default_rng(seed)

    out = Path(output_dir)
    mri_dir = out / "mri"
    if generate_mri:
        mri_dir.mkdir(parents=True, exist_ok=True)
    else:
        out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CV Lab Dataset Generator")
    print(f"{'='*60}")
    print(f"  Output dir : {out.resolve()}")
    print(f"  N subjects : {n_subjects}")
    print(f"  Seed       : {seed}")
    print(f"  MRI files  : {'yes' if generate_mri else 'no (--no-mri flag)'}")
    print(f"  Classes    : CN={class_balance[0]:.0%}  MCI={class_balance[1]:.0%}  AD={class_balance[2]:.0%}")
    print(f"{'='*60}\n")

    # ── Assign labels ──────────────────────────────────────────────────────
    fracs  = np.array(class_balance)
    fracs /= fracs.sum()
    counts = (fracs * n_subjects).astype(int)
    # Adjust rounding error
    counts[-1] = n_subjects - counts[:-1].sum()

    labels     = np.repeat([0, 1, 2], counts)
    subject_ids = [f"ADNI_{i:04d}" for i in range(n_subjects)]

    # ── Train / val / test split ───────────────────────────────────────────
    indices  = rng.permutation(n_subjects)
    n_train  = int(n_subjects * train_frac)
    n_val    = int(n_subjects * val_frac)
    n_test   = n_subjects - n_train - n_val

    split_arr = np.empty(n_subjects, dtype=object)
    split_arr[indices[:n_train]]             = "train"
    split_arr[indices[n_train:n_train+n_val]] = "val"
    split_arr[indices[n_train+n_val:]]        = "test"

    # ── Generate tabular features ──────────────────────────────────────────
    print(">> Generating clinical and cognitive features...")
    clinical_rows  = []
    cognitive_rows = []

    for sid, lab in zip(subject_ids, labels):
        clin = _sample_clinical(int(lab), rng)
        cogn = _sample_cognitive(int(lab), rng)
        clinical_rows.append({"subject_id": sid, **clin})
        cognitive_rows.append({"subject_id": sid, **cogn})

    clinical_df  = pd.DataFrame(clinical_rows).set_index("subject_id")
    cognitive_df = pd.DataFrame(cognitive_rows).set_index("subject_id")
    meta_df      = pd.DataFrame({
        "subject_id": subject_ids,
        "label":      labels,
        "split":      split_arr,
    })

    clinical_df.to_csv(out / "clinical.csv")
    cognitive_df.to_csv(out / "cognitive.csv")
    meta_df.to_csv(out / "metadata.csv", index=False)

    print(f"   Saved: clinical.csv ({len(clinical_df)} rows)")
    print(f"   Saved: cognitive.csv ({len(cognitive_df)} rows)")
    print(f"   Saved: metadata.csv  ({n_train} train / {n_val} val / {n_test} test)")

    # ── Generate MRI files ─────────────────────────────────────────────────
    if generate_mri:
        if not HAS_NIBABEL:
            print("\n[WARNING] nibabel not installed. Saving MRI as numpy .npy instead.")
            print("          Install with: pip install nibabel")
            _generate_mri_numpy(subject_ids, labels, mri_dir, rng)
        else:
            _generate_mri_nifti(subject_ids, labels, mri_dir, rng, n_subjects)
    else:
        print("\n>> Skipping MRI generation (--no-mri)")
        print("   The pipeline will use cfg.data.synthetic=true for MRI")
        print("   (tabular features will be loaded from the CSVs above)")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Dataset generation complete!")
    print(f"  Location : {out.resolve()}")
    print(f"\n  Class distribution:")
    for cls_name, cnt in zip(["CN", "MCI", "AD"], counts):
        print(f"    {cls_name}: {cnt} subjects")
    print(f"\n  To use this dataset, set in configs/config.yaml:")
    print(f"    data:")
    print(f"      synthetic: false")
    print(f"      root: {out.resolve()}")
    print(f"{'='*60}\n")


def _generate_mri_nifti(subject_ids, labels, mri_dir, rng, n_total):
    """Generate NIfTI .nii.gz files for each subject."""
    print(f"\n>> Generating {n_total} MRI NIfTI files — this may take a few minutes...")
    # Simple affine: 1mm isotropic voxels, centred at origin
    affine = np.diag([1.0, 1.0, 1.0, 1.0])

    t0 = time.time()
    for i, (sid, lab) in enumerate(zip(subject_ids, labels)):
        vol  = _generate_mri(int(lab), rng)           # [91, 109, 91]
        img  = nib.Nifti1Image(vol, affine)
        path = mri_dir / f"{sid}.nii.gz"
        nib.save(img, str(path))

        if (i + 1) % 50 == 0 or (i + 1) == n_total:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            remain  = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"   {i+1:>4}/{n_total}  ({elapsed:.0f}s elapsed, "
                  f"~{remain:.0f}s remaining)")

    print(f"   Saved {n_total} NIfTI files to {mri_dir}/")


def _generate_mri_numpy(subject_ids, labels, mri_dir, rng):
    """Fallback: save MRI as .npy if nibabel is unavailable."""
    n = len(subject_ids)
    print(f"\n>> Generating {n} MRI numpy files...")
    for i, (sid, lab) in enumerate(zip(subject_ids, labels)):
        vol  = _generate_mri(int(lab), rng)
        path = mri_dir / f"{sid}.npy"
        np.save(str(path), vol)
        if (i + 1) % 50 == 0 or (i + 1) == n:
            print(f"   {i+1:>4}/{n}")
    print(f"   Saved {n} .npy files to {mri_dir}/")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic ADNI-style dataset for the CV Lab pipeline."
    )
    parser.add_argument("--n",       type=int,   default=300,
                        help="Number of subjects to generate (default: 300)")
    parser.add_argument("--seed",    type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out",     type=str,   default="./data/raw",
                        help="Output directory (default: ./data/raw)")
    parser.add_argument("--no-mri", dest="no_mri", action="store_true",
                        help="Skip MRI NIfTI generation (tabular CSVs only, much faster)")
    parser.add_argument("--cn",     type=float,  default=0.40,
                        help="Fraction of CN subjects (default: 0.40)")
    parser.add_argument("--mci",    type=float,  default=0.30,
                        help="Fraction of MCI subjects (default: 0.30)")
    parser.add_argument("--ad",     type=float,  default=0.30,
                        help="Fraction of AD subjects (default: 0.30)")
    args = parser.parse_args()

    generate_dataset(
        output_dir    = args.out,
        n_subjects    = args.n,
        seed          = args.seed,
        generate_mri  = not args.no_mri,
        class_balance = (args.cn, args.mci, args.ad),
    )
