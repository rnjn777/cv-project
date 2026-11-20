"""
data/dataset.py
---------------
Dataset class for the multi-agent Alzheimer's detection system.

Supports two modes:
  1. Synthetic (default) — generates random tensors of the right shape.
     Use this to run the full pipeline without real data.
  2. Real ADNI/OASIS-3 — loads NIfTI MRI files + CSV feature tables.
     Activate by setting cfg.data.synthetic = False.

Labels: 0=CN, 1=MCI, 2=AD
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, Tuple, Optional


class AlzheimerDataset(Dataset):
    """
    Each sample contains:
      - mri    : FloatTensor [1, D, H, W]   — 3D structural MRI
      - clinical : FloatTensor [num_clinical]
      - cognitive : FloatTensor [num_cognitive]
      - label  : LongTensor scalar (0=CN, 1=MCI, 2=AD)
      - subject_id : str
    """

    CLASS_NAMES = ["CN", "MCI", "AD"]

    def __init__(
        self,
        cfg,
        split: str = "train",       # "train" | "val" | "test"
        transform=None,
    ):
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.mri_shape = tuple(cfg.data.mri_shape)          # (D, H, W)
        self.num_clinical  = cfg.data.num_clinical_features
        self.num_cognitive = cfg.data.num_cognitive_features

        if cfg.data.synthetic:
            self._init_synthetic(cfg.data.num_synthetic_samples)
        else:
            self._init_real(cfg.data.root)

    # ── Synthetic mode ────────────────────────────────────────────────────────
    def _init_synthetic(self, n: int):
        """Generate reproducible synthetic data for dry-run testing."""
        rng = np.random.default_rng(42)
        self.subject_ids = [f"SYN_{i:04d}" for i in range(n)]
        self.labels      = rng.integers(0, 3, size=n).tolist()

        # Pre-generate random tensors — saves time during __getitem__
        self._mri      = torch.randn(n, 1, *self.mri_shape) * 0.1
        self._clinical  = torch.randn(n, self.num_clinical)
        self._cognitive = torch.randn(n, self.num_cognitive)

        # Apply class-specific signal so the model has something to learn
        for i, label in enumerate(self.labels):
            # Hippocampal region (approximate voxel range in 91x109x91)
            if label == 2:  # AD — stronger atrophy signal
                self._mri[i, 0, 35:55, 45:65, 35:55] -= 0.5
            elif label == 1:  # MCI — moderate signal
                self._mri[i, 0, 35:55, 45:65, 35:55] -= 0.2

    # ── Real data mode ────────────────────────────────────────────────────────
    def _init_real(self, root: str):
        """
        Load from real ADNI/OASIS-3 data.
        Expected structure:
            root/
              metadata.csv          — columns: subject_id, label, split
              mri/<subject_id>.nii.gz
              clinical.csv          — indexed by subject_id
              cognitive.csv         — indexed by subject_id
        """
        meta = pd.read_csv(os.path.join(root, "metadata.csv"))
        meta = meta[meta["split"] == self.split].reset_index(drop=True)

        clinical_df  = pd.read_csv(os.path.join(root, "clinical.csv"),  index_col=0)
        cognitive_df = pd.read_csv(os.path.join(root, "cognitive.csv"), index_col=0)

        # ── Z-Score Feature Standardization (strictly fitted on training split) ──
        meta_all = pd.read_csv(os.path.join(root, "metadata.csv"))
        train_sids = meta_all[meta_all["split"] == "train"]["subject_id"].tolist()

        c_num = clinical_df.apply(pd.to_numeric, errors="coerce")
        c_train = c_num.reindex(c_num.index.intersection(train_sids))
        c_mean = c_train.mean()
        c_std  = c_train.std().replace(0, 1.0).fillna(1.0)
        clinical_df = ((c_num - c_mean) / c_std).fillna(0.0)

        cg_num = cognitive_df.apply(pd.to_numeric, errors="coerce")
        cg_train = cg_num.reindex(cg_num.index.intersection(train_sids))
        cg_mean = cg_train.mean()
        cg_std  = cg_train.std().replace(0, 1.0).fillna(1.0)
        cognitive_df = ((cg_num - cg_mean) / cg_std).fillna(0.0)

        mri_dir = os.path.join(root, "mri")
        require_mri = getattr(self.cfg.data, "require_mri", True)
        if require_mri and os.path.exists(mri_dir):
            has_mri = meta["subject_id"].apply(
                lambda sid: os.path.exists(os.path.join(mri_dir, f"{sid}.nii.gz"))
            )
            n_before = len(meta)
            if has_mri.sum() > 0:
                meta = meta[has_mri].reset_index(drop=True)
                print(f"[Dataset] Split '{self.split}': Filtered {n_before} -> {len(meta)} subjects with verified 3D MRI scans.")

        self.subject_ids = meta["subject_id"].tolist()
        self.labels      = meta["label"].tolist()
        self._clinical_df  = clinical_df
        self._cognitive_df = cognitive_df
        self.num_clinical  = len(clinical_df.columns)
        self.num_cognitive = len(cognitive_df.columns)
        self._mri_dir = mri_dir
        self._use_nifti = True

    # ── Core interface ────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.subject_ids)

    def __getitem__(self, idx: int) -> Dict:
        sid   = self.subject_ids[idx]
        label = self.labels[idx]

        if self.cfg.data.synthetic:
            mri      = self._mri[idx]
            clinical  = self._clinical[idx]
            cognitive = self._cognitive[idx]
        else:
            mri = self._load_nifti(sid)

            if sid in self._clinical_df.index:
                c_val = self._clinical_df.loc[sid]
                if isinstance(c_val, pd.DataFrame):
                    c_val = c_val.iloc[0]
                clinical = torch.tensor(
                    pd.to_numeric(c_val.values, errors="coerce").astype(np.float32)
                )
            else:
                clinical = torch.zeros(self.num_clinical, dtype=torch.float32)
            clinical = torch.nan_to_num(clinical, nan=0.0, posinf=0.0, neginf=0.0)

            if sid in self._cognitive_df.index:
                cg_val = self._cognitive_df.loc[sid]
                if isinstance(cg_val, pd.DataFrame):
                    cg_val = cg_val.iloc[0]
                cognitive = torch.tensor(
                    pd.to_numeric(cg_val.values, errors="coerce").astype(np.float32)
                )
            else:
                cognitive = torch.zeros(self.num_cognitive, dtype=torch.float32)
            cognitive = torch.nan_to_num(cognitive, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform is not None:
            mri = self.transform(mri)

        return {
            "mri":        mri,
            "clinical":   clinical,
            "cognitive":  cognitive,
            "label":      torch.tensor(label, dtype=torch.long),
            "subject_id": sid,
        }

    def _load_nifti(self, subject_id: str) -> torch.Tensor:
        """Load and preprocess a NIfTI MRI file."""
        import nibabel as nib
        path = os.path.join(self._mri_dir, f"{subject_id}.nii.gz")
        try:
            img  = nib.load(path)
            data = img.get_fdata(dtype=np.float32)

            # Normalize to zero mean, unit variance
            data = (data - data.mean()) / (data.std() + 1e-8)

            # Crop/pad to target shape
            data = _resize_volume(data, self.mri_shape)
            return torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
        except Exception as e:
            print(f"[Dataset] Warning: Failed to decompress/read {path} ({e}). Returning zero volume.")
            return torch.zeros(1, *self.mri_shape, dtype=torch.float32)


def _resize_volume(vol: np.ndarray, target: Tuple[int, int, int]) -> np.ndarray:
    """Crop or pad a 3D volume to the target shape."""
    result = np.zeros(target, dtype=np.float32)
    src_slices = tuple(slice(0, min(s, t)) for s, t in zip(vol.shape, target))
    dst_slices = tuple(slice(0, min(s, t)) for s, t in zip(vol.shape, target))
    result[dst_slices] = vol[src_slices]
    return result


# ── DataLoader factory ────────────────────────────────────────────────────────

def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.
    For synthetic data, we split a single dataset. For real data,
    each split is loaded separately using the metadata.csv 'split' column.
    """
    if cfg.data.synthetic:
        full = AlzheimerDataset(cfg, split="train")
        n    = len(full)
        n_test = int(n * cfg.data.test_split)
        n_val  = int(n * cfg.data.val_split)
        n_train = n - n_val - n_test

        train_ds, val_ds, test_ds = random_split(
            full, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_ds = AlzheimerDataset(cfg, split="train")
        val_ds   = AlzheimerDataset(cfg, split="val")
        test_ds  = AlzheimerDataset(cfg, split="test")

    loader_kwargs = dict(
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

