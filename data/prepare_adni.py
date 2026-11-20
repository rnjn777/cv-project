"""
data/prepare_adni.py
--------------------
Converts the ADNIMERGE2 R package data (.rda files) into the three CSV
files our pipeline needs:

    data/raw/
        metadata.csv   -- subject_id, label (0=CN, 1=MCI, 2=AD), split
        clinical.csv   -- 12 clinical features, indexed by subject_id
        cognitive.csv  -- 8 cognitive features, indexed by subject_id

Only baseline visits (VISCODE == 'bl') are used to avoid subject leakage
across train/val/test splits.

Usage
-----
    pip install pyreadr
    python data/prepare_adni.py
    python data/prepare_adni.py --adni-dir dataset/ADNIMERGE2 --out data/raw
    python data/prepare_adni.py --val 0.15 --test 0.15 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import pyreadr
except ImportError:
    print("\nERROR: pyreadr is not installed.")
    print("Install it by running this in your terminal:")
    print("    pip install pyreadr\n")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load a single .rda file
# ─────────────────────────────────────────────────────────────────────────────

def load_rda(adni_dir: Path, name: str) -> pd.DataFrame:
    """Load one .rda file and return its DataFrame."""
    path = adni_dir / "data" / f"{name}.rda"
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    result = pyreadr.read_r(str(path))
    # pyreadr returns a dict; .rda files usually have one object
    df = list(result.values())[0]
    return df


def show_cols(df: pd.DataFrame, name: str, head: int = 3):
    print(f"\n  {name}: {df.shape[0]} rows x {df.shape[1]} cols")
    print(f"  Columns: {list(df.columns)[:20]}")
    if head:
        print(df.head(head).to_string(max_cols=10))


# ─────────────────────────────────────────────────────────────────────────────
# Core prep
# ─────────────────────────────────────────────────────────────────────────────

def prepare_adni(
    adni_dir: str  = "dataset/ADNIMERGE2",
    out_dir:  str  = "data/raw",
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed:     int   = 42,
    verbose:  bool  = True,
):
    adni  = Path(adni_dir)
    out   = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng   = np.random.default_rng(seed)

    print(f"\n{'='*60}")
    print("  ADNI Dataset Preparation")
    print(f"{'='*60}")
    print(f"  Source : {adni.resolve()}")
    print(f"  Output : {out.resolve()}")

    # ── 1. Load core tables ───────────────────────────────────────────────
    print("\n>> Loading .rda tables...")

    dxsum   = load_rda(adni, "DXSUM")       # diagnosis per visit
    ptdemog = load_rda(adni, "PTDEMOG")     # demographics
    mmse    = load_rda(adni, "MMSE")        # MMSE scores
    cdr     = load_rda(adni, "CDR")         # CDR scores
    neurobat = load_rda(adni, "NEUROBAT")   # neuropsychological battery
    vitals  = load_rda(adni, "VITALS")      # blood pressure, weight
    medhist = load_rda(adni, "MEDHIST")     # medical history
    gdscale = load_rda(adni, "GDSCALE")     # geriatric depression scale

    if verbose:
        for name, df in [("DXSUM", dxsum), ("PTDEMOG", ptdemog),
                         ("MMSE", mmse), ("CDR", cdr), ("NEUROBAT", neurobat),
                         ("VITALS", vitals)]:
            show_cols(df, name, head=0)

    # ── 2. Standardise column names to uppercase ──────────────────────────
    for df in [dxsum, ptdemog, mmse, cdr, neurobat, vitals, medhist, gdscale]:
        df.columns = [c.upper() for c in df.columns]

    # ── 3. Filter to baseline visits only ─────────────────────────────────
    # VISCODE = 'bl' is baseline. Some tables use 'sc' (screening) — skip.
    print("\n>> Filtering to baseline visits (VISCODE='bl')...")

    def baseline(df):
        if "VISCODE" in df.columns:
            return df[df["VISCODE"].str.lower().isin(["bl"])].copy()
        elif "VISCODE2" in df.columns:
            return df[df["VISCODE2"].str.lower().isin(["bl"])].copy()
        return df  # return as-is if no visit code column

    dxsum_bl    = baseline(dxsum)
    mmse_bl     = baseline(mmse)
    cdr_bl      = baseline(cdr)
    neurobat_bl = baseline(neurobat)
    vitals_bl   = baseline(vitals)
    medhist_bl  = baseline(medhist)
    gdscale_bl  = baseline(gdscale)

    # ── 4. Build diagnosis labels ─────────────────────────────────────────
    # DXSUM has a DIAGNOSIS column or DXCHANGE column
    print("\n>> Building diagnosis labels...")

    # Try the most common column names across ADNI phases
    dx_col = None
    for col in ["DIAGNOSIS", "DXCHANGE", "DXCURREN", "DX"]:
        if col in dxsum_bl.columns:
            dx_col = col
            break

    if dx_col is None:
        print(f"  Available DXSUM columns: {list(dxsum_bl.columns)}")
        raise ValueError("Could not find a diagnosis column in DXSUM. "
                         "Check the columns above and update dx_col manually.")

    print(f"  Using diagnosis column: '{dx_col}'")
    print(f"  Unique values: {sorted(dxsum_bl[dx_col].dropna().unique())}")

    # Map to 0=CN, 1=MCI, 2=AD
    # ADNI uses various coding schemes — handle both string and numeric
    def map_diagnosis(val):
        if pd.isna(val):
            return None
        val = str(val).strip().upper()
        # String labels
        if val in ("CN", "NL", "NORMAL", "1"):
            return 0
        if val in ("MCI", "LMCI", "EMCI", "2", "3"):
            return 1
        if val in ("AD", "DEMENTIA", "4", "5", "6"):
            return 2
        return None

    dxsum_bl = dxsum_bl.copy()
    dxsum_bl["LABEL"] = dxsum_bl[dx_col].apply(map_diagnosis)
    dxsum_bl = dxsum_bl.dropna(subset=["LABEL"])
    dxsum_bl["LABEL"] = dxsum_bl["LABEL"].astype(int)

    # Keep subject ID + label
    dx = dxsum_bl[["PTID", "LABEL"]].drop_duplicates(subset="PTID")
    print(f"  Subjects with valid diagnosis: {len(dx)}")
    print(f"  CN={( dx.LABEL==0).sum()}  MCI={(dx.LABEL==1).sum()}  AD={(dx.LABEL==2).sum()}")

    subjects = set(dx["PTID"].tolist())

    # ── 5. Build clinical.csv ─────────────────────────────────────────────
    print("\n>> Building clinical features...")

    # Demographics
    demog = ptdemog[["PTID", "PTGENDER", "PTEDUC", "PTETHCAT",
                     "PTRACCAT", "PTHAND"]].copy() if "PTEDUC" in ptdemog.columns else ptdemog[["PTID"]].copy()

    # MMSE
    mmse_s = mmse_bl[["PTID", "MMSCORE"]].copy() if "MMSCORE" in mmse_bl.columns else \
             mmse_bl[["PTID", "TOTALMOD"]].rename(columns={"TOTALMOD": "MMSCORE"}).copy() \
             if "TOTALMOD" in mmse_bl.columns else mmse_bl[["PTID"]].copy()

    # CDR
    cdr_cols = [c for c in ["CDGLOBAL", "CDSOB", "CDMEMORY"] if c in cdr_bl.columns]
    cdr_s = cdr_bl[["PTID"] + cdr_cols].copy()

    # Vitals
    vital_cols = [c for c in ["VSWEIGHT", "VSWTUNIT", "VSBPSYS", "VSBPDIA",
                               "VSPULSE", "VSRESP", "VSTEMP"] if c in vitals_bl.columns]
    vitals_s = vitals_bl[["PTID"] + vital_cols].copy() if vital_cols else vitals_bl[["PTID"]].copy()

    # GDS (depression)
    gds_col = [c for c in ["GDTOTAL", "GDSCORE"] if c in gdscale_bl.columns]
    gds_s = gdscale_bl[["PTID"] + gds_col].copy() if gds_col else gdscale_bl[["PTID"]].copy()

    # Merge clinical tables
    clinical = dx[["PTID"]].copy()
    for df in [demog, mmse_s, cdr_s, vitals_s, gds_s]:
        if len(df.columns) > 1:  # has something beyond PTID
            clinical = clinical.merge(df, on="PTID", how="left")

    # Keep only subjects with a valid diagnosis
    clinical = clinical[clinical["PTID"].isin(subjects)]

    # Rename to our expected column names where possible
    rename_map = {
        "MMSCORE":    "MMSE",
        "CDGLOBAL":   "CDR",
        "VSBPSYS":    "Systolic_BP",
        "VSBPDIA":    "Diastolic_BP",
        "VSWEIGHT":   "Weight_kg",
        "PTGENDER":   "Sex",
        "PTEDUC":     "Education_years",
        "PTHAND":     "Handedness",
        "GDTOTAL":    "GDS_Depression",
        "GDSCORE":    "GDS_Depression",
        "CDSOB":      "CDR_SumOfBoxes",
    }
    clinical = clinical.rename(columns={k: v for k, v in rename_map.items() if k in clinical.columns})

    # Add Age from demographics if available
    if "PTDOB" in ptdemog.columns and "EXAMDATE" in dxsum_bl.columns:
        pass  # complex date arithmetic — skip for now, add as NaN
    elif "AGE" in ptdemog.columns:
        age_df = ptdemog[["PTID", "AGE"]].drop_duplicates("PTID")
        clinical = clinical.merge(age_df, on="PTID", how="left")

    # Fill missing values with column median (safe default)
    num_cols = clinical.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        clinical[col] = clinical[col].fillna(clinical[col].median())

    clinical = clinical.set_index("PTID")
    print(f"  Clinical features: {list(clinical.columns)}")
    print(f"  Subjects: {len(clinical)}")

    # ── 6. Build cognitive.csv ────────────────────────────────────────────
    print("\n>> Building cognitive features...")

    cog_cols_wanted = {
        # NEUROBAT column      -> our name
        "LIMMTOTAL":  "Memory_Immediate",        # RAVLT immediate
        "LDELTOTAL":  "Memory_Delayed",           # RAVLT delayed
        "TRABSCOR":   "TrailsB_Time",             # Trails B
        "TRAASCOR":   "TrailsA_Time",             # Trails A
        "DIGITSCOR":  "Digit_Span_Forward",
        "DSPANBAC":   "Digit_Span_Backward",
        "AVDEL30MIN": "RAVLT_Forgetting",
        "AVTOT6":     "RAVLT_Learning",
        "CATANIMSC":  "Category_Fluency",
        "BNTTOTAL":   "Boston_Naming",
    }

    avail = {k: v for k, v in cog_cols_wanted.items() if k in neurobat_bl.columns}
    if not avail:
        print(f"  NOTE: Expected columns not found in NEUROBAT. Available: {list(neurobat_bl.columns)[:20]}")
        # Fall back: take first numeric columns
        num_nb = neurobat_bl.select_dtypes(include=[np.number]).columns.tolist()[:8]
        cog = neurobat_bl[["PTID"] + num_nb].copy()
    else:
        cog = neurobat_bl[["PTID"] + list(avail.keys())].copy()
        cog = cog.rename(columns=avail)

    cog = cog[cog["PTID"].isin(subjects)]
    cog = cog.drop_duplicates(subset="PTID")

    # Fill missing
    num_cols = cog.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        cog[col] = cog[col].fillna(cog[col].median())

    cog = cog.set_index("PTID")
    print(f"  Cognitive features: {list(cog.columns)}")
    print(f"  Subjects: {len(cog)}")

    # ── 7. Align subjects across all three tables ─────────────────────────
    print("\n>> Aligning subjects across tables...")

    # Only keep subjects present in ALL three tables
    common = set(dx["PTID"]) & set(clinical.index) & set(cog.index)
    print(f"  Subjects in all tables: {len(common)}")

    dx       = dx[dx["PTID"].isin(common)].set_index("PTID")
    clinical = clinical[clinical.index.isin(common)]
    cog      = cog[cog.index.isin(common)]

    # ── 8. Train / val / test split ───────────────────────────────────────
    print("\n>> Creating train/val/test splits...")

    all_ids = list(common)
    n       = len(all_ids)
    perm    = rng.permutation(n)
    n_test  = int(n * test_frac)
    n_val   = int(n * val_frac)
    n_train = n - n_val - n_test

    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    split_map = {all_ids[perm[i]]: splits[i] for i in range(n)}

    # ── 9. Build metadata.csv ─────────────────────────────────────────────
    metadata = pd.DataFrame({
        "subject_id": list(common),
        "label":      [int(dx.loc[sid, "LABEL"]) for sid in common],
        "split":      [split_map[sid] for sid in common],
    })

    print(f"\n  Final dataset: {len(metadata)} subjects")
    print(f"  CN={(metadata.label==0).sum()}  MCI={(metadata.label==1).sum()}  AD={(metadata.label==2).sum()}")
    print(f"  Train={( metadata.split=='train').sum()}  Val={(metadata.split=='val').sum()}  Test={(metadata.split=='test').sum()}")

    # ── 10. Save ──────────────────────────────────────────────────────────
    metadata.to_csv(out / "metadata.csv", index=False)
    clinical.to_csv(out / "clinical.csv")
    cog.to_csv(out / "cognitive.csv")

    print(f"\n  Saved:")
    print(f"    {out/'metadata.csv'}")
    print(f"    {out/'clinical.csv'}   ({len(clinical.columns)} features)")
    print(f"    {out/'cognitive.csv'}  ({len(cog.columns)} features)")

    # ── 11. Update configs/config.yaml hint ──────────────────────────────
    n_clinical  = len(clinical.columns)
    n_cognitive = len(cog.columns)
    print(f"\n>> In configs/config.yaml, update:")
    print(f"     data:")
    print(f"       synthetic: false")
    print(f"       root: {out.resolve()}")
    print(f"       num_clinical_features:  {n_clinical}")
    print(f"       num_cognitive_features: {n_cognitive}")
    print(f"\n   And in configs/model/clinical/clinical.yaml:")
    print(f"       input_dim: {n_clinical}")
    print(f"\n   And in configs/model/cognitive/cognitive.yaml:")
    print(f"       input_dim: {n_cognitive}")

    print(f"\n{'='*60}")
    print("  ADNI preparation complete!")
    print(f"{'='*60}\n")

    return metadata, clinical, cog


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert ADNIMERGE2 .rda files to pipeline-ready CSVs"
    )
    parser.add_argument("--adni-dir", default="dataset/ADNIMERGE2",
                        help="Path to extracted ADNIMERGE2 folder (default: dataset/ADNIMERGE2)")
    parser.add_argument("--out",      default="data/raw",
                        help="Output directory for CSVs (default: data/raw)")
    parser.add_argument("--val",      type=float, default=0.15,
                        help="Validation fraction (default: 0.15)")
    parser.add_argument("--test",     type=float, default=0.15,
                        help="Test fraction (default: 0.15)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    prepare_adni(
        adni_dir  = args.adni_dir,
        out_dir   = args.out,
        val_frac  = args.val,
        test_frac = args.test,
        seed      = args.seed,
    )
