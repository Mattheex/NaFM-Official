"""
Compare SMILES Overlap: pretrain_smiles.pkl vs massbank_output.csv
==================================================================
Loads the pre-training SMILES list (pickle) and the MassBank ClassyFire
dataset (CSV), then computes:

  - Total unique SMILES in each source
  - Number of overlapping SMILES (exact string match)
  - Number of SMILES unique to each source
  - Overlap percentage relative to each source

Usage:
    python scripts/compare_smiles.py

Output:
    Console summary table + optional CSV export of overlapping SMILES.
"""

import pickle
import sys
from pathlib import Path

import pandas as pd

# Ensure the project root is on sys.path so raw_data.raw.filter can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from raw_data.raw.filter import standardize_smiles

# ── Paths ────────────────────────────────────────────────────────────────────
#PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKL_PATH     = PROJECT_ROOT / "raw_data" / "raw" / "pretrain_smiles.pkl"
CSV_PATH     = PROJECT_ROOT / "downstream_data" / "ClassyFire" / "raw" / "massbank_output.csv"
PT_PATH = PROJECT_ROOT / "downstream_data" / "ClassyFire" / "processed" / "Classyfire_superclass.pt"
OUTPUT_DIR   = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pretrain_smiles(path: Path) -> set:
    """Load the pickle file containing a list of SMILES strings."""
    print(f"[1/3] Loading pre-training SMILES from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")
    smiles_set = {s.strip() for s in data if isinstance(s, str) and s.strip()}
    print(f"      Total entries in pickle: {len(data):,}")
    print(f"      Valid unique SMILES:     {len(smiles_set):,}")
    return smiles_set


def load_massbank_smiles(path: Path) -> set:
    df = pd.read_csv(path)
    df =df[df['is_natural_product'] == 1].copy()
    df['SMILES'].map(standardize_smiles)

    smiles_set = {s for s in df["SMILES"].dropna() if s != 0}
    print(f"      Total rows in CSV:       {len(df):,}")
    print(f"      Valid unique SMILES:     {len(smiles_set):,}")
    return smiles_set


def compute_overlap(pkl_set: set, csv_set: set):
    """Compute overlap statistics between two SMILES sets."""
    overlap       = pkl_set & csv_set
    only_pkl      = pkl_set - csv_set
    only_csv      = csv_set - pkl_set

    stats = {
        "pretrain_unique":      len(pkl_set),
        "massbank_unique":      len(csv_set),
        "overlap":              len(overlap),
        "only_pretrain":        len(only_pkl),
        "only_massbank":        len(only_csv),
        "pct_of_pretrain":      100.0 * len(overlap) / len(pkl_set) if pkl_set else 0.0,
        "pct_of_massbank":      100.0 * len(overlap) / len(csv_set) if csv_set else 0.0,
    }
    return stats, overlap, only_pkl, only_csv


def print_summary(stats: dict):
    """Print a formatted summary table."""
    print()
    print("=" * 60)
    print("  SMILES Overlap Summary")
    print("=" * 60)
    print(f"  {'Pre-training SMILES (pickle)':<35} {stats['pretrain_unique']:>10,}")
    print(f"  {'MassBank SMILES (CSV)':<35} {stats['massbank_unique']:>10,}")
    print(f"  {'─' * 48}")
    print(f"  {'Overlapping SMILES':<35} {stats['overlap']:>10,}")
    print(f"  {'Only in pre-training':<35} {stats['only_pretrain']:>10,}")
    print(f"  {'Only in MassBank':<35} {stats['only_massbank']:>10,}")
    print(f"  {'─' * 48}")
    print(f"  {'Overlap % of pre-training':<35} {stats['pct_of_pretrain']:>9.2f}%")
    print(f"  {'Overlap % of MassBank':<35} {stats['pct_of_massbank']:>9.2f}%")
    print("=" * 60)


def save_overlap_list(overlap: set, path: Path):
    """Save the overlapping SMILES to a text file (one per line)."""
    with open(path, "w") as f:
        for smi in sorted(overlap):
            f.write(smi + "\n")
    print(f"\n  Saved overlapping SMILES list: {path}  ({len(overlap):,} lines)")


def main():
    pkl_set = load_pretrain_smiles(PKL_PATH)
    csv_set = load_massbank_smiles(CSV_PATH)

    print(f"\n[3/3] Computing overlap…")
    stats, overlap, only_pkl, only_csv = compute_overlap(pkl_set, csv_set)

    print_summary(stats)

    # Optional: save the overlapping SMILES list
    out_path = OUTPUT_DIR / "overlapping_smiles.txt"
    save_overlap_list(overlap, out_path)

    # Also save the MassBank-only SMILES (those not in pre-training)
    out_only_csv = OUTPUT_DIR / "massbank_only_smiles.txt"
    save_overlap_list(only_csv, out_only_csv)


if __name__ == "__main__":
    main()
