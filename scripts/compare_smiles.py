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
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, cast

import pandas as pd

# Ensure the project root is on sys.path so raw_data.raw.filter can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from raw_data.raw.filter import smiles_to_inchikey, standardize_smiles_v2

# ── Paths ────────────────────────────────────────────────────────────────────
#PROJECT_ROOT = Path(__file__).resolve().parent.parent
PKL_PATH = PROJECT_ROOT / "raw_data" / "raw" / "pretrain_smiles.pkl"
CSV_PATH = PROJECT_ROOT / "downstream_data" / "ClassyFire" / "raw" / "massbank_output.csv"
OUTPUT_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_WORKERS = 50
PKL_INCHIKEY_CSV_PATH = OUTPUT_DIR / "pretrain_smiles_inchikey.csv"


def standardize_smiles_parallel(smiles_list: Sequence[str]) -> Set[str]:
    """Standardize SMILES with `standardize_smiles_v2()` using a fixed worker pool."""
    with Pool(NUM_WORKERS) as pool:
        standardized = pool.map(standardize_smiles_v2, smiles_list)
    return {cast(str, s) for s in standardized if s != 0}


def smiles_to_inchikey_parallel(smiles_list: Sequence[str]) -> List[str]:
    """Convert SMILES to InChIKeys without standardization using a fixed worker pool."""
    with Pool(NUM_WORKERS) as pool:
        inchikeys = pool.starmap(
            smiles_to_inchikey,
            ((smiles, False) for smiles in smiles_list),
        )
    return [cast(str, inchikey) for inchikey in inchikeys]


def order_smiles(smiles_iterable: Iterable[str]) -> List[str]:
    """Return a deterministic ordering for a SMILES collection."""
    return sorted(smiles_iterable)


def build_smiles_inchikey_rows(smiles_iterable: Iterable[str]) -> Tuple[List[str], List[str]]:
    """Build aligned SMILES/InChIKey lists and discard failed conversions."""
    ordered_smiles = order_smiles(smiles_iterable)
    inchikeys = smiles_to_inchikey_parallel(ordered_smiles)

    valid_rows = [
        (smiles, inchikey)
        for smiles, inchikey in zip(ordered_smiles, inchikeys)
        if inchikey != 0
    ]

    if not valid_rows:
        return [], []

    valid_smiles, valid_inchikeys = zip(*valid_rows)
    return list(valid_smiles), list(valid_inchikeys)


def find_coconut_ids_not_in_pretrain(coconut_path: Path, output_path: Path, pretrain_inchikeys: Set[str]):
    coconut_df = pd.read_csv(coconut_path)
    id_col = "id" if "id" in coconut_df.columns else coconut_df.columns[0]

    missing_df = coconut_df[
        (coconut_df["inchikey"] != 0)
        & (~coconut_df["inchikey"].isin(pretrain_inchikeys))
    ][[id_col, "canonical_smiles", "inchikey"]].copy()
    missing_df.to_csv(output_path, index=False)
    print(f"      Saved Coconut-only IDs:   {output_path}  ({len(missing_df):,} rows)")
    return missing_df


def load_pretrain_smiles(path: Path) -> Set[str]:
    """Load the pickle file containing a list of SMILES strings."""
    print(f"[1/3] Loading pre-training SMILES from: {path}")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list, got {type(data)}")
    
    return set(data)
    # print(data)
    smiles_set = standardize_smiles_parallel(data)
    print(f"      Total entries in pickle: {len(data):,}")
    print(f"      Valid unique SMILES:     {len(smiles_set):,}")
    return smiles_set


def load_massbank_smiles(path: Path) -> Set[str]:
    df = pd.read_csv(path)
    df = df[df["is_natural_product"] == 1].copy()
    smiles_set = standardize_smiles_parallel(df["SMILES"].dropna().tolist())

    print(f"      Total rows in CSV:       {len(df):,}")
    print(f"      Valid unique SMILES:     {len(smiles_set):,}")
    return smiles_set


def compute_overlap(pkl_set: Set[str], csv_set: Set[str]):
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


def save_overlap_list(overlap: Set[str], path: Path):
    """Save the overlapping SMILES to a text file (one per line)."""
    with open(path, "w") as f:
        for smi in sorted(overlap):
            f.write(smi + "\n")
    print(f"\n  Saved overlapping SMILES list: {path}  ({len(overlap):,} lines)")


def save_pretrain_smiles_inchikey_csv(data: Set[str]):
    """Save pickle SMILES and their corresponding InChIKeys to CSV."""

    print(f"\n[3/4] Creating SMILES/InChIKey CSV from: {PKL_PATH}")
    smiles, inchikeys = build_smiles_inchikey_rows(data)
    df = pd.DataFrame({"smiles": smiles, "inchikey": inchikeys})
    df.to_csv(PKL_INCHIKEY_CSV_PATH, index=False)
    print(f"      Saved CSV:                {PKL_INCHIKEY_CSV_PATH}  ({len(df):,} rows)")


def main():
    pkl_set = load_pretrain_smiles(PKL_PATH)
    new_plk_set = load_pretrain_smiles(PROJECT_ROOT / "raw_data" / "raw" / "2026_pretrain_smiles.pkl")

    only_new     = new_plk_set - pkl_set
    print(len(only_new))
    out_path = OUTPUT_DIR / "new_COCONUT_smiles.txt"
    save_overlap_list(only_new, out_path)

    return

    csv_set = load_massbank_smiles(CSV_PATH)
    save_pretrain_smiles_inchikey_csv(pkl_set)
    

    print(f"\n[4/4] Computing overlap…")
    stats, overlap, only_pkl, only_csv = compute_overlap(pkl_set, csv_set)

    print_summary(stats)

    out_path = OUTPUT_DIR / "raw_data_smiles.txt"
    save_overlap_list(only_pkl, out_path)

    # Optional: save the overlapping SMILES list
    out_path = OUTPUT_DIR / "overlapping_smiles.txt"
    save_overlap_list(overlap, out_path)

    # Also save the MassBank-only SMILES (those not in pre-training)
    out_only_csv = OUTPUT_DIR / "massbank_only_smiles.txt"
    save_overlap_list(only_csv, out_only_csv)

    find_coconut_ids_not_in_pretrain(
        PROJECT_ROOT / "scripts" / "coconut_csv-04-2026.csv",
        OUTPUT_DIR / "coconut_ids_not_in_pretrain.csv",
        only_pkl,
    )


if __name__ == "__main__":
    main()
