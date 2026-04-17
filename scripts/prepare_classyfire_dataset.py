"""
Prepare ClassyFire ChemOnt Dataset for NaFM Robustness Evaluation
==================================================================
Reads the locally-downloaded ``ChemOnt_2_1.obo`` ontology and, for every
ChemOnt term that has a ``ChEBI_TERM`` synonym, extracts the associated ChEBI
ID and queries the ChEBI REST API to obtain a canonical SMILES.

ChEBI ID extraction (per term):
    - Scans all ``synonym:`` lines for ``ChEBI_TERM`` annotations.
    - Priority: EXACT ChEBI_TERM > RELATED ChEBI_TERM > any other type.
    - Only the first matching ChEBI ID per term is used.

Successfully resolved molecules are stored with their full six-level ChemOnt
hierarchy.

Expected local file (under ``downstream_data/ClassyFire/``):
    ChemOnt_2_1.obo   – ChemOnt ontology (OBO format)

Output
------
downstream_data/ClassyFire/raw/classyfire_data.csv
    Columns: SMILES, ChemOnt_Kingdom, ChemOnt_Superclass, ChemOnt_Class,
             ChemOnt_Subclass, ChemOnt_Level5, ChemOnt_Level6
    Label columns contain 0-indexed integer IDs; NaN where the hierarchy
    level does not exist for a given compound.

downstream_data/ClassyFire/raw/classyfire_label_maps.json
    Human-readable name → integer ID mappings for all six levels.

Usage
-----
    python scripts/prepare_classyfire_dataset.py

    # Limit to N OBO terms queried via API (useful for quick tests):
    python scripts/prepare_classyfire_dataset.py --max-api-terms 500

    # Change minimum class size filter:
    python scripts/prepare_classyfire_dataset.py --min-class-size 20

    # Override input / output directories:
    python scripts/prepare_classyfire_dataset.py \\
        --data-dir path/to/ClassyFire \\
        --output-dir path/to/output

Notes
-----
- ChemOnt hierarchy depth:
    depth 0 = root (CHEMONTID:9999999, "Chemical entities")
    depth 1 = Kingdom
    depth 2 = Superclass
    depth 3 = Class
    depth 4 = Subclass
    depth 5 = Level5
    depth 6 = Level6
- A short sleep between requests respects ChEBI's rate limits.
  Persistent 429/503 errors trigger exponential back-off.
- Requires Python >= 3.9 and the ``requests`` package
  (conda install -c conda-forge requests).
"""

import argparse
import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Optional requests import
# ---------------------------------------------------------------------------
try:
    import requests as _requests  # noqa: F401
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CLASSYFIRE_DIR = PROJECT_ROOT / "downstream_data" / "ClassyFire"
DEFAULT_DATA_DIR = str(CLASSYFIRE_DIR)
DEFAULT_OUTPUT_DIR = str(CLASSYFIRE_DIR / "raw")

OBO_FILENAME = "ChemOnt_2_1.obo"
OUTPUT_CSV = "classyfire_data.csv"
OUTPUT_MAP = "classyfire_label_maps.json"

# Hierarchy depth → output column name  (depth 0 = root, excluded)
DEPTH_TO_COL: Dict[int, str] = {
    1: "ChemOnt_Kingdom",
    2: "ChemOnt_Superclass",
    3: "ChemOnt_Class",
    4: "ChemOnt_Subclass",
    5: "ChemOnt_Level5",
    6: "ChemOnt_Level6",
}

LEVEL_COLS: List[str] = [
    "ChemOnt_Kingdom",
    "ChemOnt_Superclass",
    "ChemOnt_Class",
    "ChemOnt_Subclass",
    "ChemOnt_Level5",
    "ChemOnt_Level6",
]

# Root of the ChemOnt hierarchy
CHEMONT_ROOT = "CHEMONTID:9999999"

# ChEBI REST endpoint
# Returns XML; the <smiles> element holds the SMILES string (when available).
CHEBI_BASE = (
    "https://www.ebi.ac.uk/webservices/chebi/2.0/test/getCompleteEntity"
    "?chebiId=CHEBI:{chebi_id}"
)
# XML namespace used in ChEBI responses
CHEBI_NS = "https://www.ebi.ac.uk/webservices/chebi"

CHEBI_SLEEP = 0.25          # seconds between requests
CHEBI_RETRIES = 3
CHEBI_RETRY_SLEEP = 2.0     # base back-off on 429/503


# ---------------------------------------------------------------------------
# Step 1 – Parse ChemOnt OBO
# ---------------------------------------------------------------------------


def parse_obo(
    obo_path: Path,
) -> Tuple[
    Dict[str, int],           # depth_map   {CHEMONTID: depth}
    Dict[str, str],           # name_map    {CHEMONTID: canonical_name}
    Dict[str, Optional[str]], # parent_map  {CHEMONTID: parent | None}
    Dict[str, str],           # chebi_map   {CHEMONTID: best ChEBI numeric ID}
]:
    """Parse ``ChemOnt_2_1.obo`` and return depth, name, parent and
    best ChEBI ID mappings.

    ChEBI ID priority per term:
        1. First ``EXACT ChEBI_TERM`` synonym
        2. First ``RELATED ChEBI_TERM`` synonym — fallback when no EXACT exists
        3. First ``ChEBI_TERM`` synonym of any other type
    """
    parent_map: Dict[str, Optional[str]] = {}
    name_map: Dict[str, str] = {}
    # chebi_map: best ChEBI numeric ID (string) per CHEMONTID
    exact_chebi: Dict[str, str] = {}
    related_chebi: Dict[str, str] = {}
    other_chebi: Dict[str, str] = {}

    current_id: Optional[str] = None

    # Matches any synonym line that contains a ChEBI_TERM annotation with a
    # CHEBI accession, e.g.:
    #   synonym: "organic molecule" EXACT ChEBI_TERM [CHEBI:72695]
    #   synonym: "organosulfur compound" RELATED ChEBI_TERM [CHEBI:33261]
    _chebi_re = re.compile(
        r'^synonym:\s+"[^"]+"\s+(\w+)\s+ChEBI_TERM\s+\[CHEBI:(\d+)\]'
    )

    with open(obo_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id:"):
                current_id = line[3:].strip()
            elif line.startswith("name:") and current_id is not None:
                name_map[current_id] = line[5:].strip()
                if current_id not in parent_map:
                    parent_map[current_id] = None
            elif line.startswith("synonym:") and current_id is not None:
                m = _chebi_re.match(line)
                if m:
                    syn_type, chebi_num = m.group(1), m.group(2)
                    if syn_type == "EXACT" and current_id not in exact_chebi:
                        exact_chebi[current_id] = chebi_num
                    elif syn_type == "RELATED" and current_id not in related_chebi:
                        related_chebi[current_id] = chebi_num
                    elif current_id not in other_chebi:
                        other_chebi[current_id] = chebi_num
            elif line.startswith("is_a:") and current_id is not None:
                parent_map[current_id] = line[5:].strip().split()[0]

    # Ensure root is present
    parent_map.setdefault(CHEMONT_ROOT, None)
    name_map.setdefault(CHEMONT_ROOT, "Chemical entities")

    # Merge: other_chebi is the base; related overrides; exact overrides all
    chebi_map: Dict[str, str] = {**other_chebi, **related_chebi, **exact_chebi}

    # BFS from root to assign depths
    depth_map: Dict[str, int] = {CHEMONT_ROOT: 0}
    children: Dict[str, List[str]] = {nid: [] for nid in parent_map}
    for child, par in parent_map.items():
        if par is not None:
            children.setdefault(par, []).append(child)

    queue = [CHEMONT_ROOT]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        d = depth_map[node]
        for child in children.get(node, []):
            if child not in depth_map:
                depth_map[child] = d + 1
                queue.append(child)

    n_exact = len(exact_chebi)
    n_related_only = sum(1 for k in related_chebi if k not in exact_chebi)
    n_other_only = sum(
        1 for k in other_chebi if k not in exact_chebi and k not in related_chebi
    )
    print(
        f"  OBO parsed: {len(name_map):,} terms, "
        f"{sum(1 for d in depth_map.values() if d == 1)} Kingdoms, "
        f"{sum(1 for d in depth_map.values() if d == 2)} Superclasses, "
        f"{sum(1 for d in depth_map.values() if d == 3)} Classes, "
        f"{sum(1 for d in depth_map.values() if d == 4)} Subclasses."
    )
    print(
        f"  ChEBI IDs: {n_exact:,} from EXACT synonyms, "
        f"{n_related_only:,} from RELATED-only, "
        f"{n_other_only:,} from other synonym types."
    )
    print(f"  Total queryable terms (with ChEBI ID): {len(chebi_map):,}")
    return depth_map, name_map, parent_map, chebi_map


# ---------------------------------------------------------------------------
# Step 2 – Ancestor resolution (all 6 levels)
# ---------------------------------------------------------------------------


def get_ancestors(
    chemontid: str,
    parent_map: Dict[str, Optional[str]],
    depth_map: Dict[str, int],
    name_map: Dict[str, str],
) -> Dict[str, str]:
    """Walk from *chemontid* to the root and return {col: name} for every
    ancestor (including the node itself) at depth 1–6.
    """
    result: Dict[str, str] = {}
    node: Optional[str] = chemontid
    while node is not None:
        d = depth_map.get(node)
        if d is not None and d in DEPTH_TO_COL:
            col = DEPTH_TO_COL[d]
            if col not in result:
                result[col] = name_map.get(node, node)
        node = parent_map.get(node)
    return result


# ---------------------------------------------------------------------------
# Step 3 – ChEBI REST API helper
# ---------------------------------------------------------------------------


def query_chebi_smiles(chebi_id: str) -> Optional[str]:
    """Return the SMILES from the ChEBI API for numeric *chebi_id*, or None.

    Queries ``getCompleteEntity`` and extracts the ``<smiles>`` element from
    the XML response.  Returns ``None`` when the entry has no SMILES or the
    request fails.
    """
    if not HAS_REQUESTS:
        raise RuntimeError(
            "The 'requests' package is required for ChEBI API queries.\n"
            "Install it with:  conda install -c conda-forge requests"
        )

    import requests  # local import so the module loads without it

    url = CHEBI_BASE.format(chebi_id=chebi_id)
    for attempt in range(CHEBI_RETRIES):
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                # Parse XML; the smiles element may use a namespace prefix
                root = ET.fromstring(resp.text)
                # Try namespace-qualified lookup first, then bare tag
                ns = {"chebi": CHEBI_NS}
                smiles_el = root.find(".//chebi:smiles", ns)
                if smiles_el is None:
                    smiles_el = root.find(".//{%s}smiles" % CHEBI_NS)
                if smiles_el is None:
                    # Fallback: search without namespace
                    smiles_el = root.find(".//smiles")
                if smiles_el is not None and smiles_el.text:
                    return smiles_el.text.strip() or None
                return None
            elif resp.status_code == 404:
                return None
            elif resp.status_code in (429, 503):
                time.sleep(CHEBI_RETRY_SLEEP * (attempt + 1))
            else:
                return None
        except Exception:
            time.sleep(CHEBI_RETRY_SLEEP)
    return None


# ---------------------------------------------------------------------------
# Step 4 – OBO-based ChEBI enrichment pipeline
# ---------------------------------------------------------------------------


def enrich_from_obo(
    depth_map: Dict[str, int],
    name_map: Dict[str, str],
    parent_map: Dict[str, Optional[str]],
    chebi_map: Dict[str, str],
    max_terms: Optional[int] = None,
) -> List[Dict]:
    """For every OBO term with a ChEBI ID, query the ChEBI API for a SMILES
    and record a row with the full six-level hierarchy.

    Terms are processed deepest-first so more specific chemical classes are
    tried before broader class names.
    """
    rows: List[Dict] = []
    api_hits = 0
    api_miss = 0

    term_list = sorted(
        chebi_map.items(),
        key=lambda kv: -depth_map.get(kv[0], 0),
    )
    if max_terms is not None:
        term_list = term_list[:max_terms]

    for chemontid, chebi_id in tqdm(term_list, unit="term", desc="ChEBI API"):
        time.sleep(CHEBI_SLEEP)
        smiles = query_chebi_smiles(chebi_id)

        if not smiles:
            api_miss += 1
            continue

        api_hits += 1
        levels = get_ancestors(chemontid, parent_map, depth_map, name_map)
        if not levels:
            continue

        rows.append({"SMILES": smiles, **levels})

    print(
        f"  API enrichment: {api_hits:,} SMILES found, "
        f"{api_miss:,} not found in ChEBI."
    )
    return rows


# ---------------------------------------------------------------------------
# Step 5 – Build label maps and integer-encode
# ---------------------------------------------------------------------------


def build_label_maps(df: pd.DataFrame) -> Dict:
    """Build ``{column: {name: int_id}}`` mappings from string label cols."""
    maps: Dict = {}
    for col in LEVEL_COLS:
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            maps[col] = {name: idx for idx, name in enumerate(unique_vals)}
    return maps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Prepare NaFM-compatible ClassyFire dataset from the ChemOnt OBO "
            "ontology via PubChem API SMILES lookup."
        )
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing ChemOnt_2_1.obo "
             f"(default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for output files (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-api-terms",
        type=int,
        default=None,
        help="Cap on the number of OBO terms to query via PubChem "
             "(default: all terms with synonyms).",
    )
    parser.add_argument(
        "--min-class-size",
        type=int,
        default=10,
        help="Drop ChemOnt_Class groups with fewer than this many molecules "
             "(default: 10).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not HAS_REQUESTS:
        raise RuntimeError(
            "The 'requests' package is required.\n"
            "Install it with:  conda install -c conda-forge requests"
        )

    # ------------------------------------------------------------------ #
    # 1. Parse OBO ontology                                               #
    # ------------------------------------------------------------------ #
    obo_path = data_dir / OBO_FILENAME
    if not obo_path.exists():
        raise FileNotFoundError(f"OBO file not found: {obo_path}")
    print(f"\n[1/3] Parsing ChemOnt OBO: {obo_path}")
    depth_map, name_map, parent_map, chebi_map = parse_obo(obo_path)

    # ------------------------------------------------------------------ #
    # 2. Query ChEBI API for each ChEBI ID                                #
    # ------------------------------------------------------------------ #
    n_terms = len(chebi_map) if args.max_api_terms is None else args.max_api_terms
    print(
        f"\n[2/3] Querying ChEBI API for SMILES "
        f"({n_terms:,} terms)…"
    )
    rows = enrich_from_obo(
        depth_map,
        name_map,
        parent_map,
        chebi_map,
        max_terms=args.max_api_terms,
    )

    if not rows:
        print("No SMILES retrieved from ChEBI. Exiting.")
        return

    # ------------------------------------------------------------------ #
    # 3. Filter, encode, save                                             #
    # ------------------------------------------------------------------ #
    print(f"\n[3/3] Encoding labels and saving outputs…")

    result_df = pd.DataFrame(rows, columns=["SMILES"] + LEVEL_COLS)

    # Deduplicate on SMILES
    before_dedup = len(result_df)
    result_df = result_df.drop_duplicates(subset="SMILES").reset_index(drop=True)
    print(f"  Deduplicated: {before_dedup - len(result_df):,} duplicates removed.")

    # Drop under-represented classes
    original_len = len(result_df)
    class_counts = result_df["ChemOnt_Class"].value_counts()
    valid_classes = class_counts[class_counts >= args.min_class_size].index
    result_df = result_df[
        result_df["ChemOnt_Class"].isin(valid_classes)
    ].reset_index(drop=True)
    print(
        f"  Dropped {original_len - len(result_df):,} molecules from "
        f"ChemOnt_Class groups with < {args.min_class_size} samples. "
        f"Remaining: {len(result_df):,}"
    )

    # Build label maps (string → int)
    label_maps = build_label_maps(result_df)

    # Integer-encode (NaN stays NaN for absent levels)
    for col, name_to_id in label_maps.items():
        result_df[col] = result_df[col].map(name_to_id)

    # Save CSV
    out_csv = output_dir / OUTPUT_CSV
    result_df.to_csv(out_csv, index=False)
    print(f"\nSaved dataset : {out_csv}")
    print(f"  Total rows  : {len(result_df):,}")
    for col in LEVEL_COLS:
        n_unique = result_df[col].nunique(dropna=True)
        n_nan = result_df[col].isna().sum()
        print(f"  {col}: {n_unique} unique classes, {n_nan:,} N/A")

    # Save label maps
    out_map = output_dir / OUTPUT_MAP
    with open(out_map, "w") as f:
        json.dump(label_maps, f, indent=2)
    print(f"Saved label maps: {out_map}")

    print(
        """
Ready! To fine-tune NaFM on ClassyFire ChemOnt classes, run:

    python train.py \\
        --task finetune \\
        --dataset Ontology \\
        --dataset-root downstream_data/ClassyFire \\
        --dataset-arg ChemOnt_Class \\
        --pretrained-path NaFM.ckpt \\
        --emb-dim 1024 \\
        --num-layer 6 \\
        --drop-ratio 0.1 \\
        --num-epochs 300 \\
        --lr 1e-4 \\
        --lr-min 1e-6 \\
        --batch-size 256 \\
        --val-size 0.1 \\
        --test-size 0.1 \\
        --early-stopping-patience 50 \\
        --log-dir logs-classyfire/seed_0 \\
        --seed 0 \\
        --accelerator cpu
"""
    )


if __name__ == "__main__":
    main()
