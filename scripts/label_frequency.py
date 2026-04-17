"""
Compute the frequency of each label in the processed ClassyFire superclass dataset
and save a bar plot.

Imbalance metric: Gini coefficient
  - 0 = perfectly balanced (all classes equally represented)
  - 1 = maximally imbalanced (all samples in one class)

Usage:
    python scripts/label_frequency.py
"""

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

order = 'superclass'
# ── Paths ────────────────────────────────────────────────────────────────────
PT_FILE      = Path(f"downstream_data/ClassyFire/processed/Classyfire_{order}.pt")
ID_TO_NAME_FILE = Path("downstream_data/ClassyFire/id_to_name.json")
SPLITS_FILE  = Path("logs-classyfire-class/splits.npz")
FIGURES_DIR  = Path("figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Load label-to-name mapping ───────────────────────────────────────────────
with open(ID_TO_NAME_FILE, "r") as f:
    id_to_name: dict[str, str] = json.load(f)

# ── Load the processed dataset ───────────────────────────────────────────────
data, slices = torch.load(PT_FILE, weights_only=False)

# `data.label` is a 1-D tensor: one scalar label per sample.
labels: torch.Tensor = data.label  # shape: (num_samples,)

# ── Load splits ───────────────────────────────────────────────────────────────
splits      = np.load(SPLITS_FILE)
idx_test    = splits["idx_test"]    # integer indices into the dataset
test_labels = labels[idx_test]      # labels for the test split


# ── Helper: Gini coefficient ──────────────────────────────────────────────────
def gini(counts: list[int]) -> float:
    """
    Gini coefficient over class counts.
    0 = perfectly balanced, 1 = maximally imbalanced.
    """
    arr = np.array(sorted(counts), dtype=float)
    n   = arr.sum()
    if n == 0 or len(arr) == 1:
        return 0.0
    arr /= n                      # convert to proportions
    arr.sort()
    k   = np.arange(1, len(arr) + 1)
    return float(1.0 - 2.0 * np.sum((len(arr) - k + 1) * arr) / (len(arr) * arr.sum()))


# ── Helper: print a frequency table ──────────────────────────────────────────
def print_frequency_table(label_tensor: torch.Tensor, title: str) -> Counter:
    counter      = Counter(label_tensor.tolist())
    sorted_items = sorted(counter.items(), key=lambda x: -x[1])
    total        = label_tensor.numel()
    counts_list  = [c for _, c in sorted_items]

    # Imbalance ratio
    ir = max(counts_list) / min(counts_list) if min(counts_list) > 0 else float("inf")
    g  = gini(counts_list)

    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")
    print(f"  Total samples  : {total}")
    print(f"  Unique classes : {len(counter)}")
    print(f"  Imbalance Ratio (max/min count) : {ir:.1f}x")
    print(f"  Gini coefficient                : {g:.4f}  "
          f"({'highly imbalanced' if g > 0.5 else 'moderately imbalanced' if g > 0.2 else 'fairly balanced'})")
    print()

    header = f"  {'Label ID':>10}  {'Count':>8}  {'Frequency (%)':>14}  Name"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label_id, count in sorted_items:
        name = id_to_name.get(str(int(label_id)), "<unknown>")
        freq = 100.0 * count / total
        print(f"  {int(label_id):>10}  {count:>8}  {freq:>13.2f}%  {name}")

    return counter


# ── Print full-dataset table ──────────────────────────────────────────────────
full_counter = print_frequency_table(labels, f"FULL dataset — ClassyFire {order}")

# ── Print test-split table ────────────────────────────────────────────────────
test_counter = print_frequency_table(test_labels, f"TEST split  — ClassyFire {order}")


# ── Bar plot (test split only) ────────────────────────────────────────────────
sorted_test  = sorted(test_counter.items(), key=lambda x: -x[1])
names_test   = [id_to_name.get(str(int(lid)), f"ID {int(lid)}") for lid, _ in sorted_test]
counts_test  = [cnt for _, cnt in sorted_test]
total_test   = test_labels.numel()

fig, ax = plt.subplots(figsize=(max(12, len(names_test) * 0.18), 5))
x    = range(len(names_test))
bars = ax.bar(x, counts_test, color="steelblue", linewidth=0,width=0.7)

# for bar, cnt in zip(bars, counts_test):
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         bar.get_height() + total_test * 0.003,
#         str(cnt),
#         ha="center",
#         va="bottom",
#         fontsize=8,
#     )

ax.set_xticks(range(len(names_test)))
ax.set_xticklabels(names_test, rotation=80, ha="right", fontsize=7)
ax.set_ylabel(f"Sample count per {order}")
ax.set_xlim(-0.8, len(names_test) - 0.2)
ax.set_title(f"ClassyFire {order} — test-split label frequency")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()

ext = "png"
out = FIGURES_DIR / f"classyfire_{order}_label_freq_test.{ext}"
fig.savefig(out, dpi=150)
print(f"\nSaved: {out}")

plt.close(fig)
