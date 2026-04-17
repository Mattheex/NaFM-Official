"""
Plot RMSE Comparison: NaFM Paper (Table 2) vs. Reproduced Values
=================================================================
Generates a line chart comparing the RMSE values reported in the
original NaFM paper (Table 2, mean ± SD over 3 seeds) against the values
reproduced in this reusability study (single run, seed 0).

The paper values are shown as a line with a shaded ±1 SD band.
The reproduced values are shown as a plain line.

Usage:
    python scripts/plot_rmse_comparison.py

Output:
    figures/rmse_comparison.png
    figures/rmse_comparison.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

targets = [
    "PTP1B\n(n=612)",
    "AChE\n(n=341)",
    "COX-2\n(n=190)",
    "HIV-RT\n(n=186)",
    "Tyrosinase\n(n=186)",
    "CYP3A4\n(n=178)",
    "MRP4\n(n=177)",
    "COX-1\n(n=140)",
]

# Original paper: Table 2 — mean RMSE over 3 random seeds
paper_mean = np.array([0.8243, 1.1227, 0.9239, 1.0802, 0.6927, 0.6922, 0.2265, 0.9286])
paper_std  = np.array([0.1960, 0.1604, 0.1721, 0.1506, 0.2828, 0.1326, 0.0921, 0.1342])

# Reproduced values: single run (seed 0), from this reusability study
reproduced = np.array([1.0040, 1.1372, 1.0786, 1.3958, 0.9501, 0.6962, 0.2101, 0.8142])

n_targets = len(targets)
x = np.arange(n_targets)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(11, 5))

COLOR_PAPER = "#4C72B0"
COLOR_REPRO = "#DD8452"

# --- Paper line + shaded ±1 SD band ---
ax.fill_between(
    x,
    paper_mean - paper_std,
    paper_mean + paper_std,
    alpha=0.18,
    color=COLOR_PAPER,
    linewidth=0,
    label="_nolegend_",
)
ax.plot(
    x,
    paper_mean,
    marker="o",
    markersize=6,
    linewidth=1.8,
    color=COLOR_PAPER,
    label="Paper (mean ± SD, 3 seeds)",
)

# --- Reproduced line only (no shading) ---
ax.plot(
    x,
    reproduced,
    marker="s",
    markersize=6,
    linewidth=1.8,
    color=COLOR_REPRO,
    label="Reproduced (seed 0)",
)

# ---------------------------------------------------------------------------
# Axis formatting
# ---------------------------------------------------------------------------

ax.set_xticks(x)
ax.set_xticklabels(targets, fontsize=9)
ax.set_ylabel("RMSE (pIC50 log units)", fontsize=11)
ax.set_xlabel("Protein Target", fontsize=11)
ax.set_title(
    "Bioactivity Regression RMSE: Paper vs. Reproduced",
    fontsize=12,
    pad=12,
)

y_lo = max(0, (paper_mean - paper_std).min() - 0.1)
y_hi = max(reproduced.max(), (paper_mean + paper_std).max()) + 0.15
ax.set_ylim(y_lo, y_hi)

ax.yaxis.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="upper right", fontsize=10)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, "rmse_comparison.png")
# pdf_path = os.path.join(output_dir, "rmse_comparison.pdf")

fig.savefig(png_path, dpi=200, bbox_inches="tight")
#fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
# print(f"Saved: {pdf_path}")

plt.close(fig)

# ---------------------------------------------------------------------------
# Also print a summary table to stdout
# ---------------------------------------------------------------------------

print("\nRMSE Summary Table")
print("-" * 72)
header = f"{'Target':<18} {'Paper mean':>12} {'Paper SD':>10} {'Reproduced':>12} {'Delta':>8} {'In SD?':>8}"
print(header)
print("-" * 72)
target_labels = [
    "PTP1B", "AChE", "COX-2", "HIV type-1 RT",
    "Tyrosinase", "CYP3A4", "MRP4", "COX-1",
]
for tgt, pm, ps, rp in zip(target_labels, paper_mean, paper_std, reproduced):
    delta = rp - pm
    in_sd = abs(delta) <= ps
    flag = "YES" if in_sd else "NO ★"
    print(f"{tgt:<18} {pm:>12.4f} {ps:>10.4f} {rp:>12.4f} {delta:>+8.3f} {flag:>8}")
print("-" * 72)
