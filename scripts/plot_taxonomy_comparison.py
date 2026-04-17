"""
Plot AUPRC Comparison: NaFM Paper (Table 1) vs. Reproduced Values
==================================================================
Generates a line chart comparing the AUPRC values reported in the
original NaFM paper (Table 1, mean ± SD over 3 seeds) for natural
product taxonomy classification against the AUPRC values reproduced
in this reusability study (single run, seed 0).

The paper values are shown as a line with a shaded ±1 SD band.
The reproduced values are shown as a plain line.

Usage:
    python scripts/plot_taxonomy_comparison.py

Output:
    figures/taxonomy_comparison.png
    figures/taxonomy_comparison.pdf
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

training_samples = [4, 8, 16, 24, 40, 64]
x = np.arange(len(training_samples))

# Original paper: Table 1 — NaFM mean AUPRC (%) over 3 random seeds
paper_mean = np.array([70.10, 79.89, 87.37, 89.15, 90.77, 91.75]) / 100.0
paper_std  = np.array([ 0.92,  0.07,  1.51,  0.22,  0.26,  0.47]) / 100.0

# Reproduced AUPRC: single run (seed 0), from this reusability study
reproduced_auprc = np.array([
    0.6833313666481845,
    0.7913935916919913,
    0.8536716412359994,
    0.8803619014160159,
    0.9000399147262391,
    0.9126560955392224,
])

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))

COLOR_PAPER = "#4C72B0"
COLOR_REPRO = "#DD8452"

# --- Paper NaFM line + shaded ±1 SD band ---
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

# --- Reproduced AUPRC line ---
ax.plot(
    x,
    reproduced_auprc,
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
ax.set_xticklabels([str(s) for s in training_samples], fontsize=10)
ax.set_ylabel("AUPRC", fontsize=11)
ax.set_xlabel("Training Samples per Class", fontsize=11)
ax.set_title(
    "Natural Product Taxonomy Classification AUPRC: Paper vs. Reproduced",
    fontsize=12,
    pad=12,
)

y_lo = max(0.0, (paper_mean - paper_std).min() - 0.05)
y_hi = min(1.0, max(reproduced_auprc.max(), (paper_mean + paper_std).max()) + 0.04)
ax.set_ylim(y_lo, y_hi)

ax.yaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%")
)

ax.yaxis.grid(True, linestyle="--", alpha=0.6, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower right", fontsize=10)

plt.tight_layout()

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

output_dir = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(output_dir, exist_ok=True)

png_path = os.path.join(output_dir, "taxonomy_comparison.png")
pdf_path = os.path.join(output_dir, "taxonomy_comparison.pdf")

fig.savefig(png_path, dpi=200, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")

plt.close(fig)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

print("\nNatural Product Taxonomy Classification AUPRC Summary")
print("-" * 70)
header = (f"{'Samples/Class':>14} {'Paper AUPRC':>12} {'Paper SD':>9} "
          f"{'Repro AUPRC':>12} {'Delta':>8} {'In SD?':>7}")
print(header)
print("-" * 70)
for s, pm, ps, ra in zip(training_samples, paper_mean, paper_std, reproduced_auprc):
    delta = ra - pm
    in_sd = abs(delta) <= ps
    flag = "YES" if in_sd else "NO ★"
    print(f"{s:>14} {pm*100:>11.2f}% {ps*100:>8.2f}% "
          f"{ra*100:>11.2f}% {delta*100:>+7.2f}% {flag:>7}")
print("-" * 70)
