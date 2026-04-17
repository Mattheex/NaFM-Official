"""
Plot per-class AUPRC for NaFM on ClassyFire (or any non-Regression dataset)
============================================================================
Runs inference with a fine-tuned NaFM checkpoint, computes per-class AUPRC,
and saves a bar chart sorted alphabetically by class name.

Usage:
    python scripts/plot_classyfire_auprc.py \\
        --dataset ClassyFire \\
        --dataset-root ./downstream_data/ClassyFire \\
        --dataset-arg superclass \\
        --log-dir logs-finetune-tune \\
        --checkpoint <checkpoint_filename>

Output:
    figures/classyfire_auprc.png
    figures/classyfire_auprc.pdf
"""

import argparse
import json
import os
import sys

# Ensure repo root is on the path when called from scripts/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from tqdm import tqdm

from gnn.datasets import (
    Lotus, Ontology, External, BGC, Classyfire
)
from gnn.tune_module import LNNP as FinetunedLNNP


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser(
        description="Plot per-class AUPRC for NaFM on a classification dataset"
    )
    parser.add_argument("--dataset", default="ClassyFire", type=str,
                        choices=["Lotus", "Ontology", "External", "BGC", "ClassyFire"],
                        help="Dataset name (must not be Regression)")
    parser.add_argument("--dataset-arg", default="superclass", type=str,
                        help="Dataset argument (e.g. superclass / Class)")
    parser.add_argument("--dataset-root", default="./downstream_data/ClassyFire",
                        type=str, help="Path to dataset root directory")
    parser.add_argument("--log-dir", default="logs-finetune-tune", type=str,
                        help="Directory containing splits.npz and the checkpoint")
    parser.add_argument("--checkpoint", default=None, type=str, required=True,
                        help="Checkpoint filename (relative to --log-dir)")
    parser.add_argument("--val-fold", type=int, default=0,
                        help="Fold used as test set for External dataset")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="figures", type=str,
                        help="Directory where plots are saved")
    parser.add_argument("--min-auprc-y", type=float, default=0.2,
                        help="Lower bound of the AUPRC y-axis")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(args, device):
    # Load dataset
    if args.dataset == "Lotus":
        dataset = Lotus(root=args.dataset_root)
    elif args.dataset == "Ontology":
        dataset = Ontology(root=args.dataset_root, dataset_arg=args.dataset_arg)
    elif args.dataset == "External":
        dataset = External(root=args.dataset_root, dataset_arg=args.dataset_arg)
    elif args.dataset == "BGC":
        dataset = BGC(root=args.dataset_root)
    elif args.dataset == "ClassyFire":
        dataset = Classyfire(root=args.dataset_root, dataset_arg=args.dataset_arg)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    n_classes = dataset.num_class

    # Load test indices
    if args.dataset == "External":
        val_fold = args.val_fold
        if val_fold == -1:
            idx_test = np.arange(len(dataset))
        else:
            idx_test = np.where(dataset.data.fold.numpy() == val_fold)[0]
    else:
        splits = np.load(f"{args.log_dir}/splits.npz")
        idx_test = splits.f.idx_test

    test_set = Subset(dataset, idx_test.tolist())
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Load model
    checkpoint_path = f"{args.log_dir}/{args.checkpoint}"
    model = FinetunedLNNP.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        num_classes=n_classes,
    )
    model = model.to(device)
    model.eval()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Collect predictions
    all_outputs, all_labels = [], []
    for inputs in tqdm(test_loader, desc="Inference"):
        inputs = inputs.to(device)
        with torch.no_grad():
            logits = model(inputs)
            probs = torch.softmax(logits, dim=-1)
        all_outputs.append(probs.cpu().numpy())
        all_labels.append(inputs.label.cpu().numpy())

    all_outputs = np.concatenate(all_outputs)   # (N, C)
    all_labels  = np.concatenate(all_labels)    # (N,)

    return all_outputs, all_labels, n_classes


# ---------------------------------------------------------------------------
# Per-class AUPRC
# ---------------------------------------------------------------------------

def compute_per_class_auprc(all_outputs, all_labels, n_classes, id_to_name):
    """Returns a dict {class_name: auprc} for all valid classes."""
    binarized = np.array(label_binarize(all_labels, classes=list(range(n_classes))))

    results = {}
    for i in range(n_classes):
        if len(np.unique(binarized[:, i])) > 1:
            auprc_val = average_precision_score(binarized[:, i], all_outputs[:, i])
            name = id_to_name.get(str(i), str(i))
            results[name] = auprc_val

    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_auprc(per_class_auprc: dict, args):
    # Sort alphabetically by class name
    names  = sorted(per_class_auprc.keys())
    values = [per_class_auprc[n] for n in names]

    n = len(names)
    x = np.arange(n)

    COLOR_NAFM = "#5aafe6"   # light blue matching the reference figure

    fig, ax = plt.subplots(figsize=(max(12, n * 0.18), 5))

    bars = ax.bar(x, values, color=COLOR_NAFM, width=0.7, label="NaFM",
                 linewidth=0)

    # Mean AUPRC line
    mean_val = np.mean(values)
    #ax.axhline(float(mean_val), color="#444444", linewidth=1.2, linestyle="--",
    #           label=f"Mean AUPRC = {mean_val:.3f}")

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=80, fontsize=7, ha="right")
    ax.set_ylabel("AUPRC", fontsize=11)
    ax.set_xlabel(f"{args.dataset_arg.capitalize()}", fontsize=11)
    ax.set_ylim(top=1.02)
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_title(
        f"Per-class AUPRC – NaFM on {args.dataset} ({args.dataset_arg})",
        fontsize=12, pad=10
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"classyfire_{args.dataset_arg}_auprc"
    png_path = os.path.join(args.output_dir, f"{stem}.png")
    pdf_path = os.path.join(args.output_dir, f"{stem}.pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.close(fig)

    # Summary table
    print(f"\nPer-class AUPRC ({args.dataset} / {args.dataset_arg})")
    print(f"{'Class':<50} {'AUPRC':>8}")
    print("-" * 60)
    for name in names:
        print(f"{name:<50} {per_class_auprc[name]:>8.4f}")
    print("-" * 60)
    print(f"{'Mean (valid classes)':<50} {mean_val:>8.4f}")
    print(f"{'N valid classes':<50} {n:>8d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_outputs, all_labels, n_classes = run_inference(args, device)

    id_to_name_path = os.path.join(args.dataset_root, "id_to_name.json")
    with open(id_to_name_path) as f:
        id_to_name = dict(json.load(f))

    per_class_auprc = compute_per_class_auprc(
        all_outputs, all_labels, n_classes, id_to_name
    )

    plot_auprc(per_class_auprc, args)


if __name__ == "__main__":
    main()
