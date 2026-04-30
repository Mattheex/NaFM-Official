import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# Map the target IDs used in your bash script to the labels for the plot
TARGET_MAP = {
    "204": "AChE",
    "178": "PTP-1B",
    "109": "CYP3A4",
    "30":  "COX-1",
    "570": "5-Lipoxygenase",
    "668": "P-glycoprotein 1",
    "208": "CYP1A2"
}

# The folds you ran in your bash script
FOLD_IDS = [1, 2, 3, 4] 
SEED_IDS = [0,1,2]
DATASET_NAME = "External"

def extract_averaged_metrics():
    """
    Crawls log directories, reads metrics.csv, and averages EF values across folds.
    """
    final_data = {
        "targets": [],
        "ef1": [],
        "ef5": [],
        "ef10": [],
        "ef1_std": [], # Optional: to track variability
    }

    for target_id, target_name in TARGET_MAP.items():
        fold_ef1, fold_ef5, fold_ef10 = [], [], []

        for seed in SEED_IDS:
            for fold in FOLD_IDS:
                # Matches folder: ./log-External_204_1
                
                seed_path = f"_{seed}"
                folder_path = f"log-{DATASET_NAME}_{target_id}_{fold}{seed_path}"
                
                # Find the metrics.csv file (PyTorch Lightning often nests this in version_X)
                csv_files = glob.glob(os.path.join(folder_path, "**/metrics.csv"), recursive=True)
                
                if not csv_files:
                    print(f"[!] Missing data for {target_name} (ID: {target_id}), Fold: {fold}, Seed: {seed}")
                    continue
                
                try:
                    # Read the latest CSV file found
                    df = pd.read_csv(csv_files[-1])
                    
                    # Get the last recorded values for EFs (usually end of training)
                    # We dropna to ensure we get rows where metrics were actually logged
                    fold_ef1.append(df['ef1'].dropna().iloc[-1])
                    fold_ef5.append(df['ef5'].dropna().iloc[-1])
                    fold_ef10.append(df['ef10'].dropna().iloc[-1])
                except Exception as e:
                    print(f"[!] Error reading {csv_files[0]}: {e}")

        # Calculate means for this target across all successful folds
        final_data["targets"].append(target_name)
        final_data["ef1"].append(np.mean(fold_ef1) if fold_ef1 else 0)
        final_data["ef5"].append(np.mean(fold_ef5) if fold_ef5 else 0)
        final_data["ef10"].append(np.mean(fold_ef10) if fold_ef10 else 0)
        final_data["ef1_std"].append(np.std(fold_ef1) if fold_ef1 else 0)

    return final_data

def plot_enrichment_factors(data):
    """
    Generates the bar plot using the extracted data.
    """
    os.makedirs('figures', exist_ok=True)
    
    targets = data["targets"]
    nafm_ef1 = data["ef1"]
    nafm_ef5 = data["ef5"]
    nafm_ef10 = data["ef10"]

    x = np.arange(len(targets))
    width = 0.15 

    colors = {
        'ef1': '#b3d9eb', 'ef5': '#f2d9b1', 'ef10': '#d9e6d9',
        'ef1_g': '#d1e5f0', 'ef5_g': '#f9ebd2', 'ef10_g': '#e9f1e9'
    }

    fig, ax = plt.subplots(figsize=(16, 4), facecolor='#ffffff')
    ax.set_facecolor('#ebf1f6')

    # Plot NaFM (Our Model)
    ax.bar(x - 2*width, nafm_ef1, width, label='EF1', color=colors['ef1'], edgecolor='white', linewidth=0.5)
    ax.bar(x - width, nafm_ef5, width, label='EF5', color=colors['ef5'], edgecolor='white', linewidth=0.5)
    ax.bar(x, nafm_ef10, width, label='EF10', color=colors['ef10'], edgecolor='white', linewidth=0.5)

    # Styling
    ax.set_title('NaFM screening', fontsize=16, pad=20)
    ax.set_xticks(x-width)
    ax.set_xticklabels(targets, rotation=0)
    
    ax.set_ylabel('Enrichment Factor Value', fontsize=12)
    #ax.set_ylim(0, max(nafm_ef1 + [40]) + 10) # Dynamic height
    ax.set_yticks(np.arange(0, 81, 20))
    ax.set_ylim(0,80)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='white', linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', frameon=False)
    plt.tight_layout()
    
    output_path = 'figures/final_cv_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"[*] Plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    print("[*] Extracting metrics from logs...")
    metrics_data = extract_averaged_metrics()
    
    print("[*] Generating plot...")
    plot_enrichment_factors(metrics_data)