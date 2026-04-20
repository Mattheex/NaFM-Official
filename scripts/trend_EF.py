import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
TARGET_MAP = {
    "204": "AChE",
    #"178": "PTP-1B",
    "109": "CYP3A4",
    #"30":  "COX-1",
    "570": "5-Lipoxygenase",
    #"668": "P-glycoprotein 1",
    #"208": "CYP1A2"
}

FOLD_IDS = [1, 2, 3, 4] 
DATASET_NAME = "External"

def extract_combined_evolution():
    """Extracts epoch, ef5, and val_loss, grouped by epoch."""
    target_histories = {}

    for target_id, target_name in TARGET_MAP.items():
        all_folds_df = []
        for fold in FOLD_IDS:
            folder_path = f"log-{DATASET_NAME}_{target_id}_{fold}"
            csv_files = glob.glob(os.path.join(folder_path, "**/metrics.csv"), recursive=True)
            if not csv_files: continue
            
            try:
                df = pd.read_csv(csv_files[-1])
                cols = [c for c in ['epoch', 'ef5', 'val_loss'] if c in df.columns]
                all_folds_df.append(df[cols])
            except Exception as e:
                print(f"[!] Error: {e}")

        if all_folds_df:
            combined = pd.concat(all_folds_df)
            # This averages the 4 folds for each epoch
            evolution = combined.groupby('epoch').mean().reset_index()
            target_histories[target_name] = evolution

    return target_histories

def plot_target_grid(histories):
    """Generates a grid: Rows = Targets, Columns = (EF5, Val Loss)."""
    os.makedirs('figures', exist_ok=True)
    
    num_targets = len(histories)
    # Create a figure with N rows (one per target) and 2 columns
    fig, axes = plt.subplots(num_targets, 2, figsize=(14, 4 * num_targets), facecolor='#ffffff', sharex=True)
    
    # Define colors for variety
    colors = plt.cm.viridis(np.linspace(0, 0.8, num_targets))

    for i, (target_name, data) in enumerate(histories.items()):
        ax_ef = axes[i, 0]
        ax_loss = axes[i, 1]
        color = colors[i]

        # --- Left Column: EF5 ---
        if 'ef5' in data.columns:
            ax_ef.plot(data['epoch'], data['ef5'], color=color, linewidth=1.5, label='Avg EF5')
            ax_ef.set_ylabel(f"{target_name}\nEF5 Value")
            ax_ef.grid(True, alpha=0.3)
            ax_ef.set_ylim(0,9)
        
        # --- Right Column: Val Loss ---
        if 'val_loss' in data.columns:
            ax_loss.plot(data['epoch'], data['val_loss'], color=color, linewidth=1.5)
            ax_loss.set_ylabel("Val Loss")
            ax_loss.grid(True, alpha=0.3)

        # Set headers only for the first row
        if i == 0:
            ax_ef.set_title("EF5 Evolution", fontsize=14, pad=10,fontweight='bold')
            ax_loss.set_title("Validation Loss", fontsize=14, pad=10,fontweight='bold')

    # Label only the bottom plots' X-axis
    axes[-1, 0].set_xlabel("Epoch")
    axes[-1, 1].set_xlabel("Epoch")

    plt.tight_layout()
    output_path = 'figures/target_wise_evolution.png'
    plt.savefig(output_path, dpi=300)
    print(f"[*] Grid plot saved to: {output_path}")
    plt.show()

if __name__ == "__main__":
    data = extract_combined_evolution()
    if data:
        plot_target_grid(data)