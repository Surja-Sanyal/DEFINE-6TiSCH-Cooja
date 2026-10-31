import os
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")

MECHANISMS = {
    "p":   ("DEFINE",    "#1f77b4", "o", "-"),
    "e":   ("6TiSCH-MC", "#ff7f0e", "s", "--"),
    "r":   ("EDBA-SA",   "#d62728", "^", "-."),
    "d":   ("ECV-SA",    "#2ca02c", "v", ":"),
    "a_2": ("ATP: 2",    "#7f7f7f", "d", "--"),
    "t":   ("TRGB",      "#9467bd", "P", "-."),
    "c":   ("CSMJ",      "#8c564b", "X", ":"),
    "s":   ("EASE",      "#e377c2", "*", "--"),
}

def load_scheme_matrix(prefix, target_sf=61, target_eb=1):
    candidates = glob.glob(os.path.join(STAT_DIR, f"_{prefix}_cooja_stats_.txt"))
    if not candidates:
        candidates = glob.glob(os.path.join(STAT_DIR, f"_{prefix}_stats_.txt"))
    if not candidates:
        return None
    
    data = np.loadtxt(candidates[0], delimiter=",", usecols=range(12))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    # Filter for the specific slotframe length (col index 3) and EB multiplier (col index 4)
    # Columns in row: [n_nodes, area, n_ch, sf_len, eb_mult, num_joined, total_time, ...]
    filtered = data[(data[:, 3] == target_sf) & (data[:, 4] == target_eb)]
    if len(filtered) == 0:
        filtered = data # Fallback if exact match isn't found
        
    return filtered[filtered[:, 0].argsort()]

def plot_metric(col_idx, ylabel, filename, scale=1.0):
    plt.figure(figsize=(7, 5), dpi=300)
    for key, (label, color, marker, ls) in MECHANISMS.items():
        mat = load_scheme_matrix(key)
        if mat is None:
            continue
        x = mat[:, 0]
        y = mat[:, col_idx] * scale
        plt.plot(x, y, label=label, color=color, marker=marker, linestyle=ls, linewidth=1.6, markersize=5)

    plt.xlabel("Number of Nodes")
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(ROOT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved {out_path}")

if __name__ == "__main__":
    plot_metric(col_idx=6, ylabel="Total sync time (s)", filename="Fig9a_total_sync_time.png")
    plot_metric(col_idx=8, ylabel="Average sync time (slots)", filename="Fig9b_avg_sync_time.png")
    plot_metric(col_idx=9, ylabel="Collisions", filename="Fig9c_collisions.png")
    plot_metric(col_idx=10, ylabel="Total Tx Count / Energy", filename="Fig11a_tx_energy.png")