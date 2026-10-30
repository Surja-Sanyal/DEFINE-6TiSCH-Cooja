import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def load_stats(filepath):
    if not os.path.exists(filepath):
        return None
    data = np.loadtxt(filepath, delimiter=",", usecols=range(12))
    return data

def main():
    stat_dir = "Statistics"
    mechanisms = {
        'p': {'label': 'DEFINE (Proposed)', 'color': '#1f77b4', 'linestyle': '-'},
        'e': {'label': '6TiSCH-MC (RFC 8180)', 'color': '#aec7e8', 'linestyle': '--'},
        'r': {'label': 'EDBA-SA', 'color': '#ff7f0e', 'linestyle': '-.'},
        'd': {'label': 'ECV-SA', 'color': '#2ca02c', 'linestyle': ':'},
        'a_2': {'label': 'ATP (2 subslots)', 'color': '#d62728', 'linestyle': '-'},
        't': {'label': 'TRGB (2023)', 'color': '#9467bd', 'linestyle': '--'},
        'c': {'label': 'Chang et al. (2026)', 'color': '#8c564b', 'linestyle': '-.'},
        's': {'label': 'EASE (2026)', 'color': '#e377c2', 'linestyle': ':'}
    }

    plt.figure(figsize=(8, 6), dpi=300)
    
    for key, props in mechanisms.items():
        pattern = os.path.join(stat_dir, f"_{key}_stats_.txt")
        files = glob.glob(pattern)
        if not files:
            continue
        data = load_stats(files[0])
        if data is not None and len(data) > 0:
            if data.ndim == 1:
                data = data.reshape(1, -1)
            nodes = data[:, 0]
            sync_time = data[:, 6] # Total synchronization time
            plt.plot(nodes, sync_time, label=props['label'], color=props['color'], linestyle=props['linestyle'], marker='o', linewidth=1.5)

    plt.xlabel("Total Pledges (Network Density)", fontsize=12)
    plt.ylabel("Total Synchronization Time (s)", fontsize=12)
    plt.title("Network Synchronization Time vs. Node Density", fontsize=14, fontweight='bold')
    plt.legend(frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    output_plot = "synchronization_time_comparison.png"
    plt.savefig(output_plot, dpi=300)
    print(f"Plot successfully generated and saved as {output_plot}")

if __name__ == "__main__":
    main()