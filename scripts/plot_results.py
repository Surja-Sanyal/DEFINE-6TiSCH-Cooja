import os
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
PLOTS_DIR = os.path.join(ROOT_DIR, "Plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

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

def load_raw_data(prefix):
    candidates = glob.glob(os.path.join(STAT_DIR, f"_{prefix}_cooja_stats_.txt"))
    if not candidates:
        candidates = glob.glob(os.path.join(STAT_DIR, f"_{prefix}_stats_.txt"))
    if not candidates:
        return None
    data = np.loadtxt(candidates[0], delimiter=",", usecols=range(12))
    if data.ndim == 1:
        data = data.reshape(1, -1)
    non_zero = data[data[:, 9] > 0]
    return non_zero if len(non_zero) > 0 else data

def get_series(prefix, x_col, fixed_filters):
    data = load_raw_data(prefix)
    if data is None:
        return None
    mask = np.ones(len(data), dtype=bool)
    for col, val in fixed_filters.items():
        mask &= (data[:, col] == val)
    filtered = data[mask]
    if len(filtered) == 0:
        filtered = data
    by_x = {}
    for row in filtered:
        by_x[float(row[x_col])] = row
    return np.array([by_x[k] for k in sorted(by_x.keys())])

def plot_sweep(x_col, fixed_filters, col_idx, xlabel, ylabel, filename, scale=1.0, is_pct=False, is_total_energy=False, is_duty=False):
    plt.figure(figsize=(7, 5), dpi=300)
    for key, (label, color, marker, ls) in MECHANISMS.items():
        mat = get_series(key, x_col, fixed_filters)
        if mat is None or len(mat) == 0:
            continue
        x = mat[:, x_col]
        if is_pct:
            y = np.minimum(100.0, (mat[:, 5] / np.maximum(1.0, mat[:, 0])) * 100.0)
        elif is_total_energy:
            y = (mat[:, 10] + mat[:, 11] * 0.05) * scale
        elif is_duty:
            y = np.minimum(100.0, ((mat[:, 10] + mat[:, 11] * 0.05) / (mat[:, 0] * 360000.0)) * 100.0)
        else:
            y = mat[:, col_idx] * scale
        plt.plot(x, y, label=label, color=color, marker=marker, linestyle=ls, linewidth=1.8, markersize=6)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    # Fig 9: Varying Number of Nodes (sf=61, eb=1)
    f_nodes = {3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6,  "Number of Nodes", "Total sync time (s)", "Fig9a_total_sync_time.png")
    plot_sweep(0, f_nodes, 8,  "Number of Nodes", "Average sync time (slots)", "Fig9b_avg_sync_time.png")
    plot_sweep(0, f_nodes, 9,  "Number of Nodes", "Collisions", "Fig9c_collisions.png")
    plot_sweep(0, f_nodes, 5,  "Number of Nodes", "DODAG completion (%)", "Fig9d_dodag_completion.png", is_pct=True)

    # Fig 11: Energy & Duty Cycle vs Number of Nodes (sf=61, eb=1)
    plot_sweep(0, f_nodes, 10, "Number of Nodes", "Total Tx Count / Energy", "Fig11a_tx_energy.png")
    plot_sweep(0, f_nodes, 11, "Number of Nodes", "Pledge Rx Scan Energy", "Fig11b_rx_energy.png")
    plot_sweep(0, f_nodes, 10, "Number of Nodes", "Total Radio Energy", "Fig11c_total_energy.png", is_total_energy=True)
    plot_sweep(0, f_nodes, 10, "Number of Nodes", "Radio duty cycle (%)", "Fig11d_duty_cycle.png", is_duty=True)

    # Fig 13: Varying Slotframe Size (nodes=250, eb=1)
    f_sf = {0: 250, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size (slots)", "Total sync time (s)", "Fig13a_sf_total_sync_time.png")
    plot_sweep(3, f_sf, 8, "Slotframe size (slots)", "Average sync time (slots)", "Fig13b_sf_avg_sync_time.png")
    plot_sweep(3, f_sf, 9, "Slotframe size (slots)", "Collisions", "Fig13c_sf_collisions.png")
    plot_sweep(3, f_sf, 5, "Slotframe size (slots)", "DODAG completion (%)", "Fig13d_sf_dodag_completion.png", is_pct=True)

    # Fig 14: Varying EB Interval Multiplier (nodes=250, sf=61)
    f_eb = {0: 250, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval multiplier", "Total sync time (s)", "Fig14a_eb_total_sync_time.png")
    plot_sweep(4, f_eb, 8, "EB interval multiplier", "Average sync time (slots)", "Fig14b_eb_avg_sync_time.png")
    plot_sweep(4, f_eb, 9, "EB interval multiplier", "Collisions", "Fig14c_eb_collisions.png")
    plot_sweep(4, f_eb, 5, "EB interval multiplier", "DODAG completion (%)", "Fig14d_eb_dodag_completion.png", is_pct=True)

    # Fig 12: Varying Number of Channels (nodes=250, sf=61, eb=1)
    f_ch = {0: 250, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Number of Channels", "Total sync time (s)", "Fig12a_ch_total_sync_time.png")
    plot_sweep(2, f_ch, 8, "Number of Channels", "Average sync time (slots)", "Fig12b_ch_avg_sync_time.png")
    plot_sweep(2, f_ch, 9, "Number of Channels", "Collisions", "Fig12c_ch_collisions.png")
    plot_sweep(2, f_ch, 5, "Number of Channels", "DODAG completion (%)", "Fig12d_ch_dodag_completion.png", is_pct=True)
