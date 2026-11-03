import os
import glob
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
PLOTS_DIR = os.path.join(ROOT_DIR, "Plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

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

def plot_sweep(x_col, fixed_filters, col_idx, xlabel, ylabel, png_name, pdf_name=None, x_scale=1.0, y_scale=1.0, is_pct=False, is_total_energy=False, is_duty=False):
    plt.figure(figsize=(6.5, 4.6), dpi=300)
    for key, (label, color, marker, ls) in MECHANISMS.items():
        mat = get_series(key, x_col, fixed_filters)
        if mat is None or len(mat) == 0:
            continue
        x = mat[:, x_col] * x_scale
        if is_pct:
            y = np.minimum(100.0, (mat[:, 5] / np.maximum(1.0, mat[:, 0])) * 100.0)
        elif is_total_energy:
            y = (mat[:, 10] + mat[:, 11] * 0.05) * y_scale
        elif is_duty:
            y = np.minimum(100.0, ((mat[:, 10] + mat[:, 11] * 0.05) / (mat[:, 0] * 360000.0)) * 100.0)
        else:
            y = mat[:, col_idx] * y_scale
        plt.plot(x, y, label=label, color=color, marker=marker, linestyle=ls, linewidth=1.8, markersize=6)

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(ncol=2, fontsize=8.5)
    plt.tight_layout()
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        plt.savefig(os.path.join(out_dir, png_name), dpi=300)
    if pdf_name:
        plt.savefig(os.path.join(RESULTS_DIR, pdf_name), format="pdf")
    plt.close()
    print(f"Saved {png_name}" + (f" and Results/{pdf_name}" if pdf_name else ""))

if __name__ == "__main__":
    # Fig 9: Varying Number of Nodes (sf=61, eb=1)
    f_nodes = {3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6,  "Total pledges", "Total sync time (s)", "Fig9a_total_sync_time.png", "motewise_total_time.pdf")
    plot_sweep(0, f_nodes, 8,  "Total pledges", "Average sync time (slots)", "Fig9b_avg_sync_time.png", "motewise_average_neighbour_time.pdf")
    plot_sweep(0, f_nodes, 9,  "Total pledges", "Collisions", "Fig9c_collisions.png", "motewise_average_collisions.pdf")
    plot_sweep(0, f_nodes, 5,  "Total pledges", "DODAG completion (%)", "Fig9d_dodag_completion.png", "motewise_dodag_completion.pdf", is_pct=True)

    # Fig 10: Varying Network Density (motes per unit distance = N / 100)
    plot_sweep(0, f_nodes, 6,  "Motes per unit distance", "Total sync time (s)", "Fig10a_density_total_time.png", "densitywise_total_time.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 8,  "Motes per unit distance", "Average sync time (slots)", "Fig10b_density_avg_time.png", "densitywise_average_neighbour_time.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 9,  "Motes per unit distance", "Collisions", "Fig10c_density_collisions.png", "densitywise_average_collisions.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 5,  "Motes per unit distance", "DODAG completion (%)", "Fig10d_density_completion.png", "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True)

    # Fig 11: Energy & Duty Cycle vs Network Density
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", "Tx energy / count", "Fig11a_tx_energy.png", "densitywise_transmission_energy.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 11, "Motes per unit distance", "Rx scan energy / ticks", "Fig11b_rx_energy.png", "densitywise_reception_energy.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", "Total radio energy", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, is_total_energy=True)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", "Radio duty cycle (%)", "Fig11d_duty_cycle.png", "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True)

    # Fig 12: Varying Number of Channels (nodes=250, sf=61, eb=1)
    f_ch = {0: 200, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Total channels", "Total sync time (s)", "Fig12a_ch_total_sync_time.png", "channelwise_total_time.pdf")
    plot_sweep(2, f_ch, 8, "Total channels", "Average sync time (slots)", "Fig12b_ch_avg_sync_time.png", "channelwise_average_neighbour_time.pdf")
    plot_sweep(2, f_ch, 9, "Total channels", "Collisions", "Fig12c_ch_collisions.png", "channelwise_average_collisions.pdf")
    plot_sweep(2, f_ch, 5, "Total channels", "DODAG completion (%)", "Fig12d_ch_dodag_completion.png", "channelwise_dodag_completion.pdf", is_pct=True)

    # Fig 13: Varying Slotframe Size (nodes=250, eb=1)
    f_sf = {0: 200, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size", "Total sync time (s)", "Fig13a_sf_total_sync_time.png", "slotframewise_total_time.pdf")
    plot_sweep(3, f_sf, 8, "Slotframe size", "Average sync time (slots)", "Fig13b_sf_avg_sync_time.png", "slotframewise_average_neighbour_time.pdf")
    plot_sweep(3, f_sf, 9, "Slotframe size", "Collisions", "Fig13c_sf_collisions.png", "slotframewise_average_collisions.pdf")
    plot_sweep(3, f_sf, 5, "Slotframe size", "DODAG completion (%)", "Fig13d_sf_dodag_completion.png", "slotframewise_dodag_completion.pdf", is_pct=True)

    # Fig 14: Varying EB Interval Multiplier (nodes=250, sf=61)
    f_eb = {0: 200, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval", "Total sync time (s)", "Fig14a_eb_total_sync_time.png", "eb_intervalwise_total_time.pdf")
    plot_sweep(4, f_eb, 8, "EB interval", "Average sync time (slots)", "Fig14b_eb_avg_sync_time.png", "eb_intervalwise_average_neighbour_time.pdf")
    plot_sweep(4, f_eb, 9, "EB interval", "Collisions", "Fig14c_eb_collisions.png", "eb_intervalwise_average_collisions.pdf")
    plot_sweep(4, f_eb, 5, "EB interval", "DODAG completion (%)", "Fig14d_eb_dodag_completion.png", "eb_intervalwise_dodag_completion.pdf", is_pct=True)
