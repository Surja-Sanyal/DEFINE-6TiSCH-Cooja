import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
PLOTS_DIR = os.path.join(ROOT_DIR, "Plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Format: (Label, Color, Marker, LineStyle, LineWidth, MarkerFaceColor, Alpha)
MECHANISMS = {
    "p":   ("DEFINE",     "#0052cc", "o", "-",                2.4, "#0052cc", 1.0),
    "e":   ("6TiSCH-MC",  "#d95f02", "s", (0, (5, 2)),        1.2, "white",   0.7),
    "r":   ("EDBA-SA",    "#7570b3", "^", (0, (4, 2, 1, 2)),  1.2, "white",   0.7),
    "d":   ("ECV-SA",     "#1b9e77", "v", (0, (2, 1.5)),      1.2, "white",   0.7),
    "a_2": ("ATP: 2",     "#e7298a", "D", (0, (6, 2)),        1.2, "white",   0.7),
    "t":   ("TRGB",       "#e6ab02", "P", (0, (3, 1.5)),      1.2, "white",   0.7),
    "c":   ("CSMJ",       "#666666", "X", (0, (1.5, 1.5)),    1.2, "white",   0.7),
    "s":   ("EASE",       "#a6761d", "*", (0, (4, 1.5)),      1.2, "white",   0.7),
}

def save_standalone_legend():
    handles = []
    for _, (label, color, marker, ls, lw, mfc, alpha) in MECHANISMS.items():
        ms = 9.0 if marker == "*" else 7.0
        handles.append(
            Line2D(
                [0], [0],
                color=color,
                marker=marker,
                linestyle=ls,
                linewidth=lw,
                markersize=ms,
                markerfacecolor=mfc,
                markeredgecolor=color,
                markeredgewidth=1.4,
                alpha=alpha,
                label=label,
            )
        )
    fig = plt.figure(figsize=(7.2, 0.70), dpi=300)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        edgecolor="#444444",
        facecolor="#fcfcfc",
        fontsize=10.0,
        handlelength=3.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    plt.axis("off")
    for out_path in [
        os.path.join(ROOT_DIR, "legend_only.png"),
        os.path.join(PLOTS_DIR, "legend_only.png"),
    ]:
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(RESULTS_DIR, "legend_only.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Regenerated clean standalone legend: legend_only.pdf")

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
    plt.figure(figsize=(5.4, 3.7), dpi=300)
    # Plot baselines first with alpha=0.7, then DEFINE on top with alpha=1.0 (zorder=10)
    order = ["e", "r", "d", "a_2", "t", "c", "s", "p"]
    for key in order:
        label, color, marker, ls, lw, mfc, alpha = MECHANISMS[key]
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

        ms = 8.0 if marker == "*" else 6.0
        z = 10 if key == "p" else 2
        plt.plot(
            x, y,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=lw,
            markersize=ms,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.3,
            alpha=alpha,
            zorder=z,
        )

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        plt.savefig(os.path.join(out_dir, png_name), dpi=300, bbox_inches="tight")
    if pdf_name:
        plt.savefig(os.path.join(RESULTS_DIR, pdf_name), format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved untangled {png_name}")

if __name__ == "__main__":
    save_standalone_legend()

    f_nodes = {3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6,  "Total pledges", "Total sync time (s)", "Fig9a_total_sync_time.png", "motewise_total_time.pdf")
    plot_sweep(0, f_nodes, 8,  "Total pledges", "Average sync time (slots)", "Fig9b_avg_sync_time.png", "motewise_average_neighbour_time.pdf")
    plot_sweep(0, f_nodes, 9,  "Total pledges", "Collisions", "Fig9c_collisions.png", "motewise_average_collisions.pdf")
    plot_sweep(0, f_nodes, 5,  "Total pledges", "DODAG completion (%)", "Fig9d_dodag_completion.png", "motewise_dodag_completion.pdf", is_pct=True)

    plot_sweep(0, f_nodes, 6,  "Motes per unit distance", "Total sync time (s)", "Fig10a_density_total_time.png", "densitywise_total_time.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 8,  "Motes per unit distance", "Average sync time (slots)", "Fig10b_density_avg_time.png", "densitywise_average_neighbour_time.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 9,  "Motes per unit distance", "Collisions", "Fig10c_density_collisions.png", "densitywise_average_collisions.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 5,  "Motes per unit distance", "DODAG completion (%)", "Fig10d_density_completion.png", "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True)

    plot_sweep(0, f_nodes, 10, "Motes per unit distance", "Tx energy / count", "Fig11a_tx_energy.png", "densitywise_transmission_energy.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 11, "Motes per unit distance", "Rx scan energy / ticks", "Fig11b_rx_energy.png", "densitywise_reception_energy.pdf", x_scale=0.01)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", "Total radio energy", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, is_total_energy=True)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", "Radio duty cycle (%)", "Fig11d_duty_cycle.png", "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True)

    f_ch = {0: 200, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Total channels", "Total sync time (s)", "Fig12a_ch_total_sync_time.png", "channelwise_total_time.pdf")
    plot_sweep(2, f_ch, 8, "Total channels", "Average sync time (slots)", "Fig12b_ch_avg_sync_time.png", "channelwise_average_neighbour_time.pdf")
    plot_sweep(2, f_ch, 9, "Total channels", "Collisions", "Fig12c_ch_collisions.png", "channelwise_average_collisions.pdf")
    plot_sweep(2, f_ch, 5, "Total channels", "DODAG completion (%)", "Fig12d_ch_dodag_completion.png", "channelwise_dodag_completion.pdf", is_pct=True)

    f_sf = {0: 200, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size", "Total sync time (s)", "Fig13a_sf_total_sync_time.png", "slotframewise_total_time.pdf")
    plot_sweep(3, f_sf, 8, "Slotframe size", "Average sync time (slots)", "Fig13b_sf_avg_sync_time.png", "slotframewise_average_neighbour_time.pdf")
    plot_sweep(3, f_sf, 9, "Slotframe size", "Collisions", "Fig13c_sf_collisions.png", "slotframewise_average_collisions.pdf")
    plot_sweep(3, f_sf, 5, "Slotframe size", "DODAG completion (%)", "Fig13d_sf_dodag_completion.png", "slotframewise_dodag_completion.pdf", is_pct=True)

    f_eb = {0: 200, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval", "Total sync time (s)", "Fig14a_eb_total_sync_time.png", "eb_intervalwise_total_time.pdf")
    plot_sweep(4, f_eb, 8, "EB interval", "Average sync time (slots)", "Fig14b_eb_avg_sync_time.png", "eb_intervalwise_average_neighbour_time.pdf")
    plot_sweep(4, f_eb, 9, "EB interval", "Collisions", "Fig14c_eb_collisions.png", "eb_intervalwise_average_collisions.pdf")
    plot_sweep(4, f_eb, 5, "EB interval", "DODAG completion (%)", "Fig14d_eb_dodag_completion.png", "eb_intervalwise_dodag_completion.pdf", is_pct=True)
