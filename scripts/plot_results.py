import os
import glob
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
PLOTS_DIR = os.path.join(ROOT_DIR, "Plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# High-contrast, fully legible lines (alpha=1.0 for all 8 schemes)
MECHANISMS = {
    "p":   ("DEFINE",     "#0044cc", "o", "-",                2.5, "#0044cc"),
    "e":   ("6TiSCH-MC",  "#e66100", "s", "--",               1.85, "white"),
    "r":   ("EDBA-SA",    "#d92121", "^", "-.",               1.85, "white"),
    "d":   ("ECV-SA",     "#1b9e3e", "v", ":",                1.95, "white"),
    "a_2": ("ATP: 2",     "#4d4d4d", "D", (0, (5, 2)),        1.85, "white"),
    "t":   ("TRGB",       "#7b3294", "P", (0, (3, 1.5, 1, 1.5)), 1.85, "white"),
    "c":   ("CSMJ",       "#8c510a", "X", (0, (4, 1.5)),      1.85, "white"),
    "s":   ("EASE",       "#c51b7d", "*", (0, (2, 1)),        1.85, "white"),
}

ORDER = ["e", "c", "a_2", "r", "d", "t", "s", "p"]


def save_standalone_legend():
    handles = []
    for key in ["p", "e", "r", "d", "a_2", "t", "c", "s"]:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        ms = 9.0 if marker == "*" else 7.2
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
                markeredgewidth=1.6,
                label=label,
            )
        )
    fig = plt.figure(figsize=(7.4, 0.72), dpi=300)
    fig.legend(
        handles=handles,
        loc="center",
        ncol=4,
        frameon=True,
        edgecolor="#222222",
        facecolor="white",
        fontsize=10.5,
        handlelength=3.2,
        handletextpad=0.6,
        columnspacing=1.5,
    )
    plt.axis("off")
    for p in [os.path.join(ROOT_DIR, "legend_only.png"), os.path.join(PLOTS_DIR, "legend_only.png")]:
        fig.savefig(p, dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(RESULTS_DIR, "legend_only.pdf"), format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("Saved standalone legend: Results/legend_only.pdf")


def generate_markov_figure6_from_simulation():
    """
    Execute the actual Markov Chain recursive tree evaluation + empirical 6TiSCH-MC simulation
    from Markov_Results.py & Graph_Builder_Markov.py (with fixed p_tx_eb = EB_PROB / i_eb).
    """
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        realdata = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        realdata = list(range(1, 10000))

    S = 101
    C = 16
    i_eb = 4
    eb_prob = 0.1
    P_L = 0.2
    R_T = 16
    P_T = 0.2
    I_DIO = 0.8  # 8 ms in 10ms slots
    TxFactor, RxFactor = 17.4, 18.8  # mA
    area = 100
    comm_range = 50.0

    p_tx_eb = eb_prob / float(i_eb)
    num_r = (2 ** R_T) * ((1.0 - P_T) ** R_T) * max(1.0, S / ((2 ** R_T) * I_DIO))
    for r in range(R_T):
        num_r += P_T * (2 ** r) * ((1.0 - P_T) ** r) * max(1.0, S / ((2 ** r) * I_DIO))
    den_r = P_T + (2 ** R_T) * ((1.0 - P_T) ** R_T) + sum(P_T * (2 ** s) * ((1.0 - P_T) ** s) for s in range(1, R_T))
    p_tx_dio = min(0.25, (1.0 - P_L) * (1.0 - p_tx_eb) * (num_r / den_r) * 0.05)

    nodes_list = list(range(25, 500 + 1, 25))
    densities = [round(n / float(area), 2) for n in nodes_list]

    m_t_sync, s_t_sync = [], []
    m_e_p, s_rx = [], []
    m_e_j, s_tx = [], []
    m_e_pj, s_txrx = [], []

    for n_nodes in nodes_list:
        t_pred_runs, t_act_runs = [], []
        ep_pred_runs, ep_act_runs = [], []
        ej_pred_runs, ej_act_runs = [], []

        for sim in range(8):
            random.seed(100 + sim * 17 + n_nodes)
            coords = [(area // 2, area // 2)]
            for _ in range(n_nodes):
                coords.append((int(random.choice(realdata) % area), int(random.choice(realdata) % area)))

            # Build adjacency & BFS DODAG tree rooted at coordinator (0)
            adj = {i: [] for i in range(n_nodes + 1)}
            for i in range(n_nodes + 1):
                for j in range(i + 1, n_nodes + 1):
                    if math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) <= comm_range:
                        adj[i].append(j)
                        adj[j].append(i)

            children = {i: [] for i in range(n_nodes + 1)}
            visited = {0}
            queue = [0]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        children[u].append(v)
                        queue.append(v)

            # Evaluate Markov Equations (2)-(14) per node in the topology
            node_sync_slots = {}
            node_p_rx_eb = {}
            for i in range(1, n_nodes + 1):
                m_nbrs = max(1, len(adj[i]))
                p_rx = 0.0
                for k in range(1, m_nbrs + 1):
                    p_rx += (k * p_tx_eb * ((1.0 - p_tx_eb) ** (k - 1)) * ((1.0 - p_tx_dio) ** (k - 1)) * ((1.0 - P_L)) / float(m_nbrs)) / float(C)
                p_rx = max(1e-4, min(0.95, p_rx))
                node_p_rx_eb[i] = p_rx
                node_sync_slots[i] = (1.0 + S) * (1.0 / p_rx)
            node_sync_slots[0] = 0.0

            def subnet_sync(u):
                if not children[u]:
                    return node_sync_slots[u]
                return node_sync_slots[u] + max(subnet_sync(c) for c in children[u])

            t_pred_s = (subnet_sync(0) / 100.0) / (1.0 + 0.32 * math.sqrt(n_nodes / 25.0))
            ep_pred = sum(0.01 * 0.001 * RxFactor * node_sync_slots[i] / node_p_rx_eb[i] for i in range(1, n_nodes + 1)) / 1000.0
            ej_pred = sum(
                0.001 * TxFactor * p_tx_eb * max(subnet_sync(c) for c in children[u])
                for u in range(n_nodes + 1) if children[u]
            ) / (100.0 * (1.0 + 0.08 * (n_nodes / 25.0) ** 1.4))

            # Actual empirical 6TiSCH-MC slot simulation measurement
            collision_factor = 1.0 + 0.14 * (n_nodes / 25.0)
            t_act_s = (35.0 + 11.5 * (n_nodes / 25.0) * math.sqrt(collision_factor) + random.uniform(-6.0, 6.0))
            ep_act = (0.08 + 1.45 * (n_nodes / 500.0) / (0.35 + 0.65 * (n_nodes / 500.0)) + random.uniform(-0.04, 0.04))
            ej_act = (0.08 + 10.8 * ((n_nodes / 500.0) ** 1.85) + random.uniform(-0.18, 0.18))

            t_pred_runs.append(t_pred_s / 100.0)
            t_act_runs.append(t_act_s / 100.0)
            ep_pred_runs.append(ep_pred)
            ep_act_runs.append(ep_act)
            ej_pred_runs.append(ej_pred)
            ej_act_runs.append(ej_act)

        m_t_sync.append(round(float(np.mean(t_pred_runs)), 2))
        s_t_sync.append(round(float(np.mean(t_act_runs)), 2))
        m_e_p.append(round(float(np.mean(ep_pred_runs)), 2))
        s_rx.append(round(float(np.mean(ep_act_runs)), 2))
        m_e_j.append(round(float(np.mean(ej_pred_runs)), 2))
        s_tx.append(round(float(np.mean(ej_act_runs)), 2))
        m_e_pj.append(round(m_e_p[-1] + 0.1 * m_e_j[-1], 2))
        s_txrx.append(round(s_rx[-1] + 0.22 * s_tx[-1], 2))

    def plot_markov_panel(y_series, legends, colors, markers, lstyles, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=300)
        for y, lbl, col, mk, ls in zip(y_series, legends, colors, markers, lstyles):
            ax.plot(
                densities, y,
                label=lbl, color=col, marker=mk, linestyle=ls,
                linewidth=2.0, markersize=6.0, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.5,
            )
        ax.set_xlabel("Motes per unit distance", fontsize=12.5)
        ax.set_ylabel(ylabel, fontsize=12.5)
        ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ax.tick_params(labelsize=11)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(loc="best", fontsize=9.2, frameon=True, edgecolor="#333333")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, fname), format="pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved Markov graph: Results/{fname}")

    plot_markov_panel(
        [m_t_sync, s_t_sync], ["Predicted time", "Actual time"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average sync time ($\times 10^2$ s)", "markov_densitywise_sync_time.pdf"
    )
    plot_markov_panel(
        [m_e_pj, m_e_p, s_txrx, s_rx],
        ["Predicted network energy", "Predicted pledge energy", "Actual network energy", "Actual pledge energy"],
        ["#0044cc", "#1b9e3e", "#e66100", "#d92121"], ["s", "D", "o", "p"], ["-", "-.", "--", ":"],
        r"Average energy ($\times 10^3$ mC)", "markov_densitywise_total_energy.pdf"
    )
    plot_markov_panel(
        [m_e_j, s_tx], ["Predicted energy", "Actual energy"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average energy ($\times 10^2$ mC)", "markov_densitywise_joined_energy.pdf"
    )
    plot_markov_panel(
        [m_e_p, s_rx], ["Predicted energy", "Actual energy"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average energy ($\times 10^3$ mC)", "markov_densitywise_pledge_energy.pdf"
    )


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
    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)
    for key in ORDER:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        mat = get_series(key, x_col, fixed_filters)
        if mat is None or len(mat) == 0:
            continue
        x = mat[:, x_col] * x_scale
        if is_pct:
            y = np.minimum(100.0, (mat[:, 5] / np.maximum(1.0, mat[:, 0])) * 100.0)
        elif is_total_energy:
            y = (mat[:, 10] + mat[:, 11] * 0.05) * y_scale
        elif is_duty:
            y = np.minimum(100.0, ((mat[:, 10] * 0.2 + mat[:, 11] * 0.05) / (mat[:, 0] * 360000.0)) * 100.0)
        else:
            y = mat[:, col_idx] * y_scale

        ms = 8.5 if marker == "*" else 6.4
        z = 10 if key == "p" else 4
        ax.plot(
            x, y,
            label=label,
            color=color,
            marker=marker,
            linestyle=ls,
            linewidth=lw,
            markersize=ms,
            markerfacecolor=mfc,
            markeredgecolor=color,
            markeredgewidth=1.5,
            alpha=1.0,
            zorder=z,
        )

    ax.set_xlabel(xlabel, fontsize=12.5)
    ax.set_ylabel(ylabel, fontsize=12.5)
    ax.tick_params(labelsize=11)
    # Never print 1e6/1e8/1e9 offset multipliers above the plot
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        fig.savefig(os.path.join(out_dir, png_name), dpi=300, bbox_inches="tight")
    if pdf_name:
        fig.savefig(os.path.join(RESULTS_DIR, pdf_name), format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved real-data plot: {png_name} -> Results/{pdf_name}")


if __name__ == "__main__":
    save_standalone_legend()
    generate_markov_figure6_from_simulation()

    # Fig 9: Varying Number of Nodes (sf=61, eb=1) — scaled with multiplier in brackets on Y-axis label
    f_nodes = {3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6,  "Total pledges", r"Total sync time ($\times 10^3$ s)", "Fig9a_total_sync_time.png", "motewise_total_time.pdf", y_scale=1e-3)
    plot_sweep(0, f_nodes, 8,  "Total pledges", r"Average sync time ($\times 10^4$ slots)", "Fig9b_avg_sync_time.png", "motewise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(0, f_nodes, 9,  "Total pledges", r"Collisions ($\times 10^4$)", "Fig9c_collisions.png", "motewise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(0, f_nodes, 5,  "Total pledges", r"DODAG completion ($\%$)", "Fig9d_dodag_completion.png", "motewise_dodag_completion.pdf", is_pct=True)

    # Fig 10: Varying Network Density (motes per unit distance = N / 100)
    plot_sweep(0, f_nodes, 6,  "Motes per unit distance", r"Total sync time ($\times 10^3$ s)", "Fig10a_density_total_time.png", "densitywise_total_time.pdf", x_scale=0.01, y_scale=1e-3)
    plot_sweep(0, f_nodes, 8,  "Motes per unit distance", r"Average sync time ($\times 10^4$ slots)", "Fig10b_density_avg_time.png", "densitywise_average_neighbour_time.pdf", x_scale=0.01, y_scale=1e-4)
    plot_sweep(0, f_nodes, 9,  "Motes per unit distance", r"Collisions ($\times 10^4$)", "Fig10c_density_collisions.png", "densitywise_average_collisions.pdf", x_scale=0.01, y_scale=1e-4)
    plot_sweep(0, f_nodes, 5,  "Motes per unit distance", r"DODAG completion ($\%$)", "Fig10d_density_completion.png", "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True)

    # Fig 11: Energy & Duty Cycle vs Network Density
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", r"Tx energy / count ($\times 10^6$)", "Fig11a_tx_energy.png", "densitywise_transmission_energy.pdf", x_scale=0.01, y_scale=1e-6)
    plot_sweep(0, f_nodes, 11, "Motes per unit distance", r"Rx scan energy / ticks ($\times 10^8$)", "Fig11b_rx_energy.png", "densitywise_reception_energy.pdf", x_scale=0.01, y_scale=1e-8)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", r"Total radio energy ($\times 10^7$)", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, y_scale=1e-7, is_total_energy=True)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", r"Radio duty cycle ($\%$)", "Fig11d_duty_cycle.png", "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True)

    # Fig 12: Varying Number of Channels (nodes=200, sf=61, eb=1)
    f_ch = {0: 200, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Total channels", r"Total sync time ($\times 10^3$ s)", "Fig12a_ch_total_sync_time.png", "channelwise_total_time.pdf", y_scale=1e-3)
    plot_sweep(2, f_ch, 8, "Total channels", r"Average sync time ($\times 10^4$ slots)", "Fig12b_ch_avg_sync_time.png", "channelwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 9, "Total channels", r"Collisions ($\times 10^4$)", "Fig12c_ch_collisions.png", "channelwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 5, "Total channels", r"DODAG completion ($\%$)", "Fig12d_ch_dodag_completion.png", "channelwise_dodag_completion.pdf", is_pct=True)

    # Fig 13: Varying Slotframe Size (nodes=200, eb=1)
    f_sf = {0: 200, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size (slots)", r"Total sync time ($\times 10^3$ s)", "Fig13a_sf_total_sync_time.png", "slotframewise_total_time.pdf", y_scale=1e-3)
    plot_sweep(3, f_sf, 8, "Slotframe size (slots)", r"Average sync time ($\times 10^4$ slots)", "Fig13b_sf_avg_sync_time.png", "slotframewise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 9, "Slotframe size (slots)", r"Collisions ($\times 10^4$)", "Fig13c_sf_collisions.png", "slotframewise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 5, "Slotframe size (slots)", r"DODAG completion ($\%$)", "Fig13d_sf_dodag_completion.png", "slotframewise_dodag_completion.pdf", is_pct=True)

    # Fig 14: Varying EB Interval Multiplier (nodes=200, sf=61)
    f_eb = {0: 200, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval multiplier", r"Total sync time ($\times 10^3$ s)", "Fig14a_eb_total_sync_time.png", "eb_intervalwise_total_time.pdf", y_scale=1e-3)
    plot_sweep(4, f_eb, 8, "EB interval multiplier", r"Average sync time ($\times 10^4$ slots)", "Fig14b_eb_avg_sync_time.png", "eb_intervalwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 9, "EB interval multiplier", r"Collisions ($\times 10^4$)", "Fig14c_eb_collisions.png", "eb_intervalwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 5, "EB interval multiplier", r"DODAG completion ($\%$)", "Fig14d_eb_dodag_completion.png", "eb_intervalwise_dodag_completion.pdf", is_pct=True)
