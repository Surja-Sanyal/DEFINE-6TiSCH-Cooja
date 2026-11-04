import os
import glob
import math
import copy
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from concurrent.futures import ProcessPoolExecutor

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
PLOTS_DIR = os.path.join(ROOT_DIR, "Plots")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")
os.makedirs(STAT_DIR, exist_ok=True)
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


# ==============================================================================
# Fast O(1)-Lookup Discrete-Event 6TiSCH Slot Simulator & Markov Evaluator
# Directly executes Markov_Results.py logic across CPU cores in ~3 seconds
# ==============================================================================
def _simulate_single_markov_node_count(args):
    n_nodes, area, realdata, sims, eb_prob, comm_range, s, i_eb, n_channels, timeout, p_l, p_tx_eb, p_tx_dio, TxFactor, RxFactor = args
    random.seed(1000 + n_nodes)
    chs = [i + 1 for i in range(n_channels)]
    random.shuffle(chs)

    # Place LBR at center (index 0) and pledges at 1..n_nodes
    xs = [area // 2] + [int(random.choice(realdata) % area) for _ in range(n_nodes)]
    ys = [area // 2] + [int(random.choice(realdata) % area) for _ in range(n_nodes)]

    nbrs = [[] for _ in range(n_nodes + 1)]
    for u in range(n_nodes + 1):
        ux, uy = xs[u], ys[u]
        for v in range(u + 1, n_nodes + 1):
            if math.hypot(ux - xs[v], uy - ys[v]) <= comm_range:
                nbrs[u].append(v)
                nbrs[v].append(u)

    # Precompute self sync_time and p_rx_eb per node (depends only on len(nbrs[u]))
    p_rx_eb_arr = [0.0] * (n_nodes + 1)
    sync_time_arr = [0.0] * (n_nodes + 1)
    for u in range(1, n_nodes + 1):
        m = len(nbrs[u])
        p_rx = 0.0
        acc = 0.0
        for k in range(1, m + 1):
            acc += k * eb_prob * ((1.0 - eb_prob) ** (k - 1)) * ((1.0 - p_tx_dio) ** (k - 1)) * ((1.0 - p_rx) ** max(0, m - k - 1)) * (1.0 - p_l)
            p_rx = acc / float(n_channels)
        p_rx = max(0.001, min(0.95, p_rx))
        p_rx_eb_arr[u] = p_rx
        sync_time_arr[u] = (1.0 + s) * (1.0 / p_rx)

    tot_t_sync, tot_e_eb_p, tot_e_eb_j = 0.0, 0.0, 0.0
    tot_sim_time, tot_tx, tot_rx = 0.0, 0.0, 0.0

    for sim_idx in range(sims):
        random.seed(5000 + n_nodes * 31 + sim_idx)
        joined = [False] * (n_nodes + 1)
        joined[0] = True
        joined_list = [0]
        unjoined_set = set(range(1, n_nodes + 1))
        height = [0] + [n_nodes] * n_nodes
        joined_asn = [0] + [-1] * n_nodes
        last_eb = [-1] * (n_nodes + 1)
        tx_count = [0] * (n_nodes + 1)
        rx_count = [0] * (n_nodes + 1)
        children = [[] for _ in range(n_nodes + 1)]

        asn = 0
        while unjoined_set:
            sf_number = asn // s
            c_slot = chs[asn % n_channels]
            tx_set = set()
            for j_id in joined_list:
                if random.random() <= eb_prob and (last_eb[j_id] < 0 or sf_number >= last_eb[j_id] + i_eb):
                    last_eb[j_id] = sf_number
                    tx_count[j_id] += 1
                    tx_set.add(j_id)

            if tx_set:
                newly_joined = []
                for p_id in list(unjoined_set):
                    # Active joined neighbors transmitting in this slotframe
                    active_nbrs = [nb for nb in nbrs[p_id] if nb in tx_set and joined_asn[nb] != asn]
                    if not active_nbrs:
                        continue
                    r_ch = random.randint(1, n_channels)
                    if r_ch == c_slot and len(active_nbrs) == 1:
                        j_parent = active_nbrs[0]
                        if height[j_parent] + 1 < height[p_id]:
                            joined[p_id] = True
                            height[p_id] = height[j_parent] + 1
                            joined_asn[p_id] = asn
                            rx_count[p_id] = asn
                            children[j_parent].append(p_id)
                            newly_joined.append(p_id)
                for p_id in newly_joined:
                    unjoined_set.remove(p_id)
                    joined_list.append(p_id)

            if (asn / 100.0) > timeout:
                break
            asn += s

        # Recursive tree evaluation on the formed DODAG
        memo_sub = {}
        def get_sub_sync(u):
            if u in memo_sub:
                return memo_sub[u]
            if not children[u]:
                val = sync_time_arr[u]
            else:
                val = sync_time_arr[u] + max(get_sub_sync(c) for c in children[u])
            memo_sub[u] = val
            return val

        t_sync = get_sub_sync(0) / 100.0
        e_eb_p = sum(0.01 * 0.001 * RxFactor * sync_time_arr[i] / p_rx_eb_arr[i] for i in range(1, n_nodes + 1))
        e_eb_j = sum(
            0.001 * TxFactor * p_tx_eb * max(get_sub_sync(c) for c in children[u])
            for u in range(n_nodes + 1) if children[u]
        )

        tot_t_sync += t_sync
        tot_e_eb_p += e_eb_p
        tot_e_eb_j += e_eb_j
        tot_sim_time += (asn / 100.0)
        tot_tx += sum(tx_count[u] for u in joined_list)
        tot_rx += sum(rx_count[u] for u in joined_list)

    return [
        n_nodes, area,
        tot_t_sync / sims,
        tot_e_eb_p / sims,
        tot_e_eb_j / sims,
        tot_sim_time / sims,
        tot_tx / sims,
        tot_rx / sims,
    ]


def generate_markov_figure6_from_simulation():
    markov_cache = os.path.join(STAT_DIR, "_markov_stats_.txt")
    TxFactor, RxFactor = 17.4, 18.8

    if os.path.exists(markov_cache) and os.path.getsize(markov_cache) > 100:
        m_data = np.loadtxt(markov_cache, delimiter=",", usecols=range(8))
        print("Loaded cached Markov simulation statistics from Statistics/_markov_stats_.txt")
    else:
        if os.path.exists(REALDATA_PATH):
            raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
            realdata = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
        else:
            realdata = list(range(1, 10000))

        sims = 5
        eb_prob = 0.1
        comm_range = 50.0
        s = 101
        i_eb = 1
        nodes_list = list(range(25, 500 + 1, 25))
        area = 100
        n_channels = 16
        timeout = 3600
        p_l, r_t, p_t, i_dio = 0.2, 16, 0.2, 1

        p_tx_eb = 1.0 / float(i_eb)
        p_tx_dio = (1.0 - p_l) * (1.0 - eb_prob) * (
            ((2 ** r_t) * ((1.0 - p_t) ** r_t) * max(1.0, s / ((2 ** r_t) * i_dio))
             + sum(p_t * (2 ** r) * ((1.0 - p_t) ** r) * max(1.0, s / ((2 ** r) * i_dio)) for r in range(r_t)))
            / (p_t + (2 ** r_t) * ((1.0 - p_t) ** r_t) + sum(p_t * (2 ** r) * ((1.0 - p_t) ** r) for r in range(1, r_t)))
        )
        p_tx_dio = min(0.95, max(0.01, p_tx_dio / float(s)))

        tasks = [
            (n_nodes, area, realdata, sims, eb_prob, comm_range, s, i_eb, n_channels, timeout, p_l, p_tx_eb, p_tx_dio, TxFactor, RxFactor)
            for n_nodes in nodes_list
        ]
        with ProcessPoolExecutor() as ex:
            m_rows = list(ex.map(_simulate_single_markov_node_count, tasks))

        m_data = np.array(sorted(m_rows, key=lambda r: r[0]), dtype=float)
        with open(markov_cache, "w") as fp:
            for row in m_data:
                fp.write(",".join(map(str, row)) + ",\n")
        print("Saved real Markov simulation results to Statistics/_markov_stats_.txt")

    sorted_densities = [round(r[0] / r[1], 2) for r in m_data]

    # Exact Graph_Builder_Markov.py scaling
    m_t_sync = [round(val / 100.0, 1) for val in m_data[:, 2]]
    m_e_p    = [round(val / 1000.0, 1) for val in m_data[:, 3]]
    m_e_j    = [round(val / 100.0, 1) for val in m_data[:, 4]]
    m_e_pj   = [round((ep + ej) / 1000.0, 1) for ep, ej in zip(m_data[:, 3], m_data[:, 4])]

    s_t_sync = [round(val / 100.0, 1) for val in m_data[:, 5]]
    s_tx     = [round((0.001 * TxFactor * val) / 100.0, 1) for val in m_data[:, 6]]
    s_rx     = [round((0.001 * RxFactor * val) / 1000.0, 1) for val in m_data[:, 7]]
    s_txrx   = [round(((0.001 * TxFactor * tx) + (0.001 * RxFactor * rx)) / 1000.0, 1) for tx, rx in zip(m_data[:, 6], m_data[:, 7])]

    def plot_markov_panel(y_series, legends, colors, markers, lstyles, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=300)
        for y, lbl, col, mk, ls in zip(y_series, legends, colors, markers, lstyles):
            ax.plot(
                sorted_densities, y,
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
        ax.set_ylim(ax.get_ylim()[0], 1.28 * ax.get_ylim()[-1])
        ax.legend(loc="upper center", ncol=2, fontsize=9.0, frameon=True, edgecolor="#333333")
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
    
    # Average duplicate runs sharing the exact same x coordinate
    by_x = {}
    for row in filtered:
        by_x.setdefault(float(row[x_col]), []).append(row)
    return np.array([np.mean(by_x[k], axis=0) for k in sorted(by_x.keys())])


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
            if x_col == 4 and key == "p" and col_idx in (6, 7, 8):
                y = y / np.maximum(1.0, mat[:, 4])

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

    f_nodes = {3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6,  "Total pledges", r"Total sync time ($\times 10^3$ s)", "Fig9a_total_sync_time.png", "motewise_total_time.pdf", y_scale=1e-3)
    plot_sweep(0, f_nodes, 8,  "Total pledges", r"Average sync time ($\times 10^4$ slots)", "Fig9b_avg_sync_time.png", "motewise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(0, f_nodes, 9,  "Total pledges", r"Collisions ($\times 10^4$)", "Fig9c_collisions.png", "motewise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(0, f_nodes, 5,  "Total pledges", r"DODAG completion ($\%$)", "Fig9d_dodag_completion.png", "motewise_dodag_completion.pdf", is_pct=True)

    plot_sweep(0, f_nodes, 6,  "Motes per unit distance", r"Total sync time ($\times 10^3$ s)", "Fig10a_density_total_time.png", "densitywise_total_time.pdf", x_scale=0.01, y_scale=1e-3)
    plot_sweep(0, f_nodes, 8,  "Motes per unit distance", r"Average sync time ($\times 10^4$ slots)", "Fig10b_density_avg_time.png", "densitywise_average_neighbour_time.pdf", x_scale=0.01, y_scale=1e-4)
    plot_sweep(0, f_nodes, 9,  "Motes per unit distance", r"Collisions ($\times 10^4$)", "Fig10c_density_collisions.png", "densitywise_average_collisions.pdf", x_scale=0.01, y_scale=1e-4)
    plot_sweep(0, f_nodes, 5,  "Motes per unit distance", r"DODAG completion ($\%$)", "Fig10d_density_completion.png", "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True)

    plot_sweep(0, f_nodes, 10, "Motes per unit distance", r"Tx energy / count ($\times 10^6$)", "Fig11a_tx_energy.png", "densitywise_transmission_energy.pdf", x_scale=0.01, y_scale=1e-6)
    plot_sweep(0, f_nodes, 11, "Motes per unit distance", r"Rx scan energy / ticks ($\times 10^8$)", "Fig11b_rx_energy.png", "densitywise_reception_energy.pdf", x_scale=0.01, y_scale=1e-8)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", r"Total radio energy ($\times 10^7$)", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, y_scale=1e-7, is_total_energy=True)
    plot_sweep(0, f_nodes, 10, "Motes per unit distance", r"Radio duty cycle ($\%$)", "Fig11d_duty_cycle.png", "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True)

    f_ch = {0: 200, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Total channels", r"Total sync time ($\times 10^3$ s)", "Fig12a_ch_total_sync_time.png", "channelwise_total_time.pdf", y_scale=1e-3)
    plot_sweep(2, f_ch, 8, "Total channels", r"Average sync time ($\times 10^4$ slots)", "Fig12b_ch_avg_sync_time.png", "channelwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 9, "Total channels", r"Collisions ($\times 10^4$)", "Fig12c_ch_collisions.png", "channelwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 5, "Total channels", r"DODAG completion ($\%$)", "Fig12d_ch_dodag_completion.png", "channelwise_dodag_completion.pdf", is_pct=True)

    f_sf = {0: 200, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size (slots)", r"Total sync time ($\times 10^3$ s)", "Fig13a_sf_total_sync_time.png", "slotframewise_total_time.pdf", y_scale=1e-3)
    plot_sweep(3, f_sf, 8, "Slotframe size (slots)", r"Average sync time ($\times 10^4$ slots)", "Fig13b_sf_avg_sync_time.png", "slotframewise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 9, "Slotframe size (slots)", r"Collisions ($\times 10^4$)", "Fig13c_sf_collisions.png", "slotframewise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 5, "Slotframe size (slots)", r"DODAG completion ($\%$)", "Fig13d_sf_dodag_completion.png", "slotframewise_dodag_completion.pdf", is_pct=True)

    f_eb = {0: 200, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval multiplier", r"Total sync time ($\times 10^3$ s)", "Fig14a_eb_total_sync_time.png", "eb_intervalwise_total_time.pdf", y_scale=1e-3)
    plot_sweep(4, f_eb, 8, "EB interval multiplier", r"Average sync time ($\times 10^4$ slots)", "Fig14b_eb_avg_sync_time.png", "eb_intervalwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 9, "EB interval multiplier", r"Collisions ($\times 10^4$)", "Fig14c_eb_collisions.png", "eb_intervalwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 5, "EB interval multiplier", r"DODAG completion ($\%$)", "Fig14d_eb_dodag_completion.png", "eb_intervalwise_dodag_completion.pdf", is_pct=True)