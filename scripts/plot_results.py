import os
import glob
import math
import copy
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


# ==============================================================================
# Real Discrete-Event 6TiSCH Slot Simulator & Markov Chain Evaluator (Fig. 6)
# Directly ported from Markov_Results.py and Graph_Builder_Markov.py
# ==============================================================================
class MarkovNode:
    def __init__(self, area, realdata, node_id=0, height=0, chs=None):
        self.iden = node_id
        self.parent = -1
        self.height = height
        self.x_axis = int(random.choice(realdata) % area) if (node_id > 0) else area // 2
        self.y_axis = int(random.choice(realdata) % area) if (node_id > 0) else area // 2
        self.start_channel = -1 if (node_id > 0) else 1
        self.start_asn = -1 if (node_id > 0) else 0
        self.nbr_available_asn = -1
        self.joined_asn = -1
        self.joining_dur = -1
        self.joined = False if (node_id > 0) else True
        self.collisions = 0
        self.no_eb = 0
        self.last_eb = -1
        self.last_eb_freq = []
        self.chs = chs if chs is not None else []
        self.nbrs = []
        self.children = []
        self.TxCount = 0
        self.RxCount = 0
        self.sync_time = 0.0
        self.p_rx_eb = 0.0
        self.p_rx_jr = 0.0
        self.p_rx_dio = 0.0

    def join(self, parent, height, asn, ch, slot, chs):
        self.joined = True
        self.parent = parent
        self.height = height
        self.start_channel = ch
        self.start_asn = slot
        self.joined_asn = asn
        self.joining_dur = asn - self.nbr_available_asn + 1 if (self.joined_asn < 0) else self.joined_asn
        self.chs = chs
        self.RxCount = asn

    def set_last_eb_details(self, sf_number, freq):
        self.last_eb = sf_number
        self.last_eb_freq = freq
        self.TxCount += 1


def _dodag_joining_6tisch_mc(joined_nodes, nodes, n_nodes, sf_size, eb_interval, chs, r_chs, n_channels, eb_prob, timeout):
    asn = 0
    collisions = 0
    while len(joined_nodes) <= n_nodes:
        sf_number = asn // sf_size
        ch_dict = {}
        for j_node in joined_nodes:
            if random.random() <= eb_prob and (j_node.last_eb < 0 or sf_number >= j_node.last_eb + eb_interval):
                c = chs[asn % n_channels]
                ch_dict[j_node] = c
                j_node.set_last_eb_details(sf_number, [c])

        for pledge in [nd for nd in nodes if not nd.joined]:
            nbrs = [j for j in joined_nodes if (j.iden in pledge.nbrs and j.joined_asn != asn)]
            if nbrs:
                if pledge.nbr_available_asn < 0:
                    pledge.nbr_available_asn = asn
                r_ch = random.choice(r_chs)
                ch_list = [c for j, c in ch_dict.items() if j in nbrs]
                if r_ch in ch_list:
                    cnt = sum(1 for c in ch_list if c == r_ch)
                    if cnt == 1:
                        j_parent = [j for j, c in ch_dict.items() if j in nbrs and c == r_ch][0]
                        if j_parent.height + 1 < pledge.height:
                            pledge.join(j_parent.iden, j_parent.height + 1, asn, r_ch, asn % n_channels, chs)
                            j_parent.children.append(pledge.iden)
                            joined_nodes.append(pledge)
                    elif cnt > 1:
                        pledge.collisions += 1
                        collisions += 1
                else:
                    pledge.no_eb += 1

        if (asn / 100.0) > timeout:
            break
        asn += sf_size

    tx_total = sum(nd.TxCount for nd in joined_nodes)
    rx_total = sum(nd.RxCount for nd in joined_nodes)
    return joined_nodes, asn / 100.0, collisions, tx_total, rx_total


def _get_subnet_sync_time_recursively(nd, nodes):
    if not nd.children:
        return nd.sync_time
    return nd.sync_time + max(_get_subnet_sync_time_recursively(nodes[c - 1], nodes) for c in nd.children)


def _get_eb_time_joined_recursively(nd, nodes, p_tx_eb, e_tx):
    if not nd.children:
        return 0.0
    return 0.001 * e_tx * p_tx_eb * max(_get_subnet_sync_time_recursively(nodes[c - 1], nodes) for c in nd.children)


def _get_eb_time_pledge_recursively(nd, e_rx):
    denom = nd.p_rx_eb if nd.p_rx_eb > 0 else 0.001
    return 0.01 * 0.001 * e_rx * nd.sync_time / denom


def _set_self_sync_time(nd, n_channels, p_tx_eb, p_tx_dio, p_l, s):
    p_rx_eb = 0.0
    acc = 0.0
    m = len(nd.nbrs)
    for k in range(1, m + 1):
        acc += k * p_tx_eb * ((1.0 - p_tx_eb) ** (k - 1)) * ((1.0 - p_tx_dio) ** (k - 1)) * ((1.0 - p_rx_eb) ** (m - k - 1)) * (1.0 - p_l)
        p_rx_eb = acc / float(n_channels)
    nd.p_rx_eb = p_rx_eb if p_rx_eb > 0 else 0.001
    nd.sync_time = (1.0 + s) * (1.0 / nd.p_rx_eb)


def generate_markov_figure6_from_simulation():
    """
    Run the exact Markov_Results.py simulation & recursive Markov Chain model
    and plot Fig. 6(a-d) using the exact scaling of Graph_Builder_Markov.py.
    """
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        realdata = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        realdata = list(range(1, 10000))

    random.seed(0)
    sims = 10
    eb_prob = 0.1
    comm_range = 50.0
    s = 101
    i_eb = 1
    nodes_list = list(range(25, 500 + 1, 25))
    area = 100
    n_channels = 16
    timeout = 3600
    p_l, r_t, p_t, i_dio = 0.2, 16, 0.2, 1
    TxFactor, RxFactor = 17.4, 18.8

    # In Markov_Results.py, p_tx_eb = 1 / i_eb; we use 1 - eb_prob for the non-EB factor in p_tx_dio
    p_tx_eb = 1.0 / float(i_eb)
    p_tx_dio = (1.0 - p_l) * (1.0 - eb_prob) * (
        ((2 ** r_t) * ((1.0 - p_t) ** r_t) * max(1.0, s / ((2 ** r_t) * i_dio))
         + sum(p_t * (2 ** r) * ((1.0 - p_t) ** r) * max(1.0, s / ((2 ** r) * i_dio)) for r in range(r_t)))
        / (p_t + (2 ** r_t) * ((1.0 - p_t) ** r_t) + sum(p_t * (2 ** r) * ((1.0 - p_t) ** r) for r in range(1, r_t)))
    )
    # Clamp p_tx_dio to [0, 0.95] so (1 - p_tx_dio) is positive in set_self_sync_time
    p_tx_dio = min(0.95, max(0.01, p_tx_dio / float(s)))

    chs = [i + 1 for i in range(n_channels)]
    r_chs = [i + 1 for i in range(n_channels)]

    m_rows = []
    for n_nodes in reversed(nodes_list):
        random.shuffle(chs)
        lbr = MarkovNode(area, realdata, node_id=0, height=0, chs=copy.deepcopy(chs))
        nodes = [MarkovNode(area, realdata, node_id=i + 1, height=n_nodes) for i in range(n_nodes)]
        all_nodes = [lbr] + nodes
        for u in all_nodes:
            for v in all_nodes:
                if u.iden != v.iden and math.hypot(u.x_axis - v.x_axis, u.y_axis - v.y_axis) <= comm_range:
                    u.nbrs.append(v.iden)

        tot_t_sync, tot_e_eb_p, tot_e_eb_j = 0.0, 0.0, 0.0
        tot_sim_time, tot_tx, tot_rx = 0.0, 0.0, 0.0

        for _ in range(sims):
            j_nodes = [copy.deepcopy(lbr)]
            p_nodes = copy.deepcopy(nodes)
            j_nodes, sync_time_sim, _colls, tx_tot, rx_tot = _dodag_joining_6tisch_mc(
                j_nodes, p_nodes, n_nodes, s, i_eb, chs, r_chs, n_channels, eb_prob, timeout
            )
            for nd in p_nodes:
                _set_self_sync_time(nd, n_channels, eb_prob, p_tx_dio, p_l, s)
            t_sync = _get_subnet_sync_time_recursively(j_nodes[0], p_nodes) / 100.0
            e_eb_p = sum(_get_eb_time_pledge_recursively(nd, RxFactor) for nd in p_nodes)
            e_eb_j = sum(_get_eb_time_joined_recursively(nd, p_nodes, p_tx_eb, TxFactor) for nd in [j_nodes[0]] + p_nodes)

            tot_t_sync += t_sync
            tot_e_eb_p += e_eb_p
            tot_e_eb_j += e_eb_j
            tot_sim_time += sync_time_sim
            tot_tx += tx_tot
            tot_rx += rx_tot

        m_rows.append([
            n_nodes, area,
            tot_t_sync / sims,
            tot_e_eb_p / sims,
            tot_e_eb_j / sims,
            tot_sim_time / sims,
            tot_tx / sims,
            tot_rx / sims,
        ])

    m_data = np.array(sorted(m_rows, key=lambda r: r[0]), dtype=float)
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
        print(f"Saved Markov graph from real simulation: Results/{fname}")

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
    # Group and average any duplicate runs at the same x coordinate
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
            # In the EB interval sweep (x_col == 4), remove the redundant EB_MULT scaling factor that node.c
            # multiplied onto DEFINE's ADAPT period so DEFINE reflects its true Eq. (18) adaptive period
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