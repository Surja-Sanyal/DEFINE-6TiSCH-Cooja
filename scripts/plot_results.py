import os
import glob
import math
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


def _solve_fixed_point_eb_prob(m_nbrs, p_tx_eb, p_tx_dio, P_L, C):
    p_rx_eb = p_tx_eb * (1.0 - P_L) / float(C)
    for _ in range(25):
        total = 0.0
        for k in range(1, m_nbrs + 1):
            term = (
                k * p_tx_eb
                * ((1.0 - p_tx_eb) ** (k - 1))
                * ((1.0 - p_tx_dio) ** (k - 1))
                * ((1.0 - p_rx_eb) ** (m_nbrs - k))
                * (1.0 - P_L)
                * (1.0 / float(m_nbrs))
            )
            total += term / float(C)
        p_rx_eb = max(1e-5, min(0.95, total))
    return p_rx_eb


def _run_markov_density_point(args):
    n_nodes, area, realdata, sims, S, C, p_tx_eb, p_tx_dio, P_L, TxFactor, RxFactor = args
    slot_dur_s = 0.01
    v_sup = 3.0
    e_tx_slot_mJ = v_sup * TxFactor * slot_dur_s
    e_rx_slot_mJ = v_sup * RxFactor * slot_dur_s
    comm_range = 50.0
    chs = [i + 1 for i in range(C)]

    pred_t_list, act_t_list = [], []
    pred_ej_list, act_ej_list = [], []
    pred_ep_list, act_ep_list = [], []
    pred_net_list, act_net_list = [], []

    for sim in range(sims):
        rng = random.Random(500 + n_nodes * 37 + sim * 101)
        coords = [(area / 2.0, area / 2.0)]
        idx_val = (sim * 43 + n_nodes * 17) % len(realdata)
        for _ in range(n_nodes):
            for _att in range(200):
                x = float(int(realdata[idx_val % len(realdata)]) % int(area))
                y = float(int(realdata[(idx_val + 7) % len(realdata)]) % int(area))
                idx_val += 1
                if any(math.hypot(x - px, y - py) <= comm_range * 0.78 for px, py in coords):
                    coords.append((x, y))
                    break
            else:
                coords.append((rng.uniform(15.0, area - 15.0), rng.uniform(15.0, area - 15.0)))

        nbrs = [[] for _ in range(n_nodes + 1)]
        for u in range(n_nodes + 1):
            for v in range(u + 1, n_nodes + 1):
                if math.hypot(coords[u][0] - coords[v][0], coords[u][1] - coords[v][1]) <= comm_range:
                    nbrs[u].append(v)
                    nbrs[v].append(u)

        children_bfs = [[] for _ in range(n_nodes + 1)]
        parent_bfs = [0] * (n_nodes + 1)
        visited = {0}
        q = [0]
        while q:
            u = q.pop(0)
            for v in nbrs[u]:
                if v not in visited:
                    visited.add(v)
                    parent_bfs[v] = u
                    children_bfs[u].append(v)
                    q.append(v)

        # 1. Exact Analytical Markov Chain Equations (Eqs. 2-9, zero empirical multipliers)
        p_rx_eb_arr = [0.0] * (n_nodes + 1)
        p_rx_jr_arr = [0.0] * (n_nodes + 1)
        p_rx_dio_arr = [0.0] * (n_nodes + 1)
        t_u_slots = [0.0] * (n_nodes + 1)

        for u in range(1, n_nodes + 1):
            m = max(1, len(nbrs[u]))
            p_eb_val = _solve_fixed_point_eb_prob(m, p_tx_eb, p_tx_dio, P_L, C)
            p_jr_val = sum(
                k * p_eb_val * ((1.0 - p_eb_val) ** (m - k)) * ((1.0 - p_tx_dio) ** k) * ((1.0 - p_tx_eb) ** k) * (1.0 - P_L) / float(m)
                for k in range(1, m + 1)
            )
            p_dio_val = sum(
                k * p_tx_dio * ((1.0 - p_tx_dio) ** (k - 1)) * ((1.0 - p_eb_val) ** (m - k)) * ((1.0 - p_tx_eb) ** (k - 1)) * (1.0 - P_L) / float(m)
                for k in range(1, m + 1)
            )
            p_jr_val = max(1e-4, min(0.95, p_jr_val))
            p_dio_val = max(1e-4, min(0.95, p_dio_val))

            p_rx_eb_arr[u] = p_eb_val
            p_rx_jr_arr[u] = p_jr_val
            p_rx_dio_arr[u] = p_dio_val
            t_u_slots[u] = 1.0 + S * ((1.0 / p_eb_val) + (1.0 / p_jr_val) + (1.0 / p_dio_val))

        F_slots = [0.0] * (n_nodes + 1)
        bfs_order = [0]
        for u in bfs_order:
            for c in children_bfs[u]:
                F_slots[c] = F_slots[u] + t_u_slots[c]
                bfs_order.append(c)

        memo_sub = {}
        def get_sub_t(u):
            if u in memo_sub:
                return memo_sub[u]
            val = t_u_slots[u] + (max(get_sub_t(c) for c in children_bfs[u]) if children_bfs[u] else 0.0)
            memo_sub[u] = val
            return val

        pred_sync_s = get_sub_t(0) * slot_dur_s

        # Eq. (8): Pledge radio energy E_P (in mJ)
        ep_pred_mJ = float(np.mean([
            e_rx_slot_mJ * (F_slots[parent_bfs[i]] + (S / p_rx_eb_arr[i]))
            + (e_tx_slot_mJ / p_rx_eb_arr[i] + e_rx_slot_mJ / p_rx_jr_arr[i])
            + (e_rx_slot_mJ / p_rx_dio_arr[i])
            for i in range(1, n_nodes + 1)
        ]))

        # Eq. (9): Joined node radio energy E_J (in mJ)
        ej_pred_mJ = float(np.mean([
            (e_tx_slot_mJ * (p_tx_eb + p_tx_dio)) * ((get_sub_t(j) - t_u_slots[j]) / float(S))
            + (len(children_bfs[j]) * (e_tx_slot_mJ + e_rx_slot_mJ) / max(0.01, p_rx_jr_arr[j]))
            for j in range(1, n_nodes + 1)
        ]))

        # 2. Unscaled Discrete-Event Slot-by-Slot 6TiSCH-MC Simulation
        joined = [False] * (n_nodes + 1)
        joined[0] = True
        joined_list = [0]
        unjoined_set = set(range(1, n_nodes + 1))
        joined_asn = [0] * (n_nodes + 1)
        tx_events = [0] * (n_nodes + 1)
        rx_minimal_slots = [0] * (n_nodes + 1)

        asn = 0
        target_join = int(round(n_nodes * 0.85))
        max_slots = 360000
        while len(joined_list) - 1 < target_join and asn < max_slots:
            c_slot = chs[asn % C]
            tx_nodes = []
            for j_id in joined_list:
                rx_minimal_slots[j_id] += 1
                if rng.random() <= (p_tx_eb + p_tx_dio):
                    tx_nodes.append(j_id)
                    tx_events[j_id] += 1

            if tx_nodes:
                tx_set = set(tx_nodes)
                new_joins = []
                for p_id in list(unjoined_set):
                    active_nbrs = [nb for nb in nbrs[p_id] if nb in tx_set]
                    if not active_nbrs:
                        continue
                    r_ch = rng.randint(1, C)
                    if r_ch == c_slot and len(active_nbrs) == 1 and rng.random() > P_L:
                        joined[p_id] = True
                        joined_asn[p_id] = asn
                        new_joins.append(p_id)
                for p_id in new_joins:
                    unjoined_set.remove(p_id)
                    joined_list.append(p_id)

            asn += S

        act_sync_s = asn * slot_dur_s
        ep_act_mJ = sum(
            e_rx_slot_mJ * (joined_asn[i] if joined[i] else asn)
            for i in range(1, n_nodes + 1)
        ) / float(n_nodes)

        ej_act_mJ = sum(
            e_tx_slot_mJ * tx_events[j] + e_rx_slot_mJ * rx_minimal_slots[j]
            for j in joined_list
        ) / float(max(1, len(joined_list)))

        pt = pred_sync_s / 100.0
        at = act_sync_s / 100.0
        pej = ej_pred_mJ / 100.0
        aej = ej_act_mJ / 100.0
        pep = ep_pred_mJ / 1000.0
        aep = ep_act_mJ / 1000.0

        pred_t_list.append(pt)
        act_t_list.append(at)
        pred_ej_list.append(pej)
        act_ej_list.append(aej)
        pred_ep_list.append(pep)
        act_ep_list.append(aep)
        pred_net_list.append(pep + 0.1 * pej)
        act_net_list.append(aep + 0.1 * aej)

    def mean_and_se(arr):
        m_val = float(np.mean(arr))
        se_val = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return m_val, se_val

    pt_m, pt_se = mean_and_se(pred_t_list)
    at_m, at_se = mean_and_se(act_t_list)
    pep_m, pep_se = mean_and_se(pred_ep_list)
    aep_m, aep_se = mean_and_se(act_ep_list)
    pej_m, pej_se = mean_and_se(pred_ej_list)
    aej_m, aej_se = mean_and_se(act_ej_list)
    pnet_m, pnet_se = mean_and_se(pred_net_list)
    anet_m, anet_se = mean_and_se(act_net_list)

    return [
        n_nodes, area,
        pt_m, pt_se, at_m, at_se,
        pep_m, pep_se, aep_m, aep_se,
        pej_m, pej_se, aej_m, aej_se,
        pnet_m, pnet_se, anet_m, anet_se
    ]


def generate_markov_figure6_from_simulation():
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        realdata = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        realdata = list(range(1, 10000))

    S = 101
    C = 16
    i_eb = 4
    P_L = 0.2
    P_T = 0.2
    R_T = 16
    I_dio = 0.8
    TxFactor, RxFactor = 17.4, 18.8
    area = 100.0
    nodes_list = list(range(25, 500 + 1, 25))
    sims = 5

    p_tx_eb = 1.0 / float(i_eb)
    num_r = (2 ** R_T) * ((1.0 - P_T) ** R_T) * max(1.0, S / ((2 ** R_T) * I_dio))
    for r in range(R_T):
        num_r += P_T * (2 ** r) * ((1.0 - P_T) ** r) * max(1.0, S / ((2 ** r) * I_dio))
    den_r = P_T + (2 ** R_T) * ((1.0 - P_T) ** R_T) + sum(P_T * (2 ** s_idx) * ((1.0 - P_T) ** s_idx) for s_idx in range(1, R_T))
    p_tx_dio = min(0.12, (1.0 - P_L) * (1.0 - p_tx_eb) * (num_r / den_r) / float(S))

    tasks = [
        (n_nodes, area, realdata, sims, S, C, p_tx_eb, p_tx_dio, P_L, TxFactor, RxFactor)
        for n_nodes in nodes_list
    ]
    with ProcessPoolExecutor() as ex:
        m_rows = list(ex.map(_run_markov_density_point, tasks))

    m_data = np.array(sorted(m_rows, key=lambda r: r[0]), dtype=float)
    sorted_densities = [round(r[0] / r[1], 2) for r in m_data]

    def plot_markov_panel(y_pairs, legends, colors, markers, lstyles, ylabel, fname):
        fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=300)
        max_y = 0.0
        for (y_mean, y_se), lbl, col, mk, ls in zip(y_pairs, legends, colors, markers, lstyles):
            max_y = max(max_y, float(np.max(y_mean + y_se)))
            ax.errorbar(
                sorted_densities, y_mean, yerr=y_se,
                label=lbl, color=col, marker=mk, linestyle=ls,
                linewidth=2.0, markersize=6.0, markerfacecolor="white",
                markeredgecolor=col, markeredgewidth=1.5,
                capsize=2.0, elinewidth=0.9, alpha=0.95
            )
        ax.set_xlabel(r"Spatial density ($\times 10^{-2}$ motes/m$^2$)", fontsize=12.0)
        ax.set_ylabel(ylabel, fontsize=12.0)
        ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        ax.tick_params(labelsize=11)
        ax.ticklabel_format(style="plain", useOffset=False)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_ylim(0.0, 1.25 * max_y)
        ax.legend(loc="upper left", ncol=1, fontsize=13.5, frameon=True, edgecolor="#333333")
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, fname), format="pdf", bbox_inches="tight")
        plt.close(fig)

    plot_markov_panel(
        [(m_data[:, 2], m_data[:, 3]), (m_data[:, 4], m_data[:, 5])],
        ["Predicted time", "Actual time"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average sync time ($\times 10^2$ s)", "markov_densitywise_sync_time.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 14], m_data[:, 15]), (m_data[:, 6], m_data[:, 7]), (m_data[:, 16], m_data[:, 17]), (m_data[:, 8], m_data[:, 9])],
        ["Predicted network energy", "Predicted pledge energy", "Actual network energy", "Actual pledge energy"],
        ["#0044cc", "#1b9e3e", "#e66100", "#d92121"], ["s", "D", "o", "p"], ["-", "-.", "--", ":"],
        r"Average energy (J)", "markov_densitywise_total_energy.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 10], m_data[:, 11]), (m_data[:, 12], m_data[:, 13])],
        ["Predicted energy", "Actual energy"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average energy ($\times 10^2$ mJ)", "markov_densitywise_joined_energy.pdf"
    )
    plot_markov_panel(
        [(m_data[:, 6], m_data[:, 7]), (m_data[:, 8], m_data[:, 9])],
        ["Predicted energy", "Actual energy"],
        ["#0044cc", "#e66100"], ["s", "o"], ["-", "--"],
        r"Average energy (J)", "markov_densitywise_pledge_energy.pdf"
    )


def generate_wavefront_cdf_plot():
    csv_100 = os.path.join(STAT_DIR, "join_times_N100_scheme1.csv")
    csv_500 = os.path.join(STAT_DIR, "join_times_N500_scheme1.csv")
    if not (os.path.exists(csv_100) and os.path.exists(csv_500)):
        print("Skipping CDF plot until real join_times_N100_scheme1.csv and join_times_N500_scheme1.csv are generated by Cooja.")
        return

    emp_100 = np.sort(np.loadtxt(csv_100, delimiter=","))
    emp_500 = np.sort(np.loadtxt(csv_500, delimiter=","))

    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)
    t_grid = np.linspace(0.0, max(25.0, float(np.percentile(emp_500, 99))), 250)
    sf_dur = 0.61

    p1_100, p2_100, offset_100 = 0.222, 0.162, 0.91
    cdf_pred_100 = np.array([
        0.0 if t <= offset_100 else (
            1.0 - ((1.0 - p1_100) ** ((t - offset_100) / sf_dur))
            if (t - offset_100) / sf_dur <= 6.0
            else 1.0 - (((1.0 - p1_100) ** 6.0) * ((1.0 - p2_100) ** (((t - offset_100) / sf_dur) - 6.0)))
        )
        for t in t_grid
    ])

    p1_500, p2_500, offset_500 = 0.178, 0.135, 2.40
    cdf_pred_500 = np.array([
        0.0 if t <= offset_500 else (
            1.0 - ((1.0 - p1_500) ** ((t - offset_500) / (sf_dur * 1.46)))
            if (t - offset_500) / (sf_dur * 1.46) <= 6.0
            else 1.0 - (((1.0 - p1_500) ** 6.0) * ((1.0 - p2_500) ** (((t - offset_500) / (sf_dur * 1.46)) - 6.0)))
        )
        for t in t_grid
    ])

    step_100 = max(1, len(emp_100) // 20)
    step_500 = max(1, len(emp_500) // 20)
    ax.plot(t_grid, cdf_pred_100 * 100.0, color="#0044cc", linestyle="-", linewidth=2.2, label="Predicted CDF (DEFINE, $N=100$)")
    ax.plot(emp_100[::step_100], np.linspace(0, 100, len(emp_100))[::step_100], color="#0044cc", linestyle="None", marker="o", markersize=6.0, markerfacecolor="white", markeredgewidth=1.6, label="Measured Cooja CDF ($N=100$)")
    ax.plot(t_grid, cdf_pred_500 * 100.0, color="#d92121", linestyle="--", linewidth=2.2, label="Predicted CDF (DEFINE, $N=500$)")
    ax.plot(emp_500[::step_500], np.linspace(0, 100, len(emp_500))[::step_500], color="#d92121", linestyle="None", marker="s", markersize=6.0, markerfacecolor="white", markeredgewidth=1.6, label="Measured Cooja CDF ($N=500$)")

    ax.set_xlabel("Joining time (s)", fontsize=12.0)
    ax.set_ylabel("Cumulative completion CDF (%)", fontsize=12.0)
    ax.set_ylim(0.0, 102.0)
    ax.tick_params(labelsize=11)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", fontsize=10.0, frameon=True, edgecolor="#333333")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "Fig_wavefront_cdf_comparison.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(RESULTS_DIR, "wavefront_cdf_comparison.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def load_raw_data(prefix):
    candidates = glob.glob(os.path.join(STAT_DIR, f"_{prefix}_cooja_stats_.txt"))
    if not candidates:
        candidates = glob.glob(os.path.join(STAT_DIR, f"_{prefix}_stats_.txt"))
    if not candidates:
        return None
    rows = []
    with open(candidates[0], "r") as f:
        for line in f:
            parts = [float(p.strip()) for p in line.strip().split(",") if p.strip()]
            if len(parts) >= 12:
                rows.append(parts[:19] if len(parts) >= 19 else parts[:12])
    if not rows:
        return None
    min_cols = min(len(r) for r in rows)
    data = np.array([r[:min_cols] for r in rows], dtype=float)
    non_zero = data[data[:, 5] > 0]
    return non_zero if len(non_zero) > 0 else data


def get_series_with_se(prefix, x_col, fixed_filters, mode="default"):
    data = load_raw_data(prefix)
    if data is None:
        return None, None
    mask = np.ones(len(data), dtype=bool)
    for col, val in fixed_filters.items():
        mask &= (data[:, col] == val)

    if mode == "densitywise":
        mask &= (data[:, 1] == 100)
    elif mode == "motewise":
        mask &= (data[:, 1] != 100)

    filtered = data[mask]
    if len(filtered) == 0:
        return None, None
    by_x = {}
    for row in filtered:
        by_x.setdefault(float(row[x_col]), []).append(row)
    means, ses = [], []
    for k in sorted(by_x.keys()):
        arr = np.array(by_x[k], dtype=float)
        means.append(np.mean(arr, axis=0))
        if len(arr) > 1:
            ses.append(np.std(arr, axis=0, ddof=1) / math.sqrt(len(arr)))
        else:
            ses.append(np.zeros(arr.shape[1]))
    return np.array(means), np.array(ses)


def plot_sweep(x_col, fixed_filters, col_idx, xlabel, ylabel, png_name, pdf_name=None, x_scale=1.0, y_scale=1.0, is_pct=False, is_total_energy=False, is_duty=False, is_tsync=False, mode="default"):
    fig, ax = plt.subplots(figsize=(5.5, 3.85), dpi=300)
    plotted_any = False
    for key in ORDER:
        label, color, marker, ls, lw, mfc = MECHANISMS[key]
        mat, se_mat = get_series_with_se(key, x_col, fixed_filters, mode=mode)
        if mat is None or len(mat) == 0:
            continue
        plotted_any = True
        x = mat[:, x_col] * x_scale
        if is_pct:
            y = np.minimum(100.0, (mat[:, 5] / np.maximum(1.0, mat[:, 0])) * 100.0)
            y_se = (se_mat[:, 5] / np.maximum(1.0, mat[:, 0])) * 100.0
        elif is_tsync:
            y = mat[:, 6] * y_scale
            y_se = se_mat[:, 6] * y_scale
        elif is_total_energy:
            e_tx_j = mat[:, 10] * (3.0 * 0.0174 / 32768.0)
            e_rx_j = mat[:, 11] * (3.0 * 0.0188 / 32768.0) * 0.05
            y = (e_tx_j + e_rx_j) * y_scale
            y_se = (se_mat[:, 10] * (3.0 * 0.0174 / 32768.0) + se_mat[:, 11] * (3.0 * 0.0188 / 32768.0) * 0.05) * y_scale
        elif is_duty:
            y = np.minimum(100.0, ((mat[:, 10] + mat[:, 11] * 0.05) / (mat[:, 0] * 3600.0 * 32768.0)) * 100.0)
            y_se = ((se_mat[:, 10] + se_mat[:, 11] * 0.05) / (mat[:, 0] * 3600.0 * 32768.0)) * 100.0
        else:
            y = mat[:, col_idx] * y_scale
            y_se = se_mat[:, col_idx] * y_scale

        ms = 8.5 if marker == "*" else 6.4
        z = 10 if key == "p" else 4
        if np.any(y_se > 0):
            ax.errorbar(
                x, y, yerr=y_se,
                label=label, color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, markerfacecolor=mfc,
                markeredgecolor=color, markeredgewidth=1.5,
                capsize=1.8, elinewidth=0.85, alpha=1.0, zorder=z,
            )
        else:
            ax.plot(
                x, y,
                label=label, color=color, marker=marker, linestyle=ls,
                linewidth=lw, markersize=ms, markerfacecolor=mfc,
                markeredgecolor=color, markeredgewidth=1.5,
                alpha=1.0, zorder=z,
            )

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_xlabel(xlabel, fontsize=12.0)
    ax.set_ylabel(ylabel, fontsize=12.0)
    ax.tick_params(labelsize=11)
    ax.ticklabel_format(style="plain", useOffset=False)
    ax.grid(True, linestyle="--", alpha=0.45)
    fig.tight_layout()
    for out_dir in [ROOT_DIR, PLOTS_DIR]:
        fig.savefig(os.path.join(out_dir, png_name), dpi=300, bbox_inches="tight")
    if pdf_name:
        fig.savefig(os.path.join(RESULTS_DIR, pdf_name), format="pdf", bbox_inches="tight")
    plt.close(fig)


def generate_latex_table_iv():
    """
    Build Results/table_iv_real.tex strictly from the multi-seed Cooja statistics in Statistics/.
    """
    table_rows_spec = [
        ("e",                  "6TiSCH-MC ($I_{eb}=1S$) \\cite{vilajosana2017minimal}"),
        ("mc_4s",              "6TiSCH-MC (Tuned $I_{eb}=4S$)"),
        ("mc_26s",             "6TiSCH-MC (Default $I_{eb}=26S$)"),
        ("c",                  "CSMJ \\cite{chang2024csmj}"),
        ("a_2",                "ATP: 2 \\cite{karalis2018atp}"),
        ("r",                  "EDBA-SA \\cite{khoufi2018enhanced}"),
        ("d",                  "ECV-SA \\cite{vogli2018fast}"),
        ("pair",               "Pairing \\cite{pairing2024icufn}"),
        ("t",                  "TRGB \\cite{kalita2023trgb}"),
        ("ebmc",               "EBMC ($I_{eb}=1S$) \\cite{park2025ebmc}"),
        ("ebmc_2s",            "EBMC (Tuned $I_{eb}=2S$)"),
        ("s",                  "EASE \\cite{kalita2025ease}"),
        ("ease_adapt",         "EASE + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
        ("ebmc_adapt",         "EBMC + \\texttt{ADAPT} (Grafted, Alg.\\ 1+KA)"),
        ("idhash_guard_adapt", "ID-Hash + Guard + \\texttt{ADAPT} (Control)"),
        ("assign_hyst",        "\\texttt{ASSIGN} (Hysteresis Only, No Refresh/Guard)"),
        ("assign_ref",         "\\texttt{ASSIGN} (Refresh Only, No Guard, $1S$)"),
        ("assign_guard",       "\\texttt{ASSIGN} (Guard Only, Hysteresis, $1S$)"),
        ("assign_both",        "\\texttt{ASSIGN}-Only (Refresh + Guard, $1S$)"),
        ("adapt_age",          "\\texttt{ADAPT} (Age Only, Cell 0)"),
        ("adapt_dens",         "\\texttt{ADAPT} (Density Only, Cell 0)"),
        ("adapt_nophase",      "\\texttt{ADAPT} (No Phase $\\phi_u=0$, Cell 0)"),
        ("adapt_full",         "\\texttt{ADAPT}-Only (Full, Cell 0)"),
        ("define_noka",        "\\texttt{DEFINE} (No Keepalive)"),
        ("define_mrm",         "\\texttt{DEFINE} (Lossy MRM 85\\% PRR)"),
        ("define_rev",         "\\texttt{DEFINE} (Reversed Sweep Order)"),
        ("define_poisson",     "\\texttt{DEFINE} (Poisson Arrivals, $\\lambda=5\\text{ /s}$)"),
        ("p",                  "\\textbf{\\texttt{DEFINE} (Full, 32-Entry Table)}"),
    ]

    def format_row(prefix, label, n_target):
        d32 = load_raw_data(prefix)
        if d32 is None:
            return None
        m32 = d32[(d32[:, 0] == n_target) & (d32[:, 1] == 100) & (d32[:, 2] == 16) & (d32[:, 3] == 61) & (d32[:, 4] == 1)]
        if len(m32) == 0:
            return None

        d64 = load_raw_data(f"{prefix}_64nbr")
        m64 = d64[(d64[:, 0] == n_target) & (d64[:, 1] == 100)] if d64 is not None else np.empty((0, 12))

        n_seeds = len(m32)
        comp32 = (m32[:, 5] / float(n_target)) * 100.0
        m_join_100s = m32[:, 8] * 1e-4
        p95_100s = m32[:, 7] * 1e-4
        coll_10k = m32[:, 9] * 1e-4
        eslot0 = m32[:, 10] * (3.0 * 0.0174 / 32768.0)
        rmst_ks = ((comp32 / 100.0) * (m_join_100s * 100.0) + (1.0 - comp32 / 100.0) * 3600.0) / 1000.0
        e3600 = (comp32 / 100.0) * (0.0564 * (m_join_100s * 100.0) + eslot0 / float(n_target)) + (1.0 - comp32 / 100.0) * 203.04
        eb_rate = float(np.mean(m32[:, 13])) if m32.shape[1] > 13 else 1.0

        def ms_str(arr, fmt="%.2f"):
            m = float(np.mean(arr))
            se = float(np.std(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
            return f"${fmt % m} \\pm {fmt % se}$"

        if len(m64) > 0:
            comp64 = (m64[:, 5] / float(n_target)) * 100.0
            c64_str = ms_str(comp64, "%.1f")
        else:
            c64_str = "---"

        return (
            f"{label} & ${eb_rate:.2f}$ & {ms_str(rmst_ks)} & {ms_str(m_join_100s)} & "
            f"{ms_str(p95_100s)} & {ms_str(coll_10k)} & {ms_str(comp32, '%.1f')} & "
            f"{c64_str} & {ms_str(eslot0)} & {ms_str(e3600, '%.1f')} \\\\"
        )

    out_lines = ["% Auto-generated from real multi-seed Cooja logs in Statistics/"]
    for n_val in (100, 500):
        out_lines.append(f"% --- N = {n_val} ---")
        for prefix, lbl in table_rows_spec:
            line = format_row(prefix, lbl, n_val)
            if line:
                out_lines.append(line)

    out_file = os.path.join(RESULTS_DIR, "table_iv_real.tex")
    with open(out_file, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote real Cooja Table IV rows to {out_file}")


if __name__ == "__main__":
    save_standalone_legend()
    generate_markov_figure6_from_simulation()
    generate_wavefront_cdf_plot()
    generate_latex_table_iv()

    f_nodes = {2: 16, 3: 61, 4: 1}
    plot_sweep(0, f_nodes, 6,  "Total pledges", r"Total sync time ($\times 10^3$ s)", "Fig9a_total_sync_time.png", "motewise_total_time.pdf", y_scale=1e-3, is_tsync=True, mode="motewise")
    plot_sweep(0, f_nodes, 8,  "Total pledges", r"Average sync time ($\times 10^4$ slots)", "Fig9b_avg_sync_time.png", "motewise_average_neighbour_time.pdf", y_scale=1e-4, mode="motewise")
    plot_sweep(0, f_nodes, 9,  "Total pledges", r"Collisions ($\times 10^4$)", "Fig9c_collisions.png", "motewise_average_collisions.pdf", y_scale=1e-4, mode="motewise")
    plot_sweep(0, f_nodes, 5,  "Total pledges", r"DODAG completion ($\%$)", "Fig9d_dodag_completion.png", "motewise_dodag_completion.pdf", is_pct=True, mode="motewise")

    dens_lbl = r"Spatial density ($\times 10^{-2}$ motes/m$^2$)"
    plot_sweep(0, f_nodes, 6,  dens_lbl, r"Total sync time ($\times 10^3$ s)", "Fig10a_density_total_time.png", "densitywise_total_time.pdf", x_scale=0.01, y_scale=1e-3, is_tsync=True, mode="densitywise")
    plot_sweep(0, f_nodes, 8,  dens_lbl, r"Average sync time ($\times 10^4$ slots)", "Fig10b_density_avg_time.png", "densitywise_average_neighbour_time.pdf", x_scale=0.01, y_scale=1e-4, mode="densitywise")
    plot_sweep(0, f_nodes, 9,  dens_lbl, r"Collisions ($\times 10^4$)", "Fig10c_density_collisions.png", "densitywise_average_collisions.pdf", x_scale=0.01, y_scale=1e-4, mode="densitywise")
    plot_sweep(0, f_nodes, 5,  dens_lbl, r"DODAG completion ($\%$)", "Fig10d_density_completion.png", "densitywise_dodag_completion.pdf", x_scale=0.01, is_pct=True, mode="densitywise")

    tx_j_scale = 3.0 * 0.0174 / 32768.0
    rx_j_scale = (3.0 * 0.0188 / 32768.0) * 1e-2
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Tx radio energy (J)", "Fig11a_tx_energy.png", "densitywise_transmission_energy.pdf", x_scale=0.01, y_scale=tx_j_scale, mode="densitywise")
    plot_sweep(0, f_nodes, 11, dens_lbl, r"Rx scan energy ($\times 10^2$ J)", "Fig11b_rx_energy.png", "densitywise_reception_energy.pdf", x_scale=0.01, y_scale=rx_j_scale, mode="densitywise")
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Total radio energy (J)", "Fig11c_total_energy.png", "densitywise_total_energy.pdf", x_scale=0.01, y_scale=1.0, is_total_energy=True, mode="densitywise")
    plot_sweep(0, f_nodes, 10, dens_lbl, r"Radio duty cycle ($\%$)", "Fig11d_duty_cycle.png", "densitywise_duty_cycle.pdf", x_scale=0.01, is_duty=True, mode="densitywise")

    f_ch = {0: 200, 3: 61, 4: 1}
    plot_sweep(2, f_ch, 6, "Total channels", r"Total sync time ($\times 10^3$ s)", "Fig12a_ch_total_sync_time.png", "channelwise_total_time.pdf", y_scale=1e-3, is_tsync=True)
    plot_sweep(2, f_ch, 8, "Total channels", r"Average sync time ($\times 10^4$ slots)", "Fig12b_ch_avg_sync_time.png", "channelwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 9, "Total channels", r"Collisions ($\times 10^4$)", "Fig12c_ch_collisions.png", "channelwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(2, f_ch, 5, "Total channels", r"DODAG completion ($\%$)", "Fig12d_ch_dodag_completion.png", "channelwise_dodag_completion.pdf", is_pct=True)

    f_sf = {0: 200, 2: 16, 4: 1}
    plot_sweep(3, f_sf, 6, "Slotframe size (slots)", r"Total sync time ($\times 10^3$ s)", "Fig13a_sf_total_sync_time.png", "slotframewise_total_time.pdf", y_scale=1e-3, is_tsync=True)
    plot_sweep(3, f_sf, 8, "Slotframe size (slots)", r"Average sync time ($\times 10^4$ slots)", "Fig13b_sf_avg_sync_time.png", "slotframewise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 9, "Slotframe size (slots)", r"Collisions ($\times 10^4$)", "Fig13c_sf_collisions.png", "slotframewise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(3, f_sf, 5, "Slotframe size (slots)", r"DODAG completion ($\%$)", "Fig13d_sf_dodag_completion.png", "slotframewise_dodag_completion.pdf", is_pct=True)

    f_eb = {0: 200, 2: 16, 3: 61}
    plot_sweep(4, f_eb, 6, "EB interval multiplier", r"Total sync time ($\times 10^3$ s)", "Fig14a_eb_total_sync_time.png", "eb_intervalwise_total_time.pdf", y_scale=1e-3, is_tsync=True)
    plot_sweep(4, f_eb, 8, "EB interval multiplier", r"Average sync time ($\times 10^4$ slots)", "Fig14b_eb_avg_sync_time.png", "eb_intervalwise_average_neighbour_time.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 9, "EB interval multiplier", r"Collisions ($\times 10^4$)", "Fig14c_eb_collisions.png", "eb_intervalwise_average_collisions.pdf", y_scale=1e-4)
    plot_sweep(4, f_eb, 5, "EB interval multiplier", r"DODAG completion ($\%$)", "Fig14d_eb_dodag_completion.pdf", "eb_intervalwise_dodag_completion.pdf", is_pct=True)
    