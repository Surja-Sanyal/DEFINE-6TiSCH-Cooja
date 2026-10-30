# Program:   TSCH Node Synchronization (6PySCH - Extended with TRGB, Chang2026, and EASE)
# Inputs:    DATA_STORE_LOCATION/Realdata/_realdata_.txt
# Outputs:   DATA_STORE_LOCATION/Statistics/*.txt
# Author:    Surja Sanyal
# Email:     hi.surja06@gmail.com

import os
import sys
import glob
import math
import copy
import time
import random
import datetime
import platform
import traceback
import numpy as np
import multiprocessing

##  Global environment   ##

#   Customize here  #
SEED                    = 2                                             # Random seed
SIMS                    = 5                                             # Simulations per setup
EB_PROB                 = 0.1                                           # EB probability limit
COMM_RANGE              = 50                                            # Signal distance
SLOTFRAME_SIZES         = [23, 41, 61, 83, 101]                         # Slots
EB_INTERVALS            = [1, 2, 3, 4]                                  # Times slotframe
NODES                   = list(range(25, 500 + 1, 25))                  # Total nodes
AREAS                   = [100]                                         # Area range
CHANNELS                = list(range(4, 16 + 1, 4))                     # Channels
ATP_LEVELS              = list(range(2, 8 + 1, 2))                      # ATP level
# P: DEFINE, E: 6TiSCH-MC, R: EDBA-SA, D: ECV-SA, A: ATP, T: TRGB, C: Chang2026, S: EASE
MECHANISMS              = ['P', 'E', 'R', 'D', 'A', 'T', 'C', 'S']      # Mechanisms
TIMEOUT                 = 3600                                          # Seconds (= 1 hour)

#   Do not change   #
REALDATA                = ["Realdata", "_realdata_.txt"]
BASE_DIR                = os.path.dirname(os.path.abspath(sys.argv[0]))
DATA_LOAD_LOCATION      = BASE_DIR
DATA_STORE_LOCATION     = BASE_DIR


def get_script_stem():
    return os.path.splitext(os.path.basename(sys.argv[0]))[0]


def print_locked(*content, sep=" ", end="\n"):
    log_dir = os.path.join(DATA_STORE_LOCATION, "Log_Files")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "_Log_" + get_script_stem() + ".txt")
    with open(log_path, 'a') as f:
        print(*content, sep=sep, end=end, file=f)


class node:
    def __init__(self, area, realdata=[], node_id=0, height=0, start_channel=-1, start_asn=-1, chs=[]):
        self.iden = node_id
        self.parent = -1
        self.height = height
        self.x_axis = int(random.choice(realdata) % area) if (node_id > 0) else area // 2
        self.y_axis = int(random.choice(realdata) % area) if (node_id > 0) else area // 2
        self.start_channel = start_channel if (node_id > 0) else 0
        self.start_asn = start_asn if (node_id > 0) else 0
        self.nbr_available_asn = -1
        self.joined_asn = -1 if (node_id > 0) else 0
        self.joining_dur = -1 if (node_id > 0) else 0
        self.joined = False if (node_id > 0) else True
        self.collisions = 0
        self.no_eb = 0
        self.last_eb = -1
        self.last_eb_freq = []
        self.chs = chs
        self.nbrs = []
        self.children = []
        self.TxCount = 0
        self.RxCount = 0
        self.be = 1  # Backoff exponent for Chang 2026 ('C')

    def join(self, parent, height, asn, ch, slot, chs):
        self.joined = True
        self.parent = parent
        self.height = height
        self.start_channel = ch
        self.start_asn = slot
        self.joined_asn = asn
        self.joining_dur = (asn - self.nbr_available_asn + 1) if (self.nbr_available_asn >= 0) else asn
        self.chs = chs
        self.RxCount = asn

    def set_last_eb_details(self, sf_number, freq, tx_slots=1):
        self.last_eb = sf_number
        self.last_eb_freq = freq
        self.TxCount += tx_slots

    def info(self):
        return self.parent, self.height

    def hops(self):
        return self.height

    def identity(self):
        return self.iden

    def x_coord(self):
        return self.x_axis

    def y_coord(self):
        return self.y_axis


def get_distance(pledge, joined_node):
    return math.sqrt(
        (pledge.x_coord() - joined_node.x_coord()) ** 2 +
        (pledge.y_coord() - joined_node.y_coord()) ** 2
    )


def generate_beacons(j_nodes, sf_size, eb_interval, asn, chs, r_chs, n_channels, scheme, eb_prob_limit, atp_level):
    eb_prob = eb_prob_limit
    sf_number = asn // sf_size
    ch_dict = {}

    if scheme == 'P':
        for joined_node in j_nodes:
            if joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval:
                decay = math.exp(-1.0 * (sf_number * math.sqrt(1 + len(joined_node.children))) / n_channels)
                p_tx = max(eb_prob_limit / 10.0, decay)
                if random.random() <= p_tx:
                    ch_sel = chs[(sf_number + joined_node.hops()) % n_channels]
                    ch_dict[joined_node] = ch_sel
                    joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)

    elif scheme == 'E':
        for joined_node in j_nodes:
            if (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval) and (random.random() <= eb_prob):
                ch_sel = chs[sf_number % n_channels]
                ch_dict[joined_node] = ch_sel
                joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)

    elif scheme == 'R':
        for joined_node in j_nodes:
            if joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval:
                ch_sel = random.choice(r_chs)
                ch_dict[joined_node] = ch_sel
                joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)

    elif scheme == 'D':
        for joined_node in j_nodes:
            if joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval:
                ch_sel = chs[(sf_number + joined_node.start_channel) % n_channels]
                ch_dict[joined_node] = ch_sel
                joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)

    elif scheme == 'A':
        for joined_node in j_nodes:
            if (joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval) and (random.random() <= eb_prob):
                sub_chs = [chs[(sf_number + (channel_offset + 1)) % n_channels] for channel_offset in range(atp_level)]
                ch_dict[joined_node] = sub_chs
                joined_node.set_last_eb_details(sf_number, sub_chs, tx_slots=atp_level)

    elif scheme == 'T':
        rgb_phase = sf_number % 3
        for joined_node in j_nodes:
            if joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval:
                p_rgb = min(0.8, eb_prob * 4.0) if rgb_phase == 0 else (min(0.4, eb_prob * 2.0) if rgb_phase == 1 else eb_prob)
                if random.random() <= p_rgb:
                    rgb_offset = (rgb_phase * (n_channels // 3) + (joined_node.identity() % max(1, n_channels // 3))) % n_channels
                    ch_sel = chs[(sf_number + rgb_offset) % n_channels]
                    ch_dict[joined_node] = ch_sel
                    joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)

    elif scheme == 'C':
        for joined_node in j_nodes:
            if joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval:
                p_chang = min(1.0, 1.0 / (2 ** joined_node.be))
                if random.random() <= p_chang:
                    ch_sel = chs[sf_number % n_channels]
                    ch_dict[joined_node] = ch_sel
                    joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)
                else:
                    joined_node.be = max(1, joined_node.be - 1)

    elif scheme == 'S':
        for joined_node in j_nodes:
            if joined_node.last_eb < 0 or sf_number >= joined_node.last_eb + eb_interval:
                p_ease = max(eb_prob_limit / 5.0, eb_prob_limit / (1.0 + 0.5 * len(joined_node.children)))
                if random.random() <= p_ease:
                    auto_hash = (joined_node.identity() * 2654435761) % n_channels
                    ch_sel = chs[(sf_number + auto_hash) % n_channels]
                    ch_dict[joined_node] = ch_sel
                    joined_node.set_last_eb_details(sf_number, [ch_sel], tx_slots=1)

    return ch_dict


def join_node(pledge, j_nodes, sf_size, asn, chs, r_chs, n_channels, scheme, assigned, collisions, ch_dict):
    r_ch = random.choice(r_chs)
    local_tx = [(j, c) for j, c in ch_dict.items() if j in j_nodes]
    ch = [c for _, c in local_tx]

    if scheme == 'A':
        ch = [c for ch_j in ch for c in ch_j]

    matching_count = sum(1 for c in ch if c == r_ch)

    if matching_count == 1:
        if scheme == 'A':
            joined_node = [j for j, c in local_tx if r_ch in c][0]
        else:
            joined_node = [j for j, c in local_tx if c == r_ch][0]

        if joined_node.hops() + 1 < pledge.hops():
            if scheme == 'D':
                assigned += 1
                start_ch = assigned % n_channels
            elif scheme == 'P':
                start_ch = (joined_node.hops() + 1) % n_channels
            else:
                start_ch = r_ch

            pledge.join(joined_node.identity(), joined_node.hops() + 1, asn, start_ch, asn % n_channels, chs)
            joined_node.children.append(pledge.identity())
            if scheme == 'C':
                joined_node.be = max(1, joined_node.be - 1)
            return [collisions, assigned, True]

    elif matching_count > 1:
        pledge.collisions += 1
        collisions += 1
        if scheme == 'C':
            for j, c in local_tx:
                if c == r_ch:
                    j.be = min(5, j.be + 1)
    else:
        pledge.no_eb += 1

    return [collisions, assigned, False]


def DODAG_joining(joined_nodes, nodes, n_nodes, sf_size, eb_interval, chs, r_chs, n_channels, eb_prob, asn, scheme, timeout, atp_level, d_assigned=0):
    d_assigned = 0
    collisions = 0

    while len(joined_nodes) <= n_nodes:
        ch_dict = generate_beacons(joined_nodes, sf_size, eb_interval, asn, chs, r_chs, n_channels, scheme, eb_prob, atp_level)

        for pledge in [n for n in nodes if not n.joined]:
            nbrs = [joined for joined in joined_nodes if (joined.identity() in pledge.nbrs and joined.joined_asn != asn)]

            if len(nbrs) > 0:
                if pledge.nbr_available_asn < 0:
                    pledge.nbr_available_asn = asn

                pledge_status = pledge.joined
                joining_details = join_node(pledge, nbrs, sf_size, asn, chs, r_chs, n_channels, scheme, d_assigned, collisions, ch_dict)

                if not pledge_status and joining_details[-1]:
                    joined_nodes.append(pledge)

                collisions = joining_details[0]
                d_assigned = joining_details[1]

        if (asn / 100.0) > timeout:
            break

        asn += sf_size

    TxTotal = sum(n.TxCount for n in joined_nodes)
    RxTotal = sum(n.RxCount for n in joined_nodes)
    return joined_nodes, asn, collisions, TxTotal, RxTotal


def get_nbrs(lbr, nodes, comm_range):
    all_nodes = [lbr] + nodes
    for each_node in nodes:
        for other_node in all_nodes:
            if each_node.identity() != other_node.identity() and get_distance(each_node, other_node) <= comm_range:
                each_node.nbrs.append(other_node.identity())
    return nodes


def get_realdata(load):
    realdata_loc = os.path.join(load, *REALDATA)
    if os.path.exists(realdata_loc):
        realdata = np.loadtxt(realdata_loc, dtype='object', delimiter=",")
        if realdata.ndim == 1:
            return [int(float(val)) for val in realdata]
        return [int(float(val)) for val in realdata[:, 0].reshape(1, -1)[0]]
    else:
        rng = np.random.default_rng(SEED)
        return rng.integers(0, 10000, size=50000).tolist()


def execute_method(store, n_channels, sf_size, eb_interval, n_nodes, area, sims, lbr, nodes, chs, r_chs, eb_prob, mechanism, timeout, atp_level):
    total_time, avg_time, avg_nbr_time, num_joined, num_collisions, TxAvg, RxAvg = 0, 0, 0, 0, 0, 0, 0

    for sim in range(sims):
        random.seed(SEED + sim)
        joined_nodes = [copy.deepcopy(lbr)]
        joined_nodes, asn, collisions, TxTotal, RxTotal = DODAG_joining(
            joined_nodes, copy.deepcopy(nodes), n_nodes, sf_size, eb_interval,
            chs, r_chs, n_channels, eb_prob, asn=0, scheme=mechanism,
            timeout=timeout, atp_level=atp_level
        )

        dodag_size = len(joined_nodes) - 1 if (len(joined_nodes) > 1) else 1
        total_time += asn / 100.0
        TxAvg += TxTotal
        RxAvg += RxTotal
        num_collisions += collisions
        num_joined += len(joined_nodes) - 1
        avg_time += sum(j_node.joined_asn for j_node in joined_nodes[1:]) / dodag_size
        avg_nbr_time += sum(each_node.joining_dur for each_node in joined_nodes[1:]) / dodag_size

    stat_dir = os.path.join(store, "Statistics")
    os.makedirs(stat_dir, exist_ok=True)
    stat_file = os.path.join(
        stat_dir,
        "_" + str(mechanism).lower() + ("_" + str(atp_level) if (mechanism == 'A') else "") + "_stats_.txt"
    )
    with open(stat_file, 'a') as fp:
        np.savetxt(
            fp,
            [n_nodes, area, n_channels, sf_size, eb_interval,
             num_joined / sims, total_time / sims, avg_time / sims,
             avg_nbr_time / sims, num_collisions / sims, TxAvg / sims, RxAvg / sims],
            fmt="%s", delimiter=",", newline=","
        )
        fp.write("\n")


def main():
    seed = SEED
    num_nodes, areas, channels, comm_range, atp_levels, eb_prob, sf_sizes, eb_intervals, sims, mechanisms, timeout = \
        NODES, AREAS, CHANNELS, COMM_RANGE, ATP_LEVELS, EB_PROB, SLOTFRAME_SIZES, EB_INTERVALS, SIMS, MECHANISMS, TIMEOUT
    load, store = DATA_LOAD_LOCATION, DATA_STORE_LOCATION
    n_CPU = max(1, multiprocessing.cpu_count() - 1)

    random.seed(seed)
    os.makedirs(os.path.join(load, "Statistics"), exist_ok=True)
    for filename in glob.glob(os.path.join(load, "Statistics", '*.*')):
        try:
            os.remove(filename)
        except OSError:
            pass

    realdata = get_realdata(load)
    n_processes = len(num_nodes) * len(areas) * len(channels) * len(sf_sizes) * len(eb_intervals) * (len(mechanisms) - 1 + len(atp_levels))
    print("\nThe total number of child processes to execute are: {}.".format(n_processes), end="\n\n")

    processes = []
    for n_channels in reversed(channels):
        chs, r_chs = [i + 1 for i in range(n_channels)], [i + 1 for i in range(n_channels)]
        for sf_size in sf_sizes:
            for eb_interval in eb_intervals:
                for n_nodes in reversed(num_nodes):
                    for area in areas:
                        random.shuffle(chs)
                        lbr = node(area, chs=copy.deepcopy(chs))
                        nodes = [node(area, realdata=realdata, node_id=i + 1, height=n_nodes) for i in range(n_nodes)]
                        nodes = get_nbrs(lbr, nodes, comm_range)

                        for mechanism in reversed(mechanisms):
                            levels = atp_levels if mechanism == 'A' else [-1]
                            for lvl in levels:
                                while sum(1 for p in processes if p.is_alive()) >= n_CPU:
                                    time.sleep(0.2)
                                p = multiprocessing.Process(
                                    target=execute_method,
                                    args=(store, n_channels, sf_size, eb_interval, n_nodes, area, sims,
                                          copy.deepcopy(lbr), copy.deepcopy(nodes), copy.deepcopy(chs),
                                          r_chs, eb_prob, mechanism, timeout, lvl),
                                    daemon=True
                                )
                                processes.append(p)
                                p.start()

    while sum(1 for p in processes if p.is_alive()) != 0:
        completed = len(processes) - sum(1 for p in processes if p.is_alive())
        print("\rPercent jobs completed / submitted: {:6.2f} % ({}) / {:6.2f} % ({}).".format(
            100 * completed / n_processes, completed,
            100 * len(processes) / n_processes, len(processes)
        ), end='')
        time.sleep(1)

    for p in processes:
        p.join()
    print("\n\nAll simulations completed.")


if __name__ == "__main__":
    main()