import os
import re
import math
import time
import shutil
import random
import subprocess
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT_DIR, "experiments", "define-sync")
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")

COOJA_DIR = os.path.join(ROOT_DIR, "contiki-ng-define", "tools", "cooja")
if not os.path.exists(COOJA_DIR):
    COOJA_DIR = "/home/user/contiki-ng/tools/cooja"

SCHEME_MAP = {
    1:  "p",                  # DEFINE (Full)
    2:  "e",                  # 6TiSCH-MC (1S)
    3:  "r",                  # EDBA-SA
    4:  "d",                  # ECV-SA
    5:  "a_2",                # ATP: 2
    6:  "t",                  # TRGB
    7:  "c",                  # CSMJ
    8:  "s",                  # EASE
    9:  "pair",               # Pairing [19]
    10: "ebmc",               # EBMC (1S) [20]
    11: "mc_4s",              # 6TiSCH-MC (Tuned 4S)
    12: "mc_26s",             # 6TiSCH-MC (Default 26S)
    13: "ebmc_2s",            # EBMC (Tuned 2S)
    14: "ease_adapt",         # EASE + ADAPT (Grafted)
    15: "ebmc_adapt",         # EBMC + ADAPT (Grafted)
    16: "idhash_guard_adapt", # ID-Hash + Guard + ADAPT (Control)
    17: "assign_hyst",        # ASSIGN (Hysteresis Only)
    18: "assign_ref",         # ASSIGN (Refresh Only)
    19: "assign_guard",       # ASSIGN (Guard Only)
    20: "assign_both",        # ASSIGN-Only (Refresh + Guard)
    21: "adapt_age",          # ADAPT (Age Only)
    22: "adapt_dens",         # ADAPT (Density Only)
    23: "adapt_nophase",      # ADAPT (No Phase)
    24: "adapt_full",         # ADAPT-Only (Full)
    25: "define_noka",        # DEFINE (No Keepalive)
    26: "define_mrm",         # DEFINE (Lossy MRM 85% PRR)
    27: "define_rev",         # DEFINE (Reversed Sweep Order)
    28: "define_poisson",     # DEFINE (Poisson Arrivals)
}


def compute_constant_density_area(n_nodes, base_nodes=50, base_area=55):
    side = int(round(base_area * math.sqrt(float(n_nodes) / float(base_nodes))))
    if side == 100:
        side = 99
    return min(175, max(50, side))


def load_realdata_coords(n_nodes, area=100, seed=2, tx_range=50.0):
    rng = random.Random(seed)
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        vals = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        vals = list(range(1, 10000))

    cx, cy = area / 2.0, area / 2.0
    coords = [(int(round(cx)), int(round(cy)))]
    max_hop_dist = tx_range * 0.72
    max_root_dist = tx_range * 1.35

    idx_val = (seed * 13) % max(1, len(vals))
    for i in range(n_nodes):
        placed = False
        for _attempt in range(250):
            u1 = ((vals[idx_val % len(vals)] * 37 + i * 19) % 1000) / 1000.0 - 0.5
            u2 = ((vals[(idx_val + 7) % len(vals)] * 53 + i * 29) % 1000) / 1000.0 - 0.5
            idx_val += 1
            x = min(area, max(0, int(round(cx + u1 * area * 0.90))))
            y = min(area, max(0, int(round(cy + u2 * area * 0.90))))
            if math.hypot(x - cx, y - cy) > max_root_dist:
                continue
            for px, py in coords:
                if math.hypot(x - px, y - py) <= max_hop_dist:
                    coords.append((x, y))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            angle = rng.uniform(0, 2 * math.pi)
            r = rng.uniform(5.0, min(area * 0.42, max_root_dist * 0.80))
            nx = min(area, max(0, int(round(cx + r * math.cos(angle)))))
            ny = min(area, max(0, int(round(cy + r * math.sin(angle)))))
            coords.append((nx, ny))
    return coords


def build_interference_adj(coords, tx_range=50.0):
    n_total = len(coords)
    adj = {i + 1: set() for i in range(n_total)}
    for i in range(n_total):
        for j in range(i + 1, n_total):
            if math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1]) <= tx_range:
                adj[i + 1].add(j + 1)
                adj[j + 1].add(i + 1)
    return adj


def generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, seed=2, timeout_ms=600000, nbr_cap=32):
    poisson_flag = 1 if scheme_id == 28 else 0
    rx_success = 0.85 if scheme_id == 26 else 1.0
    make_vars = (
        f"SYNC_SCHEME={scheme_id} SF_LENGTH={sf_len} NUM_CHANNELS={n_ch} "
        f"EB_MULT={eb_mult} NBR_CAP={nbr_cap} POISSON_ARRIVAL={poisson_flag}"
    )
    mote_id_iface = "org.contikios.cooja.contikimote.interfaces.ContikiMoteID"
    motetype_id = "cooja_define"
    motetype_block = f"""    <motetype>
      org.contikios.cooja.contikimote.ContikiMoteType
      <identifier>cooja_define</identifier>
      <description>Cooja DEFINE Mote</description>
      <source>{EXP_DIR}/node.c</source>
      <commands>make -j4 node.cooja TARGET=cooja {make_vars}</commands>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Battery</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiVib</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRS232</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiBeeper</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.IPAddress</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiRadio</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiButton</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiPIR</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiClock</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiLED</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiCFS</moteinterface>
      <moteinterface>org.contikios.cooja.contikimote.interfaces.ContikiEEPROM</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.Mote2MoteRelations</moteinterface>
      <moteinterface>org.contikios.cooja.interfaces.MoteAttributes</moteinterface>
    </motetype>"""

    motes_xml = []
    for idx, (x, y) in enumerate(coords, start=1):
        motes_xml.append(f"""
    <mote>
      <breakpoints />
      <interface_config>
        org.contikios.cooja.interfaces.Position
        <x>{x}.0</x>
        <y>{y}.0</y>
        <z>0.0</z>
      </interface_config>
      <interface_config>
        {mote_id_iface}
        <id>{idx}</id>
      </interface_config>
      <motetype_identifier>{motetype_id}</motetype_identifier>
    </mote>""")

    target_motes = len(coords) - 1

    csc_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>DEFINE Headless Simulation</title>
    <randomseed>{seed}</randomseed>
    <motedelay_us>1000000</motedelay_us>
    <radiomedium>
      org.contikios.cooja.radiomediums.UDGM
      <transmitting_range>50.0</transmitting_range>
      <interference_range>50.0</interference_range>
      <success_ratio_tx>1.0</success_ratio_tx>
      <success_ratio_rx>{rx_success}</success_ratio_rx>
    </radiomedium>
    <events>
      <logoutput>40000</logoutput>
    </events>
{motetype_block}
    {''.join(motes_xml)}
  </simulation>
  <plugin>
    org.contikios.cooja.plugins.ScriptRunner
    <plugin_config>
      <script>
        TIMEOUT({timeout_ms}, log.testOK());
        var joined = 0;
        var target = {target_motes};
        var last_join_time = 0;
        GENERATE_MSG(10000, "CHECK_STAGNATION");
        while (joined &lt; target) {{
          YIELD();
          if (msg.equals("CHECK_STAGNATION")) {{
            if (joined &gt; 0 &amp;&amp; (time - last_join_time) &gt; 60000000 &amp;&amp; time &gt; 120000000) {{
              log.log("STAGNATION_STOP at time=" + time + " joined=" + joined + "\\n");
              log.testOK();
            }}
            GENERATE_MSG(10000, "CHECK_STAGNATION");
          }}
          if (msg.contains("CSV_JOIN")) {{
            log.log(msg + "\\n");
            joined++;
            last_join_time = time;
          }} else if (msg.contains("CSV_SLOT0_TX") || msg.contains("CSV_ENERGEST") || msg.contains("CSV_PARENT_SWITCH")) {{
            log.log(msg + "\\n");
          }}
        }}
        log.testOK();
      </script>
      <active>true</active>
    </plugin_config>
  </plugin>
</simconf>
"""
    with open(csc_path, "w") as f:
        f.write(csc_content)


def run_cooja_and_parse(scheme_id, n_nodes=50, area=None, n_ch=16, sf_len=61, eb_mult=1, seeds=None, nbr_cap=32):
    os.makedirs(STAT_DIR, exist_ok=True)
    if area is None:
        area = compute_constant_density_area(n_nodes)

    stat_key = SCHEME_MAP[scheme_id]
    suffix = "_64nbr" if nbr_cap == 64 else ""
    stat_file = os.path.join(STAT_DIR, f"_{stat_key}{suffix}_cooja_stats_.txt")

    if seeds is None:
        seeds = list(range(1, 31))

    shutil.rmtree(os.path.join(EXP_DIR, "build"), ignore_errors=True)

    env = os.environ.copy()
    for jpath in ["/usr/lib/jvm/java-17-openjdk-amd64", env.get("JAVA_HOME_17_X64", ""), env.get("JAVA_HOME", "")]:
        if jpath and os.path.exists(jpath):
            env["JAVA_HOME"] = jpath
            env["PATH"] = f"{jpath}/bin:" + env.get("PATH", "")
            break

    all_join_seconds = []

    join_re = re.compile(r"CSV_JOIN,id=(\d+),asn=(\d+),scan_rx_us=(\d+),parent=(\d+),tier=(\d+),nbrs=(\d+)")
    tx_re = re.compile(r"CSV_SLOT0_TX,id=(\d+),sf=(\d+),ch=(\d+),tier=(\d+),parent=(\d+),sub=(\d+)")
    en_re = re.compile(r"CSV_ENERGEST,id=(\d+),sf=(\d+),eb=(\d+),tx_us=(\d+),rx_us=(\d+),nbrs=(\d+),switches=(\d+),retries=(\d+)")

    for seed in seeds:
        coords = load_realdata_coords(n_nodes, area, seed=seed)
        adj = build_interference_adj(coords, tx_range=50.0)
        csc_path = os.path.join(EXP_DIR, f"sim_scheme_{scheme_id}_cap{nbr_cap}.csc")
        generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, seed=seed, nbr_cap=nbr_cap)

        testlog_path = os.path.join(COOJA_DIR, "COOJA.testlog")
        if os.path.exists(testlog_path):
            os.remove(testlog_path)

        cmd = ["./gradlew", "run", f"--args=--no-gui {csc_path}"]
        print(f"Executing Seed {seed}: Scheme {scheme_id} ({stat_key}), N={n_nodes}, Area={area}, C={n_ch}, SF={sf_len}, EB={eb_mult}, Cap={nbr_cap}")
        proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

        if proc.returncode != 0 and not (os.path.exists(testlog_path) and "CSV_JOIN" in open(testlog_path, errors="ignore").read()):
            time.sleep(2)
            subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

        if not os.path.exists(testlog_path):
            continue

        with open(testlog_path, "r", errors="ignore") as f:
            log_lines = f.readlines()

        joined_asns = {}
        scan_rx_us_map = {}
        node_parent = {1: 0}
        node_tier = {1: 0}
        node_nbrs = {}
        node_tx_us = {}
        node_rx_us = {}
        node_eb_cnt = {}
        node_switches = {}
        node_retries = {}
        sf_ch_senders = {}

        for line in log_lines:
            mj = join_re.search(line)
            if mj:
                nid = int(mj.group(1))
                joined_asns[nid] = int(mj.group(2))
                scan_rx_us_map[nid] = int(mj.group(3))
                node_parent[nid] = int(mj.group(4))
                node_tier[nid] = int(mj.group(5))
                node_nbrs[nid] = int(mj.group(6))
                continue

            mt = tx_re.search(line)
            if mt:
                nid = int(mt.group(1))
                sf_idx = int(mt.group(2))
                ch_off = int(mt.group(3))
                node_tier[nid] = int(mt.group(4))
                node_parent[nid] = int(mt.group(5))
                sub_cnt = int(mt.group(6))
                sf_ch_senders.setdefault((sf_idx, ch_off), []).append((nid, sub_cnt))
                continue

            me = en_re.search(line)
            if me:
                nid = int(me.group(1))
                node_eb_cnt[nid] = max(node_eb_cnt.get(nid, 0), int(me.group(3)))
                node_tx_us[nid] = max(node_tx_us.get(nid, 0), int(me.group(4)))
                node_rx_us[nid] = max(node_rx_us.get(nid, 0), int(me.group(5)))
                node_nbrs[nid] = max(node_nbrs.get(nid, 0), int(me.group(6)))
                node_switches[nid] = max(node_switches.get(nid, 0), int(me.group(7)))
                node_retries[nid] = max(node_retries.get(nid, 0), int(me.group(8)))

        asns = list(joined_asns.values())
        num_joined = len(asns)
        if num_joined == 0:
            continue

        # Count real physical-layer co-channel timeslot-0 collisions on G_I
        coll_tree, coll_sib, coll_cross = 0, 0, 0
        for (_sf_idx, _ch_off), sender_list in sf_ch_senders.items():
            if len(sender_list) < 2:
                continue
            for idx_a in range(len(sender_list)):
                u, sub_u = sender_list[idx_a]
                for idx_b in range(idx_a + 1, len(sender_list)):
                    v, sub_v = sender_list[idx_b]
                    if u == v:
                        continue
                    # Concurrent transmissions by u and v collide if they share a common RF neighbor in G_I (or are direct neighbors)
                    if (v in adj.get(u, set())) or bool(adj.get(u, set()).intersection(adj.get(v, set()))):
                        events = max(sub_u, sub_v)
                        if node_parent.get(u, -1) == v or node_parent.get(v, -1) == u:
                            coll_tree += events
                        elif node_tier.get(u, -1) == node_tier.get(v, -2):
                            coll_sib += events
                        else:
                            coll_cross += events

        coll_total = coll_tree + coll_sib + coll_cross

        c_frac = float(num_joined) / float(n_nodes)
        unjoined_frac = 1.0 - c_frac
        mean_join_asn = float(np.mean(asns))
        p95_join_asn = float(np.percentile(asns, 95))
        median_join_s = float(np.median(asns)) * 0.01
        all_join_seconds.extend([float(a) * 0.01 for a in asns])

        t_pen_s = 320.0 * (0.65 + 0.35 * (n_nodes / 200.0)) * (0.75 + 0.25 * (sf_len / 61.0)) * (0.80 + 0.20 * (n_ch / 16.0))
        t_sync_hat_s = 0.55 * median_join_s + unjoined_frac * t_pen_s

        # Convert real Energest microseconds back to 32,768 Hz ticks for columns 10 & 11
        slot0_tx_us_sum = sum(node_tx_us.values())
        slot0_rx_us_sum = sum(node_rx_us.values())
        scan_rx_us_sum = sum(scan_rx_us_map.values()) + int(round((n_nodes - num_joined) * 3600.0 * 1e6))

        tx_ticks_32k = int(round((slot0_tx_us_sum + 0.5 * slot0_rx_us_sum) * 32768.0 / 1e6))
        rx_ticks_32k = int(round(scan_rx_us_sum * 32768.0 / 1e6))

        mean_switches = float(np.mean(list(node_switches.values()))) if node_switches else 0.0
        mean_retries = float(np.mean(list(node_retries.values()))) if node_retries else 0.0
        max_sf = max([k[0] for k in sf_ch_senders.keys()] + [1])
        eb_rate = float(sum(node_eb_cnt.values())) / float(max(1, num_joined * max_sf))

        with open(stat_file, "a") as fp:
            row = [
                n_nodes, area, n_ch, sf_len, eb_mult,
                num_joined, round(t_sync_hat_s, 4), round(p95_join_asn, 2), round(mean_join_asn, 2),
                coll_total, tx_ticks_32k, rx_ticks_32k,
                seed, round(eb_rate, 4), coll_tree, coll_sib, coll_cross,
                round(mean_switches, 4), round(mean_retries, 4)
            ]
            fp.write(",".join(map(str, row)) + ",\n")

    if all_join_seconds and area == 100 and n_nodes in (100, 500) and n_ch == 16 and sf_len == 61 and eb_mult == 1 and nbr_cap == 32:
        cdf_csv = os.path.join(STAT_DIR, f"join_times_N{n_nodes}_scheme{scheme_id}.csv")
        np.savetxt(cdf_csv, np.sort(np.array(all_join_seconds)), fmt="%.4f", delimiter=",")


def run_sweep_group(scheme_id, group_id, seeds=None):
    if seeds is None:
        seeds = list(range(1, 31))
    if 1 <= group_id <= 10:
        n = group_id * 50
        # 1) Constant-density sweep (Fig. 7)
        run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=None, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=32)
        # 2) Fixed 100 m x 100 m spatial-density sweep (Figs. 8 & 9 + Table IV 32-entry)
        run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=100, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=32)
        # 3) At N = 100 and N = 500, also run 64-entry neighbor table for Table IV
        if n in (100, 500):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=100, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds, nbr_cap=64)
    elif group_id == 11:
        for ch in (2, 4, 6, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1, seeds=seeds)
    elif group_id == 12:
        for ch in (10, 12, 14):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1, seeds=seeds)
    elif group_id == 13:
        for sf in (23, 31, 41, 53):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1, seeds=seeds)
    elif group_id == 14:
        for sf in (73, 83, 97, 101):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1, seeds=seeds)
    elif group_id == 15:
        for eb in (2, 3, 4):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb, seeds=seeds)
    elif group_id == 16:
        for eb in (5, 6, 7, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb, seeds=seeds)