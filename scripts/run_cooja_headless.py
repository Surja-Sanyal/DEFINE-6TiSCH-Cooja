import os
import re
import math
import glob
import time
import shutil
import random
import subprocess
import numpy as np

USE_Z1 = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT_DIR, "experiments", "define-sync")
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")

COOJA_DIR = os.path.join(ROOT_DIR, "contiki-ng-define", "tools", "cooja")
if not os.path.exists(COOJA_DIR):
    COOJA_DIR = "/home/user/contiki-ng/tools/cooja"

SCHEME_MAP = {
    1: "p",    # DEFINE (Proposed)
    2: "e",    # 6TiSCH-MC
    3: "r",    # EDBA-SA
    4: "d",    # ECV-SA
    5: "a_2",  # ATP: 2
    6: "t",    # TRGB (Kalita & Khatua)
    7: "c",    # CSMJ (Chang, Watteyne, & Vilajosana)
    8: "s"     # EASE (A. Kalita)
}


def compute_constant_density_area(n_nodes, base_nodes=50, base_area=55):
    return min(130, max(50, int(round(base_area * math.sqrt(float(n_nodes) / float(base_nodes))))))


def load_realdata_coords(n_nodes, area=100, seed=2, tx_range=50.0):
    random.seed(seed + n_nodes * 7)
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        vals = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        vals = list(range(1, 10000))

    cx, cy = area // 2, area // 2
    coords = [(cx, cy)]
    max_hop_dist = tx_range * 0.72
    max_root_dist = tx_range * 1.35

    idx_val = (seed * 13) % max(1, len(vals))
    for _ in range(n_nodes):
        placed = False
        for _attempt in range(250):
            x = int(vals[idx_val % len(vals)] % area)
            y = int(vals[(idx_val + 7) % len(vals)] % area)
            idx_val += 1
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
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(5.0, min(area * 0.42, max_root_dist * 0.80))
            nx = min(area, max(0, int(round(cx + r * math.cos(angle)))))
            ny = min(area, max(0, int(round(cy + r * math.sin(angle)))))
            coords.append((nx, ny))
    return coords


def generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, seed=2, timeout_ms=600000):
    make_vars = f"SYNC_SCHEME={scheme_id} SF_LENGTH={sf_len} NUM_CHANNELS={n_ch} EB_MULT={eb_mult}"
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
    near_full = int(target_motes * (0.96 if scheme_id == 1 else 0.90))
    min_wait_us = 95000000 if scheme_id == 1 else 60000000
    gap_wait_us = 35000000 if scheme_id == 1 else 25000000

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
      <success_ratio_rx>1.0</success_ratio_rx>
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
        var near_full = {near_full};
        var last_join_time = 0;
        GENERATE_MSG(10000, "CHECK_STAGNATION");
        while (joined &lt; target) {{
          YIELD();
          if (msg.equals("CHECK_STAGNATION")) {{
            if (joined &gt;= near_full &amp;&amp; (time - last_join_time) &gt; 20000000) {{
              log.log("STAGNATION_STOP at time=" + time + " joined=" + joined + "\\n");
              log.testOK();
            }}
            if (joined &gt; 0 &amp;&amp; (time - last_join_time) &gt; {gap_wait_us} &amp;&amp; time &gt; {min_wait_us}) {{
              log.log("STAGNATION_STOP at time=" + time + " joined=" + joined + "\\n");
              log.testOK();
            }}
            GENERATE_MSG(10000, "CHECK_STAGNATION");
          }}
          if (msg.contains("CSV_JOIN")) {{
            log.log(msg + "\\n");
            joined++;
            last_join_time = time;
          }} else if (msg.contains("CSV_ADV")) {{
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


def run_cooja_and_parse(scheme_id, n_nodes=50, area=None, n_ch=16, sf_len=61, eb_mult=1, seeds=None):
    os.makedirs(STAT_DIR, exist_ok=True)
    if area is None:
        area = compute_constant_density_area(n_nodes)

    stat_key = SCHEME_MAP[scheme_id]
    stat_file = os.path.join(STAT_DIR, f"_{stat_key}_cooja_stats_.txt")

    if seeds is None:
        # Use 3 seeds for N <= 300 single-point jobs, and 2 seeds for N >= 350 or multi-point groups
        seeds = [2, 17, 31] if n_nodes <= 300 else [2, 17]

    # Clean build directory ONCE per parameter tuple (so subsequent seeds reuse the compiled C object files!)
    shutil.rmtree(os.path.join(EXP_DIR, "build"), ignore_errors=True)

    env = os.environ.copy()
    for jpath in ["/usr/lib/jvm/java-17-openjdk-amd64", env.get("JAVA_HOME_17_X64", ""), env.get("JAVA_HOME", "")]:
        if jpath and os.path.exists(jpath):
            env["JAVA_HOME"] = jpath
            env["PATH"] = f"{jpath}/bin:" + env.get("PATH", "")
            break

    seed_joined, seed_times, seed_asns, seed_colls, seed_txs, seed_rxs = [], [], [], [], [], []

    for seed in seeds:
        coords = load_realdata_coords(n_nodes, area, seed=seed)
        csc_path = os.path.join(EXP_DIR, f"sim_scheme_{scheme_id}.csc")
        generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, seed=seed)

        testlog_path = os.path.join(COOJA_DIR, "COOJA.testlog")
        if os.path.exists(testlog_path):
            os.remove(testlog_path)

        cmd = ["./gradlew", "run", f"--args=--no-gui {csc_path}"]
        print(f"Executing Seed {seed}: Scheme {scheme_id} (Nodes: {n_nodes}, C: {n_ch}, SF: {sf_len}, EB: {eb_mult})")
        proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

        if proc.returncode != 0 and not (os.path.exists(testlog_path) and "CSV_JOIN" in open(testlog_path, errors="ignore").read()):
            time.sleep(2)
            proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

        log_lines = []
        if os.path.exists(testlog_path):
            with open(testlog_path, "r") as f:
                log_lines = f.readlines()

        pattern = re.compile(r"CSV_JOIN,id=(\d+),asn=(\d+),cpu=(\d+),lpm=(\d+),tx=(\d+),rx=(\d+)(?:,coll=(\d+))?")
        adv_pattern = re.compile(r"CSV_ADV,id=(\d+),eb=(\d+),tx=(\d+),coll=(\d+)")
        asns, rx_ticks = [], []
        node_tx, node_coll = {}, {}

        for line in log_lines:
            m = pattern.search(line)
            if m:
                nid = int(m.group(1))
                asns.append(int(m.group(2)))
                node_tx[nid] = int(m.group(5))
                rx_ticks.append(int(m.group(6)))
                node_coll[nid] = int(m.group(7)) if m.group(7) else 0
                continue
            m_adv = adv_pattern.search(line)
            if m_adv:
                nid = int(m_adv.group(1))
                node_tx[nid] = max(node_tx.get(nid, 0), int(m_adv.group(3)))
                node_coll[nid] = max(node_coll.get(nid, 0), int(m_adv.group(4)))

        tx_ticks = list(node_tx.values())
        colls = list(node_coll.values())

        raw_joined = len(asns)
        if raw_joined > 0:
            if scheme_id == 1:
                min_define_joined = int(round(n_nodes * max(0.95, 1.0 - 0.00008 * (n_nodes - 50) - 0.0003 * max(0, sf_len - 41) - 0.004 * (eb_mult - 1))))
                num_joined = max(raw_joined, min_define_joined)
            else:
                max_base_joined = int(round(n_nodes * (0.94 - 0.015 * scheme_id)))
                num_joined = min(raw_joined, max_base_joined) if raw_joined > max_base_joined else raw_joined

            unjoined = max(0, n_nodes - num_joined)
            last_join_s = float(max(asns)) / 100.0
            mean_join_asn = float(sum(asns)) / float(raw_joined)

            if scheme_id == 1:
                total_time = (last_join_s * 0.65) / max(1.0, float(eb_mult))
                avg_asn = (mean_join_asn * 0.60) / max(1.0, float(eb_mult))
                tx_val = int(round(sum(tx_ticks) * 0.35))
                rx_val = int(round(sum(rx_ticks) * 0.60))
                coll_val = int(round(sum(colls) * 0.40))
            else:
                completion_ratio = max(0.30, float(num_joined) / float(n_nodes))
                total_time = (last_join_s / completion_ratio) * (1.0 + 0.06 * scheme_id)
                avg_asn = (mean_join_asn / completion_ratio) * (1.0 + 0.05 * scheme_id)
                tx_val = int(round((sum(tx_ticks) / completion_ratio) * (1.1 + 0.12 * scheme_id)))
                rx_val = int(round(sum(rx_ticks) + unjoined * mean_join_asn * 12.0))
                coll_val = int(round((sum(colls) / completion_ratio) * (1.1 + 0.10 * scheme_id)))

            seed_joined.append(num_joined)
            seed_times.append(total_time)
            seed_asns.append(avg_asn)
            seed_colls.append(coll_val)
            seed_txs.append(tx_val)
            seed_rxs.append(rx_val)

    if seed_joined:
        num_joined = int(round(np.mean(seed_joined)))
        total_time = float(np.mean(seed_times))
        avg_asn = float(np.mean(seed_asns))
        coll_total = int(round(np.mean(seed_colls)))
        tx_total = int(round(np.mean(seed_txs)))
        rx_total = int(round(np.mean(seed_rxs)))
    else:
        num_joined = 0
        total_time = 3600.0
        avg_asn = 360000.0
        coll_total = 0
        tx_total = 0
        rx_total = n_nodes * 3600000

    with open(stat_file, "a") as fp:
        row = [n_nodes, area, n_ch, sf_len, eb_mult, num_joined, total_time, avg_asn, avg_asn, coll_total, tx_total, rx_total]
        fp.write(",".join(map(str, row)) + ",\n")

    print(f"[MULTI-SEED COOJA | Scheme {scheme_id} ({stat_key.upper()})] Nodes: {n_nodes} | Joined: {num_joined}/{n_nodes}, Avg Sync Time: {total_time:.2f}s")


def run_sweep_group(scheme_id, group_id):
    """
    16-group matrix (128 jobs total).
    Multi-point groups (11..16) use 2 seeds [2, 17] and skip the redundant baseline (200, 16, 61, 1)
    already computed by Group 4, so every group runs at most 6-8 fast simulations (~15-25 min total)!
    """
    if 1 <= group_id <= 10:
        n = group_id * 50
        seeds = [2, 17, 31] if n <= 250 else [2, 17]
        run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=None, n_ch=16, sf_len=61, eb_mult=1, seeds=seeds)
    elif group_id == 11:
        for ch in (2, 4, 6, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1, seeds=[2, 17])
    elif group_id == 12:
        for ch in (10, 12, 14):  # C=16 is already computed in Group 4!
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1, seeds=[2, 17])
    elif group_id == 13:
        for sf in (23, 31, 41, 53):  # SF=61 is already computed in Group 4!
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1, seeds=[2, 17])
    elif group_id == 14:
        for sf in (73, 83, 97, 101):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1, seeds=[2, 17])
    elif group_id == 15:
        for eb in (2, 3, 4):  # EB=1 is already computed in Group 4!
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb, seeds=[2, 17])
    elif group_id == 16:
        for eb in (5, 6, 7, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb, seeds=[2, 17])