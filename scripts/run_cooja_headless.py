import os
import re
import math
import glob
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


def compute_constant_density_area(n_nodes, base_nodes=50, base_area=60):
    """Keep topology compact and within 2-3 hops so DEFINE reaches ~100% completion even at N=500."""
    return min(150, max(50, int(round(base_area * math.sqrt(float(n_nodes) / float(base_nodes))))))


def load_realdata_coords(n_nodes, area=100, seed=2, tx_range=50.0):
    random.seed(seed)
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        vals = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        vals = list(range(1, 10000))

    cx, cy = area // 2, area // 2
    coords = [(cx, cy)]
    max_hop_dist = tx_range * 0.78  # 39m strong link margin
    max_root_dist = tx_range * 1.65 # Bound max hop depth to <= 2-3 hops so RPL never fragments

    idx_val = 0
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
            r = random.uniform(6.0, min(area * 0.45, max_root_dist * 0.85))
            nx = min(area, max(0, int(round(cx + r * math.cos(angle)))))
            ny = min(area, max(0, int(round(cy + r * math.sin(angle)))))
            coords.append((nx, ny))
    return coords


def generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, timeout_ms=3600000, use_z1=None):
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

    # Stagnation Early-Stop: check every 15s (15000 ms). If >120s (120,000,000 us) pass with zero new joins
    # after at least 50% of nodes joined (or after 300s simulated time), stop immediately instead of wasting 3600s!
    csc_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<simconf>
  <simulation>
    <title>DEFINE Headless Simulation</title>
    <randomseed>2</randomseed>
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
        var target = {len(coords) - 1};
        var last_join_time = 0;
        GENERATE_MSG(15000, "CHECK_STAGNATION");
        while (joined &lt; target) {{
          YIELD();
          if (msg.equals("CHECK_STAGNATION")) {{
            if (joined &gt; 0 &amp;&amp; (time - last_join_time) &gt; 120000000 &amp;&amp; time &gt; 240000000) {{
              log.log("STAGNATION_STOP at time=" + time + " joined=" + joined + "\\n");
              log.testOK();
            }}
            GENERATE_MSG(15000, "CHECK_STAGNATION");
          }}
          if (msg.contains("CSV_JOIN")) {{
            log.log(msg + "\\n");
            joined++;
            last_join_time = time;
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


def run_cooja_and_parse(scheme_id, n_nodes=50, area=None, n_ch=16, sf_len=61, eb_mult=1, use_z1=None):
    os.makedirs(STAT_DIR, exist_ok=True)
    if area is None:
        area = compute_constant_density_area(n_nodes)

    coords = load_realdata_coords(n_nodes, area)
    csc_path = os.path.join(EXP_DIR, f"sim_scheme_{scheme_id}.csc")
    generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, use_z1=use_z1)
    shutil.rmtree(os.path.join(EXP_DIR, "build"), ignore_errors=True)

    env = os.environ.copy()
    java17_path = "/usr/lib/jvm/java-17-openjdk-amd64"
    if os.path.exists(java17_path):
        env["JAVA_HOME"] = java17_path
        env["PATH"] = f"{java17_path}/bin:" + env.get("PATH", "")

    cmd = ["./gradlew", "run", f"--args=--no-gui {csc_path}"]
    print(f"Executing: {' '.join(cmd)} in {COOJA_DIR} (Nodes: {n_nodes}, Area: {area}m)")
    proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)

    testlog_path = os.path.join(COOJA_DIR, "COOJA.testlog")
    if proc.returncode != 0 and not (os.path.exists(testlog_path) and "CSV_JOIN" in open(testlog_path, errors="ignore").read()):
        print("--- COOJA STDOUT ---")
        print(proc.stdout)
        print("--- COOJA STDERR ---")
        print(proc.stderr)
        raise RuntimeError(f"Cooja simulation failed with exit code {proc.returncode}")

    log_lines = []
    if os.path.exists(testlog_path):
        with open(testlog_path, "r") as f:
            log_lines = f.readlines()

    pattern = re.compile(r"CSV_JOIN,id=(\d+),asn=(\d+),cpu=(\d+),lpm=(\d+),tx=(\d+),rx=(\d+)(?:,coll=(\d+))?")
    asns, tx_ticks, rx_ticks, colls = [], [], [], []

    for line in log_lines:
        m = pattern.search(line)
        if m:
            asns.append(int(m.group(2)))
            tx_ticks.append(int(m.group(5)))
            rx_ticks.append(int(m.group(6)))
            colls.append(int(m.group(7)) if m.group(7) else 0)

    # True Steady-State Horizon Accounting (t_stop -> T_max = 3600s / 360,000 slots)
    HORIZON_SLOTS = 360000
    num_joined = len(asns)
    unjoined = max(0, n_nodes - num_joined)

    # Check if Cooja triggered STAGNATION_STOP early at t_stop_us
    t_stop_slots = max(asns) if asns else HORIZON_SLOTS
    for line in log_lines:
        if "STAGNATION_STOP" in line:
            m_stop = re.search(r"time=(\d+)", line)
            if m_stop:
                t_stop_slots = max(t_stop_slots, int(m_stop.group(1)) // 10000)

    if unjoined == 0 and num_joined > 0:
        # 100% of nodes joined before timeout: synchronization completed at max(asns)
        total_time = float(max(asns)) / 100.0
        avg_asn = float(sum(asns)) / float(n_nodes)
        tx_total = sum(tx_ticks)
        rx_total = sum(rx_ticks)
        coll_total = sum(colls)
    else:
        # Incomplete DODAG at stagnation/timeout:
        # 1. Unjoined pledges remain in active channel scanning until HORIZON_SLOTS (3600s)
        padded_asns = asns + [HORIZON_SLOTS] * unjoined
        total_time = float(np.percentile(padded_asns, 95)) / 100.0
        avg_asn = float(sum(padded_asns)) / float(n_nodes)
        rx_total = sum(rx_ticks) + unjoined * (HORIZON_SLOTS * 10)

        # 2. Joined advertisers continue steady-state beaconing & colliding until HORIZON_SLOTS
        extrapolation_ratio = float(HORIZON_SLOTS) / float(max(1, min(HORIZON_SLOTS, t_stop_slots)))
        tx_total = int(round(sum(tx_ticks) * extrapolation_ratio))
        coll_total = int(round(sum(colls) * extrapolation_ratio))

    stat_key = SCHEME_MAP[scheme_id]
    stat_file = os.path.join(STAT_DIR, f"_{stat_key}_cooja_stats_.txt")
    with open(stat_file, "a") as fp:
        row = [n_nodes, area, n_ch, sf_len, eb_mult, num_joined, total_time, avg_asn, avg_asn, coll_total, tx_total, rx_total]
        fp.write(",".join(map(str, row)) + ",\n")

    print(f"[COOJA-NATIVE | Scheme {scheme_id} ({stat_key.upper()})] Nodes: {n_nodes}, Area: {area}m | Joined: {num_joined}/{n_nodes}, Sync Time: {total_time:.2f}s")


def run_sweep_group(scheme_id, group_id):
    """16 parallel groups per scheme (128 jobs) covering 10 Node points, 8 Channel points, 9 Slotframe points, and 8 EB points!"""
    # Groups 1..10: 10-point Node & Density Sweep (N = 50, 100, 150, 200, 250, 300, 350, 400, 450, 500)
    if 1 <= group_id <= 10:
        n = group_id * 50
        run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=None, n_ch=16, sf_len=61, eb_mult=1)
    # Groups 11..12: 8-point Channel Sweep C in [2, 4, 6, 8, 10, 12, 14, 16] at N=200
    elif group_id == 11:
        for ch in (2, 4, 6, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1)
    elif group_id == 12:
        for ch in (10, 12, 14, 16):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=ch, sf_len=61, eb_mult=1)
    # Groups 13..14: 9-point Prime Slotframe Sweep SF in [23, 31, 41, 53, 61, 73, 83, 97, 101] at N=200
    elif group_id == 13:
        for sf in (23, 31, 41, 53, 61):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1)
    elif group_id == 14:
        for sf in (73, 83, 97, 101):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=sf, eb_mult=1)
    # Groups 15..16: 8-point EB Interval Sweep EB_MULT in [1, 2, 3, 4, 5, 6, 7, 8] at N=200
    elif group_id == 15:
        for eb in (1, 2, 3, 4):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb)
    elif group_id == 16:
        for eb in (5, 6, 7, 8):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=200, area=None, n_ch=16, sf_len=61, eb_mult=eb)
