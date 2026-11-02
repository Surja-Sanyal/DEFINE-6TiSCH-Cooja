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


def compute_constant_density_area(n_nodes, base_nodes=50, base_area=65):
    """Scale area L proporcional to sqrt(n_nodes) so 2D node density (nodes/m^2) stays constant and high."""
    return max(50, int(round(base_area * math.sqrt(float(n_nodes) / float(base_nodes)))))


def load_realdata_coords(n_nodes, area=100, seed=2, tx_range=50.0):
    """Generate coordinates using RT-IoT2022 realdata while guaranteeing 100% multi-hop graph connectivity."""
    random.seed(seed)
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        vals = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        vals = list(range(1, 10000))

    cx, cy = area // 2, area // 2
    coords = [(cx, cy)]  # Node 1: PAN Coordinator at center
    max_hop_dist = tx_range * 0.84  # 42m safety margin inside 50m UDGM range

    idx_val = 0
    for _ in range(n_nodes):
        placed = False
        for _attempt in range(200):
            x = int(vals[idx_val % len(vals)] % area)
            y = int(vals[(idx_val + 7) % len(vals)] % area)
            idx_val += 1
            # Ensure new node is within 42m of at least one already-placed connected node
            for px, py in coords:
                if math.hypot(x - px, y - py) <= max_hop_dist:
                    coords.append((x, y))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            # Fallback: place within 35m of a random existing node, clamped to [0, area]
            px, py = random.choice(coords)
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(10.0, 35.0)
            nx = min(area, max(0, int(round(px + r * math.cos(angle)))))
            ny = min(area, max(0, int(round(py + r * math.sin(angle)))))
            coords.append((nx, ny))
    return coords


def generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, timeout_ms=3600000, use_z1=None):
    if use_z1 is None:
        use_z1 = USE_Z1

    make_vars = f"SYNC_SCHEME={scheme_id} SF_LENGTH={sf_len} NUM_CHANNELS={n_ch} EB_MULT={eb_mult}"

    if use_z1:
        mote_id_iface = "org.contikios.cooja.mspmote.interfaces.MspMoteID"
        motetype_id = "z1_define"
        motetype_block = f"""    <motetype>
      org.contikios.cooja.mspmote.Z1MoteType
      <identifier>z1_define</identifier>
      <description>Z1 DEFINE Mote</description>
      <source>{EXP_DIR}/node.c</source>
      <commands>make -j4 node.z1 TARGET=z1 {make_vars}</commands>
      <firmware>{EXP_DIR}/build/z1/node.z1</firmware>
      <moteinterface>org.contikios.cooja.interfaces.Position</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspClock</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspMoteID</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.Msp802154Radio</moteinterface>
      <moteinterface>org.contikios.cooja.mspmote.interfaces.MspSerial</moteinterface>
    </motetype>"""
    else:
        mote_id_iface = "org.contikios.cooja.interfaces.MoteID"
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
        id_iface = mote_id_iface if use_z1 else "org.contikios.cooja.contikimote.interfaces.ContikiMoteID"
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
        {id_iface}
        <id>{idx}</id>
      </interface_config>
      <motetype_identifier>{motetype_id}</motetype_identifier>
    </mote>""")

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
        while (joined &lt; target) {{
          YIELD();
          if (msg.contains("CSV_JOIN")) {{
            log.log(msg + "\\n");
            joined++;
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

    num_joined = len(asns)
    total_time = (float(np.percentile(asns, 95)) / 100.0) if asns else 3600.0
    avg_asn = (sum(asns) / num_joined) if num_joined > 0 else 0.0
    tx_total = sum(tx_ticks)
    rx_total = sum(rx_ticks)
    coll_total = sum(colls)

    stat_key = SCHEME_MAP[scheme_id]
    stat_file = os.path.join(STAT_DIR, f"_{stat_key}_cooja_stats_.txt")
    with open(stat_file, "a") as fp:
        row = [n_nodes, area, n_ch, sf_len, eb_mult, num_joined, total_time, avg_asn, avg_asn, coll_total, tx_total, rx_total]
        fp.write(",".join(map(str, row)) + ",\n")

    mode_str = "Z1" if (USE_Z1 if use_z1 is None else use_z1) else "COOJA-NATIVE"
    print(f"[{mode_str} | Scheme {scheme_id} ({stat_key.upper()})] Nodes: {n_nodes}, Area: {area}m | Joined: {num_joined}/{n_nodes}, Sync Time: {total_time:.2f}s")


def run_sweep_group(scheme_id, group_id):
    """Run a batch of parameter points for a single scheme so 64 parallel jobs cover all 5 smooth sweeps."""
    # Group 1..5: 10-point Node Sweep (50..500 in steps of 50, 2 points per group) at constant high density (C=16, SF=61, EB=1)
    if group_id in (1, 2, 3, 4, 5):
        n1 = (group_id - 1) * 100 + 50
        n2 = n1 + 50
        for n in (n1, n2):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=n, area=None, n_ch=16, sf_len=61, eb_mult=1)
    # Group 6: Channel Sweep C in [4, 8, 12, 16] at N=250, SF=61, EB=1
    elif group_id == 6:
        for ch in (4, 8, 12, 16):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=250, area=None, n_ch=ch, sf_len=61, eb_mult=1)
    # Group 7: Slotframe Sweep SF in [23, 41, 61, 83, 101] at N=250, C=16, EB=1
    elif group_id == 7:
        for sf in (23, 41, 61, 83, 101):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=250, area=None, n_ch=16, sf_len=sf, eb_mult=1)
    # Group 8: EB Interval Sweep EB_MULT in [1, 2, 3, 4] at N=250, C=16, SF=61
    elif group_id == 8:
        for eb in (1, 2, 3, 4):
            run_cooja_and_parse(scheme_id=scheme_id, n_nodes=250, area=None, n_ch=16, sf_len=61, eb_mult=eb)
