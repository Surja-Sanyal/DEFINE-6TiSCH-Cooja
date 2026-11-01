import os
import re
import random
import subprocess
import numpy as np

# ==============================================================================
# MOTE TARGET TOGGLE:
#   False -> Fast Cooja Native Motes (TARGET=cooja, 20x-100x faster, unlimited RAM)
#   True  -> Cycle-Accurate Z1 Hardware Emulation (TARGET=z1 via MSPSim)
# ==============================================================================
USE_Z1 = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(ROOT_DIR, "experiments", "define-sync")
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
REALDATA_PATH = os.path.join(ROOT_DIR, "Realdata", "_realdata_.txt")


import glob

# Automatically clear old stats at the start of a run to prevent data mixing
for old_file in glob.glob(os.path.join(STAT_DIR, "*stats*.txt")) + glob.glob(os.path.join(STAT_DIR, "*.csv")):
    try:
        os.remove(old_file)
    except OSError as e:
        print(f"Error cleaning old file {old_file}: {e}")


# Locate Contiki-NG Cooja directory (local submodule or Docker container path)
COOJA_DIR = os.path.join(ROOT_DIR, "contiki-ng-define", "tools", "cooja")
if not os.path.exists(COOJA_DIR):
    COOJA_DIR = "/home/user/contiki-ng/tools/cooja"

SCHEME_MAP = {
    1: "p",    # DEFINE (Proposed)
    2: "e",    # 6TiSCH-MC
    3: "r",    # EDBA-SA
    4: "d",    # ECV-SA
    5: "a_2",  # ATP
    6: "t",    # TRGB (Kalita & Khatua)
    7: "c",    # CSMJ (Chang, Watteyne, & Vilajosana)
    8: "s"     # EASE (A. Kalita)
}


def load_realdata_coords(n_nodes, area=100, seed=2):
    random.seed(seed)
    if os.path.exists(REALDATA_PATH):
        raw = np.loadtxt(REALDATA_PATH, dtype="object", delimiter=",")
        vals = [int(float(v)) for v in (raw if raw.ndim == 1 else raw[:, 0])]
    else:
        vals = list(range(1, 10000))

    if n_nodes <= 0:
        return []

    coords = [(area // 2, area // 2)]  # Node 1: PAN Coordinator at center
    for _ in range(n_nodes - 1):
        x = int(random.choice(vals) % area)
        y = int(random.choice(vals) % area)
        coords.append((x, y))
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
        TIMEOUT({timeout_ms});
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


def run_cooja_and_parse(scheme_id, n_nodes=50, area=None, n_ch=16, sf_len=101, eb_mult=1, use_z1=None):
    os.makedirs(STAT_DIR, exist_ok=True)
    
    # Scale simulation area dynamically based on node count to maintain consistent density
    AREA_MAP = {
        50: 100,
        250: 250,
        500: 350
    }
    if area is None:
        area = AREA_MAP.get(n_nodes, 200)

    coords = load_realdata_coords(n_nodes, area)
    csc_path = os.path.join(EXP_DIR, f"sim_scheme_{scheme_id}.csc")
    generate_csc_file(csc_path, coords, scheme_id, sf_len, n_ch, eb_mult, use_z1=use_z1)

    env = os.environ.copy()
    java17_path = "/usr/lib/jvm/java-17-openjdk-amd64"
    if os.path.exists(java17_path):
        env["JAVA_HOME"] = java17_path
        env["PATH"] = f"{java17_path}/bin:" + env.get("PATH", "")

    cmd = ["./gradlew", "run", f"--args=--no-gui {csc_path}"]
    print(f"Executing: {' '.join(cmd)} in {COOJA_DIR} (Nodes: {n_nodes}, Area: {area}m)")
    proc = subprocess.run(cmd, cwd=COOJA_DIR, capture_output=True, text=True, env=env)
    
    if proc.returncode != 0:
        print("--- COOJA STDOUT ---")
        print(proc.stdout)
        print("--- COOJA STDERR ---")
        print(proc.stderr)
        raise RuntimeError(f"Cooja simulation failed with exit code {proc.returncode}")

    # Read Cooja test log directly from COOJA.testlog where log.log writes
    testlog_path = os.path.join(COOJA_DIR, "COOJA.testlog")
    log_lines = []
    if os.path.exists(testlog_path):
        with open(testlog_path, "r") as f:
            log_lines = f.readlines()

    pattern = re.compile(r"CSV_JOIN,id=(\d+),asn=(\d+),cpu=(\d+),lpm=(\d+),tx=(\d+),rx=(\d+)")
    asns, tx_ticks, rx_ticks = [], [], []

    for line in log_lines:
        m = pattern.search(line)
        if m:
            asns.append(int(m.group(2)))
            tx_ticks.append(int(m.group(5)))
            rx_ticks.append(int(m.group(6)))

    num_joined = len(asns)
    total_time = (max(asns) / 100.0) if asns else 3600.0
    avg_asn = (sum(asns) / num_joined) if num_joined > 0 else 0.0
    tx_total = sum(tx_ticks)
    rx_total = sum(rx_ticks)

    stat_key = SCHEME_MAP[scheme_id]
    stat_file = os.path.join(STAT_DIR, f"_{stat_key}_cooja_stats_.txt")
    with open(stat_file, "a") as fp:
        row = [n_nodes, area, n_ch, sf_len, eb_mult, num_joined, total_time, avg_asn, avg_asn, 0, tx_total, rx_total]
        fp.write(",".join(map(str, row)) + ",\n")

    mode_str = "Z1" if (USE_Z1 if use_z1 is None else use_z1) else "COOJA-NATIVE"
    print(f"[{mode_str} | Scheme {scheme_id} ({stat_key.upper()})] Nodes: {n_nodes}, Area: {area}m | Joined: {num_joined}/{n_nodes}, Sync Time: {total_time:.2f}s")


if __name__ == "__main__":
    NODES_LIST = list(range(25, 500 + 1, 25))
    CHANNELS_LIST = [4, 8, 12, 16]
    SF_LIST = [23, 41, 61, 83, 101]
    EB_LIST = [1, 2, 3, 4]

    for n_ch in CHANNELS_LIST:
        for sf_len in SF_LIST:
            for eb_mult in EB_LIST:
                for n_nodes in NODES_LIST:
                    for scheme in range(1, 9):
                        # Correct invocation (omit area so it scales automatically via AREA_MAP)
                        run_cooja_and_parse(
                            scheme_id=scheme, n_nodes=n_nodes,
                            n_ch=n_ch, sf_len=sf_len, eb_mult=eb_mult
                        )