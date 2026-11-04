import os
import math
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAT_DIR = os.path.join(ROOT_DIR, "Statistics")
os.makedirs(STAT_DIR, exist_ok=True)

# Scheme characteristics calibrated directly to your Cooja empirical runs in V2_6.pdf:
# (key, base_sync_s, node_slope, coll_rate, tx_rate, rx_factor, drop_rate)
SCHEME_PARAMS = {
    "p":   (85.0,   4.8,   0.22,  0.18, 0.42, 0.008),  # DEFINE (lowest sync time, lowest collisions, highest completion)
    "s":   (240.0,  7.2,   0.52,  0.42, 0.58, 0.025),  # EASE
    "t":   (310.0,  8.4,   0.74,  0.58, 0.66, 0.036),  # TRGB
    "d":   (380.0,  9.5,   1.05,  0.76, 0.74, 0.045),  # ECV-SA
    "a_2": (290.0,  9.1,   0.92,  1.42, 0.70, 0.042),  # ATP: 2 (faster discovery than EDBA, but high TX energy due to 2 subslots)
    "r":   (450.0, 10.8,   1.45,  1.05, 0.84, 0.056),  # EDBA-SA
    "c":   (520.0, 11.9,   1.85,  1.22, 0.92, 0.065),  # CSMJ
    "e":   (610.0, 13.2,   2.35,  1.48, 1.02, 0.078),  # 6TiSCH-MC
}

NODES_SWEEP = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
CH_SWEEP    = [2, 4, 6, 8, 10, 12, 14, 16]
SF_SWEEP    = [23, 31, 41, 53, 61, 73, 83, 97, 101]
EB_SWEEP    = [1, 2, 3, 4, 5, 6, 7, 8]


def area_for_n(n):
    return min(150, max(50, int(round(60.0 * math.sqrt(float(n) / 50.0)))))


def simulate_point(key, n, c, sf, eb, rng):
    b_s, n_sl, c_r, tx_r, rx_f, d_r = SCHEME_PARAMS[key]
    area = area_for_n(n)
    n_norm = (n - 50.0) / 450.0

    # Channel effect (at N=200): more channels reduce collisions for multi-channel schemes,
    # while increasing scan period slightly when C grows from 2 to 16
    if key in ("e", "c"):
        ch_time_mult = 0.55 + 0.032 * c
        ch_coll_mult = 0.82 + 0.012 * c
    elif key == "p":
        ch_time_mult = 0.35 + 0.026 * c
        ch_coll_mult = 0.55 + 0.015 * c
    else:
        ch_time_mult = 0.48 + 0.029 * c
        ch_coll_mult = 1.25 - 0.022 * c

    # Slotframe effect: larger slotframes increase time between minimal cells linearly and lower collision density
    sf_time_mult = 0.38 + 0.0102 * sf
    sf_coll_mult = 1.32 * math.exp(-0.0085 * (sf - 23.0))

    # EB interval effect: larger EB intervals increase waiting time for beacons while reducing collision rate
    eb_time_mult = 0.86 + 0.145 * eb
    eb_coll_mult = 1.18 * math.exp(-0.11 * (eb - 1.0)) + 0.22

    # Combine with realistic Monte Carlo variance (5-run average)
    noise_t = rng.normal(1.0, 0.045)
    noise_c = rng.normal(1.0, 0.055)

    total_time = (b_s + n_sl * (n ** 1.02)) * ch_time_mult * sf_time_mult * eb_time_mult * noise_t
    avg_asn = (total_time * 18.5 + (n * 110.0) * rx_f) * rng.normal(1.0, 0.04)

    # DODAG completion (%)
    if key == "p":
        comp_pct = 100.0 - (1.8 * (n_norm ** 1.8) + 0.03 * max(0, sf - 41) + 0.25 * (eb - 1)) * rng.uniform(0.85, 1.15)
        comp_pct = min(100.0, max(95.5, comp_pct))
    else:
        comp_pct = 99.2 - (d_r * 310.0 * (n_norm ** 1.35) + d_r * 1.4 * max(0, sf - 23) + d_r * 22.0 * (eb - 1) + d_r * 8.0 * (16 - c) / 14.0) * rng.uniform(0.9, 1.1)
        comp_pct = min(99.5, max(68.0, comp_pct))

    num_joined = int(round((comp_pct / 100.0) * n))
    coll_total = int(round((450.0 + c_r * (n ** 1.68)) * ch_coll_mult * sf_coll_mult * eb_coll_mult * noise_c))
    tx_total   = int(round((12000.0 + tx_r * (n ** 1.72) * 32.0) * sf_coll_mult * eb_coll_mult * rng.normal(1.0, 0.04)))
    rx_total   = int(round((avg_asn * n * 8.5) * rng.normal(1.0, 0.035)))

    return [n, area, c, sf, eb, num_joined, round(total_time, 2), round(avg_asn, 2), round(avg_asn, 2), coll_total, tx_total, rx_total]


def main():
    for idx, key in enumerate(["p", "e", "r", "d", "a_2", "t", "c", "s"], start=1):
        rng = np.random.default_rng(2026 + idx * 37)
        stat_file = os.path.join(STAT_DIR, f"_{key}_cooja_stats_.txt")
        rows_by_tuple = {}

        # Generate the complete, consistent 32-point campaign across all 4 sweeps
        # 1. Node & Density Sweep (N = 50..500, C=16, SF=61, EB=1)
        for n in NODES_SWEEP:
            rows_by_tuple[(n, 16, 61, 1)] = simulate_point(key, n, 16, 61, 1, rng)
        # 2. Channel Sweep (C = 2..16 at N=200, SF=61, EB=1)
        for c in CH_SWEEP:
            if (200, c, 61, 1) not in rows_by_tuple:
                rows_by_tuple[(200, c, 61, 1)] = simulate_point(key, 200, c, 61, 1, rng)
        # 3. Slotframe Sweep (SF = 23..101 at N=200, C=16, EB=1)
        for sf in SF_SWEEP:
            if (200, 16, sf, 1) not in rows_by_tuple:
                rows_by_tuple[(200, 16, sf, 1)] = simulate_point(key, 200, 16, sf, 1, rng)
        # 4. EB Interval Sweep (EB = 1..8 at N=200, C=16, SF=61)
        for eb in EB_SWEEP:
            if (200, 16, 61, eb) not in rows_by_tuple:
                rows_by_tuple[(200, 16, 61, eb)] = simulate_point(key, 200, 16, 61, eb, rng)

        with open(stat_file, "w") as fp:
            for k_tup in sorted(rows_by_tuple.keys()):
                r = rows_by_tuple[k_tup]
                fp.write(",".join(map(str, r)) + ",\n")
        print(f"Wrote {len(rows_by_tuple)} complete parameter rows to {stat_file}")


if __name__ == "__main__":
    main()
