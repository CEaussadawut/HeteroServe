#!/usr/bin/env python3
"""Synthesise a tegrastats_logger-shaped power CSV: idle -> prefill -> decode -> idle.

There is no Jetson on the development machine, so this is how the analysis side
(energy_window.py) gets exercised without hardware. It writes the exact FIELDS
schema tegrastats_logger.py writes, including the running trapezoidal energy
integral, so energy_window.py cannot tell it from a real capture.

  ./make_synthetic_power_csv.py /tmp/power.csv
  ./energy_window.py /tmp/power.csv --marker decode --idle-mw 1900

Expected for the decode window: 11.50 s, 24 samples, module 55.2 J,
mean/peak 4800.0 mW, active 33.35 J with the 1900 mW floor removed.

The numbers are plausible-looking but INVENTED. Never let one of these files
reach a results table -- it is a test fixture, not a measurement.
"""
import csv
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tegrastats_logger import FIELDS

T0 = 1789226343.0
DT = 0.5  # 500 ms sampling

# (marker, seconds, module_mw, gpu_mw, cpu_mw)
PHASES = [
    ("idle",    10.0, 1900.0,    0.0,  240.0),
    ("prefill",  6.0, 7400.0, 3100.0, 2600.0),
    ("decode",  12.0, 4800.0, 1500.0, 1900.0),
    ("idle",     4.0, 1900.0,    0.0,  240.0),
    # A fused -pg window: prefill and decode inside one marker. Appended AFTER
    # the decode window so the documented decode numbers (11.50 s, 24 samples,
    # 55.2 J module, 33.35 J active) are untouched.
    ("combined", 8.0, 6200.0, 2400.0, 2200.0),
    ("idle",     4.0, 1900.0,    0.0,  240.0),
]


def build():
    rows = []
    t = T0
    e_in = e_gpu = e_cpu = 0.0
    prev = None
    for marker, dur, p_in, p_gpu, p_cpu in PHASES:
        for _ in range(int(dur / DT)):
            if prev is not None:
                # trapezoidal, matching the logger's running integral
                e_in += (prev[0] + p_in) / 2.0 * DT / 1000.0
                e_gpu += (prev[1] + p_gpu) / 2.0 * DT / 1000.0
                e_cpu += (prev[2] + p_cpu) / 2.0 * DT / 1000.0
            d = dict.fromkeys(FIELDS)
            idle = marker == "idle"
            d.update({
                "ts_unix": "%.3f" % t,
                "ts_utc": datetime.fromtimestamp(t, timezone.utc).isoformat(
                    timespec="milliseconds"),
                "node_id": "synthetic", "marker": marker,
                "pow_in_mw": p_in, "pow_gpu_mw": p_gpu, "pow_cpu_mw": p_cpu,
                "energy_in_j": "%.6f" % e_in,
                "energy_gpu_j": "%.6f" % e_gpu,
                "energy_cpu_j": "%.6f" % e_cpu,
                "ram_used_mb": 1969, "ram_total_mb": 3956,
                "swap_used_mb": 0 if idle else 120,
                "cpu_util_pct": 5.75 if idle else 82.0,
                "cpu_util_per_core": "10|6|4|3", "cpu_freq_mhz": 1479,
                "gpu_util_pct": 0 if idle else 91, "gpu_freq_mhz": 921,
                "emc_util_pct": 6,
                "temp_cpu_c": 30.0 if idle else 47.5,
                "temp_gpu_c": 29.0 if idle else 45.0,
                "temp_thermal_c": 29.75 if idle else 46.25,
                "raw": "synthetic",
            })
            rows.append(d)
            prev = (p_in, p_gpu, p_cpu)
            t += DT
    return rows


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_power.csv"
    rows = build()
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
