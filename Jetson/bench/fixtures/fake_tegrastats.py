#!/usr/bin/env python3
"""Fake Nano tegrastats. Power follows $HS_FAKE_PHASE, so a test can tell load
energy apart from compute energy, and prefill apart from decode."""
import os, sys, time
interval = 1000
if "--interval" in sys.argv:
    interval = int(sys.argv[sys.argv.index("--interval") + 1])
phase_file = os.environ["HS_FAKE_PHASE"]
MW = {"idle": (1900, 0, 238), "load": (5200, 200, 3000),
      "prefill": (7400, 3100, 2600), "decode": (4800, 1500, 1900)}
# HS_FAKE_TEMP_RAMP makes the board heat while working and cool while idle, so
# the closed-loop cooldown gate can actually be exercised: with a constant
# temperature the gate passes on its first poll and proves nothing. Value is
# degrees per second of heating; cooling is a tenth of that, which is roughly the
# asymmetry a fanless heatsink shows.
RAMP = float(os.environ.get("HS_FAKE_TEMP_RAMP", "0"))
BASE_C = float(os.environ.get("HS_FAKE_TEMP_BASE_C", "46.2"))
temp_c = BASE_C
last = time.time()

while True:
    try:
        ph = open(phase_file, encoding="utf-8").read().strip() or "idle"
    except OSError:
        ph = "idle"
    p_in, p_gpu, p_cpu = MW.get(ph, MW["idle"])
    if RAMP:
        now = time.time()
        dt, last = now - last, now
        if ph == "idle":
            temp_c = max(BASE_C, temp_c - RAMP * 0.1 * dt)
        else:
            temp_c += RAMP * dt
    sys.stdout.write(
        "RAM 1969/3956MB (lfb 22x4MB) SWAP 120/1978MB (cached 0MB) IRAM 0/252kB(lfb 252kB) "
        "CPU [42%%@1479,38%%@1479,11%%@1479,9%%@1479] EMC_FREQ 21%%@1600 GR3D_FREQ 63%%@921 "
        "APE 25 PLL@46.5C CPU@%.1fC PMIC@100C GPU@45.5C AO@52C thermal@%.1fC "
        "POM_5V_IN %d/%d POM_5V_GPU %d/%d POM_5V_CPU %d/%d\n"
        % (temp_c, temp_c, p_in, p_in, p_gpu, p_gpu, p_cpu, p_cpu))
    sys.stdout.flush()
    time.sleep(interval / 1000.0)
