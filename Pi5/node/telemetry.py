#!/usr/bin/env python3
"""Pi 5 node telemetry sampler.

Records the things the INA226 rail cannot tell you but that a power number is
meaningless without: SoC temperature, the clock the governor actually delivered,
throttle flags from the firmware, and memory pressure. Written as CSV on the same
UTC time base as the harness and the Pico 2 DAQ, so the three streams join on
`ts_utc` at analysis time.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envfile  # noqa: E402

THERMAL = Path("/sys/class/thermal/thermal_zone0/temp")
CPU0_FREQ = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
GOVERNOR = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")

FIELDS = [
    "ts_utc", "mono_ns", "node_id",
    "soc_temp_c", "cpu_mhz", "governor",
    "throttled_hex", "under_voltage_now", "freq_capped_now", "throttled_now",
    "mem_avail_mb", "mem_used_mb", "swap_used_mb",
    "load1", "rpc_rss_mb",
]


def read_int(p, default=-1):
    try:
        return int(p.read_text().strip())
    except (OSError, ValueError):
        return default


def read_throttled():
    """vcgencmd get_throttled -- the firmware's own record of under-voltage and
    thermal capping. A cell whose trace has any bit set is reported as throttled
    rather than averaged in silently (proposal risk R5)."""
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2
        ).stdout.strip()
        return int(out.split("=", 1)[1], 16)
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        # Narrow, so a genuine bug here surfaces instead of being reported as -1 and then
        # rendered as "unknown throttle state" forever. -1 means "could not read", and the
        # agent's readiness gate treats that as blocking, not as "not throttled".
        return -1


def meminfo():
    total = avail = swap_total = swap_free = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, v = line.partition(":")
            kb = int(v.strip().split()[0])
            if k == "MemTotal":
                total = kb
            elif k == "MemAvailable":
                avail = kb
            elif k == "SwapTotal":
                swap_total = kb
            elif k == "SwapFree":
                swap_free = kb
    except (OSError, ValueError, IndexError):
        pass
    return avail // 1024, (total - avail) // 1024, (swap_total - swap_free) // 1024


def rpc_rss_mb():
    """RSS of the rpc-server, so a placement that quietly overshot its advertised
    memory budget is visible in the trace instead of only as an OOM."""
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if "rpc-server" not in (proc / "comm").read_text():
                continue
            for line in (proc / "status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
        except OSError:
            continue
    return -1


def sample(node_id):
    thr = read_throttled()
    avail, used, swap = meminfo()
    try:
        load1 = os.getloadavg()[0]
    except OSError:
        load1 = -1.0
    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "mono_ns": time.monotonic_ns(),
        "node_id": node_id,
        "soc_temp_c": round(read_int(THERMAL) / 1000.0, 2),
        "cpu_mhz": read_int(CPU0_FREQ) // 1000,
        "governor": GOVERNOR.read_text().strip() if GOVERNOR.exists() else "?",
        "throttled_hex": ("0x%x" % thr) if thr >= 0 else "",
        # bit 0 under-voltage now, bit 1 arm frequency capped now, bit 2 currently
        # throttled -- see the Raspberry Pi firmware documentation.
        "under_voltage_now": int(bool(thr & 0x1)) if thr >= 0 else "",
        "freq_capped_now": int(bool(thr & 0x2)) if thr >= 0 else "",
        "throttled_now": int(bool(thr & 0x4)) if thr >= 0 else "",
        "mem_avail_mb": avail,
        "mem_used_mb": used,
        "swap_used_mb": swap,
        "load1": round(load1, 2),
        "rpc_rss_mb": rpc_rss_mb(),
    }


def main():
    env = envfile.load()
    ap = argparse.ArgumentParser()
    ap.add_argument("--hz", type=float, default=5.0, help="sample rate")
    ap.add_argument("--out", default=env.get("DATA_DIR", "/opt/heteroserve/data") + "/telemetry")
    ap.add_argument("--tag", default="", help="run tag appended to the filename")
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = run until SIGTERM")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = ("_" + args.tag) if args.tag else ""
    path = outdir / ("%s_telemetry_%s%s.csv" % (env["NODE_ID"], stamp, tag))

    period = 1.0 / args.hz
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("now", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("now", True))

    deadline = time.monotonic() + args.seconds if args.seconds > 0 else None
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        next_t = time.monotonic()
        while not stop["now"] and (deadline is None or time.monotonic() < deadline):
            w.writerow(sample(env["NODE_ID"]))
            fh.flush()
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))

    sys.stderr.write("telemetry written to %s\n" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
