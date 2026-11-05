#!/usr/bin/env python3
"""
Log Jetson tegrastats to CSV with UTC timestamps, for energy-per-token accounting.

tegrastats on a Nano dev kit emits lines like:

  RAM 1969/3956MB (lfb 22x4MB) SWAP 0/1978MB (cached 0MB) IRAM 0/252kB(lfb 252kB)
  CPU [10%@1479,6%@1479,4%@1479,3%@1479] EMC_FREQ 6%@1600 GR3D_FREQ 0%@76
  PLL@27C CPU@30C PMIC@100C GPU@29C AO@36C thermal@29.75C
  POM_5V_IN 1904/1904 POM_5V_GPU 0/0 POM_5V_CPU 238/238

POM_* fields are milliwatts as <instantaneous>/<running average>, read from the
carrier board's INA3221 (B01: /sys/bus/i2c/drivers/ina3221x/6-0040). Because that
chip is on the CARRIER, POM_5V_IN is the whole dev kit's 5 V input -- module plus
carrier plus anything drawing off USB or HDMI -- NOT module-only power. Only the
wall adapter's conversion loss sits outside it. (Module-only is a TX2/Xavier
property; do not carry it over to the Nano.) POM_5V_GPU and POM_5V_CPU are SoC
rails and are what the prefill-vs-decode attribution rests on.

There is no second instrument: the cluster has no external inline meter, so this
logger is the only power signal, and across-device comparison is out of scope.
Keep the peripheral set identical across runs so POM_5V_IN's carrier offset is
at least constant.

What the sample rate bounds, and what it does not
  The interval sets the resolution on every window boundary and on any structure
  inside a window. It does NOT average out with repetitions: every repetition is
  sampled on the same grid relative to the workload, so a transient shorter than
  the interval is missed the same way each time and the spread ACROSS repetitions
  comes back small -- reassuringly, and wrongly. A tight spread is therefore not
  evidence the rate was sufficient.
  Two columns on every joined row carry the bound instead of leaving it to be
  re-derived: `sample_gap_max_s`, the longest hole the integral had to cross on a
  straight line, and `pow_in_plateau_frac`, how often the rail simply repeated
  its previous reading -- a high fraction means tegrastats is re-printing a value
  the INA has not refreshed, so the effective bandwidth is below the nominal rate.
  Neither detects ALIASING against the token cadence; nothing available here can.

What is NOT validated
  The INA3221's absolute accuracy. There is no reference instrument on this
  cluster, so every joule reported is as accurate as that chip and no better, and
  that cannot be checked from inside the node. `joules_vs_mean_pct` on each row
  compares the running integral against mean power x duration -- two arithmetics
  over the SAME samples, so it catches a bookkeeping error and proves nothing
  about the samples themselves. tegrastats' own running average is not a second
  instrument either: it is the same ADC, averaged.

Usage
  ./tegrastats_logger.py --out power.csv --hz 5
  ./tegrastats_logger.py --out power.csv --duration 120 --echo
  ./tegrastats_logger.py --out power.csv --marker-file /tmp/hs_marker

Stop with SIGINT/SIGTERM; the CSV is flushed per line, so a kill never costs
you the run.
"""
import argparse
import csv
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

RE_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
RE_CPU = re.compile(r"CPU \[([^\]]+)\]")
RE_EMC = re.compile(r"EMC_FREQ (\d+)%@?(\d+)?")
RE_GR3D = re.compile(r"GR3D_FREQ (\d+)%@?(\d+)?")
RE_POM = re.compile(r"(POM_[A-Z0-9_]+) (\d+)/(\d+)")
RE_TEMP = re.compile(r"([A-Za-z0-9]+)@(-?[\d.]+)C")
RE_CORE = re.compile(r"(\d+)%@(\d+)")

FIELDS = [
    "ts_unix", "ts_utc", "node_id", "marker",
    "pow_in_mw", "pow_gpu_mw", "pow_cpu_mw",
    "energy_in_j", "energy_gpu_j", "energy_cpu_j",
    "ram_used_mb", "ram_total_mb", "swap_used_mb",
    "cpu_util_pct", "cpu_util_per_core", "cpu_cores_online", "cpu_cores_seen",
    "cpu_freq_mhz",
    "gpu_util_pct", "gpu_freq_mhz", "emc_util_pct",
    "temp_cpu_c", "temp_gpu_c", "temp_thermal_c",
    "raw",
]


def parse(line):
    """Parse one tegrastats line into a FIELDS-keyed dict.

    Unknown or missing fields stay None so that a different L4T release with a
    slightly different line format degrades to empty columns instead of
    crashing a measurement run that may already be an hour in.
    """
    d = dict.fromkeys(FIELDS)
    d["raw"] = line.strip()

    m = RE_RAM.search(line)
    if m:
        d["ram_used_mb"], d["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = RE_SWAP.search(line)
    if m:
        d["swap_used_mb"] = int(m.group(1))

    m = RE_CPU.search(line)
    if m:
        utils, freqs, per_core = [], [], []
        seen = 0
        for core in m.group(1).split(","):
            core = core.strip()
            seen += 1
            mm = RE_CORE.match(core)
            if mm:
                utils.append(float(mm.group(1)))
                freqs.append(int(mm.group(2)))
                per_core.append(str(int(float(mm.group(1)))))
            else:
                # An offline core reads as "off". It is NOT 0% busy -- it is not
                # there. Counting it as 0 and dividing by the printed core count
                # halved cpu_util_pct in the 5 W mode, which parks two of the
                # four cores: a fully saturated 2-core board read as 50%. The
                # per-core string keeps the literal "off" so the power mode stays
                # legible in the row rather than only in `raw`.
                per_core.append("off")
        if utils:
            d["cpu_util_pct"] = round(sum(utils) / len(utils), 2)
        elif seen:
            # every core parked: no utilisation to report, and 0.0 would be a
            # measurement rather than the absence of one
            d["cpu_util_pct"] = None
        if per_core:
            d["cpu_util_per_core"] = "|".join(per_core)
        d["cpu_cores_online"] = len(utils)
        d["cpu_cores_seen"] = seen
        if freqs:
            d["cpu_freq_mhz"] = max(freqs)

    m = RE_EMC.search(line)
    if m:
        d["emc_util_pct"] = int(m.group(1))
    m = RE_GR3D.search(line)
    if m:
        d["gpu_util_pct"] = int(m.group(1))
        if m.group(2):
            d["gpu_freq_mhz"] = int(m.group(2))

    for name, inst, _avg in RE_POM.findall(line):
        if name.endswith("_IN"):
            d["pow_in_mw"] = int(inst)
        elif name.endswith("_GPU"):
            d["pow_gpu_mw"] = int(inst)
        elif name.endswith("_CPU"):
            d["pow_cpu_mw"] = int(inst)

    for name, val in RE_TEMP.findall(line):
        n = name.upper()
        if n == "CPU":
            d["temp_cpu_c"] = float(val)
        elif n == "GPU":
            d["temp_gpu_c"] = float(val)
        elif n == "THERMAL":
            d["temp_thermal_c"] = float(val)
    return d


def read_marker(path):
    """Current phase label written by the benchmark driver, or "".

    A plain file rather than a FIFO: the driver overwrites it, the logger reads
    whatever is there. No blocking, no lock, and a dead driver simply leaves the
    last label in place instead of stalling the logger.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--hz", type=float, default=5.0, help="samples per second (default 5)")
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until killed")
    ap.add_argument("--node-id", default=os.environ.get("NODE_ID", "unknown"))
    ap.add_argument("--marker-file", default=os.environ.get("HS_MARKER_FILE", ""),
                    help="file whose contents label each sample")
    ap.add_argument("--tegrastats", default="tegrastats")
    ap.add_argument("--echo", action="store_true", help="also print each sample")
    args = ap.parse_args()

    interval_ms = max(50, int(round(1000.0 / args.hz)))

    try:
        proc = subprocess.Popen(
            [args.tegrastats, "--interval", str(interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            # encoding pinned: universal_newlines alone decodes with the locale
            # encoding, ASCII on a C-locale 3.6, and one stray byte in a
            # tegrastats line would then kill the logger mid-sweep. A mangled
            # sample must never be fatal -- it is one row of many.
            universal_newlines=True, encoding="utf-8", errors="replace", bufsize=1,
        )
    except OSError as exc:
        sys.exit("cannot start %s: %s" % (args.tegrastats, exc))

    stop = {"flag": False}

    def _stop(_sig, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    t0 = time.time()
    # Running trapezoidal integral per rail, in joules. Carrying the integral in
    # the CSV means any window's energy is a single subtraction downstream --
    # robust even if the analysis later resamples or drops rows.
    acc = {"in": 0.0, "gpu": 0.0, "cpu": 0.0}
    prev = {"t": None, "in": None, "gpu": None, "cpu": None}
    rails = (("in", "pow_in_mw", "energy_in_j"),
             ("gpu", "pow_gpu_mw", "energy_gpu_j"),
             ("cpu", "pow_cpu_mw", "energy_cpu_j"))

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    n = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for line in proc.stdout:
            if stop["flag"]:
                break
            now = time.time()
            d = parse(line)
            d["ts_unix"] = round(now, 3)
            d["ts_utc"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            d["node_id"] = args.node_id
            d["marker"] = read_marker(args.marker_file) if args.marker_file else ""

            for key, pcol, ecol in rails:
                p = d[pcol]
                if p is None:
                    continue
                if prev["t"] is not None and prev[key] is not None:
                    dt = now - prev["t"]
                    # Guard against a stalled reader letting one sample dominate
                    # the integral.
                    if 0 < dt < 5:
                        acc[key] += (p + prev[key]) / 2.0 / 1000.0 * dt
                prev[key] = p
                d[ecol] = round(acc[key], 4)
            prev["t"] = now

            writer.writerow(d)
            fh.flush()
            n += 1
            if args.echo:
                print("%s  %6s mW  cpu %5s%%  gpu %4s%%  %s" % (
                    d["ts_utc"], d["pow_in_mw"], d["cpu_util_pct"],
                    d["gpu_util_pct"], d["marker"]), flush=True)

            if args.duration and (now - t0) >= args.duration:
                break

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("wrote %s (%d samples, %.1f s)" % (args.out, n, time.time() - t0),
          file=sys.stderr)


if __name__ == "__main__":
    main()
