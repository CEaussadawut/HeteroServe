#!/usr/bin/env python3
"""energy_phase_run.py — one board's prefill-only and decode-only energy, by the protocol in testbed/energy.md.

The strip is logged at 5 Hz (shelly_strip.py, read-only) for the whole session. An idle window comes first and another
one last. Between them run K cycles of a prefill-only llama-bench invocation (-p N -n 0) and a decode-only one
(-p 0 -n N), each with -r >= 3 and --progress. Every line the board prints is stamped with the laptop's time.time() as it
arrives (stdbuf -oL -eL on the board, stderr merged into stdout on the board), so the windows are on the strip log's
clock. A sampler on the board records get_throttled, the temperature, the ARM clock and the fan state every 5 s for the
whole session (house rule: get_throttled with every Pi run).

Preflight refuses to start if the model's SHA-256 on the board differs from --sha256, if a llama process is already
running there, or if the strip's name for --outlet is not --outlet-name. Nothing is scored here:
score_energy_phase_run.py reads the run folder. Nothing is filtered: every line goes to a file.

Usage:
  energy_phase_run.py --node pi5 --outlet 1 --model '~/models/tinyllama-1.1b-q4km.gguf' --sha256 HEX64 \\
      --pp 512:15 --tg 128:13 --cycles 3 [--threads 4] [--idle-before 300] [--idle-after 300] [--gap 10] [--out DIR]
  --local runs everything on this machine instead of over ssh (the laptop, 2026-10-05: it has no ssh server);
  --bench-args adds llama-bench arguments (the Jetson's CPU rows: -ngl 0); --require-guard refuses to start unless the
  Jetson's power guard is running there.
"""
import argparse
import datetime
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STRIP = os.path.join(ROOT, "testbed", "scripts", "shelly_strip.py")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=6"]
LLAMA_PROCS = "llama-bench|llama-cli|llama-server|rpc-server|ggml-rpc-server"

SAMPLER = r'''
end=$((SECONDS+$1))
while [ $SECONDS -lt $end ]; do
  if command -v vcgencmd >/dev/null; then
    echo "$(date +%s.%3N) $(vcgencmd get_throttled) $(vcgencmd measure_temp) $(vcgencmd measure_clock arm) fan=$(cat /sys/class/thermal/cooling_device0/cur_state 2>/dev/null) loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  else
    b=$(ls -d /sys/class/power_supply/BAT* 2>/dev/null | head -1)
    echo "$(date +%s.%3N) thermal_mC=$(cat /sys/class/thermal/thermal_zone*/temp | tr '\n' ',') freq_khz=$(awk '{s+=$1} END {if (NR) print int(s/NR)}' /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null) bat=${b:+$(cat $b/status 2>/dev/null)/$(cat $b/capacity 2>/dev/null)/$(cat $b/power_now 2>/dev/null)} loadavg=$(cut -d' ' -f1 /proc/loadavg)"
  fi
  sleep 5
done
'''


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class Run:
    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.markers = open(os.path.join(out, "markers.tsv"), "a")

    def mark(self, label):
        t = time.time()
        self.markers.write("%.6f\t%s\n" % (t, label))
        self.markers.flush()
        print("%s %s" % (datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S"), label), flush=True)
        return t


LOCAL = False


def argv(node, cmd, stdin=False):
    """The command line that runs `cmd` (a shell string) on `node`: over ssh, or here with --local."""
    if LOCAL:
        return ["bash", "-c", cmd]
    return SSH + ([] if stdin else ["-n"]) + [node, cmd]


def ssh_run(node, cmd, timeout=120):
    p = subprocess.run(argv(node, cmd), capture_output=True, text=True, timeout=timeout,
                       stdin=subprocess.DEVNULL)
    return p.returncode, p.stdout, p.stderr


def stamp_stream(stream, path):
    """Write each line of a text stream to path, prefixed by the laptop's time.time() on arrival."""
    with open(path, "a") as f:
        for line in stream:
            f.write("%.6f\t%s\n" % (time.time(), line.rstrip("\n")))
            f.flush()


def preflight(a, run):
    pre = {"at": now_iso()}
    info = subprocess.run([sys.executable, STRIP, "info", "--json"], capture_output=True, text=True, timeout=30)
    with open(os.path.join(run.out, "strip_info_before.json"), "w") as f:
        f.write(info.stdout)
    try:
        sinfo = json.loads(info.stdout)
    except ValueError:
        raise SystemExit("refused: the strip did not answer `info --json` (%s)" % info.stderr.strip()[:200])
    name = mapped = None
    for o in sinfo.get("outlets", []):
        if str(o.get("outlet")) == str(a.outlet):
            name, mapped = o.get("name"), o.get("map")
    pre["strip_outlet_name"], pre["strip_json_map"] = name, mapped
    if name != a.outlet_name or mapped != a.outlet_name:
        raise SystemExit("refused: strip outlet %s is named %r (strip.json map %r), expected %r"
                         % (a.outlet, name, mapped, a.outlet_name))
    rc, out, err = ssh_run(a.node, "hostname; cat /proc/loadavg; free -m | sed -n 2p; "
                                   "(vcgencmd get_throttled; vcgencmd measure_temp) 2>/dev/null; "
                                   "pgrep -a -x '%s' || echo 'no llama process'" % LLAMA_PROCS)
    pre["node_state"] = out.strip().splitlines()
    if "no llama process" not in out:
        raise SystemExit("refused: a llama process is already running on %s:\n%s" % (a.node, out))
    if a.require_guard:
        rc, out, err = ssh_run(a.node, "pgrep -f jetson_power_guard.sh >/dev/null && echo guard=up || echo guard=DOWN; "
                                       "cat /var/lib/nvpmodel/status 2>/dev/null; echo; tail -n 1 ~/heteroserve/logs/power-guard.log 2>/dev/null")
        pre["guard"] = out.strip().splitlines()
        if "guard=up" not in out:
            raise SystemExit("refused: --require-guard, and the power guard is not running on %s" % a.node)
        rc, tout, err = ssh_run(a.node, "tr -d '\\000' < ~/heteroserve/logs/power-guard.log | grep -a -c TRIP")
        pre["guard_trips"] = int(tout.strip()) if tout.strip().isdigit() else None
    rc, out, err = ssh_run(a.node, "cd %s && git rev-parse HEAD && git status --porcelain | wc -l && %s --version 2>&1 | head -2"
                           % (a.llama_dir, a.bench))
    pre["llama_cpp"] = out.strip().splitlines()
    rc, out, err = ssh_run(a.node, "sha256sum %s" % a.model, timeout=300)
    got = out.split()[0] if out.split() else None
    pre["model_sha256"] = got
    pre["model_sha256_ok"] = (got == a.sha256)
    if got != a.sha256:
        raise SystemExit("refused: %s on %s hashes %s, the pin is %s" % (a.model, a.node, got, a.sha256))
    return pre


def bench_cmd(a, phase, n, reps):
    pn = "-p %d -n 0" % n if phase == "pp" else "-p 0 -n %d" % n
    return ("cd %s && exec stdbuf -oL -eL %s -m %s -t %d %s -r %d %s--progress -oe jsonl 2>&1"
            % (a.llama_dir, a.bench, a.model, a.threads, pn, reps, (a.bench_args + " ") if a.bench_args else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--node", required=True, help="ssh alias of the board")
    ap.add_argument("--outlet", type=int, required=True, help="the strip outlet the board is on")
    ap.add_argument("--outlet-name", help="the strip's name for that outlet (default: --node)")
    ap.add_argument("--model", required=True, help="model path on the board (~ allowed)")
    ap.add_argument("--sha256", required=True, help="the model's pinned SHA-256 (testbed/models.md)")
    ap.add_argument("--llama-dir", default="~/llama.cpp")
    ap.add_argument("--bench", default="./build/bin/llama-bench", help="relative to --llama-dir")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--pp", required=True, help="N:R, prefill-only -p N -n 0 -r R")
    ap.add_argument("--tg", required=True, help="N:R, decode-only -p 0 -n N -r R")
    ap.add_argument("--cycles", type=int, default=3, help="prefill then decode, this many times")
    ap.add_argument("--idle-before", type=float, default=300.0)
    ap.add_argument("--idle-after", type=float, default=300.0)
    ap.add_argument("--settle", type=float, default=30.0, help="s between preflight and the idle window")
    ap.add_argument("--gap", type=float, default=10.0, help="s between invocations")
    ap.add_argument("--tail", type=float, default=5.0, help="s the strip keeps logging after the last window, so the log "
                    "brackets it (2026-10-05: without it the idle-after energies were withheld, NOT_COVERED)")
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--local", action="store_true", help="run on this machine, not over ssh")
    ap.add_argument("--bench-args", default="", help="extra llama-bench arguments, e.g. '-ngl 0'")
    ap.add_argument("--require-guard", action="store_true", help="refuse unless jetson_power_guard.sh runs on the node")
    ap.add_argument("--jetson-clocks", action="store_true", help="the Jetson's protocol (jetson_bench_tinyllama.sh, run_pool.sh): "
                    "store its clocks, lock them for the whole session (idle windows included) if it is in MAXN, restore after")
    ap.add_argument("--out")
    a = ap.parse_args()
    global LOCAL
    LOCAL = a.local
    a.outlet_name = a.outlet_name or a.node
    specs = {}
    for ph in ("pp", "tg"):
        m = re.fullmatch(r"(\d+):(\d+)", getattr(a, ph))
        if not m:
            raise SystemExit("--%s must be N:R" % ph)
        n, r = int(m.group(1)), int(m.group(2))
        if r < 3:
            raise SystemExit("refused: -r %d; the house rule is -r >= 3" % r)
        specs[ph] = (n, r)
    out = a.out or os.path.join(ROOT, "testbed", "results", "energy_phase_%s_%s" % (a.node, time.strftime("%Y%m%d_%H%M%S")))
    run = Run(out)
    manifest = {"started": now_iso(), "args": vars(a), "out": os.path.relpath(out, ROOT), "invocations": []}
    logger = sampler = None
    current = None
    locked = None
    try:
        manifest["preflight"] = preflight(a, run)
        run.mark("preflight ok")
        if a.jetson_clocks:
            store = "~/heteroserve/logs/jetson_clocks.stored.%s" % time.strftime("%Y%m%d_%H%M%S")
            rc, o, e = ssh_run(a.node, "grep -q '^pmode:0000' /var/lib/nvpmodel/status || { echo NOT_MAXN; exit 3; }; "
                                       "sudo -n jetson_clocks --store %s && sudo -n jetson_clocks && echo LOCKED" % store)
            manifest["jetson_clocks"] = {"store": store, "out": (o + e).strip().splitlines()}
            if "LOCKED" not in o:
                raise SystemExit("refused: could not lock the Jetson's clocks: %s" % (o + e).strip()[:200])
            locked = store
            run.mark("jetson clocks locked")
        logger = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(out, "strip.csv"), "--hz", str(a.hz)],
                                  stdout=open(os.path.join(out, "strip_logger.stdout"), "w"),
                                  stderr=open(os.path.join(out, "strip_logger.stderr"), "w"))
        run.mark("logger start")
        planned = a.settle + a.idle_before + a.idle_after + 900 * a.cycles * 2
        sampler = subprocess.Popen(["bash", "-s", str(int(planned))] if LOCAL else SSH + [a.node, "bash -s %d" % planned],
                                   stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        sampler.stdin.write(SAMPLER)
        sampler.stdin.close()
        threading.Thread(target=stamp_stream, args=(sampler.stdout, os.path.join(out, "node_sampler.log")), daemon=True).start()
        run.mark("sampler start")
        time.sleep(a.settle)
        if logger.poll() is not None:
            raise SystemExit("the strip logger exited early (code %s): see strip_logger.stderr" % logger.returncode)
        run.mark("idle-before start")
        time.sleep(a.idle_before)
        run.mark("idle-before end")
        for k in range(1, a.cycles + 1):
            for ph in ("pp", "tg"):
                n, r = specs[ph]
                label = "%s%d" % (ph, k)
                cmd = bench_cmd(a, ph, n, r)
                run.mark("bench start %s" % label)
                current = subprocess.Popen(argv(a.node, cmd, stdin=True), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, text=True, bufsize=1)
                stamp_stream(current.stdout, os.path.join(out, "bench_%s.log" % label))
                rc = current.wait()
                current = None
                run.mark("bench end %s exit=%d" % (label, rc))
                manifest["invocations"].append({"label": label, "phase": ph, "n": n, "reps": r, "cmd": cmd, "exit": rc})
                time.sleep(a.gap)
        run.mark("idle-after start")
        time.sleep(a.idle_after)
        run.mark("idle-after end")
        time.sleep(a.tail)
    finally:
        if current is not None:             # interrupted mid-invocation: the only llama process there is ours (preflight)
            ssh_run(a.node, "pkill -x llama-bench || true")
            run.mark("interrupted: llama-bench killed on %s" % a.node)
        if sampler is not None:
            sampler.terminate()
        if locked:      # restore only what this run locked, and only while the board is still in MAXN (run_pool.sh's rule)
            rc, o, e = ssh_run(a.node, "grep -q '^pmode:0000' /var/lib/nvpmodel/status && sudo -n jetson_clocks --restore %s "
                                       "&& echo RESTORED || echo NOT_RESTORED" % locked)
            manifest.setdefault("jetson_clocks", {})["restore"] = (o + e).strip().splitlines()
            run.mark("jetson clocks restore: %s" % (o.strip().splitlines() or ["?"])[-1])
        if logger is not None and logger.poll() is None:
            logger.send_signal(signal.SIGINT)
            try:
                logger.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.kill()
        run.mark("logger stop")
        rc, o, e = ssh_run(a.node, "hostname; cat /proc/loadavg; (vcgencmd get_throttled; vcgencmd measure_temp) 2>/dev/null; "
                                   "dmesg 2>/dev/null | grep -i -E 'under-?voltage|throttl' | tail -3")
        manifest["postflight_node_state"] = o.strip().splitlines()
        if a.require_guard:
            rc, o, e = ssh_run(a.node, "pgrep -f jetson_power_guard.sh >/dev/null && echo guard=up || echo guard=DOWN; "
                                       "cat /var/lib/nvpmodel/status; echo; tr -d '\\000' < ~/heteroserve/logs/power-guard.log | grep -a -c TRIP")
            manifest["postflight_guard"] = o.strip().splitlines()
        info = subprocess.run([sys.executable, STRIP, "info", "--json"], capture_output=True, text=True, timeout=30)
        with open(os.path.join(out, "strip_info_after.json"), "w") as f:
            f.write(info.stdout)
        manifest["finished"] = now_iso()
        with open(os.path.join(out, "run.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        run.mark("run end")


if __name__ == "__main__":
    main()
