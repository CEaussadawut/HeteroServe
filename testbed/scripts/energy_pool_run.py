#!/usr/bin/env python3
"""energy_pool_run.py — pooled cells' energy: run_pool.sh cells inside one strip log, with idle windows around them.

The strip is logged at 5 Hz (shelly_strip.py, read-only) for the whole session. An idle window comes first and another
one last. In between, each listed cell runs exactly as every pool cell runs: `CELL=<id> PRED=<registration> bash
testbed/scripts/run_pool.sh`, whose gates, workers, memory probes, Jetson clock lock and EXIT-trap cleanup are its own,
and whose raw log it keeps (testbed/results/pool_<cell>_<stamp>.log). Every line run_pool.sh prints is also stamped
here with the laptop's time.time() as it arrives. The host's llama-bench --progress lines and its jsonl results come
through that stream, so score_energy_pool_run.py can cut each cell's prefill window and decode window on the strip
log's clock.

Preflight refuses to start if a cell is missing from the registration or was skipped there, or if the strip's name for
a pool node's outlet differs from the node. A cell that fails (run_pool.sh's gates refuse it, or llama-bench exits
non-zero) is recorded and the session goes on: reported as found.

Usage:
  energy_pool_run.py --pred testbed/dataset/pool_energy_predictions_<stamp>.json \\
      --cells tinyllama.rule,tinyllama.even,tinyllama.rule,... --outlets pi4=0,jetson=2 \\
      [--idle-before 300] [--idle-after 120] [--settle 30] [--gap 20] [--tail 5] [--out DIR]
"""
import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STRIP = os.path.join(ROOT, "testbed", "scripts", "shelly_strip.py")
RUN_POOL = os.path.join(ROOT, "testbed", "scripts", "run_pool.sh")


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pred", required=True, help="the registration (run_pool.sh's PRED)")
    ap.add_argument("--cells", required=True, help="comma list, in run order (repeat a cell to repeat it)")
    ap.add_argument("--outlets", required=True, help="node=outlet,... for every node the cells use")
    ap.add_argument("--idle-before", type=float, default=300.0)
    ap.add_argument("--idle-after", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=30.0)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--tail", type=float, default=5.0)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--out")
    a = ap.parse_args()
    outlets = {k: int(v) for k, v in (x.split("=") for x in a.outlets.split(","))}
    reg = json.load(open(a.pred))
    cells = a.cells.split(",")
    for c in cells:
        cfg = reg["configs"].get(c)
        if cfg is None or "skipped" in cfg:
            raise SystemExit("refused: cell %s is %s in %s" % (c, "skipped" if cfg else "missing", a.pred))
        for p in cfg["pipeline"]:
            if p["node"] not in outlets:
                raise SystemExit("refused: cell %s uses %s, which has no --outlets entry" % (c, p["node"]))
    out = a.out or os.path.join(ROOT, "testbed", "results", "energy_pool_%s" % time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)
    mk = open(os.path.join(out, "markers.tsv"), "a")

    def mark(label):
        t = time.time()
        mk.write("%.6f\t%s\n" % (t, label)); mk.flush()
        print("%s %s" % (datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S"), label), flush=True)

    info = subprocess.run([sys.executable, STRIP, "info", "--json"], capture_output=True, text=True, timeout=30)
    open(os.path.join(out, "strip_info_before.json"), "w").write(info.stdout)
    names = {str(o["outlet"]): (o.get("name"), o.get("map")) for o in json.loads(info.stdout)["outlets"]}
    for node, o in outlets.items():
        if names.get(str(o)) != (node, node):
            raise SystemExit("refused: strip outlet %d is %r, expected %r" % (o, names.get(str(o)), node))
    manifest = {"started": now_iso(), "args": vars(a), "outlets": outlets, "out": os.path.relpath(out, ROOT), "cells": []}
    mark("preflight ok")
    logger = current = None
    try:
        logger = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(out, "strip.csv"), "--hz", str(a.hz)],
                                  stdout=open(os.path.join(out, "strip_logger.stdout"), "w"),
                                  stderr=open(os.path.join(out, "strip_logger.stderr"), "w"))
        mark("logger start")
        time.sleep(a.settle)
        if logger.poll() is not None:
            raise SystemExit("the strip logger exited early (code %s)" % logger.returncode)
        mark("idle-before start"); time.sleep(a.idle_before); mark("idle-before end")
        for i, c in enumerate(cells, 1):
            label = "%02d_%s" % (i, c)
            mark("cell start %s" % label)
            env = dict(os.environ, CELL=c, PRED=os.path.relpath(os.path.abspath(a.pred), ROOT))
            current = subprocess.Popen(["bash", RUN_POOL], cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            with open(os.path.join(out, "cell_%s.log" % label), "a") as f:
                for line in current.stdout:
                    f.write("%.6f\t%s\n" % (time.time(), line.rstrip("\n"))); f.flush()
            rc = current.wait()
            current = None
            mark("cell end %s exit=%d" % (label, rc))
            manifest["cells"].append({"label": label, "cell": c, "exit": rc})
            time.sleep(a.gap)
        mark("idle-after start"); time.sleep(a.idle_after); mark("idle-after end")
        time.sleep(a.tail)
    finally:
        if current is not None and current.poll() is None:   # run_pool.sh's EXIT trap stops its own workers and probes
            current.send_signal(signal.SIGTERM)
            try:
                current.wait(timeout=60)
            except subprocess.TimeoutExpired:
                current.kill()
            mark("interrupted: run_pool.sh stopped")
        if logger is not None and logger.poll() is None:
            logger.send_signal(signal.SIGINT)
            try:
                logger.wait(timeout=15)
            except subprocess.TimeoutExpired:
                logger.kill()
        mark("logger stop")
        info = subprocess.run([sys.executable, STRIP, "info", "--json"], capture_output=True, text=True, timeout=30)
        open(os.path.join(out, "strip_info_after.json"), "w").write(info.stdout)
        manifest["finished"] = now_iso()
        json.dump(manifest, open(os.path.join(out, "run.json"), "w"), indent=1)
        mark("run end")


if __name__ == "__main__":
    main()
