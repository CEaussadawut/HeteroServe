#!/usr/bin/env python3
"""The laptop's attention op per attention shape, timed with no model loaded (dl2, 2026-11-04; the user: "do all four
follow-up studies": a depth law per model architecture). The CPU counterpart of lc4n's fa_bench on the Jetson.

    QUIET_ALLOW_APPS=zen python3 testbed/scripts/dl2_fa_cpu_20261104.py run          # measure (the laptop quiet)
    python3 testbed/scripts/dl2_fa_cpu_20261104.py table <run dir>                    # write the CPU attention table

Why: the planner prices a CPU board's depth cost per query-head dimension (ATTENTION_V5: a (1 + d / d1) ns per query-head dim
per block per token of depth), fitted on TinyLlama alone (head size 64). dl1 found Qwen3-30B-A3B's slope 20-28 % under it
(head size 128): per query-head dimension, the cost depends on the head's layout, which one law per dimension cannot carry.
run: fa_bench_cpu (testbed/scripts/fa_bench_cpu_20261104.cpp, built against the laptop's pinned tree into
~/.cache/heteroserve-wf/fa_cpu/) for every attention shape the cluster serves, -t 4 as the laptop decodes, 15 timed graphs a
point (median), at the cache lengths NKV, in two passes over the shapes (the second in reverse order) for repeatability.
Gates first: the laptop quiet on its charger (phase_calib_20261028.laptop_quiet; QUIET_ALLOW_APPS as there). The CPU's
package temperature and the load are logged around each shape. Raw output: testbed/results/dl2fa_<stamp>/ (run.log, the
binary's sha256 and its source's).
table: per shape, the time per attention op (us) at each cache length, the mean of the two passes' medians, and their spread;
writes testbed/dataset/attention_cpu_20261104.json (the laptop's "fa_curve" law). Nothing is fitted: the planner reads the
curve.
"""
import argparse, datetime, hashlib, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
BIN = os.path.expanduser("~/.cache/heteroserve-wf/fa_cpu/fa_bench_cpu")
SRC = os.path.join(HERE, "fa_bench_cpu_20261104.cpp")
OUT = os.path.join(ROOT, "testbed", "dataset", "attention_cpu_20261104.json")
NKV = [256, 512, 768, 1024, 1280, 1536, 2048, 2560, 3072, 4096, 6144, 8192]
REPS, THREADS = 15, 4
# (head_dim, n_head, n_head_kv, attention layers in a token's graph): every model the cluster serves or benches, as
# model_shapes.json and hybrid_shapes_20261101.json give them (a hybrid's attention layers: every 4th block)
SHAPES = {
    "64x32x4":   {"models": ["tinyllama"], "layers": 22},
    "128x16x8":  {"models": ["qwen3-0.6b"], "layers": 28},
    "128x32x8":  {"models": ["qwen3-4b"], "layers": 36},
    "128x40x8":  {"models": ["qwen3-14b"], "layers": 40},
    "128x32x4":  {"models": ["qwen3-30b-a3b"], "layers": 48},
    "128x64x8":  {"models": ["qwen3-32b"], "layers": 64},
    "256x16x4":  {"models": ["qwen3.5-9b"], "layers": 8},
    "256x24x4":  {"models": ["qwen3.8-27b"], "layers": 16},
    "256x16x2":  {"models": ["qwen3.6-35b-a3b"], "layers": 10},
}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cpu_state():
    load = open("/proc/loadavg").read().split()[0]
    temps = []
    for z in sorted(os.listdir("/sys/class/thermal")):
        try:
            if open("/sys/class/thermal/%s/type" % z).read().strip() == "x86_pkg_temp":
                temps.append(int(open("/sys/class/thermal/%s/temp" % z).read()) / 1000)
        except OSError:
            pass
    return "load %s pkg %s C" % (load, temps[0] if temps else "?")


def run():
    import phase_calib_20261028 as C
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "dl2fa_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## dl2fa -> %s; binary sha256 %s, source sha256 %s" % (os.path.relpath(d, ROOT), sha(BIN)[:16], sha(SRC)[:16]))
    C.laptop_quiet(lambda s: say("gate " + s))
    order = list(SHAPES)
    for p, shapes in ((1, order), (2, order[::-1])):
        for key in shapes:
            hd, nh, nkv = (int(x) for x in key.split("x"))
            layers = SHAPES[key]["layers"]
            say("pass %d shape %s layers %d before: %s" % (p, key, layers, cpu_state()))
            r = subprocess.run([BIN, str(hd), str(nh), str(nkv), str(layers), str(REPS), str(THREADS), ",".join(map(str, NKV))],
                               capture_output=True, text=True, timeout=3600)
            for ln in (r.stdout + r.stderr).splitlines():
                say("pass %d shape %s | %s" % (p, key, ln))
            say("pass %d shape %s rc %d after: %s" % (p, key, r.returncode, cpu_state()))
    say("RUN DONE")


def table(rd):
    pts = {}
    for ln in open(os.path.join(rd, "run.log")):
        m = re.search(r"pass (\d) shape (\S+) \| n_kv (\d+) us_per_graph [\d.]+ us_per_op ([\d.]+)", ln)
        if m:
            pts.setdefault(m.group(2), {}).setdefault(int(m.group(3)), {})[int(m.group(1))] = float(m.group(4))
    out = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "_source": "dl2_fa_cpu_20261104.py table %s: fa_bench_cpu (fa_bench_cpu_20261104.cpp) on the laptop, -t %d, the "
                      "median of %d graphs a point, the mean of two passes" % (os.path.relpath(rd, ROOT), THREADS, REPS),
           "board": "laptop", "law": "fa_curve", "threads": THREADS,
           "use": "per attention layer, the op's time at a cache of n_kv cells (us); a decoded token at depth d uses "
                  "n_kv = 256 x ceil((d + 1) / 256); the cost of depth is us(n_kv(d)) - us(256)",
           "shapes": {}}
    for key, by in pts.items():
        curve, spread = {}, {}
        for n, v in sorted(by.items()):
            if len(v) == 2:
                curve[n] = round(statistics.fmean(v.values()), 2)
                spread[n] = round(100 * abs(v[1] - v[2]) / curve[n], 2)
        out["shapes"][key] = {"models": SHAPES.get(key, {}).get("models"), "layers_in_graph": SHAPES.get(key, {}).get("layers"),
                              "us_per_op": {str(n): v for n, v in curve.items()},
                              "pass_spread_pct": {str(n): v for n, v in spread.items()}}
    json.dump(out, open(OUT, "w"), indent=1)
    for key, s in out["shapes"].items():
        c = {int(n): v for n, v in s["us_per_op"].items()}
        print("%-9s %s; spread max %.1f %%" % (key, " ".join("%d:%.1f" % (n, c[n]) for n in sorted(c)),
                                                 max(s["pass_spread_pct"].values())))
    print("wrote %s" % os.path.relpath(OUT, ROOT))


V5 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261103.json")
V6 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261104.json")
CPU_TABLE = OUT                     # the published CPU table, which v6 reads (OUT may be pointed at a scratch copy: reproduce.py)


def v6(out=V6, cpu=None):
    """The planner's table with the laptop priced by its attention op per shape (law "fa_curve", predict_pool.attn_ms): V5's
    boards, the laptop's entry replaced by the curves of attention_cpu_20261104.json (its hybrid fraction kept). out and cpu:
    where to write, and which CPU table to read (by default the published one)."""
    base, cpu = json.load(open(V5)), json.load(open(cpu or CPU_TABLE))
    t = json.loads(json.dumps(base))
    t["made"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t["_source"] = ("dl2_fa_cpu_20261104.py v6: attention_cost_20261103.json's boards, with the laptop's law replaced by its "
                    "attention op per shape (attention_cpu_20261104.json, fa_bench_cpu); registered as dl2 (FA held on all three "
                    "models, 0.6-4.9 %)")
    lap = base["boards"]["laptop"]
    t["boards"]["laptop"] = {"law": "fa_curve", "shapes": {k: s["us_per_op"] for k, s in cpu["shapes"].items()},
                             "hybrid_attention_fraction": lap.get("hybrid_attention_fraction", {}),
                             "n_kv_rule": "256 x ceil((depth + 1) / 256) cells; the cost of depth is us(n_kv) - us(256) per attention layer",
                             "from": cpu["_source"]}
    json.dump(t, open(out, "w"), indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "table", "v6"))
    ap.add_argument("dir", nargs="?")
    a = ap.parse_args()
    if a.cmd == "run":
        run()
    elif a.cmd == "table":
        table(a.dir)
    else:
        v6()


if __name__ == "__main__":
    main()
