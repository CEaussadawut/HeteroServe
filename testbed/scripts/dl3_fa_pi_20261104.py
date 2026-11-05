#!/usr/bin/env python3
"""The Pis' attention op per attention shape, timed with no model loaded (dl3fa, 2026-11-04; the user: "Time the Pis' CPUs per
attention shape, so the per-shape depth table (ATTENTION_V6) covers more than the laptop"). dl2fa's method on each Pi.

    python3 testbed/scripts/dl3_fa_pi_20261104.py run pi5            # measure one board (run the two boards at once)
    python3 testbed/scripts/dl3_fa_pi_20261104.py table <pi5 run dir> <pi4 run dir>

run: fa_bench_cpu (testbed/scripts/fa_bench_cpu_20261104.cpp, built on the board against its pinned tree into
~/heteroserve/fa_cpu/), dl2fa's shapes, cache lengths, repetitions and threads exactly (dl2_fa_cpu_20261104.SHAPES, NKV, REPS,
THREADS), two passes (the second in reverse order). Gates first: the board cooler than COLD_C (vcgencmd, up to 10 min). The
board's temperature and `vcgencmd get_throttled` are logged around every shape. Raw output:
testbed/results/dl3fa_<board>_<stamp>/run.log, with the board's binary sha256 and its source's.
table: per board and shape, the time per attention op (us) at each cache length, the mean of the two passes' medians, and their
spread; writes testbed/dataset/attention_cpu_pi_20261104.json. Nothing is fitted.
"""
import argparse, datetime, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import dl2_fa_cpu_20261104 as D2                           # noqa: E402  (the laptop's method: shapes, cache lengths, reps)

SRC = os.path.join(HERE, "fa_bench_cpu_20261104.cpp")
OUT = os.path.join(ROOT, "testbed", "dataset", "attention_cpu_pi_20261104.json")
BIN = "~/heteroserve/fa_cpu/fa_bench_cpu"
COLD_C = {"pi5": 50.0, "pi4": 55.0}                         # lc4's cold gate per board
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def on(node, cmd, timeout=600):
    r = subprocess.run(SSH + [node, cmd], capture_output=True, text=True, timeout=timeout)
    keep = [l for l in (r.stdout + r.stderr).splitlines() if not re.search(r"post-quantum|store now|pq\.html", l)]
    return r.returncode, "\n".join(keep)


def state(node):
    rc, o = on(node, "vcgencmd measure_temp; vcgencmd get_throttled; cut -d' ' -f1 /proc/loadavg", timeout=30)
    return " ".join(o.split())


def temp_c(node):
    rc, o = on(node, "vcgencmd measure_temp", timeout=30)
    m = re.search(r"temp=([\d.]+)", o)
    return float(m.group(1)) if m else None


def run(node):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "dl3fa_%s_%s" % (node, stamp))
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    rc, shas = on(node, "sha256sum %s ~/heteroserve/fa_cpu/fa_bench_cpu_20261104.cpp; cd ~/llama.cpp && git rev-parse --short=8 HEAD"
                  % BIN, timeout=60)
    say("########## dl3fa %s -> %s; on the board: %s; the source here %s" % (
        node, os.path.relpath(d, ROOT), " | ".join(shas.split("\n")), D2.sha(SRC)[:16]))
    t0 = time.time()
    while True:
        t = temp_c(node)
        if t is not None and t <= COLD_C[node]:
            break
        if time.time() - t0 > 600:
            say("gate: still %s C after 10 min (cold %.0f C); going on, recorded" % (t, COLD_C[node])); break
        time.sleep(10)
    say("gate: %s (waited %d s)" % (state(node), time.time() - t0))
    order = list(D2.SHAPES)
    for p, shapes in ((1, order), (2, order[::-1])):
        for key in shapes:
            hd, nh, nkv = (int(x) for x in key.split("x"))
            layers = D2.SHAPES[key]["layers"]
            say("pass %d shape %s layers %d before: %s" % (p, key, layers, state(node)))
            rc, o = on(node, "%s %d %d %d %d %d %d %s" % (BIN, hd, nh, nkv, layers, D2.REPS, D2.THREADS, ",".join(map(str, D2.NKV))),
                       timeout=7200)
            for ln in o.splitlines():
                say("pass %d shape %s | %s" % (p, key, ln))
            say("pass %d shape %s rc %d after: %s" % (p, key, rc, state(node)))
    say("RUN DONE")


def parse(rd):
    pts = {}
    for ln in open(os.path.join(rd, "run.log")):
        m = re.search(r"pass (\d) shape (\S+) \| n_kv (\d+) us_per_graph [\d.]+ us_per_op ([\d.]+)", ln)
        if m:
            pts.setdefault(m.group(2), {}).setdefault(int(m.group(3)), {})[int(m.group(1))] = float(m.group(4))
    shapes = {}
    for key, by in pts.items():
        curve, spread = {}, {}
        for n, v in sorted(by.items()):
            if len(v) == 2:
                curve[n] = round(statistics.fmean(v.values()), 2)
                spread[n] = round(100 * abs(v[1] - v[2]) / curve[n], 2)
        shapes[key] = {"models": D2.SHAPES.get(key, {}).get("models"), "layers_in_graph": D2.SHAPES.get(key, {}).get("layers"),
                       "us_per_op": {str(n): v for n, v in curve.items()}, "pass_spread_pct": {str(n): v for n, v in spread.items()}}
    return shapes


def table(rd5, rd4):
    out = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "_source": "dl3_fa_pi_20261104.py table %s %s: fa_bench_cpu (fa_bench_cpu_20261104.cpp) built on each Pi, -t %d, the "
                      "median of %d graphs a point, the mean of two passes" % (os.path.relpath(rd5, ROOT), os.path.relpath(rd4, ROOT),
                                                                            D2.THREADS, D2.REPS),
           "law": "fa_curve", "threads": D2.THREADS,
           "use": "per attention layer, the op's time at a cache of n_kv cells (us); a decoded token at depth d uses "
                  "n_kv = 256 x ceil((d + 1) / 256); the cost of depth is us(n_kv(d)) - us(256)",
           "boards": {"pi5": {"from": os.path.relpath(rd5, ROOT), "shapes": parse(rd5)},
                      "pi4": {"from": os.path.relpath(rd4, ROOT), "shapes": parse(rd4)}}}
    json.dump(out, open(OUT, "w"), indent=1)
    for b, e in out["boards"].items():
        for key, s in e["shapes"].items():
            c = {int(n): v for n, v in s["us_per_op"].items()}
            print("%-4s %-9s %s; spread max %.1f %%" % (b, key, " ".join("%d:%.0f" % (n, c[n]) for n in sorted(c)),
                                                       max(s["pass_spread_pct"].values())))
    print("wrote %s" % os.path.relpath(OUT, ROOT))


V6 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261104.json")
V7 = os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261105.json")
PI_TABLE = OUT                      # the published Pi table, which v7 reads (OUT may be pointed at a scratch copy: reproduce.py)


def v7(out=V7, pi=None):
    """The planner's table with the Pis priced by their own attention op per shape too (law "fa_curve", predict_pool.attn_ms):
    V6's boards (the laptop's fa_curve, the Jetson's fa_table), each Pi's entry replaced by its curves from
    attention_cpu_pi_20261104.json (its hybrid fraction kept). Registered as dl3 (4 Nov): within 2.4 % on the three dense
    benches (where V5 was 6-199 % off), 38 % high on the hybrid 9B on the Pi 5 (V5: 56 % low)."""
    base, pt = json.load(open(V6)), json.load(open(pi or PI_TABLE))
    t = json.loads(json.dumps(base))
    t["made"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t["_source"] = ("dl3_fa_pi_20261104.py v7: attention_cost_20261104.json's boards (V6), with each Pi's law replaced by its "
                    "attention op per shape (attention_cpu_pi_20261104.json, fa_bench_cpu on the board); registered as dl3")
    for b in ("pi5", "pi4"):
        t["boards"][b] = {"law": "fa_curve", "shapes": {k: s["us_per_op"] for k, s in pt["boards"][b]["shapes"].items()},
                          "hybrid_attention_fraction": base["boards"][b].get("hybrid_attention_fraction", {}),
                          "n_kv_rule": "256 x ceil((depth + 1) / 256) cells; the cost of depth is us(n_kv) - us(256) per attention layer",
                          "from": pt["_source"]}
    json.dump(t, open(out, "w"), indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "table", "v7"))
    ap.add_argument("args", nargs="*")
    a = ap.parse_args()
    if a.cmd == "run":
        if len(a.args) != 1 or a.args[0] not in COLD_C:
            raise SystemExit("run pi5|pi4")
        run(a.args[0])
    elif a.cmd == "table":
        if len(a.args) != 2:
            raise SystemExit("table <pi5 run dir> <pi4 run dir>")
        table(*a.args)
    else:
        v7()


if __name__ == "__main__":
    main()
