#!/usr/bin/env python3
"""Score an energy_pool_run.py session against its registration (register_pool_energy.py).

Per cell run: the host's llama-bench prints, through run_pool.sh, "benchmark 1/2: prompt run i/R" before each timed prompt
repetition, then "benchmark 2/2: generation run i/R", and a jsonl result line after each test. A phase's window runs from the
stamp of its "run 1/R" line to the stamp of the first jsonl line after its "run R/R" line (testbed/energy.md, protocol 3).
Energy per window comes from `shelly_strip.py summarise --json` with the session's idle-before window, per outlet. The
cell's energy is the sum over its boards' outlets, gross and net, by each method. The router is reported apart and not
charged to the cell. Tokens: n_prompt x reps and n_gen x reps from the jsonl.

Checks: each run exited 0 and llama-bench reported EXIT_BENCH=0; throughput within 10 % of the engine's prediction;
the registered hypotheses (the rule's net J/token at least 25 % below the even split's, per phase, per method). The
across-run spread is reported, not scored. Nothing is filled in: a missing value stays None and never passes.

Usage: score_energy_pool_run.py RUN_DIR [--out JSON]   (the registration is the one the run used, from run.json)
"""
import argparse
import datetime
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STRIP = os.path.join(ROOT, "testbed", "scripts", "shelly_strip.py")
RUNL = re.compile(r"benchmark (\d+)/(\d+): (prompt|generation) run (\d+)/(\d+)\s*$")
TPS_TOL = 0.10


def read_markers(run):
    out = []
    for line in open(os.path.join(run, "markers.tsv")):
        t, label = line.rstrip("\n").split("\t", 1)
        out.append((float(t), label))
    return out


def marker(ms, prefix):
    hits = [t for t, label in ms if label.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("marker %r: %d matches" % (prefix, len(hits)))
    return hits[0]


def windows(path):
    lines = []
    for ln in open(path, errors="replace"):
        t, _, x = ln.rstrip("\n").partition("\t")
        lines.append((float(t), x))
    out, exit_bench = {}, None
    for t, x in lines:
        m = re.search(r"EXIT_BENCH=(\d+)", x)
        if m:
            exit_bench = int(m.group(1))
    for kind, ph in (("prompt", "pp"), ("generation", "tg")):
        runs = [(i, t, RUNL.search(x)) for i, (t, x) in enumerate(lines) if RUNL.search(x) and RUNL.search(x).group(3) == kind]
        if not runs:
            out[ph] = {"flags": ["NO_RUN_LINES"]}
            continue
        reps = int(runs[0][2].group(5))
        first = next((r for r in runs if int(r[2].group(4)) == 1), None)
        last = next((r for r in runs if int(r[2].group(4)) == reps), None)
        w = {"reps_announced": reps, "flags": []}
        if first is None or last is None:
            w["flags"].append("MISSING_RUN_1_OR_R"); out[ph] = w; continue
        w["start"] = first[1]
        jl = next(((t, x) for t, x in lines[last[0] + 1:] if x.startswith("{")), None)
        if jl is None:
            w["flags"].append("NO_JSONL_AFTER_LAST_RUN"); out[ph] = w; continue
        w["end"] = jl[0]
        j = json.loads(jl[1])
        s = j.get("samples_ns") or []
        n = j.get("n_prompt") if ph == "pp" else j.get("n_gen")
        w.update(avg_ts=j.get("avg_ts"), reps=len(s), timed_s=sum(s) / 1e9, tokens=(n * len(s)) if n else None,
                 window_overhead_s=(w["end"] - w["start"]) - sum(s) / 1e9, n=n)
        out[ph] = w
    return out, exit_bench


def summarise(run, window, idle, tokens):
    cmd = [sys.executable, STRIP, "summarise", os.path.join(run, "strip.csv"), "--window", "%.6f" % window[0], "%.6f" % window[1],
           "--idle", "%.6f" % idle[0], "%.6f" % idle[1], "--tokens", str(tokens), "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(p.stdout) if p.stdout.strip() else {"error": p.stderr.strip()[:300]}


def agg(v):
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0, "mean": None, "min": None, "max": None, "spread_pct": None}
    m = statistics.fmean(v)
    return {"n": len(v), "mean": m, "min": min(v), "max": max(v), "spread_pct": 100.0 * (max(v) - min(v)) / m if m else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--out")
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.run, "run.json")))
    reg = json.load(open(os.path.join(ROOT, man["args"]["pred"]) if not os.path.isabs(man["args"]["pred"]) else man["args"]["pred"]))
    outlets = {k: int(v) for k, v in man["outlets"].items()}
    router = 3
    ms = read_markers(a.run)
    idle = (marker(ms, "idle-before start"), marker(ms, "idle-before end"))
    s = {"run": os.path.relpath(os.path.abspath(a.run), ROOT), "registration": man["args"]["pred"], "registered": reg.get("registered"),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "idle_before": list(idle), "runs": [], "checks": []}
    for c in man["cells"]:
        cfg = reg["configs"][c["cell"]]
        nodes = [p["node"] for p in cfg["pipeline"]]
        w, exit_bench = windows(os.path.join(a.run, "cell_%s.log" % c["label"]))
        rec = {"label": c["label"], "cell": c["cell"], "exit": c["exit"], "exit_bench": exit_bench, "nodes": nodes, "phases": {}}
        for ph in ("pp", "tg"):
            x = w.get(ph, {})
            if "start" not in x or "end" not in x or not x.get("tokens"):
                rec["phases"][ph] = {"bench": x}; continue
            summ = summarise(a.run, (x["start"], x["end"]), idle, x["tokens"])
            json.dump(summ, open(os.path.join(a.run, "summarise_%s_%s.json" % (c["label"], ph)), "w"), indent=1)
            if "outlets" not in summ:
                rec["phases"][ph] = {"bench": x, "error": summ.get("error")}; continue
            per = {n: summ["outlets"][str(outlets[n])] for n in nodes}
            tot = {}
            for m in ("trap", "counter"):
                g = [per[n].get("energy_%s_j" % m) for n in nodes]
                net = [per[n].get("net_energy_%s_j" % m) for n in nodes]
                tot["gross_%s_j" % m] = sum(g) if None not in g else None
                tot["net_%s_j" % m] = sum(net) if None not in net else None
                tot["j_per_token_gross_%s" % m] = tot["gross_%s_j" % m] / x["tokens"] if tot["gross_%s_j" % m] is not None else None
                tot["j_per_token_net_%s" % m] = tot["net_%s_j" % m] / x["tokens"] if tot["net_%s_j" % m] is not None else None
            rt = summ["outlets"][str(router)]
            rec["phases"][ph] = {"bench": x, "window_s": x["end"] - x["start"], "pool": tot,
                                 "per_board": {n: {k: per[n].get(k) for k in ("label", "mean_w", "energy_trap_j", "energy_counter_j",
                                                                            "disagreement_pct", "idle_mean_w_trap", "idle_mean_w_counter")}
                                               for n in nodes},
                                 "router": {k: rt.get(k) for k in ("mean_w", "energy_trap_j", "energy_counter_j")},
                                 "flags": [f for f in summ.get("flags", []) if f.get("code") != "FRESH_BELOW_1HZ"]}
        s["runs"].append(rec)
        s["checks"].append({"check": "%s exit 0 and EXIT_BENCH=0" % c["label"], "value": [c["exit"], exit_bench],
                            "pass": c["exit"] == 0 and exit_bench == 0})
    s["cells"] = {}
    for cell in sorted(set(r["cell"] for r in s["runs"])):
        runs = [r for r in s["runs"] if r["cell"] == cell]
        pred = reg["configs"][cell]["predicted"]
        cs = {}
        for ph, pk in (("pp", "prefill_tok_s"), ("tg", "decode_tok_s")):
            ok = [r["phases"][ph] for r in runs if "pool" in r["phases"].get(ph, {})]
            cs[ph] = {"tps": agg([p["bench"].get("avg_ts") for p in ok]), "predicted_tps": pred[pk],
                      **{k: agg([p["pool"].get(k) for p in ok]) for k in ("j_per_token_net_counter", "j_per_token_net_trap",
                                                                         "j_per_token_gross_counter", "j_per_token_gross_trap")},
                      "board_mean_w": {n: agg([p["per_board"][n].get("mean_w") for p in ok]) for n in runs[0]["nodes"]},
                      "router_mean_w": agg([p["router"].get("mean_w") for p in ok])}
            t = cs[ph]["tps"]["mean"]
            s["checks"].append({"check": "%s %s t/s within %d %% of the engine's %.3f" % (cell, ph, TPS_TOL * 100, pred[pk]),
                                "value": t, "pass": None if t is None else abs(t / pred[pk] - 1) <= TPS_TOL})
        s["cells"][cell] = cs
    for h in reg.get("energy_hypotheses", []):
        for ph in ("pp", "tg"):
            for m in ("counter", "trap"):
                r = s["cells"].get(h["rule"], {}).get(ph, {}).get("j_per_token_net_%s" % m, {}).get("mean")
                e = s["cells"].get(h["even"], {}).get(ph, {}).get("j_per_token_net_%s" % m, {}).get("mean")
                # a saving is only meaningful when both cells cost energy above idle (a net <= 0 is flagged, not scored)
                sav = 100.0 * (1 - r / e) if (r is not None and e is not None and r > 0 and e > 0) else None
                s["checks"].append({"check": "%s %s: the rule's net J/token (%s) >= %g %% below the even split's" % (h["model"], ph, m, h["min_saving_pct"]),
                                    "value": sav, "pass": None if sav is None else sav >= h["min_saving_pct"]})
    json.dump(s, open(a.out or os.path.join(a.run, "score.json"), "w"), indent=1)
    report(s)


def _f(x, nd=3):
    return "?" if x is None else "%.*f" % (nd, x)


def report(s):
    for r in s["runs"]:
        for ph in ("pp", "tg"):
            p = r["phases"].get(ph, {})
            b = p.get("bench", {})
            pool = p.get("pool", {})
            print("%-22s %s  t/s %8s  win %6s s  pool net J/tok ctr %8s trap %8s  gross ctr %8s  W %s  router %s W  %s" % (
                r["label"], ph, _f(b.get("avg_ts"), 3), _f(p.get("window_s"), 1), _f(pool.get("j_per_token_net_counter"), 4),
                _f(pool.get("j_per_token_net_trap"), 4), _f(pool.get("j_per_token_gross_counter"), 4),
                " ".join("%s %s" % (n, _f(v.get("mean_w"), 2)) for n, v in p.get("per_board", {}).items()),
                _f(p.get("router", {}).get("mean_w"), 2), ",".join(b.get("flags", []) + [f["code"] for f in p.get("flags", [])]) or "-"))
    for cell, cs in s["cells"].items():
        for ph in ("pp", "tg"):
            c = cs[ph]
            print("%-16s %s: t/s %s (pred %s, spread %s %%)  net J/token ctr %s (spread %s %%) trap %s  gross ctr %s" % (
                cell, ph, _f(c["tps"]["mean"], 3), c["predicted_tps"], _f(c["tps"]["spread_pct"], 2),
                _f(c["j_per_token_net_counter"]["mean"], 4), _f(c["j_per_token_net_counter"]["spread_pct"], 2),
                _f(c["j_per_token_net_trap"]["mean"], 4), _f(c["j_per_token_gross_counter"]["mean"], 4)))
    for c in s["checks"]:
        v = c["value"]
        print("  %-5s %s  [%s]" % ({True: "PASS", False: "FAIL", None: "N/A"}[c["pass"]], c["check"], _f(v, 2) if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
