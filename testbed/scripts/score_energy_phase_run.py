#!/usr/bin/env python3
"""Score a run of energy_phase_run.py against its registration.

Per invocation: the energy window from its stamped --progress lines, and energy and J/token by both methods
(`shelly_strip.py summarise --json`, idle-subtracted with the run's idle-before window; raw output kept in the run
folder). Per phase: the mean and the spread across invocations (the house rule's across-run spread). Each is checked
against the registration's predictions. The board sampler's throttle flags, temperatures and fan states come too.

Window (testbed/energy.md, protocol 3): from the stamp of the first timed repetition's line ("prompt run 1/R" or
"generation run 1/R") to the stamp of the first result line after the last repetition's line (a JSONL line or a markdown
table row). If no result line arrived, the invocation's "bench end" marker is used, and that is flagged.

Nothing is filled in: a value the run does not give stays None, and None never passes a check.

Usage: score_energy_phase_run.py RUN_DIR --registration JSON [--out JSON]
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
RUN1 = re.compile(r"benchmark \d+/\d+: (prompt|generation) run 1/(\d+)\s*$")


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


def bench_window(run, label, ms):
    lines = []
    for ln in open(os.path.join(run, "bench_%s.log" % label), errors="replace"):
        t, _, text = ln.rstrip("\n").partition("\t")
        lines.append((float(t), text))
    out = {"flags": []}
    i1 = next((i for i, (_, x) in enumerate(lines) if RUN1.search(x)), None)
    if i1 is None:
        out["flags"].append("NO_RUN_1_LINE")
        return out
    reps = int(RUN1.search(lines[i1][1]).group(2))
    kind = RUN1.search(lines[i1][1]).group(1)
    last = re.compile(r"benchmark \d+/\d+: %s run %d/%d\s*$" % (kind, reps, reps))
    iN = next((i for i, (_, x) in enumerate(lines) if last.search(x)), None)
    out.update(start=lines[i1][0], reps_announced=reps, kind=kind)
    if iN is None:
        out["flags"].append("NO_LAST_RUN_LINE")
        return out
    res = [(t, x) for t, x in lines[iN + 1:] if x.startswith("{") or (x.startswith("|") and "---" not in x and "model" not in x)]
    jl = [x for _, x in lines[iN + 1:] if x.startswith("{")]
    if res:
        out["end"], out["end_source"] = res[0][0], "result line"
    else:
        out["end"], out["end_source"] = marker(ms, "bench end %s" % label), "bench end marker"
        out["flags"].append("NO_RESULT_LINE")
    if jl:
        j = json.loads(jl[0])
        out["jsonl"] = {k: j.get(k) for k in ("n_prompt", "n_gen", "n_threads", "avg_ts", "stddev_ts", "samples_ns",
                                              "samples_ts", "build_commit", "model_filename", "n_batch", "n_ubatch")}
        s = j.get("samples_ns") or []
        out["reps"] = len(s)
        out["timed_s"] = sum(s) / 1e9
        out["window_overhead_s"] = (out["end"] - out["start"]) - out["timed_s"]
        n = j.get("n_prompt") if kind == "prompt" else j.get("n_gen")
        out["tokens"] = (n * len(s)) if n else None
    else:
        out["flags"].append("NO_JSONL")
    return out


def summarise(run, window, idle, tokens):
    cmd = [sys.executable, STRIP, "summarise", os.path.join(run, "strip.csv"), "--window", "%.6f" % window[0],
           "%.6f" % window[1], "--json"]
    if idle:
        cmd += ["--idle", "%.6f" % idle[0], "%.6f" % idle[1]]
    if tokens:
        cmd += ["--tokens", str(tokens)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.returncode, (json.loads(p.stdout) if p.stdout.strip() else None), p.stderr


def outlet_view(summ, outlet):
    o = dict(summ["outlets"][str(outlet)])
    o["flags"] = [f for f in summ.get("flags", []) if f.get("outlet") in (outlet, "all", None)]
    if o.get("energy_counter_j") is not None and summ.get("window_s"):
        o["counter_mean_w"] = o["energy_counter_j"] / summ["window_s"]
    return o


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return {"n": 0, "mean": None, "min": None, "max": None, "spread_pct": None}
    m = statistics.fmean(v)
    return {"n": len(v), "mean": m, "min": min(v), "max": max(v), "spread_pct": 100.0 * (max(v) - min(v)) / m if m else None}


def within(x, band):
    return None if x is None else (band[0] <= x <= band[1])


def sampler(run):
    p = os.path.join(run, "node_sampler.log")
    rows, other = [], []
    if os.path.exists(p):
        for ln in open(p, errors="replace"):
            m = re.search(r"throttled=(0x[0-9a-fA-F]+) temp=([0-9.]+)'C frequency\(\d+\)=(\d+) fan=(\d*)", ln)
            if m:
                rows.append((m.group(1), float(m.group(2)), int(m.group(3)), m.group(4)))
                continue
            # a board without vcgencmd. The battery field runs to " loadavg=": its status can hold a space ("Not charging",
            # 2026-10-05 23:07 laptop shakedown: \S* read it as "Not" and failed the battery check); the Jetson's is empty
            m = re.search(r"thermal_mC=([0-9, ]*) freq_khz=(\d*) bat=(.*?)(?= loadavg=|$)", ln)
            if m:
                temps = [int(x) / 1000.0 for x in re.split(r"[ ,]+", m.group(1).strip()) if x]
                other.append((temps, int(m.group(2)) if m.group(2) else None,
                              m.group(3).split("/")[0] if m.group(3) else None))
    if other and not rows:
        # zones that never change are not temperatures of anything working (the Jetson's zone 4 reads a fixed 50 C)
        nz = min(len(x[0]) for x in other)
        live = [z for z in range(nz) if len(set(x[0][z] for x in other)) > 1] or list(range(nz))
        t = [max(x[0][z] for z in live) for x in other if x[0]]
        f = [x[1] for x in other if x[1] is not None]
        return {"n": len(other), "kind": "no-vcgencmd", "temp_max_c": max(t) if t else None, "temp_min_c": min(t) if t else None,
                "freq_khz_min": min(f) if f else None, "freq_khz_max": max(f) if f else None,
                "battery_states": sorted(set(x[2] for x in other if x[2])),
                "zones": [{"zone": z, "min_c": min(x[0][z] for x in other), "max_c": max(x[0][z] for x in other),
                           "live": z in live} for z in range(nz)]}
    if not rows:
        return {"n": 0}
    return {"n": len(rows), "throttled_values": sorted(set(r[0] for r in rows)), "temp_max_c": max(r[1] for r in rows),
            "temp_min_c": min(r[1] for r in rows), "arm_hz_min": min(r[2] for r in rows), "arm_hz_max": max(r[2] for r in rows),
            "fan_states": sorted(set(r[3] for r in rows))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--registration", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    man = json.load(open(os.path.join(a.run, "run.json")))
    outlet = man["args"]["outlet"]
    ms = read_markers(a.run)
    idle_b = (marker(ms, "idle-before start"), marker(ms, "idle-before end"))
    idle_a = (marker(ms, "idle-after start"), marker(ms, "idle-after end"))
    s = {"run": os.path.relpath(os.path.abspath(a.run), ROOT), "registration": os.path.relpath(os.path.abspath(a.registration), ROOT),
         "registered": reg.get("registered"), "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "outlet": outlet, "idle_before_window": list(idle_b), "idle_after_window": list(idle_a), "invocations": []}
    for nm, w in (("idle_before", idle_b), ("idle_after", idle_a)):
        rc, summ, err = summarise(a.run, w, None, None)
        s[nm] = outlet_view(summ, outlet) if summ else {"error": err.strip()[:300]}
    for inv in man["invocations"]:
        bw = bench_window(a.run, inv["label"], ms)
        rec = {"label": inv["label"], "phase": inv["phase"], "exit": inv["exit"], "bench": bw}
        if "start" in bw and "end" in bw and bw.get("tokens"):
            rc, summ, err = summarise(a.run, (bw["start"], bw["end"]), idle_b, bw["tokens"])
            with open(os.path.join(a.run, "summarise_%s.json" % inv["label"]), "w") as f:
                json.dump(summ, f, indent=1)
            rec["energy"] = outlet_view(summ, outlet) if summ else {"error": err.strip()[:300]}
            rec["window_s"] = bw["end"] - bw["start"]
        s["invocations"].append(rec)
    P = reg["predictions"]
    s["phases"], s["checks"] = {}, []
    for ph in ("pp", "tg"):
        inv = [r for r in s["invocations"] if r["phase"] == ph and "energy" in r]
        e = [r["energy"] for r in inv]
        ph_out = {
            "invocations": [r["label"] for r in inv],
            "tps": agg([r["bench"].get("jsonl", {}).get("avg_ts") for r in inv]),
            "mean_w": agg([x.get("mean_w") for x in e]),
            "counter_mean_w": agg([x.get("counter_mean_w") for x in e]),
            "j_per_token_net_counter": agg([x.get("j_per_token_net_counter") for x in e]),
            "j_per_token_net_trap": agg([x.get("j_per_token_net_trap") for x in e]),
            "j_per_token_gross_counter": agg([x.get("j_per_token_gross_counter") for x in e]),
            "j_per_token_gross_trap": agg([x.get("j_per_token_gross_trap") for x in e]),
            "wh_per_1000_tokens_net_counter": agg([x.get("wh_per_1000_tokens_net_counter") for x in e]),
            "wh_per_1000_tokens_net_trap": agg([x.get("wh_per_1000_tokens_net_trap") for x in e]),
            "disagreement_pct": agg([x.get("disagreement_pct") for x in e]),
            "window_s": agg([r.get("window_s") for r in inv]),
        }
        s["phases"][ph] = ph_out
        pp = P[ph]
        s["checks"] += [
            {"check": "%s tokens/s mean in %s" % (ph, pp["tps"]), "value": ph_out["tps"]["mean"], "pass": within(ph_out["tps"]["mean"], pp["tps"])},
            {"check": "%s mean apower W in %s" % (ph, pp["mean_w"]), "value": ph_out["mean_w"]["mean"], "pass": within(ph_out["mean_w"]["mean"], pp["mean_w"])},
        ]
        for m in ("counter", "trap"):
            v = ph_out["j_per_token_net_%s" % m]
            band = pp["j_per_token_net"][m] if isinstance(pp["j_per_token_net"], dict) else pp["j_per_token_net"]
            s["checks"] += [
                {"check": "%s net J/token (%s) mean in %s" % (ph, m, band), "value": v["mean"], "pass": within(v["mean"], band)},
                {"check": "%s net J/token (%s) spread across invocations <= %g %%" % (ph, m, P["spread_pct_max"]), "value": v["spread_pct"],
                 "pass": None if v["spread_pct"] is None else v["spread_pct"] <= P["spread_pct_max"]},
            ]
        for r in inv:
            g = r["energy"].get("disagreement_pct")
            s["checks"].append({"check": "%s trapezoid-vs-counter gap %% in %s" % (r["label"], P["gap_pct"]), "value": g, "pass": within(g, P["gap_pct"])})
    s["sampler"] = sampler(a.run)
    if s["sampler"].get("kind") == "no-vcgencmd":
        bs = s["sampler"].get("battery_states")
        if bs:      # the laptop: its outlet is its charger, which is the laptop's own draw only while the battery is full
            s["checks"].append({"check": "battery Full (or Not charging) throughout", "value": bs,
                                "pass": all(b in ("Full", "Not charging") for b in bs)})
    else:
        s["checks"].append({"check": "get_throttled 0x0 throughout", "value": s["sampler"].get("throttled_values"),
                            "pass": (s["sampler"].get("throttled_values") == ["0x0"]) if s["sampler"].get("n") else None})
    pre_t = (man.get("preflight") or {}).get("guard_trips")
    post = man.get("postflight_guard")
    if pre_t is not None and post:
        post_t = int(post[-1]) if post[-1].strip().isdigit() else None
        s["guard"] = {"trips_before": pre_t, "trips_after": post_t, "state_after": post[:-1]}
        s["checks"].append({"check": "no power-guard trip during the run (Jetson)", "value": [pre_t, post_t],
                            "pass": None if post_t is None else (post_t == pre_t and "guard=up" in post[0])})
    with open(a.out or os.path.join(a.run, "score.json"), "w") as f:
        json.dump(s, f, indent=1)
    report(s)


def _f(x, nd=3):
    return "?" if x is None else "%.*f" % (nd, x)


def report(s):
    print("idle before: %s W apower, %s W counter | idle after: %s W, %s W" % (
        _f(s["idle_before"].get("mean_w")), _f(s["idle_before"].get("counter_mean_w")),
        _f(s["idle_after"].get("mean_w")), _f(s["idle_after"].get("counter_mean_w"))))
    print("%-5s %7s %8s %8s %8s %9s %11s %11s %8s  flags" % ("inv", "t/s", "win s", "W trap", "W ctr", "gap %", "J/tok net c", "J/tok net t", "overhd s"))
    for r in s["invocations"]:
        b, e = r["bench"], r.get("energy", {})
        fl = b.get("flags", []) + [f["code"] for f in e.get("flags", []) if f["code"] != "FRESH_BELOW_1HZ"]
        print("%-5s %7s %8s %8s %8s %9s %11s %11s %8s  %s" % (
            r["label"], _f(b.get("jsonl", {}).get("avg_ts"), 2), _f(r.get("window_s"), 1), _f(e.get("mean_w")),
            _f(e.get("counter_mean_w")), _f(e.get("disagreement_pct"), 2), _f(e.get("j_per_token_net_counter"), 4),
            _f(e.get("j_per_token_net_trap"), 4), _f(b.get("window_overhead_s"), 2), ",".join(fl) or "-"))
    for ph, p in s["phases"].items():
        print("%s: t/s %s (spread %s %%), W %s, net J/token counter %s (spread %s %%) trap %s (spread %s %%), Wh/1000 tok net counter %s" % (
            ph, _f(p["tps"]["mean"], 2), _f(p["tps"]["spread_pct"], 2), _f(p["mean_w"]["mean"], 2),
            _f(p["j_per_token_net_counter"]["mean"], 4), _f(p["j_per_token_net_counter"]["spread_pct"], 2),
            _f(p["j_per_token_net_trap"]["mean"], 4), _f(p["j_per_token_net_trap"]["spread_pct"], 2),
            _f(p["wh_per_1000_tokens_net_counter"]["mean"], 4)))
    sm = s["sampler"]
    if sm.get("kind") == "no-vcgencmd":
        print("board: temp %s-%s C, mean CPU clock %s-%s MHz, battery %s (n %s)" % (sm.get("temp_min_c"), sm.get("temp_max_c"),
              (sm.get("freq_khz_min") or 0) // 1000, (sm.get("freq_khz_max") or 0) // 1000, sm.get("battery_states"), sm.get("n")))
    else:
      print("board: throttled %s, temp %s-%s C, ARM %s-%s MHz, fan states %s (n %s)" % (
        sm.get("throttled_values"), sm.get("temp_min_c"), sm.get("temp_max_c"),
        (sm.get("arm_hz_min") or 0) // 1000000, (sm.get("arm_hz_max") or 0) // 1000000, sm.get("fan_states"), sm.get("n")))
    for c in s["checks"]:
        v = c["value"]
        print("  %-5s %s  [%s]" % ({True: "PASS", False: "FAIL", None: "N/A"}[c["pass"]], c["check"],
                                  _f(v, 4) if isinstance(v, float) else v))


if __name__ == "__main__":
    main()
