#!/usr/bin/env python3
"""Score a run of testbed/scripts/energy_mapcheck_idle.sh against its registration
(testbed/dataset/energy_predictions_20261005.json, registered 2026-10-05 14:26:38, read-only).

A, the outlet-map check: each outlet's mean apower over each load window minus its baseline mean, and the registered
   rule (largest rise among outlets 0-2 on the mapped outlet, >= 0.5 W; the other two board outlets move < 0.3 W).
B, the idle floors: `shelly_strip.py summarise --json` over the registered 300 s window (both energies, their gap,
   freshness, flags; raw output kept beside the run), by_minute against the counter over complete minutes, request
   round trips, and outlet 2 (AC) against the Jetson guard's 1 Hz INA3221 input-rail power (DC, module input only).

Nothing is filled in: a missing reading stays missing, and a window with no samples scores None, never 0.

Usage: score_energy_mapcheck_idle.py RUN_DIR [--registration JSON] [--out JSON]
"""
import argparse
import collections
import csv
import datetime
import json
import os
import re
import statistics
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
STRIP = os.path.join(ROOT, "testbed", "scripts", "shelly_strip.py")
REG = os.path.join(ROOT, "testbed", "dataset", "energy_predictions_20261005.json")
N = 4
BOARD_OUTLETS = (0, 1, 2)
RISE_MIN_W, OTHERS_MAX_W = 0.5, 0.3      # the registered rule


def fnum(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_markers(run):
    out = []
    with open(os.path.join(run, "markers.tsv")) as f:
        for line in f:
            t, label = line.rstrip("\n").split("\t", 1)
            out.append((float(t), label))
    return out


def marker(ms, prefix):
    hits = [t for t, label in ms if label.startswith(prefix)]
    if len(hits) != 1:
        raise SystemExit("marker %r: %d matches in markers.tsv" % (prefix, len(hits)))
    return hits[0]


def read_rows(path):
    out = []
    with open(path, newline="") as f:
        for d in csv.DictReader(f):
            out.append({"t": float(d["wall_time"]) + float(d["rtt_s"]) / 2.0, "rtt": float(d["rtt_s"]),
                        "ok": d["ok"] == "1", "error": d["error"], "d": d})
    return out


def mean_w(rows, i, a, b):
    v = [fnum(r["d"]["o%d_apower" % i]) for r in rows if r["ok"] and a <= r["t"] <= b]
    v = [x for x in v if x is not None]
    return {"mean_w": statistics.fmean(v) if v else None, "n": len(v),
            "min_w": min(v) if v else None, "max_w": max(v) if v else None}


def requests(rows, a=None, b=None):
    sel = [r for r in rows if (a is None or a <= r["t"] <= b)]
    rtt = sorted(r["rtt"] for r in sel if r["ok"])
    p95 = rtt[min(len(rtt) - 1, int(round(0.95 * (len(rtt) - 1))))] if rtt else None
    return {"requests": len(sel), "failed": sum(1 for r in sel if not r["ok"]),
            "errors": sorted(set(r["error"].split(":")[0] for r in sel if not r["ok"])),
            "rtt_median_s": statistics.median(rtt) if rtt else None, "rtt_p95_s": p95, "rtt_max_s": rtt[-1] if rtt else None,
            "rtt_over_0.5s": sum(1 for x in rtt if x > 0.5)}


def board_load_out(run, board):
    p = os.path.join(run, "load_%s.out" % board)
    if not os.path.exists(p):
        return {"raw": None}
    raw = open(p, errors="replace").read()
    m = re.search(r"ran ([0-9.]+) s", raw)
    left = re.search(r"busy loops left: (\d+)", raw)
    return {"raw": raw.strip(), "board_clock_ran_s": float(m.group(1)) if m else None,
            "busy_loops_left": int(left.group(1)) if left else None}


def score_map(run, reg):
    ms, rows = read_markers(run), read_rows(os.path.join(run, "mapcheck.csv"))
    rule = reg["A_map_check"]
    mp = rule["map_under_test"]
    a0 = marker(ms, "A: logger start")
    loads = [(L["board"], marker(ms, "A: load start %s " % L["board"]), marker(ms, "A: load end %s" % L["board"]))
             for L in rule["loads"]]
    base = (a0 + 5.0, min(s for _, s, _ in loads) - 2.0)
    base_m = {i: mean_w(rows, i, *base) for i in range(N)}
    out = {"baseline_window": list(base), "baseline": {str(i): base_m[i] for i in range(N)}, "boards": [],
           "requests": requests(rows)}
    confirmed = True
    for board, s, e in loads:
        w = (s + 10.0, e - 2.0)
        m = {i: mean_w(rows, i, *w) for i in range(N)}
        delta = {i: (m[i]["mean_w"] - base_m[i]["mean_w"])
                 if (m[i]["mean_w"] is not None and base_m[i]["mean_w"] is not None) else None for i in range(N)}
        expected = int(next(k for k, v in mp.items() if v == board))
        scored = [i for i in BOARD_OUTLETS if delta[i] is not None]
        top = max(scored, key=lambda i: delta[i]) if scored else None
        others = [i for i in BOARD_OUTLETS if i != expected]
        ok = (top == expected and delta[expected] >= RISE_MIN_W
              and all(delta[i] is not None and abs(delta[i]) < OTHERS_MAX_W for i in others))
        confirmed = confirmed and ok
        out["boards"].append({"board": board, "mapped_outlet": expected, "load_marks": [s, e], "window": list(w),
                              "load": {str(i): m[i] for i in range(N)},
                              "delta_w": {str(i): delta[i] for i in range(N)}, "largest_rise_outlet": top,
                              "passes": ok, "board_side": board_load_out(run, board)})
    out["map_confirmed"] = confirmed
    return out


def by_minute_check(rows, i):
    """Per complete minute [M, M+60): the strip's by_minute against its own counter.

    boundary_total[M] is aenergy.total at the first ok row whose minute_ts is M: the same aenergy object, refreshed at
    most ~1 s after the boundary. by_minute_0 read while minute_ts is M+60 is the strip's energy for [M, M+60)."""
    first_total, bm_seen = {}, {}
    for r in rows:
        if not r["ok"]:
            continue
        d = r["d"]
        mts, bm0, tot = (fnum(d["o%d_%s" % (i, k)]) for k in ("aenergy_minute_ts", "by_minute_0", "aenergy_total"))
        if mts is None:
            continue
        if tot is not None and mts not in first_total:
            first_total[mts] = tot
        if bm0 is not None:
            bm_seen.setdefault(mts, collections.Counter())[bm0] += 1
    mins = sorted(first_total)
    out = []
    for M in mins:
        if M + 60 in first_total and (M + 60) in bm_seen and mins[0] < M:    # M itself must be a seen transition
            seen = bm_seen[M + 60]
            bm = seen.most_common(1)[0][0]                 # the value the strip held for that minute
            dc = (first_total[M + 60] - first_total[M]) * 1000.0
            out.append({"minute_ts": M, "by_minute_mwh": bm, "by_minute_values_seen": dict((str(k), v) for k, v in seen.items()),
                        "counter_delta_mwh": round(dc, 3),
                        "diff_pct": 100.0 * (bm - dc) / dc if dc else None})
    s_bm, s_dc = sum(x["by_minute_mwh"] for x in out), sum(x["counter_delta_mwh"] for x in out)
    return {"minutes": out, "n": len(out), "sum_by_minute_mwh": s_bm, "sum_counter_mwh": s_dc,
            "sum_diff_pct": 100.0 * (s_bm - s_dc) / s_dc if s_dc else None}


def guard_dc(run, a, b):
    p = os.path.join(run, "jetson_power_guard.log")
    if not os.path.exists(p):
        return {"n": 0, "note": "no guard log"}
    day = datetime.datetime.fromtimestamp(a).date()          # the guard writes the Jetson's local time, synced
    pts, trips = [], []
    for ln in open(p, errors="replace"):
        if "TRIP" in ln:
            trips.append(ln.strip())
        m = re.match(r"^(\d\d):(\d\d):(\d\d) V=(\d+) I=(\d+) P=(\d+) alarms=(\d+)", ln)
        if not m:
            continue
        t = datetime.datetime.combine(day, datetime.time(*map(int, m.group(1, 2, 3)))).timestamp()
        if a <= t <= b:
            pts.append((int(m.group(4)), int(m.group(5)), int(m.group(6)), int(m.group(7))))
    if not pts:
        return {"n": 0, "trips": trips}
    return {"n": len(pts), "mean_p_w": statistics.fmean(p[2] for p in pts) / 1000.0,
            "min_p_w": min(p[2] for p in pts) / 1000.0, "max_p_w": max(p[2] for p in pts) / 1000.0,
            "mean_v": statistics.fmean(p[0] for p in pts) / 1000.0, "min_v": min(p[0] for p in pts) / 1000.0,
            "alarms_max": max(p[3] for p in pts), "trips": trips}


def score_idle(run, reg):
    ms, rows = read_markers(run), read_rows(os.path.join(run, "idle.csv"))
    b0 = marker(ms, "B: logger start")
    a, b = b0 + 30.0, b0 + 330.0
    cmd = [sys.executable, STRIP, "summarise", os.path.join(run, "idle.csv"), "--window", "%.3f" % a, "%.3f" % b, "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True)
    with open(os.path.join(run, "idle_summarise.json"), "w") as f:
        f.write(p.stdout)
    with open(os.path.join(run, "idle_summarise.stderr"), "w") as f:
        f.write(p.stderr)
    summ = json.loads(p.stdout) if p.stdout.strip() else None
    out = {"window": [a, b], "summarise_exit": p.returncode, "summarise": summ, "requests": requests(rows, a, b),
           "apower": {str(i): mean_w(rows, i, a, b) for i in range(N)},
           "by_minute": {str(i): by_minute_check(rows, i) for i in range(N)}}
    dc = guard_dc(run, a, b)
    ac = (summ or {}).get("outlets", {}).get("2", {}).get("mean_w")
    out["jetson_ac_vs_dc"] = {"outlet2_ac_mean_w": ac, "guard_dc": dc,
                              "ac_minus_dc_w": (ac - dc["mean_p_w"]) if (ac is not None and dc.get("n")) else None,
                              "dc_over_ac": (dc["mean_p_w"] / ac) if (ac and dc.get("n")) else None}
    return out


def states(run):
    return {n: open(os.path.join(run, n), errors="replace").read().strip().splitlines()
            for n in sorted(os.listdir(run)) if n.startswith("state_")}


def report(s):
    A, B = s["A"], s["B"]
    print("A  outlet-map check (registered rule: largest rise on the mapped outlet, >= %.1f W; others < %.1f W)" % (RISE_MIN_W, OTHERS_MAX_W))
    print("   baseline W: " + "  ".join("o%s %s" % (i, _f(A["baseline"][i]["mean_w"])) for i in sorted(A["baseline"])))
    for x in A["boards"]:
        print("   %-6s -> outlet %d   rise W: %s   largest on %s   %s   (board clock ran %s s, loops left %s)" % (
            x["board"], x["mapped_outlet"], "  ".join("o%s %+.2f" % (i, v) if v is not None else "o%s ?" % i
                                                      for i, v in sorted(x["delta_w"].items())),
            x["largest_rise_outlet"], "PASS" if x["passes"] else "FAIL",
            _f(x["board_side"].get("board_clock_ran_s")), x["board_side"].get("busy_loops_left")))
    print("   MAP %s   requests %d, failed %d" % ("CONFIRMED" if A["map_confirmed"] else "NOT CONFIRMED",
                                                  A["requests"]["requests"], A["requests"]["failed"]))
    su = B["summarise"] or {}
    print("B  idle floors over %.0f s" % (B["window"][1] - B["window"][0]))
    for i in range(N):
        o = su.get("outlets", {}).get(str(i), {})
        w = B["apower"][str(i)]
        bm = B["by_minute"][str(i)]
        print("   o%d %-7s mean %s W (min %s max %s)  E trap %s J  counter %s J  gap %s %%  fresh %s/s  by_minute vs counter %s %% over %d min" % (
            i, o.get("label"), _f(o.get("mean_w")), _f(w["min_w"], 1), _f(w["max_w"], 1), _f(o.get("energy_trap_j"), 1),
            _f(o.get("energy_counter_j"), 1), _f(o.get("disagreement_pct"), 2), _f(o.get("fresh_any_hz"), 2),
            _f(bm["sum_diff_pct"], 2), bm["n"]))
    r = B["requests"]
    print("   requests %d, failed %d; rtt median %s s, p95 %s s, max %s s, %d over 0.5 s" % (
        r["requests"], r["failed"], _f(r["rtt_median_s"], 3), _f(r["rtt_p95_s"], 3), _f(r["rtt_max_s"], 3), r["rtt_over_0.5s"]))
    j = B["jetson_ac_vs_dc"]
    print("   Jetson: outlet 2 AC %s W vs guard DC %s W (n %s, min V %s)  AC-DC %s W" % (
        _f(j["outlet2_ac_mean_w"]), _f(j["guard_dc"].get("mean_p_w")), j["guard_dc"].get("n"),
        _f(j["guard_dc"].get("min_v")), _f(j["ac_minus_dc_w"])))
    for f in su.get("flags", []):
        print("   FLAG %s %s: %s" % (f.get("outlet"), f.get("code"), f.get("message")))


def _f(x, nd=3):
    return "?" if x is None else ("%.*f" % (nd, x))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--registration", default=REG)
    ap.add_argument("--out")
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    s = {"run": os.path.relpath(os.path.abspath(a.run), ROOT), "registration": os.path.relpath(a.registration, ROOT),
         "registered": reg["registered"], "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "A": score_map(a.run, reg), "B": score_idle(a.run, reg), "states": states(a.run)}
    with open(a.out or os.path.join(a.run, "score.json"), "w") as f:
        json.dump(s, f, indent=1)
    report(s)


if __name__ == "__main__":
    main()
