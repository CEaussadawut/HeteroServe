#!/usr/bin/env python3
"""Score the test of VISIT (register_visit_20261009.py) against its registered hypotheses, as written.

    python3 testbed/scripts/score_visit_20261009.py testbed/results/ring_laps_<stamp> [--reg <registration>]

Reads <dir>/run.log (ring_laps.sh, parsed as score_ring_laps_20261009.parse does) and <dir>/throttle_pi4.log. Per cell:
decode and prefill ms/token (the mean of llama-bench's samples), each RPC device's model buffer checked against its stage's
blocks (and the output layer on the last device) within 0.5 MiB, with no other RPC device, the Pi 4's thermal verdict over
the cell (full_eval.thermal), and the bench's exit.
  VISIT-J    least-squares slope of decode over Jetson visits in h1, h2, h4 (every round) minus RTT1's predicted slope, in
             VISIT_MS +- 0.5 ms
  VISIT-P    the same over loopback visits in a1 and p4
  VISIT-MAE  VISIT's mean absolute decode error over the six layouts (each layout's mean over rounds) below RTT1's
A cell that did not exit 0, or whose layout check fails, is reported and left out; a hypothesis that then lacks a registered
cell is INCOMPLETE. VISIT is ADOPTED when VISIT-J and VISIT-MAE hold. Exploratory: the residual over RTT1 fitted on every
cell as a constant plus a cost per Jetson visit and per loopback visit; prefill's slope over Jetson visits.
Writes <dir>/score.json.
"""
import hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import score_ring_laps_20261009 as SR                        # noqa: E402

REG = os.path.join(ROOT, "testbed", "dataset", "visit_predictions_20261009.json")
MIB = 2 ** 20
BAND_MS = 0.5


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def expected_buffers(layout, q):
    """{endpoint: MiB} for a registered layout: each RPC stage's blocks, and the output layer on the last device. An
    in-process host stage (the first, "pi4 (host)") is no device; its blocks still count toward the block index."""
    eps = layout["rpc"].split(",")
    out, start, i = {}, 0, 0
    for st in layout["stages"]:
        if st["board"] != "pi4 (host)":
            b = sum(q["layer_bytes"][start:start + st["blocks"]]) + (q["head_bytes"] if i == len(eps) - 1 else 0)
            out[eps[i]] = b / MIB
            i += 1
        start += st["blocks"]
    return out


def lstsq3(rows):
    """Least squares of y on (1, a, b) for rows (a, b, y): (const, coef a, coef b), or None when singular."""
    X = [(1.0, a, b) for a, b, _ in rows]
    y = [r[2] for r in rows]
    A = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
    v = [sum(x[i] * t for x, t in zip(X, y)) for i in range(3)]
    for c in range(3):                                         # Gauss-Jordan with partial pivoting
        p = max(range(c, 3), key=lambda r: abs(A[r][c]))
        if abs(A[p][c]) < 1e-12:
            return None
        A[c], A[p], v[c], v[p] = A[p], A[c], v[p], v[c]
        for r in range(3):
            if r != c:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
                v[r] -= f * v[c]
    return tuple(v[i] / A[i][i] for i in range(3))


def score(reg, q, cells, poll_lines=None, thermal=None):
    labs, V = list(reg["layouts"]), reg["rule_under_test"]["visit_ms"]
    good, bad = {}, {}
    for key, c in cells.items():
        r, st, lab = key
        lay = reg["layouts"][lab]
        exp = expected_buffers(lay, q)
        diff = {ep: round(c["bufs"].get(ep, -1.0) - mib, 2) for ep, mib in exp.items()}
        ok_layout = all(abs(d) <= 0.5 for d in diff.values()) and set(c["bufs"]) == set(exp)
        th = thermal(poll_lines, c["t0"], c["t1"])["verdict"] if (poll_lines and thermal and c["t1"]) else "UNKNOWN"
        row = {"round": r, "setting": st, "layout": lab, "jetson_visits": lay["jetson_visits"],
               "loopback_visits": lay["loopback_visits"], "decode_ms": c["tg"], "prefill_ms": c["pp"], "exit": c["exit"],
               "layout_ok": ok_layout, "buffer_diff_mib": diff, "thermal": th}
        if c["exit"] != 0 or not ok_layout or c["tg"] is None or c["pp"] is None:
            bad["%d.%s.%s" % key] = row
        else:
            good[key] = row
    want = {(r, "s0", lab) for r in range(1, reg["rounds"] + 1) for lab in labs}
    missing = sorted("%d.%s.%s" % k for k in want - set(good))
    hyp = {}
    for h, series in reg["series"].items():
        kind = "jetson_visits" if h == "VISIT-J" else "loopback_visits"
        rows = [v for v in good.values() if v["layout"] in series]
        complete = len(rows) == len(series) * reg["rounds"]
        sl = SR.slope([v[kind] for v in rows], [v["decode_ms"] for v in rows]) if len({v[kind] for v in rows}) > 1 else None
        pred = reg["predicted_slopes_ms_per_visit"][h]
        beyond = sl - pred["rtt1"] if sl is not None else None
        inside = beyond is not None and V - BAND_MS <= beyond <= V + BAND_MS
        hyp[h] = {"verdict": ("HOLDS" if inside else "FAILS") if complete else "INCOMPLETE",
                  "slope_ms_per_visit": round(sl, 3) if sl is not None else None, "predicted_slope_rtt1": pred["rtt1"],
                  "beyond_rtt1_ms": round(beyond, 3) if beyond is not None else None,
                  "band_ms": [round(V - BAND_MS, 3), round(V + BAND_MS, 3)], "cells": len(rows)}
    acc = {}
    for lab in labs:
        xs = [v["decode_ms"] for v in good.values() if v["layout"] == lab]
        if xs:
            m = statistics.fmean(xs)
            p1, pv = (reg["layouts"][lab]["predicted"][k]["decode_ms"] for k in ("rtt1", "visit"))
            acc[lab] = {"measured_ms": round(m, 2), "rtt1_ms": p1, "visit_ms": pv, "err_rtt1_pct": round(100 * (p1 / m - 1), 2),
                        "err_visit_pct": round(100 * (pv / m - 1), 2), "rounds": len(xs)}
    complete = len(acc) == len(labs) and all(v["rounds"] == reg["rounds"] for v in acc.values())
    mae = {k: round(statistics.fmean(abs(v["err_%s_pct" % k]) for v in acc.values()), 2) if acc else None
           for k in ("rtt1", "visit")}
    hyp["VISIT-MAE"] = {"verdict": ("HOLDS" if mae["visit"] < mae["rtt1"] else "FAILS") if complete else "INCOMPLETE",
                        "mae_pct": mae, "layouts": acc}
    adopted = hyp["VISIT-J"]["verdict"] == "HOLDS" and hyp["VISIT-MAE"]["verdict"] == "HOLDS"
    # exploratory: the residual over RTT1 as a constant + a cost per Jetson visit + a cost per loopback visit
    res = [(v["jetson_visits"], v["loopback_visits"],
            v["decode_ms"] - reg["layouts"][v["layout"]]["predicted"]["rtt1"]["decode_ms"]) for v in good.values()]
    fit = lstsq3(res) if len(res) >= 4 else None
    hs = [v for v in good.values() if v["layout"] in reg["series"]["VISIT-J"]]
    pre = SR.slope([v["jetson_visits"] for v in hs], [v["prefill_ms"] for v in hs]) if len({v["jetson_visits"] for v in hs}) > 1 else None
    expl = {"residual_over_rtt1_fit": {"const_ms": round(fit[0], 3), "per_jetson_visit_ms": round(fit[1], 3),
                                       "per_loopback_visit_ms": round(fit[2], 3), "cells": len(res)} if fit else None,
            "prefill_slope_ms_per_token_per_jetson_visit_h_series": round(pre, 4) if pre is not None else None}
    return {"cells": {"%d.%s.%s" % k: v for k, v in sorted(good.items())}, "left_out": bad, "missing": missing,
            "hypotheses": hyp, "adoption": "ADOPTED" if adopted else "NOT ADOPTED", "exploratory": expl}


def main(argv):
    if not argv:
        sys.exit(__doc__)
    d = argv[0]
    reg_path = argv[argv.index("--reg") + 1] if "--reg" in argv else REG
    import full_eval as F
    import predict_pool as P
    reg = json.load(open(reg_path))
    q = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))[P.MODELS[reg["model"]][0]]
    lines = open(os.path.join(d, "run.log"), encoding="utf-8", errors="replace").read().splitlines()
    cells, pings = SR.parse(lines)
    tp = os.path.join(d, "throttle_pi4.log")
    poll = open(tp, encoding="utf-8", errors="replace").readlines() if os.path.exists(tp) else None
    s = score(reg, q, cells, poll, F.thermal)
    code = [f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh")) and os.path.exists(os.path.join(HERE, f))
            and sha(os.path.join(HERE, f)) != h]
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": code, "pings_ms": {"%d.%s" % k: v for k, v in sorted(pings.items())}, **s}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print("%-4s %8s %8s %8s %8s %8s %8s  %s" % ("lay", "r1", "r2", "r3", "mean", "RTT1", "VISIT", "thermal"))
    for lab, lay in reg["layouts"].items():
        xs = [s["cells"].get("%d.s0.%s" % (r, lab), {}).get("decode_ms") for r in range(1, reg["rounds"] + 1)]
        th = sorted({s["cells"].get("%d.s0.%s" % (r, lab), {}).get("thermal") for r in range(1, reg["rounds"] + 1)} - {None})
        ok = [x for x in xs if x is not None]
        print("%-4s %s %8s %8.2f %8.2f  %s" % (lab, " ".join("%8.2f" % x if x is not None else "       -" for x in xs),
                                              "%.2f" % statistics.fmean(ok) if ok else "-", lay["predicted"]["rtt1"]["decode_ms"],
                                              lay["predicted"]["visit"]["decode_ms"], th))
    print("pings: %s" % s["pings_ms"])
    for h, v in s["hypotheses"].items():
        print("%-9s %s  %s" % (h, v["verdict"], {k: w for k, w in v.items() if k not in ("verdict", "layouts")}))
    print("VISIT-MAE layouts: %s" % s["hypotheses"]["VISIT-MAE"]["layouts"])
    print("adoption: %s" % s["adoption"])
    print("exploratory: %s" % s["exploratory"])
    print("left out: %s; missing: %s; code changed since registration: %s" % (list(s["left_out"]), s["missing"], code))
    print("wrote %s" % os.path.relpath(os.path.join(d, "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
