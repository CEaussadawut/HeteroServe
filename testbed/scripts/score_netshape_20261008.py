#!/usr/bin/env python3
"""Score the network study (register_netshape_20261008.py) against its registered hypotheses, as written.

    python3 testbed/scripts/score_netshape_20261008.py testbed/results/netshape_<stamp>

Reads the registration, compare_pool.py's results (testbed/dataset/pool_results_20261008netshape.json: each cell's measured
decode and prefill, exit, thermal verdict) and each setting's measured link (<dir>/link_<setting>.log, link_to_json.parse).
Every cell is RE-PRICED with predict_pool.predict at its setting's measured link: the Jetson's link takes the measured
Mbit/s (the slower direction's mean) and RTT; the rest is the registration's own inputs, which must be unchanged.
  NS-FLIP   per setting: where the re-priced layouts differ by >= 3 %, the layout predicted faster measures faster
  NS-CROSS  the measured crossover (linear in 1/bandwidth between the bracketing settings s0, b200, b100, b50, b20) within a
            factor 1.5 of the re-priced crossover (bisection over bandwidth with the same inputs)
  NS-ACC    every cell's measured decode within 10 % of its re-priced prediction
  NS-RTT    per layout, the slope through the origin of decode's increase over s0 (d1, d5) against the measured RTT
            increase, in [1.5, 2.5]
A cell that did not exit 0 and finish is reported and left out; a hypothesis that then lacks its cells is INCOMPLETE.
Writes <dir>/score.json.
"""
import copy, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402
import link_to_json as lj                                   # noqa: E402

PRED = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261008netshape_r2.json")   # r2 supersedes the first (router change)
RES = os.path.join(ROOT, "testbed", "dataset", "pool_results_20261008netshape_r2.json")
BW = ("s0", "b200", "b100", "b50", "b20")
FLIP_MIN, ACC_MAX, CROSS_FACTOR, RTT_SLOPE = 0.03, 0.10, 1.5, (1.5, 2.5)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def measured_crossover(points):
    """points: [(mbps, gap_ms)], gap = host - last. The bandwidth where the gap crosses zero, from the first adjacent pair
    (in falling bandwidth) whose signs differ, linear in 1/bandwidth; None without a sign change."""
    pts = sorted(points, key=lambda p: -p[0])
    for (b1, g1), (b2, g2) in zip(pts, pts[1:]):
        if g1 == 0:
            return b1
        if (g1 > 0) != (g2 > 0):
            x1, x2 = 1.0 / b1, 1.0 / b2
            x = x1 + (0 - g1) * (x2 - x1) / (g2 - g1)
            return 1.0 / x
    return None


def slope_through_origin(xs, ys):
    """Least squares y = k x through the origin; None for no points or all x zero."""
    sxx = sum(x * x for x in xs)
    return None if not xs or sxx == 0 else sum(x * y for x, y in zip(xs, ys)) / sxx


def main(argv):
    if not argv:
        sys.exit(__doc__)
    out_dir = argv[0]
    reg, res = json.load(open(PRED)), json.load(open(RES))
    ds = os.path.join(ROOT, "testbed", "dataset")
    # The registered bytes of each input (2026-10-20): the live file when it still matches, else the read-only dated copy
    # kept when it changed (characterisation.20261020_before_merge.csv, model_shapes.20261011_before_q6k.json,
    # memory_at_rest.20260927.json), found by its sha256. Nothing matching: refused, as before.
    src = {}
    for f in ("characterisation.csv", "model_shapes.json", "memory_at_rest.json"):
        stem, ext = os.path.splitext(f)
        cands = [os.path.join(ds, f)] + sorted(glob.glob(os.path.join(ds, "%s.*%s" % (stem, ext))))
        src[f] = next((c for c in cands if sha(c) == reg["inputs_sha256"][f]), None)
    changed = [f for f, p in src.items() if p is None]
    if changed:
        sys.exit("refused: %s changed since the registration; the re-pricing would not be the registered model" % changed)
    code_changed = [f for f in reg["inputs_sha256"] if f.endswith((".py", ".sh"))
                    and sha(os.path.join(HERE, f)) != reg["inputs_sha256"][f]]                  # reported, not refused
    shapes = json.load(open(src["model_shapes.json"]))
    R = pp.rows(src["characterisation.csv"], "char")
    first = reg["configs"][reg["run_order"][0]]
    q = shapes[first["model"]]
    C, _ = pp.costs(R, shapes, first["quant"])
    assign = [(p["node"], p["blocks"]) for p in first["pipeline"]]
    links = {}
    for s in reg["settings"]:
        p = os.path.join(out_dir, "link_%s.log" % s["id"])
        if os.path.exists(p):
            _, _, r = lj.parse(p)
            links[s["id"]] = {"mbps": r["mbps"], "rtt_ms": r["rtt_ms"], "log": os.path.relpath(p, ROOT),
                              "directions": {d: v["mbps_mean"] for d, v in r["directions"].items()}}
    cells, skipped = {}, {}
    for cid in reg["run_order"]:
        c, m = reg["configs"][cid], res["cells"].get(cid)
        sid = c["setting"]
        if not m or m.get("exit") != 0 or not m.get("finished") or sid not in links:
            skipped[cid] = "no result" if not m else ("no link log" if sid not in links else "exit %s" % m.get("exit"))
            continue
        L = {"jetson": {"mbps": links[sid]["mbps"], "rtt_ms": links[sid]["rtt_ms"]}}
        dec, pre = pp.predict(assign, q, C, L, head_at=c["head_at"])
        md, mpre = m["decode"]["measured"]["ms_per_token"], m["prefill"]["measured"]["ms_per_token"]
        cells[cid] = {"setting": sid, "layout": c["layout"], "decode_ms": md, "prefill_ms": mpre,
                      "repriced_decode_ms": round(dec, 1), "repriced_prefill_ms": round(pre, 1),
                      "nominal_decode_ms": c["predicted"]["decode_ms_per_token"],
                      "decode_err_pct": round(100 * (dec / md - 1), 1), "prefill_err_pct": round(100 * (pre / mpre - 1), 1),
                      "thermal": (m.get("thermal") or {}).get("verdict"), "page_in": (m.get("page_in") or {}).get("flag")}
    by = {(x["setting"], x["layout"]): x for x in cells.values()}
    hyp = {}
    # NS-ACC
    acc = [abs(x["decode_err_pct"]) <= 100 * ACC_MAX for x in cells.values()]
    hyp["NS-ACC"] = {"verdict": ("HOLDS" if all(acc) else "FAILS") if len(cells) == len(reg["run_order"]) else
                     ("FAILS" if not all(acc) else "INCOMPLETE"),
                     "worst_pct": max((x["decode_err_pct"] for x in cells.values()), key=abs, default=None)}
    # NS-FLIP
    flips, judged = {}, []
    for s in reg["settings"]:
        a, b = by.get((s["id"], "last")), by.get((s["id"], "host"))
        if not (a and b):
            flips[s["id"]] = "INCOMPLETE"; continue
        gap_pred = (b["repriced_decode_ms"] - a["repriced_decode_ms"]) / min(a["repriced_decode_ms"], b["repriced_decode_ms"])
        gap_meas = b["decode_ms"] - a["decode_ms"]
        if abs(gap_pred) < FLIP_MIN:
            flips[s["id"]] = "not judged (re-priced gap %+.1f %%; measured host-last %+.1f ms)" % (100 * gap_pred, gap_meas)
            continue
        ok = (gap_pred > 0) == (gap_meas > 0)
        judged.append(ok)
        flips[s["id"]] = "%s (re-priced %+.1f %%, measured host-last %+.1f ms)" % ("holds" if ok else "FAILS", 100 * gap_pred,
                                                                                   gap_meas)
    hyp["NS-FLIP"] = {"verdict": "INCOMPLETE" if "INCOMPLETE" in flips.values() and all(judged) else
                      ("HOLDS" if all(judged) else "FAILS"), "settings": flips}
    # NS-CROSS
    pts = [(links[s]["mbps"], by[(s, "host")]["decode_ms"] - by[(s, "last")]["decode_ms"])
           for s in BW if (s, "host") in by and (s, "last") in by]
    mx = measured_crossover(pts)
    base = {"jetson": {"mbps": 0.0, "rtt_ms": links["s0"]["rtt_ms"] if "s0" in links else reg["base_link"]["rtt_ms"]}}

    def gap(mbps):
        L = copy.deepcopy(base); L["jetson"]["mbps"] = mbps
        return pp.predict(assign, q, C, L, head_at=0)[0] - pp.predict(assign, q, C, L, head_at=None)[0]
    lo, hi = 5.0, 1000.0
    px = None
    if (gap(lo) > 0) != (gap(hi) > 0):
        for _ in range(60):
            m_ = (lo * hi) ** 0.5
            if (gap(m_) > 0) == (gap(hi) > 0):
                hi = m_
            else:
                lo = m_
        px = (lo * hi) ** 0.5
    ratio = (mx / px) if (mx and px) else None
    hyp["NS-CROSS"] = {"verdict": "INCOMPLETE" if len(pts) < len(BW) and mx is None else
                       ("HOLDS" if ratio and 1 / CROSS_FACTOR <= ratio <= CROSS_FACTOR else "FAILS"),
                       "measured_mbps": round(mx, 1) if mx else None, "repriced_mbps": round(px, 1) if px else None,
                       "ratio": round(ratio, 2) if ratio else None, "points": [(round(b, 1), round(g, 1)) for b, g in pts]}
    # NS-RTT
    slopes = {}
    for lay in ("last", "host"):
        base_c = by.get(("s0", lay))
        xs, ys = [], []
        for s in ("d1", "d5"):
            x = by.get((s, lay))
            if base_c and x and s in links and "s0" in links:
                xs.append(links[s]["rtt_ms"] - links["s0"]["rtt_ms"]); ys.append(x["decode_ms"] - base_c["decode_ms"])
        k = slope_through_origin(xs, ys) if len(xs) == 2 else None
        slopes[lay] = {"slope": round(k, 2) if k is not None else None, "rtt_increase_ms": [round(v, 3) for v in xs],
                       "decode_increase_ms": [round(v, 1) for v in ys]}
    ks = [v["slope"] for v in slopes.values()]
    hyp["NS-RTT"] = {"verdict": "INCOMPLETE" if None in ks else
                     ("HOLDS" if all(RTT_SLOPE[0] <= k <= RTT_SLOPE[1] for k in ks) else "FAILS"), "layouts": slopes}
    drift = {lay: round(100 * (by[("s0r", lay)]["decode_ms"] / by[("s0", lay)]["decode_ms"] - 1), 2)
             for lay in ("last", "host") if ("s0", lay) in by and ("s0r", lay) in by}
    score = {"registration_sha256": sha(PRED), "results_sha256": sha(RES), "code_changed_since_registration": code_changed,
             "links": links, "cells": cells, "skipped": skipped, "hypotheses": hyp, "drift_s0r_vs_s0_pct": drift}
    json.dump(score, open(os.path.join(out_dir, "score.json"), "w"), indent=1)
    print("%-20s %-5s %-5s %8s %8s %7s %8s %8s  %s" % ("cell", "set", "head", "decode", "repriced", "err%", "prefill", "repriced",
                                                       "thermal"))
    for cid, x in cells.items():
        print("%-20s %-5s %-5s %8.1f %8.1f %+6.1f%% %8.1f %8.1f  %s" % (cid, x["setting"], x["layout"], x["decode_ms"],
              x["repriced_decode_ms"], x["decode_err_pct"], x["prefill_ms"], x["repriced_prefill_ms"], x["thermal"]))
    for s, v in links.items():
        print("link %-5s %7.1f Mbit/s %7.3f ms  %s" % (s, v["mbps"], v["rtt_ms"], v["directions"]))
    for h, v in hyp.items():
        print("%-9s %s  %s" % (h, v["verdict"], {k: w for k, w in v.items() if k != "verdict"}))
    print("drift s0r vs s0 (decode, %%): %s; skipped: %s; code changed since registration: %s" % (drift, skipped, code_changed))
    print("wrote %s" % os.path.relpath(os.path.join(out_dir, "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
