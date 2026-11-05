#!/usr/bin/env python3
"""RP-SHAPED and RP-BETTER for the RPC link term (register_rpc4plan_20261024.py), called by ab_score_20261024.py as its
scorer_extra: extra(session, reg, score) -> dict.
  RP-SHAPED  bs4_extra_20261024.py's check under this study's name: every shaped arm-run's line in <session>/ns4_shape.log
             ok and earlier than its first scored request, and its netem counters in <session>/ns4c_counters.log at least
             MIN_PKTS packets in each direction.
  RP-BETTER  for every arm with a measurement, |measured / new prediction - 1| < |measured / old prediction - 1|, where the
             new prediction is the arm's "predicted" and the old one its "predicted_old" (mean ms per token)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402
MIN_PKTS = 1000


def extra(session, reg, score):
    path = os.path.join(session, "ns4_shape.log")
    lines = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    cpath = os.path.join(session, "ns4c_counters.log")
    counts = [json.loads(x) for x in open(cpath)] if os.path.exists(cpath) else []
    shaped = [r for r in score["runs"] if reg["arms"].get(r["arm"], {}).get("pre")]
    checks = []
    for k, r in enumerate(shaped):
        rr = S4.reqs(os.path.join(ROOT, r["log"])) if r.get("log") else []
        first = min((x["sent_wall"] for x in rr), default=None)
        sw = lines[k] if k < len(lines) else None
        ct = {x: y for x, y in counts[k].items() if x != "raw"} if k < len(counts) else None
        through = bool(ct and (ct.get("egress_pkts") or 0) >= MIN_PKTS and (ct.get("ingress_pkts") or 0) >= MIN_PKTS)
        ok = bool(sw and sw.get("ok") and first and float(sw.get("applied_epoch", 1e20)) < first) and through
        checks.append({"run": r["i"], "arm": r["arm"], "shape": sw, "first_scored_request_epoch": first, "counters": ct,
                       "traffic_through_netem": through, "ok": ok})
    shaped_v = {"verdict": "INCOMPLETE" if not shaped else ("HOLDS" if all(c["ok"] for c in checks) else "FAILS"), "checks": checks}
    errs, complete = {}, True
    for k, arm in reg["arms"].items():
        m = score["arms"].get(k, {}).get("mean_ms_per_token")
        if not m or not score["arms"].get(k, {}).get("complete"):
            complete = False
            continue
        new, old = arm["predicted"]["mean_ms_per_token"], arm["predicted_old"]["mean_ms_per_token"]
        errs[k] = {"measured": m, "new": new, "old": old, "err_new_pct": round(100 * (m / new - 1), 2),
                   "err_old_pct": round(100 * (m / old - 1), 2)}
    better = {"verdict": "INCOMPLETE" if not complete else
              ("HOLDS" if all(abs(v["err_new_pct"]) < abs(v["err_old_pct"]) for v in errs.values()) else "FAILS"), "arms": errs}
    return {"hypotheses": {"RP-SHAPED": shaped_v, "RP-BETTER": better}}
