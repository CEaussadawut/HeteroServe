#!/usr/bin/env python3
"""NS4-GAP and NS4-SHAPED for the network study's third registration (register_ns4c_20261024.py), called by
ab_score_20261024.py as its scorer_extra: extra(session, reg, score) -> dict. As ns4_extra_20261024.py, with NS4-SHAPED
also requiring the shaped arm-run's traffic to have passed through the shaping: ns4b's qdiscs were installed and
verified while no RPC packet matched their filter (the flow was on the overlay addresses). ns4c_shape_20261024.sh appends,
before each clear, one line of the netem qdiscs' own counters to <session>/ns4c_counters.log (in session order); each
shaped arm-run needs at least MIN_PKTS packets through netem in each direction. ns4_extra_20261024.py's notes follow.

NS4-GAP and NS4-SHAPED for the four-node network study (register_ns4_20261024.py), called by ab_score_20261024.py as its
scorer_extra: extra(session, reg, score) -> dict.

NS4-GAP     the head-last penalty, mean ms per token of L minus H at a setting, rises as the rate falls: U < 20 < 5.
NS4-SHAPED  ns4_shape_20261024.sh appends one JSON line per shaped arm-run to <session>/ns4_shape.log (in session order),
            with the epoch the shaping was in place; every shaped arm-run's line must be ok and earlier than its first
            scored request (qr3's sent_wall of request 0)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402
MIN_PKTS = 1000      # an arm-run is ~1,700 tokens and every decoded token moves at least 8 KB each way over the RPC link


def extra(session, reg, score):
    arms = score["arms"]
    v = lambda k: (arms.get(k) or {}).get("mean_ms_per_token")
    have = all(v(h + s) is not None and (arms.get(h + s) or {}).get("complete") for s in ("U", "20", "5") for h in "HL")
    gaps = {s: round(v("L" + s) - v("H" + s), 2) for s in ("U", "20", "5")} if have else None
    gap = {"verdict": "INCOMPLETE" if not have else ("HOLDS" if gaps["U"] < gaps["20"] < gaps["5"] else "FAILS"), "gaps_ms": gaps}
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
    sh = {"verdict": "INCOMPLETE" if not shaped else ("HOLDS" if all(c["ok"] for c in checks) else "FAILS"), "checks": checks}
    return {"hypotheses": {"NS4-GAP": gap, "NS4-SHAPED": sh}}
