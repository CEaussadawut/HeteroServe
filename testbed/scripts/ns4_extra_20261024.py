#!/usr/bin/env python3
"""NS4-GAP and NS4-SHAPED for the four-node network study (register_ns4_20261024.py), called by ab_score_20261024.py as its
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


def extra(session, reg, score):
    arms = score["arms"]
    v = lambda k: (arms.get(k) or {}).get("mean_ms_per_token")
    have = all(v(h + s) is not None and (arms.get(h + s) or {}).get("complete") for s in ("U", "20", "5") for h in "HL")
    gaps = {s: round(v("L" + s) - v("H" + s), 2) for s in ("U", "20", "5")} if have else None
    gap = {"verdict": "INCOMPLETE" if not have else ("HOLDS" if gaps["U"] < gaps["20"] < gaps["5"] else "FAILS"), "gaps_ms": gaps}
    path = os.path.join(session, "ns4_shape.log")
    lines = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    shaped = [r for r in score["runs"] if reg["arms"].get(r["arm"], {}).get("pre")]
    checks = []
    for k, r in enumerate(shaped):
        rr = S4.reqs(os.path.join(ROOT, r["log"])) if r.get("log") else []
        first = min((x["sent_wall"] for x in rr), default=None)
        sw = lines[k] if k < len(lines) else None
        ok = bool(sw and sw.get("ok") and first and float(sw.get("applied_epoch", 1e20)) < first)
        checks.append({"run": r["i"], "arm": r["arm"], "shape": sw, "first_scored_request_epoch": first, "ok": ok})
    sh = {"verdict": "INCOMPLETE" if not shaped else ("HOLDS" if all(c["ok"] for c in checks) else "FAILS"), "checks": checks}
    return {"hypotheses": {"NS4-GAP": gap, "NS4-SHAPED": sh}}
