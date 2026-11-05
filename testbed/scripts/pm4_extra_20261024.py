#!/usr/bin/env python3
"""PM4-SWITCH for the four-node power-mode study (register_pm4_20261024.py), called by ab_score_20261024.py as its
scorer_extra: extra(session, reg, score) -> dict.

pm4_mode_20261024.py set-when-served appends one JSON line per W arm-run to <session>/pm4_switch.log, with the epoch of the
verified switch. The k-th line belongs to the k-th W arm-run (both in session order). PM4-SWITCH HOLDS when every W arm-run
has a verified switch earlier than its first scored request (qr3's sent_wall of request 0), FAILS otherwise, INCOMPLETE
without a W arm-run."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402


def extra(session, reg, score):
    path = os.path.join(session, "pm4_switch.log")
    lines = [json.loads(x) for x in open(path)] if os.path.exists(path) else []
    w_runs = [r for r in score["runs"] if r["arm"] == "W"]
    checks = []
    for k, r in enumerate(w_runs):
        rr = S4.reqs(os.path.join(ROOT, r["log"])) if r.get("log") else []
        first = min((x["sent_wall"] for x in rr), default=None)
        sw = lines[k] if k < len(lines) else None
        ok = bool(sw and sw.get("ok") and first and sw.get("verified_epoch", 1e20) < first)
        checks.append({"run": r["i"], "switch": sw, "first_scored_request_epoch": first,
                       "margin_s": round(first - sw["verified_epoch"], 1) if (sw and first and "verified_epoch" in sw) else None,
                       "ok": ok})
    verdict = "INCOMPLETE" if not w_runs else ("HOLDS" if all(c["ok"] for c in checks) else "FAILS")
    return {"hypotheses": {"PM4-SWITCH": {"verdict": verdict, "checks": checks}}}
