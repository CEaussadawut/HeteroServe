#!/usr/bin/env python3
"""Score sp10 (register_sp10_20261103.py, run by sp10_session_20261103.py): the planner's checking cost on two plans it was not
fitted on.

    python3 testbed/scripts/score_sp10_20261103.py testbed/results/sp10_<stamp> --registration testbed/dataset/sp10_predictions_20261103.json

Per plan, from the session's results.json (spec_cost_20261103.py's reading): its decode step (none's mean ms per token over
both prompts), R(m) = the scan's median T(m) / that step, each MTP arm's speedup per prompt (none's ms per token for that prompt
over the arm's), its tokens per pass and pass time, and the measured draft cost (the least-squares slope of
pass(n) - dec x R(n + 1) over the three settings, both prompts' pass times averaged). A plan is complete with every registered
setting's requests and every scanned size's repetitions. Hypotheses SP-CHECK, SP-SPEEDUP, SP-SIGN, SP-BEST and SP-DRAFT as
registered; reported, not tested: the tokens per pass against the exploration's, and each arm's predicted speedup at its
measured tokens per pass. Writes score.json and score.txt into the session folder.
"""
import argparse, datetime, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import spec_cost_20261103 as SC                             # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402

SPECS = {"mtp1": 1, "mtp2": 2, "mtp3": 3}


def measure(session, reg, c):
    recs = [r for r in json.load(open(os.path.join(session, "results.json"))) if r["config"] == c]
    if not recs:
        return {"complete": False}
    cfg = reg["configs"][c]
    sp = {s["spec"] + "/" + s["prompt"]: s for s in SC.specs(session, recs)}
    none = next((r for r in recs if r["spec"] == "none"), None)
    complete = all(sp.get(s + "/" + p, {}).get("requests", 0) >= reg["reqs"] for s in cfg["specs"] for p in reg["prompt_order"]) \
        and none is not None and all(sum(1 for x in none["scan"] if x["m"] == m and (x.get("timings") or {}).get("prompt_ms"))
                                         >= reg["scan_reps"] for m in reg["scan"])
    out = {"complete": complete}
    if not (sp.get("none/free", {}).get("ms_per_token") and sp.get("none/copy", {}).get("ms_per_token")):
        return out
    dec = statistics.fmean([sp["none/free"]["ms_per_token"], sp["none/copy"]["ms_per_token"]])
    R = {s["m"]: s["prompt_ms"] / dec for s in SC.scan(none) if s["prompt_ms"]} if none else {}
    arms, pts = {}, []
    for name, n in SPECS.items():
        for pk in reg["prompt_order"]:
            a = sp.get(name + "/" + pk, {})
            if a.get("ms_per_token"):
                arms[name + "/" + pk] = {"speedup": round(sp["none/" + pk]["ms_per_token"] / a["ms_per_token"], 4),
                                         "ms_per_token": a["ms_per_token"], "tokens_per_pass": a.get("tokens_per_pass"),
                                         "pass_ms": a.get("ms_per_pass"), "acceptance": a.get("acceptance")}
        ps = [sp.get(name + "/" + pk, {}).get("ms_per_pass") for pk in reg["prompt_order"]]
        if all(ps) and (n + 1) in R:
            pts.append((n, statistics.fmean(ps) - dec * R[n + 1]))
    d = None
    if len(pts) == len(SPECS):
        mn = statistics.fmean(x for x, _ in pts); mr = statistics.fmean(y for _, y in pts)
        d = sum((x - mn) * (y - mr) for x, y in pts) / sum((x - mn) ** 2 for x, _ in pts)
    out.update({"decode_ms": round(dec, 2), "R": {m: round(v, 4) for m, v in R.items()}, "arms": arms,
                "draft_ms": round(d, 2) if d is not None else None, "pass_residuals": [(n, round(r, 1)) for n, r in pts]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    H = reg["hypotheses"]
    res = {c: measure(a.session, reg, c) for c in reg["order"]}
    ok = all(res[c].get("complete") for c in reg["order"])
    vs = lambda cond: ("HOLDS" if cond else "FAILS") if ok else "INCOMPLETE"
    check, speed, sign, best, draft, at_E = [], [], [], [], [], []
    for c in reg["order"]:
        p, m = reg["predictions"][c], res[c]
        for k in H["SP-CHECK"]["m"]:
            pr, me = p["R"][str(k)], m.get("R", {}).get(k)
            check.append({"plan": c, "m": k, "predicted": round(pr, 3), "measured": round(me, 3) if me else None,
                          "err_pct": round(100 * (me / pr - 1), 2) if me else None})
        for arm, pa in p["arms"].items():
            ma = m.get("arms", {}).get(arm, {})
            me = ma.get("speedup")
            speed.append({"plan": c, "arm": arm, "predicted": pa["speedup"], "measured": me,
                          "err_pct": round(100 * (me / pa["speedup"] - 1), 2) if me else None})
            if pa["speedup"] >= 1 + H["SP-SIGN"]["margin"] or pa["speedup"] <= 1 - H["SP-SIGN"]["margin"]:
                sign.append({"plan": c, "arm": arm, "predicted": pa["speedup"], "measured": me,
                             "same_side": (me is not None) and ((me > 1) == (pa["speedup"] > 1))})
            if ma.get("tokens_per_pass"):
                at_E.append({"plan": c, "arm": arm, "E_registered": pa["E"], "E_measured": ma["tokens_per_pass"],
                             "predicted_at_measured_E": round(p["decode_ms"] * ma["tokens_per_pass"] / pa["pass_ms"], 3),
                             "measured": me})
        for pk in reg["prompt_order"]:
            cand = {"none": 1.0, **{s: p["arms"][s + "/" + pk]["speedup"] for s in SPECS}}
            meas = {"none": 1.0, **{s: m.get("arms", {}).get(s + "/" + pk, {}).get("speedup") for s in SPECS}}
            pick = max(cand, key=cand.get)
            top = max((v for v in meas.values() if v), default=None)
            best.append({"plan": c, "prompt": pk, "pick": pick, "pick_measured": meas[pick], "best_measured": top,
                         "best_setting": max((k for k in meas if meas[k]), key=lambda k: meas[k]) if top else None,
                         "within": (meas[pick] is not None and top is not None and meas[pick] >= top * (1 - H["SP-BEST"]["tol"]))})
        md = m.get("draft_ms")
        draft.append({"plan": c, "predicted": p["draft_ms"], "measured": md,
                      "err_pct": round(100 * (md / p["draft_ms"] - 1), 2) if md is not None else None})
    within = lambda rows, tol: all(r["err_pct"] is not None and abs(r["err_pct"]) <= 100 * tol for r in rows)
    hyp = {"SP-CHECK": {"verdict": vs(within(check, H["SP-CHECK"]["tol"])), "rows": check},
           "SP-SPEEDUP": {"verdict": vs(within(speed, H["SP-SPEEDUP"]["tol"])), "rows": speed},
           "SP-SIGN": {"verdict": vs(bool(sign) and all(r["same_side"] for r in sign)), "rows": sign},
           "SP-BEST": {"verdict": vs(all(r["within"] for r in best)), "rows": best},
           "SP-DRAFT": {"verdict": vs(within(draft, H["SP-DRAFT"]["tol"])), "rows": draft}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "plans": res, "hypotheses": hyp, "at_measured_E": at_E,
         "run_notes": re.findall(r"^\S+ ((?:\S+ \S+ ERROR|\S+ \S+ TIMED OUT|\S+ \S+ start rc [1-9]).*)$",
                                 open(os.path.join(a.session, "run.log"), encoding="utf-8", errors="replace").read(), re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["sp10: the planner's checking cost on two plans it was not fitted on; code changed since registration: %s"
         % (s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %s" % (k, h["verdict"]))
        for r in h["rows"]:
            L.append("    %s" % r)
    L.append("")
    for c in reg["order"]:
        L.append("plan %s (%s %s): measured decode %s ms, draft %s ms, R %s" % (
            c, reg["configs"][c]["MODEL"], reg["configs"][c]["PLACE"], res[c].get("decode_ms"), res[c].get("draft_ms"),
            res[c].get("R")))
    L.append("")
    L += ["at measured E: %s" % r for r in at_E]
    L += ["run: %s" % n for n in s["run_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
