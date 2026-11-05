#!/usr/bin/env python3
"""Score sp11 (register_sp11_20261104.py, run by sp11_session_20261104.py): the MTP 1 default across many prompts, and why a
plan's acceptance on free text moves.

    python3 testbed/scripts/score_sp11_20261104.py testbed/results/sp11_<stamp> --registration testbed/dataset/sp11_predictions_20261104.json

Per plan and prompt, from the session's results.json: none's ms per token, MTP 1's ms per token, the speedup (none's over
MTP 1's), MTP 1's tokens per pass E = (predicted_n - 1) / passes with passes = predicted_n - 1 - draft_n_accepted (as
spec_cost_20261103.py), and the model's speedup at that E, decode_ms x E / pass_ms (the registered decode step and MTP 1 pass of
the plan). Texts: each request's token ids; the first position where two texts of a prompt differ; at that position, the top
two probabilities of the plain run's n_probs pass (the config marked "nprobs"), read from its logprobs.
A plan is complete with every prompt's request under both settings. Hypotheses as registered (MT-COST, MT-SIGN, MT-MEAN,
MT-WORST, TX-TIE, TX-E); every one is INCOMPLETE unless every plan is complete. Writes score.json and score.txt.
"""
import argparse, datetime, json, math, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402


def first_diff(a, b):
    """The first position where token lists a and b differ (None when one is a prefix of the other and they are as long)."""
    if a is None or b is None:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def per_plan(recs, reg, c):
    out = {"complete": False, "prompts": {}}
    by = {r["spec"]: r for r in recs if r["config"] == c}
    if not ("none" in by and "mtp1" in by):
        return out
    req = {s: {q["prompt"]: q for q in by[s]["requests"] if q.get("timings")} for s in ("none", "mtp1")}
    out["complete"] = all(pk in req["none"] and pk in req["mtp1"] for pk in reg["prompt_order"])
    p = reg["predictions"][c]
    for pk in reg["prompt_order"]:
        n, m = req["none"].get(pk), req["mtp1"].get(pk)
        row = {"tokens_none": n.get("tokens") if n else None, "tokens_mtp1": m.get("tokens") if m else None}
        if n and m:
            tn, tm = n["timings"], m["timings"]
            passes = (tm.get("predicted_n") or 0) - 1 - (tm.get("draft_n_accepted") or 0)
            E = ((tm.get("predicted_n") or 0) - 1) / passes if passes else None
            sp = tn["predicted_per_token_ms"] / tm["predicted_per_token_ms"]
            model = p["decode_ms"] * E / p["pass_ms"] if E else None
            row.update({"none_ms": round(tn["predicted_per_token_ms"], 2), "mtp1_ms": round(tm["predicted_per_token_ms"], 2),
                        "speedup": round(sp, 4), "E": round(E, 3) if E else None, "drafted": tm.get("draft_n"),
                        "accepted": tm.get("draft_n_accepted"), "model_at_E": round(model, 4) if model else None,
                        "cost_err_pct": round(100 * (sp / model - 1), 2) if model else None,
                        "text_diverges_at": first_diff(n.get("tokens"), m.get("tokens"))})
        out["prompts"][pk] = row
    probs = {q["prompt"]: q for q in by["none"].get("probs_pass", []) if q.get("probs")}
    out["probs"] = probs
    return out


def margin(probs_rec, k):
    """The plain run's top-two probability gap at position k (from the n_probs pass), and whether its token there is the
    timed plain run's."""
    if probs_rec is None or k is None or k >= len(probs_rec["probs"]):
        return None
    top = probs_rec["probs"][k]["top"]
    if len(top) < 2:
        return None
    p1, p2 = math.exp(top[0][1]), math.exp(top[1][1])
    return {"position": k, "p1": round(p1, 4), "p2": round(p2, 4), "gap": round(p1 - p2, 4), "top_ids": [top[0][0], top[1][0]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    H = reg["hypotheses"]
    recs = json.load(open(os.path.join(a.session, "results.json")))
    res = {c: per_plan(recs, reg, c) for c in reg["order"]}
    ok = all(res[c]["complete"] for c in reg["order"])
    vs = lambda cond: ("HOLDS" if cond else "FAILS") if ok else "INCOMPLETE"
    cost, sign, mean, worst = [], [], [], []
    for c in reg["order"]:
        p = reg["predictions"][c]
        Estar = p["pass_ms"] / p["decode_ms"]
        sps = []
        for pk in reg["prompt_order"]:
            r = res[c]["prompts"].get(pk, {})
            if r.get("speedup") is None:
                continue
            sps.append(r["speedup"])
            cost.append({"plan": c, "prompt": pk, "E": r["E"], "measured": r["speedup"], "model_at_E": r["model_at_E"],
                         "err_pct": r["cost_err_pct"]})
            if abs(r["speedup"] - 1) > H["MT-SIGN"]["dead_band"]:
                sign.append({"plan": c, "prompt": pk, "E": r["E"], "E_breakeven": round(Estar, 3), "measured": r["speedup"],
                             "agrees": (r["speedup"] > 1) == (r["E"] > Estar)})
        g = math.exp(statistics.fmean(math.log(x) for x in sps)) if sps else None
        mean.append({"plan": c, "geomean_speedup": round(g, 4) if g else None, "n": len(sps),
                     "faster_on": sum(1 for x in sps if x > 1), "registered_E_speedup": p.get("speedup_at_registered_E")})
        worst.append({"plan": c, "slowest": round(min(sps), 4) if sps else None})
    # the texts (TX-*): pairs against the plain run of the nprobs config, then the two plans' MTP 1 texts
    tie, tx_e = [], []
    pc = next((c for c in reg["order"] if reg["configs"][c].get("nprobs")), None)
    pair = H["TX-E"]["plans"] if "TX-E" in H else None     # 2026-11-04: sp12 (the 35B alone) registers no TX-E
    for pk in reg["prompt_order"]:
        base = res[pc]["prompts"].get(pk, {}) if pc else {}
        pr = res[pc]["probs"].get(pk) if pc else None
        plain = base.get("tokens_none")
        if pr is not None and plain is not None and pr.get("tokens") != plain:
            tie.append({"prompt": pk, "note": "the n_probs pass's text differs from the timed plain run's at %s"
                        % first_diff(pr.get("tokens"), plain)})
        for c in reg["order"]:
            if reg["configs"][c]["MODEL"] != reg["configs"][pc]["MODEL"]:
                continue
            for s in ("none", "mtp1"):
                if c == pc and s == "none":
                    continue
                other = res[c]["prompts"].get(pk, {}).get("tokens_" + s)
                k = first_diff(plain, other)
                if k is not None:
                    mg = margin(pr, k)
                    tie.append({"prompt": pk, "vs": "%s %s" % (c, s), "position": k, "plain_top2": mg,
                                "near_tie": (mg is not None and mg["gap"] < H["TX-TIE"]["gap"])})
        A, B = (res[x]["prompts"].get(pk, {}) for x in pair) if pair else ({}, {})
        if A.get("E") and B.get("E"):
            same = A.get("tokens_mtp1") is not None and A.get("tokens_mtp1") == B.get("tokens_mtp1")
            dE = 100 * abs(A["E"] / B["E"] - 1)
            tx_e.append({"prompt": pk, "texts_same": same, "E": [A["E"], B["E"]], "E_diff_pct": round(dE, 2),
                         "agrees": True if dE <= H["TX-E"]["tol_pct"] else (not same)})
    within = lambda rows, tol: bool(rows) and all(r["err_pct"] is not None and abs(r["err_pct"]) <= 100 * tol for r in rows)
    hyp = {"MT-COST": {"verdict": vs(within(cost, H["MT-COST"]["tol"])), "rows": cost},
           "MT-SIGN": {"verdict": vs(bool(sign) and all(r["agrees"] for r in sign)), "rows": sign},
           "MT-MEAN": {"verdict": vs(all(r["geomean_speedup"] is not None and r["geomean_speedup"] >= H["MT-MEAN"]["min"] for r in mean)),
                       "rows": mean},
           "MT-WORST": {"verdict": vs(all(r["slowest"] is not None and r["slowest"] >= H["MT-WORST"]["min"] for r in worst)),
                        "rows": worst},
           "TX-TIE": {"verdict": vs(bool([r for r in tie if "near_tie" in r]) and all(r["near_tie"] for r in tie if "near_tie" in r)
                                    and not any("note" in r for r in tie)), "rows": tie},
           "TX-E": {"verdict": vs(bool(tx_e) and all(r["agrees"] for r in tx_e)), "rows": tx_e}}
    hyp = {k: v for k, v in hyp.items() if k in H}          # only what was registered (sp11: all six; sp12: no TX-E)
    for c in res:
        for r in res[c]["prompts"].values():
            r.pop("tokens_none", None); r.pop("tokens_mtp1", None)
        res[c].pop("probs", None)
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "plans": res, "hypotheses": hyp,
         "run_notes": re.findall(r"^\S+ ((?:\S+ \S+ ERROR|\S+ \S+ TIMED OUT|\S+ \S+ start rc [1-9]).*)$",
                                 open(os.path.join(a.session, "run.log"), encoding="utf-8", errors="replace").read(), re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["sp11: the MTP 1 default across %d prompts, and the texts behind a plan's acceptance; code changed since registration: %s"
         % (len(reg["prompt_order"]), s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-9s %s" % (k, h["verdict"]))
        for r in h["rows"]:
            L.append("    %s" % r)
    L.append("")
    for c in reg["order"]:
        L.append("plan %s (%s %s), complete %s:" % (c, reg["configs"][c]["MODEL"], reg["configs"][c]["PLACE"], res[c]["complete"]))
        for pk, r in res[c]["prompts"].items():
            L.append("    %-9s none %s ms, mtp1 %s ms, speedup %s, E %s, model at E %s, text leaves plain at %s" % (
                pk, r.get("none_ms"), r.get("mtp1_ms"), r.get("speedup"), r.get("E"), r.get("model_at_E"), r.get("text_diverges_at")))
    L += ["run: %s" % n for n in s["run_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
