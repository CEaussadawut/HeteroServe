#!/usr/bin/env python3
"""Score the two-user test against testbed/dataset/twouser_predictions_20260929.json, applying its registered rules as written:

    python3 .handoff/assets/pd-2026-09-29/score_twouser.py testbed/results/pd_twouser_<stamp>.log
    python3 .handoff/assets/pd-2026-09-29/score_twouser.py <log> --reg testbed/dataset/twouser_predictions_20260929_gen128.json
                                       --out testbed/dataset/twouser_results_20260929_gen128.json     # TWOUSER128

HANDOFF, NOREPACK, ACCURACY, STATIC-CHECK and STEADY are read from the registration; every figure comes from the run's own log
(the "result" lines, and the servers' timings in the user lines for STATIC-CHECK). Writes testbed/dataset/twouser_results_20260929.json.
"""
import json, os, re, statistics, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REG = os.path.join(ROOT, "testbed", "dataset", "twouser_predictions_20260929.json")
OUT = os.path.join(ROOT, "testbed", "dataset", "twouser_results_20260929.json")


def parse(path):
    res, users = {}, {}
    for line in open(path):
        m = re.match(r"(\S+) r(\d+) (\S+) (result|user1|user2|user2 prefill|user2 decode) (\{.*\})$", line.strip())
        if not m:
            continue
        rep, arm, kind, obj = int(m.group(2)), m.group(3), m.group(4), json.loads(m.group(5))
        if kind == "result":
            res.setdefault(arm, []).append(obj)
        else:
            users.setdefault(arm, {}).setdefault(kind, []).append(obj)
    return res, users


def main(path):
    reg = json.load(open(REG))
    pred = reg["predicted"]
    res, users = parse(path)
    out = {"log": os.path.relpath(path, ROOT), "registration_sha256": None, "arms": {}, "rules": {}}
    import hashlib
    out["registration_sha256"] = hashlib.sha256(open(REG, "rb").read()).hexdigest()
    mean = lambda v: statistics.fmean(v) if v else float("nan")
    spread = lambda v: 100 * (max(v) - min(v)) / statistics.fmean(v) if len(v) > 1 else 0.0
    for arm, rs in res.items():
        md = [(r["user1"]["done_s"] + r["user2"]["done_s"]) / 2 for r in rs]
        a = {"n": len(rs), "mean_done_s": round(mean(md), 2), "mean_done_by_rep": [round(x, 2) for x in md],
             "makespan_s": round(mean([r["makespan_s"] for r in rs]), 2),
             "mean_ttft_s": round(mean([(r["user1"]["ttft_s"] + r["user2"]["ttft_s"]) / 2 for r in rs]), 2),
             "mean_tpot_ms": round(mean([(r["user1"]["tpot_ms"] + r["user2"]["tpot_ms"]) / 2 for r in rs]), 1),
             "steady": spread(md) <= 5, "spread_pct": round(spread(md), 2)}
        for u in ("user1", "user2"):
            a[u] = {k: round(mean([r[u][k] for r in rs]), 2) for k in ("ttft_s", "done_s", "tpot_ms", "tpot_max_ms")}
        p = pred.get(arm, {})
        if "mean_done_range_s" in p:
            lo, hi = p["mean_done_range_s"]
            near = lo if abs(a["mean_done_s"] - lo) < abs(a["mean_done_s"] - hi) else hi
            a["predicted_mean_done_s"], a["accuracy_basis_s"] = p["mean_done_s"], near
        else:
            a["predicted_mean_done_s"] = near = p.get("mean_done_s")
        a["error_pct"] = round(100 * (near - a["mean_done_s"]) / a["mean_done_s"], 2) if near else None
        a["accuracy_holds"] = a["error_pct"] is not None and abs(a["error_pct"]) <= 10
        out["arms"][arm] = a
    A = out["arms"]
    # REVERSAL (TWOUSER128): shared_nr's mean completion below the hand-off's
    if any(r.startswith("REVERSAL") for r in reg["rules"]) and all(k in A for k in ("handoff", "shared_nr")):
        out["rules"]["REVERSAL"] = {"holds": A["shared_nr"]["mean_done_s"] < A["handoff"]["mean_done_s"],
                                    "shared_nr": A["shared_nr"]["mean_done_s"], "handoff": A["handoff"]["mean_done_s"]}
    # the alternative prediction, when the registration carries one: reported, not a rule
    for alt, v in (reg.get("predicted_alt") or {}).items():
        for arm, a in A.items():
            if arm in v:
                a.setdefault("alt", {})[alt] = {"mean_done_s": v[arm]["mean_done_s"],
                                                "error_pct": round(100 * (v[arm]["mean_done_s"] - a["mean_done_s"]) / a["mean_done_s"], 2)}
    # HANDOFF
    if all(k in A for k in ("handoff", "shared", "shared_nr")):
        ok = A["handoff"]["mean_done_s"] < A["shared"]["mean_done_s"] and A["handoff"]["mean_done_s"] < A["shared_nr"]["mean_done_s"]
        out["rules"]["HANDOFF"] = {"holds": ok, "handoff": A["handoff"]["mean_done_s"], "shared": A["shared"]["mean_done_s"],
                                   "shared_nr": A["shared_nr"]["mean_done_s"]}
    # NOREPACK
    if all(k in A for k in ("shared", "shared_nr")):
        out["rules"]["NOREPACK"] = {"holds": A["shared_nr"]["mean_tpot_ms"] < A["shared"]["mean_tpot_ms"],
                                    "shared_nr_tpot_ms": A["shared_nr"]["mean_tpot_ms"], "shared_tpot_ms": A["shared"]["mean_tpot_ms"]}
    # ACCURACY
    out["rules"]["ACCURACY"] = {arm: {"holds": a["accuracy_holds"], "error_pct": a["error_pct"]} for arm, a in A.items()}
    # STATIC-CHECK: the servers' own timings for the static arm's users against the sweep's 1-slot figures
    rows = reg["rows"]
    st = users.get("static", {})
    chk = {}
    for u, ref in (("user1", rows["pi5_alone"]), ("user2", rows["pi4_alone"])):
        t = [x.get("timings") or {} for x in st.get(u, [])]
        pre = mean([x["prompt_ms"] / 1000 for x in t if "prompt_ms" in x])
        dec = mean([x["predicted_per_token_ms"] / 1000 for x in t if "predicted_per_token_ms" in x])
        ep, ed = 100 * (pre - ref["prefill_s"]) / ref["prefill_s"], 100 * (dec - ref["decode_s_per_token"]) / ref["decode_s_per_token"]
        chk[u] = {"prefill_s": round(pre, 3), "sweep_prefill_s": ref["prefill_s"], "prefill_diff_pct": round(ep, 2),
                  "decode_s_per_token": round(dec, 4), "sweep_decode_s": ref["decode_s_per_token"], "decode_diff_pct": round(ed, 2),
                  "holds": abs(ep) <= 5 and abs(ed) <= 5}
    out["rules"]["STATIC-CHECK"] = chk
    out["rules"]["STEADY"] = {arm: {"steady": a["steady"], "spread_pct": a["spread_pct"]} for arm, a in A.items()}
    json.dump(out, open(OUT, "w"), indent=1)
    for arm, a in A.items():
        print("%-9s n=%d mean done %6.2f s (predicted %s, error %+.1f %%, %s) makespan %6.2f  mean TTFT %6.2f  mean TPOT %7.1f ms  "
              "u1 ttft/done %.1f/%.1f  u2 ttft/done %.1f/%.1f  spread %.1f %%"
              % (arm, a["n"], a["mean_done_s"], a["predicted_mean_done_s"], a["error_pct"] or 0, "holds" if a["accuracy_holds"] else "MISSES",
                 a["makespan_s"], a["mean_ttft_s"], a["mean_tpot_ms"], a["user1"]["ttft_s"], a["user1"]["done_s"], a["user2"]["ttft_s"],
                 a["user2"]["done_s"], a["spread_pct"]))
    for k, v in out["rules"].items():
        print(k, json.dumps(v))
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    if "--out" in sys.argv:                                        # another result file (TWOUSER128, or a test of the parsing)
        OUT = sys.argv[sys.argv.index("--out") + 1]
    if "--reg" in sys.argv:                                        # another registration (TWOUSER128)
        REG = sys.argv[sys.argv.index("--reg") + 1]
    main(sys.argv[1])
