#!/usr/bin/env python3
"""The laptop's refitted power rows (power_refit_20261101.json) on the four-node sessions they were NOT fitted on -- a look
after v4a's E-REFIT failed (2026-11-01; post hoc, not registered). Laptop only; reads published scores, registrations and
strip logs, re-scores nothing.

    python3 testbed/scripts/energy_holdout_20261101.py

Per engine arm of each session since the evaluation (draft arms and prima.cpp's own arms left out: a draft adds laptop work
the busy split does not price, and prima.cpp is not this predictor's plan): its measured time per token T and gross J per
decoded token at all four outlets (the session's score), and three predictions at that measured T, with the busy fractions
priced as PLANNER_V3 prices the plan (energy_model.per_node with the RPC term and the per-model multipliers on the blocks):
  old       energy_model.power_table() (the laptop's 2026-10-06 row: idle 14.05 W, busy 32.12)
  refit     energy_model.energy_refit (the laptop at the evaluation's idle, 20.47 W)
  refit+idle   the same with the laptop's idle and busy rows shifted to the session's own idle-before window
                (the baseline moves with the laptop's state: 17.1 W in v4a's and in ph1/ph2's sessions)
Writes testbed/results/energy_holdout_20261101/holdout.json.
"""
import json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import full_eval as FE                                      # noqa: E402

OUTD = os.path.join(ROOT, "testbed", "results", "energy_holdout_20261101")
SESSIONS = ["sched2_20261022_032239", "prima4_20261023_030920", "prima4_20261023_111055", "order35_20261023_154835",
            "spec4_20261023_134113", "c35_20261024_025725", "c30_20261024_201522", "def1_20261028_015958",
            "def2_20261028_033530", "v4a_20261101_061538"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}


def measured_ms(a):
    for k in ("server_mean_ms",):
        if a.get(k):
            return a[k]
    if isinstance(a.get("server_tpot"), dict) and a["server_tpot"].get("mean"):
        return a["server_tpot"]["mean"]
    return a.get("mean_ms_per_token")


def engine_arm(arm):
    """(assign, head) of an engine arm from its registration entry, or None (a draft, prima.cpp, no placement)."""
    env = dict(arm.get("serve_env") or {}, **(arm.get("place_env") or {}))
    text = json.dumps(arm).lower()
    if (env.get("SERVE_DRAFT") not in (None, "", "none")) or "prima" in json.dumps({k: v for k, v in arm.items()
                                                                                   if k in ("what", "system", "runtime")}).lower():
        return None
    place = env.get("PLACE")
    if not place:
        return None
    assign = [(n, int(k)) for n, k in (x.split(":") for x in place.split(","))]
    return assign, env.get("PLACE_HEAD") or "last"


def session_idle(d):
    text = open(os.path.join(d, "session.log"), encoding="utf-8", errors="replace").read()
    import re
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    a, b = t("IDLE_BEFORE start"), t("IDLE_BEFORE end")
    en = FE.window_energy(d, a, b, (a, b), list(OUTLETS.values()))
    return {n: en[o][0] / (b - a) for n, o in OUTLETS.items()}


def main():
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    os.makedirs(OUTD, exist_ok=True)
    refit = EM.refit_table()
    lap0 = refit["laptop"]
    power = EM.power_table()
    rows = []
    for s in SESSIONS:
        d = os.path.join(ROOT, "testbed", "results", s)
        sc = json.load(open(os.path.join(d, "score.json")))
        reg = json.load(open(os.path.join(ROOT, sc["registration"])))
        idle = session_idle(d)
        for k, a in (sc.get("arms") or {}).items():
            ea = engine_arm(reg["arms"].get(k, {}))
            T, J = measured_ms(a), a.get("j_per_token_all")
            if not ea or not T or not J:
                continue
            assign, head = ea
            model = reg["arms"][k].get("model") or reg["model"]           # sched2: each arm its own quantisation
            r = P.plan_fixed(model, assign, head, int(reg["ctx"]), json.load(open(os.path.join(ROOT, reg["memfile"]))),
                             load_mode="none", **P.PLANNER_V3)
            members = [n for n, _ in assign]
            old = EM.energy(r["assign"], r["q"], r["C"], r["links"], power, "tg", head_at=r["head_at"], t_ms=T)
            old_all = old["gross_j"] + sum(power[n]["idle"] for n in OUTLETS if n not in members) * T / 1000
            new = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], refit, head_at=r["head_at"], t_ms=T, rpc_msg=P.RPC_MSG)
            sh = dict(refit, laptop=dict(lap0, idle_w=idle["laptop"], active_w=lap0["active_w"] + idle["laptop"] - lap0["idle_w"]))
            new2 = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], sh, head_at=r["head_at"], t_ms=T, rpc_msg=P.RPC_MSG)
            lw = (a.get("j_per_token_by_outlet") or a.get("j_per_token_by_board") or {}).get("laptop")
            rows.append({"session": s, "arm": k, "model": model, "plan": " + ".join("%s %d" % x for x in assign), "head": head,
                         "T_ms": round(T, 2), "measured_j": J, "laptop_idle_before_w": round(idle["laptop"], 2),
                         "laptop_busy_frac": round(new["busy_frac"].get("laptop", 0.0), 3),
                         "measured_laptop_j": lw,
                         "old_j": round(old_all, 4), "refit_j": round(new["all_j"], 4), "refit_idle_j": round(new2["all_j"], 4),
                         "laptop_w": {"refit": round(new["board_w"]["laptop"], 2), "refit_idle": round(new2["board_w"]["laptop"], 2)}})
    err = lambda r, k: 100 * (r[k] / r["measured_j"] - 1)
    print("%-24s %-4s %-14s %-38s %8s %7s | %6s %6s %6s  (%% error vs measured J/token, all outlets)" % (
        "session", "arm", "model", "plan", "T ms", "idle W", "old", "refit", "+idle"))
    for r in rows:
        print("%-24s %-4s %-14s %-38s %8.1f %7.2f | %+6.1f %+6.1f %+6.1f" % (r["session"], r["arm"], r["model"][:14], r["plan"][:38],
              r["T_ms"], r["laptop_idle_before_w"], err(r, "old_j"), err(r, "refit_j"), err(r, "refit_idle_j")))
    summ = {}
    for k in ("old_j", "refit_j", "refit_idle_j"):
        e = [err(r, k) for r in rows]
        summ[k] = {"n": len(e), "median_pct": round(statistics.median(e), 2), "median_abs_pct": round(statistics.median(abs(x) for x in e), 2),
                   "worst_pct": round(max(e, key=abs), 2), "within_10pct": sum(abs(x) <= 10 for x in e)}
        print("%-13s median %+6.1f %%, |error| median %5.1f %%, worst %+6.1f %%, within 10 %%: %d of %d" % (
            k, summ[k]["median_pct"], summ[k]["median_abs_pct"], summ[k]["worst_pct"], summ[k]["within_10pct"], summ[k]["n"]))
    json.dump({"what": "power_refit_20261101.json on the sessions it was not fitted on (post hoc, not registered)", "rows": rows,
               "summary": summ}, open(os.path.join(OUTD, "holdout.json"), "w"), indent=1)
    print("wrote %s" % os.path.relpath(os.path.join(OUTD, "holdout.json"), ROOT))


if __name__ == "__main__":
    main()
