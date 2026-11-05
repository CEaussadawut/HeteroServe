#!/usr/bin/env python3
"""Score e2 (register_e2_20261102.py, run by spec4_session_20261023.sh): the laptop's refitted power rows with the session's
own laptop idle, on Qwen3-32B Q6_K.

    python3 testbed/scripts/score_e2_20261102.py testbed/results/e2_<stamp> \\
        --registration testbed/dataset/e2_predictions_20261102.json

Per arm-run (its qr3 log): the server's per-token timings (mean), and the gross energy at all four outlets over each request's
decode window with the session's idle-before window, plus every outlet's mean W over those windows. idle_s, the formula's one
measured input: the laptop outlet's mean W over the session's IDLE_BEFORE window. The predictions come from the registration's
own numbers (its formula and recorded rows), not from today's code. Hypotheses E2-REFIT, E2-POWER, E2-BETTER, E2-REPEAT as
registered; an arm is complete with every registered arm-run's requests. Writes score.json and score.txt into the session
folder. Nothing is filled in.
"""
import argparse, datetime, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402


def j_formula(T_ms, idle_s, fx):
    """The registered formula: J = T x (boards_w_sum + idle_s + active_minus_idle_w x f_laptop + pool_host_w) + per_token_j."""
    return T_ms / 1000.0 * (fx["boards_w_sum"] + idle_s + fx["active_minus_idle_w"] * fx["f_laptop"] + fx["pool_host_w"]) \
        + fx["per_token_j"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    outlets = reg["outlets"]
    text = open(os.path.join(a.session, "session.log"), encoding="utf-8", errors="replace").read()
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
    iw = FE.window_energy(a.session, idle[0], idle[1], idle, list(outlets.values()))
    idle_w = {n: iw[o][0] / (idle[1] - idle[0]) for n, o in outlets.items()}
    idle_s = idle_w["laptop"]
    runs, arms = [], {}
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = S4.reqs(path) if path else []
        sg_all, cg_all, j_all, gaps, win_s = [], [], 0.0, 0, 0.0
        e_out = {o: 0.0 for o in outlets.values()}
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if not ev:
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
            en = FE.window_energy(a.session, t0, t1, idle, list(outlets.values()))
            for o, v in en.items():
                e_out[o] += v[0]
            j_all += sum(v[0] for v in en.values()); gaps += max(1, ntok - 1); win_s += t1 - t0
            sg_all += sg; cg_all += cg
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr), "server_tpot": Q.stats(sg_all),
               "client_tpot": Q.stats(cg_all), "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "outlet_w": {n: round(e_out[o] / win_s, 3) for n, o in outlets.items()} if win_s else None,
               "laptop_battery": S4.battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])}
        runs.append(run)
        d = arms.setdefault(arm, {"sg": [], "j": 0.0, "gaps": 0, "win": 0.0, "e": {o: 0.0 for o in outlets.values()},
                                  "requests": 0, "runs": [], "run_means": []})
        d["sg"] += sg_all; d["j"] += j_all; d["gaps"] += gaps; d["win"] += win_s; d["requests"] += len(rr); d["runs"].append(int(i))
        for o in e_out:
            d["e"][o] += e_out[o]
        if run["server_tpot"].get("mean") is not None:
            d["run_means"].append(run["server_tpot"]["mean"])
    names = list(reg["arms"])
    want = {k: reg["reqs"] * reg["order"].count(k) for k in names}
    res = {}
    for arm in names:
        d = arms.get(arm)
        if not d:
            res[arm] = {"complete": False}
            continue
        p = reg["arms"][arm]["predicted"]
        fx = p["formula"]
        m = Q.stats(d["sg"]).get("mean")
        j = round(d["j"] / d["gaps"], 4) if d["gaps"] else None
        jp, jm = j_formula(p["decode_ms"], idle_s, fx), (j_formula(m, idle_s, fx) if m else None)
        rm = d["run_means"]
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want[arm],
                    "server_mean_ms": m, "time_err_pct": round(100 * (m / p["decode_ms"] - 1), 2) if m else None,
                    "j_per_token_all": j,
                    "outlet_w": {n: round(d["e"][o] / d["win"], 3) for n, o in outlets.items()} if d["win"] else None,
                    "predicted_j_t_pred": round(jp, 4), "predicted_j_t_meas": round(jm, 4) if jm else None,
                    "refit_err_pct": round(100 * (jp / j - 1), 2) if j else None,
                    "power_err_pct": round(100 * (jm / j - 1), 2) if (j and jm) else None,
                    "old_err_pct": round(100 * (p["j_per_token_all_old"] / j - 1), 2) if j else None,
                    "fixed_idle_err_pct": round(100 * (p["j_per_token_all_refit_fixed_idle"] / j - 1), 2) if j else None,
                    "run_means": rm, "run_spread_pct": round(100 * (max(rm) / min(rm) - 1), 2) if len(rm) >= 2 else None}
    H = reg["hypotheses"]
    ok = all(res[k].get("complete") for k in names)
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    ref = {k: res[k].get("refit_err_pct") for k in names}
    pw = {k: res[k].get("power_err_pct") for k in names}
    old = {k: res[k].get("old_err_pct") for k in names}
    hyp = {"E2-REFIT": {"verdict": vs(all(v is not None and abs(v) <= 100 * H["E2-REFIT"]["tol"] for v in ref.values())),
                        "refit_err_pct": ref},
           "E2-POWER": {"verdict": vs(all(v is not None and abs(v) <= 100 * H["E2-POWER"]["tol"] for v in pw.values())),
                        "power_err_pct": pw},
           "E2-BETTER": {"verdict": vs(all(ref[k] is not None and old[k] is not None and abs(ref[k]) < abs(old[k]) for k in names)),
                         "abs_err_pct": {k: [abs(ref[k]) if ref[k] is not None else None, abs(old[k]) if old[k] is not None else None]
                                         for k in names}},
           "E2-REPEAT": {"verdict": vs(all(res[k].get("run_spread_pct") is not None and res[k]["run_spread_pct"] <= 100 * H["E2-REPEAT"]["tol"]
                                           for k in names)),
                         "run_spread_pct": {k: res[k].get("run_spread_pct") for k in names}}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "idle_s_w": round(idle_s, 3), "idle_before_w": {n: round(v, 3) for n, v in idle_w.items()},
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER|!!!!!).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["e2: the laptop's refitted power with the session's own idle, on %s at ctx %d; code changed since registration: %s" % (
        reg["model"], reg["ctx"], s["code_changed_since_registration"] or "none"),
        "idle before: %s W (idle_s, the laptop: %.3f W)" % (s["idle_before_w"], idle_s), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for k in names:
        x = reg["arms"][k]
        L.append("arm %s %s head %s: predicted %s ms; measured %s" % (
            k, x["place_env"]["PLACE"], x["head"], x["predicted"]["decode_ms"],
            {y: z for y, z in res[k].items() if y in ("server_mean_ms", "time_err_pct", "j_per_token_all", "predicted_j_t_pred",
                                                      "predicted_j_t_meas", "refit_err_pct", "power_err_pct", "old_err_pct",
                                                      "fixed_idle_err_pct", "outlet_w", "run_means", "complete")}))
    for r in runs:
        L.append("run %d %s rc %d: %d requests, server mean %s ms, J/token %s, outlets %s W, battery %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["server_tpot"].get("mean"), r["j_per_token_all"], r["outlet_w"],
            r["laptop_battery"]))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
