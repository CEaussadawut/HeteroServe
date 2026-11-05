#!/usr/bin/env python3
"""Score v4a (register_v4a_20261101.py, run by spec4_session_20261023.sh): PLANNER_V4's hybrid memory on Qwen3.6-35B-A3B at
16,384 cells (K3 V3's pick, K4 V4's) and the laptop's refitted power rows.

    python3 testbed/scripts/score_v4a_20261101.py testbed/results/v4a_<stamp> \\
        --registration testbed/dataset/v4a_predictions_20261101.json

Per arm-run (its qr3 log): the server's per-token timings (mean, the registered measure), the client TPOT P50, TTFT, the gross
energy at all four outlets over each request's decode window with the session's idle-before window, the per-outlet mean W over
those windows, and the memory each board held (serve_pool.sh's own log, named in the qr3 log's "served" line: MemAvailable at
the gate less at READY, against the charge it printed). Per arm, pooled over its runs. Hypotheses V4-KV, V4-ACC, V4-MEM,
E-REFIT, E-OLD, V4-REPEAT as registered; an arm is complete with every registered arm-run's requests. Writes score.json and
score.txt into the session folder. Nothing is filled in.
"""
import argparse, datetime, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402

MEM = re.compile(r"^mem     (\w+): MemAvailable (\d+) MiB at READY, (\d+) at the gate: the serve holds (-?\d+) MiB, charged (\d+) MiB", re.M)


def served_log(qr3_log):
    for line in open(qr3_log, encoding="utf-8", errors="replace"):
        if " served {" in line:
            return json.loads(line.split(" served ", 1)[1]).get("serve_log")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    outlets = reg["outlets"]
    text = open(os.path.join(a.session, "session.log"), encoding="utf-8", errors="replace").read()
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
    idle_w = {n: v[0] / (idle[1] - idle[0]) for n, v in FE.window_energy(a.session, idle[0], idle[1], idle, list(outlets.values())).items()}
    runs, arms = [], {}
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = S4.reqs(path) if path else []
        cg_all, sg_all, ttft, j_all, gaps, win_s = [], [], [], 0.0, 0, 0.0
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
            cg_all += cg; sg_all += sg; ttft.append(tc)
        sl = served_log(path) if path else None
        mem = {m.group(1): {"held_mib": int(m.group(4)), "charged_mib": int(m.group(5))}
               for m in MEM.finditer(open(os.path.join(ROOT, sl), encoding="utf-8", errors="replace").read())} if sl else {}
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "serve_log": sl, "requests": len(rr),
               "server_tpot": Q.stats(sg_all), "client_tpot": Q.stats(cg_all),
               "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "outlet_w": {n: round(e_out[o] / win_s, 3) for n, o in outlets.items()} if win_s else None, "memory": mem,
               "laptop_battery": S4.battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])}
        runs.append(run)
        d = arms.setdefault(arm, {"sg": [], "cg": [], "ttft": [], "j": 0.0, "gaps": 0, "win": 0.0, "e": {o: 0.0 for o in outlets.values()},
                                  "requests": 0, "runs": [], "run_means": [], "memory": []})
        d["sg"] += sg_all; d["cg"] += cg_all; d["ttft"] += [x for x in ttft if x]; d["j"] += j_all; d["gaps"] += gaps
        d["win"] += win_s; d["requests"] += len(rr); d["runs"].append(int(i)); d["memory"].append(mem)
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
        m = Q.stats(d["sg"]).get("mean")
        p = reg["arms"][arm]["predicted"]
        j = round(d["j"] / d["gaps"], 4) if d["gaps"] else None
        rm = d["run_means"]
        # each board's worst (largest) held - charged over the arm's serves: positive = held more than charged
        over = {}
        for mm in d["memory"]:
            for n, v in mm.items():
                over[n] = max(over.get(n, -10 ** 9), v["held_mib"] - v["charged_mib"])
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want[arm],
                    "server_mean_ms": m, "server": Q.stats(d["sg"]), "client_p50_ms": Q.stats(d["cg"]).get("p50"),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None, "j_per_token_all": j,
                    "outlet_w": {n: round(d["e"][o] / d["win"], 3) for n, o in outlets.items()} if d["win"] else None,
                    "err_pct": round(100 * (m / p["decode_ms"] - 1), 2) if m else None,
                    "refit_err_pct": round(100 * (p["j_per_token_all_refit"] / j - 1), 2) if j else None,
                    "old_err_pct": round(100 * (p["j_per_token_all_old"] / j - 1), 2) if j else None,
                    "held_minus_charged_worst_mib": over, "run_means": rm,
                    "run_spread_pct": round(100 * (max(rm) / min(rm) - 1), 2) if len(rm) >= 2 else None}
    H = reg["hypotheses"]
    ok = all(res[k].get("complete") for k in names)
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    k3, k4 = res.get("K3", {}).get("server_mean_ms"), res.get("K4", {}).get("server_mean_ms")
    gain = (1 - k4 / k3) if (k3 and k4) else None
    err = {k: res[k].get("err_pct") for k in names}
    ref = {k: res[k].get("refit_err_pct") for k in names}
    old = {k: res[k].get("old_err_pct") for k in names}
    over4 = res.get("K4", {}).get("held_minus_charged_worst_mib") or {}
    hyp = {"V4-KV": {"verdict": vs(gain is not None and gain >= H["V4-KV"]["min_gain"]),
                     "measured_gain_pct": round(100 * gain, 2) if gain is not None else None,
                     "predicted_gain_pct": round(100 * reg["predicted_gain_k4_vs_k3"], 2)},
           "V4-ACC": {"verdict": vs(all(v is not None and abs(v) <= 100 * H["V4-ACC"]["tol"] for v in err.values())), "err_pct": err},
           "V4-MEM": {"verdict": vs(bool(over4) and len(over4) == len(reg["nodes"])
                                    and all(v <= H["V4-MEM"]["tol_mib"] for v in over4.values())),
                      "k4_held_minus_charged_worst_mib": over4},
           "E-REFIT": {"verdict": vs(all(v is not None and abs(v) <= 100 * H["E-REFIT"]["tol"] for v in ref.values())),
                       "refit_err_pct": ref},
           "E-OLD": {"verdict": vs(all(v is not None and v <= -100 * H["E-OLD"]["min_low"] for v in old.values())), "old_err_pct": old},
           "V4-REPEAT": {"verdict": vs(all(res[k].get("run_spread_pct") is not None and res[k]["run_spread_pct"] <= 100 * H["V4-REPEAT"]["tol"]
                                           for k in names)),
                         "run_spread_pct": {k: res[k].get("run_spread_pct") for k in names}}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "idle_before_w": {n: round(idle_w[o], 3) for n, o in outlets.items()},
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER|!!!!!).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["v4a: PLANNER_V4's hybrid memory and the laptop's refitted power on %s at ctx %d; code changed since registration: %s" % (
        reg["model"], reg["ctx"], s["code_changed_since_registration"] or "none"),
        "idle before: %s W" % s["idle_before_w"], ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for k in names:
        x = reg["arms"][k]
        L.append("arm %s %s head %s: predicted %s ms, J/token refit %s / old %s; measured %s" % (
            k, x["place_env"]["PLACE"], x["head"], x["predicted"]["decode_ms"], x["predicted"]["j_per_token_all_refit"],
            x["predicted"]["j_per_token_all_old"],
            {y: z for y, z in res[k].items() if y in ("server_mean_ms", "client_p50_ms", "ttft_ms_p50", "j_per_token_all", "err_pct",
                                                      "refit_err_pct", "old_err_pct", "outlet_w", "run_means", "complete")}))
    for r in runs:
        L.append("run %d %s rc %d: %d requests, server mean %s ms, J/token %s, outlets %s W, memory %s, battery %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["server_tpot"].get("mean"), r["j_per_token_all"], r["outlet_w"],
            r["memory"], r["laptop_battery"]))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
