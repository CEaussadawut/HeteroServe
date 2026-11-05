#!/usr/bin/env python3
"""Score q9 (register_q9_20261102.py, run by spec4_session_20261023.sh): the plan by decode time (D) against the plan by the
whole 512 + 64 request (R), on Qwen3-30B-A3B Q6_K.

    python3 testbed/scripts/score_q9_20261102.py testbed/results/q9_<stamp> --registration testbed/dataset/q9_predictions_20261102.json

Per request (its qr3 log): the time to the first token (the client's), the request's time from sending to its last token
(done_s), and the server's per-token timings. Per arm, pooled over its runs: the mean time to the first token, the mean
request time, the server mean TPOT; per run the mean request time (for the repeat). Gross energy at all four outlets over each
request, sent to last token, is reported, not tested. Hypotheses Q9-REQ, Q9-TRADE, Q9-ACC, Q9-REPEAT as registered; an arm
is complete with every registered arm-run's requests. Writes score.json and score.txt into the session folder.
"""
import argparse, datetime, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    outlets = reg["outlets"]
    text = open(os.path.join(a.session, "session.log"), encoding="utf-8", errors="replace").read()
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
    runs, arms = [], {}
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = S4.reqs(path) if path else []
        ttft, total, sg_all, j = [], [], [], []
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if not ev or not x.get("done_s"):
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            ttft.append(tc / 1000.0 if tc else None); total.append(x["done_s"]); sg_all += sg
            en = FE.window_energy(a.session, x["sent_wall"], x["sent_wall"] + x["done_s"], idle, list(outlets.values()))
            j.append(sum(v[0] for v in en.values()))
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(total),
               "ttft_s": [round(v, 2) if v else None for v in ttft], "request_s": [round(v, 2) for v in total],
               "request_mean_s": round(statistics.fmean(total), 3) if total else None, "server_tpot": Q.stats(sg_all),
               "request_j": [round(v, 1) for v in j]}
        runs.append(run)
        d = arms.setdefault(arm, {"ttft": [], "total": [], "sg": [], "j": [], "runs": [], "run_means": []})
        d["ttft"] += [v for v in ttft if v]; d["total"] += total; d["sg"] += sg_all; d["j"] += j; d["runs"].append(int(i))
        if total:
            d["run_means"].append(statistics.fmean(total))
    names = list(reg["arms"])
    want = {k: reg["reqs"] * reg["order"].count(k) for k in names}
    res = {}
    for arm in names:
        d = arms.get(arm)
        if not d:
            res[arm] = {"complete": False}
            continue
        p = reg["arms"][arm]["predicted"]
        mt, mr, ms = (statistics.fmean(d["ttft"]) if d["ttft"] else None), (statistics.fmean(d["total"]) if d["total"] else None), \
            Q.stats(d["sg"]).get("mean")
        rm = d["run_means"]
        res[arm] = {"runs": d["runs"], "requests": len(d["total"]), "complete": len(d["total"]) >= want[arm],
                    "ttft_mean_s": round(mt, 3) if mt else None, "request_mean_s": round(mr, 3) if mr else None,
                    "server_mean_ms": ms, "request_j_mean": round(statistics.fmean(d["j"]), 1) if d["j"] else None,
                    "ttft_err_pct": round(100 * (mt / p["ttft_s"] - 1), 2) if mt else None,
                    "request_err_pct": round(100 * (mr / p["request_s"] - 1), 2) if mr else None,
                    "tpot_err_pct": round(100 * (ms / p["decode_ms"] - 1), 2) if ms else None,
                    "run_means": [round(v, 3) for v in rm], "run_spread_pct": round(100 * (max(rm) / min(rm) - 1), 2) if len(rm) >= 2 else None}
    H = reg["hypotheses"]
    ok = all(res[k].get("complete") for k in names)
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    D, R = res.get("D", {}), res.get("R", {})
    gain = (1 - R["request_mean_s"] / D["request_mean_s"]) if (R.get("request_mean_s") and D.get("request_mean_s")) else None
    hyp = {"Q9-REQ": {"verdict": vs(gain is not None and gain >= H["Q9-REQ"]["min_gain"]),
                      "measured_gain_pct": round(100 * gain, 2) if gain is not None else None,
                      "predicted_gain_pct": round(100 * reg["predicted_request_gain"], 2)},
           "Q9-TRADE": {"verdict": vs(bool(R.get("ttft_mean_s") and D.get("ttft_mean_s") and R.get("server_mean_ms") and D.get("server_mean_ms"))
                                      and R["ttft_mean_s"] < D["ttft_mean_s"] and R["server_mean_ms"] > D["server_mean_ms"]),
                        "ttft_s": {k: res[k].get("ttft_mean_s") for k in names}, "tpot_ms": {k: res[k].get("server_mean_ms") for k in names}},
           "Q9-ACC": {"verdict": vs(all(res[k].get("ttft_err_pct") is not None and abs(res[k]["ttft_err_pct"]) <= 100 * H["Q9-ACC"]["tol_ttft"]
                                        and res[k].get("tpot_err_pct") is not None and abs(res[k]["tpot_err_pct"]) <= 100 * H["Q9-ACC"]["tol_tpot"]
                                        for k in names)),
                      "ttft_err_pct": {k: res[k].get("ttft_err_pct") for k in names}, "tpot_err_pct": {k: res[k].get("tpot_err_pct") for k in names}},
           "Q9-REPEAT": {"verdict": vs(all(res[k].get("run_spread_pct") is not None and res[k]["run_spread_pct"] <= 100 * H["Q9-REPEAT"]["tol"]
                                           for k in names)),
                         "run_spread_pct": {k: res[k].get("run_spread_pct") for k in names}}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER|!!!!!).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["q9: the request priced as prefill and decode, on %s at ctx %d (%d + %d); code changed since registration: %s" % (
        reg["model"], reg["ctx"], reg["n"], reg["gen"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for k in names:
        x = reg["arms"][k]
        L.append("arm %s %s head %s: predicted %s; measured %s" % (k, x["place_env"]["PLACE"], x["head"], x["predicted"],
                 {y: z for y, z in res[k].items() if y not in ("runs",)}))
    for r in runs:
        L.append("run %d %s rc %d: %d requests, TTFT %s s, request %s s, server mean %s ms, J %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["ttft_s"], r["request_s"], r["server_tpot"].get("mean"), r["request_j"]))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
