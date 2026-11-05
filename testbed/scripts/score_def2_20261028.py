#!/usr/bin/env python3
"""Score def2 (register_def2_20261028.py, run by spec4_session_20261023.sh): the per-board multipliers on the blocks only
(node_mult "measured_blocks") against "measured", on three new head-last Qwen3.6-35B-A3B placements.

    python3 testbed/scripts/score_def2_20261028.py testbed/results/def2_<stamp> \\
        --registration testbed/dataset/def2_predictions_20261028.json

Per arm-run (its qr3 log): the server's per-token timings (mean, the registered measure), the client TPOT P50, TTFT, and the
gross energy at all four outlets over each request's decode window with the session's idle-before window; per arm, pooled
over its runs. Each arm's error against both predictions: measured / predicted - 1. Hypotheses D2-BLOCKS, D2-BETTER, D2-REPEAT
as registered. An arm is complete with every registered arm-run's requests. Writes score.json and score.txt into
the session folder. Nothing is filled in.
"""
import argparse, datetime, json, os, re, sys

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
        cg_all, sg_all, ttft, j_all, gaps = [], [], [], 0.0, 0
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if not ev:
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            en = FE.window_energy(a.session, x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"], idle, list(outlets.values()))
            j_all += sum(v[0] for v in en.values()); gaps += max(1, ntok - 1)
            cg_all += cg; sg_all += sg; ttft.append(tc)
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr), "server_tpot": Q.stats(sg_all),
               "client_tpot": Q.stats(cg_all), "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "laptop_battery": S4.battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])}
        runs.append(run)
        d = arms.setdefault(arm, {"sg": [], "cg": [], "ttft": [], "j": 0.0, "gaps": 0, "requests": 0, "runs": [], "run_means": []})
        d["sg"] += sg_all; d["cg"] += cg_all; d["ttft"] += [x for x in ttft if x]; d["j"] += j_all; d["gaps"] += gaps
        d["requests"] += len(rr); d["runs"].append(int(i))
        if run["server_tpot"].get("mean") is not None:
            d["run_means"].append(run["server_tpot"]["mean"])
    names = list(reg["arms"])
    want = {k: reg["reqs"] * reg.get("min_runs", {}).get(k, reg["order"].count(k)) for k in names}
    res = {}
    for arm in names:
        d = arms.get(arm)
        if not d:
            res[arm] = {"complete": False}
            continue
        m = Q.stats(d["sg"]).get("mean")
        p = reg["arms"][arm]["predicted"]
        rm = d["run_means"]
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want[arm],
                    "server_mean_ms": m, "server": Q.stats(d["sg"]), "client_p50_ms": Q.stats(d["cg"]).get("p50"),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j"] / d["gaps"], 4) if d["gaps"] else None,
                    "err_blocks_pct": round(100 * (m / p["blocks_decode_ms"] - 1), 2) if m else None,
                    "err_measured_pct": round(100 * (m / p["measured_decode_ms"] - 1), 2) if m else None,
                    "err_shipped_pct": round(100 * (m / p["shipped_decode_ms"] - 1), 2) if m else None,
                    "run_means": rm, "run_spread_pct": round(100 * (max(rm) / min(rm) - 1), 2) if len(rm) >= 2 else None}
    H = reg["hypotheses"]
    ok = all(res[k].get("complete") for k in names)
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    ev3 = {k: res[k].get("err_blocks_pct") for k in names}
    esh = {k: res[k].get("err_measured_pct") for k in names}
    have = all(v is not None for v in ev3.values()) and all(v is not None for v in esh.values())
    hyp = {"D2-BLOCKS": {"verdict": vs(have and all(abs(v) <= 100 * H["D2-BLOCKS"]["tol"] for v in ev3.values())), "err_blocks_pct": ev3},
           "D2-BETTER": {"verdict": vs(have and all(abs(ev3[k]) < abs(esh[k]) for k in names)),
                         "abs_err_pct": {k: [abs(ev3[k]) if ev3[k] is not None else None, abs(esh[k]) if esh[k] is not None else None]
                                         for k in names}},
           "D2-REPEAT": {"verdict": vs(all(res[k].get("run_spread_pct") is not None and res[k]["run_spread_pct"] <= 100 * H["D2-REPEAT"]["tol"]
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
    L = ["def2: the multipliers on the blocks only against measured on %s, three new head-last placements; code changed since registration: %s" % (
        reg["model"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for k in names:
        x = reg["arms"][k]
        L.append("arm %s %s head %s: predicted measured %s / blocks %s / shipped %s ms; measured %s" % (
            k, x["place_env"]["PLACE"], x["head"], x["predicted"]["measured_decode_ms"], x["predicted"]["blocks_decode_ms"],
            x["predicted"]["shipped_decode_ms"],
            {y: z for y, z in res[k].items() if y in ("server_mean_ms", "client_p50_ms", "ttft_ms_p50", "j_per_token_all",
                                                      "err_blocks_pct", "err_measured_pct", "err_shipped_pct", "run_means", "complete")}))
    for r in runs:
        L.append("run %d %s rc %d: %d requests, server mean %s ms, client P50 %s ms, battery %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["server_tpot"].get("mean"), r["client_tpot"].get("p50"), r["laptop_battery"]))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
