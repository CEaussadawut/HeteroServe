#!/usr/bin/env python3
"""Score the order check on Qwen3.6-35B-A3B (register_order35_20261023.py, run by spec4_session_20261023.sh).

    python3 testbed/scripts/score_order35_20261023.py testbed/results/order35_<stamp> \\
        --registration testbed/dataset/order35_predictions_20261023.json

As score_prima4_20261021.py, for arms P (the engine's pick) and F (the predictor's best order): per request the client TPOT
samples (qr3_tpot.samples), the server's per-token timings, TTFT, the gross energy at all four outlets over its decode window
with the session's idle-before window; pooled per arm. Hypotheses OR-FASTER, OR-ACC, OR-ENERGY, OR-TEXT. Each arm-run
carries the laptop battery's state over its requests. Writes score.json and score.txt into the session folder.
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
    runs, arms, texts = [], {}, {}
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = S4.reqs(path) if path else []
        cg_all, sg_all, ttft, j_all, gaps, by = [], [], [], 0.0, 0, {}
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if not ev:
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
            en = FE.window_energy(a.session, t0, t1, idle, list(outlets.values()))
            g = max(1, ntok - 1)
            j_all += sum(v[0] for v in en.values()); gaps += g
            for node, o in outlets.items():
                by[node] = by.get(node, 0.0) + en[o][0]
            cg_all += cg; sg_all += sg; ttft.append(tc)
            texts.setdefault(x.get("i"), {}).setdefault(arm, []).append(S4.text16(x))
        bat = S4.battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr), "client_tpot": Q.stats(cg_all),
               "server_tpot": Q.stats(sg_all), "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "j_per_token_by_board": {n: round(v / gaps, 4) for n, v in by.items()} if gaps else None, "laptop_battery": bat}
        runs.append(run)
        d = arms.setdefault(arm, {"cg": [], "sg": [], "ttft": [], "j": 0.0, "gaps": 0, "requests": 0, "runs": []})
        d["cg"] += cg_all; d["sg"] += sg_all; d["ttft"] += [x for x in ttft if x]; d["j"] += j_all; d["gaps"] += gaps
        d["requests"] += len(rr); d["runs"].append(int(i))
    want = {k: reg["reqs"] * reg.get("min_runs", {}).get(k, reg["order"].count(k)) for k in ("P", "F")}
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want.get(arm, 0),
                    "client_tpot": Q.stats(d["cg"]), "server_tpot": Q.stats(d["sg"]),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j"] / d["gaps"], 4) if d["gaps"] else None}
    H = reg["hypotheses"]
    Pr, Fr = res.get("P", {}), res.get("F", {})
    ok = bool(Pr.get("complete") and Fr.get("complete"))
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    pp, pf = Pr.get("client_tpot", {}).get("p50"), Fr.get("client_tpot", {}).get("p50")
    jp, jf = Pr.get("j_per_token_all"), Fr.get("j_per_token_all")
    cheap = "F" if reg["arms"]["F"]["predicted"]["decode_gross_j_all"] < reg["arms"]["P"]["predicted"]["decode_gross_j_all"] else "P"
    jj = {"P": jp, "F": jf}
    acc = {}
    for k, x in (("P", Pr), ("F", Fr)):
        m, p = x.get("server_tpot", {}).get("mean"), reg["arms"][k]["predicted"]["decode_ms"]
        acc[k] = {"server_mean": m, "predicted": p, "err_pct": round(100 * (m / p - 1), 1) if m else None}
    same = {str(k): d for k, d in sorted(texts.items(), key=lambda kv: kv[0])}
    match = {k: (len({x for arm in d.values() for x in arm}) == 1) for k, d in same.items() if "P" in d and "F" in d}
    other = "P" if cheap == "F" else "F"
    hyp = {"OR-FASTER": {"verdict": vs(pp is not None and pf is not None and pf < pp), "P_p50": pp, "F_p50": pf,
                         "P_over_F": round(pp / pf, 4) if pp and pf else None, "predicted": H["OR-FASTER"]["predicted"]},
           "OR-ACC": {"verdict": vs(all(x["err_pct"] is not None and abs(x["err_pct"]) <= 100 * H["OR-ACC"]["tol"] for x in acc.values())),
                      **acc},
           "OR-ENERGY": {"verdict": vs(jj[cheap] is not None and jj[other] is not None and jj[cheap] < jj[other]),
                         "predicted_lower": cheap, "P_j": jp, "F_j": jf},
           "OR-TEXT": {"verdict": vs(match.get("0") is True), "match_by_request": match, "texts": same}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER|!!!!!).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["the order check on %s: P %s against F %s, ctx %d; code changed since registration: %s" % (
        reg["model"], reg["arms"]["P"]["place_env"]["PLACE"], reg["arms"]["F"]["place_env"]["PLACE"], reg["ctx"],
        s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x not in ("verdict", "texts")}))
    L.append("")
    for r in runs:
        L.append("run %d %s rc %d: %d requests, client TPOT P50 %s ms, server mean %s, TTFT P50 %s ms, J/token all %s, battery %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["client_tpot"].get("p50"), r["server_tpot"].get("mean"), r["ttft_ms_p50"],
            r["j_per_token_all"], r["laptop_battery"]))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
