#!/usr/bin/env python3
"""Score speculative decoding on scheduler v2's plan (spec4_session_20261023.sh) by its registration's hypotheses.

    python3 testbed/scripts/score_spec4_20261023.py testbed/results/spec4_<stamp> \\
        --registration testbed/dataset/spec4_predictions_20261023.json

Per request (each arm-run's qr3 log): the decode window, from the first token's arrival to the last's, over the tokens that
came after the first event -- with a draft, accepted tokens arrive several to an event, which qr3_tpot.samples() leaves out
of its client gaps, so the window is the measure that counts every token; TTFT; the gross energy at all four outlets over the
decode window with the session's idle-before window (full_eval.window_energy), per gen - 1 tokens. Per arm, pooled over its
runs: the mean time per token (windows over tokens), the client-gap P50 (single-token events only, for reference), the
server's mean, TTFT P50, J per token. The draft's acceptance from each run's copied server log where it prints it. Then the
registered hypotheses (SP-FASTER, SP-PRED, SP-ENERGY, SP-ACC, SP-TEXT). An arm is complete with min_runs arm-runs' worth of
requests where the registration has min_runs, else every registered arm-run's. Each arm-run carries the laptop battery's
state over its requests (battery.tsv). Writes score.json and score.txt into the session folder. Nothing is filled in.
"""
import argparse, datetime, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402  (battery(), text16(), reqs(), sha256())


def acceptance(server_log):
    """(accepted, drafted) summed from the server log's per-check or per-request lines; None when it prints neither."""
    if not server_log or not os.path.exists(server_log):
        return None
    t = open(server_log, errors="replace").read()
    per_req = re.findall(r"draft acceptance rate = [\d.]+ \(\s*(\d+) accepted /\s*(\d+) generated\)", t)
    if per_req:
        return {"accepted": sum(int(a) for a, _ in per_req), "drafted": sum(int(b) for _, b in per_req), "source": "per request"}
    per_check = re.findall(r"accepted (\d+)/(\d+) draft tokens", t)
    if per_check:
        return {"accepted": sum(int(a) for a, _ in per_check), "drafted": sum(int(b) for _, b in per_check), "source": "per check"}
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
    runs, arms, texts = [], {}, {}
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        path = os.path.join(ROOT, lg) if lg else None
        rr = S4.reqs(path) if path else []
        win, tok, cg_all, sg_all, ttft, j_all, gaps, by = 0.0, 0, [], [], [], 0.0, 0, {}
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if len(ev) < 2:
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            w, k = ev[-1][0] - ev[0][0], ntok - len(ev[0][1])
            if k < 1:
                continue
            win += w; tok += k
            t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
            en = FE.window_energy(a.session, t0, t1, idle, list(outlets.values()))
            g = max(1, ntok - 1)
            j_all += sum(v[0] for v in en.values()); gaps += g
            for node, o in outlets.items():
                by[node] = by.get(node, 0.0) + en[o][0]
            cg_all += cg; sg_all += sg; ttft.append(tc)
            texts.setdefault(x.get("i"), {}).setdefault(arm, []).append(S4.text16(x))
        bat = S4.battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])
        acc = acceptance(path[:-4] + "_server.log") if path else None
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr),
               "mean_ms_per_token": round(1000 * win / tok, 2) if tok else None, "tokens_after_first": tok,
               "client_gap_p50_single_token_events": Q.stats(cg_all).get("p50"), "server": Q.stats(sg_all),
               "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "j_per_token_by_board": {n: round(v / gaps, 4) for n, v in by.items()} if gaps else None,
               "draft_acceptance": acc, "laptop_battery": bat}
        runs.append(run)
        d = arms.setdefault(arm, {"win": 0.0, "tok": 0, "cg": [], "sg": [], "ttft": [], "j": 0.0, "gaps": 0, "requests": 0,
                                  "runs": [], "acc": [0, 0]})
        d["win"] += win; d["tok"] += tok; d["cg"] += cg_all; d["sg"] += sg_all; d["ttft"] += [x for x in ttft if x]
        d["j"] += j_all; d["gaps"] += gaps; d["requests"] += len(rr); d["runs"].append(int(i))
        if acc:
            d["acc"][0] += acc["accepted"]; d["acc"][1] += acc["drafted"]
    want = {k: reg["reqs"] * reg.get("min_runs", {}).get(k, reg["order"].count(k)) for k in ("V", "S")}
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want.get(arm, 0),
                    "mean_ms_per_token": round(1000 * d["win"] / d["tok"], 2) if d["tok"] else None,
                    "client_gap_p50_single_token_events": Q.stats(d["cg"]).get("p50"), "server": Q.stats(d["sg"]),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j"] / d["gaps"], 4) if d["gaps"] else None,
                    "draft_acceptance_rate": round(d["acc"][0] / d["acc"][1], 4) if d["acc"][1] else None}
    H = reg["hypotheses"]
    V, S = res.get("V", {}), res.get("S", {})
    ok = bool(V.get("complete") and S.get("complete"))
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    mv, ms = V.get("mean_ms_per_token"), S.get("mean_ms_per_token")
    sp = mv / ms if mv and ms else None
    jv, js = V.get("j_per_token_all"), S.get("j_per_token_all")
    vmean, vpred = V.get("server", {}).get("mean"), reg["arms"]["V"]["predicted"]["decode_ms"]
    same = {str(k): d for k, d in sorted(texts.items(), key=lambda kv: kv[0])}
    match = {k: (len({x for arm in d.values() for x in arm}) == 1) for k, d in same.items() if "V" in d and "S" in d}
    hyp = {"SP-FASTER": {"verdict": vs(sp is not None and sp > 1), "V_mean_ms": mv, "S_mean_ms": ms,
                         "speedup": round(sp, 4) if sp else None},
           "SP-PRED": {"verdict": vs(sp is not None and abs(sp / H["SP-PRED"]["predicted"] - 1) <= H["SP-PRED"]["tol"]),
                       "speedup": round(sp, 4) if sp else None, "predicted": H["SP-PRED"]["predicted"]},
           "SP-ENERGY": {"verdict": vs(jv is not None and js is not None and js < jv), "V_j": jv, "S_j": js,
                         "S_over_V": round(js / jv, 4) if jv and js else None},
           "SP-ACC": {"verdict": vs(vmean is not None and abs(vmean / vpred - 1) <= H["SP-ACC"]["tol"]), "V_server_mean": vmean,
                      "V_predicted": vpred, "err_pct": round(100 * (vmean / vpred - 1), 1) if vmean else None},
           "SP-TEXT": {"verdict": vs(match.get("0") is True), "match_by_request": match, "texts": same}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER|!!!!!).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["speculative decoding on v2's plan (%s %s), ctx %d; code changed since registration: %s" % (
        reg["model"], reg["arms"]["V"]["place_env"]["PLACE"], reg["ctx"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x not in ("verdict", "texts")}))
    L.append("")
    for k, x in res.items():
        L.append("arm %s: %s" % (k, x))
    for r in runs:
        L.append("run %d %s rc %d: %d requests, mean %s ms/token, TTFT P50 %s ms, J/token all %s, draft %s, battery %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["mean_ms_per_token"], r["ttft_ms_p50"], r["j_per_token_all"],
            r["draft_acceptance"], r["laptop_battery"]))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
