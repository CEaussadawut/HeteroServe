#!/usr/bin/env python3
"""Score the draft-length study registered as spec4d (register_spec4d_20261024.py, run by spec4_session_20261023.sh).

A copy of score_spec4c_20261024.py with one fix (2026-10-24): an arm-run that served no request (spec4c's arm-run 4,
refused by the memory gate) may have had no server of its own, so no server log is read for it.
spec4c's scorer read the next arm-run's log for it and pooled that run's draft acceptance into the refused arm
(S3 0.5024 where its own run gives 0.4515). Everything else is unchanged; the original's docstring follows.

Score the complete draft-length study (register_spec4c_20261024.py, run by spec4_session_20261023.sh).

    python3 testbed/scripts/score_spec4d_20261024.py testbed/results/spec4d_<stamp> \\
        --registration testbed/dataset/spec4d_predictions_20261024.json

As score_spec4b_20261023.py (2026-10-23), whose measurement code this keeps unchanged, for arms S1, S2, S3: per request the decode window over the tokens after the first event, TTFT,
and the gross energy at all four outlets over the decode window (full_eval.window_energy, the session's idle-before window);
pooled per arm. The draft's acceptance per arm-run from the host's own server log, ~/logs/llama-server_serve_<stamp>.log on
the laptop -- the first one started after the arm-run's qr3 log (qr3_tpot.py's copy of it does not reach a laptop host) --
read in this build's wording ("draft acceptance = r (a accepted / g generated)", one line per request; the first request of
each server, the warm-up, left out). Hypotheses SC-ORDER, SC-BEST, SC-ENERGY, SC-ACC, SC-TEXT (text compared within an
arm, between its two arm-runs). Writes score.json and score.txt into the session folder.
"""
import argparse, datetime, glob, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402
import score_prima4_20261021 as S4                          # noqa: E402

ACC = re.compile(r"task +(\d+) \| draft acceptance = [\d.]+ \( *(\d+) accepted / *(\d+) generated\), mean len = +([\d.]+)")


def server_log(qr3_log):
    m = re.search(r"_(\d{8}_\d{6})\.log$", qr3_log or "")
    if not m:
        return None
    later = sorted(p for p in glob.glob(os.path.expanduser("~/logs/llama-server_serve_*.log"))
                   if re.search(r"_(\d{8}_\d{6})\.log$", p).group(1) >= m.group(1))
    return later[0] if later else None


def acceptance(path):
    if not path:
        return None
    rows = [(int(t), int(a), int(g), float(ml)) for t, a, g, ml in ACC.findall(open(path, errors="replace").read())]
    rows = rows[1:]                                          # the warm-up request
    if not rows:
        return None
    a, g = sum(r[1] for r in rows), sum(r[2] for r in rows)
    return {"server_log": path, "requests": len(rows), "accepted": a, "drafted": g, "rate": round(a / g, 4) if g else None,
            "server_mean_len": [r[3] for r in rows]}      # the server's own tokens per check, per request


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
        win, tok, ttft, j_all, gaps = 0.0, 0, [], 0.0, 0
        for x in rr:
            ev = [e for e in x["events"] if e[1]]
            if len(ev) < 2:
                continue
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            w, k = ev[-1][0] - ev[0][0], ntok - len(ev[0][1])
            if k < 1:
                continue
            win += w; tok += k
            en = FE.window_energy(a.session, x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"], idle, list(outlets.values()))
            j_all += sum(v[0] for v in en.values()); gaps += max(1, ntok - 1); ttft.append(tc)
            texts.setdefault(str(x.get("i")), {}).setdefault(arm, []).append(S4.text16(x))
        acc = acceptance(server_log(lg)) if rr else None   # a run that served no request may have no server, so no log
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr),
               "mean_ms_per_token": round(1000 * win / tok, 2) if tok else None,
               "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None, "draft": acc,
               "laptop_battery": S4.battery(a.session, [(x["sent_wall"], x["sent_wall"] + x["done_s"]) for x in rr if x.get("done_s")])}
        runs.append(run)
        d = arms.setdefault(arm, {"win": 0.0, "tok": 0, "ttft": [], "j": 0.0, "gaps": 0, "requests": 0, "runs": [], "acc": [0, 0]})
        d["win"] += win; d["tok"] += tok; d["ttft"] += [x for x in ttft if x]; d["j"] += j_all; d["gaps"] += gaps
        d["requests"] += len(rr); d["runs"].append(int(i))
        if acc:
            d["acc"][0] += acc["accepted"]; d["acc"][1] += acc["drafted"]
    names = list(reg["arms"])
    want = {k: reg["reqs"] * reg.get("min_runs", {}).get(k, reg["order"].count(k)) for k in names}
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want.get(arm, 0),
                    "mean_ms_per_token": round(1000 * d["win"] / d["tok"], 2) if d["tok"] else None,
                    "predicted": reg["arms"][arm]["predicted"]["mean_ms_per_token"],
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j"] / d["gaps"], 4) if d["gaps"] else None,
                    "draft_acceptance_rate": round(d["acc"][0] / d["acc"][1], 4) if d["acc"][1] else None}
    ok = all(res.get(k, {}).get("complete") for k in names)
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    mt = {k: res.get(k, {}).get("mean_ms_per_token") for k in names}
    jt = {k: res.get(k, {}).get("j_per_token_all") for k in names}
    have = all(v is not None for v in mt.values())
    fastest = min(mt, key=lambda k: mt[k]) if have else None
    acc_ok = {k: (abs(mt[k] / res[k]["predicted"] - 1) <= reg["hypotheses"]["SC-ACC"]["tol"]) if mt.get(k) else None for k in names}
    within = {arm: (len(set(texts.get("0", {}).get(arm, []))) == 1 and len(texts.get("0", {}).get(arm, [])) >= 2)
              for arm in names}                              # request 0's text, the same in both arm-runs of an arm
    across = {k: len({x for arm in d.values() for x in arm}) == 1 and len(d) == len(names) for k, d in texts.items()}
    hyp = {"SC-ORDER": {"verdict": vs(have and mt["S1"] < mt["S2"] < mt["S3"]), "mean_ms_per_token": mt},
           "SC-BEST": {"verdict": vs(have and fastest == reg["hypotheses"]["SC-BEST"]["predicted"]), "fastest": fastest,
                       "predicted": reg["hypotheses"]["SC-BEST"]["predicted"]},
           "SC-ENERGY": {"verdict": vs(have and all(jt[k] is not None for k in names) and min(jt, key=lambda k: jt[k]) == fastest),
                         "j_per_token": jt},
           "SC-ACC": {"verdict": vs(have and all(acc_ok.values())),
                      "err_pct": {k: round(100 * (mt[k] / res[k]["predicted"] - 1), 1) if mt.get(k) else None for k in names}},
           "SC-TEXT": {"verdict": vs(all(within.values())), "within_arm": within,
                       "across_arms_by_request (observation)": across, "texts": texts}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": S4.sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp,
         "session_notes": re.findall(r"^((?:DEADLINE|NOT ON THE CHARGER|NOT QUIET|FATAL|STOPPED_BY_USER|!!!!!).*)$", text, re.M)}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["draft length on v2's plan, complete (%s %s), ctx %d; code changed since registration: %s" % (
        reg["model"], reg["arms"]["S1"]["place_env"]["PLACE"], reg["ctx"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x not in ("verdict", "texts")}))
    L.append("")
    for k in names:
        L.append("arm %s: %s" % (k, res.get(k)))
    for r in runs:
        L.append("run %d %s rc %d: %d requests, mean %s ms/token, TTFT P50 %s ms, J/token all %s, draft %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["mean_ms_per_token"], r["ttft_ms_p50"], r["j_per_token_all"],
            {x: y for x, y in (r["draft"] or {}).items() if x != "server_log"}))
    L += ["session: %s" % n for n in s["session_notes"]]
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
