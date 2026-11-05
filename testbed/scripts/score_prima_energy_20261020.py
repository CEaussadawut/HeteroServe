#!/usr/bin/env python3
"""Score the engine-against-prima.cpp energy session (prima_energy_20261020.sh) by its registration's hypotheses.

    python3 testbed/scripts/score_prima_energy_20261020.py testbed/results/prima_energy_<stamp> \\
        --registration testbed/dataset/prima_energy_predictions_20261020.json

Per request (each arm-run's qr3 log, "req" records): client TPOT samples (qr3_tpot.samples), TTFT, and the gross energy over
its decode window -- from its first token's event to its last -- by shelly_strip.py summarise on the session's strip log,
with the session's idle-before window (full_eval.py's window_energy). Per arm, pooled over its runs: client TPOT P50 / P95,
the server's P50 where llama-server gives per-token timings (arm A), TTFT P50, and J per decoded token at the Pi 5 + Pi 4
outlets and at all four (the energy over the gaps, gen - 1 per request). Then PE-TPOT, PE-ENERGY, PE-FOLLOW and PE-REPL.
Writes score.json and score.txt into the session folder. Nothing is filled in.
--also DIR (2026-10-20, after registration): further session folders of the same registration, each with its own arms.tsv,
session.log idle window and strip.csv, pooled into the arms (the session of 06:16 was stopped at 07:44 by the operator, in
its fourth arm-run's cold gate; that arm-run ran again in a continuation folder). Without --also, it scores as registered.
"""
import argparse, datetime, hashlib, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402
import full_eval as FE                                      # noqa: E402



def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


REQ = re.compile(r"^\S+ req \d+ (\{.*\})\s*$")      # qr3_tpot.py's request record: "<time> req <i> {json}"


def reqs(log):
    out = []
    for ln in open(log, encoding="utf-8", errors="replace"):
        m = REQ.match(ln)
        if m:
            out.append(json.loads(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True)
    ap.add_argument("--also", nargs="*", default=[])
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    outlets = reg["outlets"]
    pool = [outlets["pi5"], outlets["pi4"]]
    runs, arms = [], {}
    for sess in [a.session] + a.also:
      text = open(os.path.join(sess, "session.log"), encoding="utf-8", errors="replace").read()
      t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
      idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
      for ln in open(os.path.join(sess, "arms.tsv")):
          i, arm, rc, lg = ln.rstrip("\n").split("\t")
          rr = reqs(os.path.join(ROOT, lg)) if lg else []
          cg_all, sg_all, ttft, j_pool, j_all, gaps = [], [], [], 0.0, 0.0, 0
          for x in rr:
              cg, sg, merged, tc, ts, ntok = Q.samples(x)
              ev = [e for e in x["events"] if e[1]]
              if not ev:
                  continue
              t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
              en = FE.window_energy(sess, t0, t1, idle, list(outlets.values()))
              g = max(1, ntok - 1)
              j_pool += sum(en[o][0] for o in pool); j_all += sum(v[0] for v in en.values()); gaps += g
              cg_all += cg; sg_all += sg; ttft.append(tc)
          run = {"session": os.path.relpath(os.path.abspath(sess), ROOT), "i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr), "client_tpot": Q.stats(cg_all),
                 "server_tpot": Q.stats(sg_all), "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
                 "j_per_token_pi5_pi4": round(j_pool / gaps, 4) if gaps else None, "j_per_token_all": round(j_all / gaps, 4) if gaps else None}
          runs.append(run)
          d = arms.setdefault(arm, {"cg": [], "sg": [], "ttft": [], "j_pool": 0.0, "j_all": 0.0, "gaps": 0, "requests": 0, "runs": []})
          d["cg"] += cg_all; d["sg"] += sg_all; d["ttft"] += [x for x in ttft if x]; d["j_pool"] += j_pool; d["j_all"] += j_all
          d["gaps"] += gaps; d["requests"] += len(rr); d["runs"].append(int(i))
    want = reg["reqs"] * 2
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want,
                    "client_tpot": Q.stats(d["cg"]), "server_tpot": Q.stats(d["sg"]),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_pi5_pi4": round(d["j_pool"] / d["gaps"], 4) if d["gaps"] else None,
                    "j_per_token_all": round(d["j_all"] / d["gaps"], 4) if d["gaps"] else None}
    H = reg["hypotheses"]
    ok = all(res.get(k, {}).get("complete") for k in ("A", "B"))
    A, B = res.get("A", {}), res.get("B", {})
    pa, pb = A.get("client_tpot", {}).get("p50"), B.get("client_tpot", {}).get("p50")
    ja, jb = A.get("j_per_token_pi5_pi4"), B.get("j_per_token_pi5_pi4")
    tr = pb / pa if pa and pb else None
    jr = jb / ja if ja and jb else None
    v = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    hyp = {"PE-TPOT": {"verdict": v(pa is not None and pb is not None and pa < pb), "A_p50": pa, "B_p50": pb},
           "PE-ENERGY": {"verdict": v(ja is not None and jb is not None and ja < jb), "A_j": ja, "B_j": jb},
           "PE-FOLLOW": {"verdict": v(tr is not None and jr is not None and abs(jr - tr) <= H["PE-FOLLOW"]["tol"]),
                         "tpot_ratio": round(tr, 4) if tr else None, "energy_ratio": round(jr, 4) if jr else None},
           "PE-REPL": {"verdict": v(tr is not None and abs(tr - H["PE-REPL"]["expected"]) <= H["PE-REPL"]["tol"]),
                       "tpot_ratio": round(tr, 4) if tr else None, "expected": H["PE-REPL"]["expected"]}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT), "also": [os.path.relpath(os.path.abspath(x), ROOT) for x in a.also], "registration": os.path.relpath(os.path.abspath(a.registration), ROOT),
         "registration_sha256": sha256(a.registration), "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "hypotheses": hyp}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["the engine (A) against prima.cpp (B), Qwen3-14B on Pi 5 + Pi 4; code changed since registration: %s" % (
        s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for r in runs:
        L.append("run %d %s rc %d: %d requests, client TPOT P50 %s ms, server P50 %s, TTFT P50 %s ms, J/token Pi 5 + Pi 4 %s, all %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["client_tpot"].get("p50"), r["server_tpot"].get("p50"), r["ttft_ms_p50"],
            r["j_per_token_pi5_pi4"], r["j_per_token_all"]))
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
