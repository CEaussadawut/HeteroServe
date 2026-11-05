#!/usr/bin/env python3
"""Score scheduler v2 against v1 (sched2_session_20261022.sh + sched2_quality_20261022.sh) by the registration's hypotheses.

    python3 testbed/scripts/score_sched2_20261022.py testbed/results/sched2_<stamp> \\
        --registration testbed/dataset/sched2_predictions_20261022.json --quality testbed/results/sched2_quality_<stamp>.log

Speed and energy as score_prima4_20261021.py (per request: client TPOT samples, the server's per-token timings, TTFT, and the
gross energy at all four outlets over its decode window with the session's idle-before window; per arm pooled over its runs).
Quality from llama-perplexity's own KL report in the quality log, per variant section: Mean PPL(Q)/PPL(base), Mean KLD, Same top p.
Writes score.json and score.txt into the session folder. Nothing is filled in; a missing figure scores INCOMPLETE.
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


def reqs(log):
    pat = re.compile(r"^\S+ req \d+ (\{.*\})\s*$")
    return [json.loads(m.group(1)) for m in (pat.match(ln) for ln in open(log, encoding="utf-8", errors="replace")) if m]


def quality(log):
    """{variant: {"ppl_ratio", "kld", "same_top_pct", "ppl_q", "ppl_base"}} from the sections sched2_quality_20261022.sh writes."""
    text = open(log, encoding="utf-8", errors="replace").read()
    out = {}
    parts = re.split(r"^===== \S+ (q6_k|q4_k_m) against the Q8_0 logits$", text, flags=re.M)
    for name, body in zip(parts[1::2], parts[2::2]):
        g = lambda pat: (lambda m: float(m.group(1)) if m else None)(re.search(pat, body, re.M))
        out[name] = {"ppl_q": g(r"Mean PPL\(Q\)\s*:\s*([\d.]+)"), "ppl_base": g(r"Mean PPL\(base\)\s*:\s*([\d.]+)"),
                     "ppl_ratio": g(r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([\d.]+)"), "kld": g(r"Mean\s+KLD:\s*([\d.]+)"),
                     "same_top_pct": g(r"Same top p:\s*([\d.]+)")}
    m = re.search(r"^model\s+q8_0: (\S+) sha256 (\w+)", text, re.M)
    return out, {"q8_0_file": m.group(1) if m else None, "done": "QUALITY DONE" in text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session"); ap.add_argument("--registration", required=True); ap.add_argument("--quality", required=True)
    a = ap.parse_args()
    reg = json.load(open(a.registration))
    outlets = reg["outlets"]
    text = open(os.path.join(a.session, "session.log"), encoding="utf-8", errors="replace").read()
    t = lambda k: float(re.search(r"^%s (\d+\.\d+)$" % k, text, re.M).group(1))
    idle = (t("IDLE_BEFORE start"), t("IDLE_BEFORE end"))
    runs, arms = [], {}
    for ln in open(os.path.join(a.session, "arms.tsv")):
        i, arm, rc, lg = ln.rstrip("\n").split("\t")
        rr = reqs(os.path.join(ROOT, lg)) if lg else []
        cg_all, sg_all, ttft, j_all, gaps, by = [], [], [], 0.0, 0, {}
        for x in rr:
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            ev = [e for e in x["events"] if e[1]]
            if not ev:
                continue
            t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
            en = FE.window_energy(a.session, t0, t1, idle, list(outlets.values()))
            g = max(1, ntok - 1)
            j_all += sum(v[0] for v in en.values()); gaps += g
            for node, o in outlets.items():
                by[node] = by.get(node, 0.0) + en[o][0]
            cg_all += cg; sg_all += sg; ttft.append(tc)
        run = {"i": int(i), "arm": arm, "rc": int(rc), "log": lg, "requests": len(rr), "client_tpot": Q.stats(cg_all),
               "server_tpot": Q.stats(sg_all), "ttft_ms_p50": round(Q.pct([x for x in ttft if x], 50), 1) if ttft else None,
               "j_per_token_all": round(j_all / gaps, 4) if gaps else None,
               "j_per_token_by_board": {n: round(v / gaps, 4) for n, v in by.items()} if gaps else None}
        runs.append(run)
        d = arms.setdefault(arm, {"cg": [], "sg": [], "ttft": [], "j": 0.0, "gaps": 0, "requests": 0, "runs": [], "by": {}})
        d["cg"] += cg_all; d["sg"] += sg_all; d["ttft"] += [x for x in ttft if x]; d["j"] += j_all; d["gaps"] += gaps
        d["requests"] += len(rr); d["runs"].append(int(i))
        for n, v in by.items():
            d["by"][n] = d["by"].get(n, 0.0) + v
    want = reg["reqs"] * 2
    res = {}
    for arm, d in arms.items():
        res[arm] = {"runs": d["runs"], "requests": d["requests"], "complete": d["requests"] >= want,
                    "client_tpot": Q.stats(d["cg"]), "server_tpot": Q.stats(d["sg"]),
                    "ttft_ms_p50": round(Q.pct(d["ttft"], 50), 1) if d["ttft"] else None,
                    "j_per_token_all": round(d["j"] / d["gaps"], 4) if d["gaps"] else None,
                    "j_per_token_by_board": {n: round(v / d["gaps"], 4) for n, v in d["by"].items()} if d["gaps"] else None}
    H = reg["hypotheses"]
    A, V = res.get("A", {}), res.get("V", {})
    ok = bool(A.get("complete") and V.get("complete"))
    vs = lambda c, k=ok: ("HOLDS" if c else "FAILS") if k else "INCOMPLETE"
    pa, pv = A.get("client_tpot", {}).get("p50"), V.get("client_tpot", {}).get("p50")
    sp = pa / pv if pa and pv else None
    acc = {}
    for k, x in (("A", A), ("V", V)):
        m, p = x.get("server_tpot", {}).get("mean"), reg["arms"][k]["predicted"]["decode_ms"]
        acc[k] = {"server_mean": m, "predicted": p, "err_pct": round(100 * (m / p - 1), 1) if m else None}
    qa, qmeta = quality(a.quality)
    q6, q4 = qa.get("q6_k", {}), qa.get("q4_k_m", {})
    qok = bool(qmeta["done"] and q6.get("kld") is not None and q4.get("kld") is not None)
    hyp = {"Q6-KLD": {"verdict": vs(q6.get("kld") is not None and q6["kld"] <= H["Q6-KLD"]["max"], qok), "q6_k": q6.get("kld")},
           "Q6-TOP": {"verdict": vs(q6.get("same_top_pct") is not None and q6["same_top_pct"] >= H["Q6-TOP"]["min_pct"], qok),
                      "q6_k": q6.get("same_top_pct")},
           "Q6-PPL": {"verdict": vs(q6.get("ppl_ratio") is not None and abs(q6["ppl_ratio"] - 1) <= H["Q6-PPL"]["tol"], qok),
                      "ratio": q6.get("ppl_ratio")},
           "Q-ORDER": {"verdict": vs(qok and q4["kld"] > q6["kld"], qok), "q6_k": q6.get("kld"), "q4_k_m": q4.get("kld")},
           "S2-TPOT": {"verdict": vs(sp is not None and sp >= H["S2-TPOT"]["bar"]), "A_p50": pa, "V_p50": pv,
                       "speedup": round(sp, 3) if sp else None, "predicted": H["S2-TPOT"]["predicted"]},
           "S2-ACC": {"verdict": vs(all(x["err_pct"] is not None and abs(x["err_pct"]) <= 100 * H["S2-ACC"]["tol"] for x in acc.values())),
                      **acc},
           "S2-TTFT": {"verdict": vs(A.get("ttft_ms_p50") and V.get("ttft_ms_p50") and V["ttft_ms_p50"] < A["ttft_ms_p50"]),
                       "A": A.get("ttft_ms_p50"), "V": V.get("ttft_ms_p50")},
           "S2-ENERGY": {"verdict": vs(A.get("j_per_token_all") and V.get("j_per_token_all") and V["j_per_token_all"] < A["j_per_token_all"]),
                         "A": A.get("j_per_token_all"), "V": V.get("j_per_token_all")}}
    s = {"session": os.path.relpath(os.path.abspath(a.session), ROOT), "quality_log": os.path.relpath(os.path.abspath(a.quality), ROOT),
         "registration": os.path.relpath(os.path.abspath(a.registration), ROOT), "registration_sha256": sha256(a.registration),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and sha256(os.path.join(ROOT, f)) != h),
         "arms": res, "runs": runs, "quality": qa, "hypotheses": hyp}
    json.dump(s, open(os.path.join(a.session, "score.json"), "w"), indent=1)
    L = ["scheduler v2 (V: %s %s) against v1 (A: %s %s), ctx %d; code changed since registration: %s" % (
        reg["arms"]["V"]["model"], reg["arms"]["V"]["place_env"]["PLACE"], reg["arms"]["A"]["model"],
        reg["arms"]["A"]["place_env"]["PLACE"], reg["ctx"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for r in runs:
        L.append("run %d %s rc %d: %d requests, client TPOT P50 %s ms, server mean %s, TTFT P50 %s ms, J/token all %s %s" % (
            r["i"], r["arm"], r["rc"], r["requests"], r["client_tpot"].get("p50"), r["server_tpot"].get("mean"), r["ttft_ms_p50"],
            r["j_per_token_all"], r["j_per_token_by_board"]))
    for v, x in qa.items():
        L.append("quality %s: PPL ratio %s, mean KLD %s, same top %s %%" % (v, x.get("ppl_ratio"), x.get("kld"), x.get("same_top_pct")))
    open(os.path.join(a.session, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
