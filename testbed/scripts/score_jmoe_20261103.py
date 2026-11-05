#!/usr/bin/env python3
"""Score jmoe's prefill (register_jmoe_20261103.py; run by lc4_20261024.py, whose own score.json holds decode).

    python3 testbed/scripts/score_jmoe_20261103.py testbed/results/jmoe_<stamp>

Per bench run (bench_<group>_<label>.log, llama-bench -o jsonl): pp512's mean over its samples (ms) and its spread; per label,
the mean over its runs. Hypotheses JM-PRE and JM-SHIFT as registered. Reported: J0's drift (its two runs), decode (tg16) per
label, and the fit of pp512 against the Jetson's blocks k: the least-squares slope over k 1, 3, 5 (ms per block moved from the
Pi 5 to the Jetson) and the step from k 0 to the fit's k 0 (what the Jetson costs as a stage). Writes score_prefill.json and
score_prefill.txt into the run folder.
"""
import argparse, datetime, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import score_prima4_20261021 as S4                          # noqa: E402


def tests(path):
    out = {}
    for ln in open(path, errors="replace"):
        if ln.startswith('{"build'):
            x = json.loads(ln)
            name = "pp%d" % x["n_prompt"] if x.get("n_prompt", 0) > 0 and x.get("n_gen", 0) == 0 else \
                   "tg%d" % x["n_gen"] if x.get("n_gen", 0) > 0 and x.get("n_prompt", 0) == 0 else None
            if name:
                out[name] = [s / 1e6 for s in x["samples_ns"]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    a = ap.parse_args()
    text = open(os.path.join(a.run, "run.log")).read()
    regrel = re.search(r"registration (\S+) \(", text).group(1)
    reg = json.load(open(os.path.join(ROOT, regrel)))
    runs = {}
    for m in re.finditer(r"##### BENCH group=(\d+) label=(\S+) .* (ok|FAILED)$", text, re.M):
        g, label, ok = m.groups()
        t = tests(os.path.join(a.run, "bench_%s_%s.log" % (g, label))) if ok == "ok" else {}
        runs.setdefault(label, []).append({"group": int(g), "ok": ok == "ok",
                                           "pp512_ms": round(statistics.fmean(t["pp512"]), 1) if t.get("pp512") else None,
                                           "pp512_spread_pct": round(100 * (max(t["pp512"]) / min(t["pp512"]) - 1), 2) if t.get("pp512") else None,
                                           "tg16_ms_per_token": round(statistics.fmean(t["tg16"]) / 16, 2) if t.get("tg16") else None})
    want = {}
    for group in reg["order"]:
        for label in group:
            want[label] = want.get(label, 0) + 1
    res = {}
    for label, b in reg["benches"].items():
        rs = [r for r in runs.get(label, []) if r["pp512_ms"]]
        pp = statistics.fmean(r["pp512_ms"] for r in rs) if rs else None
        res[label] = {"k": reg["jetson_k"][label], "runs": rs, "complete": len(rs) >= want.get(label, 0),
                      "pp512_ms": round(pp, 1) if pp else None, "predicted_pp512_ms": b["predicted"]["pp512_ms"],
                      "err_pct": round(100 * (pp / b["predicted"]["pp512_ms"] - 1), 2) if pp else None,
                      "tg16_ms_per_token": round(statistics.fmean(r["tg16_ms_per_token"] for r in rs if r["tg16_ms_per_token"]), 2)
                      if any(r["tg16_ms_per_token"] for r in rs) else None}
    ok = all(v["complete"] for v in res.values())
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    H = reg["prefill_hypotheses"]
    gain = res["J0"]["pp512_ms"] - res["J5"]["pp512_ms"] if (res["J0"]["pp512_ms"] and res["J5"]["pp512_ms"]) else None
    hyp = {"JM-PRE": {"verdict": vs(all(v["err_pct"] is not None and abs(v["err_pct"]) <= 100 * H["JM-PRE"]["tol"] for v in res.values())),
                      "err_pct": {k: v["err_pct"] for k, v in res.items()}},
           "JM-SHIFT": {"verdict": vs(gain is not None and gain >= H["JM-SHIFT"]["min_share"] * reg["predicted_gain_J0_J5_ms"]),
                        "measured_gain_ms": round(gain, 1) if gain is not None else None, "predicted_gain_ms": reg["predicted_gain_J0_J5_ms"]}}
    pts = [(v["k"], v["pp512_ms"]) for v in res.values() if v["k"] > 0 and v["pp512_ms"]]
    fit = None
    if len(pts) >= 2 and res["J0"]["pp512_ms"]:
        mk = statistics.fmean(k for k, _ in pts); mp = statistics.fmean(p for _, p in pts)
        sl = sum((k - mk) * (p - mp) for k, p in pts) / sum((k - mk) ** 2 for k, _ in pts)
        fit = {"ms_per_block_moved": round(sl, 1), "stage_step_ms": round(mp - sl * mk - res["J0"]["pp512_ms"], 1),
               "points": pts}
    j0 = [r["pp512_ms"] for r in runs.get("J0", []) if r["pp512_ms"]]
    s = {"run": os.path.relpath(os.path.abspath(a.run), ROOT), "registration": regrel,
         "registration_sha256": S4.sha256(os.path.join(ROOT, regrel)),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and S4.sha256(os.path.join(ROOT, f)) != h),
         "benches": res, "hypotheses": hyp, "fit": fit,
         "j0_drift_pct": round(100 * (max(j0) / min(j0) - 1), 2) if len(j0) >= 2 else None}
    json.dump(s, open(os.path.join(a.run, "score_prefill.json"), "w"), indent=1)
    L = ["jmoe: the Jetson prefilling Qwen3-30B-A3B Q6_K's MoE blocks; code changed since registration: %s"
         % (s["code_changed_since_registration"] or "none"), ""]
    L += ["%-9s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}) for k, h in hyp.items()]
    L.append("")
    for k, v in res.items():
        L.append("%s (Jetson %d): pp512 %s ms (predicted %s, %s %%), decode %s ms/token; runs %s" % (
            k, v["k"], v["pp512_ms"], v["predicted_pp512_ms"], v["err_pct"], v["tg16_ms_per_token"],
            [(r["pp512_ms"], r["pp512_spread_pct"]) for r in v["runs"]]))
    L += ["fit: %s" % fit, "J0 drift: %s %%" % s["j0_drift_pct"]]
    open(os.path.join(a.run, "score_prefill.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
