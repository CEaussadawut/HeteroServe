#!/usr/bin/env python3
"""Score lc4n (register_lc4n_20261028.py; run by lc4_20261024.py): the Jetson's long-context cost against fa_bench.

    python3 testbed/scripts/score_lc4n_20261028.py testbed/results/lc4n_<stamp>

Per bench and depth (llama-bench's jsonl, tee'd by the runner): ms per decoded token per repetition (each sample over n_gen)
and their mean. Per bench, the least-squares slope over its registered long range, against fa_bench's; the 768 - 512
difference; the 384 - 256 difference; and the hypotheses J5-FA, J5-SWITCH, J5-STAIR, J5-QUANT, J5-REPEAT as registered.
Depth 0 is reported as found: its ms per token, and how far the depth-256 point sits above it compared with fa_bench's
difference between 512 and 256 cells. A bench that failed or lacks a depth makes the hypotheses that need it INCOMPLETE.
Writes score_lc4n.json and score_lc4n.txt into the run folder. Nothing is filled in.
"""
import argparse, datetime, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lc4_20261024 as L                                     # noqa: E402


def reps_by_depth(path):
    out = {}
    for ln in open(path, errors="replace"):
        if ln.startswith('{"build'):
            x = json.loads(ln)
            if x.get("n_gen", 0) > 0 and x.get("n_prompt", 0) == 0:
                out[int(x.get("n_depth", 0))] = [s / 1e6 / x["n_gen"] for s in x["samples_ns"]]
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run")
    a = ap.parse_args()
    d = os.path.abspath(a.run)
    text = open(os.path.join(d, "run.log")).read()
    regrel = re.search(r"registration (\S+) \(", text).group(1)
    reg = json.load(open(os.path.join(ROOT, regrel)))
    H = reg["hypotheses_lc4n"]
    res = {}
    for m in re.finditer(r"##### BENCH group=(\d+) label=(\S+) .* (ok|FAILED)$", text, re.M):
        g, label, ok = m.groups()
        reps = reps_by_depth(os.path.join(d, "bench_%s_%s.log" % (g, label))) if ok == "ok" else {}
        b = reg["benches"][label]["predicted"]
        mean = {k: statistics.fmean(v) for k, v in reps.items()}
        r = {"ok": ok == "ok", "ms_per_token_by_depth": {str(k): round(v, 3) for k, v in sorted(mean.items())},
             "reps_ms": {str(k): [round(x, 3) for x in v] for k, v in sorted(reps.items())},
             "rep_spread_pct": {str(k): round(100 * (max(v) / min(v) - 1), 3) for k, v in sorted(reps.items()) if v}}
        lr = b["long_depths"]
        if all(k in mean for k in lr):
            r["long_slope_ms_per_1k"] = round(1000 * L.slope([(k, mean[k]) for k in lr]), 4)
            r["long_slope_err_pct"] = round(100 * (r["long_slope_ms_per_1k"] / b["long_slope_ms_per_1k"] - 1), 2)
        if 512 in mean and 768 in mean:
            r["delta_768_minus_512_ms"] = round(mean[768] - mean[512], 3)
        if 256 in mean and 384 in mean:
            r["stair_384_vs_256_pct"] = round(100 * (mean[384] / mean[256] - 1), 3)
        if 0 in mean and 256 in mean:
            fa = b["fa_ms_per_token_by_depth"]
            r["depth0"] = {"ms_per_token": round(mean[0], 3), "step_0_to_256_ms": round(mean[256] - mean[0], 3),
                           "fa_step_0_to_256_ms": round(fa["256"] - fa["0"], 3)}
        res[label] = r
    labels = list(reg["benches"])
    have = lambda *ks: all(k in res and res[k]["ok"] for k in ks)
    hyp = {}
    if have(*labels) and all("long_slope_err_pct" in res[k] for k in labels):
        e = {k: res[k]["long_slope_err_pct"] for k in labels}
        hyp["J5-FA"] = {"verdict": "HOLDS" if all(abs(v) <= 100 * H["J5-FA"]["tol"] for v in e.values()) else "FAILS",
                        "err_pct": e, "measured": {k: res[k]["long_slope_ms_per_1k"] for k in labels},
                        "fa_bench": {k: reg["benches"][k]["predicted"]["long_slope_ms_per_1k"] for k in labels}}
    else:
        hyp["J5-FA"] = {"verdict": "INCOMPLETE"}
    if have("tq", "t8", "q6") and all("delta_768_minus_512_ms" in res[k] for k in ("tq", "t8", "q6")):
        dl = {k: res[k]["delta_768_minus_512_ms"] for k in ("tq", "t8", "q6")}
        hyp["J5-SWITCH"] = {"verdict": "HOLDS" if dl["tq"] < 0 and dl["t8"] < 0 and dl["q6"] > 0 else "FAILS",
                            "delta_768_minus_512_ms": dl,
                            "fa_bench": {k: reg["benches"][k]["predicted"]["delta_768_minus_512_ms"] for k in ("tq", "t8", "q6")}}
    else:
        hyp["J5-SWITCH"] = {"verdict": "INCOMPLETE"}
    if have(*labels) and all("stair_384_vs_256_pct" in res[k] for k in labels):
        st = {k: res[k]["stair_384_vs_256_pct"] for k in labels}
        hyp["J5-STAIR"] = {"verdict": "HOLDS" if all(abs(v) <= 100 * H["J5-STAIR"]["tol"] for v in st.values()) else "FAILS",
                           "pct_384_vs_256": st}
    else:
        hyp["J5-STAIR"] = {"verdict": "INCOMPLETE"}
    if have("tq", "t8") and "long_slope_ms_per_1k" in res["tq"] and "long_slope_ms_per_1k" in res["t8"]:
        q = res["t8"]["long_slope_ms_per_1k"] / res["tq"]["long_slope_ms_per_1k"] - 1
        hyp["J5-QUANT"] = {"verdict": "HOLDS" if abs(q) <= H["J5-QUANT"]["tol"] else "FAILS", "t8_vs_tq_pct": round(100 * q, 2)}
    else:
        hyp["J5-QUANT"] = {"verdict": "INCOMPLETE"}
    want = {k: [str(x) for x in reg["benches"][k]["predicted"]["n_kv_by_depth"]] for k in labels}
    if have(*labels) and all(set(want[k]) <= set(res[k]["rep_spread_pct"]) for k in labels):
        worst = max((v, "%s@%s" % (k, dp)) for k in labels for dp, v in res[k]["rep_spread_pct"].items())
        hyp["J5-REPEAT"] = {"verdict": "HOLDS" if worst[0] <= 100 * H["J5-REPEAT"]["tol"] else "FAILS",
                            "max_spread_pct": worst[0], "at": worst[1]}
    else:
        hyp["J5-REPEAT"] = {"verdict": "INCOMPLETE"}
    s = {"run": os.path.relpath(d, ROOT), "registration": regrel, "registration_sha256": L.sha(os.path.join(ROOT, regrel)),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items()
                                                   if f.endswith((".py", ".sh", ".cpp")) and L.sha(os.path.join(ROOT, f)) != h),
         "benches": res, "hypotheses": hyp,
         "guard": re.findall(r"^\S+ (jetson (?:guard|clocks).*)$", text, re.M)}
    json.dump(s, open(os.path.join(d, "score_lc4n.json"), "w"), indent=1)
    out = ["lc4n: the Jetson's depth cost against fa_bench; code changed since registration: %s" % (
        s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        out.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    out.append("")
    for k, r in res.items():
        out.append("%s: %s" % (k, {x: y for x, y in r.items() if x not in ("reps_ms",)}))
    out += s["guard"]
    open(os.path.join(d, "score_lc4n.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
