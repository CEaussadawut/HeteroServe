#!/usr/bin/env python3
"""Score upgrade 1's step 1 (register_spec_pass_20261010.py) against its registered hypotheses, as written.

    python3 testbed/scripts/score_spec_pass_20261010.py <pool_sp.q3_14b.pass_<stamp>.log> [--reg PATH] [--out PATH]

Reads the cell's run_pool.sh log: llama-batched-bench's JSONL lines ({"n_kv_max": ..., "pl": k, "tg": 8, "t_tg": s, ...}),
the THROTTLE line, EXIT_BENCH, and the RPC model buffer that load_tensors logs. T(k) = t_tg / tg x 1000, ms per pass of
k tokens, each with logits; per k the mean of its lines (registered: 3) and their spread (max - min) / mean.
  SP-K1    T(1) within 5 % of the registered decode ms per token
  SP-BAND  every T(k), k = 2..5, within its registered band [0.95 x ROOF(k), 1.05 x ADD(k)]
  SP-PAYS  for some k in 2..5, T(k) / T(1) < the registered E(k - 1; alpha)
A k with fewer lines than the registered rounds makes every hypothesis that needs it INCOMPLETE; a bench that did not
exit 0, or a Jetson buffer more than 0.5 MiB from the plan's, is reported beside the verdicts. Exploratory: each model's
error per k (100 x (predicted / measured - 1)) and mean absolute error; the overlap fraction phi of
T(k) - T(1) = (k - 1) phi sum(pre_i), least squares through the origin; the ceiling speedup max_k E(k - 1; a) T(1) / T(k)
for a = 0.5 .. 0.9 (no drafter cost). Writes --out (default: score.json beside the log).
"""
import hashlib, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))

REG = os.path.join(ROOT, "testbed", "dataset", "spec_pass_predictions_20261010.json")
CELL = "sp.q3_14b.pass"
BUF = re.compile(r"load_tensors: +RPC\d*\[([^\]]+)\] model buffer size = +([\d.]+) MiB")
MIB = 2 ** 20


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parse(lines):
    """{"passes": {k: [ms, ...]}, "throttle": {node: {field: value}}, "exit_bench": int|None, "buffers": {ep: MiB}}."""
    out = {"passes": {}, "throttle": {}, "exit_bench": None, "buffers": {}}
    for ln in lines:
        s = ln.strip()
        if s.startswith('{"n_kv_max"'):
            try:
                r = json.loads(s)
            except ValueError:
                continue
            if r.get("tg"):
                out["passes"].setdefault(int(r["pl"]), []).append(1000.0 * float(r["t_tg"]) / int(r["tg"]))
        elif s.startswith("THROTTLE "):
            parts = s.split()
            out["throttle"][parts[1]] = dict(p.split("=", 1) for p in parts[2:] if "=" in p)
        elif s.startswith("EXIT_BENCH="):
            out["exit_bench"] = int(s.split("=", 1)[1])
        else:
            m = BUF.search(s)
            if m:
                out["buffers"][m.group(1)] = float(m.group(2))
    return out


def tokens_per_check(k, a):
    return (1 - a ** k) / (1 - a)


def score(reg, data, q=None):
    c = reg["configs"][CELL]
    pk, dec = c["predicted"]["per_k"], c["predicted"]["decode_ms"]
    ks, rounds = reg["ks"], reg["rounds"]
    T, spread, n = {}, {}, {}
    for k in ks:
        xs = data["passes"].get(k, [])
        n[k] = len(xs)
        if xs:
            T[k] = statistics.fmean(xs)
            spread[k] = (max(xs) - min(xs)) / T[k] if len(xs) > 1 else None
    full = {k: n[k] >= rounds for k in ks}
    hyp = {}
    if full.get(1):
        e = T[1] / dec - 1
        hyp["SP-K1"] = {"verdict": "HOLDS" if abs(e) <= 0.05 else "FAILS", "t1_ms": round(T[1], 1), "decode_ms": dec,
                        "err_pct": round(100 * e, 2)}
    else:
        hyp["SP-K1"] = {"verdict": "INCOMPLETE"}
    rest = [k for k in ks if k != 1]
    if all(full.get(k) for k in rest):
        inside = {k: pk[str(k)]["band_ms"][0] <= T[k] <= pk[str(k)]["band_ms"][1] for k in rest}
        hyp["SP-BAND"] = {"verdict": "HOLDS" if all(inside.values()) else "FAILS",
                          "inside": {str(k): v for k, v in inside.items()},
                          "measured_ms": {str(k): round(T[k], 1) for k in rest},
                          "band_ms": {str(k): pk[str(k)]["band_ms"] for k in rest}}
    else:
        hyp["SP-BAND"] = {"verdict": "INCOMPLETE"}
    if full.get(1) and all(full.get(k) for k in rest):
        ratio = {k: T[k] / T[1] for k in rest}
        pays = {k: ratio[k] < pk[str(k)]["tokens_per_check_at_alpha"] for k in rest}
        hyp["SP-PAYS"] = {"verdict": "HOLDS" if any(pays.values()) else "FAILS",
                          "ratio": {str(k): round(v, 3) for k, v in ratio.items()},
                          "e_at_alpha": {str(k): pk[str(k)]["tokens_per_check_at_alpha"] for k in rest},
                          "pays": {str(k): v for k, v in pays.items()}}
    else:
        hyp["SP-PAYS"] = {"verdict": "INCOMPLETE"}
    # exploratory
    err = {m: {str(k): round(100 * (pk[str(k)]["%s_ms" % m] / T[k] - 1), 2) for k in ks if k in T} for m in ("add", "roof")}
    mae = {m: round(statistics.fmean(abs(v) for v in err[m].values()), 2) if err[m] else None for m in err}
    phi = None
    if 1 in T and any(k in T for k in rest):
        pre = c["predicted"]["pre_sum_ms_per_token"]
        num = sum((k - 1) * (T[k] - T[1]) for k in rest if k in T)
        den = pre * sum((k - 1) ** 2 for k in rest if k in T)
        phi = round(num / den, 3) if den else None
    ceil = {}
    if 1 in T:
        for a in (0.5, 0.6, 0.7, 0.8, 0.9):
            best = max(((tokens_per_check(k, a) * T[1] / T[k], k) for k in ks if k in T), default=None)
            if best:
                ceil["%.1f" % a] = {"speedup": round(best[0], 3), "check_k": best[1], "draft_tokens": best[1] - 1}
    jet = None
    if q is not None:
        start = sum(p["blocks"] for p in c["pipeline"][:1])
        want = sum(q["layer_bytes"][start:start + c["pipeline"][1]["blocks"]]) / MIB
        got = list(data["buffers"].values())
        jet = {"expected_mib": round(want, 2), "logged_mib": got,
               "ok": len(got) == 1 and abs(got[0] - want) <= 0.5}
    return {"per_k": {str(k): {"samples": n[k], "mean_ms": round(T[k], 1) if k in T else None,
                               "spread": round(spread[k], 4) if spread.get(k) is not None else None,
                               "ratio_to_k1": round(T[k] / T[1], 3) if (k in T and 1 in T) else None} for k in ks},
            "hypotheses": hyp, "exit_bench": data["exit_bench"], "thermal": data["throttle"], "jetson_buffer": jet,
            "exploratory": {"model_err_pct": err, "model_mae_pct": mae, "phi": phi, "ceiling_speedup": ceil}}


def main(argv):
    if not argv:
        sys.exit(__doc__)
    log = argv[0]
    reg_path = argv[argv.index("--reg") + 1] if "--reg" in argv else REG
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(os.path.dirname(log), "score.json")
    reg = json.load(open(reg_path))
    import predict_pool as P
    q = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))[P.MODELS[reg["model"]][0]]
    data = parse(open(log, encoding="utf-8", errors="replace").read().splitlines())
    s = score(reg, data, q)
    code = [f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh")) and os.path.exists(os.path.join(HERE, f))
            and sha(os.path.join(HERE, f)) != h]
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha(reg_path), "log": os.path.relpath(log, ROOT),
         "code_changed_since_registration": code, **s}
    json.dump(s, open(out, "w"), indent=1)
    print("%-3s %8s %10s %8s %8s %9s %9s" % ("k", "samples", "T(k) ms", "spread", "x T(1)", "ADD err", "ROOF err"))
    for k, v in s["per_k"].items():
        print("%-3s %8d %10s %8s %8s %9s %9s" % (k, v["samples"], v["mean_ms"], v["spread"], v["ratio_to_k1"],
                                                s["exploratory"]["model_err_pct"]["add"].get(k), s["exploratory"]["model_err_pct"]["roof"].get(k)))
    for h, v in s["hypotheses"].items():
        print("%-8s %s  %s" % (h, v["verdict"], {k: w for k, w in v.items() if k != "verdict"}))
    print("thermal %s; exit %s; jetson buffer %s" % ({n: t.get("verdict") for n, t in s["thermal"].items()}, s["exit_bench"],
                                                     s["jetson_buffer"]))
    print("exploratory: mae %s, phi %s" % (s["exploratory"]["model_mae_pct"], s["exploratory"]["phi"]))
    print("ceiling speedup (no drafter cost): %s" % s["exploratory"]["ceiling_speedup"])
    print("code changed since registration: %s" % code)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
