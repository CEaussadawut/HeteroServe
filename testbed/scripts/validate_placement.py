#!/usr/bin/env python3
"""Validate the placement engine against the measured 14B layer-split sweep. Read-only.

    python3 testbed/scripts/validate_placement.py [--dataset testbed/dataset]

The question: can the engine (ThinkPad/control/placement.py), fed only single-node characterisation of a small
model, predict how a model that fits no single node behaves when pooled -- which split is fastest, how fast, and
where memory runs out?

Measured side: the Pi 5 + Pi 4 sweep of Qwen2.5-14B Q4_K_M (run_14b_pooled.sh at b23701f, -r 3): llama-bench on
the Pi 5 with -rpc <Pi 4> -ngl N. At this pin -ngl N offloads the OUTPUT LAYER first and then N-1 repeating layers
(llama-model.cpp load_tensors logs "offloading output layer" and "offloading N-1 repeating layers"), so the Pi 4
holds the last N-1 blocks plus the output head, and the Pi 5 the first 49-N blocks plus the token embedding. The
project's earlier write-ups read ngl=N as "N layers on the Pi 4"; every split here is restated in blocks.

Predicted side, with no pooled data used: each node's cost per unit of work, from its TinyLlama row at the TARGET's
quantisation (Q4_K_M -- on some nodes the cost per byte or per parameter moves with the quant's kernel, so the other
quants' rows are reported as sensitivity only):
  decode   ms/token per byte read per token   (blocks + output head; the embedding is one row per token)
  prefill  ms/token per block parameter       (llama-bench computes logits for the last prompt token only)
scaled block by block through the 14B's tensor table, plus the output head on the node that holds it and the
per-token transfers (hidden state out, logits back) at the measured link. Every measured split is predicted.

Engine checks: the memory caps memory_cap() gives each node, with each run's own MemAvailable at start, against
the splits that ran (43 blocks on the Pi 5 at ngl=6; 35 + head on the Pi 4 at ngl=36) and the one that was
OOM-killed (45 blocks on the Pi 5 at ngl=4, 2026-09-20); and the split plan_optimal picks, latency-only (power is
not measured), against the measured fastest.

Caveat printed with the result: the sweep used pp64 / tg16, the single-node rows pp512 / tg128. Prefill cost per
token depends on batch size, so the prefill comparison is weaker than decode's.
"""
import csv, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control.placement import DeviceProfile, ModelProfile, memory_cap, plan_optimal  # noqa: E402

SWEEP = os.path.join(ROOT, "testbed", "results", "pool14b_20260925_025332.log")
OOM_RUN = os.path.join(ROOT, "testbed", "results", "pool14b_20260920_162912.log")
TARGET = ("qwen2.5-14b-instruct-q4_k_m", "Q4_K_M")
TINY = {"Q4_K_M": "tinyllama-1.1b-q4km", "Q8_0": "tinyllama-1.1b-q8_0", "Q4_0": "tinyllama-1.1b-q4_0"}


def load_run(path):
    """({ngl: {"pp": [t/s samples], "tg": [...]}}, meta, {node: MemAvailable MiB at start}, [killed ngl])."""
    out, ngl, meta, avail, killed = {}, None, {}, {}, []
    for ln in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"(pi5|pi4) avail MiB : (\d+)", ln)
        if m:
            avail[m.group(1)] = int(m.group(2)); continue
        m = re.match(r"##### ngl=(\d+)", ln)
        if m:
            ngl = int(m.group(1)); out.setdefault(ngl, {}); continue
        if ngl is not None and re.search(r"\bKilled\b.*llama-bench", ln):
            killed.append(ngl)
        if ln.startswith('{"build_commit"') and ngl is not None:
            r = json.loads(ln)
            out[ngl].setdefault("pp" if r.get("n_prompt") else "tg", []).extend(r.get("samples_ts") or [])
            meta = {"build_commit": r["build_commit"], "n_prompt": r.get("n_prompt") or meta.get("n_prompt"),
                    "n_gen": r.get("n_gen") or meta.get("n_gen")}
    return {k: v for k, v in out.items() if v.get("pp") and v.get("tg")}, meta, avail, killed


def fit(xs, ys):
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return a, b, 1 - ss_res / sum((y - my) ** 2 for y in ys)


def tiny_ms(char_csv, node):
    """{quant: {"pp": ms/token, "tg": ms/token}} from the node's unflagged TinyLlama CPU rows."""
    out = {}
    for r in csv.DictReader(open(char_csv)):
        if r["node_id"] == node and r["device"] == "cpu" and not r["flags"] and "tinyllama" in r["model"]:
            out.setdefault(r["quant"], {})[r["phase"]] = float(r["ms_per_token"])
    return out


def unit_costs(ms, shape):
    """(ms per byte read per decoded token, ms per block parameter per prompt token)."""
    read_bytes = sum(shape["layer_bytes"]) + shape["head_bytes"]
    return ms["tg"] / read_bytes, ms["pp"] / (shape["params_per_layer"] * shape["n_layers"])


def predict(x, phase, u5, u4, q, link, n_prompt):
    """ms per token with x blocks (the last x) plus the output head on the Pi 4, the rest on the Pi 5."""
    L, lb = q["n_layers"], q["layer_bytes"]
    n_vocab = (q["head_params"] - q["n_embd"]) / q["n_embd"]         # output.weight is n_embd x n_vocab
    bw = link["mbps"] * 1e6 / 8 / 1000                                 # bytes per ms
    if phase == "tg":
        compute = u5[0] * sum(lb[:L - x]) + u4[0] * (sum(lb[L - x:]) + q["head_bytes"])
        return compute + 2 * link["rtt_ms"] + (q["activation_bytes"] + 4 * n_vocab) / bw
    ppl = q["params_per_layer"]
    compute = u5[1] * ppl * (L - x) + u4[1] * ppl * x + u4[1] * q["head_params"] / n_prompt
    return compute + (2 * link["rtt_ms"] + (n_prompt * q["activation_bytes"] + 4 * n_vocab) / bw) / n_prompt


def main(argv):
    ds = argv[argv.index("--dataset") + 1] if "--dataset" in argv else os.path.join(ROOT, "testbed", "dataset")
    shapes = json.load(open(os.path.join(ds, "model_shapes.json")))
    link = json.load(open(os.path.join(ds, "link.json")))["pairs"]["pi5-pi4"]
    q = shapes[TARGET[0]]
    if q.get("bytes_per_layer_source") != "tensor table":
        raise SystemExit("model_shapes.json has no tensor-table figures for %s; regenerate it" % TARGET[0])
    data, meta, avail, _ = load_run(SWEEP)
    _, _, oom_avail, oom_killed = load_run(OOM_RUN)
    L, n_prompt = q["n_layers"], meta["n_prompt"]
    ms = {n: tiny_ms(os.path.join(ds, "characterisation.csv"), n) for n in ("pi5", "pi4")}
    units = {qt: (unit_costs(ms["pi5"][qt], shapes[TINY[qt]]), unit_costs(ms["pi4"][qt], shapes[TINY[qt]]))
             for qt in TINY if qt in ms["pi5"] and qt in ms["pi4"]}
    u5, u4 = units[TARGET[1]]
    xs = sorted(n - 1 for n in data)                                   # blocks on the Pi 4 (plus the head)
    res = {"sweep_log": os.path.relpath(SWEEP, ROOT), "oom_log": os.path.relpath(OOM_RUN, ROOT),
           "build_commit": meta["build_commit"], "layout": "ngl=N: Pi 4 holds the last N-1 blocks + output head; "
           "Pi 5 the first 49-N blocks + token embedding", "blocks_on_pi4": xs, "link": link,
           "predictor": "TinyLlama %s rows, per byte (decode) / per block parameter (prefill)" % TARGET[1]}
    print("14B sweep at %s: blocks on the Pi 4 (each plus the output head): %s" % (meta["build_commit"], xs))

    for ph, name in (("tg", "decode"), ("pp", "prefill")):
        meas = [1000.0 / statistics.fmean(data[x + 1][ph]) for x in xs]
        pred = [predict(x, ph, u5, u4, q, link, n_prompt) for x in xs]
        a, b, r2 = fit(xs, meas)
        ap, bp, _ = fit(xs, pred)
        err = [100 * (p - m) / m for p, m in zip(pred, meas)]
        sens = {qt: [round(predict(x, ph, v[0], v[1], q, link, n_prompt), 1) for x in (xs[0], xs[-1])]
                for qt, v in units.items()}
        res[name] = {"measured_ms": {x: round(m, 1) for x, m in zip(xs, meas)},
                     "predicted_ms": {x: round(p, 1) for x, p in zip(xs, pred)},
                     "error_pct": {x: round(e, 1) for x, e in zip(xs, err)},
                     "mean_abs_error_pct": round(statistics.fmean(abs(e) for e in err), 1),
                     "fit": {"intercept_ms": round(a, 1), "slope_ms_per_block": round(b, 2), "r2": round(r2, 4)},
                     "predicted_fit": {"intercept_ms": round(ap, 1), "slope_ms_per_block": round(bp, 2)},
                     "slope_error_pct": round(100 * (bp - b) / b, 1),
                     "sensitivity_by_quant_ms_at_%d_and_%d_blocks" % (xs[0], xs[-1]): sens}
        print("\n%s (%s%d): measured ms/token = %.1f + %.2f x   R^2 %.4f;  predicted %.1f + %.2f x  (slope %+.1f%%)"
              % (name, ph, n_prompt if ph == "pp" else meta["n_gen"], a, b, r2, ap, bp, res[name]["slope_error_pct"]))
        for x, m, p, e in zip(xs, meas, pred, err):
            print("  %2d blocks on the Pi 4: measured %7.1f  predicted %7.1f  (%+.1f%%)" % (x, m, p, e))
        print("  mean |error| %.1f%%;  other quants' rows would predict (at %d / %d blocks): %s"
              % (res[name]["mean_abs_error_pct"], xs[0], xs[-1], sens))

    # Memory: the caps the engine gives, with each run's own MemAvailable at start.
    mem = ModelProfile(TARGET[0], L, q["bytes_per_layer"], q["kv_bytes_per_layer_per_token"], q["activation_bytes"],
                       context_len=n_prompt + meta["n_gen"], head_bytes=q["head_bytes"], embd_bytes=q["embd_bytes"],
                       host_in_pipeline=True)
    old = ModelProfile(TARGET[0], L, q["file_size"] / L, q["kv_bytes_per_layer_per_token"], q["activation_bytes"],
                       context_len=n_prompt + meta["n_gen"])
    ran5, ran4 = L + 1 - min(data), max(data) - 1                      # most blocks each node held in a run that ran
    killed5 = [L + 1 - n for n in oom_killed]
    checks, caps = {}, {}
    for run, av in (("2026-09-25 sweep", avail), ("2026-09-20 run", oom_avail)):
        d5 = DeviceProfile("pi5", 0, 0, usable_mem_mb=av["pi5"]); d4 = DeviceProfile("pi4", 0, 0, usable_mem_mb=av["pi4"])
        caps[run] = {"mem_available_mib": av, "pi5_first_with_embedding": memory_cap(d5, mem, holds_embd=True),
                     "pi4_last_with_head": memory_cap(d4, mem, holds_head=True),
                     "old_accounting_pi5": memory_cap(d5, old), "old_accounting_pi4": memory_cap(d4, old)}
    c25, c20 = caps["2026-09-25 sweep"], caps["2026-09-20 run"]
    checks["pi5 holds %d blocks (ngl=%d ran)" % (ran5, min(data))] = c25["pi5_first_with_embedding"] >= ran5
    checks["pi4 holds %d blocks + head (ngl=%d ran)" % (ran4, max(data))] = c25["pi4_last_with_head"] >= ran4
    for k in killed5:
        checks["pi5 refuses %d blocks (ngl=%d OOM-killed)" % (k, L + 1 - k)] = c20["pi5_first_with_embedding"] < k
    res["memory_caps"], res["memory_checks"] = caps, checks
    print("\nmemory caps, blocks (usable = the run's MemAvailable at start; 350 MB overhead):")
    for run, c in caps.items():
        print("  %-17s %s" % (run, c))
    for k, v in checks.items():
        print("  %-40s %s" % (k, "OK" if v else "CONTRADICTS the measurement"))

    # The rule, latency-only, with the predicted per-block costs (uniform blocks: the mean over the model).
    devs = {"pi5": DeviceProfile("pi5", u5[0] * q["bytes_per_layer"], u5[1] * q["params_per_layer"], avail["pi5"],
                                 link_mbps=link["mbps"], rtt_ms=link["rtt_ms"]),
            "pi4": DeviceProfile("pi4", u4[0] * q["bytes_per_layer"], u4[1] * q["params_per_layer"], avail["pi4"],
                                 link_mbps=link["mbps"], rtt_ms=link["rtt_ms"], is_straggler=True)}
    # The sweep ran llama-bench on the Pi 5, the only board holding the 14B file: the rule chooses the split for that host.
    # head_placement="last": the sweep ran llama.cpp's own layout, the output head on the last board (the Pi 4), and
    # this checks the pick against it; plan_optimal's default became "best" on 2026-10-08.
    rule = plan_optimal(devs, mem, "decode", host_candidates=["pi5"], head_placement="last")
    rule_old = plan_optimal(devs, old, "decode")
    best = min(xs, key=lambda x: res["decode"]["measured_ms"][x])
    res["engine"] = {"pick": {"order": rule.order, "blocks": rule.layers},
                     "pick_old_accounting": {"order": rule_old.order, "blocks": rule_old.layers},
                     "measured_fastest": {"pi5": L - best, "pi4": best}}
    print("\nengine pick (latency-only): %s %s   old accounting: %s   measured fastest: Pi 5 %d / Pi 4 %d + head"
          % (rule.order, rule.layers, rule_old.layers, L - best, best))
    if 23 in xs:
        m_even, m_best = res["decode"]["measured_ms"][23], res["decode"]["measured_ms"][best]
        p_even, p_best = res["decode"]["predicted_ms"][23], res["decode"]["predicted_ms"][best]
        res["tpot_reduction_vs_ngl24_pct"] = {"measured": round(100 * (m_even - m_best) / m_even, 1),
                                              "predicted": round(100 * (p_even - p_best) / p_even, 1)}
        print("decode TPOT, fastest vs the near-even ngl=24 (Pi 5 25 / Pi 4 23 + head): measured %.0f -> %.0f ms "
              "(%.1f%% lower), predicted %.1f%% lower. QR8 stretch: 25%%."
              % (m_even, m_best, res["tpot_reduction_vs_ngl24_pct"]["measured"],
                 res["tpot_reduction_vs_ngl24_pct"]["predicted"]))
    print("\ncaveat: the sweep ran pp%d / tg%d; the single-node rows pp512 / tg128. Prefill cost per token depends"
          " on batch size, so prefill is the weaker comparison." % (n_prompt, meta["n_gen"]))
    out = os.path.join(ds, "placement_validation.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
