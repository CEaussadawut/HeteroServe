#!/usr/bin/env python3
"""Register the first measured run of the default exact plan (2026-10-07), before it runs. Reads files only.

    python3 testbed/scripts/register_exact_20261007.py [--dry-run]

Qwen3-14B on Pi 4 host + Jetson, planned by placement.plan_optimal with its defaults since 2026-10-07: the exact fill
(real block sizes) and EXACT_FILL_HEADROOM_MB kept free per board, head where cheaper. It is run A-B-A against the plan
it replaces -- the average fill without headroom, measured at 16:45-17:15 today as x.q3_14b.engine_host -- because the
predicted difference, 1.3 %, is near the ~1 % drift of one plan run twice:
  x.q3_14b.default       the default plan (must reproduce compare_halda's engine_default: pi4 27 + jetson 13, 2025.9 ms)
  x.q3_14b.ref_average   the plan it replaces (pi4 26 + jetson 14, 2000.5 ms, as registered in ..._20261007h2h_v2.json)
  x.q3_14b.default_rep   the default plan again
Memory per board is the engine's accounting at the planned 4096 context, without the headroom, so the margins show what
the headroom left free. Output: testbed/dataset/pool_predictions_20261007exact.json, read-only; never overwritten.
"""
import datetime, hashlib, json, os, stat, sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402
import compare_halda as ch                                  # noqa: E402
from control import placement                              # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261007exact.json")
N_CTX, HOST, NODES, KEY = 4096, "pi4", ["pi4", "jetson"], "qwen3_14b"
PUBLISHED = {"default": ([("pi4", 27), ("jetson", 13)], 2025.9), "ref_average": ([("pi4", 26), ("jetson", 14)], 2000.5)}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = pp.rows(inputs["characterisation.csv"], "char")
    shape, quant, _ = pp.MODELS[KEY]
    q = shapes[shape]
    C, src = pp.costs(R, shapes, quant)
    L, missing = pp.link_view(pairs, HOST, NODES)
    if missing:
        sys.exit("refused: no measured link %s" % missing)
    devs = pp.profiles(C, q, mem, L, NODES)
    mp = replace(pp.model_profile(q, shape), context_len=N_CTX)          # carries layer_bytes: the default is exact
    plans = {"default": placement.plan_optimal(devs, mp, "decode", host_candidates=[HOST], head_placement="best"),
             "ref_average": placement.plan_optimal(devs, mp, "decode", host_candidates=[HOST], head_placement="best",
                                                   fill="average", headroom_mb=0)}
    reg = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "the default exact plan (exact fill + %.0f MiB headroom) measured, A-B-A against the plan it "
                       "replaces; Qwen3-14B on Pi 4 host + Jetson (Vulkan)" % placement.EXACT_FILL_HEADROOM_MB,
           "default_headroom_mib": placement.EXACT_FILL_HEADROOM_MB,
           "inputs_sha256": {f: sha(p) for f, p in inputs.items()},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "predictor_sha256": {f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", os.path.basename(__file__))},
           "n_ctx_planned": N_CTX, "protocol": "-t 4 -p 64 -n 16 -r 3 -oe jsonl -v",
           "run_order": ["x.q3_14b.default", "x.q3_14b.ref_average", "x.q3_14b.default_rep"], "configs": {}}
    for name, p in plans.items():
        if not p.feasible:
            sys.exit("refused: the %s plan is infeasible: %s" % (name, p.reason[:160]))
        assign, head_at = ch.pipeline(p), (0 if p.head_node else None)
        dec, pre = pp.predict(assign, q, C, L, head_at=head_at)
        want_assign, want_dec = PUBLISHED[name]
        if assign != want_assign or abs(round(dec, 1) - want_dec) > 0.1:
            sys.exit("refused: %s plans %s at %.1f ms; compare_halda published %s at %.1f" % (name, assign, dec, want_assign, want_dec))
        m = pp.memory(assign, q, mem, devs, mp, head_at=head_at)
        reg["configs"]["x.q3_14b.%s" % name] = {
            "model": shape, "quant": quant,
            "scheduler": {"default": "the engine's default: exact fill + %.0f MiB headroom, head where cheaper" % placement.EXACT_FILL_HEADROOM_MB,
                          "ref_average": "the plan it replaces: average fill, no headroom, head where cheaper"}[name],
            "why": "the first measured run of the default exact plan, 2026-10-07",
            "pipeline": [{"node": n, "blocks": k} for n, k in assign],
            "head_on": assign[0][0] if head_at == 0 else assign[-1][0], "head_at": head_at,
            "bench": pp.bench_args(assign, KEY, head_at=head_at), "memory": m,
            "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                          "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2)},
            "cost_sources": {n: src[n] for n, _ in assign}}
    c = reg["configs"]
    c["x.q3_14b.default_rep"] = dict(c["x.q3_14b.default"], why="the A-B-A repeat of x.q3_14b.default, run last: drift, not a new plan")
    for cid in reg["run_order"]:
        x = c[cid]
        print("%-22s %-20s head %-6s decode %7.1f  prefill %6.1f  margins %s   %s %s" % (
            cid, " + ".join("%s %d" % (s["node"], s["blocks"]) for s in x["pipeline"]), x["head_on"],
            x["predicted"]["decode_ms_per_token"], x["predicted"]["prefill_ms_per_token"],
            {n: v["margin_mib"] for n, v in x["memory"].items()}, x["bench"]["host"], x["bench"]["args"]))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
