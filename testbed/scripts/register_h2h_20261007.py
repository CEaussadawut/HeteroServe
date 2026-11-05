#!/usr/bin/env python3
"""Register the engine-vs-Halda head-to-head (2026-10-07) before it runs. Reads files only; writes one new file.

    python3 testbed/scripts/register_h2h_20261007.py [--dry-run] [--out FILE --supersedes "WHY"]

Pi 4 hosts, the Jetson is its worker (the Pi 5 is out of this session). Every cell is computed here with the same
planner, Halda port and predictor as compare_halda.py, at the same planning context (n_ctx 4096), and must reproduce
the offline comparison's published prediction (testbed/dataset/halda_offline_20261007.json) to 0.1 ms, or nothing is
written:
  x.tiny.engine            the engine's plan for TinyLlama         pi4 1 + jetson 21, head on the Jetson
  x.tiny.halda             Halda's layout                          pi4 1 + jetson 21, head on the Pi 4 (-ot)
  x.q3_14b.engine_host     the engine, head where cheaper          pi4 26 + jetson 14, head on the Pi 4 (-ot)
  x.q3_14b.engine_last     the engine, head on the last board      pi4 32 + jetson 8, head on the Jetson
plus a repeat of each model's first cell, run last (A-B-A): the 14B's predicted gap, 3.7 %, is near the ~2 % drift of
one configuration minutes apart, and the bracket measures that drift in the same session. Halda's own 14B plan (pi4 25 +
jetson 15) fails the engine's validated memory accounting by 126 MiB and is recorded under not_run, not run.
The head on the host is -ot "output\\.weight=CPU" at the pin (predict_pool.bench_args, head_at=0). Memory is the
engine's accounting at the planned 4096 context, so the run's gate asks for at least what the plan would need there.
Output: testbed/dataset/pool_predictions_20261007h2h.json, made read-only. An existing file is never overwritten.
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
from control.gguf import read_metadata                     # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261007h2h.json")
N_CTX = 4096
HOST, NODES = "pi4", ["pi4", "jetson"]
PUBLISHED = {("qwen3_14b", "engine_last"): 2077.3, ("qwen3_14b", "engine_host"): 2000.5,
             ("tinyllama", "engine"): 95.4, ("tinyllama", "halda"): 101.7}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    global OUT
    if "--out" in argv:
        OUT = os.path.abspath(argv[argv.index("--out") + 1])
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = pp.rows(inputs["characterisation.csv"], "char")
    reg = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "engine vs Halda's layout on the same runtime: Pi 4 host + Jetson (Vulkan), pre-registered",
           "inputs_sha256": {f: sha(p) for f, p in inputs.items()},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "halda.py", "gguf.py")},
           "predictor_sha256": {f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "compare_halda.py", os.path.basename(__file__))},
           "n_ctx_planned": N_CTX, "protocol": "-t 4 -p 64 -n 16 -r 3 -oe jsonl -v",
           "run_order": ["x.tiny.engine", "x.tiny.halda", "x.tiny.engine_rep",
                         "x.q3_14b.engine_host", "x.q3_14b.engine_last", "x.q3_14b.engine_host_rep"],
           "not_run": {}, "configs": {}}
    if "--supersedes" in argv:
        reg["supersedes"] = argv[argv.index("--supersedes") + 1]
    for key in ("tinyllama", "qwen3_14b"):
        shape, quant, path = pp.MODELS[key]
        q = shapes[shape]
        kv = read_metadata(os.path.join(ch.CACHE, ch.FILES[shape]))
        n_ff = int(kv[kv["general.architecture"] + ".feed_forward_length"])
        C, src = pp.costs(R, shapes, quant)
        L, missing = pp.link_view(pairs, HOST, NODES)
        if missing:
            sys.exit("refused: no measured link %s" % missing)
        devs = pp.profiles(C, q, mem, L, NODES)
        mp = replace(pp.model_profile(q, shape), context_len=N_CTX)
        plans = {}
        for name, hp in (("engine_last", "last"), ("engine_best", "best")):
            # the 2026-10-07 registrations were planned with the average fill (the default until that evening)
            r = placement.plan_optimal(devs, mp, "decode", host_candidates=[HOST], head_placement=hp, fill="average")
            if not r.feasible:
                sys.exit("refused: %s %s infeasible: %s" % (key, name, r.reason[:160]))
            plans[name] = (ch.pipeline(r), 0 if r.head_node else None)
        ring, hres, _ = ch.halda_plans(["jetson"], HOST, q, C, mem, N_CTX, n_ff)[0]
        h_assign = [(n, hres["layers"][n]) for n in ring if n in hres["layers"]] if hres["feasible"] else None
        if key == "tinyllama":
            cells = {"engine": plans["engine_best"], "halda": (h_assign, 0)}
        else:
            if plans["engine_best"][1] != 0:
                sys.exit("refused: the engine's best 14B plan no longer puts the head on the host")
            cells = {"engine_host": plans["engine_best"], "engine_last": plans["engine_last"]}
            hm = pp.memory(h_assign, q, mem, devs, mp, head_at=0)
            worst = min(min(v["margin_mib"], v.get("weights_margin_mib", v["margin_mib"])) for v in hm.values())
            reg["not_run"]["x.q3_14b.halda"] = {"pipeline": [{"node": n, "blocks": k} for n, k in h_assign], "head_on": HOST,
                                                "min_margin_mib": worst,
                                                "why": "Halda's own plan fails the engine's validated memory accounting"}
        short = "tiny" if key == "tinyllama" else "q3_14b"
        for cname, (assign, head_at) in cells.items():
            dec, pre = pp.predict(assign, q, C, L, head_at=head_at)
            want = PUBLISHED[(key, cname)]
            if abs(round(dec, 1) - want) > 0.1:
                sys.exit("refused: %s %s predicts %.1f ms, the offline comparison published %.1f" % (key, cname, dec, want))
            m = pp.memory(assign, q, mem, devs, mp, head_at=head_at)
            reg["configs"]["x.%s.%s" % (short, cname)] = {
                "model": shape, "quant": quant,
                "scheduler": {"engine": "the engine (head where cheaper)", "halda": "Halda's layout (head on the host)",
                              "engine_host": "the engine (head where cheaper: the host)",
                              "engine_last": "the engine with the head on the last board (llama.cpp's -ngl default)"}[cname],
                "why": "head-to-head 2026-10-07, %s on Pi 4 host + Jetson" % key,
                "pipeline": [{"node": n, "blocks": k} for n, k in assign],
                "head_on": assign[0][0] if head_at == 0 else assign[-1][0], "head_at": head_at,
                "bench": pp.bench_args(assign, key, head_at=head_at), "memory": m,
                "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                              "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2)},
                "cost_sources": {n: src[n] for n, _ in assign}}
    c = reg["configs"]
    c["x.tiny.engine_rep"] = dict(c["x.tiny.engine"], why="the A-B-A repeat of x.tiny.engine, run last: drift, not a new plan")
    c["x.q3_14b.engine_host_rep"] = dict(c["x.q3_14b.engine_host"],
                                         why="the A-B-A repeat of x.q3_14b.engine_host, run last: drift, not a new plan")
    for cid in reg["run_order"]:
        x = c[cid]
        print("%-26s %-22s head %-6s decode %7.1f  prefill %6.1f  min margin %5d MiB   %s %s" % (
            cid, " + ".join("%s %d" % (p["node"], p["blocks"]) for p in x["pipeline"]), x["head_on"],
            x["predicted"]["decode_ms_per_token"], x["predicted"]["prefill_ms_per_token"],
            min(v["margin_mib"] for v in x["memory"].values()), x["bench"]["host"], x["bench"]["args"]))
    for cid, x in reg["not_run"].items():
        print("%-26s NOT RUN: %s (min margin %d MiB)" % (cid, x["why"], x["min_margin_mib"]))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
