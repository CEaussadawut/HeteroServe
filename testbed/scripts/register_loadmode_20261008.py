#!/usr/bin/env python3
"""Pre-register the load-mode A/B of 2026-10-08: a Pi 4 host's weights memory-mapped (every cell so far) against read into
memory (llama-bench -lm none, as serve_pool.sh now serves). Reads files only.

    python3 testbed/scripts/register_loadmode_20261008.py [--dry-run]

Why. A Pi 4 host serves its blocks from the page cache, and the first prefill repetition is sometimes COLD: llama-bench's
warm-up does not keep them resident, and a timed repetition reads them back from the SD card (2026-09-28: "the first
prefill repetition is cold"; 2026-10-07: x.q3_14b.default_rep's first pp64 repetition 76.0 s against 53.2/53.4, +446 MiB of
RssFile inside it). Ten Pi 4-hosted cells show it (compare_pool.py's page-in flag, 2026-10-08). Serving the same plan mapped
thrashed outright (4 tokens in 5.5 min) and served at the benchmark's speed with -lm none. This A/B decides whether the
benchmark protocol should load the weights into memory too.
  lm.q3_14b.mmap   Qwen3-14B, the engine's default plan on Pi 4 + Jetson (Pi 4 27 + Jetson 13, the head on the Pi 4), as run
                   on 2026-10-07 (x.q3_14b.default)
  lm.q3_14b.none   the same, with -lm none
Run mmap, none, mmap, none (run_pool.sh, one cell at a time), at the pool protocol (-t 4 -p 64 -n 16 -r 3 -oe jsonl -v).
Hypotheses (scored from compare_pool.py's results and the jsonl repetitions, and reported as found):
  H-LM-COLD   in both `none` cells every scored test's three repetitions agree within 5 % (max/min), and compare_pool flags
              no page-in
  H-LM-SPEED  the `none` cells' decode ms/token within +-3 % of the `mmap` cells' (means over both runs), and their prefill
              within +-3 % of the `mmap` cells' fastest repetition (a cold repetition is the effect under test, not a speed)
Output: testbed/dataset/pool_predictions_20261008loadmode.json, read-only; never overwritten.
"""
import datetime, hashlib, json, os, stat, sys
from dataclasses import replace

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261008loadmode.json")
N_CTX, HOST, NODES, KEY = 4096, "pi4", ["pi4", "jetson"], "qwen3_14b"


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
    mp = replace(pp.model_profile(q, shape), context_len=N_CTX)
    plan = pp.plan_optimal(devs, mp, "decode", host_candidates=[HOST])
    assign = [(n, plan.layers[n]) for n in plan.order if plan.layers.get(n, 0) > 0]
    head_at = 0 if plan.head_node == assign[0][0] else None
    if (assign, head_at) != ([("pi4", 27), ("jetson", 13)], 0):
        sys.exit("refused: the default plan is %s head_at %s, not the one registered on 10-07" % (assign, head_at))
    dec, pre = pp.predict(assign, q, C, L, head_at=head_at)
    m = pp.memory(assign, q, mem, devs, mp, head_at=head_at)
    reg = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "load-mode A/B: a Pi 4 host's weights mapped (every cell so far) against read into memory (-lm none)",
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()},
                             **{f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "run_pool.sh", "compare_pool.py",
                                                                         os.path.basename(__file__))}},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "n_ctx_planned": N_CTX, "protocol": "-t 4 -p 64 -n 16 -r 3 -oe jsonl -v",
           "run_order": ["lm.q3_14b.mmap", "lm.q3_14b.none", "lm.q3_14b.mmap", "lm.q3_14b.none"], "configs": {},
           "hypotheses": [
               {"id": "H-LM-COLD", "claim": "in both none cells every scored test's repetitions agree within 5 % (max/min), "
                                            "and compare_pool flags no page-in"},
               {"id": "H-LM-SPEED", "claim": "none's decode ms/token within +-3 % of mmap's (means over both runs); none's "
                                             "prefill within +-3 % of mmap's fastest repetition"}]}
    for cid, extra in (("lm.q3_14b.mmap", ""), ("lm.q3_14b.none", " -lm none")):
        bench = pp.bench_args(assign, KEY, head_at=head_at)
        bench["args"] += extra
        reg["configs"][cid] = {
            "model": shape, "quant": quant, "why": "load-mode A/B: %s" % ("weights memory-mapped (the default)" if not extra
                                                                          else "weights read into memory (-lm none)"),
            "pipeline": [{"node": n, "blocks": k} for n, k in assign], "head_on": HOST, "head_at": head_at,
            "bench": bench, "memory": m,
            "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                          "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2)},
            "cost_sources": {n: src[n] for n, _ in assign}}
    for cid in ("lm.q3_14b.mmap", "lm.q3_14b.none"):
        c = reg["configs"][cid]
        print("%-16s %s  decode %.1f  prefill %.1f  margins %s  %s" % (cid, " + ".join("%s %d" % a for a in assign), dec, pre,
                                                                      {n: v["margin_mib"] for n, v in m.items()}, c["bench"]["args"]))
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
