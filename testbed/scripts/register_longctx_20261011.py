#!/usr/bin/env python3
"""Register the long-context plan (2026-10-11; the user: "build the measured context-length cost into the planner
(registered), then try a long-context plan where the laptop holds some layers. The Jetson has no memory to take more").
Writes, read-only, testbed/dataset/longctx_predictions_20261011.json.

    python3 testbed/scripts/register_longctx_20261011.py [--dry-run] [--out PATH]

What is new in the planner (predict_pool.py, opt-in, so every earlier registration reproduces):
- the KV term: a decoded token at context depth d costs each block d x attn_ms(board) more (predict(depth=, attn=),
  profiles(depth=, attn=)). The laws are attention_cost_20261011.json's:
  - Pi 4: 1.894 ns per query-head dimension (from the 14B, both depths hot);
  - Jetson: 0.0703 ns per KV byte;
  - laptop: 0.714 ns per query-head dimension.
  So plan_optimal itself weighs attention when it splits;
- the laptop as an RPC worker ("laptop"; cluster.json t580): its worker row (laptop_worker_20261011.json; it never
  repacks as a worker), its measured link to the Pi 4 (2026-10-10 12:59, 933.9 Mbit/s, RTT 0.846 ms), and its memory
  (MemAvailable now less 8 GiB kept for the desktop).

The planner, host fixed on the Pi 4 (it holds the model file and serves today), boards Pi 4 + laptop + Jetson, priced
at RTT1, -lm none, its pick recorded at d = 0, 1,024, 2,048 and 4,096:
- the laptop is the fastest board per block (16.8 ms against the Jetson's 29.7 and the Pi 4's 58.4), so up to 2,048 the
  pick is Pi 4 1 + laptop 39;
- at 4,096 the laptop's attention (15 ms per block) makes the Jetson's cheaper attention (1.2) worth a visit, and the pick
  becomes Pi 4 1 + Jetson 13 + laptop 26. The term changes the plan, not only its price.
The plan tested is the pick at the planning depth 4,096, the serving default's context. It is benched as served, -lm none,
with `llama-bench -t 4 -p 0 -n 16 -d 4096,0 -r 3`: d = 0 decodes seconds after d = 4,096, at the same heat. Three runs.
Predicted at both depths with the term and, for contrast, without it.

Hypotheses (15 %: the laptop's worker row spread 10.4 % between runs, and it holds 26 of the plan's blocks):
  LC-0      at d = 0 the measured ms per token (mean of the three runs) within 15 % of predicted
  LC-4096   at d = 4,096, the same
  LC-SLOPE  the measured rise (4,096 against 0) within 25 % of predicted (the laptop's attention cell spread 24 %)
  LC-GAIN   at d = 0 the plan is faster than the served plan (Pi 4 27 + Jetson 13, measured hot in kv14b: 1,991.1 ms per
            token), by the predicted factor within 15 %
Reported: the served plan's predicted 4,096 against this plan's measured 4,096 (the served plan's own prefill to 4,096 on
the Pi 4 is about an hour a run, not run); the planner's pick at each depth.
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
import link_to_json as LJ                                   # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "longctx_predictions_20261011.json")
LINK_LAPTOP = os.path.join(ROOT, "testbed", "results", "spec_shape_20261010_125542", "link_s0.log")
KV14B = os.path.join(ROOT, "testbed", "results", "kv14_20261010_213514", "score.json")
LAPTOP_RESERVE_MIB = 8192
PLAN_DEPTH = 4096
DEPTHS = [0, 1024, 2048, 4096]
TOL, TOL_SLOPE = 15.0, 25.0
CELL = "lc.q3_14b.d4096"
INPUTS = ["testbed/scripts/predict_pool.py", "ThinkPad/control/placement.py", "testbed/scripts/longctx_20261011.py",
          "testbed/dataset/attention_cost_20261011.json", "testbed/dataset/laptop_worker_20261011.json",
          "testbed/dataset/characterisation.csv", "testbed/dataset/model_shapes.json", "testbed/dataset/link.json",
          "testbed/dataset/memory_at_rest.json", "testbed/results/spec_shape_20261010_125542/link_s0.log",
          "testbed/results/kv14_20261010_213514/score.json", "testbed/scripts/jetson_power_guard.sh",
          "testbed/scripts/throttle_poll.sh"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def inputs(laptop_avail_mib):
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    q = shapes["qwen3-14b-q4_k_m"]
    R = P.rows(os.path.join(DS, "characterisation.csv"), "char")
    lap = P.laptop_costs(R, shapes, "Q4_K_M", os.path.join(DS, "laptop_worker_20261011.json"))
    _, _, lk = LJ.parse(LINK_LAPTOP)
    pairs = {"pi4-laptop": {"mbps": lk["mbps"], "rtt_ms": lk["rtt_ms"]}}
    mem = json.load(open(os.path.join(DS, "memory_at_rest.json")))
    mem = dict(mem, laptop={"mem_available_mib": laptop_avail_mib - LAPTOP_RESERVE_MIB})
    T = P.attention_table()
    attn = {n: P.attn_ms(T, n, q) for n in ("pi4", "jetson", "laptop")}
    return q, lap, pairs, mem, attn


def plan(depth, mem, attn, lap, pairs, use_attn=True):
    return P.plan_host_choice("qwen3_14b", ["pi4", "laptop", "jetson"], depth + 16, mem, host="pi4", rtt_trips=1,
                              load_mode="none", depth=depth, attn=attn if use_attn else None, extra_costs={"laptop": lap},
                              extra_pairs=pairs)


def price(assign, head, depth, mem, attn, lap, pairs):
    return P.plan_fixed("qwen3_14b", assign, head, depth + 16, mem, rtt_trips=1, load_mode="none", depth=depth, attn=attn,
                        extra_costs={"laptop": lap}, extra_pairs=pairs)


def build(laptop_avail_mib):
    q, lap, pairs, mem, attn = inputs(laptop_avail_mib)
    picks = {}
    for d in DEPTHS:
        r = plan(d, mem, attn, lap, pairs)
        picks[str(d)] = {"assign": [list(a) for a in r["assign"]], "head": "host" if r["head_at"] == 0 else "last",
                         "decode_ms": round(r["decode_ms"], 2)}
    pick = plan(PLAN_DEPTH, mem, attn, lap, pairs)
    assign, head = pick["assign"], ("host" if pick["head_at"] == 0 else "last")
    pred, pred_noattn, served = {}, {}, {}
    for d in (0, PLAN_DEPTH):
        pred[str(d)] = price(assign, head, d, mem, attn, lap, pairs)["decode_ms"]
        pred_noattn[str(d)] = price(assign, head, d, mem, None, lap, pairs)["decode_ms"]
        served[str(d)] = price([("pi4", 27), ("jetson", 13)], "host", d, mem, attn, lap, pairs)["decode_ms"]
    memo = P.memory(assign, q, mem, pick["devices"], pick["model"], head_at=pick["head_at"])
    if any(m["margin_mib"] < 0 for m in memo.values()):
        raise SystemExit("the pick does not fit: %s" % memo)
    args = P.bench_args(assign, "qwen3_14b", head_at=pick["head_at"])["args"] + " -lm none"
    kv14b = json.load(open(KV14B))
    served_d0 = sum(r["d0"] for r in kv14b["per_run_ms"]) / len(kv14b["per_run_ms"])
    return {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "testbed/scripts/register_longctx_20261011.py",
        "question": "the planner with the measured context-length cost, and the laptop as a worker: its long-context plan",
        "model": "qwen3_14b", "host": "pi4", "boards": ["pi4", "laptop", "jetson"], "rtt_trips": 1, "load_mode": "none",
        "attn_ms_per_block_per_depth_token": attn, "laptop_costs": lap, "extra_pairs": pairs,
        "laptop_mem_available_mib_at_registration": laptop_avail_mib, "laptop_reserve_mib": LAPTOP_RESERVE_MIB,
        "planner_picks_by_depth": picks, "plan_depth": PLAN_DEPTH,
        "plan": {"assign": [list(a) for a in assign], "head": head, "memory": memo},
        "cell": CELL, "bench": {"host": "pi4", "args": args, "protocol": "-t 4 -p 0 -n 16 -d %d,0 -r 3 -oe jsonl" % PLAN_DEPTH,
                                "workers": {"jetson": "ggml-rpc-server -H 192.168.1.114 -p 50052 -d Vulkan0 (serve_pool.sh's)",
                                            "laptop": "ggml-rpc-server -H 192.168.1.108 -p 50053 -t 4 (the characterised worker)"}},
        "runs": 3,
        "predictions_ms": pred, "predictions_without_the_term_ms": pred_noattn, "served_plan_predicted_ms": served,
        "served_plan_measured_d0_ms_kv14b": round(served_d0, 2),
        "predicted_gain_d0": round(served_d0 / pred["0"], 4),
        "tolerance_pct": TOL, "tolerance_slope_pct": TOL_SLOPE,
        "hypotheses": {
            "LC-0": "measured ms per token at d = 0 (mean of the runs) within 15 % of predictions_ms.0",
            "LC-4096": "the same at d = 4,096 against predictions_ms.4096",
            "LC-SLOPE": "the measured rise (4,096 - 0) within 25 % of the predicted rise",
            "LC-GAIN": "served_plan_measured_d0_ms_kv14b / measured d = 0 within 15 % of predicted_gain_d0"},
        "inputs_sha256": {p: sha(os.path.join(ROOT, p)) for p in INPUTS if os.path.exists(os.path.join(ROOT, p))},
    }


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else OUT
    if os.path.exists(out) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % out)
    avail = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable"))) // 1024
    reg = build(avail)
    print(json.dumps({k: reg[k] for k in ("planner_picks_by_depth", "plan", "bench", "predictions_ms",
                                          "predictions_without_the_term_ms", "served_plan_predicted_ms",
                                          "served_plan_measured_d0_ms_kv14b", "predicted_gain_d0")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(out, "w"), indent=1)
    if out == OUT:
        os.chmod(out, 0o444)
    print("registered %s (%s)" % (os.path.relpath(out, ROOT), sha(out)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
