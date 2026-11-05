#!/usr/bin/env python3
"""Register the KV term's discriminating test on the served Qwen3-14B (2026-10-10, after kv_depth_20261010 failed). Writes,
read-only, testbed/dataset/kv14_predictions_20261010.json: a run_pool.sh cell plus two competing predictions.

    python3 testbed/scripts/register_kv14_20261010.py [--dry-run]

What kv_depth_20261010 found (Qwen3-0.6B whole on each board, testbed/results/kv_depth_20261010_193538/score.json):
- a KV byte costs 1.00 ns on the Pi 4's CPU and 0.070 ns on the Jetson's GPU, not a weight byte's 0.273 and 0.134;
- the 0.6B has 2 query heads per KV head (16 / 8). Qwen3-14B has 5 (40 / 8). Which one the extra cost follows decides
  the 14B's long-context speed, and the 0.6B cannot tell. Two models, fixed here before any 14B data:
  A  the cost follows the KV bytes:     slope = 1000 x 4,096 x (27 x c_kv(pi4) + 13 x c_kv(jetson))
  B  on the Pi 4 it follows the query heads (attention is compute-bound on the A72); on the Jetson, the KV bytes:
                                        slope = 1000 x 4,096 x (27 x c_kv(pi4) x 40/16 + 13 x c_kv(jetson))
  c_kv(b) = the 0.6B's measured slope on b / (28 layers x 4,096 bytes).
The cell: the served plan, Pi 4 27 + Jetson 13 with the head on the host, `-lm none` as serve_pool.sh serves.
run_pool.sh runs `llama-bench -t 4 -p 0 -n 16 -d 0,1024 -r 3`; kv14_20261010.py runs the cell three times.
Hypotheses:
  HK14-A  the measured slope (ms at d = 1,024 minus at d = 0, per 1,000 tokens, averaged over the three runs) within 15 %
          of A
  HK14-B  the same against B
At most one can hold: they differ by 2.3x. If neither holds, both mechanisms are wrong.
Reported: the morning's registered term (weight-byte cost, 38.3 ms per 1,000) against the measured slope, and each run's
Pi 4 thermal verdict.
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
from control import gguf                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "kv14_predictions_20261010.json")
KVD = os.path.join(ROOT, "testbed", "results", "kv_depth_20261010_193538", "score.json")
KVR = os.path.join(DS, "kv_depth_predictions_20261010.json")
CELL = "kv14.d1024"
TOL = 15.0
INPUTS = ["testbed/scripts/kv14_20261010.py", "testbed/scripts/run_pool.sh", "testbed/scripts/predict_pool.py",
          "testbed/results/kv_depth_20261010_193538/score.json", "testbed/dataset/kv_depth_predictions_20261010.json",
          "testbed/scripts/jetson_power_guard.sh", "testbed/scripts/throttle_poll.sh"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build():
    kvd, kvr = json.load(open(KVD)), json.load(open(KVR))
    small = kvr["model"]
    per_tok = small["n_layers"] * small["kv_bytes_per_layer_per_token"]          # 28 x 4,096 bytes per token of depth
    ckv = {b: kvd["boards"][b]["slope_ms_per_1k_mean"] / (1000.0 * per_tok) for b in ("pi4", "jetson")}   # ms per KV byte
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    q = shapes["qwen3-14b-q4_k_m"]
    kv = q["kv_bytes_per_layer_per_token"]
    s06 = gguf.shape(os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf"))
    small_heads = s06["n_head"] / s06["n_head_kv"]                                # 16 / 8 = 2
    gqa_ratio = (q["n_head"] / q["n_head_kv"]) / small_heads
    A = 1000 * kv * (27 * ckv["pi4"] + 13 * ckv["jetson"])
    B = 1000 * kv * (27 * ckv["pi4"] * gqa_ratio + 13 * ckv["jetson"])
    mem = json.load(open(os.path.join(DS, "memory_at_rest.json")))
    r = P.plan_fixed("qwen3_14b", [("pi4", 27), ("jetson", 13)], "host", 1040, mem, load_mode="none")
    m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
    memo = {n: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in x.items()} for n, x in m.items()}
    memo["jetson"]["reserve_mib"] = 450                      # as the 10-10 spec_pass cell on the same board
    memo["jetson"]["margin_mib"] = memo["jetson"]["available_mib"] - memo["jetson"]["need_mib"] - 450
    args = P.bench_args(r["assign"], "qwen3_14b", head_at=r["head_at"])["args"] + " -lm none"
    cell = {"pipeline": [{"node": "pi4", "blocks": 27}, {"node": "jetson", "blocks": 13}],
            "bench": {"host": "pi4", "tool": "llama-bench", "args": args, "protocol": "-t 4 -p 0 -n 16 -d 0,1024 -r 3 -oe jsonl -v"},
            "memory": memo}
    return {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "testbed/scripts/register_kv14_20261010.py",
        "question": "the KV term on the served 14B: does the Pi 4's attention cost follow the KV bytes (A) or the query heads (B)?",
        "configs": {CELL: cell}, "cell": CELL, "runs": 3,
        "kv_cost_ms_per_byte_from_0_6b": ckv, "query_heads_per_kv_head": {"qwen3-14b": q["n_head"] / q["n_head_kv"], "qwen3-0.6b": small_heads},
        "predictions": {"A_ms_per_1k": A, "B_ms_per_1k": B, "registered_weight_cost_term_ms_per_1k": kvr["served_14b"]["slope_ms_per_1k_tokens"]},
        "tolerance_pct": TOL,
        "hypotheses": {"HK14-A": "measured slope within 15 % of predictions.A_ms_per_1k",
                       "HK14-B": "measured slope within 15 % of predictions.B_ms_per_1k"},
        "inputs_sha256": {p: sha(os.path.join(ROOT, p)) for p in INPUTS if os.path.exists(os.path.join(ROOT, p))},
    }


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % OUT)
    reg = build()
    print(json.dumps({k: reg[k] for k in ("kv_cost_ms_per_byte_from_0_6b", "query_heads_per_kv_head", "predictions", "configs")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, 0o444)
    print("registered %s (%s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
