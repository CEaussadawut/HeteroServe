#!/usr/bin/env python3
"""Register the Jetson's attention law against a third model (study lc4m, 2026-10-25; the user: "complete everything lefts",
and "yes to the qwen download"), before it runs. Run and scored by lc4_20261024.py (its "slope_near" type).

    python3 testbed/scripts/register_lc4m_20261025.py [--dry-run]

lc4j and lc4k measured the Jetson GPU's decode slope against context depth on two models: Qwen3-0.6B Q8_0 8.18 ms per 1,000
tokens (28 layers; 16 query heads x 128, 8 KV heads: 2,048 query dims and 4,096 KV bytes per layer per token) and TinyLlama
Q4_K_M 4.221 (22 layers; 32 x 64, 4 KV heads: 2,048 query dims, 1,024 KV bytes). The KV-byte law of 11 October holds on the
first and under-prices the second 2.7x, so the cost is not KV bytes alone; two models cannot tell more.
Qwen3-4B Q4_K_M (official Qwen GGUF, revision bc640142, downloaded 2026-10-25 with the user's yes; 36 layers; 32 query heads x
128, 8 KV heads: 4,096 query dims and 4,096 KV bytes per layer per token) has Qwen3-0.6B's KV bytes per layer and twice its
query dims, so the candidate laws part:
  J4-KV   the cost follows the KV bytes (the 11 October law): 4,096 B x 0.070251 ns x 36 layers
  J4-QW   it follows the query dims, per layer as Qwen3-0.6B's (lc4j): 8.18 / 28 x (4,096 / 2,048) x 36
  J4-TWO  per layer a x query dims + b x KV bytes, a and b fitted exactly on lc4j's and lc4k's slopes
each within 10 % (their bands do not overlap; at most one can hold). The bench is lc4j's on the Jetson with this model,
Vulkan, the clocks locked, the guard live (lc4_20261024.py's gates): llama-bench -ngl 99 -ub 128 -p 0 -n 32
-d 0,256,512,768,1024 -r 3.
The depth range was set by hand smokes before this registration (single runs, -r 1, not reported; disclosed here):
  - -d 0,3072: the OOM killer stopped llama-bench at 3,072 (the weights are 2.32 GiB of the board's 4 GB);
  - -ub 128 (smaller prefill batches; decode, the timed part, is unchanged) -d 0,2048: it fits, but the decode collapsed from
    3.05 to 0.10 tokens/s at 2,048 -- memory pressure, not attention;
  - -ub 128 -d 0,512,1024,1536: 3.05, 2.95, 2.85 tokens/s, then 0.24 at 1,536. So the clean range is up to 1,024 tokens.
    Over it those single runs give roughly 22.5 ms per 1,000 tokens. The three predictions below are computed from lc4j's
    and lc4k's slopes only, as they were before any smoke; their tolerance is the one written before it.
Writes testbed/dataset/lc4m_predictions_20261025.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
from control import gguf  # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "lc4m_predictions_20261025.json")
MODEL_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-4B-Q4_K_M.gguf")
MODEL_SHA = "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5"      # the Hugging Face LFS oid at bc640142
LC4J, LC4K = "testbed/results/lc4j_20261024_200720/score.json", "testbed/results/lc4k_20261024_211512/score.json"
SHAPES = {"qwen3-0.6b": (28, 16 * 128, 4096), "tinyllama": (22, 32 * 64, 1024)}   # layers, query dims, KV bytes per layer


def sha(p):
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    s06 = json.load(open(os.path.join(ROOT, LC4J)))["benches"]["jq"]["slope_ms_per_1k"]
    stl = json.load(open(os.path.join(ROOT, LC4K)))["benches"]["jt"]["slope_ms_per_1k"]
    law = json.load(open(os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261011b.json")))["boards"]["jetson"]
    q = gguf.shape(MODEL_LOCAL)
    L, qd, kv = q["n_layers"], q["n_head"] * q["head_dim"], q["kv_bytes_per_layer_per_token"]
    # per layer, ms per 1,000 tokens of depth: the two measured models
    (l6, q6, k6), (lt, qt, kt) = SHAPES["qwen3-0.6b"], SHAPES["tinyllama"]
    p6, pt = s06 / l6, stl / lt
    # a x qd + b x kv through both points (equal query dims, so b from the KV difference, then a)
    b = (p6 - pt) / (k6 - kt)
    a_ = (pt - b * kt) / qt
    v_kv = round(kv * law["ns_per_kv_byte"] * 1e-6 * L * 1000, 3)
    v_qw = round(p6 * (qd / q6) * L, 3)
    v_two = round((a_ * qd + b * kv) * L, 3)
    hyps = {"J4-KV": (v_kv, "the cost follows the KV bytes (the 11 October law)"),
            "J4-QW": (v_qw, "it follows the query dims, per layer as Qwen3-0.6B's"),
            "J4-TWO": (v_two, "per layer a x query dims + b x KV bytes, fitted on lc4j and lc4k")}
    bands = sorted((v * 0.9, v * 1.1) for v, _ in hyps.values())
    if any(bands[i][1] >= bands[i + 1][0] for i in range(len(bands) - 1)):
        raise SystemExit("refused: the 10 %% bands overlap: %s" % hyps)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_lc4m_20261025.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "lc4m",
           "what": "the Jetson's attention law against a third model: Qwen3-4B Q4_K_M whole on the Jetson's GPU, decode against "
                   "context depth 0-1,024, the slope tested against three candidate laws",
           "model": "qwen3-4b", "cold_c": {"pi5": 50, "pi4": 55},
           "smokes_before_registration": {"d0,3072": "OOM-killed at 3,072", "ub128 d0,2048": "3.05 then 0.10 tokens/s",
                                          "ub128 d0,512,1024,1536": "3.05, 2.95, 2.85, 0.24 tokens/s (single runs, -r 1; "
                                                                    "roughly 22.5 ms per 1,000 tokens over 0-1,024)"},
           "shape": {k: q[k] for k in ("n_layers", "n_head", "n_head_kv", "head_dim", "kv_bytes_per_layer_per_token")},
           "calibration": {"qwen3_0.6b_slope_ms_per_1k": s06, "tinyllama_slope_ms_per_1k": stl, "kv_law": law,
                           "two_term": {"a_ms_per_1k_per_layer_per_query_dim": a_, "b_ms_per_1k_per_layer_per_kv_byte": b},
                           "from": [LC4J, LC4K, "testbed/dataset/attention_cost_20261011b.json"]},
           "benches": {"jq4": {"host": "jetson", "model_path": "~/models/qwen3-4b-q4_k_m.gguf", "model_sha256": MODEL_SHA,
                               "args": "-ngl 99 -ub 128 -p 0 -n 32 -d 0,256,512,768,1024 -r 3",
                               "predicted": {"slope_ms_per_1k": v_two, "kv_law": v_kv, "query_width": v_qw}}},
           "order": [["jq4"]],
           "hypotheses": {k: {"type": "slope_near", "label": "jq4", "value": v, "tol": 0.10,
                              "claim": "%s: %.3f ms per 1,000 tokens within 10 %%" % (c, v)} for k, (v, c) in hyps.items()},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_lc4m_20261025.py", "testbed/scripts/jetson_power_guard.sh",
               "testbed/scripts/throttle_poll.sh", "testbed/dataset/attention_cost_20261011b.json", LC4J, LC4K,
               "ThinkPad/config/cluster.json")}}
    if sha(MODEL_LOCAL) != MODEL_SHA:
        raise SystemExit("refused: the laptop's Qwen3-4B is not the pinned bytes")
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("shape", "calibration", "hypotheses")}, indent=1)); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s): %s" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16],
                                             ", ".join("%s %.3f" % (k, v) for k, (v, _) in hyps.items())))


if __name__ == "__main__":
    main()
