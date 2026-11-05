#!/usr/bin/env python3
"""Register the Jetson's attention law, tested between two candidates (study lc4j, 2026-10-24; the user: "continue what's left
for project 2"), before it runs. Run and scored by lc4_20261024.py (its "slope_near" type).

    python3 testbed/scripts/register_lc4j_20261024.py [--dry-run]

lc4charb (24 Oct) measured TinyLlama's decode on the Jetson's GPU growing 3.522 ms per 1,000 tokens of context, 2.2x what
the board's law predicted: the KV-byte law (attention_cost_20261011b.json: 0.070251 ns per KV byte, fitted on Qwen3-0.6B
whole on the Jetson, 11 Oct). Two explanations predict very different slopes for Qwen3-0.6B Q8_0, which has the SAME query-head
width per layer as TinyLlama (16 x 128 = 32 x 64 = 2,048) but 4x its KV bytes (4,096 B per layer per token, against 1,024):
  J-QHEAD  the cost follows the query heads (as on the laptop's CPU), calibrated on today's TinyLlama: 3.522 / 22 x 28 =
           4.483 ms per 1,000 tokens (within 15 %)
  J-KV     the cost follows the KV bytes, the 11 October law: 4,096 x 0.070251 ns x 28 = 8.057 ms per 1,000 tokens (within 15 %)
At most one can hold. The bench is lc4charb's on the Jetson with the other model: llama-bench -ngl 99 -p 0 -n 32
-d 0,2048,4096,8192 -r 3, Vulkan, the clocks locked, the guard live (lc4_20261024.py's gates).
Writes testbed/dataset/lc4j_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
from control import gguf  # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "lc4j_predictions_20261024.json")
MODEL_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf")
MODEL_SHA = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"      # testbed/models.md
TINY_LAYERS, TINY_QDIMS = 22, 32 * 64      # TinyLlama 1.1B: 22 layers, 32 query heads x 64


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    tiny = json.load(open(os.path.join(ROOT, "testbed", "results", "lc4charb_20261024_065032", "score.json")))["benches"]["jet"]["slope_ms_per_1k"]
    law = json.load(open(os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261011b.json")))["boards"]["jetson"]
    q = gguf.shape(MODEL_LOCAL)
    qd = q["n_head"] * q["head_dim"]
    v_qhead = round(tiny / TINY_LAYERS * q["n_layers"] * qd / TINY_QDIMS, 3)
    v_kv = round(q["kv_bytes_per_layer_per_token"] * law["ns_per_kv_byte"] * 1e-6 * q["n_layers"] * 1000, 3)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_lc4j_20261024.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "lc4j",
           "what": "the Jetson's attention law between two candidates: Qwen3-0.6B Q8_0 whole on the Jetson's GPU, decode against "
                   "context depth 0-8,192, the slope tested against the query-head law (calibrated on TinyLlama today) and the "
                   "11 October KV-byte law",
           "model": "qwen3-0.6b", "cold_c": {"pi5": 50, "pi4": 55},
           "shape": {k: q[k] for k in ("n_layers", "n_head", "n_head_kv", "head_dim", "kv_bytes_per_layer_per_token")},
           "calibration": {"tinyllama_jetson_slope_ms_per_1k": tiny, "from": "testbed/results/lc4charb_20261024_065032/score.json",
                           "kv_law": law},
           "benches": {"jq": {"host": "jetson", "model_path": "~/models/qwen3-0.6b-q8_0.gguf", "model_sha256": MODEL_SHA,
                              "args": "-ngl 99 -p 0 -n 32 -d 0,2048,4096,8192 -r 3",
                              "predicted": {"slope_ms_per_1k": v_qhead, "alternative_slope_ms_per_1k": v_kv}}},
           "order": [["jq"]],
           "hypotheses": {
               "J-QHEAD": {"type": "slope_near", "label": "jq", "value": v_qhead, "tol": 0.15,
                           "claim": "the slope follows the query heads: %.3f ms per 1,000 tokens within 15 %%" % v_qhead},
               "J-KV": {"type": "slope_near", "label": "jq", "value": v_kv, "tol": 0.15,
                        "claim": "the slope follows the KV bytes (the 11 October law): %.3f ms per 1,000 tokens within 15 %%" % v_kv}},
           "inputs_sha256": {f: sha(os.path.join(ROOT, f)) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_lc4j_20261024.py", "testbed/scripts/jetson_power_guard.sh",
               "testbed/scripts/throttle_poll.sh", "testbed/dataset/attention_cost_20261011b.json",
               "testbed/results/lc4charb_20261024_065032/score.json", "ThinkPad/config/cluster.json")}}
    if sha(MODEL_LOCAL) != MODEL_SHA:
        raise SystemExit("refused: the laptop's Qwen3-0.6B is not the pinned bytes")
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("shape", "calibration", "benches", "hypotheses")}, indent=1)); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s): J-QHEAD %.3f, J-KV %.3f ms per 1,000 tokens" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16], v_qhead, v_kv))


if __name__ == "__main__":
    main()
