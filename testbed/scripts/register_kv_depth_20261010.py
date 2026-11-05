#!/usr/bin/env python3
"""Register scheduler upgrade 4's first term (2026-10-10): the KV-cache bytes read at the context depth. Writes,
read-only, testbed/dataset/kv_depth_predictions_20261010.json.

    python3 testbed/scripts/register_kv_depth_20261010.py [--dry-run]

The question (the survey's upgrade 4, "add the KV-cache bytes read at the current context depth"). The predictor prices
a decoded token by the weight bytes each board reads. At depth d every layer also reads its K and V for the d earlier
positions, which costs nothing in today's model. At a 4,096-token context that is about 7.8 % more bytes for Qwen3-14B.

The term, fixed before any data:
  extra ms per decoded token at depth d = n_layers(b) x kv_bytes_per_layer_per_token x d x c(b)
- kv_bytes_per_layer_per_token = 2 (K and V) x n_head_kv x head_dim x 2 bytes (F16, llama.cpp's default cache);
- c(b) = the board's decode cost per byte, the predictor's own (predict_pool.costs at the model's quant). The KV cache
  is assumed to cost what a weight byte costs.
So the slope against d is n_layers x kv_bytes x c(b) per token of depth.

Fresh cells (none of these has ever been run):
- Qwen3-0.6B Q8_0 (28 layers, 4,096 KV bytes per layer per token), whole on one board:
  - the Pi 4 (CPU, -t 4);
  - the Jetson (Vulkan, -ngl 99, clocks locked as every run_pool.sh cell).
- llama-bench -p 0 -n 32 -d 0,1024,2048,4096 -r 3, three invocations per board (llama-bench caches each depth's state,
  so each depth is filled once per invocation). The flash-attention setting is llama-bench's default, as in serving, and
  is recorded from its output.

Hypotheses, decided here before any data:
  HK-PI4  the Pi 4's slope (least squares of ms per decoded token against d, per invocation, then averaged) within 15 %
          of predicted
  HK-JET  the Jetson's, the same
Upgrade 4's KV term holds when both do.
Reported, not judged:
- the spread of the slope across the three invocations;
- the served 14B plan's predicted slope (Pi 4 27 + Jetson 13): the term the planner would add at a 4,096-token
  context.
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as pp                                   # noqa: E402
from control import gguf                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "kv_depth_predictions_20261010.json")
MODEL_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf")
DEPTHS = [0, 1024, 2048, 4096]
TOL_PCT = 15.0
INPUTS = ["testbed/scripts/kv_depth_20261010.py", "testbed/scripts/predict_pool.py", "testbed/dataset/characterisation.csv",
          "testbed/dataset/model_shapes.json", "testbed/scripts/jetson_power_guard.sh", "testbed/scripts/throttle_poll.sh",
          "testbed/scripts/power_modes_char_20261010.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build():
    R = pp.rows(os.path.join(DS, "characterisation.csv"), "char")
    shapes = json.load(open(os.path.join(DS, "model_shapes.json")))
    small = gguf.shape(MODEL_LOCAL)
    kv = 2 * small["n_head_kv"] * small["head_dim"] * 2
    assert kv == small["kv_bytes_per_layer_per_token"], (kv, small["kv_bytes_per_layer_per_token"])
    C8, src8 = pp.costs(R, shapes, "Q8_0")
    pred = {b: {"cost_ms_per_byte": C8[b]["host"][0], "cost_source": src8[b]["host"],
                "slope_ms_per_1k_tokens": small["n_layers"] * kv * 1000 * C8[b]["host"][0]} for b in ("pi4", "jetson")}
    q14 = shapes["qwen3-14b-q4_k_m"]
    C4, _ = pp.costs(R, shapes, "Q4_K_M")
    kv14 = q14["kv_bytes_per_layer_per_token"]
    served = {"plan": "pi4 27 (host) + jetson 13 (worker), head on the host", "kv_bytes_per_layer_per_token": kv14,
              "slope_ms_per_1k_tokens": 1000 * kv14 * (27 * C4["pi4"]["host"][0] + 13 * C4["jetson"]["worker"][0])}
    served["extra_ms_at_4096"] = served["slope_ms_per_1k_tokens"] * 4.096
    return {
        "registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "testbed/scripts/register_kv_depth_20261010.py",
        "question": "scheduler upgrade 4, the KV term: decode cost grows with the context depth by the KV bytes read",
        "model": {"file": "~/models/qwen3-0.6b-q8_0.gguf", "sha256": sha(MODEL_LOCAL), "n_layers": small["n_layers"],
                  "n_head_kv": small["n_head_kv"], "head_dim": small["head_dim"], "kv_bytes_per_layer_per_token": kv},
        "bench": {"args": "-p 0 -n 32 -d %s -r 3" % ",".join(map(str, DEPTHS)), "depths": DEPTHS, "invocations": 3,
                  "pi4": "-t 4 (CPU)", "jetson": "-ngl 99 (Vulkan0), clocks locked"},
        "predictions": pred, "served_14b": served, "tolerance_pct": TOL_PCT,
        "hypotheses": {"HK-PI4": "the Pi 4's mean slope over three invocations within 15 % of predictions.pi4",
                       "HK-JET": "the Jetson's, within 15 % of predictions.jetson",
                       "verdict": "the KV term holds when HK-PI4 and HK-JET hold"},
        "inputs_sha256": {p: sha(os.path.join(ROOT, p)) for p in INPUTS if os.path.exists(os.path.join(ROOT, p))},
    }


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % OUT)
    reg = build()
    print(json.dumps({k: reg[k] for k in ("model", "predictions", "served_14b")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, 0o444)
    print("registered %s (%s)" % (os.path.relpath(OUT, ROOT), sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
