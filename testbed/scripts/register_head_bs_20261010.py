#!/usr/bin/env python3
"""Register scheduler upgrade 6 (2026-10-10: backend sampling on the board that holds the output head) BEFORE it runs.
Writes, read-only, testbed/dataset/head_bs_predictions_20261010.json.

    python3 testbed/scripts/register_head_bs_20261010.py [--dry-run] [--out PATH]
        --out writes the same registration elsewhere (a smoke run's plumbing; never the dataset's file)

The question (the survey's upgrade 6). When the output head sits on a worker, every token's logits (n_vocab x 4 bytes,
607,744 for Qwen3) cross the link back to the host, which then samples. llama.cpp's backend sampling
(the request field "backend_sampling", in the pin) can sample on the board that holds the logits. Then only the token
comes back. Does it work over RPC, and does it save exactly the logits' transfer?

Why the pin should allow it (read before any data):
- the RPC device reports every op as supported (ggml-rpc.cpp, ggml_backend_rpc_device_supports_op: "return true"), so
  the scheduler puts the sampling ops on the Jetson;
- the Jetson's Vulkan backend supports TOP_K, SOFT_MAX, CUMSUM and ARGMAX (ggml-vulkan.cpp supports_op);
- with a sampler on the sequence, the raw logits are copied out only when needed (llama-context.cpp, needs_raw_logits);
- the chain must be backend-capable end to end, or it stops at the first sampler that is not: so the requests name
  "samplers": ["top_k"] with top_k 1 (chain: top_k, then dist; both have backend versions), greedy in effect, in BOTH arms.

The setup: Qwen3-0.6B Q8_0 (on the Pi 4 since 10-10, sha256 pinned), Pi 4 14 layers + Jetson 14 layers + the head on the
Jetson. llama-server on the Pi 4 (the pinned build), ggml-rpc-server on the Jetson exactly as serve_pool.sh starts it.
- Why this model and not the 14B: with its head the 14B fits the Jetson only at 7 layers, which decodes slower than the
  served plan (the planner: 2,104 against 2,027 ms). Its ~2 s token would also bury a 5 ms saving in run-to-run noise.
  The 0.6B has the same vocabulary, so the same 607,744-byte logits, and a ~0.1 s token. serve_pool.sh plans only the
  models in predict_pool.MODELS, so this runner starts the two servers itself, with serve_pool.sh's commands and gates.
- Settings: s0 (unshaped gigabit), b100 and b50 (netem on the Pi 4, both directions, peer the Jetson, ports 50052,5202).
  Each setting's link is measured first, with nothing loaded (measure_link.sh, PAIR="pi4 jetson", port 5202).
- Arms, per request: backend_sampling false (off) and true (on). One server load; at each setting the four prompts run
  off, on, on, off (ABBA), 64 tokens each, greedy, cache_prompt false.

The model: the saving per token (off - on, the server's ms per token) is the logits' transfer, V / g.
- V = n_vocab x 4 = 607,744 bytes, from the GGUF.
- g = the measured goodput from the Jetson to the Pi 4 (bytes per ms; measure_link.sh's receiver-timed rate in that
  direction).
Nominal (payload) rates, for the record: s0 at 941 Mbit/s gives 5.17 ms; b100 at 95.6 gives 50.9; b50 at 47.8 gives 101.7.

Hypotheses, decided here before any data:
  HB-BYTES  at s0, the bytes the Jetson sends per decoded token fall by at least 0.9 V with the option on (its socket's
            bytes_acked, read on the Jetson around each request). This is the mechanism: the logits stop crossing.
  HB-SAVE   at each setting, the measured saving within 1 ms of V / g (the survey's test). Three tests.
  HB-SLOPE  the least-squares slope of the saving against 1 / g over the three settings within 10 % of V: the transfer
            removed is exactly the logits, whatever constant remains.
  HB-SAME   the greedy text of each prompt identical in both arms at every setting.
The option works over RPC when HB-BYTES and HB-SAME hold. Upgrade 6 holds when all six tests hold.
Reported, not judged: the constant left over (the intercept of the slope fit).
"""
import datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control import gguf                                    # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "head_bs_predictions_20261010.json")
SHAPE_REG = os.path.join(DS, "spec_shape_predictions_20261010.json")
MODEL_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf")
MODEL_ON_PI4 = "~/models/qwen3-0.6b-q8_0.gguf"
SETTINGS = [("s0", 0), ("b100", 100), ("b50", 50)]
PAYLOAD = 1448.0 / 1514.0                                   # netshape.sh: payload goodput = rate x 1448 / 1514
GIGABIT_PAYLOAD_MBIT = 941.0
INPUTS = ["testbed/scripts/head_bs_20261010.py", "testbed/scripts/netshape.sh", "testbed/scripts/measure_link.sh",
          "testbed/scripts/link_to_json.py", "testbed/scripts/jetson_power_guard.sh", "testbed/scripts/throttle_poll.sh",
          "testbed/scripts/power_modes_char_20261010.py"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def saving_ms(V, mbit):
    """V / g in ms, g in Mbit/s of goodput."""
    return V / (mbit * 1e6 / 8 / 1000.0)


def build():
    shp = gguf.shape(MODEL_LOCAL)
    n_vocab = int(round(shp["embd_params"] / shp["n_embd"]))
    V = 4 * n_vocab
    preg = json.load(open(SHAPE_REG))
    nominal = {s: (GIGABIT_PAYLOAD_MBIT if r == 0 else r * PAYLOAD) for s, r in SETTINGS}
    return {
        "registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "script": "testbed/scripts/register_head_bs_20261010.py",
        "question": "scheduler upgrade 6: sample on the board that holds the output head, so only the token crosses the link",
        "model_file": MODEL_ON_PI4, "model_sha256": sha(MODEL_LOCAL), "n_layers": shp["n_layers"], "n_embd": shp["n_embd"],
        "n_vocab": n_vocab, "V_bytes": V, "split": {"pi4": 14, "jetson": 14, "head": "jetson"},
        "server_args": "-m %s --rpc 192.168.1.114:50052 -ngl 15 -c 1024 -np 1 -t 4 -lv 4 -lm none --host 192.168.1.110 --port 8080"
                       % MODEL_ON_PI4,
        "host_url": preg["host_url"], "chat_format": preg["chat_format"], "warmup_prompt": preg["warmup_prompt"],
        "prompts": preg["prompts"], "n_predict": 64,
        "sampling": {"samplers": ["top_k"], "top_k": 1, "temperature": 0, "cache_prompt": False},
        "arms": {"off": {"backend_sampling": False}, "on": {"backend_sampling": True}},
        "abba": ["off", "on", "on", "off"],
        "settings": [{"id": s, "rate_mbit": r} for s, r in SETTINGS],
        "shaping": {"peer": "192.168.1.114", "ports": "50052,5202", "measure_port": 5202, "pair": "pi4 jetson"},
        "nominal_goodput_mbit": nominal,
        "predicted_saving_ms_nominal": {s: round(saving_ms(V, m), 3) for s, m in nominal.items()},
        "thresholds": {"bytes_fraction": 0.9, "save_ms": 1.0, "slope_pct": 10.0},
        "hypotheses": {
            "HB-BYTES": "at s0, (bytes_acked per decoded token off - on) on the Jetson's RPC socket >= 0.9 V",
            "HB-SAVE": "at each setting, |(ms per token off - on) - V / g| <= 1 ms, g = the measured Jetson -> Pi 4 goodput",
            "HB-SLOPE": "the least-squares slope of the saving against 1 / g over the three settings within 10 % of V",
            "HB-SAME": "each prompt's greedy text identical in both arms at every setting",
            "works_over_rpc": "HB-BYTES and HB-SAME hold", "verdict": "upgrade 6 holds when all six tests hold"},
        "inputs_sha256": {p: sha(os.path.join(ROOT, p)) for p in INPUTS if os.path.exists(os.path.join(ROOT, p))},
    }


def main(argv):
    out = argv[argv.index("--out") + 1] if "--out" in argv else OUT
    if os.path.exists(out) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % out)
    reg = build()
    print(json.dumps({k: reg[k] for k in ("n_vocab", "V_bytes", "split", "nominal_goodput_mbit", "predicted_saving_ms_nominal")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(out, "w"), indent=1, ensure_ascii=False)
    if out == OUT:
        os.chmod(out, 0o444)
    print("registered %s (%s)" % (os.path.relpath(out, ROOT), sha(out)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
