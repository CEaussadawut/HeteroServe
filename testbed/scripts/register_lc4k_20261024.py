#!/usr/bin/env python3
"""Register a repeat of lc4charb's Jetson bench (study lc4k, 2026-10-24; the user: "continue what's left for project 2"),
before it runs. Run and scored by lc4_20261024.py.

    python3 testbed/scripts/register_lc4k_20261024.py [--dry-run]

Why: lc4j (testbed/results/lc4j_20261024_200720) found the Jetson's 11 October KV-byte law holding on its own model again
(Qwen3-0.6B: 8.18 ms per 1,000 tokens against 8.057, +1.5 %; the series smooth: 92.1, 109.4, 125.8, 159.3 ms), and the
query-head law calibrated on TinyLlama's slope failing (+82.5 %). The TinyLlama slope it was calibrated on, lc4charb's 3.522
ms per 1,000 tokens (2.2x the KV-byte law's 1.583), came from a series that is not monotonic: 86.87, 112.48, 104.80, 121.46 ms
at 0, 2,048, 4,096 and 8,192 tokens. If that series repeats, TinyLlama's depth cost on the Jetson's GPU is a property of the
board and the model, and the KV-byte law does not carry to it; if not, lc4charb's Jetson bench was disturbed, and the law may.
The bench is lc4charb's own, alone this time (lc4charb ran three nodes' benches at once; they share nothing but the laptop
that started them): TinyLlama Q4_K_M, llama-bench -ngl 99 -p 0 -n 32 -d 0,2048,4096,8192 -r 3, Vulkan, the clocks locked,
the guard live (lc4_20261024.py's gates).
  K-REPEAT  every depth within 3 % of lc4charb's Jetson figure
  K-KV      the slope within 15 % of the KV-byte law's 1.583 ms per 1,000 tokens
  K-SAME    the slope within 15 % of lc4charb's 3.522
K-KV and K-SAME cannot both hold. Writes testbed/dataset/lc4k_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(ROOT, "testbed", "dataset", "lc4k_predictions_20261024.json")
SRC_REG = "testbed/dataset/lc4charb_predictions_20261024.json"
SRC_RUN = "testbed/results/lc4charb_20261024_065032/score.json"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dry-run", action="store_true"); a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    src = json.load(open(os.path.join(ROOT, SRC_REG)))["benches"]["jet"]
    was = json.load(open(os.path.join(ROOT, SRC_RUN)))["benches"]["jet"]
    law = src["predicted"]["slope_ms_per_1k"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_lc4k_20261024.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "lc4k",
           "what": "lc4charb's Jetson bench repeated alone: TinyLlama Q4_K_M whole on the Jetson's GPU, decode against context "
                   "depth 0-8,192; does its non-monotonic series repeat, and which slope does it give",
           "model": "tinyllama", "cold_c": {"pi5": 50, "pi4": 55},
           "benches": {"jt": {"host": "jetson", "model_path": src["model_path"], "model_sha256": src["model_sha256"],
                              "args": src["args"],
                              "predicted": {"ms_per_token_by_depth": {str(k): v for k, v in was["ms_per_token_by_depth"].items()},
                                            "slope_ms_per_1k": was["slope_ms_per_1k"], "kv_law_slope_ms_per_1k": law}}},
           "order": [["jt"]],
           "hypotheses": {
               "K-REPEAT": {"type": "acc_at", "labels": ["jt"], "tol": 0.03,
                            "claim": "every depth within 3 %% of lc4charb's Jetson figure (%s ms)"
                                     % ", ".join("%.2f" % v for v in was["ms_per_token_by_depth"].values())},
               "K-KV": {"type": "slope_near", "label": "jt", "value": law, "tol": 0.15,
                        "claim": "the slope within 15 %% of the KV-byte law's %.3f ms per 1,000 tokens" % law},
               "K-SAME": {"type": "slope_near", "label": "jt", "value": was["slope_ms_per_1k"], "tol": 0.15,
                          "claim": "the slope within 15 %% of lc4charb's %.3f ms per 1,000 tokens" % was["slope_ms_per_1k"]}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_lc4k_20261024.py", "testbed/scripts/jetson_power_guard.sh",
               "testbed/scripts/throttle_poll.sh", SRC_REG, SRC_RUN, "testbed/results/lc4j_20261024_200720/score.json",
               "ThinkPad/config/cluster.json")}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("benches", "hypotheses")}, indent=1)); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
