#!/usr/bin/env python3
"""Register jmoe (2026-11-03; the user: "... then ... Measure the Jetson's prefill speed on mixture-of-experts models"): what
the Jetson's GPU costs to prefill MoE blocks, against what the planner prices.

    python3 testbed/scripts/register_jmoe_20261103.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

In q9b the request objective moved five of Qwen3-30B-A3B Q6_K's blocks from the Pi 5 to the Jetson to prefill sooner. It
predicted the first token 3.50 s sooner and measured 0.90 s. The planner prices the Jetson's prefill from TinyLlama's Vulkan
rows, a dense model. Here q9b's plans and two between them are benched, the laptop's 36 blocks fixed and the Jetson taking k
of the Pi 5's 12 (head on the laptop, `-ot ^output.weight=CPU`, `-lm none`, as predict_pool.bench_args builds them):
  J0  laptop 36 + Pi 5 12 (q9b's D)
  J1  laptop 36 + Pi 5 11 + Jetson 1
  J3  laptop 36 + Pi 5 9 + Jetson 3
  J5  laptop 36 + Pi 5 7 + Jetson 5 (q9b's R)
llama-bench -t 4 -p 512 -n 16 -r 3, order J0 J5 J1 J3 J0 (J0 twice for drift), run by lc4_20261024.py: its gates (a quiet
laptop, the pin, the model's sha256 on the laptop, the Pis cool), the RPC servers, the Jetson's guard and locked clocks, a
throttle poll on the Pi 5. lc4 scores decode (score.json); score_jmoe_20261103.py scores prefill (score_prefill.json).
Predicted per bench: the planner's (PLANNER_V4) prefill per token x 512 and decode per token.
Hypotheses (registered, prefill):
  JM-PRE   every bench's pp512 within 10 % of the planner's prediction
  JM-SHIFT the measured pp512 gain from J0 to J5 at least half the planner's predicted gain
Reported, not tested: the fit of pp512 against the Jetson's k (the cost of a block moved from the Pi 5 to the Jetson, and the
step from k 0 to k 1), J0's drift, and decode per bench.
Writes testbed/dataset/jmoe_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "jmoe_predictions_20261103.json")
KEY, CTX = "qwen3moe_q6k", 4096
PLANS = {"J0": [("laptop", 36), ("pi5", 12)], "J1": [("laptop", 36), ("pi5", 11), ("jetson", 1)],
         "J3": [("laptop", 36), ("pi5", 9), ("jetson", 3)], "J5": [("laptop", 36), ("pi5", 7), ("jetson", 5)]}
SERVERS = {"pi5": {"node": "pi5", "ip": "192.168.10.11", "port": 50052, "extra": "-t 4"},
           "jetson": {"node": "jetson", "ip": "192.168.1.114", "port": 50052, "extra": "-d Vulkan0"}}
BENCH = " -t 4 -p 512 -n 16 -r 3"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    path, msha = Q.MODEL_FILE[KEY]
    benches = {}
    for label, assign in PLANS.items():
        r = P.plan_fixed(KEY, assign, "host", CTX, mem, load_mode="none", **P.PLANNER_V4)
        args = P.bench_args(assign, KEY, head_at=0, load_mode="none")["args"]
        assert args.startswith("-m %s " % path), args
        for w, _ in assign[1:]:
            assert SERVERS[w]["ip"] + ":" + str(SERVERS[w]["port"]) == P.ENDPOINT[w]
        benches[label] = {"host": "laptop", "model_path": path, "model_sha256": msha, "args": args[len("-m %s " % path):] + BENCH,
                          "assign": [list(x) for x in assign], "head_at": 0, "servers": [SERVERS[w] for w, _ in assign[1:]],
                          "predicted": {"pp512_ms": round(512 * r["prefill_ms"], 1), "prefill_ms_per_token": round(r["prefill_ms"], 3),
                                        "ms_per_token_by_depth": {"0": round(r["decode_ms"], 2)}}}
    gain = benches["J0"]["predicted"]["pp512_ms"] - benches["J5"]["predicted"]["pp512_ms"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_jmoe_20261103.py", "runner": "testbed/scripts/lc4_20261024.py",
           "scorer": "testbed/scripts/score_jmoe_20261103.py", "out_prefix": "jmoe",
           "what": "the Jetson's GPU prefilling Qwen3-30B-A3B Q6_K's MoE blocks as a worker, against the planner's price (q9b's miss)",
           "model": KEY, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "cold_c": {"pi5": 50, "pi4": 55},
           "benches": benches, "order": [["J0"], ["J5"], ["J1"], ["J3"], ["J0"]],
           "jetson_k": {"J0": 0, "J1": 1, "J3": 3, "J5": 5}, "predicted_gain_J0_J5_ms": round(gain, 1),
           "hypotheses": {},          # lc4's own (decode) hypotheses: none; the prefill ones below are score_jmoe's
           "prefill_hypotheses": {
               "JM-PRE": {"tol": 0.10, "claim": "every bench's pp512 within 10 % of the planner's prediction"},
               "JM-SHIFT": {"min_share": 0.5, "claim": "the measured pp512 gain from J0 to J5 at least half the predicted %.0f ms" % gain}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/register_jmoe_20261103.py", "testbed/scripts/score_jmoe_20261103.py", "testbed/scripts/lc4_20261024.py",
               "testbed/scripts/predict_pool.py", "testbed/scripts/throttle_poll.sh", "testbed/scripts/jetson_power_guard.sh",
               "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    for label, b in benches.items():
        print("%s %s: pp512 %.0f ms predicted (%.2f ms/token), decode %.1f ms; args %s" % (
            label, b["assign"], b["predicted"]["pp512_ms"], b["predicted"]["prefill_ms_per_token"],
            b["predicted"]["ms_per_token_by_depth"]["0"], b["args"]))
    print("predicted pp512 gain J0 -> J5: %.0f ms" % gain)
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
