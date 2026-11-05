#!/usr/bin/env python3
"""Register dl1 (2026-11-03; the user: "... A depth law that bends, so decode can be priced at real context depths"): the bent
depth law (attention_cost_20261103.json, attention_bend_20261103.py) on fresh depths and a dense model.

    python3 testbed/scripts/register_dl1_20261103.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

PLANNER_V4's depth law is a straight line per CPU board; the cost of a token of depth grows with depth (lc4charb). The bent law,
a (1 + d / d1) per query-head dimension per block per token of depth on the laptop and the Pi 5, was fitted on lc4charb's
TinyLlama curves alone. After the fact, on data it was not fitted on:
- v4d's 30B at 4,096: the depth term within 6.1 and 6.6 % (V4: 52 % high);
- q9b's TPOT at ~544: within 4.9 and 0.5 % (V4: 26 and 17 % high);
- depth-priced planning picks the laptop-alone plan v4d measured faster (V4 picked the slower one).
Here, at depths neither fit nor check used, on three laptop-hosted plans:
  A  Qwen3-30B-A3B Q3_K_M, the laptop alone (v4d's A), depths 0, 1,024, 2,048
  B  the same, laptop 40 + Jetson 8, head on the laptop (v4d's B), depths 0, 1,024, 2,048
  C  Qwen3-14B Q4_K_M (dense), the laptop alone, depths 0, 512, 1,024 (llama-bench prefills each repetition's depth, and the
     laptop prefills this model at ~190 ms a token)
llama-bench -t 4 -p 0 -n 32 -d <depths> -r 3, order A B C, by lc4_20261024.py (its gates, servers, the Jetson's procedure).
The Pi 5's bend is not benched here: its network driver stalled pooled serves on 3 Nov (testbed/results/pi5_tx_stall_20261103),
and it waits for that fix. Predicted per bench and depth: PLANNER_V4's decode with ATTENTION_V5 (the bent law); V4's straight
prediction is recorded beside it, not tested.
Hypotheses (registered, lc4's types):
  DL-SLOPE  every bench's slope (ms per 1,000 tokens of depth, least squares over its depths) within 15 % of the bent law's
  DL-PICK   at 2,048 tokens of depth, the plan the bent law prices faster of A and B is faster
  DL-ACC    A, B and C within 15 % of their predictions at every depth
Writes testbed/dataset/dl1_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "dl1_predictions_20261103.json")
CTX = 4096
SERVER = {"pi5": ("192.168.10.11", 50052, "-t 4"), "jetson": ("192.168.1.114", 50052, "-d Vulkan0")}
PLANS = {"A": ("qwen3moe", [("laptop", 48)], None, (0, 1024, 2048)),
         "B": ("qwen3moe", [("laptop", 40), ("jetson", 8)], 0, (0, 1024, 2048)),
         "C": ("qwen3_14b", [("laptop", 40)], None, (0, 512, 1024))}
V5 = dict(P.PLANNER_V4, attn_table=P.ATTENTION_V5)


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def slope(pred):
    pts = [(int(d), v) for d, v in pred.items()]
    md = statistics.fmean(d for d, _ in pts); mv = statistics.fmean(v for _, v in pts)
    return 1000 * sum((d - md) * (v - mv) for d, v in pts) / sum((d - md) ** 2 for d, _ in pts)


def bench(key, assign, head_at, mem, depths):
    BENCH = "-t 4 -p 0 -n 32 -d %s -r 3" % ",".join(str(d) for d in depths)
    head = "host" if head_at == 0 else "last"
    pr = lambda d, pl: round(P.plan_fixed(key, assign, head, CTX, mem, load_mode="none", depth=d, **pl)["decode_ms"], 2)
    p5 = {str(d): pr(d, V5) for d in depths}
    p4 = {str(d): pr(d, P.PLANNER_V4) for d in depths}
    if len(assign) == 1:
        args = "-ngl 0 " + BENCH
    else:
        a = P.bench_args(assign, key, head_at=head_at)["args"]            # "-m <path> -rpc <endpoints> -ngl N [-ts ...] [-ot ...]"
        assert a.startswith("-m %s " % Q.MODEL_FILE[key][0]), a
        args = a.split(" ", 2)[2] + " " + BENCH                            # without "-m <path>": the runner adds the model
    return {"host": "laptop", "model": key, "model_path": Q.MODEL_FILE[key][0], "model_sha256": Q.MODEL_FILE[key][1], "args": args,
            "assign": [list(x) for x in assign], "head_at": head_at,
            "servers": [{"node": n, "ip": SERVER[n][0], "port": SERVER[n][1], "extra": SERVER[n][2]} for n, _ in assign[1:]],
            "predicted": {"ms_per_token_by_depth": p5, "slope_ms_per_1k": round(slope(p5), 3)},
            "predicted_v4_straight": {"ms_per_token_by_depth": p4, "slope_ms_per_1k": round(slope(p4), 3)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    benches = {k: bench(key, assign, h, mem, depths) for k, (key, assign, h, depths) in PLANS.items()}
    pa, pb = benches["A"]["predicted"]["ms_per_token_by_depth"]["2048"], benches["B"]["predicted"]["ms_per_token_by_depth"]["2048"]
    fast, slow = ("A", "B") if pa < pb else ("B", "A")
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_dl1_20261103.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "dl1",
           "what": "the bent depth law (ATTENTION_V5) on fresh depths (512-2,048) and a dense model",
           "follows": "v4d (testbed/dataset/v4d_predictions_20261101.json; the straight law 52 % high at 4,096)",
           "ctx": CTX, "cold_c": {"pi5": 50, "pi4": 55}, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "benches": benches, "order": [["A"], ["B"], ["C"]],
           "hypotheses": {
               "DL-SLOPE": {"type": "slope_within", "labels": ["A", "B", "C"], "tol": 0.15,
                            "claim": "every bench's slope within 15 % of the bent law's"},
               "DL-PICK": {"type": "faster_at", "a": fast, "b": slow, "depth": 2048,
                           "claim": "at 2,048 tokens of depth, %s is faster than %s (bent law: %.1f against %.1f ms)" % (
                               fast, slow, min(pa, pb), max(pa, pb))},
               "DL-ACC": {"type": "acc_at", "labels": ["A", "B", "C"], "tol": 0.15,
                          "claim": "A, B and C within 15 % of their predictions at every depth"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_dl1_20261103.py", "testbed/scripts/predict_pool.py",
               "testbed/scripts/attention_bend_20261103.py", "testbed/dataset/attention_cost_20261103.json",
               "testbed/dataset/attention_cost_20261101.json", "testbed/dataset/hybrid_shapes_20261101.json",
               "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv", "testbed/dataset/link.json",
               "ThinkPad/config/cluster.json", "testbed/scripts/throttle_poll.sh", "testbed/scripts/jetson_power_guard.sh",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    for k, b in benches.items():
        print("%s %s: bent %s (slope %.1f per 1k); straight %s (slope %.1f); args %s" % (
            k, b["assign"], b["predicted"]["ms_per_token_by_depth"], b["predicted"]["slope_ms_per_1k"],
            b["predicted_v4_straight"]["ms_per_token_by_depth"], b["predicted_v4_straight"]["slope_ms_per_1k"], b["args"]))
    print("DL-PICK: %s faster than %s at 2,048 (bent law)" % (fast, slow))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
