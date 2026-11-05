#!/usr/bin/env python3
"""Register v4d (2026-11-01; the user: "do item 1 to 4 then 6-7"): the scheduler review's item 4 -- context depth priced by
default -- tested as lc4planb tested the term on TinyLlama (2026-10-24), now on serve_pool.sh's default model at a moderate
depth, where the CPU laws are least sure.

    python3 testbed/scripts/register_v4d_20261101.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]
    python3 testbed/scripts/lc4_20261024.py run testbed/dataset/v4d_predictions_20261101.json

Qwen3-30B-A3B Q3_K_M (serve_pool.sh's default MODEL), two plans from the same snapshot, each a laptop-hosted llama-bench
(-p 0 -n 32 -d 0,4096 -r 3, order A B B A, lc4_20261024.py's runner: its gates, the Jetson's guard and locked clocks):
  A  PLANNER_V3's pick at ctx 8,192 (depth not priced): the laptop alone;
  B  PLANNER_V4's pick at ctx 8,192, priced at half the context, 4,096 tokens (serve_plan_20261101.plan_depth): part of the
     model on the Jetson's GPU, whose attention barely grows with depth (lc4n), the head on the laptop.
Each bench is predicted at depth 0 and 4,096 by PLANNER_V4 (attention_cost_20261101.json: the laptop's query-head law,
0.547 ns per query-head dimension, and the Jetson's fa_table). The laptop's law is a straight line through depths whose cost
curves upward (lc4charb: TinyLlama 30.0 -> 57.1 -> 100.4 -> 232.6 ms at 0 / 2,048 / 4,096 / 8,192), so at 4,096 it may
over-price the laptop -- the risk this test takes.
Hypotheses (registered):
  D4-SHORT  at depth 0, A is faster than B
  D4-LONG   at 4,096 tokens of depth, B is faster than A (the plan the depth term chose)
  D4-ACC    every bench within 15 % of its prediction at both depths (lc4planb's bar)
  D4-LAW    A's slope (ms per 1,000 tokens, least squares over its two depths) within 15 % of the laptop's law for this shape
Writes testbed/dataset/v4d_predictions_20261101.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
import serve_plan_20261101 as SP                            # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "v4d_predictions_20261101.json")
KEY, CTX, NODES = "qwen3moe", 8192, ["laptop", "pi5", "pi4", "jetson"]
DEPTHS = (0, 4096)
BENCH = "-t 4 -p 0 -n 32 -d 0,4096 -r 3"
SERVER = {"pi5": ("192.168.10.11", 50052, "-t 4"), "pi4": ("192.168.10.12", 50052, "-t 4"), "jetson": ("192.168.1.114", 50052, "-d Vulkan0")}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def bench(plan, mem):
    assign, head_at = [tuple(x) for x in plan["assign"]], plan["head_at"]
    head = "host" if head_at == 0 else "last"
    if assign[0][0] != "laptop":
        raise SystemExit("refused: the plan is not laptop-hosted: %s" % (assign,))
    pred = {str(d): round(P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", depth=d, **P.PLANNER_V4)["decode_ms"], 2)
            for d in DEPTHS}
    if len(assign) == 1:
        args = "-ngl 0 " + BENCH
    else:
        a = P.bench_args(assign, KEY, head_at=head_at)["args"]          # "-m <path> -rpc <endpoints> -ngl N [-ts ...] [-ot ...]"
        args = a.split(" ", 2)[2] + " " + BENCH                          # without "-m <path>": the runner adds the model
    return {"host": "laptop", "model_path": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "args": args,
            "assign": [list(x) for x in assign], "head_at": head_at,
            "servers": [{"node": n, "ip": SERVER[n][0], "port": SERVER[n][1], "extra": SERVER[n][2]} for n, _ in assign[1:]],
            "predicted": {"ms_per_token_by_depth": pred}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    depth = SP.plan_depth(CTX, {})
    pa = P.plan_host_choice(KEY, NODES, CTX, mem, load_mode="none", **P.PLANNER_V3)
    pb = P.plan_host_choice(KEY, NODES, CTX, mem, load_mode="none", depth=depth, **P.PLANNER_V4)
    if (pa["assign"], pa["head_at"]) == (pb["assign"], pb["head_at"]):
        raise SystemExit("refused: the depth term does not change the plan on this snapshot: %s" % (pa["assign"],))
    benches = {"A": bench(pa, mem), "B": bench(pb, mem)}
    q = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))[P.MODELS[KEY][0]]
    tab = P.attention_table(P.ATTENTION_V4)
    law_slope = 1000 * P.attn_ms(tab, "laptop", q) * q["n_layers"]       # ms per 1,000 tokens of depth, the laptop alone
    pa_, pb_ = benches["A"]["predicted"]["ms_per_token_by_depth"], benches["B"]["predicted"]["ms_per_token_by_depth"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_v4d_20261101.py", "runner": "testbed/scripts/lc4_20261024.py", "out_prefix": "v4d",
           "what": "the scheduler review's item 4: context depth priced by default (PLANNER_V4 at half the context), on serve_pool.sh's "
                   "default model at a moderate depth -- the plan without the term against the plan with it",
           "follows": "lc4planb (testbed/dataset/lc4planb_predictions_20261024.json; TinyLlama at 8,192, all three held)",
           "model": KEY, "ctx": CTX, "plan_depth": depth, "cold_c": {"pi5": 50, "pi4": 55},
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "laptop_law_slope_ms_per_1k": round(law_slope, 3),
           "benches": benches, "order": [["A"], ["B"], ["B"], ["A"]],
           "hypotheses": {
               "D4-SHORT": {"type": "faster_at", "a": "A", "b": "B", "depth": 0,
                            "claim": "at depth 0, A is faster than B (predicted %.1f against %.1f ms)" % (pa_["0"], pb_["0"])},
               "D4-LONG": {"type": "faster_at", "a": "B", "b": "A", "depth": 4096,
                           "claim": "at 4,096 tokens of depth, B is faster than A (predicted %.1f against %.1f ms)" % (pb_["4096"], pa_["4096"])},
               "D4-ACC": {"type": "acc_at", "labels": ["A", "B"], "tol": 0.15,
                          "claim": "every bench within 15 % of its prediction at both depths"},
               "D4-LAW": {"type": "slope_near", "label": "A", "value": round(law_slope, 3), "tol": 0.15,
                          "claim": "A's slope within 15 %% of the laptop's law for this shape (%.1f ms per 1,000 tokens)" % law_slope}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/lc4_20261024.py", "testbed/scripts/register_v4d_20261101.py", "testbed/scripts/predict_pool.py",
               "testbed/scripts/serve_plan_20261101.py", "testbed/dataset/attention_cost_20261101.json",
               "testbed/dataset/hybrid_shapes_20261101.json", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "testbed/dataset/link.json", "ThinkPad/config/cluster.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    for k, b in benches.items():
        print("%s %s head %s: %s ms by depth; args %s" % (k, " + ".join("%s %d" % tuple(x) for x in b["assign"]),
                                                       "laptop" if b["head_at"] == 0 or len(b["assign"]) == 1 else b["assign"][-1][0],
                                                       b["predicted"]["ms_per_token_by_depth"], b["args"]))
    print("the laptop's law for this shape: %.1f ms per 1,000 tokens" % law_slope)
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
