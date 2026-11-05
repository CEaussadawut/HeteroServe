#!/usr/bin/env python3
"""Order audit (2026-10-23): does the engine's pick (predict_pool.plan_host_choice: placement.plan_optimal's own linear
objective chooses the order, the head and the layers; predict() then prices that one plan) match the best plan by the
predictor itself? For each model: every order of every subset of the nodes that starts at the host, both head positions,
the same exact fill as plan_optimal (placement._fill_exact, the same budgets and headroom), each feasible plan priced by
predict(); the fastest is compared with the engine's pick. Reads files only.

    python3 testbed/scripts/order_audit_20261023.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--models k1,k2]
        [--reserve "laptop=7168"] [--ctx 4096]
"""
import argparse, dataclasses, itertools, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
from control import placement as PL                         # noqa: E402


def audit(key, nodes, ctx, mem, host="laptop"):
    e = P.plan_host_choice(key, nodes, ctx, mem, host=host, load_mode="none")
    q, C, L, devs, mp = e["q"], e["C"], e["links"], e["devices"], e["model"]
    hd = PL.with_headroom(devs, PL.EXACT_FILL_HEADROOM_MB) if mp.layer_bytes else devs
    rows = []
    others = [n for n in devs if n != host]
    for size in range(0, len(others) + 1):
        for sub in itertools.combinations(others, size):
            for perm in itertools.permutations(sub):
                order = [host] + list(perm)
                heads = [None, host] if len(order) > 1 else [None]
                for hn in heads:
                    layers, cost = PL._fill_exact(order, hd, mp, "decode", 0.0, 0.0, head_node=hn)
                    if layers is None:
                        continue
                    assign = [(n, layers[n]) for n in order if layers.get(n, 0) > 0]
                    if len(assign) < len(order):
                        continue
                    head_at = 0 if hn and len(assign) > 1 else None
                    dec, pre = P.predict(assign, q, C, L, head_at=head_at, rtt_trips=P.RTT_TRIPS_DEFAULT)
                    rows.append({"assign": assign, "head": "host" if head_at == 0 else "last", "objective": cost,
                                 "decode_ms": round(dec, 1), "prefill_ms": round(pre, 1)})
    rows.sort(key=lambda r: r["decode_ms"])
    pick = {"assign": e["assign"], "head": "host" if e["head_at"] == 0 else "last", "decode_ms": round(e["decode_ms"], 1),
            "prefill_ms": round(e["prefill_ms"], 1)}
    return pick, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--models", default="qwen36moe_q6k,qwen3_32b_q8,qwen3_32b_q6k,qwen3moe_q6k,qwen3moe,qwen38_27b,nemo12b,qwen3_14b,qwen35_9b")
    ap.add_argument("--reserve", default="")
    a = ap.parse_args()
    os.environ["HETEROSERVE_RESERVE_MB"] = a.reserve
    mem = json.load(open(a.memfile))
    nodes = ["pi5", "pi4", "jetson", "laptop"]
    for key in a.models.split(","):
        ctx = 2048 if key.startswith("qwen3_32b") else a.ctx
        try:
            pick, rows = audit(key, nodes, ctx, mem)
        except Exception as x:
            print("%-14s %s" % (key, str(x)[:120])); continue
        best = rows[0] if rows else None
        gap = 100 * (pick["decode_ms"] / best["decode_ms"] - 1) if best else 0
        print("%-14s ctx %d pick %-44s %s %8.1f ms | best by predictor %-44s %s %8.1f ms | pick slower by %.1f %% (%d plans)" % (
            key, ctx, " + ".join("%s %d" % x for x in pick["assign"]), pick["head"], pick["decode_ms"],
            " + ".join("%s %d" % tuple(x) for x in best["assign"]), best["head"], best["decode_ms"], gap, len(rows)))


if __name__ == "__main__":
    main()
