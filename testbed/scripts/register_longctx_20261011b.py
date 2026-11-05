#!/usr/bin/env python3
"""Registration b of the long-context plan (2026-10-11), superseding longctx_predictions_20261011.json BEFORE the full run.
Writes, read-only, testbed/dataset/longctx_predictions_20261011b.json.

    python3 testbed/scripts/register_longctx_20261011b.py [--dry-run]

One change of input, and the planner re-run on it:
- the smoke (longctx_smoke_20261011_001353, plumbing only, never scored) decoded the plan at 668 ms against 892 predicted.
  That traced to a's laptop row, measured with llama-bench's polling threads on the same 4 cores as the worker;
- b prices the laptop from attention_cost_20261011b.json and laptop_worker_20261011b.json: the same cells re-run with the
  Pi 4 as a remote host (laptop_worker_remote_20261011.py), its worker row net of the network.
Everything else is a's: the planner, the boards, the planning depth 4,096, the protocol, the three runs, the tolerances and
the hypotheses. With b's costs the planner's pick is Pi 4 1 + laptop 39 at every depth up to 4,096: the laptop beats the
Jetson per block even there, so the Jetson is dropped. So b adds a second cell, the plan that keeps the Jetson (a's pick,
and the cluster the question assumed), measured at d = 0, to test the planner's ranking (LC-RANK) and its price (LC-JET).
a's predictions are reported beside b's.
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import register_longctx_20261011 as A                       # noqa: E402

OUT = os.path.join(A.DS, "longctx_predictions_20261011b.json")
ATTN_B = os.path.join(A.DS, "attention_cost_20261011b.json")
LAP_B = os.path.join(A.DS, "laptop_worker_20261011b.json")


def inputs_b(laptop_avail_mib):
    shapes = json.load(open(os.path.join(A.DS, "model_shapes.json")))
    q = shapes["qwen3-14b-q4_k_m"]
    R = A.P.rows(os.path.join(A.DS, "characterisation.csv"), "char")
    lap = A.P.laptop_costs(R, shapes, "Q4_K_M", LAP_B)
    _, _, lk = A.LJ.parse(A.LINK_LAPTOP)
    pairs = {"pi4-laptop": {"mbps": lk["mbps"], "rtt_ms": lk["rtt_ms"]}}
    mem = json.load(open(os.path.join(A.DS, "memory_at_rest.json")))
    mem = dict(mem, laptop={"mem_available_mib": laptop_avail_mib - A.LAPTOP_RESERVE_MIB})
    T = A.P.attention_table(ATTN_B)
    attn = {n: A.P.attn_ms(T, n, q) for n in ("pi4", "jetson", "laptop")}
    return q, lap, pairs, mem, attn


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % OUT)
    avail = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable"))) // 1024
    a = json.load(open(A.OUT))
    A.inputs = inputs_b                                     # build() resolves inputs() at call time
    reg = A.build(avail)
    # The second cell: the plan that keeps the Jetson (Pi 4 1 + Jetson 13 + laptop 26, head on the laptop; a's pick, and
    # the cluster the question assumed). The planner prices it slower; measured at d = 0 beside the pick, it tests the
    # planner's ranking (LC-RANK) and its price (LC-JET).
    q, lap, pairs, mem, attn = inputs_b(avail)
    forced = [("pi4", 1), ("jetson", 13), ("laptop", 26)]
    fp = A.price(forced, "last", 0, mem, attn, lap, pairs)
    fmem = A.P.memory(forced, q, mem, fp["devices"], fp["model"], head_at=None)
    if any(m["margin_mib"] < 0 for m in fmem.values()):
        raise SystemExit("the forced plan does not fit: %s" % fmem)
    reg["cells"] = [
        {"name": "pick", "args": reg["bench"]["args"], "protocol": reg["bench"]["protocol"], "runs": reg["runs"],
         "memory": reg["plan"]["memory"]},
        {"name": "jetson_forced", "assign": [list(a) for a in forced], "head": "last",
         "args": A.P.bench_args(forced, "qwen3_14b", head_at=None)["args"] + " -lm none",
         "protocol": "-t 4 -p 0 -n 16 -d 0 -r 3 -oe jsonl", "runs": 3, "memory": fmem,
         "prediction_ms": {"0": fp["decode_ms"]}}]
    reg["hypotheses"].update({
        "LC-JET": "the forced plan's measured ms per token at d = 0 (mean of three runs) within 15 % of cells[1].prediction_ms.0",
        "LC-RANK": "the planner's pick measures faster than the forced plan at d = 0 (predicted ratio cells[1] / pick)"})
    reg.update({"script": "testbed/scripts/register_longctx_20261011b.py",
                "supersedes": {"file": "testbed/dataset/longctx_predictions_20261011.json", "sha256": A.sha(A.OUT),
                               "why": "a's laptop row was measured with the host's polling threads on the laptop's cores; "
                                      "b re-measured it with the Pi 4 as a remote host"},
                "a_predictions_ms": a["predictions_ms"], "a_plan": a["plan"]["assign"]})
    for p in ("testbed/dataset/attention_cost_20261011b.json", "testbed/dataset/laptop_worker_20261011b.json",
              "testbed/scripts/register_longctx_20261011b.py", "testbed/scripts/laptop_worker_remote_20261011.py"):
        reg["inputs_sha256"][p] = A.sha(os.path.join(ROOT, p))
    print(json.dumps({k: reg[k] for k in ("planner_picks_by_depth", "plan", "cells", "predictions_ms", "a_predictions_ms",
                                          "served_plan_predicted_ms", "predicted_gain_d0")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (%s)" % (os.path.relpath(OUT, ROOT), A.sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
