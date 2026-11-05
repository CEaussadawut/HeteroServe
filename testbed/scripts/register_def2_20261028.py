#!/usr/bin/env python3
"""Register def2 (2026-10-28, after def1): the per-board multipliers on the BLOCKS only, tested on new head-last placements.

    python3 testbed/scripts/register_def2_20261028.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

def1 (testbed/results/def1_20261028_015958) held D1-SHIP, D1-BETTER and D1-REPEAT and failed D1-V3: with the head on the
host PLANNER_V3 came within 0.6 %, but with the head on the Pi 4 (H3) it was 6.3 % slow. Found AFTER the data: the multipliers
also scaled the 417 MB output head on the Pi 4; with the head at the board's unscaled rate (node_mult "measured_blocks") H3
comes within 1.6 %, and c35's five splits and def1's other two within 0.7-1.3 %. This registration tests that rule on three
NEW placements, each with the head on the LAST board, a different board each time, every one exactly filled for its order:
  L1  laptop, Jetson, Pi 4, Pi 5 -- the head on the Pi 5;
  L2  laptop, Pi 5, Pi 4, Jetson -- the head on the Jetson;
  L3  laptop, Pi 5, Jetson, Pi 4 -- the head on the Pi 4 (H3 had laptop, Jetson, Pi 5, Pi 4).
Each is priced with the RPC term and the multipliers two ways: "measured" (def1's PLANNER_V3) and "measured_blocks".
Order L1 L2 L3 L3 L2 L1, five requests of 128 + 64 per arm-run, spec4_session_20261023.sh (tag_prefix "d2", scorer
score_def2_20261028.py), cluster.json's reserves.
Hypotheses (registered):
  D2-BLOCKS  every arm's server mean TPOT within 3 % of its "measured_blocks" prediction
  D2-BETTER  "measured_blocks"' |error| is below "measured"'s on every arm
  D2-REPEAT  each arm's two runs within 2 % of each other
Writes testbed/dataset/def2_predictions_20261028.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
from control import placement as PL                         # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "def2_predictions_20261028.json")
KEY, CTX, NODES = "qwen36moe_q6k", 4096, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
C35 = ["laptop:25,jetson:3,pi5:10,pi4:3", "laptop:22,jetson:3,pi5:10,pi4:6", "laptop:25,jetson:3,pi5:7,pi4:6",
       "laptop:25,jetson:1,pi5:10,pi4:5", "laptop:23,jetson:2,pi5:10,pi4:6"]          # c35's splits, all head on the host


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm(name, what, assign, head, mem):
    place = ",".join("%s:%d" % tuple(x) for x in assign)
    base = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none")
    a = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", node_mult="measured", rpc_msg=P.RPC_MSG)
    b = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", node_mult="measured_blocks", rpc_msg=P.RPC_MSG)
    m = P.memory(assign, base["q"], mem, base["devices"], base["model"], head_at=base["head_at"])
    return {"what": what, "assign": [list(x) for x in assign], "head": head,
            "place_env": {"PLACE": place, "PLACE_HEAD": head},
            "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": ""},
            "predicted": {"measured_decode_ms": round(a["decode_ms"], 1), "blocks_decode_ms": round(b["decode_ms"], 1),
                          "shipped_decode_ms": round(base["decode_ms"], 1)},
            "memory": m}


def fill(order, full):
    lay, _ = PL._fill_exact(order, PL.with_headroom(full["devices"], PL.EXACT_FILL_HEADROOM_MB), full["model"], "decode",
                            0.0, 0.0, head_node=None)                      # head_node None: the head on the last board
    if lay is None:
        raise SystemExit("refused: the order %s has no exact fill with the head last" % order)
    return [(n, lay[n]) for n in order if lay.get(n, 0) > 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=5)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="L1 L2 L3 L3 L2 L1"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""                       # cluster.json's reserves
    mem = json.load(open(a.memfile))
    full = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", **P.PLANNER_V3)
    orders = {"L1": ["laptop", "jetson", "pi4", "pi5"], "L2": ["laptop", "pi5", "pi4", "jetson"], "L3": ["laptop", "pi5", "jetson", "pi4"]}
    whats = {"L1": "the head on the Pi 5", "L2": "the head on the Jetson", "L3": "the head on the Pi 4, a new order"}
    arms = {}
    for k, order in orders.items():
        assign = fill(order, full)
        if [n for n, _ in assign] != order:
            raise SystemExit("refused: %s's fill drops a board: %s" % (k, assign))
        arms[k] = arm(k, whats[k], assign, "last", mem)
    for k, x in arms.items():
        if min(v["margin_mib"] for v in x["memory"].values()) < 0:
            raise SystemExit("refused: %s does not fit at rest: %s" % (k, x["memory"]))
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_def2_20261028.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_def2_20261028.py", "tag_prefix": "d2", "out_prefix": "def2",
           "what": "the per-board multipliers on the blocks only (node_mult measured_blocks) against measured, on three new "
                   "head-last Qwen3.6-35B-A3B placements, four nodes",
           "follows": "def1 (def1_predictions_20261028.json): D1-V3 failed on H3 (head on the Pi 4, -6.3 %); the blocks-only rule "
                      "was found after that data",
           "node_mult_table": P.NODE_MULT.get(KEY),
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "reserve_mb": "",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55}, "arms": arms,
           "hypotheses": {
               "D2-BLOCKS": {"tol": 0.03, "claim": "every arm's server mean TPOT within 3 % of its measured_blocks prediction"},
               "D2-BETTER": {"claim": "measured_blocks' |error| is below measured's on every arm"},
               "D2-REPEAT": {"tol": 0.02, "claim": "each arm's two runs within 2 % of each other (server mean TPOT)"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_def2_20261028.py",
               "testbed/scripts/register_def2_20261028.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: {x: y for x, y in v.items() if x != "memory"} for k, v in arms.items()}, indent=1)); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, x in arms.items():
        print("%s %s head %s: measured %.1f ms, blocks %.1f ms (shipped %.1f)" % (
            k, x["place_env"]["PLACE"], x["head"], x["predicted"]["measured_decode_ms"], x["predicted"]["blocks_decode_ms"],
            x["predicted"]["shipped_decode_ms"]))


if __name__ == "__main__":
    main()
