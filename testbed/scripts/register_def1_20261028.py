#!/usr/bin/env python3
"""Register def1 (2026-10-28; the user: "let's do 4 and 1"): do the planner's proposed defaults predict NEW placements better
than the shipped ones? Registered before anything is measured.

    python3 testbed/scripts/register_def1_20261028.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The proposed defaults (predict_pool.PLANNER_V3): order_check, the RPC message term (RPC_MSG) and the measured per-board
multipliers where a model has them (NODE_MULT: Qwen3.6-35B-A3B only, from c35). Backend sampling stays off: serving does not
sample on the head's board unless SERVE_EXTRA asks for it. On the gigabit cluster the RPC term moves a prediction by 0.0-0.4 %
(below the run-to-run noise; its effect is on slow links, registered in rpc4plan), so this test is on the model the
multipliers change, Qwen3.6-35B-A3B, where the shipped predictor is ~20 % optimistic (c35, OR-ACC).
c35 fitted the multipliers on five splits in ONE ring order (laptop, Jetson, Pi 5, Pi 4, the head on the host) and predicted a
sixth (E) within 0.04 %. def1's three placements are new in structure, each the planner's own best (PLANNER_V3) for its shape:
  H1  without the Jetson (laptop, Pi 5, Pi 4);
  H2  the full cluster in a ring order never measured (laptop, Pi 5, Pi 4, Jetson; c35 and order35b used laptop-Jetson-Pi 5-
      Pi 4 and laptop-Pi 4-Pi 5-Jetson), the head on the host, filled exactly for that order (placement._fill_exact). A plan
      without the Pi 4 does not fit this model (41 blocks of ~708 MB), found by the dry run;
  H3  the full-cluster pick with the output head on the LAST board instead of the host.
Each is priced by the shipped predictor (no options) and by PLANNER_V3's. Order H1 H2 H3 H3 H2 H1, five requests of 128 + 64
per arm-run, run by spec4_session_20261023.sh (tag_prefix "d1", scorer score_def1_20261028.py), cluster.json's reserves.
Hypotheses (registered):
  D1-V3      every arm's server mean TPOT within 5 % of its PLANNER_V3 prediction
  D1-SHIP    the shipped prediction misses every arm by more than 10 % (the defaults matter for this model)
  D1-BETTER  PLANNER_V3's |error| is below the shipped one's on every arm
  D1-REPEAT  each arm's two runs within 2 % of each other (server mean TPOT)
Writes testbed/dataset/def1_predictions_20261028.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
from control import placement as PL                         # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "def1_predictions_20261028.json")
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
    ship = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none")
    v3 = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", node_mult=P.PLANNER_V3["node_mult"],
                      rpc_msg=P.PLANNER_V3["rpc_msg"])
    m = P.memory(assign, ship["q"], mem, ship["devices"], ship["model"], head_at=ship["head_at"])
    return {"what": what, "assign": [list(x) for x in assign], "head": head,
            "place_env": {"PLACE": place, "PLACE_HEAD": head},
            "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": ""},
            "predicted": {"shipped_decode_ms": round(ship["decode_ms"], 1), "v3_decode_ms": round(v3["decode_ms"], 1),
                          "shipped_prefill_ms": round(ship["prefill_ms"], 1), "v3_prefill_ms": round(v3["prefill_ms"], 1)},
            "memory": m, "in_c35": head == "host" and place in C35}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=5)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="H1 H2 H3 H3 H2 H1"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""                       # cluster.json's reserves
    mem = json.load(open(a.memfile))
    kw = dict(host="laptop", load_mode="none", **P.PLANNER_V3)
    h1 = P.plan_host_choice(KEY, ["pi5", "pi4", "laptop"], CTX, mem, **kw)
    full = P.plan_host_choice(KEY, NODES, CTX, mem, **kw)
    order2 = ["laptop", "pi5", "pi4", "jetson"]
    lay2, _ = PL._fill_exact(order2, PL.with_headroom(full["devices"], PL.EXACT_FILL_HEADROOM_MB), full["model"], "decode",
                             0.0, 0.0, head_node="laptop")
    if lay2 is None:
        raise SystemExit("refused: the order %s has no exact fill" % order2)
    h2_assign = [(n, lay2[n]) for n in order2 if lay2.get(n, 0) > 0]
    arms = {"H1": arm("H1", "the planner's best without the Jetson", h1["assign"], "host" if h1["head_at"] == 0 else "last", mem),
            "H2": arm("H2", "the full cluster in a new ring order (laptop, Pi 5, Pi 4, Jetson), the head on the host", h2_assign,
                      "host", mem),
            "H3": arm("H3", "the full-cluster pick with the head on the last board", full["assign"],
                      "last" if full["head_at"] == 0 else "host", mem)}
    for k, x in arms.items():
        if min(v["margin_mib"] for v in x["memory"].values()) < 0:
            raise SystemExit("refused: %s does not fit at rest: %s" % (k, x["memory"]))
        if x["in_c35"]:
            raise SystemExit("refused: %s repeats a c35 split (%s); it would not be held out" % (k, x["place_env"]))
    places = [x["place_env"]["PLACE"] + "/" + x["head"] for x in arms.values()]
    if len(set(places)) < 3:
        raise SystemExit("refused: two arms coincide: %s" % places)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_def1_20261028.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_def1_20261028.py", "tag_prefix": "d1", "out_prefix": "def1",
           "what": "the planner's proposed defaults (PLANNER_V3) against the shipped predictor, on three new Qwen3.6-35B-A3B "
                   "placements, four nodes",
           "planner_v3": {"order_check": True, "rpc_msg": "RPC_MSG", "node_mult": "measured", "backend_sampling": False,
                          "node_mult_table": P.NODE_MULT.get(KEY)},
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "reserve_mb": "",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "c35_splits": C35, "arms": arms,
           "hypotheses": {
               "D1-V3": {"tol": 0.05, "claim": "every arm's server mean TPOT within 5 % of its PLANNER_V3 prediction"},
               "D1-SHIP": {"min": 0.10, "claim": "the shipped prediction misses every arm by more than 10 %"},
               "D1-BETTER": {"claim": "PLANNER_V3's |error| is below the shipped one's on every arm"},
               "D1-REPEAT": {"tol": 0.02, "claim": "each arm's two runs within 2 % of each other (server mean TPOT)"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_def1_20261028.py",
               "testbed/scripts/register_def1_20261028.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: {x: y for x, y in v.items() if x != "memory"} for k, v in arms.items()}, indent=1)); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, x in arms.items():
        print("%s %s head %s: shipped %.1f ms, v3 %.1f ms" % (k, x["place_env"]["PLACE"], x["head"],
                                                            x["predicted"]["shipped_decode_ms"], x["predicted"]["v3_decode_ms"]))


if __name__ == "__main__":
    main()
