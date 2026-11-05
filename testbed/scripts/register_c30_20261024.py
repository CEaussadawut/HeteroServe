#!/usr/bin/env python3
"""Register whether Qwen3.6-35B-A3B's per-board multipliers transfer to Qwen3-30B-A3B Q6_K (study c30, 2026-10-24; the user:
"continue what's left for project 2"), before it is measured.

    python3 testbed/scripts/register_c30_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Why: c35 (testbed/results/c35_20261024_025725) fitted one multiplier per board on the 35B's predicted busy time (laptop 1.172,
Jetson 1.278, Pi 5 1.344, Pi 4 1.152; predict_pool.NODE_MULT), and they predicted its held-out split within 0.04 % and the
23 October runs within 2.2 %. They are an option in the planner for that model only. Whether they are a property of the
boards reading MoE experts, or of that one model (a hybrid: Gated DeltaNet plus attention), decides whether a new MoE model
needs its own characterisation (~1.3 h) before the planner can price it. Qwen3-30B-A3B is the other MoE on file: the same
hidden size (2,048, so the same link term), full attention in all 48 blocks, Q6_K like the 35B.

Design (c35's, on the 30B): one ring order (laptop host, Jetson, Pi 5, Pi 4: order35b's F), the output head on the host, so
every split has the same links and head and only the block counts move. Split A is the planner's best split on that order
with every board holding at least MIN_EACH blocks (the planner's own pick for this model uses two boards; at least three
blocks each keeps every board's multiplier in every split and leaves room for the moves); B, C and D move 3, 3 and 2 blocks
from the laptop, the Pi 5 and the Jetson to the Pi 4, as c35's. Order A B C D A (A first and last, the drift check). Each
arm-run is qr3_tpot.py's warm-up and `reqs` requests of n + gen tokens at ctx 4096, through ab_session_20261024.sh; the
metric is the server's own mean gap, as c35's.

Registered, per split: the prediction with the 35B's multipliers (arms[k].predicted.server_mean_ms, what C30-TRANSFER tests)
and the predictor as it is (arms[k].terms.decode_ms, what C30-RAW tests). Scored by ab_score_20261024.py with scorer_extra
c30_fit_20261024.py (C30-RAW; the 30B's own multipliers, fitted exactly on A-D as c35's were, with how far 1 % of noise on
each split moves them; C30-PLAUSIBLE). Writes testbed/dataset/c30_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "c30_predictions_20261024.json")
KEY, CTX = "qwen3moe_q6k", 4096
SOURCE = "qwen36moe_q6k"                                    # the model whose multipliers are tested here
ORDER_NODES = ["laptop", "jetson", "pi5", "pi4"]           # order35b's F, c35's order
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
MOVES = {"B": {"laptop": -3, "pi4": 3}, "C": {"pi5": -3, "pi4": 3}, "D": {"jetson": -2, "pi4": 2}}
MIN_EACH = 3
MIN_MARGIN = 256                                            # MiB on every board (placement.EXACT_FILL_HEADROOM_MB)
C35_RUN = "testbed/results/c35_20261024_025725"


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def priced(assign, mem, k35):
    """The split priced as it is and with the 35B's multipliers; the second is checked against plan_fixed's own pricing."""
    e = P.plan_fixed(KEY, assign, "host", CTX, mem, load_mode="none")
    m = P.memory(assign, e["q"], mem, e["devices"], e["model"], head_at=0)
    busy, xfer = EM.per_node(assign, e["q"], e["C"], e["links"], head_at=0, rtt_trips=P.RTT_TRIPS_DEFAULT)
    b = {n: round(busy[n]["tg"], 4) for n, _ in assign}
    if abs(sum(b.values()) + xfer["tg"] - e["decode_ms"]) > 0.05:
        raise SystemExit("per_node does not sum to predict() for %s" % assign)
    scaled = sum(k35[n] * b[n] for n in b) + xfer["tg"]
    e2 = P.plan_fixed(KEY, assign, "host", CTX, mem, load_mode="none", node_mult=k35)
    if abs(scaled - e2["decode_ms"]) > 0.05:
        raise SystemExit("the multipliers do not price %s as plan_fixed does: %.3f vs %.3f" % (assign, scaled, e2["decode_ms"]))
    return e, m, {"busy_ms": b, "xfer_ms": round(xfer["tg"], 4), "decode_ms": round(e["decode_ms"], 2),
                  "prefill_ms": round(e["prefill_ms"], 2), "decode_ms_35b_mult": round(scaled, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=6)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="A B C D A"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    mem = json.load(open(a.memfile))
    k35 = dict(P.NODE_MULT[SOURCE])
    c35 = json.load(open(os.path.join(ROOT, C35_RUN, "score.json")))["extra"]["k"]
    if any(abs(c35[n] - k35[n]) > 1e-4 for n in ORDER_NODES):
        raise SystemExit("predict_pool.NODE_MULT %s is not c35's fit %s" % (k35, c35))
    # split A: the best (as priced today) on the order with every board >= MIN_EACH blocks that fits
    n_layers = None
    best = None
    for j in range(MIN_EACH, 49):
        for p5 in range(MIN_EACH, 49):
            for p4 in range(MIN_EACH, 49):
                lap = 48 - j - p5 - p4
                if lap < MIN_EACH:
                    continue
                assign = [("laptop", lap), ("jetson", j), ("pi5", p5), ("pi4", p4)]
                e = P.plan_fixed(KEY, assign, "host", CTX, mem, load_mode="none")
                n_layers = e["q"]["n_layers"]
                m = P.memory(assign, e["q"], mem, e["devices"], e["model"], head_at=0)
                if all(v["margin_mib"] >= MIN_MARGIN for v in m.values()) and (best is None or e["decode_ms"] < best[0]):
                    best = (e["decode_ms"], assign)
    if best is None or n_layers != 48:
        raise SystemExit("no split with every board >= %d blocks fits (%s blocks)" % (MIN_EACH, n_layers))
    base = dict(best[1])
    splits = {"A": base}
    for k, mv in MOVES.items():
        s = dict(base)
        for n, d in mv.items():
            s[n] += d
        splits[k] = s
    pick = P.plan_host_choice(KEY, ORDER_NODES, CTX, mem, host="laptop", load_mode="none", order_check=True)
    arms, power = {}, EM.power_table()
    for k, s in splits.items():
        assign = [(n, s[n]) for n in ORDER_NODES]
        if min(c for _, c in assign) < 1:
            raise SystemExit("split %s leaves a board empty: %s" % (k, assign))
        e, m, t = priced(assign, mem, k35)
        low = {n: v["margin_mib"] for n, v in m.items() if v["margin_mib"] < MIN_MARGIN}
        if low:
            raise SystemExit("split %s does not fit with %d MiB to spare: %s" % (k, MIN_MARGIN, low))
        en = EM.energy(assign, e["q"], e["C"], e["links"], power, "tg", head_at=0, rtt_trips=P.RTT_TRIPS_DEFAULT)
        arms[k] = {"what": "split %s: %s" % (k, " + ".join("%s %d" % x for x in assign)) + ("" if k == "A" else " (A %s)" % MOVES[k]),
                   "assign": assign, "head": "host",
                   "place_env": {"PLACE": ",".join("%s:%d" % x for x in assign), "PLACE_HEAD": "host"},
                   "serve_env": {"SERVE_DRAFT": "none"}, "terms": t, "memory": m,
                   "predicted": {"server_mean_ms": t["decode_ms_35b_mult"], "mean_ms_per_token": t["decode_ms_35b_mult"],
                                 "j_per_token_all": round(en["gross_j"], 4)}}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_c30_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/c30_fit_20261024.py",
           "tag_prefix": "c30", "out_prefix": "c30",
           "what": "whether Qwen3.6-35B-A3B's per-board multipliers (c35) price Qwen3-30B-A3B Q6_K: four splits on c35's ring "
                   "order, predicted with them; the 30B's own multipliers fitted on the same splits",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1],
           "ctx": CTX, "nodes": ORDER_NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "engine_pick": {"assign": pick["assign"], "head_at": pick["head_at"], "decode_ms": round(pick["decode_ms"], 2)},
           "split_a_rule": "the lowest predicted (as priced today) split on the order with every board >= %d blocks and %d MiB "
                           "to spare on each" % (MIN_EACH, MIN_MARGIN),
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "rtt_trips": P.RTT_TRIPS_DEFAULT, "arms": arms,
           "fit": {"on": ["A", "B", "C", "D"], "metric": "server_mean_ms", "k35": k35, "k35_from": C35_RUN + "/score.json"},
           "hypotheses": {
               "C30-TRANSFER": {"type": "acc", "metric": "server_mean_ms", "tol": 0.05,
                                "claim": "the 35B's per-board multipliers predict every split's server mean within 5 %"},
               "C30-DRIFT": {"type": "spread", "metric": "server_mean_ms", "arm": "A", "tol": 0.02,
                             "claim": "split A's first and last arm-runs within 2 % of each other"}},
           "hypotheses_extra": {
               "C30-RAW": {"lo": 0.10, "hi": 0.30, "claim": "the predictor as it is under-predicts every split by 10-30 % "
                                                           "(the 35B: 19.5-21.7 %)"},
               "C30-PLAUSIBLE": {"lo": 0.8, "hi": 2.0, "claim": "every multiplier fitted on the 30B lies in [0.8, 2.0]"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/c30_fit_20261024.py", "testbed/scripts/register_c30_20261024.py",
               "testbed/scripts/score_prima4_20261021.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", C35_RUN + "/score.json",
               "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        for k, v in arms.items():
            print(k, v["place_env"]["PLACE"], v["terms"], {n: x["margin_mib"] for n, x in v["memory"].items()})
        print("engine pick", reg["engine_pick"]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, v in arms.items():
        print(k, v["place_env"]["PLACE"], "predicted %.1f ms with the 35B's multipliers (%.1f as it is)"
              % (v["terms"]["decode_ms_35b_mult"], v["terms"]["decode_ms"]))


if __name__ == "__main__":
    main()
