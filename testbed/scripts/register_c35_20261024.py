#!/usr/bin/env python3
"""Register Qwen3.6-35B-A3B Q6_K's own characterisation on the four nodes (2026-10-24), before it is measured.

    python3 testbed/scripts/register_c35_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Why: the 23 October order check (order35b, testbed/results/order35_20261023_154835) found the predictor ranks this model's
orders right but is 20 % optimistic (server means 236.9 and 225.0 against 197.0 and 188.1 ms): its per-block costs come
from per-byte rows measured on dense models. The model is too large for any single node (30.95 GB), so it is characterised
where it runs: served on the four nodes in five splits, one ring order (laptop host, then Jetson, Pi 5, Pi 4: order35b's
F), the output head on the host, so every split has the same links and the same head and only the block counts move.

The model of a split s (predict_pool's own terms, split by board with energy_model.per_node): the predicted decode is
    T(s) = sum over boards d of busy_d(s) + xfer
and the measured server mean is fitted as
    T_meas(s) = sum over d of k_d busy_d(s) + xfer
one multiplier k_d per board on its predicted busy time (4 unknowns). Splits A-D (A the engine's pick on this order today;
B, C, D move 3, 3 and 2 blocks from the laptop, the Pi 5 and the Jetson to the Pi 4) determine k exactly; E (2 from the
laptop and 1 from the Jetson to the Pi 4) is held out. A runs twice (first and last) as a drift check. Each arm-run is
qr3_tpot.py's warm-up and `reqs` requests of n + gen tokens, ctx 4096 as order35b, through ab_session_20261024.sh; the
metric is the server's own mean gap (order35b's OR-ACC metric). Scored by ab_score_20261024.py with
scorer_extra c35_fit_20261024.py (the fit and the fit-based hypotheses). Writes
testbed/dataset/c35_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "c35_predictions_20261024.json")
KEY, CTX = "qwen36moe_q6k", 4096
ORDER_NODES = ["laptop", "jetson", "pi5", "pi4"]           # order35b's F
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
MOVES = {"B": {"laptop": -3, "pi4": 3}, "C": {"pi5": -3, "pi4": 3}, "D": {"jetson": -2, "pi4": 2},
         "E": {"laptop": -2, "jetson": -1, "pi4": 3}}
MIN_MARGIN = 256                                            # MiB on every board (placement.EXACT_FILL_HEADROOM_MB)
PAST = {"P": 236.88, "F": 225.01}                           # order35b's server means (score.txt), the C35-PAST targets


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def terms(assign, e):
    busy, xfer = EM.per_node(assign, e["q"], e["C"], e["links"], head_at=0, rtt_trips=P.RTT_TRIPS_DEFAULT)
    dec, pre = P.predict(assign, e["q"], e["C"], e["links"], head_at=0, rtt_trips=P.RTT_TRIPS_DEFAULT)
    b = {n: round(busy[n]["tg"], 4) for n, _ in assign}
    if abs(sum(b.values()) + xfer["tg"] - dec) > 0.05:
        raise SystemExit("per_node does not sum to predict() for %s: %.3f vs %.3f" % (assign, sum(b.values()) + xfer["tg"], dec))
    return {"busy_ms": b, "xfer_ms": round(xfer["tg"], 4), "decode_ms": round(dec, 2), "prefill_ms": round(pre, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=6)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="A B C D E A"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, ORDER_NODES, CTX, mem, host="laptop", load_mode="none", order_check=True)
    pick = dict(e["assign"])
    if [n for n, _ in e["assign"]] != ORDER_NODES or e["head_at"] != 0:
        print("note: the engine's pick is %s (head_at %s); split A takes its counts on order %s, head on the host"
              % (e["assign"], e["head_at"], ORDER_NODES))
    base = {n: pick.get(n, 0) for n in ORDER_NODES}
    if min(base.values()) < 1 or sum(base.values()) != len(e["q"]["layer_bytes"]):
        raise SystemExit("the pick %s does not hold every block on all four nodes" % e["assign"])
    splits = {"A": base}
    for k, mv in MOVES.items():
        s = dict(base)
        for n, d in mv.items():
            s[n] += d
        splits[k] = s
    arms, power = {}, EM.power_table()
    for k, s in splits.items():
        assign = [(n, s[n]) for n in ORDER_NODES]
        if min(c for _, c in assign) < 1:
            raise SystemExit("split %s leaves a board empty: %s" % (k, assign))
        m = P.memory(assign, e["q"], mem, e["devices"], e["model"], head_at=0)
        low = {n: v["margin_mib"] for n, v in m.items() if v["margin_mib"] < MIN_MARGIN}
        if low:
            raise SystemExit("split %s does not fit with %d MiB to spare: %s" % (k, MIN_MARGIN, low))
        t = terms(assign, e)
        en = EM.energy(assign, e["q"], e["C"], e["links"], power, "tg", head_at=0, rtt_trips=P.RTT_TRIPS_DEFAULT)
        arms[k] = {"what": "split %s: %s" % (k, " + ".join("%s %d" % x for x in assign)) + ("" if k == "A" else " (A %s)" % MOVES[k]),
                   "assign": assign, "head": "host",
                   "place_env": {"PLACE": ",".join("%s:%d" % x for x in assign), "PLACE_HEAD": "host"},
                   "serve_env": {"SERVE_DRAFT": "none"}, "terms": t, "memory": m,
                   "predicted": {"server_mean_ms": t["decode_ms"], "mean_ms_per_token": t["decode_ms"],
                                 "j_per_token_all": round(en["gross_j"], 4)}}
    past = {}
    r35 = json.load(open(os.path.join(ROOT, "testbed", "dataset", "order35b_predictions_20261023.json")))
    for k in ("P", "F"):
        assign = [tuple(x) for x in r35["arms"][k]["assign"]]
        head_at = 0 if r35["arms"][k]["head"] == "host" else None
        busy, xfer = EM.per_node(assign, e["q"], e["C"], e["links"], head_at=head_at, rtt_trips=P.RTT_TRIPS_DEFAULT)
        past[k] = {"assign": assign, "head": r35["arms"][k]["head"], "busy_ms": {n: round(busy[n]["tg"], 4) for n, _ in assign},
                   "xfer_ms": round(xfer["tg"], 4), "measured_server_mean_ms": PAST[k]}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_c35_20261024.py", "session": "testbed/scripts/ab_session_20261024.sh",
           "scorer": "testbed/scripts/ab_score_20261024.py", "scorer_extra": "testbed/scripts/c35_fit_20261024.py",
           "tag_prefix": "c35", "out_prefix": "c35",
           "what": "Qwen3.6-35B-A3B Q6_K characterised where it runs: five splits on one ring order, a multiplier per board fitted "
                   "on A-D, E held out",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1],
           "ctx": CTX, "nodes": ORDER_NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "engine_pick": {"assign": e["assign"], "head_at": e["head_at"], "decode_ms": round(e["decode_ms"], 2)},
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "rtt_trips": P.RTT_TRIPS_DEFAULT, "arms": arms, "fit": {"on": ["A", "B", "C", "D"], "held_out": "E",
                                                                    "metric": "server_mean_ms", "past": past},
           "hypotheses": {
               "C35-RAW": {"type": "bias", "metric": "server_mean_ms", "lo": 0.10, "hi": 0.30,
                           "claim": "the predictor as it is under-predicts every split by 10-30 % (order35b: +19.6 and +20.2 %)"},
               "C35-DRIFT": {"type": "spread", "metric": "server_mean_ms", "arm": "A", "tol": 0.02,
                             "claim": "split A's first and last arm-runs within 2 % of each other"}},
           "hypotheses_extra": {
               "C35-HOLDOUT": {"tol": 0.05, "claim": "the multipliers fitted on A-D predict E's server mean within 5 %"},
               "C35-PAST": {"tol": 0.05, "claim": "the same multipliers predict order35b's measured P and F (23 Oct, other "
                                                 "splits and orders) within 5 %"},
               "C35-PLAUSIBLE": {"lo": 0.8, "hi": 2.0, "claim": "every fitted multiplier lies in [0.8, 2.0]"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/ab_session_20261024.sh", "testbed/scripts/ab_score_20261024.py",
               "testbed/scripts/c35_fit_20261024.py", "testbed/scripts/register_c35_20261024.py",
               "testbed/scripts/score_prima4_20261021.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "testbed/dataset/order35b_predictions_20261023.json",
               "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        for k, v in arms.items():
            print(k, v["place_env"]["PLACE"], v["terms"], {n: x["margin_mib"] for n, x in v["memory"].items()})
        print("past", json.dumps(past)); print("engine pick", reg["engine_pick"]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, v in arms.items():
        print(k, v["place_env"]["PLACE"], "predicted %.1f ms" % v["terms"]["decode_ms"])


if __name__ == "__main__":
    main()
