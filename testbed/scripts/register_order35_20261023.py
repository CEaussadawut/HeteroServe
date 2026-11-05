#!/usr/bin/env python3
"""Register the order check on Qwen3.6-35B-A3B Q6_K (all four nodes), before anything is measured.

    python3 testbed/scripts/register_order35_20261023.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The brief's open item (2026-10-23): the engine's pick is chosen by placement.plan_optimal's own linear objective and only
then priced by predict(); order_audit_20261023.py prices every order of every subset (the same exact fill, both head
positions) and found one model, this one, where the predictor ranks an order the engine did not pick 2.5 % faster. The user
(~10:40): "run everything else now" -- this is also Qwen3.6-35B-A3B's evaluation on four nodes: every standard baseline
fails the engine's memory accounting here (fe4_placements_20261020.py at the same snapshot: the boards alone, any single node,
an equal split, llama.cpp's default split by free memory, -1,292 MiB on its tightest node; Halda's windows need a dense
n_ff), so the order is the one alternative that fits. Writes testbed/dataset/order35_predictions_20261023.json, read-only.
  P  the engine's pick (predict_pool.plan_host_choice), cluster.json's reserves (the laptop's 8 GiB);
  F  the fastest plan by the same predictor over every order (order_audit_20261023.audit), the same reserves.
Order P F P F, run by spec4_session_20261023.sh (tag_prefix "or", scorer score_order35_20261023.py).
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
import order_audit_20261023 as OA                           # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "order35_predictions_20261023.json")
KEY, CTX, NODES = "qwen36moe_q6k", 4096, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm(assign, head, e, mem, power, what):
    head_at = 0 if head == "host" else None
    dec, pre = P.predict(assign, e["q"], e["C"], e["links"], head_at=head_at, rtt_trips=P.RTT_TRIPS_DEFAULT)
    m = P.memory(assign, e["q"], mem, e["devices"], e["model"], head_at=head_at)
    en = EM.energy(assign, e["q"], e["C"], e["links"], power, "tg", head_at=head_at, rtt_trips=P.RTT_TRIPS_DEFAULT)
    idle = sum(power[n]["idle"] for n in NODES if n not in dict(assign)) * en["T_ms"] / 1000
    return {"what": what, "assign": [list(x) for x in assign], "head": head,
            "place_env": {"PLACE": ",".join("%s:%d" % tuple(x) for x in assign), "PLACE_HEAD": head},
            "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": ""},
            "predicted": {"decode_ms": round(dec, 1), "prefill_ms": round(pre, 1), "decode_gross_j_all": round(en["gross_j"] + idle, 4)},
            "memory": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=8)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="P F P F"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=OUT, help="the registration file (15:50: order35b supersedes order35, whose session "
                                               "stopped at its first line on a session-script bug)")
    ap.add_argument("--supersedes", default="", help="why, recorded in the file")
    a = ap.parse_args()
    globals()["OUT"] = os.path.abspath(a.out)
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    power = EM.power_table()
    pick, rows = OA.audit(KEY, NODES, CTX, mem)
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none")
    best = rows[0]
    if [tuple(x) for x in best["assign"]] == [tuple(x) for x in pick["assign"]] and best["head"] == pick["head"]:
        raise SystemExit("refused: the engine's pick is already the predictor's best at this snapshot; nothing to compare")
    Pa = arm([tuple(x) for x in pick["assign"]], pick["head"], e, mem, power, "the engine's pick (plan_optimal's objective)")
    Fa = arm([tuple(x) for x in best["assign"]], best["head"], e, mem, power, "the fastest plan by the predictor over every order")
    for x in (Pa, Fa):
        if min(v["margin_mib"] for v in x["memory"].values()) < 0:
            raise SystemExit("refused: a plan does not fit at rest: %s" % x["memory"])
    sp = Pa["predicted"]["decode_ms"] / Fa["predicted"]["decode_ms"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           **({"supersedes": a.supersedes} if a.supersedes else {}),
           "script": "testbed/scripts/register_order35_20261023.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_order35_20261023.py", "tag_prefix": "or", "out_prefix": "order35",
           "what": "the order check on Qwen3.6-35B-A3B Q6_K, all four nodes: the engine's pick against the predictor's best order",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "reserve_mb": "",
           "reserve": "cluster.json's (the laptop's 8 GiB desktop reserve)",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "baselines_that_do_not_fit": "fe4_placements_20261020.py, memory_snapshot_20261023_104214: boards alone, single node, "
                                        "equal split, llama.cpp's default split (-1,292 MiB), Halda (no dense n_ff)",
           "plans_priced": len(rows), "arms": {"P": Pa, "F": Fa},
           "hypotheses": {
               "OR-FASTER": {"predicted": round(sp, 4), "claim": "F's client TPOT P50 is below P's (predicted %.1f against %.1f ms)" % (
                   Fa["predicted"]["decode_ms"], Pa["predicted"]["decode_ms"])},
               "OR-ACC": {"tol": 0.10, "claim": "each arm's server mean TPOT within 10 % of its predicted decode"},
               "OR-ENERGY": {"claim": "the arm predicted cheaper (%s) has the lower gross J per decoded token at all four outlets" % (
                   "F" if Fa["predicted"]["decode_gross_j_all"] < Pa["predicted"]["decode_gross_j_all"] else "P")},
               "OR-TEXT": {"claim": "request 0's first 16 tokens of text are the same in every P and F run (greedy; the others "
                                    "recorded)"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/predict_pool.py",
               "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py", "testbed/scripts/full_eval.py",
               "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_order35_20261023.py",
               "testbed/scripts/register_order35_20261023.py", "testbed/scripts/order_audit_20261023.py",
               "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("arms", "hypotheses")}, indent=1)[:4000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    for k, x in (("P", Pa), ("F", Fa)):
        print("%s %s head %s: %.1f ms, %.2f J" % (k, x["place_env"]["PLACE"], x["head"], x["predicted"]["decode_ms"],
                                                  x["predicted"]["decode_gross_j_all"]))


if __name__ == "__main__":
    main()
