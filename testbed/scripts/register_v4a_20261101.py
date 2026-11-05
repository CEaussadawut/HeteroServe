#!/usr/bin/env python3
"""Register v4a (2026-11-01; the user: "do item 1 to 4 then 6-7"): the scheduler review's items 2 and 3 on new data -- a
hybrid's memory charged as llama.cpp allocates it (PLANNER_V4's hybrid_memory), and the laptop's refitted power rows.

    python3 testbed/scripts/register_v4a_20261101.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Qwen3.6-35B-A3B Q6_K at a 16,384-token context, all four nodes, two arms planned on the same snapshot:
  K3  PLANNER_V3's pick: every block charged a full-attention KV cache (41 x 2,048 B x 16,384 = 1,312 MiB);
  K4  PLANNER_V4's pick: KV on the 10 full-attention blocks (320 MiB) and the 30 recurrent blocks' fixed state (62.8 MiB)
      -- the rule hybrid_shapes_20261101.py checked against every device's buffers of the 28 Oct serve at 4,096 cells.
Both arms are priced by PLANNER_V4 at depth 0 (the two planners price time alike; they differ in memory), served with
SERVE_PLANNER=v4 so the gate charges each as PLANNER_V4 does. Energy per decoded token at all four outlets is predicted two
ways at the predicted time per token: energy_model.energy_refit (the laptop refitted from the four-node evaluation,
power_refit_20261101.json; the 35B is not in its data) and energy_model.energy with power_table() (the model as it was).
Order K3 K4 K4 K3, five requests of 128 + 64 per arm-run, spec4_session_20261023.sh (tag_prefix "v4", scorer
score_v4a_20261101.py), cluster.json's reserves.
Hypotheses (registered):
  V4-KV      K4's server mean TPOT at least 4 % below K3's (predicted: see the arms; half the predicted gain)
  V4-ACC     each arm's server mean TPOT within 3 % of its PLANNER_V4 prediction
  V4-MEM     on K4, every board held no more than 256 MiB (the exact fill's headroom) beyond what PLANNER_V4 charged it
             (serve_pool.sh's "mem" lines: MemAvailable at the gate less at READY, against its charge)
  E-REFIT    each arm's gross J per decoded token, all four outlets, within 10 % of energy_refit's prediction
  E-OLD      the old model's prediction at least 10 % below each arm's measured J per token (the bias the refit removes)
  V4-REPEAT  each arm's two runs within 2 % of each other (server mean TPOT)
Writes testbed/dataset/v4a_predictions_20261101.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "v4a_predictions_20261101.json")
KEY, CTX, NODES = "qwen36moe_q6k", 16384, ["laptop", "pi5", "pi4", "jetson"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm(what, plan, mem):
    head = "host" if plan["head_at"] == 0 else "last"
    assign = [tuple(x) for x in plan["assign"]]
    r = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", **P.PLANNER_V4)
    m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
    v3 = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", **P.PLANNER_V3)
    m3 = P.memory(v3["assign"], v3["q"], mem, v3["devices"], v3["model"], head_at=v3["head_at"])
    T = r["decode_ms"]
    new = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], head_at=r["head_at"], rpc_msg=P.RPC_MSG, t_ms=T)
    old = EM.energy(r["assign"], r["q"], r["C"], r["links"], EM.power_table(), "tg", head_at=r["head_at"], t_ms=T)
    return {"what": what, "assign": [list(x) for x in assign], "head": head,
            "place_env": {"PLACE": ",".join("%s:%d" % x for x in assign), "PLACE_HEAD": head},
            "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": "", "SERVE_PLANNER": "v4"},
            "predicted": {"decode_ms": round(T, 1), "v3_decode_ms": round(v3["decode_ms"], 1),
                          "j_per_token_all_refit": round(new["all_j"], 4), "j_per_token_all_old": round(old["gross_j"], 4),
                          "board_w_refit": {n: round(w, 3) for n, w in new["board_w"].items()},
                          "board_w_old": {n: round(w, 3) for n, w in old["board_w"].items()},
                          "busy_frac": {n: round(f, 4) for n, f in new["busy_frac"].items()}},
            "memory_v4": m, "memory_v3_accounting": {n: {k: v[k] for k in ("need_mib", "margin_mib")} for n, v in m3.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=5)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="K3 K4 K4 K3"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""                       # cluster.json's reserves
    mem = json.load(open(a.memfile))
    p3 = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", **P.PLANNER_V3)
    p4 = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", **P.PLANNER_V4)
    if (p3["assign"], p3["head_at"]) == (p4["assign"], p4["head_at"]):
        raise SystemExit("refused: the two planners pick the same plan on this snapshot: %s" % (p3["assign"],))
    arms = {"K3": arm("PLANNER_V3's pick (every block charged a full-attention KV cache)", p3, mem),
            "K4": arm("PLANNER_V4's pick (KV on the full-attention blocks, recurrent state on the rest)", p4, mem)}
    for k, x in arms.items():
        if min(v["margin_mib"] for v in x["memory_v4"].values()) < 0:
            raise SystemExit("refused: %s does not fit at rest: %s" % (k, x["memory_v4"]))
    gain = 1 - arms["K4"]["predicted"]["decode_ms"] / arms["K3"]["predicted"]["decode_ms"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_v4a_20261101.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_v4a_20261101.py", "tag_prefix": "v4", "out_prefix": "v4a",
           "what": "the scheduler review's items 2 and 3 on new data: PLANNER_V4's hybrid memory (Qwen3.6-35B-A3B at 16,384 "
                   "cells, V3's pick against V4's) and the laptop's refitted power rows (all four outlets)",
           "follows": "testbed/JOURNAL.md 2026-11-01 (the review; its items 2 and 3); hybrid_shapes_20261101.json; "
                      "power_refit_20261101.json",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "reserve_mb": "",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           "cold_c": {"pi5": 50, "pi4": 55}, "arms": arms, "predicted_gain_k4_vs_k3": round(gain, 4),
           "power_refit": EM.refit_table()["laptop"],
           "hypotheses": {
               "V4-KV": {"min_gain": 0.04, "claim": "K4's server mean TPOT at least 4 %% below K3's (predicted %.1f %%)" % (100 * gain)},
               "V4-ACC": {"tol": 0.03, "claim": "each arm's server mean TPOT within 3 % of its PLANNER_V4 prediction"},
               "V4-MEM": {"tol_mib": 256, "claim": "on K4, every board held no more than 256 MiB beyond its PLANNER_V4 charge "
                                                     "(serve_pool.sh's mem lines at READY)"},
               "E-REFIT": {"tol": 0.10, "claim": "each arm's gross J per decoded token, all four outlets, within 10 % of "
                                                 "energy_refit's prediction"},
               "E-OLD": {"min_low": 0.10, "claim": "the old model's prediction at least 10 % below each arm's measured J per token"},
               "V4-REPEAT": {"tol": 0.02, "claim": "each arm's two runs within 2 % of each other (server mean TPOT)"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/serve_plan_20261101.py", "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py",
               "testbed/scripts/shelly_strip.py", "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh",
               "testbed/scripts/score_v4a_20261101.py", "testbed/scripts/register_v4a_20261101.py",
               "testbed/dataset/model_shapes.json", "testbed/dataset/hybrid_shapes_20261101.json",
               "testbed/dataset/attention_cost_20261101.json", "testbed/dataset/power_refit_20261101.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    for k, x in arms.items():
        p = x["predicted"]
        print("%s %s head %s: %.1f ms (V3 prices it %.1f); J/token refit %.3f, old %.3f; margins v4 %s / v3 accounting %s" % (
            k, x["place_env"]["PLACE"], x["head"], p["decode_ms"], p["v3_decode_ms"], p["j_per_token_all_refit"],
            p["j_per_token_all_old"], {n: v["margin_mib"] for n, v in x["memory_v4"].items()},
            {n: v["margin_mib"] for n, v in x["memory_v3_accounting"].items()}))
    print("predicted gain K4 against K3: %.1f %%" % (100 * gain))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
