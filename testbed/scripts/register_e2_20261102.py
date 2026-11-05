#!/usr/bin/env python3
"""Register e2 (2026-11-02; the user: "run the energy refit test"): the laptop's refitted power rows with the SESSION'S OWN
laptop idle, on a model they were not fitted on.

    python3 testbed/scripts/register_e2_20261102.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Why: v4a's E-REFIT failed (+21.8, +19.8 %): the refit (power_refit_20261101.json) fixed the laptop's idle at the four-node
evaluation's 20.47 W, and v4a's session idled at 17.1. Over 24 engine arms of 10 held-out sessions the refit with each
session's own laptop idle came within a median -1.2 % (21 of 24 within 10 %; energy_holdout_20261101.py, post hoc). The worst
were scheduler v2's Qwen3-32B Q6_K plans (-12.4, -10.2 %), so this test uses that model.
Qwen3-32B Q6_K at ctx 2,048 (scheduler v2's model and context), all four nodes, two arms planned on the same snapshot:
  E1  PLANNER_V4's pick;
  E2  the plan PLANNER_V4 picks with 6 GiB less free on the laptop: more blocks on the boards, a smaller laptop share.
Each arm's prediction is fixed here as a formula whose one measured input is the laptop's mean W over the session's 180 s
idle-before window (idle_s), read by the scorer from the session's strip log:
  J(T, idle_s) = T x (sum of the boards' W + idle_s + (active - idle_fit) x f_laptop + pool_host) + per_token
with the refit's rows (active 35.74, idle_fit 20.47, per_token 0.217 J, pool_host 1.62 W: the laptop hosts both arms with
workers), each board's W from energy_model.power_table() at its busy fraction (as the refit prices the boards), and the busy
fractions f from energy_model.per_node with the plan's PLANNER_V4 pricing (the RPC term), all recorded below.
Order E1 E2 E2 E1, three requests of 128 + 64 per arm-run (spec4d's), spec4_session_20261023.sh (tag_prefix "e2", scorer
score_e2_20261102.py), cluster.json's reserves, served with SERVE_PLANNER=v4.
Hypotheses (registered):
  E2-REFIT   each arm's gross J per decoded token, all four outlets, within 10 % of J(T_pred, idle_s), T_pred PLANNER_V4's time
  E2-POWER   the same within 10 % of J(T_meas, idle_s), at the arm's measured server mean TPOT (the power model alone)
  E2-BETTER  J(T_pred, idle_s)'s |error| below the old model's (energy_model.energy, power_table) on every arm
  E2-REPEAT  each arm's two runs within 2 % of each other (server mean TPOT)
Writes testbed/dataset/e2_predictions_20261102.json, read-only; refuses to overwrite it.
"""
import argparse, copy, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "e2_predictions_20261102.json")
KEY, CTX, NODES = "qwen3_32b_q6k", 2048, ["laptop", "pi5", "pi4", "jetson"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
LESS_MIB = 6144


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def j_formula(T_ms, idle_s, fx):
    """J per token at time T_ms with the laptop idling at idle_s: the registered formula (the scorer has its own copy)."""
    return T_ms / 1000.0 * (fx["boards_w_sum"] + idle_s + fx["active_minus_idle_w"] * fx["f_laptop"] + fx["pool_host_w"]) \
        + fx["per_token_j"]


def arm(what, plan, mem, refit, power):
    head = "host" if plan["head_at"] == 0 else "last"
    assign = [tuple(x) for x in plan["assign"]]
    r = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", **P.PLANNER_V4)
    if assign[0][0] != "laptop" or len(assign) < 2:
        raise SystemExit("refused: %s is not a laptop-hosted pooled plan" % (assign,))
    m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
    T = r["decode_ms"]
    busy, xfer = EM.per_node(r["assign"], r["q"], r["C"], r["links"], r["head_at"], rpc_msg=P.RPC_MSG)
    tp = sum(b["tg"] for b in busy.values()) + xfer["tg"]
    f = {n: busy[n]["tg"] / tp for n in busy}
    lap = refit["laptop"]
    bw = {n: (power[n]["tg"] * f[n] + power[n]["idle"] * (1 - f[n])) if n in f else power[n]["idle"] for n in ("pi5", "pi4", "jetson")}
    fx = {"boards_w": bw, "boards_w_sum": sum(bw.values()), "f_laptop": f["laptop"],          # unrounded: the scorer's formula
          "active_minus_idle_w": lap["active_w"] - lap["idle_w"],                              # must give exactly these numbers
          "per_token_j": lap["per_token_j"], "pool_host_w": lap["pool_host_w"]}
    old = EM.energy(r["assign"], r["q"], r["C"], r["links"], power, "tg", head_at=r["head_at"], t_ms=T)
    fixed = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], refit, head_at=r["head_at"], t_ms=T, rpc_msg=P.RPC_MSG)
    if abs(j_formula(T, lap["idle_w"], fx) / fixed["all_j"] - 1) > 1e-9:
        raise SystemExit("the formula does not reproduce energy_refit at the fitted idle: %r against %r" % (
            j_formula(T, lap["idle_w"], fx), fixed["all_j"]))
    return {"what": what, "assign": [list(x) for x in assign], "head": head,
            "place_env": {"PLACE": ",".join("%s:%d" % x for x in assign), "PLACE_HEAD": head},
            "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": "", "SERVE_PLANNER": "v4"},
            "predicted": {"decode_ms": round(T, 1), "busy_frac": {n: round(v, 5) for n, v in f.items()}, "formula": fx,
                          "j_per_token_all_old": round(old["gross_j"], 4), "j_per_token_all_refit_fixed_idle": round(fixed["all_j"], 4),
                          "j_at_idle": {str(w): round(j_formula(T, w, fx), 4) for w in (14.0, 16.0, 17.0, 18.0, 20.47)}},
            "memory": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="E1 E2 E2 E1"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    refit, power = EM.refit_table(), EM.power_table()
    p1 = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", **P.PLANNER_V4)
    less = copy.deepcopy(mem)
    less["laptop"]["mem_available_mib"] -= LESS_MIB
    p2 = P.plan_host_choice(KEY, NODES, CTX, less, host="laptop", load_mode="none", **P.PLANNER_V4)
    if (p1["assign"], p1["head_at"]) == (p2["assign"], p2["head_at"]):
        raise SystemExit("refused: E2 is E1: %s" % (p1["assign"],))
    arms = {"E1": arm("PLANNER_V4's pick", p1, mem, refit, power),
            "E2": arm("PLANNER_V4's pick with %d MiB less free on the laptop" % LESS_MIB, p2, mem, refit, power)}
    for k, x in arms.items():
        if min(v["margin_mib"] for v in x["memory"].values()) < 0:
            raise SystemExit("refused: %s does not fit at rest: %s" % (k, x["memory"]))
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_e2_20261102.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_e2_20261102.py", "tag_prefix": "e2", "out_prefix": "e2",
           "what": "the laptop's refitted power rows with the session's own laptop idle (energy_refit laptop_idle_w), on Qwen3-32B "
                   "Q6_K, a model outside their fit: V4's pick and the pick with 6 GiB less on the laptop",
           "follows": "v4a (E-REFIT failed: the fixed idle); energy_holdout_20261101 (post hoc: with each session's own idle, a "
                      "median -1.2 %, 21 of 24 within 10 %)",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "reserve_mb": "",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           "cold_c": {"pi5": 50, "pi4": 55}, "arms": arms, "power_refit": refit["laptop"],
           "formula": "J(T, idle_s) = T x (boards_w_sum + idle_s + active_minus_idle_w x f_laptop + pool_host_w) + per_token_j; T in s",
           "idle_s_definition": "the laptop outlet's mean W over the session's IDLE_BEFORE window (trapezoid apower), all outlets on",
           "hypotheses": {
               "E2-REFIT": {"tol": 0.10, "claim": "each arm's J per token (all four outlets) within 10 % of J(T_pred, idle_s)"},
               "E2-POWER": {"tol": 0.10, "claim": "each arm's J per token within 10 % of J(T_meas, idle_s)"},
               "E2-BETTER": {"claim": "J(T_pred, idle_s)'s |error| below the old model's on every arm"},
               "E2-REPEAT": {"tol": 0.02, "claim": "each arm's two runs within 2 % of each other (server mean TPOT)"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/serve_plan_20261101.py", "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py",
               "testbed/scripts/shelly_strip.py", "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh",
               "testbed/scripts/score_e2_20261102.py", "testbed/scripts/register_e2_20261102.py",
               "testbed/dataset/power_refit_20261101.json", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    for k, x in arms.items():
        p = x["predicted"]
        print("%s %s head %s: %.1f ms; f_laptop %.3f; J at idle 17 W %.3f (fixed-idle refit %.3f, old %.3f); margins %s" % (
            k, x["place_env"]["PLACE"], x["head"], p["decode_ms"], p["formula"]["f_laptop"], p["j_at_idle"]["17.0"],
            p["j_per_token_all_refit_fixed_idle"], p["j_per_token_all_old"], {n: v["margin_mib"] for n, v in x["memory"].items()}))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
