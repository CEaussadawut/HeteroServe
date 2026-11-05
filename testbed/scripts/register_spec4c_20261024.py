#!/usr/bin/env python3
"""Register the draft-length study again, complete, as a NEW registration (spec4b, 2026-10-23, was INCOMPLETE: its S3 never
ran before its deadline; a registration is never reused) -- before it is measured.

    python3 testbed/scripts/register_spec4c_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The user (2026-10-24 ~00:30): "Do all lefts". Scheduler v2's plan for Qwen3-32B Q6_K on all four nodes, the Qwen3-0.6B draft
on the laptop host's CPU (draft-simple, -devd none -ngld 0), d = 1, 2, 3 drafted tokens per check (S1, S2, S3), order
S1 S2 S3 S3 S2 S1 (each arm twice), three requests of 128 + 64 per arm-run, run by spec4_session_20261023.sh (tag_prefix
"sc", scorer score_spec4c_20261024.py). Writes testbed/dataset/spec4c_predictions_20261024.json, read-only.
Predictions are INFORMED BY THE 23 OCTOBER MEASUREMENTS, stated as such (nothing new is assumed beyond the two lines marked):
  mean time per token(d) = (T1 r(d + 1) + d t_d) / tau(d), T1 = spec4's measured no-draft decode on this plan (1,481.47 ms),
  t_d = 31.0 ms (the predicted draft step of spec4b's registration);
  r(k), the cost of a k-token check over T1, solved from each measured arm: spec4 S (d 1) 1.2972, spec4b S1 (d 1) 1.2818,
  spec4b S2 (d 2) 1.7262; r(2) is their d-1 mean; ASSUMED: r(4) = r(3) + (r(3) - r(2)) (linear past the measured points);
  tau(d) = 1 + d x the acceptance rate, measured for d 1 (spec4 0.6973, spec4b 0.6261) and d 2 (spec4b 0.4663); ASSUMED:
  each drafted position accepted with one probability p, fitted to d 2 (1 + p + p^2 = 1.9326, p = 0.5875), so
  tau(3) = 1.9326 + p^3.
  r(2) and tau(1) are the means of the two d-1 sessions. Energy per token: the cluster's measured power over the decode
  windows (J/token over ms/token) was 44.7-44.9 W in all three measured arms, so J(d) = 44.8 W x T(d).
  ASSUMED, the plan: the laptop's MemAvailable at registration (26,051 MiB on 2026-10-24 00:39, a second Claude desktop app
  open) is 1.5 GB below 23 October's 27,523, so under the same 7 GiB reserve + the draft's 800 MiB the planner now gives
  laptop 43 + Pi 5 18 + Jetson 3 (predicted no-draft decode 1,682.0 ms) where 23 October ran laptop 47 + Jetson 1 + Pi 5 16
  (1,565.5 ms). T1 is scaled by that ratio of the predictor's decodes; r, tau and t_d are kept.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "spec4c_predictions_20261024.json")
KEY, CTX, NODES = "qwen3_32b_q6k", 2048, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
DRAFT_FILE, DRAFT_BYTES = os.path.expanduser("~/models/qwen3-0.6b-q8_0.gguf"), 639446688
DRAFT_SHA = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"   # testbed/models.md
DRAFT_HOST_MIB, RESERVE = 800, 7168
T1_MEAS, TD = 1481.47, 31.0                                # spec4 V (score.txt); spec4b's predicted draft step
MEAS = {1: [(1150.46, 0.6973, 51.4365, "spec4_20261023_134113 S"), (1186.86, 0.6261, 53.2659, "spec4b_20261023_170315 S1")],
        2: [(1355.31, 0.4663, 60.5885, "spec4b_20261023_170315 S2")]}   # (ms/token, acceptance, J/token all, source)
WATTS = 44.8                                                # J/token over s/token in all three measured arms: 44.7-44.9 W


def r_of(ms, acc, d):
    return (ms * (1 + d * acc) - d * TD) / T1_MEAS


T1_PRED_23OCT = 1565.5                                      # the predictor's decode for 23 Oct's plan (spec4b's registration)


def predictions(t1_pred_now):
    scale = t1_pred_now / T1_PRED_23OCT                     # ASSUMED: the plan's no-draft decode moves as the predictor's
    t1 = T1_MEAS * scale
    r2 = sum(r_of(ms, acc, 1) for ms, acc, _, _ in MEAS[1]) / len(MEAS[1])
    r3 = r_of(MEAS[2][0][0], MEAS[2][0][1], 2)
    r4 = r3 + (r3 - r2)                                     # ASSUMED: linear past the measured points
    tau2 = 1 + 2 * MEAS[2][0][1]
    p = (-1 + (1 + 4 * (tau2 - 1)) ** 0.5) / 2              # ASSUMED: one acceptance probability per drafted position
    tau3 = tau2 + p ** 3
    tau1 = 1 + sum(m[1] for m in MEAS[1]) / len(MEAS[1])
    r, tau = {1: r2, 2: r3, 3: r4}, {1: tau1, 2: tau2, 3: tau3}
    ms = {d: (t1 * r[d] + d * TD) / tau[d] for d in (1, 2, 3)}
    j = {d: WATTS * ms[d] / 1000 for d in (1, 2, 3)}
    return ms, j, {"t1_scale_assumed": round(scale, 4), "t1_ms": round(t1, 1), "r2": round(r2, 4), "r3": round(r3, 4),
                   "r4_assumed": round(r4, 4), "p_fitted": round(p, 4),
                   "tau": {1: round(tau1, 4), 2: round(tau2, 4), 3: round(tau3, 4)}}


def args_for(d):
    return ("-md ~/models/qwen3-0.6b-q8_0.gguf --spec-type draft-simple --spec-draft-n-max %d --spec-draft-n-min 0 "
            "--spec-draft-p-min 0 -devd none -ngld 0" % d)


def sha(p, root=True):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p) if root else p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="S1 S2 S3 S3 S2 S1"); ap.add_argument("--deadline", default="")
    ap.add_argument("--arm-min", type=int, default=25); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    dsha = sha(DRAFT_FILE, root=False)
    if dsha != DRAFT_SHA:
        raise SystemExit("refused: the draft's sha256 %s is not the pinned %s" % (dsha, DRAFT_SHA))
    os.environ["HETEROSERVE_RESERVE_MB"] = "laptop=%d" % (RESERVE + DRAFT_HOST_MIB)
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", order_check=True)
    m = P.memory(e["assign"], e["q"], mem, e["devices"], e["model"], head_at=e["head_at"])
    if min(v["margin_mib"] for v in m.values()) < 0:
        raise SystemExit("refused: the plan does not fit at rest: %s" % m)
    power = EM.power_table()
    en = EM.energy(e["assign"], e["q"], e["C"], e["links"], power, "tg", head_at=e["head_at"], rtt_trips=P.RTT_TRIPS_DEFAULT)
    idle = sum(power[n]["idle"] for n in NODES if n not in dict(e["assign"])) * en["T_ms"] / 1000
    jv, t1 = en["gross_j"] + idle, e["decode_ms"]
    td = DRAFT_BYTES * e["C"]["laptop"]["host"][0]
    head = "host" if e["head_at"] == 0 else "last"
    place = {"PLACE": ",".join("%s:%d" % tuple(x) for x in e["assign"]), "PLACE_HEAD": head}
    arms = {}
    pred, pj, model_terms = predictions(t1)
    for d in (1, 2, 3):
        arms["S%d" % d] = {"what": "the Qwen3-0.6B draft on the laptop CPU, %d drafted token%s per check" % (d, "" if d == 1 else "s"),
                           "assign": e["assign"], "head": head, "place_env": place,
                           "serve_env": {"SERVE_DRAFT": args_for(d), "DRAFT_HOST_MIB": str(DRAFT_HOST_MIB),
                                         "HETEROSERVE_RESERVE_MB": "laptop=%d" % RESERVE},
                           "predicted": {"mean_ms_per_token": round(pred[d], 1), "j_per_token_all": round(pj[d], 4),
                                         "basis": ("r and tau from %s" % " and ".join(m[3] for m in MEAS[1])) if d == 1 else
                                                  ("r and tau from %s" % MEAS[2][0][3]) if d == 2 else
                                                  "r(4) and tau(3) extrapolated as above"}}
    best = min(pred, key=pred.get)
    if [x[0] for x in sorted(pred.items(), key=lambda kv: kv[1])] != [1, 2, 3]:
        raise SystemExit("the predicted order is not S1 < S2 < S3: %s (the hypotheses below assume it)" % pred)
    deadline = None
    if a.deadline:
        t = datetime.datetime.fromisoformat(a.deadline)
        deadline = {"at": t.isoformat(), "epoch": int(t.timestamp()), "arm_min": {k: a.arm_min for k in arms},
                    "rule": "an arm-run starts only if, at its allowed minutes, it ends before the deadline; otherwise the session "
                            "ends there as at its normal end; an arm is complete with one arm-run's worth of requests (min_runs)"}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_spec4c_20261024.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_spec4c_20261024.py", "tag_prefix": "sc", "out_prefix": "spec4c",
           "supersedes": "testbed/dataset/spec4b_predictions_20261023.json (INCOMPLETE: S3 never ran)",
           "what": "the draft length for speculative decoding on scheduler v2's plan (Qwen3-32B Q6_K, all four nodes): 1, 2 and 3 "
                   "drafted tokens per check, each arm twice, energy at the wall, all four outlets",
           "plan_options": {"order_check": True, "load_mode": "none", "host": "laptop"},
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1],
           "draft": {"file": "~/models/qwen3-0.6b-q8_0.gguf", "sha256": dsha, "host_mib": DRAFT_HOST_MIB},
           "ctx": CTX, "nodes": NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "reserve_mb": "laptop=%d" % RESERVE, "reserve": "as spec4: the user's 7 GiB laptop reserve for the Qwen3-32B runs, the "
                                                         "draft's 800 MiB planned inside it",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"deadline": deadline, "min_runs": {k: 1 for k in arms}} if deadline else {}),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "t1_predictor_ms": round(t1, 1), "t1_measured_ms": T1_MEAS, "draft_step_ms": TD, "measured_inputs": MEAS,
           "model_terms": model_terms, "arms": arms,
           "hypotheses": {
               "SC-ORDER": {"claim": "mean time per token rises with the draft length: S1 < S2 < S3 (predicted %.1f, %.1f, %.1f ms)"
                                     % (pred[1], pred[2], pred[3])},
               "SC-BEST": {"predicted": "S%d" % best, "claim": "S%d has the lowest mean time per token of the three" % best},
               "SC-ENERGY": {"claim": "the arm with the lowest mean time per token also has the lowest gross J per decoded token"},
               "SC-ACC": {"tol": 0.10, "claim": "each arm's mean time per token within 10 % of its prediction"},
               "SC-TEXT": {"claim": "within each arm, request 0's first 16 tokens of text are the same in both of its arm-runs "
                                    "(spec4b's S1 and S2 texts differed from each other, so text is compared within an arm)"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_spec4c_20261024.py",
               "testbed/scripts/register_spec4c_20261024.py", "testbed/scripts/score_prima4_20261021.py",
               "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("model_terms", "arms", "hypotheses")}, indent=1)[:5000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    print("plan %s head %s; predicted %s" % (place["PLACE"], head, ", ".join("S%d %.1f ms %.2f J" % (d, pred[d], pj[d])
                                                                               for d in (1, 2, 3))))


if __name__ == "__main__":
    main()
