#!/usr/bin/env python3
"""Register the draft-length study a third time, as a NEW registration (spec4c, 2026-10-24, was INCOMPLETE: its arm-run 4,
S3's second, was refused by serve_pool.sh's memory gate when a third desktop app took ~1.8 GB of the laptop's memory; a
registration is never reused) -- before it is measured.

    python3 testbed/scripts/register_spec4d_20261024.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The design is spec4c's (register_spec4c_20261024.py): scheduler v2's plan for Qwen3-32B Q6_K on the four nodes, the
Qwen3-0.6B draft on the laptop host's CPU (draft-simple, -devd none -ngld 0), d = 1, 2, 3 drafted tokens per check (S1, S2,
S3), order S1 S2 S3 S3 S2 S1, three requests of 128 + 64 per arm-run, run by spec4_session_20261023.sh (tag_prefix "sd").
Scored by score_spec4d_20261024.py: spec4c's scorer with one fix (an arm-run that served no request has no server log of
its own; spec4c's pooled the next run's draft acceptance into its refused arm). Two changes to the design:
  HEADROOM: the plan is made with 1,024 MiB more laptop memory held back than the serve-time gate requires (planning reserve
  7,168 + the draft's 800 + 1,024 MiB; the gate keeps the user's 7 GiB + 800). spec4c's plan left 377 MiB at rest, and
  ~1.15 GB of drift in the laptop's other applications refused an arm. The plan may hold fewer layers on the laptop; the
  study compares draft lengths on one plan, which it still does.
  PREDICTIONS, INFORMED BY EVERY MEASURED ARM-RUN, stated as such: for each arm-run at draft length d, its mean time per
  token over its plan's no-draft decode, x = ms / T1(plan); the prediction is T1(this plan) x mean x at d.
    T1(23 October's plan) = 1,481.47 ms, measured (spec4 V); ASSUMED, as spec4c: another plan's T1 is that times the ratio
    of the predictor's no-draft decodes (23 October's 1,565.5 ms; spec4c's 1,682.0 ms; this plan's at registration).
    The measured arm-runs: spec4 S (d 1), spec4b S1 and S2 (d 1, 2), and spec4c's completed arm-runs as its score.json
    lists them (rc 0 with all requests).
  Energy per token: the cluster's mean power over the decode windows of every measured arm-run (J/token over s/token) times
  the predicted time.
Refuses to register unless the predicted order is S1 < S2 < S3, because the scorer's SC-ORDER (spec4c's) tests exactly
that order; a different order needs its own scorer. Writes testbed/dataset/spec4d_predictions_20261024.json, read-only.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "spec4d_predictions_20261024.json")
KEY, CTX, NODES = "qwen3_32b_q6k", 2048, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
DRAFT_FILE = os.path.expanduser("~/models/qwen3-0.6b-q8_0.gguf")
DRAFT_SHA = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"   # testbed/models.md
DRAFT_HOST_MIB, RESERVE, HEADROOM_MIB = 800, 7168, 1024
T1_MEAS = 1481.47                                           # spec4 V (score.txt), on 23 October's plan
T1_PRED = {"23oct": 1565.5, "spec4c": 1682.0}               # the predictor's no-draft decode of each measured plan
SPEC4C = "testbed/results/spec4c_20261024_004103/score.json"
MEAS_23OCT = [  # (d, ms/token, acceptance, J/token all, plan, source) as spec4c's registration quotes them
    (1, 1150.46, 0.6973, 51.4365, "23oct", "spec4_20261023_134113 S"),
    (1, 1186.86, 0.6261, 53.2659, "23oct", "spec4b_20261023_170315 S1"),
    (2, 1355.31, 0.4663, 60.5885, "23oct", "spec4b_20261023_170315 S2")]


def sha(p, root=True):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p) if root else p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def measured():
    """Every measured arm-run: 23 October's three, then spec4c's completed ones from its published score."""
    s = json.load(open(os.path.join(ROOT, SPEC4C)))
    reqs = json.load(open(os.path.join(ROOT, s["registration"])))["reqs"]
    out = list(MEAS_23OCT)
    for r in s["runs"]:
        if r["rc"] == 0 and r["requests"] >= reqs and r["mean_ms_per_token"] and r["j_per_token_all"]:
            acc = r["draft"]["accepted"] / r["draft"]["drafted"] if r.get("draft") and r["draft"].get("drafted") else None
            out.append((int(r["arm"][1:]), r["mean_ms_per_token"], acc, r["j_per_token_all"], "spec4c",
                        "spec4c_20261024_004103 run %d %s" % (r["i"], r["arm"])))
    return out


def predictions(t1_pred_now, meas):
    t1 = {k: T1_MEAS * v / T1_PRED["23oct"] for k, v in T1_PRED.items()}     # ASSUMED beyond 23oct, as stated
    t1_now = T1_MEAS * t1_pred_now / T1_PRED["23oct"]
    x = {d: [ms / t1[plan] for dd, ms, _, _, plan, _ in meas if dd == d] for d in (1, 2, 3)}
    if not all(x.values()):
        raise SystemExit("refused: no measured arm-run at some draft length: %s" % {d: len(v) for d, v in x.items()})
    ratio = {d: sum(v) / len(v) for d, v in x.items()}
    watts = [j / (ms / 1000) for _, ms, _, j, _, _ in meas]
    w = sum(watts) / len(watts)
    ms = {d: t1_now * ratio[d] for d in (1, 2, 3)}
    return ms, {d: w * ms[d] / 1000 for d in ms}, {
        "t1_ms_by_plan": {k: round(v, 1) for k, v in t1.items()}, "t1_ms_this_plan": round(t1_now, 1),
        "ratio_by_d": {d: round(r, 4) for d, r in ratio.items()}, "n_runs_by_d": {d: len(v) for d, v in x.items()},
        "watts_mean": round(w, 2), "watts_range": [round(min(watts), 2), round(max(watts), 2)]}


def args_for(d):
    return ("-md ~/models/qwen3-0.6b-q8_0.gguf --spec-type draft-simple --spec-draft-n-max %d --spec-draft-n-min 0 "
            "--spec-draft-p-min 0 -devd none -ngld 0" % d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="S1 S2 S3 S3 S2 S1"); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    dsha = sha(DRAFT_FILE, root=False)
    if dsha != DRAFT_SHA:
        raise SystemExit("refused: the draft's sha256 %s is not the pinned %s" % (dsha, DRAFT_SHA))
    os.environ["HETEROSERVE_RESERVE_MB"] = "laptop=%d" % (RESERVE + DRAFT_HOST_MIB + HEADROOM_MIB)
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", order_check=True)
    m = P.memory(e["assign"], e["q"], mem, e["devices"], e["model"], head_at=e["head_at"])
    if min(v["margin_mib"] for v in m.values()) < 0:
        raise SystemExit("refused: the plan does not fit at rest: %s" % m)
    t1_pred = e["decode_ms"]
    head = "host" if e["head_at"] == 0 else "last"
    place = {"PLACE": ",".join("%s:%d" % tuple(x) for x in e["assign"]), "PLACE_HEAD": head}
    meas = measured()
    pred, pj, model_terms = predictions(t1_pred, meas)
    if [x[0] for x in sorted(pred.items(), key=lambda kv: kv[1])] != [1, 2, 3]:
        raise SystemExit("refused: the predicted order is not S1 < S2 < S3 (%s); the scorer's SC-ORDER tests that "
                         "order, so this needs its own scorer" % {d: round(v, 1) for d, v in pred.items()})
    arms = {}
    for d in (1, 2, 3):
        arms["S%d" % d] = {"what": "the Qwen3-0.6B draft on the laptop CPU, %d drafted token%s per check" % (d, "" if d == 1 else "s"),
                           "assign": e["assign"], "head": head, "place_env": place,
                           "serve_env": {"SERVE_DRAFT": args_for(d), "DRAFT_HOST_MIB": str(DRAFT_HOST_MIB),
                                         "HETEROSERVE_RESERVE_MB": "laptop=%d" % RESERVE},
                           "predicted": {"mean_ms_per_token": round(pred[d], 1), "j_per_token_all": round(pj[d], 4),
                                         "basis": "T1 of this plan times the mean measured ms / T1(plan) at d = %d, over %s"
                                                  % (d, "; ".join(s for dd, _, _, _, _, s in meas if dd == d))}}
    memfile = os.path.relpath(os.path.abspath(a.memfile), ROOT)
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_spec4d_20261024.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_spec4d_20261024.py", "tag_prefix": "sd", "out_prefix": "spec4d",
           "supersedes": "testbed/dataset/spec4c_predictions_20261024.json (INCOMPLETE: arm-run 4 refused by the laptop's memory gate)",
           "what": "the draft length for speculative decoding on scheduler v2's plan (Qwen3-32B Q6_K, all four nodes): 1, 2 and 3 "
                   "drafted tokens per check, each arm twice, energy at the wall, all four outlets",
           "plan_options": {"order_check": True, "load_mode": "none", "host": "laptop",
                            "planning_reserve_mb": "laptop=%d" % (RESERVE + DRAFT_HOST_MIB + HEADROOM_MIB),
                            "headroom_mib": HEADROOM_MIB},
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1],
           "draft": {"file": "~/models/qwen3-0.6b-q8_0.gguf", "sha256": dsha, "host_mib": DRAFT_HOST_MIB},
           "ctx": CTX, "nodes": NODES, "outlets": OUTLETS, "memfile": memfile,
           "reserve_mb": "laptop=%d" % RESERVE,
           "reserve": "as spec4: the user's 7 GiB laptop reserve for the Qwen3-32B runs at the serve-time gate, the draft's 800 MiB "
                      "charged on top; the plan holds back 1,024 MiB more (HEADROOM)",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "t1_predictor_ms": round(t1_pred, 1), "t1_measured_ms": T1_MEAS,
           "measured_inputs": [{"d": d, "ms": ms, "acceptance": acc, "j_per_token_all": j, "plan": plan, "source": s}
                               for d, ms, acc, j, plan, s in meas],
           "model_terms": model_terms, "memory_at_rest_vs_planning_reserve_mib": {n: v["margin_mib"] for n, v in m.items()},
           "laptop_margin_vs_gate_mib": m["laptop"]["margin_mib"] + HEADROOM_MIB, "arms": arms,
           "hypotheses": {
               "SC-ORDER": {"claim": "mean time per token rises with the draft length: S1 < S2 < S3 (predicted %.1f, %.1f, %.1f ms)"
                                     % (pred[1], pred[2], pred[3])},
               "SC-BEST": {"predicted": "S1", "claim": "S1 has the lowest mean time per token of the three"},
               "SC-ENERGY": {"claim": "the arm with the lowest mean time per token also has the lowest gross J per decoded token"},
               "SC-ACC": {"tol": 0.10, "claim": "each arm's mean time per token within 10 % of its prediction"},
               "SC-TEXT": {"claim": "within each arm, request 0's first 16 tokens of text are the same in both of its arm-runs"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_spec4d_20261024.py",
               "testbed/scripts/register_spec4d_20261024.py", "testbed/scripts/score_prima4_20261021.py", SPEC4C,
               "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json", memfile)}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("model_terms", "memory_at_rest_vs_planning_reserve_mib", "laptop_margin_vs_gate_mib",
                                              "arms", "hypotheses")}, indent=1)[:6000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    print("plan %s head %s; predicted %s" % (place["PLACE"], head, ", ".join("S%d %.1f ms %.2f J" % (d, pred[d], pj[d])
                                                                               for d in (1, 2, 3))))


if __name__ == "__main__":
    main()
