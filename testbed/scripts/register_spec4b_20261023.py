#!/usr/bin/env python3
"""Register the draft length for speculative decoding on scheduler v2 (Qwen3-32B Q6_K, four nodes), before it is measured.

    python3 testbed/scripts/register_spec4b_20261023.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The user (2026-10-23 ~17:00): "pick best pieces to do and finish it on time for today's advisor report". spec4 (13:41) served
one drafted token per check: 1.288x, every hypothesis holding, with the check's cost measured on that plan for the first time.
This asks whether two or three drafted tokens do better. Writes testbed/dataset/spec4b_predictions_20261023.json, read-only.
Three arms on ONE placement (planned as spec4: the draft's 800 MiB inside the user's 7 GiB laptop reserve):
  S1, S2, S3  the Qwen3-0.6B draft on the laptop host's CPU (draft-simple, -devd none -ngld 0), d = 1, 2, 3 drafted tokens per
              check (--spec-draft-n-max d, --spec-draft-n-min 0, --spec-draft-p-min 0).
Prediction, from spec4's measurements and stated assumptions (nothing else measured):
  mean time per token(d) = T1 x (r(d + 1) + d t_d / T1) / tau(d), with T1 the predictor's decode for the plan with no draft;
  r(k) = 1 + 0.2972 (k - 1): a k-token check, linear in k from spec4's measured two-token check (1,150.46 ms x 1.6973 tokens per
  check, less the predicted 31.0 ms draft step, over V's 1,481.47 ms: r(2) = 1.2972); the linearity is the assumption;
  tau(d): spec4's measured tokens per check for d = 1 (1.6973: 129 of 185 drafted tokens accepted), and for d = 2, 3 the
  2026-10-10 Qwen3-14B values (2.2795, 2.7037) scaled by spec4's ratio to that campaign's d = 1 (1.6973 / 1.7281);
  t_d one draft step on the laptop (the 0.6B file's bytes at the laptop's per-byte Q8_0 decode cost).
Order S1 S2 S3 S3 S2 S1, three requests of 128 + 64 per arm-run, run by spec4_session_20261023.sh (tag_prefix "sl", scorer
score_spec4b_20261023.py). Speed is each request's decode window over its tokens (accepted tokens arrive in bursts).
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "spec4b_predictions_20261023.json")
KEY, CTX, NODES = "qwen3_32b_q6k", 2048, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
DRAFT_FILE, DRAFT_BYTES = os.path.expanduser("~/models/qwen3-0.6b-q8_0.gguf"), 639446688
DRAFT_SHA = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"   # testbed/models.md
DRAFT_HOST_MIB, RESERVE = 800, 7168
R_STEP = 0.2972                                             # spec4: (1150.46 x 1.6973 - 31.0) / 1481.47 - 1
TAU = {1: 1.6973}
TAU.update({d: t * 1.6973 / 1.7281 for d, t in ((2, 2.2795), (3, 2.7037))})   # spec_accept_20261010_013837, scaled


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
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none")
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
    arms, pred = {}, {}
    for d in (1, 2, 3):
        f = (1 + R_STEP * d + d * td / t1) / TAU[d]
        pred[d] = t1 * f
        arms["S%d" % d] = {"what": "the Qwen3-0.6B draft on the laptop CPU, %d drafted token%s per check" % (d, "" if d == 1 else "s"),
                           "assign": e["assign"], "head": head, "place_env": place,
                           "serve_env": {"SERVE_DRAFT": args_for(d), "DRAFT_HOST_MIB": str(DRAFT_HOST_MIB),
                                         "HETEROSERVE_RESERVE_MB": "laptop=%d" % RESERVE},
                           "predicted": {"mean_ms_per_token": round(pred[d], 1), "factor_of_t1": round(f, 4),
                                         "tau": round(TAU[d], 4), "check_over_t1": round(1 + R_STEP * d, 4),
                                         "decode_gross_j_all": round(jv * f, 4)}}
    best = min(pred, key=pred.get)
    deadline = None
    if a.deadline:
        t = datetime.datetime.fromisoformat(a.deadline)
        deadline = {"at": t.isoformat(), "epoch": int(t.timestamp()), "arm_min": {k: a.arm_min for k in arms},
                    "rule": "an arm-run starts only if, at its allowed minutes, it ends before the deadline; otherwise the session "
                            "ends there as at its normal end; an arm is complete with one arm-run's worth of requests (min_runs)"}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_spec4b_20261023.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_spec4b_20261023.py", "tag_prefix": "sl", "out_prefix": "spec4b",
           "what": "the draft length for speculative decoding on scheduler v2's plan (Qwen3-32B Q6_K, all four nodes): 1, 2 and 3 "
                   "drafted tokens per check, energy at the wall, all four outlets",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1],
           "draft": {"file": "~/models/qwen3-0.6b-q8_0.gguf", "sha256": dsha, "host_mib": DRAFT_HOST_MIB},
           "ctx": CTX, "nodes": NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "reserve_mb": "laptop=%d" % RESERVE, "reserve": "as spec4: the user's 7 GiB laptop reserve for the Qwen3-32B runs, the "
                                                         "draft's 800 MiB planned inside it",
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           **({"deadline": deadline, "min_runs": {k: 1 for k in arms}} if deadline else {}),
           **({"conditions": a.conditions} if a.conditions else {}), "cold_c": {"pi5": 50, "pi4": 55},
           "t1_predicted_ms": round(t1, 1), "draft_step_ms": round(td, 1), "arms": arms,
           "hypotheses": {
               "SL-D2": {"claim": "S2's mean time per token is below S1's (predicted %.1f against %.1f ms)" % (pred[2], pred[1])},
               "SL-D3": {"claim": "S3's mean time per token is below S1's (predicted %.1f against %.1f ms)" % (pred[3], pred[1])},
               "SL-BEST": {"predicted": "S%d" % best, "claim": "the arm predicted fastest (S%d) has the lowest mean time per token of "
                                                              "the three; S2 and S3 are predicted within 1 %%, so this is close" % best},
               "SL-ENERGY": {"claim": "the arm with the lowest mean time per token also has the lowest gross J per decoded token"},
               "SL-ACC": {"tol": 0.10, "claim": "each arm's mean time per token within 10 % of its prediction (T1 is the predictor's "
                                                "no-draft decode, which spec4 measured 7.2 % high)"},
               "SL-TEXT": {"claim": "request 0's first 16 tokens of text are the same in every arm-run"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_spec4b_20261023.py",
               "testbed/scripts/register_spec4b_20261023.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("t1_predicted_ms", "arms", "hypotheses")}, indent=1)[:4000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    print("plan %s head %s; T1 %.1f ms; predicted %s" % (place["PLACE"], head, t1,
                                                          ", ".join("S%d %.1f ms" % (d, pred[d]) for d in (1, 2, 3))))


if __name__ == "__main__":
    main()
