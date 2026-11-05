#!/usr/bin/env python3
"""Register speculative decoding on top of scheduler v2 (Qwen3-32B Q6_K, all four nodes), before anything is measured.

    python3 testbed/scripts/register_spec4_20261023.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The advisor (2026-10-16) wanted speculative decoding in the comparison; the user (2026-10-23 ~10:40): "run everything else
now". Writes testbed/dataset/spec4_predictions_20261023.json, read-only; refuses to overwrite one. Two arms, ONE placement
(planned with the draft's host charge inside the laptop's reserve, so both arms serve the same plan):
  V  scheduler v2's model, Qwen3-32B Q6_K, the engine's pick, no draft (SERVE_DRAFT=none);
  S  the same, with the Qwen3-0.6B draft (sha-pinned, testbed/models.md) on the laptop -- the host -- CPU: draft-simple, one
     drafted token per check (the 2026-10-10 campaign's setting), -devd none -ngld 0, its host charge 800 MiB (serve_pool.sh
     DRAFT_HOST_MIB, its documented figure for a draft loaded on the host).
The laptop's reserve is the user's 7 GiB for the Qwen3-32B runs (2026-10-21); the plan is made with 7,168 + 800 MiB.
Prediction, stated and NOT measured on this plan: S's speed-up = tau(1) / (T(2)/T(1) + t_d/T(1)), with tau(1) = 1.728 tokens
per check (Qwen3-0.6B drafting for Qwen3-14B, spec_accept_20261010) and T(2)/T(1) = 1.228 (a two-token check on Qwen3-14B,
Pi 4 + Jetson, spec_pass_20261010) -- neither measured for this target or this plan -- and t_d one draft step on the laptop
(the 0.6B file's bytes at the laptop's per-byte Q8_0 decode cost). S's energy per token is predicted as V's over the speed-up
(the same average power): an assumption, stated.
Speed is compared by the MEAN client time per token (each request's decode window over its gaps, pooled): with a draft,
accepted tokens arrive in bursts, so a gap's P50 says little. Scored by score_spec4_20261023.py.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "spec4_predictions_20261023.json")
KEY, CTX, NODES = "qwen3_32b_q6k", 2048, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}
DRAFT_FILE, DRAFT_BYTES = os.path.expanduser("~/models/qwen3-0.6b-q8_0.gguf"), 639446688
DRAFT_SHA = "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031"   # testbed/models.md
DRAFT_ARGS = ("-md ~/models/qwen3-0.6b-q8_0.gguf --spec-type draft-simple --spec-draft-n-max 1 --spec-draft-n-min 0 "
              "--spec-draft-p-min 0 -devd none -ngld 0")
DRAFT_HOST_MIB, RESERVE = 800, 7168
TAU1, R2 = 1.7281, 1.228                                   # spec_accept_20261010_013837, spec_pass_20261010_011447


def sha(p, root=True):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p) if root else p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=5)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--order", default="V S V S"); ap.add_argument("--deadline", default="")
    ap.add_argument("--arm-min", type=int, default=40); ap.add_argument("--conditions", default="")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    dsha = sha(DRAFT_FILE, root=False)
    if dsha != DRAFT_SHA:
        raise SystemExit("refused: the draft's sha256 %s is not the pinned %s" % (dsha, DRAFT_SHA))
    os.environ["HETEROSERVE_RESERVE_MB"] = "laptop=%d" % (RESERVE + DRAFT_HOST_MIB)      # the draft charged inside the reserve
    mem = json.load(open(a.memfile))
    e = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none")
    m = P.memory(e["assign"], e["q"], mem, e["devices"], e["model"], head_at=e["head_at"])
    if min(v["margin_mib"] for v in m.values()) < 0:
        raise SystemExit("refused: the plan does not fit at rest: %s" % m)
    power = EM.power_table()
    en = EM.energy(e["assign"], e["q"], e["C"], e["links"], power, "tg", head_at=e["head_at"], rtt_trips=P.RTT_TRIPS_DEFAULT)
    idle = sum(power[n]["idle"] for n in NODES if n not in dict(e["assign"])) * en["T_ms"] / 1000
    jv = en["gross_j"] + idle
    t1 = e["decode_ms"]
    td = DRAFT_BYTES * e["C"]["laptop"]["host"][0]                 # ms: the 0.6B's bytes at the laptop's Q8_0 per-byte cost
    spd = TAU1 / (R2 + td / t1)
    head = "host" if e["head_at"] == 0 else "last"
    place = {"PLACE": ",".join("%s:%d" % tuple(x) for x in e["assign"]), "PLACE_HEAD": head}
    order = a.order.split()
    deadline = None
    if a.deadline:
        t = datetime.datetime.fromisoformat(a.deadline)
        deadline = {"at": t.isoformat(), "epoch": int(t.timestamp()), "arm_min": {"V": a.arm_min, "S": a.arm_min},
                    "rule": "an arm-run starts only if, at its allowed minutes, it ends before the deadline; otherwise the session "
                            "ends there as at its normal end; an arm is complete with one arm-run's worth of requests (min_runs)"}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_spec4_20261023.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "what": "speculative decoding on scheduler v2's plan (Qwen3-32B Q6_K, all four nodes): the Qwen3-0.6B draft on the "
                   "laptop host against no draft, energy at the wall, all four outlets",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1],
           "draft": {"file": "~/models/qwen3-0.6b-q8_0.gguf", "sha256": dsha, "args": DRAFT_ARGS, "host_mib": DRAFT_HOST_MIB},
           "ctx": CTX, "nodes": NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "reserve_mb": "laptop=%d" % RESERVE,
           "reserve": "the user's 7 GiB laptop reserve for the Qwen3-32B runs (2026-10-21); planned with the draft's 800 MiB "
                      "inside it (laptop=%d), served with laptop=%d plus serve_pool.sh's DRAFT_HOST_MIB in arm S" % (
                          RESERVE + DRAFT_HOST_MIB, RESERVE),
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": order,
           **({"deadline": deadline, "min_runs": {"V": 1, "S": 1}} if deadline else {}),
           **({"conditions": a.conditions} if a.conditions else {}),
           "cold_c": {"pi5": 50, "pi4": 55},
           "arms": {"V": {"what": "scheduler v2's model, the engine's pick, no draft", "assign": e["assign"], "head": head,
                          "place_env": place, "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": "laptop=%d" % RESERVE},
                          "predicted": {"decode_ms": round(t1, 1), "prefill_ms": round(e["prefill_ms"], 1),
                                        "decode_gross_j_all": round(jv, 4)},
                          "memory": m, "considered": e["considered"]},
                    "S": {"what": "the same plan with the Qwen3-0.6B draft on the laptop CPU, one drafted token per check",
                          "assign": e["assign"], "head": head, "place_env": place,
                          "serve_env": {"SERVE_DRAFT": DRAFT_ARGS, "DRAFT_HOST_MIB": str(DRAFT_HOST_MIB),
                                        "HETEROSERVE_RESERVE_MB": "laptop=%d" % RESERVE},
                          "predicted": {"decode_ms": round(t1 / spd, 1), "speedup": round(spd, 3),
                                        "decode_gross_j_all": round(jv / spd, 4), "draft_step_ms": round(td, 1),
                                        "tau1": TAU1, "t2_over_t1": R2}}},
           "hypotheses": {
               "SP-FASTER": {"claim": "S's mean client time per token is below V's"},
               "SP-PRED": {"tol": 0.15, "predicted": round(spd, 3),
                           "claim": "S's speed-up (V's mean / S's mean) within 15 %% of the predicted %.2fx (its inputs were measured "
                                    "on another target and plan, so the tolerance is wide)" % spd},
               "SP-ENERGY": {"claim": "S's gross J per decoded token at all four outlets is below V's"},
               "SP-ACC": {"tol": 0.10, "claim": "V's server mean time per token within 10 % of its predicted decode"},
               "SP-TEXT": {"claim": "request 0's first 16 tokens of text are the same in every V and S run (greedy; the others "
                                    "recorded); a batched check can flip a near-tie, so a FAIL is read with the texts"}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_spec4_20261023.py",
               "testbed/scripts/register_spec4_20261023.py", "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv",
               "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("arms", "hypotheses")}, indent=1)[:4000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    print("plan %s head %s: V %.1f ms/token, %.2f J | S predicted %.1f ms (%.2fx, draft step %.1f ms), %.2f J" % (
        place["PLACE"], head, t1, jv, t1 / spd, spd, td, jv / spd))


if __name__ == "__main__":
    main()
