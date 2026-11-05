#!/usr/bin/env python3
"""Register sp12 (2026-11-04; the user: "Test MTP 1 on the 35B across prompts", with "I'll close apps for it"): sp11's test on
the 35B, the case sp11 dropped before registering when no plan held Qwen3.6-35B-A3B with the laptop's 8 GiB reserve.

    python3 testbed/scripts/register_sp12_20261104.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Why. MTP 1 is the default when a person serves a pooled plan of qwen35_9b or qwen36moe_q6k (4 Nov). sp11 found it faster on
all 12 prompts on both pooled 9B plans (geometric means 1.155 and 1.274), the checking-cost model within 3.4 % on 23 of 24 arms.
On the 35B only sp10 / sp10c's two prompts were measured (MTP 1-3 gained 1.12-1.43x).
One plan, Q35: Qwen3.6-35B-A3B Q6_K on the plan a person serving it from a terminal gets now (serve_plan_20261101.choose,
PLANNER_V4, the four nodes, ctx 4,096, depth 0, no remembered plan), on this registration's memory snapshot, the laptop's
reserve kept at 8 GiB (the owner closes apps for the run instead). No plan holding the 35B: nothing is registered (refused).
Served with no speculation and with MTP 1 (the default's arguments), sp11's 12 prompts of 128 greedy tokens, one request each
(sp11_session_20261104.py, request_timeout_s 900); then, NOT timed, each prompt once more with the top two probabilities of
every plain token (n_probs 2, before sampling).
The checking-cost model, unchanged: Q35 priced as sp11 would have priced it (spec_model_20261103.predict from the exploration's
A, as sp10c's P3). MTP 1's speedup at tokens per pass E is decode_ms x E / pass_ms; it breaks even at E* = pass_ms / decode_ms.
Hypotheses, sp11's on this plan (score_sp11_20261104.py):
  MT-COST   each prompt's speedup within 5 % of the model's at that prompt's measured E
  MT-SIGN   MTP 1 faster than none exactly on the prompts whose measured E exceeds E* (prompts within 2 % of even left out)
  MT-MEAN   the geometric mean of the 12 speedups at least 1.00
  MT-WORST  no prompt more than 10 % slower under MTP 1
  TX-TIE    every first point where Q35's MTP 1 text of a prompt leaves its plain text is a near-tie there: the plain top two
            probabilities within 0.10 of each other
Writes testbed/dataset/sp12_predictions_20261104.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import spec_model_20261103 as SM                            # noqa: E402
import explore_spec_20261102 as X                           # noqa: E402
import serve_plan_20261101 as SP                            # noqa: E402
import predict_pool as P                                    # noqa: E402
import register_sp11_20261104 as R11                        # noqa: E402  (its prompts, explorations and settings)

OUT = os.path.join(ROOT, "testbed", "dataset", "sp12_predictions_20261104.json")


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    fits = {k: SM.exploration_fit(os.path.join(ROOT, f), c) for k, (f, c, _, _) in R11.EXP.items()}
    terms = {k: SM.plan_terms(k, pl, hd, mem) for k, (_, _, pl, hd) in R11.EXP.items()}
    kappa, rpc = SM.calibrate(fits["qwen35_9b"], terms["qwen35_9b"], fits["qwen36moe_q6k"], terms["qwen36moe_q6k"])
    try:
        r, why = SP.choose("qwen36moe_q6k", ["laptop", "pi5", "pi4", "jetson"], 4096, mem, 0, last=None, **P.PLANNER_V4)
    except ValueError as ex:
        p3 = P.plan_fixed("qwen36moe_q6k", [("laptop", 24), ("pi5", 9), ("pi4", 6), ("jetson", 2)], "host", 4096, mem,
                          load_mode="none", **P.PLANNER_V4)
        raise SystemExit("refused: no subset of the cluster holds Qwen3.6-35B-A3B Q6_K with the laptop's 8 GiB reserve on this "
                         "snapshot (the laptop's MemAvailable %d MiB; sp10c's P3 placement leaves it %d MiB short). Nothing "
                         "registered. Planner: %s" % (mem["laptop"]["mem_available_mib"],
                                                      -SP.fits(p3, mem)[1]["laptop"]["margin_mib"], str(ex)[:200]))
    q35 = [(n, int(k)) for n, k in r["assign"]]
    head35 = "host" if r["head_at"] == 0 else "last"
    t35 = SM.plan_terms("qwen36moe_q6k", q35, head35, mem)
    src = SM.predict(fits["qwen36moe_q6k"], terms["qwen36moe_q6k"], t35, kappa, rpc)
    arm = src["arms"]["mtp1/free"]
    preds = {"Q35": {"decode_ms": src["decode_ms"], "pass_ms": arm["pass_ms"],
                     "E_breakeven": round(arm["pass_ms"] / src["decode_ms"], 4),
                     "speedup_at_registered_E": {pk: src["arms"]["mtp1/" + pk]["speedup"] for pk in ("free", "copy")},
                     "registered_E": {pk: src["arms"]["mtp1/" + pk]["E"] for pk in ("free", "copy")},
                     "from": "spec_model_20261103.predict on the exploration's A, as sp10c's P3", "terms": t35}}
    configs = {"Q35": {"MODEL": "qwen36moe_q6k", "CTX": "4096", "PLACE": ",".join("%s:%d" % x for x in q35), "PLACE_HEAD": head35,
                       "specs": R11.SPECS, "nprobs": True, "planned": why, "predicted_decode_ms": round(r["decode_ms"], 1)}}
    margins = {n: v["margin_mib"] for n, v in SP.fits(r, mem)[1].items()} if hasattr(SP, "fits") else None
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_sp12_20261104.py", "session": "testbed/scripts/sp11_session_20261104.py",
           "scorer": "testbed/scripts/score_sp11_20261104.py", "out_prefix": "sp12", "request_timeout_s": 900,
           "what": "the MTP 1 default across sp11's 12 prompts on the 35B's interactive plan, and the texts behind its acceptance",
           "follows": {"registrations": ["testbed/dataset/sp11_predictions_20261104.json",
                                         "testbed/dataset/sp10c_predictions_20261103.json"],
                       "why": "sp11 dropped its 35B case before registering: no plan held it with the laptop's 8 GiB reserve"},
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "memory_margins_mib": margins,
           "order": list(configs), "configs": configs,
           "spec_args": {k: X.SPEC[k] for k in R11.SPECS}, "prompts": R11.PROMPTS, "prompt_order": list(R11.PROMPTS), "gen": 128,
           "model": {"kappa": round(kappa, 4), "rpc_draft_ms": round(rpc, 2)},
           "conditions": "the owner closed apps for this run (the laptop's 8 GiB reserve kept; the snapshot's MemAvailable %d MiB); "
                         "aunsync.timer (a Google Drive sync, every 16 min, ~20 s) runs; the laptop's quietness is not gated during "
                         "the session (speed ratios within one plan's serves, not energy)" % mem["laptop"]["mem_available_mib"],
           "predictions": preds,
           "hypotheses": {
               "MT-COST": {"tol": 0.05, "claim": "each prompt's MTP 1 speedup within 5 % of the model's at its measured E"},
               "MT-SIGN": {"dead_band": 0.02, "claim": "MTP 1 faster than none exactly where the measured E exceeds E*"},
               "MT-MEAN": {"min": 1.00, "claim": "the geometric mean of the 12 speedups at least 1.00"},
               "MT-WORST": {"min": 0.90, "claim": "no prompt more than 10 % slower under MTP 1"},
               "TX-TIE": {"gap": 0.10, "claim": "every first divergence of Q35's MTP 1 text from its plain text is a near-tie there "
                                                "(the plain top-two probabilities within 0.10)"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/sp11_session_20261104.py", "testbed/scripts/score_sp11_20261104.py",
               "testbed/scripts/register_sp12_20261104.py", "testbed/scripts/register_sp11_20261104.py",
               "testbed/scripts/spec_model_20261103.py", "testbed/scripts/explore_spec_20261102.py",
               "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh", "testbed/scripts/serve_plan_20261101.py",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               "testbed/results/explspec_20261103_143210/results.json", "testbed/results/explspec_20261103_133920/results.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    print("Q35: %s (head %s), %s; predicted decode %.1f ms; margins %s" % (configs["Q35"]["PLACE"], head35, why, r["decode_ms"],
                                                                         margins))
    p = preds["Q35"]
    print("Q35: decode %.1f ms, MTP 1 pass %.1f ms, E* %.3f, speedup at the registered E %s" % (
        p["decode_ms"], p["pass_ms"], p["E_breakeven"], p["speedup_at_registered_E"]))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
