#!/usr/bin/env python3
"""Register sp10 (2026-11-03; the scheduler review's item 10, the user: "... then i tem 9-10", then "continue"): the planner's
checking cost for speculation with no draft model, priced on two plans it was not fitted on.

    python3 testbed/scripts/register_sp10_20261103.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

The model (spec_model_20261103.py) comes from the exploration (explore_spec_20261102.py, 3 Nov, not registered):
  A  Qwen3.6-35B-A3B Q6_K on laptop 24 + Jetson 2 + Pi 5 9 + Pi 4 6 (head on the laptop; the MTP block on the Pi 4):
     MTP 1-3 were 1.12-1.35x faster than none; a batch of 2 cost 1.31x a decode step
     (testbed/results/explspec_20261103_143210);
  B  Qwen3.5-9B Q4_K_M on the laptop alone: MTP lost on free text (0.86-0.99x) and won on copying (1.01-1.10x); a batch of
     2 cost 1.38x a step (testbed/results/explspec_20261103_133920).
Plans the model has not seen, each served with no speculation (then the prompt-length scan, m 1-8), MTP 1, 2 and 3, three
requests of 128 tokens on each of the exploration's two prompts, order P2 then P1 (sp10_session_20261103.py):
  P2  the 35B with the Pi 5 last, laptop 24 + Jetson 2 + Pi 4 6 + Pi 5 9 (head on the laptop): the MTP block moves to the
      Pi 5, a third of the Pi 4's cost per block;
  P1  the 9B on laptop 24 + Pi 5 9 (head on the laptop): the MTP block on the Pi 5, a worker, and the check spread over the
      laptop and the Pi 5, with B's curve.
Predicted per plan and arm (spec_model.predict): the decode step, R(m), the draft cost and each arm's speedup per prompt.
Hypotheses (registered):
  SP-CHECK    each plan's R(m) = T(m) / its decode step, m 2, 3 and 4, within 10 % of predicted
  SP-SPEEDUP  each MTP arm's speedup over the plan's none (per prompt) within 7 % of predicted
  SP-SIGN     every arm predicted at least 5 % faster or slower than none measures on that side of 1
  SP-BEST     per plan and prompt, the setting predicted fastest (none counting 1.0) within 3 % of the fastest measured
  SP-DRAFT    each plan's measured draft cost within 20 % of predicted
Reported, not tested: the tokens per pass against the exploration's (greedy decoding on the same prompts), and each arm's
predicted speedup at its measured tokens per pass.
Writes testbed/dataset/sp10_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import spec_model_20261103 as SM                            # noqa: E402
import explore_spec_20261102 as X                           # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "sp10_predictions_20261103.json")
EXP = {"qwen36moe_q6k": ("testbed/results/explspec_20261103_143210", "A", [("laptop", 24), ("jetson", 2), ("pi5", 9), ("pi4", 6)], "host"),
       "qwen35_9b": ("testbed/results/explspec_20261103_133920", "B", [("laptop", 33)], "last")}
PLANS = {"P2": ("qwen36moe_q6k", [("laptop", 24), ("jetson", 2), ("pi4", 6), ("pi5", 9)], "host"),
         "P1": ("qwen35_9b", [("laptop", 24), ("pi5", 9)], "host")}
SPECS = ["none", "mtp1", "mtp2", "mtp3"]


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
    fits = {k: SM.exploration_fit(os.path.join(ROOT, f), c) for k, (f, c, _, _) in EXP.items()}
    terms = {k: SM.plan_terms(k, pl, hd, mem) for k, (_, _, pl, hd) in EXP.items()}
    kappa, rpc = SM.calibrate(fits["qwen35_9b"], terms["qwen35_9b"], fits["qwen36moe_q6k"], terms["qwen36moe_q6k"])
    preds, configs = {}, {}
    for c, (key, pl, hd) in PLANS.items():
        tn = SM.plan_terms(key, pl, hd, mem)
        p = SM.predict(fits[key], terms[key], tn, kappa, rpc)
        p["R"] = {str(m): round(v, 4) for m, v in p["R"].items() if m <= 8}
        p.update(terms=tn, exploration={"folder": fits[key]["folder"], "config": fits[key]["config"]})
        preds[c] = p
        configs[c] = {"MODEL": key, "CTX": "4096", "PLACE": ",".join("%s:%d" % x for x in pl), "PLACE_HEAD": hd, "specs": SPECS}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_sp10_20261103.py", "session": "testbed/scripts/sp10_session_20261103.py",
           "scorer": "testbed/scripts/score_sp10_20261103.py", "out_prefix": "sp10",
           "what": "the scheduler review's item 10: the planner's checking cost for MTP speculation (spec_model_20261103.py), "
                   "fitted on the 3 Nov exploration's plans A and B, priced on two plans it has not seen",
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "order": list(PLANS), "configs": configs,
           "spec_args": {k: X.SPEC[k] for k in SPECS}, "prompts": X.PROMPTS, "prompt_order": ["free", "copy"], "reqs": 3,
           "gen": 128, "scan": [1, 2, 3, 4, 6, 8], "scan_reps": 3,
           "model": {"kappa": round(kappa, 4), "rpc_draft_ms": round(rpc, 2),
                     "fits": {k: {x: f[x] for x in ("folder", "config", "decode_ms", "draft_ms", "intercept_ms", "residuals", "E")}
                              for k, f in fits.items()},
                     "exploration_terms": terms},
           "predictions": preds,
           "hypotheses": {
               "SP-CHECK": {"m": [2, 3, 4], "tol": 0.10, "claim": "each plan's R(m) = T(m) / its decode step, m 2-4, within 10 % of predicted"},
               "SP-SPEEDUP": {"tol": 0.07, "claim": "each MTP arm's speedup over the plan's none, per prompt, within 7 % of predicted"},
               "SP-SIGN": {"margin": 0.05, "claim": "every arm predicted at least 5 % faster or slower than none measures on that side of 1"},
               "SP-BEST": {"tol": 0.03, "claim": "per plan and prompt, the setting predicted fastest within 3 % of the fastest measured"},
               "SP-DRAFT": {"tol": 0.20, "claim": "each plan's measured draft cost within 20 % of predicted"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/sp10_session_20261103.py", "testbed/scripts/score_sp10_20261103.py",
               "testbed/scripts/register_sp10_20261103.py", "testbed/scripts/spec_model_20261103.py",
               "testbed/scripts/spec_cost_20261103.py", "testbed/scripts/explore_spec_20261102.py",
               "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh", "testbed/scripts/serve_plan_20261101.py",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               "testbed/results/explspec_20261103_143210/results.json", "testbed/results/explspec_20261103_133920/results.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    print("KAPPA %.4f, RPC_DRAFT_MS %.2f" % (kappa, rpc))
    for c, p in preds.items():
        print("%s %s %s: decode %.1f ms, draft %.1f ms, R %s; speedups %s" % (
            c, configs[c]["MODEL"], configs[c]["PLACE"], p["decode_ms"], p["draft_ms"], p["R"],
            {k: v["speedup"] for k, v in p["arms"].items()}))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
