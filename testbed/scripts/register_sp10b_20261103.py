#!/usr/bin/env python3
"""Register sp10b (2026-11-03): sp10's plan P2 again, complete. sp10 (register_sp10_20261103.py, registered 15:13) lost P2's
MTP 3 arm: its server's connection to the Pi 5, which holds the MTP block on P2, hung at 15:41:40 and the server aborted at
16:07:34 ("Remote RPC server crashed or returned malformed response"), so sp10 can only score INCOMPLETE
(testbed/results/sp10_20261103_151444).

    python3 testbed/scripts/register_sp10b_20261103.py --memfile testbed/dataset/memory_snapshot_20261103_125622.json [--dry-run]

P2 as in sp10: Qwen3.6-35B-A3B Q6_K on laptop 24 + Jetson 2 + Pi 4 6 + Pi 5 9 (head on the laptop), served with no
speculation (then the prompt-length scan, m 1-8), MTP 1, 2 and 3, three requests of 128 tokens on each of the exploration's two
prompts (sp10_session_20261103.py, a request that times out after request_timeout_s ending its arm's requests). The predictions
are re-derived by register_sp10's code from the same inputs (the model, the explorations, the snapshot), and the script
refuses unless they equal sp10's registered P2 predictions. Hypotheses: sp10's, on P2's rows: SP-CHECK (R within 10 %),
SP-SPEEDUP (within 7 %), SP-SIGN, SP-BEST (3 %), SP-DRAFT (20 %). Scorer: score_sp10_20261103.py.
Writes testbed/dataset/sp10b_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import register_sp10_20261103 as R                          # noqa: E402
import spec_model_20261103 as SM                            # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "sp10b_predictions_20261103.json")
SP10 = os.path.join(ROOT, "testbed", "dataset", "sp10_predictions_20261103.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    old = json.load(open(SP10))
    mem = json.load(open(a.memfile))
    fits = {k: SM.exploration_fit(os.path.join(ROOT, f), c) for k, (f, c, _, _) in R.EXP.items()}
    terms = {k: SM.plan_terms(k, pl, hd, mem) for k, (_, _, pl, hd) in R.EXP.items()}
    kappa, rpc = SM.calibrate(fits["qwen35_9b"], terms["qwen35_9b"], fits["qwen36moe_q6k"], terms["qwen36moe_q6k"])
    key, pl, hd = R.PLANS["P2"]
    p = SM.predict(fits[key], terms[key], SM.plan_terms(key, pl, hd, mem), kappa, rpc)
    if {k: v["speedup"] for k, v in p["arms"].items()} != {k: v["speedup"] for k, v in old["predictions"]["P2"]["arms"].items()} \
            or round(p["draft_ms"], 1) != old["predictions"]["P2"]["draft_ms"] or round(p["decode_ms"], 1) != old["predictions"]["P2"]["decode_ms"]:
        raise SystemExit("refused: P2's predictions no longer equal sp10's: %s" % {k: v["speedup"] for k, v in p["arms"].items()})
    reg = {k: old[k] for k in ("session", "scorer", "spec_args", "prompts", "prompt_order", "reqs", "gen", "scan", "scan_reps",
                               "model", "hypotheses")}
    reg.update({"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                "script": "testbed/scripts/register_sp10b_20261103.py", "out_prefix": "sp10b",
                "what": "sp10's plan P2 again, complete (sp10 lost P2's MTP 3 arm to a hung connection to the Pi 5)",
                "follows": {"registration": "testbed/dataset/sp10_predictions_20261103.json",
                            "session": "testbed/results/sp10_20261103_151444",
                            "why": "P2's MTP 3 serve: the server's connection to the Pi 5 hung at 15:41:40; it aborted at 16:07:34"},
                "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "order": ["P2"],
                "configs": {"P2": old["configs"]["P2"]}, "predictions": {"P2": old["predictions"]["P2"]},
                "request_timeout_s": 600,
                "inputs_sha256": {f: R.sha(f) for f in list(old["inputs_sha256"]) + ["testbed/scripts/register_sp10b_20261103.py",
                                                                                       "testbed/dataset/sp10_predictions_20261103.json"]}})
    print("P2 re-derived, equal to sp10's: decode %.1f ms, draft %.1f ms; speedups %s" % (
        p["decode_ms"], p["draft_ms"], {k: v["speedup"] for k, v in p["arms"].items()}))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), R.sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
