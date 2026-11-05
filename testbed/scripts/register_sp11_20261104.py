#!/usr/bin/env python3
"""Register sp11 (2026-11-04; the user: "do all four follow-up studies"): the MTP 1 default across many prompts, and why a
plan's acceptance on free text moves.

    python3 testbed/scripts/register_sp11_20261104.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

Why. MTP 1 is the default when a person serves a pooled plan of qwen35_9b or qwen36moe_q6k (4 Nov). Its registered gains rest
on two prompts (sp10c, sp10e); a live check on a third prompt (mtp_default_check_20261104_031253) was 4-6 % slower, at 59 %
of drafts accepted. And sp10e's P5 accepted 2.54 tokens a pass on free text under MTP 2 where P6 accepted 2.19: read after
the data (benchmarks.md, 4 Nov), every MTP run left plain decoding's text at character 93 ("If you gather" became "have" on
P5 and "split" on P6), and the two plans' MTP texts then differed; the copy prompt's texts never did, and its acceptance agreed
within 2 %. The reading: on free text, a near-tie between the top two tokens is resolved by numbers that differ slightly
between a plan's one-token decode and its batched check, and between plans; once a token differs, the text and so the
acceptance differ.
Three plans, each served with no speculation and with MTP 1 (the default's arguments), 12 prompts of 128 greedy tokens, one
request each (sp11_session_20261104.py, request_timeout_s 600):
  Q5   Qwen3.5-9B Q4_K_M on laptop 20 + Pi 5 7 + Pi 4 6 (sp10e's P5), head on the laptop; then, NOT timed, each prompt once more
       with the top two probabilities of every plain token (n_probs 2, before sampling)
  Q6   the same on laptop 12 + Pi 5 11 + Pi 4 10 (sp10e's P6)
  Q35  Qwen3.6-35B-A3B Q6_K on the plan a person serving it from a terminal gets now (serve_plan_20261101.choose, PLANNER_V4,
       the four nodes, ctx 4,096, depth 0, no remembered plan), on this registration's memory snapshot; when no plan holds the
       35B there (the laptop's 8 GiB reserve on its live memory), Q35 is dropped before registering and the reason recorded
The checking-cost model, unchanged: Q5 and Q6 carry sp10e's registered decode step and MTP 1 pass (per-node curves); Q35 is
priced as sp10c priced P3 (spec_model_20261103.predict from the exploration's A). MTP 1's speedup at tokens per pass E is
decode_ms x E / pass_ms; it breaks even at E* = pass_ms / decode_ms.
Hypotheses:
  MT-COST   on every plan, each prompt's speedup within 5 % of the model's at that prompt's measured E (the cost side, across
            prompts)
  MT-SIGN   on every plan, MTP 1 is faster than none exactly on the prompts whose measured E exceeds E* (prompts within 2 %
            of even left out)
  MT-MEAN   on every plan, the geometric mean of the 12 speedups is at least 1.00 (the default does not slow serving on
            average)
  MT-WORST  on every plan, no prompt is more than 10 % slower under MTP 1
  TX-TIE    every first point where another 9B text of a prompt (Q6's plain, Q5's or Q6's MTP 1) leaves Q5's plain text is a
            near-tie there: Q5's plain top two probabilities within 0.10 of each other
  TX-E      for each prompt, Q5's and Q6's MTP 1 tokens per pass differ by more than 3 % only where their MTP 1 texts differ
Writes testbed/dataset/sp11_predictions_20261104.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import spec_model_20261103 as SM                            # noqa: E402
import explore_spec_20261102 as X                           # noqa: E402
import serve_plan_20261101 as SP                            # noqa: E402
import predict_pool as P                                    # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "sp11_predictions_20261104.json")
SP10E = "testbed/dataset/sp10e_predictions_20261103.json"
EXP = {"qwen36moe_q6k": ("testbed/results/explspec_20261103_143210", "A", [("laptop", 24), ("jetson", 2), ("pi5", 9), ("pi4", 6)], "host"),
       "qwen35_9b": ("testbed/results/explspec_20261103_133920", "B", [("laptop", 33)], "last")}
PROMPTS = {
    "free": X.PROMPTS["free"],
    "copy": X.PROMPTS["copy"],
    "story": "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "email": "Write a polite email to a landlord asking for a leaking kitchen tap to be fixed this week.",
    "code": "Write a Python function that returns the n-th Fibonacci number by iteration, with a docstring and two example "
            "calls.",
    "tips": "List eight practical ways to cut the electricity bill of a small home office, one line each.",
    "hash": "Explain how a hash table handles two keys that land in the same bucket, and what that costs.",
    "summary": "Summarise the following paragraph in two sentences. Solar panels on a school roof produced 41 megawatt-hours "
               "last year, about a third of the school's use. The rest came from the grid, mostly in the evenings and on "
               "cloudy days, when the panels produced little. The school plans to add a battery so that the midday surplus "
               "can cover the evening classes, and expects the battery to pay for itself in nine years.",
    "thai": "Translate into Thai: The meeting has moved to Friday at ten in the morning. Please bring the printed report and "
            "your laptop.",
    "math": "A train leaves at 08:40 and arrives at 11:15. How long is the trip in minutes? Show each step.",
    "compare": "What are the main differences between a Raspberry Pi 5 and a Raspberry Pi 4 for running small language models?",
    "chat": "User: I can't sleep the night before an exam. What should I do?\nAssistant:",
}
SPECS = ["none", "mtp1"]


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
    # Q35: the plan a person at a terminal gets now (serve_pool.sh's interactive planner, no remembered plan). When no plan
    # holds the 35B (the laptop's 8 GiB reserve on its live memory), Q35 is dropped here, before registering, with the reason.
    fits = {k: SM.exploration_fit(os.path.join(ROOT, f), c) for k, (f, c, _, _) in EXP.items()}
    terms = {k: SM.plan_terms(k, pl, hd, mem) for k, (_, _, pl, hd) in EXP.items()}
    kappa, rpc = SM.calibrate(fits["qwen35_9b"], terms["qwen35_9b"], fits["qwen36moe_q6k"], terms["qwen36moe_q6k"])
    e = json.load(open(os.path.join(ROOT, SP10E)))
    sources, dropped = [("Q5", e["predictions"]["P5"]), ("Q6", e["predictions"]["P6"])], None
    try:
        r, why = SP.choose("qwen36moe_q6k", ["laptop", "pi5", "pi4", "jetson"], 4096, mem, 0, last=None, **P.PLANNER_V4)
        q35 = [(n, int(k)) for n, k in r["assign"]]
        head35 = "host" if r["head_at"] == 0 else "last"
        t35 = SM.plan_terms("qwen36moe_q6k", q35, head35, mem)
        sources.append(("Q35", SM.predict(fits["qwen36moe_q6k"], terms["qwen36moe_q6k"], t35, kappa, rpc)))
    except ValueError as ex:
        r = None
        p3 = P.plan_fixed("qwen36moe_q6k", [("laptop", 24), ("pi5", 9), ("pi4", 6), ("jetson", 2)], "host", 4096, mem,
                          load_mode="none", **P.PLANNER_V4)
        dropped = {"config": "Q35", "why": "no subset of the cluster holds Qwen3.6-35B-A3B Q6_K with the laptop's 8 GiB reserve "
                   "on this snapshot (the laptop's MemAvailable %d MiB); sp10c's P3 placement leaves the laptop %d MiB short. The "
                   "reserve is the owner's (the review's item 5, not approved), so the 35B case waits for a run with the memory"
                   % (mem["laptop"]["mem_available_mib"], -SP.fits(p3, mem)[1]["laptop"]["margin_mib"]),
                   "planner": str(ex)[:300]}
    preds = {}
    for c, src in sources:
        arm = src["arms"]["mtp1/free"]
        preds[c] = {"decode_ms": src["decode_ms"], "pass_ms": arm["pass_ms"], "E_breakeven": round(arm["pass_ms"] / src["decode_ms"], 4),
                    "speedup_at_registered_E": {pk: src["arms"]["mtp1/" + pk]["speedup"] for pk in ("free", "copy")},
                    "registered_E": {pk: src["arms"]["mtp1/" + pk]["E"] for pk in ("free", "copy")},
                    "from": SP10E + " predictions." + {"Q5": "P5", "Q6": "P6"}[c] if c != "Q35" else
                            "spec_model_20261103.predict on the exploration's A, as sp10c's P3"}
    configs = {"Q5": {"MODEL": "qwen35_9b", "CTX": "4096", "PLACE": "laptop:20,pi5:7,pi4:6", "PLACE_HEAD": "host", "specs": SPECS,
                      "nprobs": True},
               "Q6": {"MODEL": "qwen35_9b", "CTX": "4096", "PLACE": "laptop:12,pi5:11,pi4:10", "PLACE_HEAD": "host", "specs": SPECS}}
    if r is not None:
        preds["Q35"]["terms"] = t35
        configs["Q35"] = {"MODEL": "qwen36moe_q6k", "CTX": "4096", "PLACE": ",".join("%s:%d" % x for x in q35), "PLACE_HEAD": head35,
                          "specs": SPECS, "planned": why, "predicted_decode_ms": round(r["decode_ms"], 1)}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_sp11_20261104.py", "session": "testbed/scripts/sp11_session_20261104.py",
           "scorer": "testbed/scripts/score_sp11_20261104.py", "out_prefix": "sp11", "request_timeout_s": 600,
           "what": "the MTP 1 default across 12 prompts on three pooled plans, and the texts behind a plan's acceptance",
           "follows": {"registrations": [SP10E, "testbed/dataset/sp10c_predictions_20261103.json"],
                       "live_check": "testbed/results/mtp_default_check_20261104_031253",
                       "why": "MTP 1's registered gains rest on two prompts; sp10e's P5 acceptance moved on free text"},
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "order": list(configs), "configs": configs,
           "dropped_before_registering": dropped,
           "spec_args": {k: X.SPEC[k] for k in SPECS}, "prompts": PROMPTS, "prompt_order": list(PROMPTS), "gen": 128,
           "model": {"kappa": round(kappa, 4), "rpc_draft_ms": round(rpc, 2)},
           "conditions": "Zen may stay open (QUIET_ALLOW_APPS=zen, the owner, 4 Nov); aunsync.timer (a Google Drive sync, every "
                         "16 min, ~20 s) runs; the laptop's quietness is not gated during the session (speed ratios within one "
                         "plan's serves, not energy)",
           "predictions": preds,
           "hypotheses": {
               "MT-COST": {"tol": 0.05, "claim": "each prompt's MTP 1 speedup within 5 % of the model's at its measured E, every plan"},
               "MT-SIGN": {"dead_band": 0.02, "claim": "MTP 1 faster than none exactly where the measured E exceeds E*, every plan"},
               "MT-MEAN": {"min": 1.00, "claim": "the geometric mean of the 12 speedups at least 1.00, every plan"},
               "MT-WORST": {"min": 0.90, "claim": "no prompt more than 10 % slower under MTP 1, every plan"},
               "TX-TIE": {"gap": 0.10, "claim": "every first divergence of another 9B text from Q5's plain text is a near-tie there "
                                                "(Q5's plain top-two probabilities within 0.10)"},
               "TX-E": {"plans": ["Q5", "Q6"], "tol_pct": 3.0,
                        "claim": "Q5's and Q6's MTP 1 tokens per pass differ by more than 3 % only where their MTP 1 texts differ"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/sp11_session_20261104.py", "testbed/scripts/score_sp11_20261104.py",
               "testbed/scripts/register_sp11_20261104.py", "testbed/scripts/spec_model_20261103.py",
               "testbed/scripts/explore_spec_20261102.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/serve_plan_20261101.py", "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py",
               "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               "testbed/results/explspec_20261103_143210/results.json", "testbed/results/explspec_20261103_133920/results.json",
               SP10E, os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    print("Q35: %s" % (("%s (head %s), %s; predicted decode %.1f ms" % (configs["Q35"]["PLACE"], head35, why, r["decode_ms"]))
                         if r is not None else "DROPPED: %s" % dropped["why"]))
    for c, p in preds.items():
        print("%s: decode %.1f ms, MTP 1 pass %.1f ms, E* %.3f, speedup at the registered E %s" % (
            c, p["decode_ms"], p["pass_ms"], p["E_breakeven"], p["speedup_at_registered_E"]))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
