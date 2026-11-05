#!/usr/bin/env python3
"""Register sp10d (2026-11-03 evening; the user: "... Per-node batch curves for the speculation cost model ... Then a re-test"):
the checking cost of the 9B's pooled plans, from each node's own batch curve.

    python3 testbed/scripts/register_sp10d_20261103.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

sp10 and sp10c carried the laptop-alone plan's R(m) (explspec B) onto the 9B's laptop + board plans and missed the batch of 4
by 12-18 % (P1, P4): the laptop's batch of 4 is cheaper than its batch of 3, a board's is not. The model here
(spec_model_20261103.py, per-node functions):
  - each node's own curve c_n(m) = T_n(m) / T_n(1), in its role: the laptop as host (B's scan), each Pi as an RPC worker
    (bench_worker_role.sh through a loopback server, testbed/results/worker_role_20261103_195154: a worker gets no repacked
    weights, so the Pi 5's batch of 4 costs 1.71x one there, 1.15x as a host);
  - a plan's curve: the planner's split of its decode step (energy_model.per_node), each node's share times its own curve,
    the links at 1;
  - R(m) = the pooled offset x the plan's curve. The pooled offset (a one-token batch against a decode step) is 1.0185, the
    mean of P1's 1.0201 (sp10) and P4's 1.0169 (sp10c); the laptop alone was 0.955.
  - a pass: dec x R(n + 1) + n x d + i. d is priced as before (KAPPA, RPC_DRAFT_MS from explspec A and B). i is P4's measured
    intercept (sp10c's pass fit; the MTP block on the Pi 4, as here). dec is the planner's, calibrated on B. E is B's.
Checked after the fact on P1 and P4 (their own offsets used, so not a test): R within 0.5-3 % at m 2 and 3, about 6 % low at 4.
Two plans none of it was fitted on, each with the Pi 4 last (serve_pool.sh refuses MTP with the Pi 5 last):
  P5  laptop 20 + Pi 5 7 + Pi 4 6
  P6  laptop 12 + Pi 5 11 + Pi 4 10
Each served with no speculation (then the scan, m 1-8), MTP 1, 2 and 3, two requests of 128 tokens on each of the
exploration's two prompts, order P5 then P6 (sp10_session_20261103.py, request_timeout_s 600). Scorer and hypotheses as
sp10's: SP-CHECK (R within 10 %), SP-SPEEDUP (within 7 %), SP-SIGN, SP-BEST (3 %), SP-DRAFT (20 %).
Writes testbed/dataset/sp10d_predictions_20261103.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import spec_model_20261103 as SM                            # noqa: E402
import explore_spec_20261102 as X                           # noqa: E402
import register_sp10_20261103 as R                          # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "sp10d_predictions_20261103.json")
KEY = "qwen35_9b"
WORKER = "testbed/results/worker_role_20261103_195154"
POOLED = {"P1": "testbed/results/sp10_20261103_151444/score.json", "P4": "testbed/results/sp10c_20261103_164359/score.json"}
PLANS = {"P5": [("laptop", 20), ("pi5", 7), ("pi4", 6)], "P6": [("laptop", 12), ("pi5", 11), ("pi4", 10)]}
SPECS = ["none", "mtp1", "mtp2", "mtp3"]
M = [1, 2, 3, 4, 6, 8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    fits = {k: SM.exploration_fit(os.path.join(ROOT, f), c) for k, (f, c, _, _) in R.EXP.items()}
    terms = {k: SM.plan_terms(k, pl, hd, mem) for k, (_, _, pl, hd) in R.EXP.items()}
    kappa, rpc = SM.calibrate(fits["qwen35_9b"], terms["qwen35_9b"], fits["qwen36moe_q6k"], terms["qwen36moe_q6k"])
    fB = fits[KEY]
    curves = {"laptop": SM.scan_curve(fB), "pi5": SM.worker_curve(os.path.join(ROOT, WORKER, "pi5.log")),
              "pi4": SM.worker_curve(os.path.join(ROOT, WORKER, "pi4.log"))}
    sc = {k: json.load(open(os.path.join(ROOT, f)))["plans"][k] for k, f in POOLED.items()}
    offset = statistics.fmean(sc[k]["R"]["1"] for k in sc)
    p4 = sc["P4"]["pass_residuals"]
    mn = statistics.fmean(x for x, _ in p4); mr = statistics.fmean(y for _, y in p4)
    slope = sum((x - mn) * (y - mr) for x, y in p4) / sum((x - mn) ** 2 for x, _ in p4)
    intercept = mr - slope * mn
    fit = dict(fB, intercept_ms=intercept)
    preds, configs = {}, {}
    for c, pl in PLANS.items():
        busy, links = SM.node_busy(KEY, pl, "host", mem)
        Rn = {m: round(offset * SM.plan_curve(busy, links, curves, m), 4) for m in M}
        tn = SM.plan_terms(KEY, pl, "host", mem)
        p = SM.predict_nodes(fit, terms[KEY], tn, kappa, rpc, Rn)
        p["R"] = {str(m): v for m, v in Rn.items()}
        p.update(terms=tn, busy={n: round(v, 1) for n, v in busy.items()}, links_ms=round(links, 2),
                 exploration={"folder": fB["folder"], "config": fB["config"]})
        preds[c] = p
        configs[c] = {"MODEL": KEY, "CTX": "4096", "PLACE": ",".join("%s:%d" % x for x in pl), "PLACE_HEAD": "host", "specs": SPECS}
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_sp10d_20261103.py", "session": "testbed/scripts/sp10_session_20261103.py",
           "scorer": "testbed/scripts/score_sp10_20261103.py", "out_prefix": "sp10d", "request_timeout_s": 600,
           "what": "the 9B's checking cost from each node's own batch curve (per-node), on two three-node plans it was not fitted on",
           "follows": {"registrations": ["testbed/dataset/sp10_predictions_20261103.json", "testbed/dataset/sp10c_predictions_20261103.json"],
                       "why": "carrying the laptop-alone R missed the 9B's laptop + board plans at m 4 (12-18 %)"},
           "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "order": list(PLANS), "configs": configs,
           "spec_args": {k: X.SPEC[k] for k in SPECS}, "prompts": X.PROMPTS, "prompt_order": ["free", "copy"], "reqs": 2,
           "gen": 128, "scan": M, "scan_reps": 3,
           "model": {"kappa": round(kappa, 4), "rpc_draft_ms": round(rpc, 2), "curves": {n: {str(m): round(v, 4) for m, v in c.items()}
                                                                                         for n, c in curves.items()},
                     "pooled_offset": round(offset, 4), "pooled_offset_from": {k: sc[k]["R"]["1"] for k in sc},
                     "intercept_ms": round(intercept, 2), "intercept_from": "P4's pass residuals %s (sp10c)" % p4,
                     "fit_B": {x: fB[x] for x in ("folder", "config", "decode_ms", "draft_ms", "intercept_ms", "E")}},
           "predictions": preds,
           "hypotheses": {
               "SP-CHECK": {"m": [2, 3, 4], "tol": 0.10, "claim": "each plan's R(m) = T(m) / its decode step, m 2-4, within 10 % of predicted"},
               "SP-SPEEDUP": {"tol": 0.07, "claim": "each MTP arm's speedup over the plan's none, per prompt, within 7 % of predicted"},
               "SP-SIGN": {"margin": 0.05, "claim": "every arm predicted at least 5 % faster or slower than none measures on that side of 1"},
               "SP-BEST": {"tol": 0.03, "claim": "per plan and prompt, the setting predicted fastest within 3 % of the fastest measured"},
               "SP-DRAFT": {"tol": 0.20, "claim": "each plan's measured draft cost within 20 % of predicted"}},
           "inputs_sha256": {f: R.sha(f) for f in (
               "testbed/scripts/sp10_session_20261103.py", "testbed/scripts/score_sp10_20261103.py",
               "testbed/scripts/register_sp10d_20261103.py", "testbed/scripts/register_sp10_20261103.py",
               "testbed/scripts/spec_model_20261103.py", "testbed/scripts/spec_cost_20261103.py", "testbed/scripts/explore_spec_20261102.py",
               "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh", "testbed/scripts/serve_plan_20261101.py",
               "testbed/scripts/predict_pool.py", "testbed/scripts/energy_model.py", "testbed/dataset/model_shapes.json",
               "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               "testbed/results/explspec_20261103_143210/results.json", "testbed/results/explspec_20261103_133920/results.json",
               WORKER + "/pi5.log", WORKER + "/pi4.log", POOLED["P1"], POOLED["P4"],
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    print("KAPPA %.4f, RPC_DRAFT_MS %.2f, pooled offset %.4f, intercept %.1f ms (P4's)" % (kappa, rpc, offset, intercept))
    for c, p in preds.items():
        print("%s %s: busy %s, decode %.1f ms, draft %.1f ms, R %s; speedups %s" % (
            c, configs[c]["PLACE"], p["busy"], p["decode_ms"], p["draft_ms"], {k: v for k, v in p["R"].items() if k in ("2", "3", "4")},
            {k: v["speedup"] for k, v in p["arms"].items()}))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), R.sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
