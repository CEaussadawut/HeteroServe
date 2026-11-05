#!/usr/bin/env python3
"""Register q9 (2026-11-02; the user: "... then i tem 9-10"): the scheduler review's item 9 -- a request priced as prefill
AND decode, not decode alone.

    python3 testbed/scripts/register_q9_20261102.py --memfile testbed/dataset/memory_snapshot_<stamp>.json [--dry-run]

plan_host_choice and best_by_predictor scored a request workload on its prefill alone; predict_pool.objective_ms now prices
it as P x prefill + G x decode (P prompt tokens, G generated), prefill predict()'s per-token figure as the planner prices it.
Over the planner's models, only Qwen3-30B-A3B Q6_K changes plan with it on today's memory (2026-11-02): for 512 + 64 the
request plan moves blocks from the Pi 5 to the Jetson's GPU, which prefills faster per block and decodes slower.
Qwen3-30B-A3B Q6_K at ctx 4,096, all four nodes (PLANNER_V4), two arms planned on the same snapshot:
  D  the plan by decode time (phase "decode", what serving plans by);
  R  the plan by the whole request, {"prefill": 512, "decode": 64}.
Each arm-run serves three requests of 512 prompt tokens and 64 generated (qr3_tpot.py: a different text offset per request,
so no prompt is cached), order D R R D, spec4_session_20261023.sh (tag_prefix "q9", scorer score_q9_20261102.py).
Predicted per arm: the time to the first token, P x prefill; the request, P x prefill + (G - 1) x decode; decode per token.
Hypotheses (registered):
  Q9-REQ     R's mean request time (sent to last token) at least 2.5 % below D's (half the predicted gain)
  Q9-TRADE   R's time to the first token below D's, and its time per output token above D's (the trade the objective makes)
  Q9-ACC     each arm's mean time to the first token within 15 % of its prediction and its server mean TPOT within 10 %
  Q9-REPEAT  each arm's two runs within 2 % of each other (mean request time)
Writes testbed/dataset/q9_predictions_20261102.json, read-only; refuses to overwrite it.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "q9_predictions_20261102.json")
KEY, CTX, NODES, PN, GN = "qwen3moe_q6k", 4096, ["laptop", "pi5", "pi4", "jetson"], 512, 64
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm(what, plan, mem):
    head = "host" if plan["head_at"] == 0 else "last"
    assign = [tuple(x) for x in plan["assign"]]
    r = P.plan_fixed(KEY, assign, head, CTX, mem, load_mode="none", **P.PLANNER_V4)
    m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
    pre, dec = r["prefill_ms"], r["decode_ms"]
    return {"what": what, "assign": [list(x) for x in assign], "head": head,
            "place_env": {"PLACE": ",".join("%s:%d" % x for x in assign), "PLACE_HEAD": head},
            "serve_env": {"SERVE_DRAFT": "none", "HETEROSERVE_RESERVE_MB": "", "SERVE_PLANNER": "v4"},
            "predicted": {"decode_ms": round(dec, 1), "prefill_ms_per_token": round(pre, 2), "ttft_s": round(PN * pre / 1000, 2),
                          "request_s": round((PN * pre + (GN - 1) * dec) / 1000, 2),
                          "objective_s": round(P.objective_ms({"prefill": PN, "decode": GN}, dec, pre) / 1000, 2)},
            "memory": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=3)
    ap.add_argument("--order", default="D R R D"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    mem = json.load(open(a.memfile))
    pd = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", **P.PLANNER_V4)
    pr = P.plan_host_choice(KEY, NODES, CTX, mem, host="laptop", load_mode="none", phase={"prefill": PN, "decode": GN},
                            **P.PLANNER_V4)
    if (pd["assign"], pd["head_at"]) == (pr["assign"], pr["head_at"]):
        raise SystemExit("refused: the request objective picks the decode plan on this snapshot: %s" % (pd["assign"],))
    arms = {"D": arm("the plan by decode time", pd, mem), "R": arm("the plan by the whole request (512 + 64)", pr, mem)}
    for k, x in arms.items():
        if min(v["margin_mib"] for v in x["memory"].values()) < 0:
            raise SystemExit("refused: %s does not fit at rest: %s" % (k, x["memory"]))
    gain = 1 - arms["R"]["predicted"]["request_s"] / arms["D"]["predicted"]["request_s"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_q9_20261102.py", "session": "testbed/scripts/spec4_session_20261023.sh",
           "scorer": "testbed/scripts/score_q9_20261102.py", "tag_prefix": "q9", "out_prefix": "q9",
           "what": "the scheduler review's item 9: a request priced as prefill and decode (predict_pool.objective_ms), on Qwen3-30B-A3B "
                   "Q6_K, the plan by decode time against the plan by the whole 512 + 64 request",
           "model": KEY, "model_file": Q.MODEL_FILE[KEY][0], "model_sha256": Q.MODEL_FILE[KEY][1], "ctx": CTX, "nodes": NODES,
           "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT), "reserve_mb": "",
           "reqs": a.reqs, "n": PN, "gen": GN, "warm": [Q.WARM_N, Q.WARM_GEN], "order": a.order.split(),
           "cold_c": {"pi5": 50, "pi4": 55}, "arms": arms, "predicted_request_gain": round(gain, 4),
           "hypotheses": {
               "Q9-REQ": {"min_gain": 0.025, "claim": "R's mean request time at least 2.5 %% below D's (predicted %.1f %%)" % (100 * gain)},
               "Q9-TRADE": {"claim": "R's time to the first token below D's, and its time per output token above D's"},
               "Q9-ACC": {"tol_ttft": 0.15, "tol_tpot": 0.10,
                          "claim": "each arm's mean time to the first token within 15 % of its prediction, its server mean TPOT within 10 %"},
               "Q9-REPEAT": {"tol": 0.02, "claim": "each arm's two runs within 2 % of each other (mean request time)"}},
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/serve_defaults.sh",
               "testbed/scripts/serve_plan_20261101.py", "testbed/scripts/predict_pool.py", "testbed/scripts/shelly_strip.py",
               "testbed/scripts/full_eval.py", "testbed/scripts/spec4_session_20261023.sh", "testbed/scripts/score_q9_20261102.py",
               "testbed/scripts/register_q9_20261102.py", "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv",
               "ThinkPad/config/cluster.json", os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    for k, x in arms.items():
        p = x["predicted"]
        print("%s %s head %s: TTFT %.1f s, request %.1f s, decode %.1f ms; margins %s" % (
            k, x["place_env"]["PLACE"], x["head"], p["ttft_s"], p["request_s"], p["decode_ms"],
            {n: v["margin_mib"] for n, v in x["memory"].items()}))
    print("predicted request gain R against D: %.1f %%" % (100 * gain))
    if a.dry_run:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))


if __name__ == "__main__":
    main()
