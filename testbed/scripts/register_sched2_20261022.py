#!/usr/bin/env python3
"""Register scheduler v2 -- precision as a plan variable -- against v1 on Qwen3-32B, four nodes, before anything is measured.

    python3 testbed/scripts/register_sched2_20261022.py --memfile testbed/dataset/memory_snapshot_<stamp>.json

The user (2026-10-21 ~23:20): "after techniques we have been developed so far, try your very best to develop further at huge
improvement of the scheduler"; then "Both files" (Qwen3-32B Q6_K and Q4_K_M) and "near-lossless only" (a plan may serve Q6_K or
better: predict_pool.MIN_BITS). Writes testbed/dataset/sched2_predictions_20261022.json, read-only; refuses to overwrite one.
  A  v1: the engine's pick for the Q8_0 file (plan_host_choice), the plan tonight's A1 ran (laptop 32 + Pi 4 13 + Pi 5 14 +
     Jetson 5); its 3,547.7 ms P50 there is context, not part of this study.
  V  v2: plan_precision over Qwen3-32B's official variants with the user's floor: every variant is planned by the same
     placement search and priced by the same predictor; the fastest allowed one is served as registered (PLACE).
Quality (sched2_quality_20261022.sh, the laptop alone, llama-perplexity against Q8_0's logits on GPL-3, 8 x 512 tokens) and
speed / energy (sched2_session_20261022.sh, A V A V, all four outlets) are scored by score_sched2_20261022.py.
"""
import argparse, datetime, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import qr3_tpot as Q                                        # noqa: E402
OUT = os.path.join(ROOT, "testbed", "dataset", "sched2_predictions_20261022.json")
CTX, NODES = 2048, ["pi5", "pi4", "jetson", "laptop"]
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}


def sha(p):
    h = hashlib.sha256()
    with open(os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm(r, power):
    m = P.memory(r["assign"], r["q"], r["mem"], r["devices"], r["model"], head_at=r["head_at"])
    e = EM.energy(r["assign"], r["q"], r["C"], r["links"], power, "tg", head_at=r["head_at"], rtt_trips=P.RTT_TRIPS_DEFAULT)
    idle = sum(power[n]["idle"] for n in NODES if n not in dict(r["assign"])) * e["T_ms"] / 1000
    head = "host" if r["head_at"] == 0 else "last"
    return {"assign": r["assign"], "head": head,
            "place_env": {"PLACE": ",".join("%s:%d" % tuple(x) for x in r["assign"]), "PLACE_HEAD": head},
            "predicted": {"decode_ms": round(r["decode_ms"], 1), "prefill_ms": round(r["prefill_ms"], 1),
                          "decode_gross_j_all": round(e["gross_j"] + idle, 4)},
            "memory": m, "considered": r["considered"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memfile", required=True); ap.add_argument("--reqs", type=int, default=5)
    ap.add_argument("--n", type=int, default=128); ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if os.path.exists(OUT) and not a.dry_run:
        raise SystemExit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    os.environ["HETEROSERVE_RESERVE_MB"] = ""                       # cluster.json's reserves (the laptop's 8 GiB)
    mem = json.load(open(a.memfile))
    power = EM.power_table()
    kw = dict(host="laptop", load_mode="none")
    v1 = P.plan_host_choice("qwen3_32b_q8", NODES, CTX, mem, **kw); v1["mem"] = mem
    best, considered = P.plan_precision("qwen3_32b", NODES, CTX, mem, **kw)
    v2 = best["plan"]; v2["mem"] = mem
    A, V = arm(v1, power), arm(v2, power)
    for x in (A, V):
        if min(m["margin_mib"] for m in x["memory"].values()) < 0:
            raise SystemExit("refused: a plan does not fit at rest: %s" % x["memory"])
    speed = A["predicted"]["decode_ms"] / V["predicted"]["decode_ms"]
    reg = {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "script": "testbed/scripts/register_sched2_20261022.py",
           "what": "scheduler v2: precision as a plan variable (plan_precision over a model's official quantisations, the user's "
                   "near-lossless floor) against v1, Qwen3-32B on laptop + Pi 5 + Pi 4 + Jetson, energy at the wall",
           "ctx": CTX, "nodes": NODES, "outlets": OUTLETS, "memfile": os.path.relpath(os.path.abspath(a.memfile), ROOT),
           "reqs": a.reqs, "n": a.n, "gen": a.gen, "warm": [Q.WARM_N, Q.WARM_GEN], "order": ["A", "V", "A", "V"],
           "cold_c": {"pi5": 50, "pi4": 55}, "min_bits": P.MIN_BITS,
           "arms": {"A": dict(A, model="qwen3_32b_q8", model_sha256=Q.MODEL_FILE["qwen3_32b_q8"][1],
                              what="v1: the engine's pick for the Q8_0 file"),
                    "V": dict(V, model=best["variant"], model_sha256=Q.MODEL_FILE[best["variant"]][1], bits=best["bits"],
                              what="v2: plan_precision's pick (the fastest variant with >= %.1f bits)" % P.MIN_BITS)},
           "variants_considered": [{"variant": e["variant"], "bits": e["bits"], "below_floor": e["below_floor"],
                                    **({"decode_ms": round(e["plan"]["decode_ms"], 1), "prefill_ms": round(e["plan"]["prefill_ms"], 1),
                                        "assign": e["plan"]["assign"]} if "plan" in e else {"infeasible": e.get("infeasible")})}
                                   for e in considered],
           "prior": {"20261021_A1": "the v1 plan, served 22:15-23:07 (prima4 session, arm-run A1): client TPOT P50 3,547.7 ms"},
           "quality": {"script": "testbed/scripts/sched2_quality_20261022.sh", "text": "/usr/share/common-licenses/GPL-3",
                       "protocol": "llama-perplexity at the pin, the laptop alone, -c 512 --chunks 8 -t 4, Q8_0 logits as the base"},
           "hypotheses": {
               "Q6-KLD": {"max": 0.01, "claim": "Q6_K's mean KL divergence from Q8_0 is at most 0.01"},
               "Q6-TOP": {"min_pct": 97.0, "claim": "Q6_K ranks the same token first as Q8_0 at >= 97 % of positions"},
               "Q6-PPL": {"tol": 0.01, "claim": "Q6_K's perplexity within 1 % of Q8_0's"},
               "Q-ORDER": {"claim": "Q4_K_M's mean KL divergence from Q8_0 exceeds Q6_K's (why the floor sits above it)"},
               "S2-TPOT": {"bar": 1.8, "predicted": round(speed, 3), "claim": "V's client TPOT P50 is at least 1.8x faster than A's "
                                                                                 "(predicted %.2fx)" % speed},
               "S2-ACC": {"tol": 0.10, "claim": "each arm's server mean TPOT within 10 % of its predicted decode"},
               "S2-TTFT": {"claim": "V's TTFT P50 below A's (predicted prefill %.1f against %.1f ms/token)" % (
                   V["predicted"]["prefill_ms"], A["predicted"]["prefill_ms"])},
               "S2-ENERGY": {"claim": "V's gross J per decoded token at all four outlets below A's (predicted %.2f against %.2f)" % (
                   V["predicted"]["decode_gross_j_all"], A["predicted"]["decode_gross_j_all"])}},
           "energy_definition": "per request, the gross energy at all four outlets over its decode window (first token to last), "
                                "with the session's idle-before window; summed over requests and divided by gen - 1 per request",
           "conditions": "the laptop quiet (no working apps, no battery charging, Wi-Fi off); aunsync.timer runs every 16 min "
                         "for ~20 s (a Google Drive sync), as in every laptop run since 2026-10-06",
           "inputs_sha256": {f: sha(f) for f in (
               "testbed/scripts/qr3_tpot.py", "testbed/scripts/serve_pool.sh", "testbed/scripts/predict_pool.py",
               "testbed/scripts/energy_model.py", "testbed/scripts/shelly_strip.py", "testbed/scripts/full_eval.py",
               "testbed/scripts/sched2_session_20261022.sh", "testbed/scripts/score_sched2_20261022.py",
               "testbed/scripts/sched2_quality_20261022.sh", "testbed/scripts/register_sched2_20261022.py",
               "testbed/scripts/sched2_chain_20261022.sh", "testbed/scripts/models_20261022.py",
               "testbed/dataset/model_shapes.json", "testbed/dataset/characterisation.csv", "ThinkPad/config/cluster.json",
               os.path.relpath(os.path.abspath(a.memfile), ROOT))}}
    if a.dry_run:
        print(json.dumps({k: reg[k] for k in ("arms", "variants_considered", "hypotheses")}, indent=1)[:5000]); return
    json.dump(reg, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s (sha256 %s)" % (os.path.relpath(OUT, ROOT), sha(os.path.relpath(OUT, ROOT))[:16]))
    print("A %s: %.1f ms, %.1f prefill, %.2f J | V %s %s: %.1f ms, %.1f prefill, %.2f J | %.2fx" % (
        A["place_env"]["PLACE"], A["predicted"]["decode_ms"], A["predicted"]["prefill_ms"], A["predicted"]["decode_gross_j_all"],
        best["variant"], V["place_env"]["PLACE"], V["predicted"]["decode_ms"], V["predicted"]["prefill_ms"],
        V["predicted"]["decode_gross_j_all"], speed))


if __name__ == "__main__":
    main()
