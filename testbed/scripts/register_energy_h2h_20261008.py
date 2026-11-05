#!/usr/bin/env python3
"""Pre-register the ENERGY head-to-head of the engine against Halda on today's boards (2026-10-08). Reads files only.

    python3 testbed/scripts/register_energy_h2h_20261008.py [--dry-run]

User: "benchmark energy compare to the prima and halda using what we have". On the boards that remain (Pi 4 + Jetson; the
Pi 5 was returned 2026-10-05, and prima.cpp cannot run on the Jetson), the one model for which Halda has a plan that fits
our validated memory accounting is TinyLlama (halda_offline_20261007.json). Halda's plan is the engine's split with the
output head on the host -- its rule -- realised at the pin with -ot "^output\\.weight=CPU"; on 2026-10-07 it measured 6.4 %
slower in decode (101.7 against 95.6 ms). Both are run on llama.cpp at the pin, so only the plans differ:
  eh.tiny.engine   the engine's plan (plan_optimal's default since 2026-10-08, head where cheaper): Pi 4 1 + Jetson 21,
                   the head on the Jetson
  eh.tiny.halda    Halda's plan: Pi 4 1 + Jetson 21, the head on the Pi 4
Run engine, halda, engine, halda, engine inside one strip log (energy_pool_run.py; outlets pi4=0, jetson=2), at
register_pool_energy.py's energy protocol (TinyLlama: -p 512 -n 128, -r sized for >= 120 s per phase). The Jetson's clocks
are locked for the WHOLE session, idle windows included (energy.md, 2026-10-06: a pooled registration should do this, so
that the idle floor is the locked one the cells wait at); run_pool.sh then finds them ALREADY_LOCKED and leaves them.

Predictions: throughput from predict_pool.predict (head_at as planned); energy from energy_model.energy, validated
2026-10-08 against all 16 measured pooled phase-runs (gross mean |error| 3.4 %, net 5.2 %), at the predicted time.
Hypotheses (scored by score_energy_h2h_20261008.py from score_energy_pool_run.py's score.json):
  H-DEC  the engine's mean gross decode J/token (trapezoid, and counter) is below Halda's: predicted -2.6 %
  H-PRE  prefill gross J/token within +-2 % of each other: predicted -0.5 %
  ACC    every run's gross J/token within 10 % of the model's at the run's measured time per token
Output: testbed/dataset/energy_h2h_predictions_20261008.json, read-only; never overwritten.
"""
import datetime, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "energy_h2h_predictions_20261008.json")
NODES, HOST, KEY, PHASE_S = ["pi4", "jetson"], "pi4", "tinyllama", 120.0
ORDER = ["eh.tiny.engine", "eh.tiny.halda", "eh.tiny.engine", "eh.tiny.halda", "eh.tiny.engine"]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json",
                                                "memory_at_rest.json", "halda_offline_20261007.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = P.rows(inputs["characterisation.csv"], "char")
    shape, quant, _ = P.MODELS[KEY]
    q = shapes[shape]
    C, src = P.costs(R, shapes, quant)
    L, missing = P.link_view(pairs, HOST, NODES)
    if missing:
        sys.exit("refused: no measured link %s" % missing)
    devs, mp = P.profiles(C, q, mem, L, NODES), P.model_profile(q, shape)
    eng = P.plan_optimal(devs, mp, "decode", host_candidates=[HOST])
    e_pipe = [(n, eng.layers[n]) for n in eng.order if eng.layers.get(n, 0) > 0]
    e_head = 0 if eng.head_node == e_pipe[0][0] else None
    hal = [c for c in json.load(open(inputs["halda_offline_20261007.json"]))["configs"]
           if c["model"] == KEY and sorted(c["boards"]) == sorted(NODES)][0]
    hs = [h for h in hal["halda"] if h.get("pipeline") and h.get("fits_engine_accounting")]
    h = min(hs, key=lambda x: x["decode_ms"])
    h_pipe = EM._assign(h["pipeline"])
    h_head = 0 if h["head_on"] == h_pipe[0][0] else None
    if (e_pipe, e_head) != ([("pi4", 1), ("jetson", 21)], None) or (h_pipe, h_head) != ([("pi4", 1), ("jetson", 21)], 0):
        sys.exit("refused: the plans are not the ones the docstring names: engine %s head_at %s, Halda %s head_at %s"
                 % (e_pipe, e_head, h_pipe, h_head))
    power = EM.power_table()
    res = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "energy head-to-head, engine against Halda's plan, on Pi 4 + Jetson (TinyLlama Q4_K_M), same runtime",
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()},
                             **{f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "energy_model.py", "run_pool.sh",
                                                                         "energy_pool_run.py", "score_energy_pool_run.py",
                                                                         "shelly_strip.py", os.path.basename(__file__))},
                             "strip.json": sha(os.path.join(ROOT, "testbed", "energy", "strip.json"))},
           "engine_sha256": {f: sha(os.path.join(ROOT, "ThinkPad", "control", f)) for f in ("placement.py", "gguf.py")},
           "power_table": power, "run_order": ORDER, "outlets": {"pi4": 0, "jetson": 2},
           "conditions": "the Jetson's clocks locked (jetson_clocks, stored first) before the session's idle window and "
                         "restored after it, so every window sees the same locked floor; the Jetson's power guard running "
                         "(run_pool.sh's gate requires it); the laptop runs the session and is not metered (outlet 1)",
           "configs": {}, "energy_hypotheses": [], "h2h_hypotheses": []}
    pp_n, tg_n = 512, 128
    for cid, pipe, head_at, why in (("eh.tiny.engine", e_pipe, e_head, "the engine's plan (default head_placement='best')"),
                                    ("eh.tiny.halda", h_pipe, h_head, "Halda's plan: its split, the head on the host")):
        dec, pre = P.predict(pipe, q, C, L, head_at=head_at)
        rep_pp, rep_tg = pp_n * pre / 1000.0, tg_n * dec / 1000.0
        reps = max(3, math.ceil(PHASE_S / rep_pp), math.ceil(PHASE_S / rep_tg))
        bench = P.bench_args(pipe, KEY, head_at=head_at)
        bench["protocol"] = "-t 4 -p %d -n %d -r %d --progress -oe jsonl" % (pp_n, tg_n, reps)
        e = {ph: EM.energy(pipe, q, C, L, power, ph, head_at=head_at) for ph in ("tg", "pp")}
        res["configs"][cid] = {
            "model": shape, "quant": quant, "why": why, "pipeline": [{"node": n, "blocks": k} for n, k in pipe],
            "head_on": pipe[0][0] if head_at == 0 else pipe[-1][0], "head_at": head_at, "bench": bench,
            "memory": P.memory(pipe, q, mem, devs, mp, head_at=head_at),
            "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                          "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2),
                          "timed_s": {"pp": round(reps * rep_pp, 1), "tg": round(reps * rep_tg, 1)},
                          "energy_model": {ph: {"gross_j_per_token": round(v["gross_j"], 4), "net_j_per_token": round(v["net_j"], 4),
                                                "board_w": {n: round(w, 2) for n, w in v["board_w"].items()}}
                                           for ph, v in e.items()}},
            "cost_sources": {n: src[n] for n, _ in pipe}}
    ce, ch = res["configs"]["eh.tiny.engine"]["predicted"]["energy_model"], res["configs"]["eh.tiny.halda"]["predicted"]["energy_model"]
    res["h2h_hypotheses"] = [
        {"id": "H-DEC", "claim": "engine mean gross decode J/token below Halda's, by trapezoid and by counter",
         "predicted_pct": round(100 * (ce["tg"]["gross_j_per_token"] / ch["tg"]["gross_j_per_token"] - 1), 2)},
        {"id": "H-PRE", "claim": "prefill mean gross J/token within +-2 % of each other",
         "predicted_pct": round(100 * (ce["pp"]["gross_j_per_token"] / ch["pp"]["gross_j_per_token"] - 1), 2)},
        {"id": "ACC", "claim": "every run's gross J/token (trapezoid) within 10 % of energy_model's at the run's measured time"}]
    for cid, c in res["configs"].items():
        p = c["predicted"]
        print("%-15s %-20s head %-6s decode %.1f ms  prefill %.1f ms  -r %s  energy decode %.4f / prefill %.4f J gross  margins %s"
              % (cid, " + ".join("%s %d" % (s["node"], s["blocks"]) for s in c["pipeline"]), c["head_on"],
                 p["decode_ms_per_token"], p["prefill_ms_per_token"], c["bench"]["protocol"].split("-r ")[1].split()[0],
                 p["energy_model"]["tg"]["gross_j_per_token"], p["energy_model"]["pp"]["gross_j_per_token"],
                 {n: int(v["margin_mib"]) for n, v in c["memory"].items()}))
    print("hypotheses:", [(h["id"], h.get("predicted_pct")) for h in res["h2h_hypotheses"]])
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return
    json.dump(res, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), res["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
