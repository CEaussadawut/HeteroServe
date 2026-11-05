#!/usr/bin/env python3
"""Pre-register the host-choice cell of 2026-10-08 (the energy follow-up to the engine-vs-Halda head-to-head). Reads
files only.

    python3 testbed/scripts/register_energy_host_20261008.py [--gpu-mhz MHZ] [--dry-run]

--gpu-mhz (the DVFS follow-up): the same cell with the Jetson's GPU held at MHZ for the whole session (min = max, after
jetson_clocks; the EMC stays at 1,600 MHz), registered as EXPLORATORY: its throughput is not predicted at that clock (the
cost rows are at 921.6 MHz), so the predicted figures below stay the full-clock ones and its throughput check is expected to
miss. Output then: energy_host_predictions_20261008_gpu<MHZ>.json. Hypothesis H-DVFS: decode gross J/token below the
full-clock cell's (same night), at whatever speed; reported as found.

The head-to-head (energy_h2h_predictions_20261008.json) runs TinyLlama hosted on the Pi 4, the host every TinyLlama cell has
used. The engine's planner, given the strip's measured power and allowed to choose the host among the boards holding the
file (the Jetson holds it: its single-board energy runs of 2026-10-05), picks the Jetson ALONE: predicted 87.0 ms/token
against 95.4 pooled, at 0.97 J/token with the Pi 4 switched off and 1.18 with it idling, against 1.24 pooled. Neither
Halda nor prima.cpp chooses the host (prima.cpp's head is the device the user starts as rank 0). This cell measures it:
  eh.tiny.jetson   TinyLlama Q4_K_M, all 22 blocks and the head on the Jetson's GPU (llama-bench on the Jetson, -ngl 99)
run twice inside one strip log, outlets pi4=0 and jetson=2, with the Jetson's clocks locked for the whole session as in the
head-to-head, and the Pi 4 left powered and idle (so its idle draw is measured, not assumed).
Predictions: throughput from predict_pool.predict with the Jetson as host (its Vulkan row); energy from energy_model.energy,
plus the Pi 4's idle power for the time.
Hypothesis H-HOST: the Jetson-alone cell's gross decode J/token, the Pi 4's idle outlet included, is below the head-to-head's
engine cells' mean (same night, same locked clocks), and its decode is faster. Scored by hand from the two runs' score.json
files (score_energy_pool_run.py) plus the Pi 4's outlet over the cell's windows, and reported as found.
Output: testbed/dataset/energy_host_predictions_20261008.json, read-only; never overwritten.
"""
import datetime, hashlib, json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402
import energy_model as EM                                   # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "energy_host_predictions_20261008.json")
GPU_MHZ = (12 * 76.8, 11 * 76.8, 10 * 76.8, 9 * 76.8, 8 * 76.8, 7 * 76.8, 6 * 76.8, 5 * 76.8, 4 * 76.8)   # the devfreq table
KEY, PHASE_S = "tinyllama", 120.0


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main(argv):
    global OUT
    gpu = float(argv[argv.index("--gpu-mhz") + 1]) if "--gpu-mhz" in argv else None
    if gpu is not None:
        if not any(abs(gpu - g) < 0.05 for g in GPU_MHZ):
            sys.exit("refused: %s MHz is not one of the Jetson GPU's frequencies %s" % (gpu, GPU_MHZ))
        OUT = OUT.replace(".json", "_gpu%d.json" % round(gpu))
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    ds = os.path.join(ROOT, "testbed", "dataset")
    inputs = {f: os.path.join(ds, f) for f in ("characterisation.csv", "model_shapes.json", "link.json", "memory_at_rest.json")}
    shapes = json.load(open(inputs["model_shapes.json"]))
    pairs = json.load(open(inputs["link.json"]))["pairs"]
    mem = json.load(open(inputs["memory_at_rest.json"]))
    R = P.rows(inputs["characterisation.csv"], "char")
    shape, quant, mfile = P.MODELS[KEY]
    q = shapes[shape]
    C, src = P.costs(R, shapes, quant)
    power = EM.power_table()
    L, _ = P.link_view(pairs, "jetson", ["jetson"])
    pipe = [("jetson", q["n_layers"])]
    dec, pre = P.predict(pipe, q, C, L)
    devs, mp = P.profiles(C, q, mem, L, ["jetson"]), P.model_profile(q, shape)
    pp_n, tg_n = 512, 128
    rep_pp, rep_tg = pp_n * pre / 1000.0, tg_n * dec / 1000.0
    reps = max(3, math.ceil(PHASE_S / rep_pp), math.ceil(PHASE_S / rep_tg))
    e = {ph: EM.energy(pipe, q, C, L, power, ph) for ph in ("tg", "pp")}
    pi4_idle = power["pi4"]["idle"]
    res = {"registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "campaign": "host choice: TinyLlama on the Jetson alone, the Pi 4 idle and metered (follow-up to the energy head-to-head)",
           "inputs_sha256": {**{f: sha(p) for f, p in inputs.items()},
                             **{f: sha(os.path.join(HERE, f)) for f in ("predict_pool.py", "energy_model.py", "run_pool.sh",
                                                                         "energy_pool_run.py", "score_energy_pool_run.py",
                                                                         "shelly_strip.py", os.path.basename(__file__))},
                             "strip.json": sha(os.path.join(ROOT, "testbed", "energy", "strip.json"))},
           "power_table": power, "run_order": ["eh.tiny.jetson", "eh.tiny.jetson"], "outlets": {"pi4": 0, "jetson": 2},
           "conditions": "as energy_h2h_predictions_20261008.json: the Jetson's clocks locked for the whole session; the guard "
                         "running; the Pi 4 powered, idle, on outlet 0" + (
                             "; the Jetson's GPU held at %.1f MHz (devfreq min = max) for the whole session, the EMC at "
                             "1,600 MHz: EXPLORATORY, the predicted throughput below is the full-clock one" % gpu if gpu else ""),
           "gpu_mhz": gpu or 921.6,
           "configs": {"eh.tiny.jetson": {
               "model": shape, "quant": quant, "why": "the planner's pick with the host chosen among the boards holding the file",
               "pipeline": [{"node": "jetson", "blocks": q["n_layers"]}], "head_on": "jetson", "head_at": None,
               "bench": {"host": "jetson", "args": "-m %s -ngl 99" % mfile,
                         "protocol": "-t 4 -p %d -n %d -r %d --progress -oe jsonl" % (pp_n, tg_n, reps)},
               "memory": P.memory(pipe, q, mem, devs, mp),
               "predicted": {"decode_ms_per_token": round(dec, 1), "decode_tok_s": round(1000 / dec, 3),
                             "prefill_ms_per_token": round(pre, 1), "prefill_tok_s": round(1000 / pre, 2),
                             "timed_s": {"pp": round(reps * rep_pp, 1), "tg": round(reps * rep_tg, 1)},
                             "energy_model": {ph: {"gross_j_per_token": round(v["gross_j"], 4),
                                                   "gross_j_per_token_with_pi4_idle": round(v["gross_j"] + pi4_idle * v["T_ms"] / 1000, 4)}
                                              for ph, v in e.items()}},
               "cost_sources": {"jetson": src["jetson"]}}},
           "energy_hypotheses": [],
           "h2h_hypotheses": [{"id": "H-DVFS", "claim": "decode gross J/token below the full-clock eh.tiny.jetson cell's "
                                                        "(same night), at whatever speed"}] if gpu else
                             [{"id": "H-HOST", "claim": "gross decode J/token with the Pi 4's idle outlet included below the "
                                                        "head-to-head engine cells' mean, and decode faster than theirs"}]}
    c = res["configs"]["eh.tiny.jetson"]
    print("eh.tiny.jetson  jetson 22  decode %.1f ms  prefill %.1f ms  %s  energy decode %s  margins %s" % (
        dec, pre, c["bench"]["protocol"], c["predicted"]["energy_model"]["tg"], {n: int(v["margin_mib"]) for n, v in c["memory"].items()}))
    if "--dry-run" in argv:
        print("dry run: nothing written")
        return
    json.dump(res, open(OUT, "w"), indent=1)
    os.chmod(OUT, 0o444)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), res["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
