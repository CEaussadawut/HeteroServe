#!/usr/bin/env python3
"""Register scheduler upgrade 5 (2026-10-10: the Jetson's power mode as a plan variable) BEFORE the served run. Writes,
read-only, testbed/dataset/power_modes_predictions_20261010.json.

    python3 testbed/scripts/register_power_modes_20261010.py testbed/results/pm_char_<stamp> [--dry-run] [--out PATH]
    python3 testbed/scripts/register_power_modes_20261010.py --plan-only --out PATH
        --out writes the same registration elsewhere (a smoke run's plumbing; never the dataset's file);
        --plan-only writes only the plan the runner reads, with no predictions (a smoke before the characterisation)

The question (the survey's upgrade 5). In the served plan each board waits for its turn for most of every token, and
the Jetson waits in its locked state, at full clocks. Does letting it drop to a low-power idle, or running it in its
5 W mode, save energy per token? At what cost in time? And does a per-mode characterisation predict both within 5 %?

The served plan: Qwen3-14B with Pi 4 27 + Jetson 13 and the head on the host, ctx 1,024, no draft, greedy. The prompts
are the four of the shaping run (e01, c01, w01, th01), 64 tokens each.
- One server load. The Jetson's mode is switched live between blocks, in the order L U 5 5 U L, so each mode gets two
  blocks and 8 requests. Every switch is verified from sysfs.
- The runner owns the Jetson's clocks: serve_pool.sh runs with CLOCKS=asfound, and the runner's locked mode is
  serve_pool.sh's lock (jetson_clocks), verified the same way.

The model.
- The prior is the shaping run's base arm: the same plan, prompts and protocol, with the Jetson locked (T0 and E0 below,
  recomputed here from that run's raw data).
- The characterisation (power_modes_char_20261010.py, its analysis.json) ran the same 13-layer block on the Jetson,
  hosted by the laptop, one forward pass per request with a sleep G after each, in each mode: 4 gaps x 2 blocks per mode.
- The Jetson's power over a cycle of length T (its block, then waiting):
    P(T) = Pw + A / T          Pw = its waiting power, A = the energy its block costs above waiting
  fitted per mode by least squares over the mode's 8 block-cells, each at its own mean cycle (the power window's length
  over the passes in it). So the Jetson's energy per token is Ej = Pw x T + A, and Pw is the waiting power the survey
  meant to cut.
- The block's time in mode m against locked: dT(m) = the mean over the gaps of the difference of the cells' lower
  deciles of the forward pass. Contention on the laptop only ever lengthens a pass, so the lower decile is the
  undisturbed pass, and the laptop's part cancels in the difference.
- Then, per mode m:
    T(m)   = T0 + dT(m)
    Ej(m)  = Pw(m) x T(m) + A(m)
    E(m)   = E0 + [Ej(m) - Ej(locked)] + Pw_pi4 x dT(m)      the Pi 4 waits dT longer, at its waiting power
  Pw_pi4 is the Pi 4's idle power in the planner's power table (energy_model.power_table).
- Why not the interpolation first written here (16:50): the characterisation found the laptop's part varying by up to
  ~200 ms in median between cells (contention from this session's own app), so a cell's period no longer pins the
  Jetson's cycle. The fit uses each cell's own measured cycle instead, and the lower decile resists the contention.
  Changed before any served data, and before this registration was written.

Hypotheses, decided here before any data:
  PM-T     for each mode, the measured ms per token (the server's sum of predicted_ms / sum of predicted_n over the
           mode's 8 requests) within 5 % of T(m). Three tests: PM-T-locked, PM-T-unlocked, PM-T-5w.
  PM-E     for each mode, the measured gross J per decoded token within 5 % of E(m). Three tests. The measure: the
           strip's trapezoid at the Pi 4 and Jetson outlets over each request's decode window [send + prompt_ms, reply],
           summed over the mode's requests, divided by the sum of predicted_n.
  PM-PICK  the mode predicted to use the least energy per token measures the least. Judged only if its predicted margin
           over the next mode is at least 2 %; otherwise "not judged: too close to call".
Upgrade 5 holds when all six PM-T and PM-E tests hold.
Reported, not judged:
- dE(m) and dT(m) against the run's own locked arm (the day's drift cancels);
- the Jetson's outlet alone against Ej(m);
- the J per token saved per watt of waiting power cut (the survey estimated about 1.75).
A request missing from a mode makes that mode's tests INCOMPLETE. A guard trip, or a switch that fails its check, stops
the run.
"""
import datetime, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))

DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "power_modes_predictions_20261010.json")
PRIOR_RUN = os.path.join(ROOT, "testbed", "results", "spec_shape_20261010_125542")
PRIOR_REG = os.path.join(DS, "spec_shape_predictions_20261010.json")
STEP3_RUN = os.path.join(ROOT, "testbed", "results", "spec_serve_20261010_024045", "run.jsonl")
MODES = ("locked", "unlocked", "5w")
ORDER = ["locked", "unlocked", "5w", "5w", "unlocked", "locked"]
TOL, PICK_MARGIN = 0.05, 0.02
INPUTS = ["testbed/scripts/power_modes_20261010.py", "testbed/scripts/power_modes_char_20261010.py",
          "testbed/scripts/jetson_clock_poll.sh", "testbed/scripts/jetson_power_guard.sh", "testbed/scripts/serve_pool.sh",
          "testbed/scripts/serve_defaults.sh", "testbed/scripts/energy_model.py", "testbed/scripts/full_eval.py",
          "testbed/scripts/shelly_strip.py", "testbed/scripts/throttle_poll.sh"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def prior(run=PRIOR_RUN, reg_path=PRIOR_REG):
    """T0 (ms per token) and E0 (gross J per decoded token, Pi 4 + Jetson) of the shaping run's base arm, recomputed
    from its raw data with explore_spec_shape_20261010.py's method; per session too."""
    import full_eval as F
    reg = json.load(open(reg_path))
    lines = [json.loads(l) for l in open(os.path.join(run, "run.jsonl"))]
    idle = next(l["idle_before"] for l in lines if "idle_before" in l)
    rows = [l for l in lines if l.get("arm") == "base" and l.get("timings") and l["timings"].get("predicted_n")]
    pool = [reg["outlets"]["pi4"], reg["outlets"]["jetson"]]
    per = {}
    for r in rows:
        w = F.window_energy(run, r["t0"] + r["timings"]["prompt_ms"] / 1000.0, r["t1"], idle, pool)
        s = per.setdefault(r["session"], {"ms": 0.0, "n": 0, "j": 0.0, "j_jetson": 0.0})
        s["ms"] += r["timings"]["predicted_ms"]; s["n"] += r["timings"]["predicted_n"]
        s["j"] += sum(w[o][0] for o in pool); s["j_jetson"] += w[reg["outlets"]["jetson"]][0]
    n = sum(s["n"] for s in per.values())
    return {"run": os.path.relpath(run, ROOT), "requests": len(rows), "tokens": n,
            "T0_ms": sum(s["ms"] for s in per.values()) / n, "E0_j": sum(s["j"] for s in per.values()) / n,
            "E0_jetson_outlet_j": sum(s["j_jetson"] for s in per.values()) / n,
            "by_session": {str(k): {"T_ms": round(s["ms"] / s["n"], 1), "E_j": round(s["j"] / s["n"], 3)} for k, s in sorted(per.items())}}


def step3_same_prompts(ids):
    """Step 3's base arm on the same prompts (ms per token): the day's reproducibility, for the record."""
    rows = [json.loads(l) for l in open(STEP3_RUN)]
    rows = [r for r in rows if r.get("arm") == "base" and r.get("id") in ids and r.get("timings")]
    n = sum(r["timings"]["predicted_n"] for r in rows)
    return {"requests": len(rows), "T_ms": round(sum(r["timings"]["predicted_ms"] for r in rows) / n, 1) if n else None}


def fit_power(points):
    """Least squares of P on 1/T over [(T ms, P W)]: P = Pw + A / T. Returns (Pw W, A J, r2, residuals W)."""
    xs = [1000.0 / t for t, _ in points]
    ys = [p for _, p in points]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    a = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    pw = my - a * mx
    res = [y - (pw + a * x) for x, y in zip(xs, ys)]
    tot = sum((y - my) ** 2 for y in ys)
    return pw, a, (1 - sum(r * r for r in res) / tot) if tot else None, res


def predict(char, T0, E0, pw_pi4):
    """The registered model (module docstring), from the characterisation's analysis.json. Pure, for the tests."""
    cells = char["cells"]
    gaps = sorted({c["gap_s"] for c in cells.values()})
    cell = lambda m, g: cells["%s@%.1f" % (m, g)]
    out = {"modes": {}}
    for m in MODES:
        pts = [(b["period_power_ms"], b["strip_w"]) for g in gaps for b in cell(m, g)["blocks"]]
        if len(pts) < 3 or len({round(t) for t, _ in pts}) < 2:
            raise ValueError("%s: too few cells to fit its power (%d)" % (m, len(pts)))
        pw, a, r2, res = fit_power(pts)
        dT = statistics.fmean(cell(m, g)["forward_ms_p10"] - cell("locked", g)["forward_ms_p10"] for g in gaps)
        dT_med = statistics.fmean(cell(m, g)["forward_ms_median"] - cell("locked", g)["forward_ms_median"] for g in gaps)
        T = T0 + dT
        out["modes"][m] = {"Pw_w": pw, "A_j": a, "fit_r2": r2, "fit_rms_w": (sum(r * r for r in res) / len(res)) ** 0.5,
                           "dT_ms": dT, "dT_ms_by_median": dT_med, "T_ms": T, "Ej_j": pw * T / 1000.0 + a}
    ejl = out["modes"]["locked"]["Ej_j"]
    for m in MODES:
        x = out["modes"][m]
        x["E_j"] = E0 + (x["Ej_j"] - ejl) + pw_pi4 * x["dT_ms"] / 1000.0
        x["dE_j"] = x["E_j"] - E0
    ranked = sorted(MODES, key=lambda m: out["modes"][m]["E_j"])
    margin = out["modes"][ranked[1]]["E_j"] / out["modes"][ranked[0]]["E_j"] - 1
    out["pick"] = {"least_energy": ranked[0], "margin_over_next": margin, "judged": margin >= PICK_MARGIN}
    return out


def plan(preg):
    """The served setup the runner reads. Everything but the predictions."""
    return {
        "model": "qwen3_14b", "ctx": 1024, "place": "pi4:27,jetson:13", "place_head": "host", "nodes": "pi4 jetson",
        "serve_env": {"SERVE_DRAFT": "none", "CLOCKS": "asfound"},
        "host_url": preg["host_url"], "sampling": preg["sampling"], "chat_format": preg["chat_format"],
        "warmup_prompt": preg["warmup_prompt"], "prompts": preg["prompts"], "n_predict": preg["n_predict"],
        "order": ORDER, "settle_tokens": 8, "idle_before_s": 120, "idle_after_s": 60, "outlets": preg["outlets"],
        "models_on_host": {"~/models/qwen3-14b-q4_k_m.gguf": preg["models_on_host"]["~/models/qwen3-14b-q4_k_m.gguf"]},
        "modes": {
            "locked": "MAXN, jetson_clocks: CPU 1479 MHz x4, GPU 921.6 MHz, EMC 1600 MHz, cpuidle and railgating off (every run since 09-30)",
            "unlocked": "MAXN, governors free: nvhost_podgov, schedutil, EMC by its activity monitor, GPU railgated after 500 ms idle",
            "5w": "nvpmodel -m 1 (cores 2-3 offline), then jetson_clocks at its maxima: CPU 921.6 MHz x2, GPU 614.4 MHz"}}


def main(argv):
    out_path = argv[argv.index("--out") + 1] if "--out" in argv else OUT
    args = [a for a in argv if not a.startswith("--") and a != out_path]
    preg = json.load(open(PRIOR_REG))
    if "--plan-only" in argv:
        if "--out" not in argv:
            sys.exit("--plan-only needs --out: a plan without predictions is never the dataset's registration")
        reg = dict(plan(preg), plan_only=True, note="smoke plumbing: no predictions, never scored",
                   registered_at=datetime.datetime.now().astimezone().isoformat(timespec="seconds"))
        json.dump(reg, open(out_path, "w"), indent=1, ensure_ascii=False)
        print("wrote the plan only: %s" % out_path)
        return
    if not args:
        sys.exit(__doc__)
    char_dir = args[0]
    if os.path.exists(out_path) and "--dry-run" not in argv:
        sys.exit("%s exists: a registration is never rewritten" % out_path)
    import energy_model as EM
    char_p = os.path.join(char_dir, "analysis.json")
    char = json.load(open(char_p))
    if char.get("prompt_n_not_1", 0) > len(char["cells"]) * 2:     # more than the pass after each cell's burst: broken cache
        sys.exit("the characterisation has %d requests that were not one forward pass" % char["prompt_n_not_1"])
    pr = prior()
    pw = EM.power_table()["pi4"]
    pred = predict(char, pr["T0_ms"], pr["E0_j"], pw["idle"])
    used = {}
    for k, c in char["cells"].items():
        used[k] = {f: c[f] for f in ("forward_ms_median", "forward_ms_p10", "period_ms_median", "strip_w", "ina_w",
                                     "gpu_mhz_mean", "railgated_share", "block_spread_forward_pct", "block_spread_strip_pct")}
        used[k]["blocks"] = [{f: x[f] for f in ("block", "period_power_ms", "strip_w", "ina_w", "forward_ms_p10",
                                                "forward_ms_median", "gpu_mhz_mean")} for x in c["blocks"]]
    reg = dict(
        {"registered_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "script": "testbed/scripts/register_power_modes_20261010.py",
         "question": "scheduler upgrade 5: the Jetson's power mode while it waits (locked, low-power idle, 5 W) as a plan "
                     "variable; energy and time per token predicted per mode within 5 %"},
        **plan(preg))
    reg.update({
        "prior": dict(pr, step3_same_prompts=step3_same_prompts({p["id"] for p in preg["prompts"]})),
        "characterisation": {"analysis": os.path.relpath(char_p, ROOT), "sha256": sha(char_p), "cells": used,
                             "idle": {m: [{k: x.get(k) for k in ("strip_w", "ina_w", "railgated_share")} for x in v]
                                      for m, v in char["idle"].items()}},
        "pw_pi4_w": pw["idle"], "power_table_source": pw["source"], "power_table_sha256": pw["sha256"],
        "emulated_locked_jetson_j_at_T0": round(pred["modes"]["locked"]["Ej_j"], 4),
        "predictions": {m: {k: round(v, 4) for k, v in x.items()} for m, x in pred["modes"].items()},
        "pick": {"least_energy": pred["pick"]["least_energy"], "margin_over_next": round(pred["pick"]["margin_over_next"], 4),
                 "judged": pred["pick"]["judged"]},
        "hypotheses": {
            "PM-T": "for each mode, measured ms per token (sum predicted_ms / sum predicted_n over its 8 requests) within "
                    "%.0f %% of predictions[mode].T_ms" % (100 * TOL),
            "PM-E": "for each mode, measured gross J per decoded token at the pi4 + jetson outlets (strip trapezoid over each "
                    "request's [t0 + prompt_ms, t1], summed, over the sum of predicted_n) within %.0f %% of "
                    "predictions[mode].E_j" % (100 * TOL),
            "PM-PICK": "the mode predicted least in energy (pick.least_energy) measures least; judged only when "
                       "pick.margin_over_next >= %.2f" % PICK_MARGIN,
            "verdict": "upgrade 5 holds when all six PM-T and PM-E tests hold"},
        "tolerance": TOL, "pick_margin": PICK_MARGIN,
        "inputs_sha256": {p: sha(os.path.join(ROOT, p)) for p in INPUTS if os.path.exists(os.path.join(ROOT, p))},
    })
    print(json.dumps({k: reg[k] for k in ("prior", "emulated_locked_jetson_j_at_T0", "predictions", "pick")}, indent=1))
    if "--dry-run" in argv:
        print("(dry run: nothing written)")
        return
    json.dump(reg, open(out_path, "w"), indent=1, ensure_ascii=False)
    os.chmod(out_path, 0o444)
    print("registered %s (%s)" % (os.path.relpath(out_path, ROOT), sha(out_path)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
