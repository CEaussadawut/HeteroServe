#!/usr/bin/env python3
"""Power rows per model and phase for the phase scheduler, with the laptop's turbo burst (2026-10-28, after ph1).

    python3 testbed/scripts/phase_power_20261028.py fit      # ph1's strip data -> testbed/dataset/phase_power_20261028.json

ph1's energies ran 3-14 % above the scheduler's predictions, which used energy_model.power_table() (single-board TinyLlama
runs of 5-20 October). Its strip data (testbed/results/ph1_20261028_090849: all four outlets at 5 Hz, every request's
phase times) says why: the laptop's prefill of Qwen3-0.6B draws ~41 W for its first ~20 s of load and ~30 W after (the CPU's
short-term turbo budget, the same that slowed its prefill after a busy request), against the table's 29.3 W; the laptop
idles at ~17 W while serving, against 14.1; the Jetson decodes at ~9.4 W, against 11.2; during a hand-off the laptop works
(~24 W) rather than idles. fit reads, per request of ph1, the mean power of each outlet over its phases (shelly_strip.py
summarise) and fits:
  laptop prefill: P(t) = burst W for the first tau s of load, then sustained W (tau by grid, the two powers by least
                  squares over the L requests' prefill windows);
  laptop decode:  burst W while the load is still inside tau (decode follows the prefill), sustained W after;
  Jetson prefill and decode: one mean each (the J requests); the hand-off: each node's mean over the H requests' hand-offs;
  idle: each outlet over ph1's idle window (both servers loaded, nothing running).
energy(costs, plan, n, gen, rows) prices a plan with phase_plan_20261028's times and these rows (every other outlet at its
idle while one node works). Registered test of these rows: ph2 (register_ph2_20261028.py), on held-out prompt lengths.
"""
import json, os, statistics, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_plan_20261028 as PP                            # noqa: E402

PH1 = os.path.join(ROOT, "testbed", "results", "ph1_20261028_090849")
OUT = os.path.join(ROOT, "testbed", "dataset", "phase_power_20261028.json")
OUTLETS = {"pi4": 0, "jetson": 1, "pi5": 2, "laptop": 3}


def load(path=OUT):
    return json.load(open(path))


def _power(d, idle, t0, t1):
    p = subprocess.run([sys.executable, os.path.join(HERE, "shelly_strip.py"), "summarise", os.path.join(d, "strip.csv"),
                        "--window", "%.3f" % t0, "%.3f" % t1, "--idle", "%.3f" % idle[0], "%.3f" % idle[1], "--tokens", "1",
                        "--json"], capture_output=True, text=True)
    s = json.loads(p.stdout)
    return {n: s["outlets"][str(o)]["energy_trap_j"] / (t1 - t0) for n, o in OUTLETS.items()}


def _ls2(rows):
    """Least squares y = a x1 + b x2 (no intercept): rows of (x1, x2, y)."""
    s11 = sum(r[0] * r[0] for r in rows); s22 = sum(r[1] * r[1] for r in rows); s12 = sum(r[0] * r[1] for r in rows)
    s1y = sum(r[0] * r[2] for r in rows); s2y = sum(r[1] * r[2] for r in rows)
    det = s11 * s22 - s12 * s12
    return (s1y * s22 - s2y * s12) / det, (s2y * s11 - s1y * s12) / det


def fit(run=PH1, out_path=OUT, model="Qwen3-0.6B Q8_0"):
    """The rows from one phase study's strip data: ph1's by default (as registered); run / out_path / model for another
    model's study (2026-11-03: Qwen3-4B, ph3)."""
    rows = [json.loads(l) for l in open(os.path.join(run, "run.jsonl"))]
    idle = (next(r["t"] for r in rows if r["kind"] == "idle-before start"), next(r["t"] for r in rows if r["kind"] == "idle-before end"))
    idle_w = _power(run, idle, *idle)
    lpre, ldec, jpre, jdec, hand = [], [], [], [], {"laptop": [], "jetson": []}
    for r in rows:
        if r["kind"] != "req":
            continue
        x = r["data"]
        if x["plan"] in ("L", "J"):
            tp = x["prefill_ms"] / 1000.0
            pre = _power(run, idle, x["t_start"] + 0.5, x["t_start"] + tp - 0.5)
            dec = _power(run, idle, x["t_start"] + tp + 0.5, x["t_end"] - 0.5)
            if x["plan"] == "L":
                lpre.append((tp, pre["laptop"])); ldec.append((tp, x["t_end"] - x["t_start"] - tp, dec["laptop"]))
            else:
                jpre.append(pre["jetson"]); jdec.append(dec["jetson"])
        else:
            t_h0, t_h1 = x["t_first"], x["t_first"] + x["handoff_wall_s"]
            h = _power(run, idle, t_h0, t_h1)
            hand["laptop"].append(h["laptop"]); hand["jetson"].append(h["jetson"])
    best = None
    for k in range(50, 401):                                  # tau from 5.0 to 40.0 s
        tau = k / 10.0
        lr = [(min(t, tau) / t, max(0.0, t - tau) / t, p) for t, p in lpre]
        a, b = _ls2(lr)
        err = sum((a * r[0] + b * r[1] - r[2]) ** 2 for r in lr)
        if best is None or err < best[0]:
            best = (err, tau, a, b)
    _, tau, pb, ps = best
    dr = [(min(dd, max(0.0, tau - tp)) / dd, (dd - min(dd, max(0.0, tau - tp))) / dd, p) for tp, dd, p in ldec]
    tb, ts = _ls2(dr) if any(r[0] > 0 for r in dr) else (statistics.fmean(p for *_, p in ldec),) * 2
    out = {"made": "phase_power_20261028.py fit", "from": os.path.relpath(run, ROOT), "model": model,
           # a study without hand-off requests (ph3, 2026-11-04: the 4B's hand-off thrashed the Jetson) has no hand-off rows
           "rows": {"laptop": {"idle": idle_w["laptop"], "pp_burst": pb, "pp": ps, "burst_s": tau, "tg_burst": tb, "tg": ts,
                               **({"handoff": statistics.fmean(hand["laptop"])} if hand["laptop"] else {})},
                    "jetson": {"idle": idle_w["jetson"], "pp": statistics.fmean(jpre), "tg": statistics.fmean(jdec),
                               **({"handoff": statistics.fmean(hand["jetson"])} if hand["jetson"] else {})},
                    "pi5": {"idle": idle_w["pi5"]}, "pi4": {"idle": idle_w["pi4"]}},
           "fit_points": {"laptop_prefill_s_and_W": lpre, "laptop_decode_after_s_dur_s_W": ldec,
                          "jetson_prefill_W": jpre, "jetson_decode_W": jdec, "handoff_W": hand},
           "inputs": {"run": os.path.relpath(run, ROOT)}}
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote %s" % os.path.relpath(out_path, ROOT)); print(json.dumps(out["rows"], indent=1))


def energy(c, name, n, gen, rows):
    """J at all four outlets for plan `name`, with phase_plan_20261028's times and these rows."""
    t = PP.plan(c, name, n, gen, {o: {"idle": rows[o]["idle"], "pp": rows[o].get("pp", 0), "tg": rows[o].get("tg", 0)}
                                  for o in OUTLETS})
    P, D = PP.PLANS[name]
    idle = {o: rows[o]["idle"] for o in OUTLETS}
    others = lambda busy: sum(v for o, v in idle.items() if o not in busy)
    t_pre, t_dec, t_h = t["ttft_s"], t["decode_s"], t.get("handoff_s", 0.0)
    if P == "laptop":
        L = rows["laptop"]; tau = L["burst_s"]
        e = (L["pp_burst"] * min(t_pre, tau) + L["pp"] * max(0.0, t_pre - tau)) + t_pre * others({"laptop"})
    else:
        e = t_pre * (rows["jetson"]["pp"] + others({"jetson"}))
    if t_h:
        e += t_h * (rows["laptop"]["handoff"] + rows["jetson"]["handoff"] + others({"laptop", "jetson"}))
    if D == "laptop":
        L = rows["laptop"]; left = max(0.0, L["burst_s"] - t_pre)
        e += L["tg_burst"] * min(t_dec, left) + L["tg"] * max(0.0, t_dec - left) + t_dec * others({"laptop"})
    else:
        e += t_dec * (rows["jetson"]["tg"] + others({"jetson"}))
    return dict(t, energy_j=e)


if __name__ == "__main__":
    if sys.argv[1:] == ["fit"]:
        fit()
    elif sys.argv[1:2] == ["fit"] and len(sys.argv) == 5:      # fit <run dir> <out json> <model name> (2026-11-03)
        fit(os.path.join(ROOT, sys.argv[2]), os.path.join(ROOT, sys.argv[3]), sys.argv[4])
    else:
        raise SystemExit("usage: phase_power_20261028.py fit [<run dir> <out json> <model name>]")
