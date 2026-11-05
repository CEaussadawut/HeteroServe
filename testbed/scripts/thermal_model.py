#!/usr/bin/env python3
"""Upgrade 3, offline (2026-10-10): a two-parameter first-order thermal model of the Pi 4, fitted on the 1 Hz polls on file.

    python3 testbed/scripts/thermal_model.py fit [--out JSON] [--before EPOCH]

The model:
- The equation: dT/dt = (T_a + k u(t) - T) / tau.
  - u is 1 when the poll saw the governor at the top clock (gov_khz 1,800,000: computing), else 0. scaling_cur_freq keeps
    the governor's request under the firmware's soft limit too, so a throttled but busy board still reads u = 1.
  - T_a is the temperature the board settles to with nothing running: one per log, each session's own idle level, which
    the room sets.
  - k is the rise under continuous full load, and tau the time constant. Those two are the model's parameters, shared by
    every log.
- Discretised over each step's own dt: T[n+1] = a T[n] + (1 - a)(T_a + k u[n]), a = exp(-dt / tau), simulated from each
  segment's first sample.
  - A gap over MAX_GAP_S starts a new segment with the same T_a.
  - For a fixed tau the simulated trajectory is linear in (k, T_a per log). So tau is found on a log-spaced grid, refined
    once, and the rest by least squares.
- Steady state: T_ss = T_a + k d, with d the duty, the share of time at the top clock. A step from T0 comes within 1 C of
  T_ss after tau ln(|T0 - T_ss| / 1 C).
--before EPOCH fits only the logs that ended before EPOCH (a hold-out check on a later log).

The logs: every testbed/results/throttle_pool_*_pi4.log and every testbed/results/*/throttle_pi4.log (throttle_poll.sh v2,
parsed by throttle_summary.parse).
"""
import glob, hashlib, json, math, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RESD = os.path.join(ROOT, "testbed", "results")
sys.path.insert(0, HERE)
import throttle_summary as TS                               # noqa: E402

TOP_KHZ = 1800000
MAX_GAP_S = 3


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def log_files():
    return sorted(glob.glob(os.path.join(RESD, "throttle_pool_*_pi4.log"))) + sorted(glob.glob(os.path.join(RESD, "*", "throttle_pi4.log")))


def read(path):
    """[(epoch, temp C, u)] in time order; lines the parser rejects are skipped."""
    out = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        p = TS.parse(ln)
        if not p:
            continue
        t, kv = p
        m = re.match(r"([\d.]+)", kv.get("temp", ""))
        g = kv.get("gov_khz", "")
        if m and g.isdigit():
            out.append((t, float(m.group(1)), 1.0 if int(g) >= TOP_KHZ else 0.0))
    out.sort()
    return out


def segments(samples):
    segs, cur = [], []
    for s in samples:
        if cur and s[0] - cur[-1][0] > MAX_GAP_S:
            segs.append(cur); cur = []
        cur.append(s)
    if cur:
        segs.append(cur)
    return [s for s in segs if len(s) >= 2]


def design(logs, tau):
    """Rows of the linear system for a fixed tau: (target T_obs - a^n T0, coefficient of k, log index, coefficient of T_a)."""
    rows = []
    for j, segs in enumerate(logs):
        for seg in segs:
            T0, g, decay = seg[0][1], 0.0, 1.0
            for (t0, _, u), (t1, T, _) in zip(seg, seg[1:]):
                a = math.exp(-(t1 - t0) / tau)
                g = a * g + (1 - a) * u                  # the forced response to the busy signal, per unit k
                decay *= a                               # a^n over the elapsed time
                rows.append((T - decay * T0, g, j, 1.0 - decay))
    return rows


def solve(logs, tau):
    import numpy as np
    rows = design(logs, tau)
    J = len(logs)
    X = np.zeros((len(rows), 1 + J)); y = np.zeros(len(rows))
    for i, (tgt, gk, j, ga) in enumerate(rows):
        X[i, 0], X[i, 1 + j], y[i] = gk, ga, tgt
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    rmse = float(np.sqrt(np.mean((X @ b - y) ** 2)))
    return {"tau_s": tau, "k_c": float(b[0]), "t_a": [float(x) for x in b[1:]], "rmse_c": rmse, "n": len(rows)}


def fit(files, before=None):
    logs, used = [], []
    for f in files:
        s = read(f)
        if not s or (before is not None and s[-1][0] >= before):
            continue
        segs = segments(s)
        if segs:
            logs.append(segs); used.append(f)
    grid = [10 * 1.12 ** i for i in range(60)]           # 10 s .. ~8,000 s
    best = min((solve(logs, t) for t in grid), key=lambda r: r["rmse_c"])
    lo, hi = best["tau_s"] / 1.12, best["tau_s"] * 1.12
    best = min((solve(logs, lo + (hi - lo) * i / 40) for i in range(41)), key=lambda r: r["rmse_c"])
    best["logs"] = [{"file": os.path.relpath(f, ROOT), "sha256": sha(f), "t_a_c": round(ta, 2),
                     "samples": sum(len(s) for s in segs)} for f, segs, ta in zip(used, logs, best["t_a"])]
    del best["t_a"]
    return best


def steady(t_a, k, duty):
    return t_a + k * duty


def predict_trajectory(samples, t_a, k, tau):
    """Simulated temperatures for one log's samples, from its first, with its own busy signal."""
    out, T = [samples[0][1]], samples[0][1]
    for (t0, _, u), (t1, _, _) in zip(samples, samples[1:]):
        a = math.exp(-(t1 - t0) / tau)
        T = a * T + (1 - a) * (t_a + k * u)
        out.append(T)
    return out


def main(argv):
    if not argv or argv[0] != "fit":
        sys.exit(__doc__)
    before = float(argv[argv.index("--before") + 1]) if "--before" in argv else None
    r = fit(log_files(), before)
    print("tau %.1f s, k %.2f C, RMSE %.3f C over %d steps, %d logs" % (r["tau_s"], r["k_c"], r["rmse_c"], r["n"], len(r["logs"])))
    print("T_a by log: min %.1f, max %.1f C" % (min(l["t_a_c"] for l in r["logs"]), max(l["t_a_c"] for l in r["logs"])))
    if "--out" in argv:
        json.dump(r, open(argv[argv.index("--out") + 1], "w"), indent=1)
    return r


if __name__ == "__main__":
    main(sys.argv[1:])
