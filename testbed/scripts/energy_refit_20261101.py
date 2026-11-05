#!/usr/bin/env python3
"""The laptop's power rows refitted from the four-node evaluation's own strip data (2026-11-01, the scheduler review's item 2;
the user: "do item 1 to 4 then 6-7"). Laptop only; reads the scored run directories and their registrations, re-scores nothing.

    python3 testbed/scripts/energy_refit_20261101.py

The energy model (energy_model.py) under-predicted the four-node evaluation's gross J per decoded token, all four outlets, by a
median 21.6 % (52 placements, -2 to -32 %), and the per-outlet split put nearly all of it on the laptop: its power table is a
single-board run of 2026-10-06 (idle 14.05 W, decode 32.12 W), while in the evaluation's sessions it idled at about 20.5 W
and decoded alone at 35-42 W. The boards match their table. This fits the laptop's power from the evaluation itself, per
placement over the decode windows of its requests (first token to last; the strip's apower, trapezoid), as
    P_laptop = idle + (active - idle) x f + per_token / T   (the host: its serving work per token)
                                           + pool_host       (the host of a plan with workers)
with f the laptop's busy fraction of a token by the predictor's own terms (energy_model.per_node, as each registration
priced it) and T the measured time per token. A laptop outside the plan draws idle (it runs the harness). Fitted:
  idle       the mean of the sessions' idle-before windows (the laptop's outlet)
  active, per_token   least squares of P on 1 / T over the placements with the laptop alone (f = 1)
  pool_host  the mean residual over the placements the laptop hosts with workers
Checked leave one model out (each model's placements predicted from a fit without them), all four outlets, against the old
model's registered predictions. The boards keep energy_model.power_table(). The model files are the evaluation's final runs
(fe4_tables_20261023.RUNS, its exclusions applied). Writes testbed/dataset/power_refit_20261101.json (never overwritten).
"""
import datetime, hashlib, json, os, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import fe4_tables_20261023 as F                             # noqa: E402
import shelly_strip as S                                    # noqa: E402
import energy_model as EM                                   # noqa: E402
import predict_pool as P                                    # noqa: E402

OUT = os.path.join(ROOT, "testbed", "dataset", "power_refit_20261101.json")
BOARDS = ("pi5", "pi4", "jetson")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def points():
    """One record per placement: its plan, measured time per token, per-outlet mean W over its decode windows, the session's
    idle-before W, the predictor's busy fractions, and the old model's registered all-outlet prediction."""
    os.environ["HETEROSERVE_RESERVE_MB"] = ""
    out, files = [], {}
    for label, run, regf in F.RUNS:
        d = os.path.join(F.R, run)
        regp = os.path.join(ROOT, "testbed", "dataset", regf)
        reg = json.load(open(regp))
        rows = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
        files[os.path.relpath(regp, ROOT)] = sha(regp)
        for f in ("run.jsonl", "strip.csv"):
            files[os.path.relpath(os.path.join(d, f), ROOT)] = sha(os.path.join(d, f))
        outlets = next(r for r in rows if r["kind"] == "start")["data"]["outlets"]
        log = S.load_log(os.path.join(d, "strip.csv"))
        pts = {n: [(r["t"], float(r["raw"]["o%d_apower" % o])) for r in log if r["ok"] and r["raw"]["o%d_apower" % o] != ""]
               for n, o in outlets.items()}
        i0 = next(r["t"] for r in rows if r["kind"] == "idle-before start")
        i1 = next(r["t"] for r in rows if r["kind"] == "idle-before end")
        idle = {n: S.integrate(pts[n], i0, i1) / (i1 - i0) for n in outlets}
        acc = {}
        for r in rows:
            if r["kind"] != "req" or r["data"]["placement"] in F.EXCLUDE:
                continue
            x = r["data"]
            ev = [e for e in x["events"] if e[1]]
            t0, t1 = x["sent_wall"] + ev[0][0], x["sent_wall"] + x["done_s"]
            a = acc.setdefault(x["placement"], {"T": 0.0, "gaps": 0, "E": {n: 0.0 for n in outlets}})
            a["T"] += t1 - t0
            a["gaps"] += max(1, len(ev) - 1)
            for n in outlets:
                a["E"][n] += S.integrate(pts[n], t0, t1)
        shapes, pairs, C = EM._inputs(P.MODELS[reg["model"]][1])
        q = shapes[P.MODELS[reg["model"]][0]]
        for lab, a in acc.items():
            c = reg["configs"][lab]
            assign = [tuple(x) for x in c["assign"]]
            head_at = 0 if c["head"] == "host" and len(assign) > 1 else None
            L, _ = P.link_view(pairs, assign[0][0], [n for n, _ in assign])
            busy, xfer = EM.per_node(assign, q, C, L, head_at=head_at, rtt_trips=reg["rtt_trips"])
            tpred = sum(b["tg"] for b in busy.values()) + xfer["tg"]
            role = "host" if assign[0][0] == "laptop" else ("worker" if "laptop" in dict(assign) else "none")
            out.append({"model": label, "key": reg["model"], "placement": lab, "assign": [list(x) for x in assign],
                        "head_at": head_at, "laptop_role": role, "pool": role == "host" and len(assign) > 1,
                        "T_s": a["T"] / a["gaps"], "W": {n: a["E"][n] / a["T"] for n in outlets},
                        "idle_before_W": idle, "busy_frac": {n: busy[n]["tg"] / tpred for n in busy},
                        "measured_all_j": sum(a["E"].values()) / a["gaps"],
                        "old_predicted_all_j": c["predicted"]["decode_gross_j_all_boards"]})
    return out, files


def fit(pts, idle):
    alone = [p for p in pts if p["laptop_role"] == "host" and not p["pool"]]
    xs, ys = [1.0 / p["T_s"] for p in alone], [p["W"]["laptop"] for p in alone]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    per_token = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    active = my - per_token * mx
    lap = {"idle_w": idle, "active_w": active, "per_token_j": per_token, "pool_host_w": 0.0}
    pools = [p for p in pts if p["pool"]]
    lap["pool_host_w"] = statistics.fmean(p["W"]["laptop"] - laptop_w(lap, p["busy_frac"].get("laptop", 0.0), p["T_s"], "host", False)
                                         for p in pools)
    return lap


def laptop_w(lap, f, T_s, role, pool):
    """The refit's laptop W: outside the plan idle; in it, idle + (active - idle) x f, its host's per-token work and the
    pool-host term (energy_model.laptop_refit_w is the same formula; this copy fits it)."""
    if role == "none":
        return lap["idle_w"]
    w = lap["idle_w"] + (lap["active_w"] - lap["idle_w"]) * f
    if role == "host":
        w += lap["per_token_j"] / T_s + (lap["pool_host_w"] if pool else 0.0)
    return w


def all_j(p, lap, power):
    j = laptop_w(lap, p["busy_frac"].get("laptop", 0.0), p["T_s"], p["laptop_role"], p["pool"]) * p["T_s"]
    for n in BOARDS:
        f = p["busy_frac"].get(n)
        j += (power[n]["idle"] if f is None else power[n]["tg"] * f + power[n]["idle"] * (1 - f)) * p["T_s"]
    return j


def summary(errs):
    a = [abs(e) for e in errs]
    return {"n": len(errs), "median_pct": round(statistics.median(errs), 2), "median_abs_pct": round(statistics.median(a), 2),
            "worst_pct": round(max(errs, key=abs), 2), "within_5pct": sum(x <= 5 for x in a), "within_10pct": sum(x <= 10 for x in a)}


def main():
    if os.path.exists(OUT):
        raise SystemExit("%s exists: never overwritten" % os.path.relpath(OUT, ROOT))
    pts, files = points()
    power = EM.power_table()
    idle = statistics.fmean(p["idle_before_W"]["laptop"] for p in pts)
    lap = fit(pts, idle)
    old, ins, lomo = [], [], []
    for p in pts:
        m = p["measured_all_j"]
        old.append(100 * (p["old_predicted_all_j"] / m - 1))
        ins.append(100 * (all_j(p, lap, power) / m - 1))
    for mdl in sorted(set(p["model"] for p in pts)):
        l2 = fit([p for p in pts if p["model"] != mdl], idle)
        lomo += [100 * (all_j(p, l2, power) / p["measured_all_j"] - 1) for p in pts if p["model"] == mdl]
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "why": "the scheduler review's item 2 (testbed/JOURNAL.md, 2026-11-01): the energy model under-predicted the four-node "
                  "evaluation by a median 21.6 %, nearly all of it the laptop",
           "model": "P_laptop = idle_w + (active_w - idle_w) x f + per_token_j / T (the host) + pool_host_w (the host of a plan with "
                    "workers); outside the plan idle_w. f: the laptop's busy fraction of a token (energy_model.per_node); T: the time "
                    "per token. The boards: energy_model.power_table(), as before (active x f + idle x (1 - f); outside the plan idle)",
           "laptop": {k: round(v, 4) for k, v in lap.items()},
           "boards": {n: {k: power[n][k] for k in ("idle", "tg", "source", "sha256")} for n in BOARDS},
           "fit": {"placements": len(pts), "laptop_alone": sum(1 for p in pts if p["laptop_role"] == "host" and not p["pool"]),
                   "laptop_hosts_a_pool": sum(1 for p in pts if p["pool"]),
                   "laptop_outside": sum(1 for p in pts if p["laptop_role"] == "none"),
                   "all_outlets_error": {"old_model_registered": summary(old), "refit_in_sample": summary(ins),
                                         "refit_leave_one_model_out": summary(lomo)}},
           "points": pts,
           "inputs_sha256": dict(files, **{os.path.relpath(os.path.join(HERE, f), ROOT): sha(os.path.join(HERE, f))
                                          for f in ("energy_refit_20261101.py", "fe4_tables_20261023.py", "energy_model.py",
                                                    "predict_pool.py", "shelly_strip.py")})}
    json.dump(res, open(OUT, "w"), indent=1)
    print("laptop: idle %.2f W, active %.2f W, per token %.3f J (host), pool host %.2f W" % (
        lap["idle_w"], lap["active_w"], lap["per_token_j"], lap["pool_host_w"]))
    for k, v in res["fit"]["all_outlets_error"].items():
        print("%-28s median %+.1f %%, |error| median %.1f %%, worst %+.1f %%, within 5 %%: %d of %d, within 10 %%: %d" % (
            k, v["median_pct"], v["median_abs_pct"], v["worst_pct"], v["within_5pct"], v["n"], v["within_10pct"]))
    print("wrote %s" % os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
