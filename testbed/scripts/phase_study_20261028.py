#!/usr/bin/env python3
"""The phase study (ph1, 2026-10-28; item 3): the scheduler's phase placement against measurement, energy at the wall.

    python3 testbed/scripts/phase_study_20261028.py run   --reg testbed/dataset/ph1_predictions_20261028.json
    python3 testbed/scripts/phase_study_20261028.py score testbed/results/ph1_<stamp>

run: the registration lists "arms" ({id: {n, gen, plan}}) and "order" (arm ids, interleaved). Gates: the registration
read-only; the laptop quiet on its charger (phase_calib_20261028.laptop_quiet); the Jetson's procedure
(jetson_proc_20261028.py). The strip logs all four outlets at 5 Hz (shelly_strip.py log) from before the servers start to
after they stop. phase_serve_20261028.py's two servers; the prompts tokenized on both and compared; one unmeasured warm-up request per plan
(512 tokens, 8 generated: a server's first request compiles its pipelines); an idle window of
IDLE_S seconds with both servers loaded ("IDLE_BEFORE start/end"); then each arm in order via phase_serve.run(), PAUSE_S
seconds apart. Everything goes to run.jsonl and the raw responses to phase.log, in testbed/results/ph1_<stamp>/.
score: per request, the gross energy at all four outlets over [t_start, t_end] (shelly_strip.py summarise, the trapezoid
method); per arm, the means over its requests of total time, time to first token and energy, against the registration's
predictions; the hypotheses as registered. Writes score.json and score.txt. Nothing is filled in.
"""
import argparse, datetime, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_serve_20261028 as S                           # noqa: E402
import phase_calib_20261028 as C                           # noqa: E402
import jetson_proc_20261028 as JP                           # noqa: E402
import pd_handoff as PH                                     # noqa: E402

STRIP = os.path.join(HERE, "shelly_strip.py")
IDLE_S, PAUSE_S = 60, 10


def sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p if os.path.isabs(p) else os.path.join(ROOT, p), "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(regpath):
    reg = json.load(open(regpath))
    if os.access(regpath, os.W_OK):
        raise SystemExit("FATAL: %s is writable (register first)" % regpath)
    if reg.get("phase_model", "qwen3_0.6b") != S.PHASE_MODEL:      # 2026-11-03: phase_serve's model comes from PHASE_MODEL
        raise SystemExit("FATAL: the registration's model is %s; run with PHASE_MODEL=%s (now %s)" % (
            reg.get("phase_model", "qwen3_0.6b"), reg.get("phase_model", "qwen3_0.6b"), S.PHASE_MODEL))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "%s_%s" % (reg["out_prefix"], stamp))
    os.makedirs(d)
    J = open(os.path.join(d, "run.jsonl"), "a")

    def rec(kind, data=None):
        J.write(json.dumps({"t": time.time(), "kind": kind, "data": data}) + "\n"); J.flush()
    log = S.Log("study", os.path.join(d, "phase.log"))
    say = lambda s: log("run", s)
    say("########## %s: registration %s (%s)" % (reg["out_prefix"], os.path.relpath(regpath, ROOT), sha(regpath)[:16]))
    rec("start", {"registration": os.path.relpath(regpath, ROOT), "registration_sha256": sha(regpath), "outlets": reg["outlets"]})
    C.laptop_quiet(say)
    st = JP.up(lambda s: log("jetson", s), "%s_%s" % (reg["out_prefix"], stamp))
    lp = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(d, "strip.csv"), "--hz", "5"],
                          stdout=subprocess.DEVNULL, stderr=open(os.path.join(d, "strip.stderr"), "w"))
    try:
        time.sleep(5)
        S.start("laptop", log, "%s_%s" % (reg["out_prefix"], stamp)); S.start("jetson", log, "%s_%s" % (reg["out_prefix"], stamp))
        ids = {n: S.prompt_ids("laptop", n) for n in sorted({a["n"] for a in reg["arms"].values()})}
        for n in ids:
            if S.prompt_ids("jetson", n) != ids[n]:
                raise SystemExit("FATAL: the servers tokenize the %d-token prompt differently" % n)
        # warm-up, not measured: a server's first request compiles its pipelines (the smoke's first Jetson prefill of 1,036
        # tokens took 13.9 s, the calibration's, after a 524-token one, 8.6 s); one short request per plan before the idle window
        wu = S.prompt_ids("laptop", 512)
        # the registration's plans only (2026-11-04: ph3 registers L and J; ph1 and ph2 all three, as before)
        for plan in [p for p in ("L", "J", "H") if any(a["plan"] == p for a in reg["arms"].values())]:
            toks, r = S.run(plan, wu, 8, log, "warm-up")
            rec("warm-up", {k: v for k, v in r.items() if k != "tokens"})
        rec("idle-before start"); time.sleep(IDLE_S); rec("idle-before end")
        for i, aid in enumerate(reg["order"], 1):
            a = reg["arms"][aid]
            try:
                toks, r = S.run(a["plan"], ids[a["n"]], a["gen"], log, "%s r%d" % (aid, i))
                r.update({"arm": aid, "i": i, "tokens": toks})
                rec("req", r)
            except Exception as e:
                rec("req-failed", {"arm": aid, "i": i, "error": repr(e)[:400]})
                say("!!!!! arm %s run %d failed: %r" % (aid, i, e))
            time.sleep(reg.get("pause_s", PAUSE_S))         # ph2 registers 60 s, so the laptop's turbo budget refills
    finally:
        S.stop("laptop", log); S.stop("jetson", log)
        JP.down(lambda s: log("jetson", s), st)
        time.sleep(5)
        lp.terminate()
        try:
            lp.wait(timeout=20)
        except subprocess.TimeoutExpired:
            lp.kill()
        rec("end")
    say("RUN DONE %s" % os.path.relpath(d, ROOT))
    score(d)


def energy(d, t0, t1, idle, outlets):
    p = subprocess.run([sys.executable, STRIP, "summarise", os.path.join(d, "strip.csv"), "--window", "%.6f" % t0, "%.6f" % t1,
                        "--idle", "%.6f" % idle[0], "%.6f" % idle[1], "--tokens", "1", "--json"], capture_output=True, text=True)
    s = json.loads(p.stdout)
    return {o: s["outlets"][str(o)]["energy_trap_j"] for o in outlets}


def score(d):
    rows = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    start = next(r for r in rows if r["kind"] == "start")["data"]
    reg = json.load(open(os.path.join(ROOT, start["registration"])))
    idle = (next(r["t"] for r in rows if r["kind"] == "idle-before start"), next(r["t"] for r in rows if r["kind"] == "idle-before end"))
    outlets = reg["outlets"]
    per, reqs = {}, []
    for r in rows:
        if r["kind"] != "req":
            continue
        x = r["data"]
        e = energy(d, x["t_start"], x["t_end"], idle, outlets.values())
        q = {"arm": x["arm"], "i": x["i"], "plan": x["plan"], "n": x["prompt_n"], "total_s": round(x["t_end"] - x["t_start"], 3),
             "ttft_s": round(x["ttft_s"], 3), "energy_j": round(sum(e.values()), 2),
             "energy_by_outlet_j": {k: round(e[v], 2) for k, v in outlets.items()},
             "handoff_s": round(x["handoff_wall_s"], 3) if x.get("handoff_wall_s") is not None else None,
             "tokens": x.get("tokens")}
        reqs.append(q)
        per.setdefault(x["arm"], []).append(q)
    arms = {}
    for aid, a in reg["arms"].items():
        got = per.get(aid, [])
        p = a["predicted"]
        if len(got) < reg["order"].count(aid):
            arms[aid] = {"complete": False, "requests": len(got)}
            continue
        m = {k: statistics.fmean(g[k] for g in got) for k in ("total_s", "ttft_s", "energy_j")}
        arms[aid] = {"complete": True, "requests": len(got), "n": a["n"], "plan": a["plan"], **{k: round(v, 3) for k, v in m.items()},
                     "err_total_pct": round(100 * (m["total_s"] / p["total_s"] - 1), 2),
                     "err_energy_pct": round(100 * (m["energy_j"] / p["energy_j"] - 1), 2),
                     "err_ttft_pct": round(100 * (m["ttft_s"] / p["ttft_s"] - 1), 2),
                     "spread_total_pct": round(100 * (max(g["total_s"] for g in got) / min(g["total_s"] for g in got) - 1), 2),
                     "spread_energy_pct": round(100 * (max(g["energy_j"] for g in got) / min(g["energy_j"] for g in got) - 1), 2)}
    H = reg["hypotheses"]
    ok = all(v.get("complete") for v in arms.values())
    vs = lambda c: ("HOLDS" if c else "FAILS") if ok else "INCOMPLETE"
    ns = sorted({a["n"] for a in reg["arms"].values()})
    by_n = {n: {a["plan"]: aid for aid, a in reg["arms"].items() if a["n"] == n} for n in ns}
    hyp = {}
    if ok:
        pick_e = {n: min(by_n[n], key=lambda pl: arms[by_n[n][pl]]["energy_j"]) for n in ns}
        pick_t = {n: min(by_n[n], key=lambda pl: arms[by_n[n][pl]]["total_s"]) for n in ns}
    else:
        pick_e = pick_t = {}
    hyp["PH-PICK-E"] = {"verdict": vs(all(pick_e.get(n) == reg["picks"]["energy"][str(n)] for n in ns)),
                        "measured": pick_e, "predicted": reg["picks"]["energy"]}
    hyp["PH-PICK-T"] = {"verdict": vs(all(pick_t.get(n) == reg["picks"]["time"][str(n)] for n in ns)),
                        "measured": pick_t, "predicted": reg["picks"]["time"]}
    hyp["PH-TIME"] = {"verdict": vs(all(abs(v["err_total_pct"]) <= 100 * H["PH-TIME"]["tol"] for v in arms.values() if v.get("complete"))),
                      "err_total_pct": {k: v.get("err_total_pct") for k, v in arms.items()}}
    hyp["PH-ENERGY"] = {"verdict": vs(all(abs(v["err_energy_pct"]) <= 100 * H["PH-ENERGY"]["tol"] for v in arms.values() if v.get("complete"))),
                        "err_energy_pct": {k: v.get("err_energy_pct") for k, v in arms.items()}}
    if ok:
        tt = {n: arms[by_n[n]["H"]]["ttft_s"] / arms[by_n[n]["L"]]["ttft_s"] - 1 for n in ns if "H" in by_n[n] and "L" in by_n[n]}
    else:
        tt = {}
    if "PH-TTFT" in H:                                     # registered with a hand-off arm (ph1, ph2; not ph3)
        hyp["PH-TTFT"] = {"verdict": vs(all(abs(v) <= H["PH-TTFT"]["tol"] for v in tt.values())), "h_vs_l_pct": {n: round(100 * v, 2) for n, v in tt.items()}}
    hyp["PH-REPEAT"] = {"verdict": vs(all(v["spread_total_pct"] <= 100 * H["PH-REPEAT"]["tol"] for v in arms.values() if v.get("complete"))),
                        "spread_total_pct": {k: v.get("spread_total_pct") for k, v in arms.items()}}
    if "PH-BETTER" in H:                                   # ph2: the per-model power rows against the TinyLlama power table
        tab = {k: round(100 * (v["energy_j"] / reg["arms"][k]["predicted"]["energy_j_table"] - 1), 2)
               for k, v in arms.items() if v.get("complete")}
        hyp["PH-BETTER"] = {"verdict": vs(all(abs(arms[k]["err_energy_pct"]) < abs(tab[k]) for k in tab)),
                            "err_rows_pct": {k: arms[k]["err_energy_pct"] for k in tab}, "err_table_pct": tab}
    agree = {}
    for n in ns:                                           # the text: how far each plan's tokens agree with the laptop alone's
        first = {pl: next((q["tokens"] for q in reqs if q["arm"] == by_n[n][pl]), None) for pl in by_n[n]}
        if first.get("L"):
            agree[n] = {pl: PH.prefix(first[pl] or [], first["L"]) for pl in first}
    s = {"run": os.path.relpath(d, ROOT), "registration": start["registration"], "registration_sha256": sha(start["registration"]),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, h in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and sha(f) != h),
         "arms": arms, "hypotheses": hyp, "token_agreement_with_L": agree,
         "requests": [{k: v for k, v in q.items() if k != "tokens"} for q in reqs],
         "failed": [r["data"] for r in rows if r["kind"] == "req-failed"]}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    L = ["%s: the phase scheduler against measurement; code changed since registration: %s" % (
        reg["out_prefix"], s["code_changed_since_registration"] or "none"), ""]
    for k, h in hyp.items():
        L.append("%-10s %-10s %s" % (k, h["verdict"], {x: y for x, y in h.items() if x != "verdict"}))
    L.append("")
    for aid, v in arms.items():
        p = reg["arms"][aid]["predicted"]
        L.append("arm %s: predicted total %.2f s, ttft %.2f s, %.1f J; measured %s" % (aid, p["total_s"], p["ttft_s"], p["energy_j"], v))
    L.append("token agreement with L (tokens in common from the start): %s" % agree)
    if s["failed"]:
        L.append("FAILED requests: %s" % s["failed"])
    open(os.path.join(d, "score.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run", "score")); ap.add_argument("path", nargs="?")
    ap.add_argument("--reg")
    a = ap.parse_args()
    run(a.reg) if a.cmd == "run" else score(a.path)
