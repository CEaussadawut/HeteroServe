#!/usr/bin/env python3
"""Calibrate the phase scheduler's costs through its own serving path (2026-10-28; item 3). Not a registered test: its
fits are the inputs the registration (phase study ph1) predicts from, and its runs are not reused as test data.

    python3 testbed/scripts/phase_calib_20261028.py run            # measure (the laptop quiet, the Jetson's procedure)
    python3 testbed/scripts/phase_calib_20261028.py fit <log>      # fit and write testbed/dataset/phase_costs_20261028.json

run: phase_serve_20261028.py's two servers (Qwen3-0.6B Q8_0, the pin); per prompt length n in NS, plans L and J (33 tokens:
the prefill of n tokens, then 32 decoded at depth n..n+32), two repetitions, alternating; then plan H at HS (17 tokens), two
repetitions, for the hand-off's save, transfer and restore. Gates first: the laptop quiet on its charger (as
lc4_20261024.py's gate: no listed apps, load < 1.5, AC on, not charging, Wi-Fi off; three passes 30 s apart, up to 2 h), and
the Jetson's procedure (jetson_proc_20261028.py). The raw log is testbed/results/phase_calib_<stamp>.log.
fit: per node, prefill ms = a + b n + c n^2 over (n, prompt_ms); decode ms per token = e + f d over (n + 16, the server's
predicted_per_token_ms); the hand-off's three parts (wall seconds, as the orchestrator sees them: save, transfer with its
receiver's ssh login, restore), each = i + s x state bytes; the state's bytes = fixed + per-token x tokens, fitted on the
saved files' sizes.
"""
import argparse, datetime, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import phase_serve_20261028 as S                           # noqa: E402
import jetson_proc_20261028 as JP                           # noqa: E402

NS = [512, 1024, 2048, 4096, 8192]
HS = [1024, 4096, 8192]
APPS = ("zen", "antigravity-ide", "notion-snap-reb")
OUT = os.path.join(ROOT, "testbed", "dataset", "phase_costs_20261028.json")
if S.PHASE_MODEL != "qwen3_0.6b":     # 2026-11-03: another model (phase_serve's PHASE_MODEL) gets its own costs and sizes
    NS = [n for n in (256, 512, 1024, 2048) if n + 300 <= S.CTX]
    HS = [n for n in (512, 1024, 2048) if n + 300 <= S.CTX]
    OUT = os.path.join(ROOT, "testbed", "dataset", "phase_costs_%s_20261103.json" % S.PHASE_MODEL)


def laptop_quiet(say):
    # 2026-11-04 (the user: "it's ok with zen open, RAM 8GB is reserved"): QUIET_ALLOW_APPS="zen" lets the listed apps stay
    # open; the gate line records them. Unset, the gate is as before.
    allowed = set(os.environ.get("QUIET_ALLOW_APPS", "").replace(",", " ").split())
    t0, ok = time.time(), 0
    while True:
        ps = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout.split()
        load = float(open("/proc/loadavg").read().split()[0])
        ac = open("/sys/class/power_supply/AC/online").read().strip()
        charging = any(open(p).read().strip() == "Charging" for p in
                       [os.path.join("/sys/class/power_supply", x, "status") for x in os.listdir("/sys/class/power_supply") if x.startswith("BAT")])
        wifi = subprocess.run(["nmcli", "radio", "wifi"], capture_output=True, text=True).stdout.strip()
        apps = sorted({x for x in ps if x in APPS and x not in allowed})
        good = not apps and load < 1.5 and ac == "1" and not charging and wifi != "enabled"
        ok = ok + 1 if good else 0
        if ok >= 3:
            break
        if time.time() - t0 >= 7200:
            if ac != "1":
                say("FATAL: the laptop is not on its charger after 2 h"); sys.exit(1)
            say("NOT QUIET after 2 h (apps %s load %.2f charging %s wifi %s): going on, recorded" % (apps, load, charging, wifi))
            break
        time.sleep(30)
    say("gate laptop: apps %s load %.2f ac %s charging %s wifi %s (waited %d s)" % (apps, load, ac, charging, wifi, time.time() - t0)
        + ("; open and allowed (QUIET_ALLOW_APPS): %s" % sorted({x for x in ps if x in allowed}) if allowed else ""))


def run():
    log = S.Log("calib")
    say = lambda s: log("gate", s)
    laptop_quiet(say)
    tag = os.path.basename(log.path)[:-4]
    st = JP.up(lambda s: log("jetson", s), tag)
    try:
        S.start("laptop", log, tag); S.start("jetson", log, tag)
        ids = {n: S.prompt_ids("laptop", n) for n in sorted(set(NS + HS))}
        for n in ids:
            if S.prompt_ids("jetson", n) != ids[n]:
                raise SystemExit("FATAL: the servers tokenize the %d-token prompt differently" % n)
        for r in (1, 2):
            for n in NS:
                for plan in (("L", "J") if r == 1 else ("J", "L")):
                    toks, rec = S.run(plan, ids[n], 33, log, "calib r%d n%d" % (r, n))
                    log("calib result", rec)
        for r in (1, 2):
            for n in HS:
                try:
                    toks, rec = S.run("H", ids[n], 17, log, "calib r%d n%d" % (r, n))
                    log("calib result", rec)
                except Exception as e:                  # a hand-off that does not fit is a finding, not a crash
                    log("calib result", {"plan": "H", "prompt_n": len(ids[n]), "failed": repr(e)[:300]})
    finally:
        S.stop("laptop", log); S.stop("jetson", log)
        JP.down(lambda s: log("jetson", s), st)
    log("calib", "DONE %s" % os.path.relpath(log.path, ROOT))


def lstsq(X, y):
    """Least squares by the normal equations (no numpy on purpose: the record's scripts run on stock Python)."""
    k = len(X[0])
    A = [[sum(r[i] * r[j] for r in X) for j in range(k)] for i in range(k)]
    b = [sum(r[i] * v for r, v in zip(X, y)) for i in range(k)]
    for i in range(k):                                     # Gauss-Jordan with partial pivoting
        p = max(range(i, k), key=lambda r: abs(A[r][i]))
        A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
        for r in range(k):
            if r != i:
                f = A[r][i] / A[i][i]
                A[r] = [x - f * z for x, z in zip(A[r], A[i])]; b[r] -= f * b[i]
    return [b[i] / A[i][i] for i in range(k)]


def fit(path):
    recs = []
    for ln in open(path):
        m = re.match(r"^\S+ calib result (\{.*\})$", ln.strip())
        if m:
            recs.append(json.loads(m.group(1)))
    out = {"made": "phase_calib_20261028.py fit", "at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
           "from": os.path.relpath(os.path.abspath(path), ROOT), "model": S.MODEL["name"], "model_sha256": S.MODEL_SHA,
           "prefill": {}, "decode": {}, "handoff": {}, "points": {}}
    for plan, node in (("L", "laptop"), ("J", "jetson")):
        pts = [r for r in recs if r.get("plan") == plan and r.get("prefill_ms")]
        cp = lstsq([[1.0, r["prompt_n"], r["prompt_n"] ** 2] for r in pts], [r["prefill_ms"] for r in pts])
        cd = lstsq([[1.0, r["prompt_n"] + 16] for r in pts], [r["decode_ms_per_token"] for r in pts])
        resid = lambda f, xs: max(abs(f(r) / v - 1) for r, v in xs)
        out["prefill"][node] = {"coef": cp, "max_rel_residual": round(resid(lambda r: cp[0] + cp[1] * r["prompt_n"] + cp[2] * r["prompt_n"] ** 2,
                                                                            [(r, r["prefill_ms"]) for r in pts]), 4)}
        out["decode"][node] = {"coef": cd, "max_rel_residual": round(resid(lambda r: cd[0] + cd[1] * (r["prompt_n"] + 16),
                                                                           [(r, r["decode_ms_per_token"]) for r in pts]), 4)}
        out["points"][node] = [{k: r[k] for k in ("prompt_n", "prefill_ms", "decode_ms_per_token")} for r in pts]
    hs = [r for r in recs if r.get("plan") == "H" and r.get("state_bytes")]
    hfail = [r for r in recs if r.get("plan") == "H" and r.get("failed")]
    if hfail:
        # 2026-11-04: the 4B's hand-offs to the 4 GB Jetson thrashed its memory from n = 1,024 (the slot file in /dev/shm plus
        # the restore's copy); a line through such a run would be meaningless, so the hand-off is not fitted, only recorded
        out["handoff_not_fitted"] = "%d of the run's hand-off requests failed" % len(hfail)
        out["points"]["handoff"] = [{k: r.get(k) for k in ("prompt_n", "state_bytes", "save_wall_s", "transfer_wall_s_with_ssh",
                                                            "restore_wall_s", "wall_s")} for r in hs]
    elif len({r["prompt_n"] for r in hs}) >= 2:
        sb = lstsq([[1.0, r["prompt_n"]] for r in hs], [r["state_bytes"] for r in hs])
        out["state"] = {"fixed_bytes": sb[0], "bytes_per_token": sb[1]}
        for k, f in (("save", "save_wall_s"), ("transfer", "transfer_wall_s_with_ssh"), ("restore", "restore_wall_s")):
            out["handoff"][k] = {"coef": lstsq([[1.0, r["state_bytes"]] for r in hs], [r[f] for r in hs]), "field": f}
        out["points"]["handoff"] = [{k: r.get(k) for k in ("prompt_n", "state_bytes", "save_wall_s", "transfer_wall_s_with_ssh",
                                                            "restore_wall_s", "transfer_s", "MB_per_s")} for r in hs]
    out["failed"] = [r for r in recs if r.get("failed")]
    json.dump(out, open(OUT, "w"), indent=1)
    print("wrote %s" % os.path.relpath(OUT, ROOT))
    print(json.dumps({k: out[k] for k in ("prefill", "decode", "handoff", "failed")} | ({"state": out["state"]} if "state" in out else {}), indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run", "fit")); ap.add_argument("log", nargs="?")
    a = ap.parse_args()
    run() if a.cmd == "run" else fit(a.log)
