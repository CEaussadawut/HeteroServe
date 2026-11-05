#!/usr/bin/env python3
"""The long-context plan with the laptop as a worker, exactly as registered (register_longctx_20261011.py).

    python3 testbed/scripts/longctx_20261011.py run testbed/dataset/longctx_predictions_20261011.json [--smoke]
    python3 testbed/scripts/longctx_20261011.py score testbed/results/longctx_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit, with the laptop otherwise quiet:
- prechecks: no serve_pool.sh state; the link unshaped; the Pi 4 and the Jetson idle, their llama.cpp trees and the laptop's
  at the pin and clean; the model's sha256 on the Pi 4; the Jetson as found; memory now against each board's registered
  need (the laptop's against its need plus the 8 GiB kept for the desktop); the laptop's worker port free;
- the Jetson's guard started by its NOTE's procedure and stopped last. Its clocks are locked as run_pool.sh locks a cell's
  (stored, locked, restored, checked equal). The Pi 4 is polled at 1 Hz (throttle_poll.sh);
- each run: the Jetson's ggml-rpc-server (serve_pool.sh's command line) and the laptop's (-H 192.168.1.108 -p 50053 -t 4,
  the characterised worker) started; the registered llama-bench on the Pi 4, its raw output tee'd there and copied back;
  both workers stopped by their exact argv, never pkill -f. The laptop's load, memory and clock are logged around each run.
--smoke: one run of `-p 0 -n 4 -d 0 -r 1` into longctx_smoke_<stamp>/, the plumbing only (never scored or reported).
score: per run, ms per token = 1000 / avg_ts at each depth. The means over the runs and their spread; LC-0, LC-4096,
LC-SLOPE and LC-GAIN as written. Writes <dir>/score.json.
"""
import datetime, json, os, re, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, sha, GUARD_PROCS
import power_modes_char_20261010 as PC                      # noqa: E402  STATE_CMD, parse_state, set_mode

TPOLL = os.path.join(HERE, "throttle_poll.sh")
NETSHAPE = os.path.join(HERE, "netshape.sh")
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
PIN = "b23701f77d47dad9de834d59ebfcbe25c9e8b46f"
LAPTOP_IP, LAPTOP_PORT, JETSON_IP, RPC_PORT = "192.168.1.108", 50053, "192.168.1.114", 50052
LAPTOP_RPC = "/opt/heteroserve/llama.cpp/build/bin/ggml-rpc-server"
LAPTOP_ARGS = ["-H", LAPTOP_IP, "-p", str(LAPTOP_PORT), "-t", "4"]
MODEL_SHA = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"
BUSY = 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true'
TREE = "git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l"
SMOKE_PROTO = "-t 4 -p 0 -n 4 -d 0 -r 1 -oe jsonl"


def jetson_rpc_pids():
    rc, o = SS.ssh("jetson", "for p in $(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/$p/cmdline | grep -qF -- '-H %s -p %d ' "
                             "&& echo $p; done" % (JETSON_IP, RPC_PORT))
    return o.split()


def laptop_state():
    la = open("/proc/loadavg").read().split()[0]
    av = next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable"))
    mhz = [float(l.split(":")[1]) for l in open("/proc/cpuinfo") if l.startswith("cpu MHz")]
    return "load=%s memavail_mib=%d mhz=%.0f" % (la, int(av) // 1024, sum(mhz) / len(mhz))


def run(reg_path, smoke=False):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("longctx_smoke_%s" if smoke else "longctx_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")
    t_start = time.time()

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    # registration b (2026-10-11) names several cells; a has one ("bench"), run as before
    cells = reg.get("cells") or [{"name": "pick", "args": reg["bench"]["args"], "protocol": reg["bench"]["protocol"],
                                  "runs": reg["runs"], "memory": reg["plan"]["memory"]}]
    if smoke:
        cells = [dict(c, protocol=SMOKE_PROTO, runs=1) for c in cells[:1]]
    say("registration %s (%s)%s; cells: %s" % (os.path.relpath(reg_path, ROOT), SS.sha(reg_path)[:16], " SMOKE: never scored" if smoke else "",
                                               "; ".join("%s: %s %s x%d" % (c["name"], c["args"], c["protocol"], c["runs"]) for c in cells)))
    if os.path.exists(STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if subprocess.run(["bash", NETSHAPE, "check"], capture_output=True).returncode:
        say("FATAL: the Pi 4 is shaped, or did not answer"); sys.exit(1)
    for n in ("pi4", "jetson"):
        rc, b = SS.ssh(n, BUSY)
        if b.strip():
            say("FATAL: %s is busy: %s" % (n, b.strip())); sys.exit(1)
        rc, t = SS.ssh(n, TREE)
        if t.split() != [PIN, "0"]:
            say("FATAL: %s's llama.cpp is %r, not the pin clean" % (n, t)); sys.exit(1)
    head = subprocess.run(["git", "-C", "/opt/heteroserve/llama.cpp", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", "/opt/heteroserve/llama.cpp", "status", "--porcelain", "--untracked-files=no"],
                           capture_output=True, text=True).stdout.strip()
    if head != PIN or dirty:
        say("FATAL: the laptop's llama.cpp is %s%s" % (head[:12], " (dirty)" if dirty else "")); sys.exit(1)
    if (" %s:%d " % (LAPTOP_IP, LAPTOP_PORT)) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
        say("FATAL: the laptop's port %d is taken" % LAPTOP_PORT); sys.exit(1)
    rc, out = SS.ssh("pi4", "sha256sum ~/models/qwen3-14b-q4_k_m.gguf | cut -d' ' -f1", timeout=900)
    if out.strip() != MODEL_SHA:
        say("FATAL: the model on the Pi 4 is %s" % out.strip()[:16]); sys.exit(1)
    for c in cells:
        mem = c["memory"]
        for n in [x for x in ("pi4", "jetson") if x in mem]:
            rc, a = SS.ssh(n, "awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo")
            need = mem[n]["need_mib"] + mem[n]["reserve_mib"]
            if int(a.strip() or 0) < need:
                say("REFUSED: %s has %s MiB, cell %s needs %d" % (n, a.strip(), c["name"], need)); sys.exit(1)
        lav = int(next(l.split()[1] for l in open("/proc/meminfo") if l.startswith("MemAvailable"))) // 1024
        if lav < mem["laptop"]["need_mib"] + reg["laptop_reserve_mib"]:
            say("REFUSED: the laptop has %d MiB, cell %s needs %d + %d kept for the desktop" % (lav, c["name"], mem["laptop"]["need_mib"],
                                                                                         reg["laptop_reserve_mib"]))
            sys.exit(1)
    rc, asfound = SS.ssh("jetson", PC.STATE_CMD)
    asfound = asfound.strip()
    k = PC.parse_state(asfound)
    if not (k["pmode"] == "pmode:0000" and k["online"] == "0-3" and k["railgate"] == "1" and k["emcov"] == "0"):
        say("FATAL: the Jetson is not as found: %s" % asfound); sys.exit(1)
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    say("prechecks passed: boards idle, at the pin; model sha256 OK; memory fits; the laptop %s" % laptop_state())
    stored = "~/heteroserve/run/jetson_clocks_lc_%s.stored" % stamp
    started_guard, tpoll, stored_ok, lap = False, None, False, None
    logs = []
    try:
        rc, n = SS.ssh("jetson", SS.GUARD_PROCS + " | wc -l")
        if n.strip() == "0":
            SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
            started_guard = True
            time.sleep(5)
        say("guard %s" % ("started" if started_guard else "already running"))
        rc, o = SS.ssh("jetson", "mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store %s && echo STORED" % stored)
        if "STORED" not in o:
            say("FATAL: jetson_clocks --store failed: %r" % o); sys.exit(1)
        stored_ok = True
        ok, state, said = PC.set_mode("locked", stored, asfound)
        if not ok:
            say("FATAL: the Jetson's clocks did not lock: %s" % state); sys.exit(1)
        subprocess.run(["scp", "-q", TPOLL, "pi4:throttle_poll.sh"], check=True)
        rc, p = SS.ssh("pi4", "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_lc_%s.log > /dev/null 2>&1 "
                              "< /dev/null & echo $!" % stamp)
        tpoll = p.strip()
        stop = False
        for c in cells:
            jet = JETSON_IP in c["args"]
            for i in range(1, c["runs"] + 1):
                tag = "%s_%d" % (c["name"], i)
                if jet:
                    SS.ssh("jetson", "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H %s -p %d -d Vulkan0 "
                                     "> ~/logs/rpc-server_lc_%s_%s.log 2>&1 < /dev/null &" % (JETSON_IP, RPC_PORT, stamp, tag))
                lap = subprocess.Popen([LAPTOP_RPC] + LAPTOP_ARGS, stdout=open(os.path.join(d, "rpc_laptop_%s.log" % tag), "w"),
                                       stderr=subprocess.STDOUT, start_new_session=True)
                for _ in range(60):
                    rc, o = SS.ssh("jetson", "ss -ltnH 'sport = :%d' | wc -l" % RPC_PORT) if jet else (0, "1")
                    lok = (" %s:%d " % (LAPTOP_IP, LAPTOP_PORT)) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
                    if o.strip() not in ("", "0") and lok:
                        break
                    time.sleep(1)
                else:
                    say("FATAL: a worker never listened (%s)" % tag); sys.exit(1)
                say("%s: workers up (%slaptop pid %d); the laptop %s" % (tag, "Jetson, " if jet else "", lap.pid, laptop_state()))
                log = "~/logs/lc_%s_%s.log" % (stamp, tag)
                cmd = ("~/llama.cpp/build/bin/llama-bench %s %s 2>&1 | tee %s > /dev/null; echo EXIT_BENCH=${PIPESTATUS[0]}"
                       % (c["args"], c["protocol"], log))
                t0 = time.time()
                rc, o = SS.ssh("pi4", cmd, timeout=7200)
                subprocess.run(["scp", "-q", "pi4:%s" % log.replace("~/", ""), os.path.join(d, "bench_%s.log" % tag)])
                logs.append({"cell": c["name"], "log": "bench_%s.log" % tag})
                rows = [json.loads(l) for l in open(os.path.join(d, "bench_%s.log" % tag), errors="replace") if l.startswith("{")]
                say("%s: %s in %.0f s; %s; the laptop %s" % (tag, o.strip().splitlines()[-1] if o.strip() else "no exit line", time.time() - t0,
                    ", ".join("d%d %.1f ms/token" % (r["n_depth"], 1000.0 / r["avg_ts"]) for r in rows), laptop_state()))
                for p in jetson_rpc_pids():
                    SS.ssh("jetson", "kill %s" % p)
                os.killpg(lap.pid, signal.SIGTERM); lap.wait(timeout=60); lap = None
                if "EXIT_BENCH=0" not in o:
                    say("!!!!! %s did not exit 0; stopping" % tag); stop = True; break
            if stop:
                break
    finally:
        def safe(what, fn):
            try:
                fn()
            except Exception as e:                          # one failed step must not skip the others
                say("!!!!! cleanup step %s failed: %s" % (what, e))

        def stop_workers():
            for p in jetson_rpc_pids():
                SS.ssh("jetson", "kill %s" % p)
            if lap:
                os.killpg(lap.pid, signal.SIGTERM)

        def restore():
            ok, final, said = PC.set_mode("unlocked", stored, asfound)
            say("jetson restored: %s -- %s" % (final, "EQUAL to as found" if final == asfound else "!!!!! NOT EQUAL to as found " + asfound))
            open(os.path.join(d, "restored.json"), "w").write(json.dumps({"restored": final, "asfound": asfound, "equal": final == asfound}))

        def stop_tpoll():
            SS.ssh("pi4", "kill %s" % tpoll)
            subprocess.run(["scp", "-q", "pi4:logs/throttle_lc_%s.log" % stamp, os.path.join(d, "throttle_pi4.log")])

        def guard_tail():
            rc, o = SS.ssh("jetson", "tail -n %d ~/heteroserve/logs/power-guard.log" % (int(time.time() - t_start) + 120))
            open(os.path.join(d, "power_guard_tail.log"), "w").write(o)
            say("guard: %d TRIP lines in the run's window" % sum(1 for l in o.splitlines() if "TRIP" in l))

        def stop_guard():
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        safe("workers", stop_workers)
        if stored_ok:
            safe("restore", restore)
        if tpoll:
            safe("throttle poll", stop_tpoll)
        safe("guard log", guard_tail)
        if started_guard:
            safe("guard", stop_guard)
        json.dump({"logs": logs}, open(os.path.join(d, "runs.json"), "w"), indent=1)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


# ------------------------------------------------------------------ score
def score_rows(reg, per_run):
    """per_run = [{"0": ms, "4096": ms}, ...]. Pure, for the tests."""
    D = str(reg["plan_depth"])
    ok = [r for r in per_run if r.get("0") and r.get(D)]
    if len(ok) != reg["runs"]:
        v = {"verdict": "INCOMPLETE"}
        return {"hypotheses": {k: v for k in ("LC-0", "LC-%s" % D, "LC-SLOPE", "LC-GAIN")}, "runs": len(ok)}
    m0 = sum(r["0"] for r in ok) / len(ok)
    mD = sum(r[D] for r in ok) / len(ok)
    p0, pD = reg["predictions_ms"]["0"], reg["predictions_ms"][D]
    e = lambda m, p: 100 * (m / p - 1)
    hyp = {}
    for key, m, p, tol in (("LC-0", m0, p0, reg["tolerance_pct"]), ("LC-%s" % D, mD, pD, reg["tolerance_pct"]),
                           ("LC-SLOPE", mD - m0, pD - p0, reg["tolerance_slope_pct"])):
        hyp[key] = {"verdict": "HOLDS" if abs(e(m, p)) <= tol else "FAILS", "measured_ms": round(m, 2), "predicted_ms": round(p, 2),
                    "err_pct": round(e(m, p), 2)}
    g = reg["served_plan_measured_d0_ms_kv14b"] / m0
    hyp["LC-GAIN"] = {"verdict": "HOLDS" if abs(100 * (g / reg["predicted_gain_d0"] - 1)) <= reg["tolerance_pct"] else "FAILS",
                      "measured_gain": round(g, 4), "predicted_gain": reg["predicted_gain_d0"],
                      "err_pct": round(100 * (g / reg["predicted_gain_d0"] - 1), 2)}
    spread = lambda xs: round(100 * (max(xs) - min(xs)) / (sum(xs) / len(xs)), 2)
    return {"hypotheses": hyp, "runs": len(ok), "mean_ms": {"0": round(m0, 2), D: round(mD, 2)},
            "spread_pct": {"0": spread([r["0"] for r in ok]), D: spread([r[D] for r in ok])},
            "served_plan_predicted_ms_at_depth": reg["served_plan_predicted_ms"][D],
            "against_the_served_plan_at_depth": round(reg["served_plan_predicted_ms"][D] / mD, 3),
            "without_the_term_err_pct_at_depth": round(e(mD, reg["predictions_without_the_term_ms"][D]), 2)}


def score_forced(reg, pick_runs, forced_runs):
    """Registration b's second cell: the planner's pick against the plan that keeps the Jetson (LC-JET, LC-RANK). Pure."""
    c = next(x for x in reg["cells"] if x["name"] == "jetson_forced")
    ok = [r for r in forced_runs if r.get("0")]
    pk = [r for r in pick_runs if r.get("0")]
    if len(ok) != c["runs"] or len(pk) != reg["runs"]:
        return {"hypotheses_forced": {"LC-JET": {"verdict": "INCOMPLETE"}, "LC-RANK": {"verdict": "INCOMPLETE"}}}
    mf = sum(r["0"] for r in ok) / len(ok)
    mp = sum(r["0"] for r in pk) / len(pk)
    pf = c["prediction_ms"]["0"]
    e = 100 * (mf / pf - 1)
    return {"hypotheses_forced": {
        "LC-JET": {"verdict": "HOLDS" if abs(e) <= reg["tolerance_pct"] else "FAILS", "measured_ms": round(mf, 2),
                   "predicted_ms": round(pf, 2), "err_pct": round(e, 2)},
        "LC-RANK": {"verdict": "HOLDS" if mp < mf else "FAILS", "pick_ms": round(mp, 2), "forced_ms": round(mf, 2),
                    "predicted_ratio": round(pf / reg["predictions_ms"]["0"], 4), "measured_ratio": round(mf / mp, 4)}}}


def score(d, reg_path=None):
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "longctx_predictions_20261011.json")
    reg = json.load(open(reg_path))
    logs = json.load(open(os.path.join(d, "runs.json")))["logs"]
    by_cell = {}
    for f in logs:
        cell, f = (f["cell"], f["log"]) if isinstance(f, dict) else ("pick", f)
        rows = [json.loads(l) for l in open(os.path.join(d, f), errors="replace") if l.startswith("{")]
        by_cell.setdefault(cell, []).append({str(r["n_depth"]): 1000.0 / r["avg_ts"] for r in rows if r.get("n_gen") and not r.get("n_prompt")})
    per_run = by_cell.get("pick", [])
    th = None
    tp = os.path.join(d, "throttle_pi4.log")
    if os.path.exists(tp):
        bits = [int(m.group(1), 16) for m in re.finditer(r"throttled=(0x[0-9a-f]+)", open(tp).read())]
        th = {"polls": len(bits), "soft_limit_s": sum(1 for b in bits if b & 0x8), "capped_s": sum(1 for b in bits if b & 0x2)}
    s = score_rows(reg, per_run)
    if reg.get("cells") and len(reg["cells"]) > 1:
        s.update(score_forced(reg, per_run, by_cell.get("jetson_forced", [])))
    rp = os.path.join(d, "restored.json")
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": SS.sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": sorted(p for p, h in reg.get("inputs_sha256", {}).items()
                                                   if os.path.exists(os.path.join(ROOT, p)) and SS.sha(os.path.join(ROOT, p)) != h),
         "per_run_ms": {c: [{k: round(v, 2) for k, v in r.items()} for r in rs] for c, rs in by_cell.items()}, **s,
         "pi4_throttle": th,
         "jetson_restored_equal": json.load(open(rp))["equal"] if os.path.exists(rp) else None}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "score"):
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1], "--smoke" in a)
    else:
        score(a[1], a[a.index("--reg") + 1] if "--reg" in a else None)
