#!/usr/bin/env python3
"""The KV term's discriminating test on the served Qwen3-14B, exactly as registered (register_kv14_20261010.py).

    python3 testbed/scripts/kv14_20261010.py run testbed/dataset/kv14_predictions_20261010.json
    python3 testbed/scripts/kv14_20261010.py score testbed/results/kv14_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit (spec_pass_campaign_20261010.sh's pattern):
- prechecks: no serve_pool.sh state; the link unshaped; the guard's precheck. The guard is started by its NOTE's procedure
  and stopped last;
- the registered cell through run_pool.sh, three times in a row. run_pool.sh carries every gate: pin, memory, MAXN, the
  guard, the clocks locked and restored, the throttle poll and the memory probe. Each run's log is copied into the run
  directory.
score: per run, ms per decoded token = 1000 / avg_ts at d = 0 and at d = 1,024 (llama-bench's -r 3 mean). The slope is the
difference per 1,000 tokens of depth, averaged over the runs, with its spread. HK14-A and HK14-B are scored as written;
each run's Pi 4 thermal verdict and EXIT_BENCH are reported. Writes <dir>/score.json.
"""
import datetime, glob, json, os, re, shutil, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import spec_serve_20261010 as SS                            # noqa: E402  ssh, sha, GUARD_PROCS

STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")


def run(reg_path):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "kv14_%s" % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    say("registration %s (%s)" % (os.path.relpath(reg_path, ROOT), SS.sha(reg_path)[:16]))
    if os.path.exists(STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if subprocess.run(["bash", os.path.join(HERE, "netshape.sh"), "check"], capture_output=True).returncode:
        say("FATAL: the Pi 4 is shaped, or did not answer"); sys.exit(1)
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    started = False
    logs = []
    try:
        rc, n = SS.ssh("jetson", SS.GUARD_PROCS + " | wc -l")
        if n.strip() == "0":
            SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
            started = True
            time.sleep(5)
        say("guard %s" % ("started" if started else "already running"))
        cell = reg["cell"]
        for i in range(1, reg["runs"] + 1):
            mark = time.time()
            env = dict(os.environ, CELL=cell, PRED=os.path.relpath(reg_path, ROOT))
            with open(os.path.join(d, "%s.run%d.console" % (cell, i)), "w") as con:
                p = subprocess.run(["bash", os.path.join(HERE, "run_pool.sh")], env=env, cwd=ROOT, stdout=con, stderr=subprocess.STDOUT,
                                   timeout=7200)
            found = [f for f in glob.glob(os.path.join(ROOT, "testbed", "results", "pool_%s_2*.log" % cell)) if os.path.getmtime(f) >= mark]
            if not found:
                say("!!!!! run %d wrote no log (exit %d)" % (i, p.returncode)); break
            log = sorted(found)[-1]
            shutil.copy(log, d)
            logs.append(os.path.basename(log))
            text = open(log, errors="replace").read()
            rows = [json.loads(l) for l in text.splitlines() if l.startswith("{")]
            th = re.findall(r"^THROTTLE pi4 .* verdict=(\S+)", text, re.M)
            say("run %d: exit %d, %s; Pi 4 %s; %s" % (i, p.returncode, os.path.basename(log), th[-1] if th else "no THROTTLE line",
                ", ".join("d%d %.1f ms/token" % (r["n_depth"], 1000.0 / r["avg_ts"]) for r in rows if r.get("n_gen"))))
            if p.returncode:
                break
    finally:
        if started:
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        json.dump({"logs": logs}, open(os.path.join(d, "runs.json"), "w"), indent=1)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


def score_rows(reg, per_run):
    """per_run = [{"d0": ms, "d1024": ms}, ...]. Pure, for the tests."""
    ok = [r for r in per_run if r.get("d0") and r.get("d1024")]
    hyp = {}
    if len(ok) != reg["runs"]:
        return {"hypotheses": {"HK14-A": {"verdict": "INCOMPLETE"}, "HK14-B": {"verdict": "INCOMPLETE"}}, "runs": len(ok)}
    s = [(r["d1024"] - r["d0"]) / 1.024 for r in ok]
    m = sum(s) / len(s)
    for k, key in (("HK14-A", "A_ms_per_1k"), ("HK14-B", "B_ms_per_1k")):
        e = 100 * (m / reg["predictions"][key] - 1)
        hyp[k] = {"verdict": "HOLDS" if abs(e) <= reg["tolerance_pct"] else "FAILS", "measured_ms_per_1k": round(m, 2),
                  "predicted_ms_per_1k": round(reg["predictions"][key], 2), "err_pct": round(e, 2)}
    return {"hypotheses": hyp, "slope_ms_per_1k_each": [round(x, 2) for x in s], "slope_ms_per_1k_mean": round(m, 2),
            "spread_pct": round(100 * (max(s) - min(s)) / m, 2), "runs": len(ok),
            "against_the_registered_weight_cost_term": round(m / reg["predictions"]["registered_weight_cost_term_ms_per_1k"], 2)}


def score(d, reg_path=None):
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "kv14_predictions_20261010.json")
    reg = json.load(open(reg_path))
    logs = json.load(open(os.path.join(d, "runs.json")))["logs"]
    per_run, th, ex = [], [], []
    for f in logs:
        text = open(os.path.join(d, f), errors="replace").read()
        rows = [json.loads(l) for l in text.splitlines() if l.startswith("{")]
        by = {r["n_depth"]: 1000.0 / r["avg_ts"] for r in rows if r.get("n_gen") and not r.get("n_prompt")}
        per_run.append({"d0": by.get(0), "d1024": by.get(1024)})
        t = re.findall(r"^THROTTLE pi4 .* verdict=(\S+)", text, re.M)
        th.append(t[-1] if t else None)
        e = re.findall(r"^EXIT_BENCH=(\d+)", text, re.M)
        ex.append(int(e[-1]) if e else None)
    s = score_rows(reg, per_run)
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": SS.sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": sorted(p for p, h in reg.get("inputs_sha256", {}).items()
                                                   if os.path.exists(os.path.join(ROOT, p)) and SS.sha(os.path.join(ROOT, p)) != h),
         "per_run_ms": per_run, "pi4_thermal": th, "exit_bench": ex, **s}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "score"):
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1])
    else:
        score(a[1], a[a.index("--reg") + 1] if "--reg" in a else None)
