#!/usr/bin/env python3
"""Scheduler upgrade 4, the KV term: decode cost against the context depth, exactly as registered
(register_kv_depth_20261010.py).

    python3 testbed/scripts/kv_depth_20261010.py run testbed/dataset/kv_depth_predictions_20261010.json
    python3 testbed/scripts/kv_depth_20261010.py score testbed/results/kv_depth_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit:
- prechecks: no serve_pool.sh state; neither board busy; both llama.cpp trees at the pin and clean; the model's sha256
  on the Pi 4. On the Jetson the model is copied from the laptop's pinned copy if absent, and checked there too;
- the Jetson's guard started by its NOTE's procedure and stopped last. Its clocks are locked as every run_pool.sh cell
  locks them (stored, locked, restored, checked equal to the snapshot). The Pi 4 is polled at 1 Hz (throttle_poll.sh);
- three invocations, each running llama-bench on both boards at once (each board runs alone; they share nothing), the
  registered arguments plus -o jsonl. The raw output is tee'd to a log on the board and copied back whole.
score: per board and invocation, ms per decoded token = 1000 / avg_ts at each depth. The slope is the least squares
against the depth, in ms per 1,000 tokens, averaged over the invocations and with its spread. HK-PI4 and HK-JET are
scored as written. Writes <dir>/score.json.
"""
import datetime, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, sha, GUARD_PROCS
import power_modes_char_20261010 as PC                      # noqa: E402  STATE_CMD, parse_state, set_mode

TPOLL = os.path.join(HERE, "throttle_poll.sh")
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
PIN = "b23701f77d47dad9de834d59ebfcbe25c9e8b46f"
MODEL_LOCAL = os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf")
BUSY = 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true'
TREE = "git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l"
EXTRA = {"pi4": "-t 4", "jetson": "-ngl 99"}


def run(reg_path):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "kv_depth_%s" % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")
    t_start = time.time()

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    say("registration %s (%s)" % (os.path.relpath(reg_path, ROOT), SS.sha(reg_path)[:16]))
    if os.path.exists(STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    mf, want = reg["model"]["file"], reg["model"]["sha256"]
    for n in ("pi4", "jetson"):
        rc, b = SS.ssh(n, BUSY)
        if b.strip():
            say("FATAL: %s is busy: %s" % (n, b.strip())); sys.exit(1)
        rc, t = SS.ssh(n, TREE)
        if t.split() != [PIN, "0"]:
            say("FATAL: %s's llama.cpp is %r, not the pin clean" % (n, t)); sys.exit(1)
        rc, o = SS.ssh(n, "test -f %s && sha256sum %s | cut -d' ' -f1 || echo ABSENT" % (mf, mf), timeout=300)
        if o.strip() == "ABSENT" and n == "jetson":
            if SS.sha(MODEL_LOCAL) != want:
                say("FATAL: the laptop's copy is not the registered model"); sys.exit(1)
            subprocess.run(["scp", "-q", MODEL_LOCAL, "jetson:models/qwen3-0.6b-q8_0.gguf"], check=True)
            rc, o = SS.ssh(n, "sha256sum %s | cut -d' ' -f1" % mf, timeout=300)
            say("model copied to the Jetson from the laptop's pinned copy")
        if o.strip() != want:
            say("FATAL: %s on %s is %s, registered %s" % (mf, n, o.strip()[:16], want[:16])); sys.exit(1)
        say("%s: idle, at the pin, model sha256 OK" % n)
    rc, asfound = SS.ssh("jetson", PC.STATE_CMD)
    asfound = asfound.strip()
    k = PC.parse_state(asfound)
    if not (k["pmode"] == "pmode:0000" and k["online"] == "0-3" and k["railgate"] == "1" and k["emcov"] == "0"):
        say("FATAL: the Jetson is not as found (MAXN, governors free): %s" % asfound); sys.exit(1)
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    stored = "~/heteroserve/run/jetson_clocks_kvd_%s.stored" % stamp
    started_guard, tpoll, stored_ok = False, None, False
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
        say("jetson clocks locked (%s)" % state)
        subprocess.run(["scp", "-q", TPOLL, "pi4:throttle_poll.sh"], check=True)
        rc, p = SS.ssh("pi4", "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_kvd_%s.log > /dev/null 2>&1 "
                              "< /dev/null & echo $!" % stamp)
        tpoll = p.strip()
        for inv in range(1, reg["bench"]["invocations"] + 1):
            procs = {}
            for n in ("pi4", "jetson"):
                log = "~/logs/kvd_%s_%s_%d.log" % (n, stamp, inv)
                cmd = ("mkdir -p ~/logs; ~/llama.cpp/build/bin/llama-bench -m %s %s %s -o jsonl 2>&1 | tee %s > /dev/null; "
                       "echo EXIT ${PIPESTATUS[0]}" % (mf, reg["bench"]["args"], EXTRA[n], log))
                procs[n] = (subprocess.Popen(SS.SSH + [n, cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True), log)
            say("invocation %d: llama-bench running on both boards" % inv)
            for n, (p, log) in procs.items():
                out, _ = p.communicate(timeout=3600)
                subprocess.run(["scp", "-q", "%s:%s" % (n, log.replace("~/", "")), os.path.join(d, "kvd_%s_%d.log" % (n, inv))])
                rows = [json.loads(l) for l in open(os.path.join(d, "kvd_%s_%d.log" % (n, inv)), errors="replace") if l.startswith("{")]
                say("invocation %d %-6s %s; %s" % (inv, n, out.strip().splitlines()[-1] if out.strip() else "no output",
                                                     ", ".join("d%d %.2f ms/token" % (r["n_depth"], 1000.0 / r["avg_ts"]) for r in rows)))
    finally:
        def safe(what, fn):
            try:
                fn()
            except Exception as e:                          # one failed step must not skip the others
                say("!!!!! cleanup step %s failed: %s" % (what, e))

        def restore():
            ok, final, said = PC.set_mode("unlocked", stored, asfound)
            say("jetson restored: %s -- %s" % (final, "EQUAL to as found" if final == asfound else "!!!!! NOT EQUAL to as found " + asfound))
            open(os.path.join(d, "restored.json"), "w").write(json.dumps({"restored": final, "asfound": asfound, "equal": final == asfound}))

        def stop_tpoll():
            SS.ssh("pi4", "kill %s" % tpoll)
            subprocess.run(["scp", "-q", "pi4:logs/throttle_kvd_%s.log" % stamp, os.path.join(d, "throttle_pi4.log")])

        def guard_tail():
            rc, o = SS.ssh("jetson", "tail -n %d ~/heteroserve/logs/power-guard.log" % (int(time.time() - t_start) + 120))
            open(os.path.join(d, "power_guard_tail.log"), "w").write(o)
            say("guard: %d TRIP lines in the run's window" % sum(1 for l in o.splitlines() if "TRIP" in l))

        def stop_guard():
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        if stored_ok:
            safe("restore", restore)
        if tpoll:
            safe("throttle poll", stop_tpoll)
        safe("guard log", guard_tail)
        if started_guard:
            safe("guard", stop_guard)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


# ------------------------------------------------------------------ score
def slope(points):
    """Least squares of ms per token on depth over [(depth, ms)], in ms per 1,000 tokens; and the intercept."""
    n = len(points)
    mx = sum(p[0] for p in points) / n; my = sum(p[1] for p in points) / n
    s = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return 1000 * s, my - s * mx


def score_rows(reg, runs):
    """runs = {board: [[(depth, ms per token), ...] per invocation]}. Pure, for the tests."""
    hyp, per = {}, {}
    for b, key in (("pi4", "HK-PI4"), ("jetson", "HK-JET")):
        invs = [sorted(r) for r in runs.get(b, []) if sorted(x[0] for x in r) == reg["bench"]["depths"]]
        if len(invs) != reg["bench"]["invocations"]:
            hyp[key] = {"verdict": "INCOMPLETE", "invocations": len(invs)}
            continue
        fits = [slope(r) for r in invs]
        s = [f[0] for f in fits]
        m = sum(s) / len(s)
        pred = reg["predictions"][b]["slope_ms_per_1k_tokens"]
        err = 100 * (m / pred - 1)
        per[b] = {"slope_ms_per_1k_each": [round(x, 3) for x in s], "slope_ms_per_1k_mean": round(m, 3),
                  "spread_pct": round(100 * (max(s) - min(s)) / m, 2), "intercept_ms_each": [round(f[1], 2) for f in fits],
                  "predicted_ms_per_1k": round(pred, 3), "err_pct": round(err, 2),
                  "ms_by_depth_mean": {str(dd): round(sum(r[i][1] for r in invs) / len(invs), 3) for i, dd in enumerate(reg["bench"]["depths"])}}
        hyp[key] = {"verdict": "HOLDS" if abs(err) <= reg["tolerance_pct"] else "FAILS", "measured_ms_per_1k": round(m, 3),
                    "predicted_ms_per_1k": round(pred, 3), "err_pct": round(err, 2)}
    v = [h["verdict"] for h in hyp.values()]
    return {"boards": per, "hypotheses": hyp,
            "kv_term": "INCOMPLETE" if "INCOMPLETE" in v else ("HOLDS" if all(x == "HOLDS" for x in v) else "FAILS")}


def score(d, reg_path=None):
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "kv_depth_predictions_20261010.json")
    reg = json.load(open(reg_path))
    runs, fa = {}, {}
    for b in ("pi4", "jetson"):
        for inv in range(1, reg["bench"]["invocations"] + 1):
            p = os.path.join(d, "kvd_%s_%d.log" % (b, inv))
            if not os.path.exists(p):
                continue
            rows = [json.loads(l) for l in open(p, errors="replace") if l.startswith("{")]
            rows = [r for r in rows if r.get("n_gen") and not r.get("n_prompt")]
            runs.setdefault(b, []).append([(r["n_depth"], 1000.0 / r["avg_ts"]) for r in rows])
            fa.setdefault(b, sorted({str(r.get("flash_attn")) for r in rows}))
    s = score_rows(reg, runs)
    rp = os.path.join(d, "restored.json")
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": SS.sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": sorted(p for p, h in reg.get("inputs_sha256", {}).items()
                                                   if os.path.exists(os.path.join(ROOT, p)) and SS.sha(os.path.join(ROOT, p)) != h),
         **s, "flash_attn": fa, "jetson_restored_equal": json.load(open(rp))["equal"] if os.path.exists(rp) else None,
         "served_14b_predicted": reg["served_14b"]}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps({k: s[k] for k in ("boards", "hypotheses", "kv_term", "flash_attn", "jetson_restored_equal",
                                        "code_changed_since_registration")}, indent=1))
    print("wrote %s" % os.path.relpath(os.path.join(d, "score.json"), ROOT))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "score"):
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1])
    else:
        score(a[1], a[a.index("--reg") + 1] if "--reg" in a else None)
