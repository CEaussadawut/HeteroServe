#!/usr/bin/env python3
"""Scheduler upgrade 5, step 2: the Jetson's power modes SERVED, exactly as registered (register_power_modes_20261010.py).

    python3 testbed/scripts/power_modes_20261010.py run testbed/dataset/power_modes_predictions_20261010.json [--smoke]
    python3 testbed/scripts/power_modes_20261010.py score testbed/results/power_modes_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit:
- prechecks: no serve_pool.sh state, the link unshaped (netshape.sh check), the model's sha256 on the Pi 4, and the
  Jetson idle and as found (MAXN, governors free: power_modes_char_20261010.py's checks);
- the Jetson's guard started by its NOTE's procedure and stopped last. Its as-found clocks are stored
  (jetson_clocks --store). The strip is logged at 5 Hz, the Pi 4 polled at 1 Hz (throttle_poll.sh) and the Jetson's
  clocks at 10 Hz (jetson_clock_poll.sh), all for the whole session, with an idle window first and last;
- serve_pool.sh start with the registered plan, SERVE_DRAFT=none and CLOCKS=asfound (this runner owns the Jetson's
  clocks), then one warm-up request (never scored);
- each block of the registered order: the mode switched and verified (power_modes_char_20261010.set_mode), one settle
  request of settle_tokens (never scored), the registered prompts in order, and the guard's log checked for a trip;
- back to as found, the server's log copied back, serve_pool.sh stop. The Jetson is restored and checked equal to the
  snapshot on every way out.
--smoke: one block per mode, one prompt of 8 tokens, idle windows of 10 s, into power_modes_smoke_<stamp>/ (the
plumbing only: never scored).
score: per mode, over its requests:
- ms per token = the server's sum of predicted_ms / sum of predicted_n;
- gross J per decoded token at the Pi 4 + Jetson outlets (the strip trapezoid over each request's
  [t0 + prompt_ms, t1], summed, over the sum of predicted_n), and the Jetson's outlet alone;
- the Pi 4's thermal verdicts;
- the Jetson's INA3221 power (the guard's 1 Hz log) and its mean GPU and EMC clocks (the 10 Hz poll) over the same
  windows, reported only;
- whether the greedy text matches across modes.
The registered PM-T, PM-E and PM-PICK are scored as written; the reported extras beside them. Writes <dir>/score.json.
"""
import datetime, json, os, signal, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, post, sha, GUARD_PROCS
import power_modes_char_20261010 as PC                      # noqa: E402  STATE_CMD, parse_state, set_mode, mode_cmd

STRIP = os.path.join(HERE, "shelly_strip.py")
TPOLL = os.path.join(HERE, "throttle_poll.sh")
CPOLL = os.path.join(HERE, "jetson_clock_poll.sh")
SERVE = os.path.join(HERE, "serve_pool.sh")
NETSHAPE = os.path.join(HERE, "netshape.sh")
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
MODES = ("locked", "unlocked", "5w")
TRIPS = "awk '/guard start/ {t=0} /TRIP/ {t++} END {print t+0}' ~/heteroserve/logs/power-guard.log"


def run(reg_path, smoke=False):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("power_modes_smoke_%s" if smoke else "power_modes_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")
    t_start = time.time()

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    say("registration %s (%s)%s" % (os.path.relpath(reg_path, ROOT), SS.sha(reg_path)[:16], " SMOKE: never scored" if smoke else ""))
    order = list(MODES) if smoke else reg["order"]
    prompts = reg["prompts"][:1] if smoke else reg["prompts"]
    n_predict = 8 if smoke else reg["n_predict"]
    idle = (10, 10) if smoke else (reg["idle_before_s"], reg["idle_after_s"])
    # ---- prechecks, nothing changed yet
    if os.path.exists(STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if subprocess.run(["bash", NETSHAPE, "check"], capture_output=True).returncode:
        say("FATAL: the Pi 4 is shaped, or did not answer (netshape.sh check)"); sys.exit(1)
    rc, b = SS.ssh("jetson", 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%%s " $p; done; '
                             '%s | wc -l' % PC.POLL_PROCS)
    if b.split() != ["0"]:
        say("FATAL: the Jetson is busy or already polled: %r" % b); sys.exit(1)
    rc, asfound = SS.ssh("jetson", PC.STATE_CMD)
    asfound = asfound.strip()
    k = PC.parse_state(asfound)
    if not (k["pmode"] == "pmode:0000" and k["online"] == "0-3" and k["gpu"].split("-")[0] != k["gpu"].split("-")[1]
            and k["railgate"] == "1" and k["emcov"] == "0" and k["cpuidle"] == "00"):
        say("FATAL: the Jetson is not as found (MAXN, governors free): %s" % asfound); sys.exit(1)
    say("jetson as found: %s" % asfound)
    for path, want in reg["models_on_host"].items():
        rc, out = SS.ssh("pi4", "sha256sum %s | cut -d' ' -f1" % path, timeout=900)
        if out.strip() != want:
            say("FATAL: %s on the Pi 4 is %s, registered %s" % (path, out.strip()[:16], want[:16])); sys.exit(1)
        say("model %s sha256 OK on the Pi 4" % path)
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    out = open(os.path.join(d, "run.jsonl"), "a")
    rec = lambda o: (out.write(json.dumps(o, ensure_ascii=False) + "\n"), out.flush())
    stored = "~/heteroserve/run/jetson_clocks_pmserve_%s.stored" % stamp
    started_guard, logger, tpoll, cpoll_up, stored_ok = False, None, None, False, False
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
        say("as-found clocks stored: %s" % stored)
        logger = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(d, "strip.csv"), "--hz", "5"],
                                  stdout=open(os.path.join(d, "strip_logger.stdout"), "w"), stderr=subprocess.STDOUT,
                                  start_new_session=True)
        subprocess.run(["scp", "-q", TPOLL, "pi4:throttle_poll.sh"], check=True)
        rc, p = SS.ssh("pi4", "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_pm_%s.log > /dev/null 2>&1 "
                              "< /dev/null & echo $!" % stamp)
        tpoll = p.strip()
        subprocess.run(["scp", "-q", CPOLL, "jetson:heteroserve/tools/jetson_clock_poll.sh"], check=True)
        SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_clock_poll.sh ~/heteroserve/logs/clockpoll_%s.log "
                         "</dev/null >/dev/null 2>&1 & echo started" % stamp)
        cpoll_up = True
        say("strip logger pid %d, Pi 4 throttle poll pid %s, Jetson clock poll up; idle %d s" % (logger.pid, tpoll, idle[0]))
        t_i0 = time.time(); time.sleep(idle[0]); t_i1 = time.time()
        rec({"idle_before": [t_i0, t_i1], "state": asfound})
        env = dict(os.environ, MODEL=reg["model"], CTX=str(reg["ctx"]), PLACE=reg["place"], PLACE_HEAD=reg["place_head"],
                   NODES=reg["nodes"], **reg["serve_env"])
        r = subprocess.run(["bash", SERVE, "start"], env=env, capture_output=True, text=True, timeout=2400)
        open(os.path.join(d, "serve.console"), "w").write(r.stdout + r.stderr)
        if r.returncode or "READY" not in r.stdout:
            say("FATAL: serve_pool.sh start failed (exit %d); see serve.console" % r.returncode); sys.exit(1)
        say("ready: %s" % " | ".join(l for l in r.stdout.splitlines() if l.startswith(("split", "mem", "draft", "jetson  clocks"))))
        SS.post(reg["host_url"] + "/completion", dict(reg["sampling"], prompt=reg["chat_format"] % reg["warmup_prompt"],
                                                       n_predict=16), timeout=3600)
        say("warm-up done")
        for bi, mode in enumerate(order):
            ok, state, said = PC.set_mode(mode, stored, asfound)
            rec({"block": bi, "mode": mode, "switch": state, "said": said, "ok": ok, "t": time.time()})
            if not ok:
                say("FATAL: block %d: the Jetson is not in %s after the switch: %s" % (bi, mode, state)); sys.exit(1)
            say("===== block %d: %s (%s)" % (bi, mode, state))
            SS.post(reg["host_url"] + "/completion", dict(reg["sampling"], prompt=reg["chat_format"] % reg["warmup_prompt"],
                                                           n_predict=reg["settle_tokens"]), timeout=3600)
            for pr in prompts:
                body = dict(reg["sampling"], prompt=reg["chat_format"] % pr["text"], n_predict=n_predict)
                t0 = time.time()
                try:
                    resp = SS.post(reg["host_url"] + "/completion", body, timeout=3600)
                except Exception as e:
                    say("!!!!! block %d %s %s failed: %s" % (bi, mode, pr["id"], e)); continue
                t1 = time.time()
                tm = resp.get("timings") or {}
                rec({"block": bi, "mode": mode, "id": pr["id"], "t0": t0, "t1": t1, "tokens_predicted": resp.get("tokens_predicted"),
                     "timings": tm, "content": resp.get("content")})
                say("%-8s %-5s %3s tokens, %6.1f ms/token, prompt %.1f s" % (
                    mode, pr["id"], resp.get("tokens_predicted"), (tm.get("predicted_ms") or 0) / max(1, tm.get("predicted_n") or 1),
                    (tm.get("prompt_ms") or 0) / 1000))
            rc, t = SS.ssh("jetson", TRIPS)
            if t.strip() != "0":
                say("FATAL: the guard tripped (%s since its start) during block %d" % (t.strip(), bi)); sys.exit(1)
        ok, state, said = PC.set_mode("unlocked", stored, asfound)
        say("back to as found: %s (%s)" % (ok, state))
        rc, name = SS.ssh("pi4", "ls -t ~/logs/llama-server_serve_*.log | head -1")
        subprocess.run(["scp", "-q", "pi4:%s" % name.strip(), os.path.join(d, "server.log")])
        r = subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600)
        open(os.path.join(d, "serve.console"), "a").write(r.stdout + r.stderr)
        say("stopped: %s" % " | ".join(l for l in r.stdout.splitlines() if "stop" in l))
        t_i0 = time.time(); time.sleep(idle[1]); t_i1 = time.time()
        rec({"idle_after": [t_i0, t_i1]})
    finally:
        def safe(what, fn):
            try:
                fn()
            except Exception as e:                          # one failed step must not skip the others
                say("!!!!! cleanup step %s failed: %s" % (what, e))

        def stop_serve():
            subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600); say("serve_pool.sh stopped (cleanup)")

        def restore():
            ok, final, said = PC.set_mode("unlocked", stored, asfound)
            say("jetson restored: %s -- %s" % (final, "EQUAL to as found" if final == asfound else "!!!!! NOT EQUAL to as found " + asfound))
            rec({"restored": final, "asfound": asfound, "equal": final == asfound})

        def stop_tpoll():
            SS.ssh("pi4", "kill %s" % tpoll)
            subprocess.run(["scp", "-q", "pi4:logs/throttle_pm_%s.log" % stamp, os.path.join(d, "throttle_pi4.log")])

        def stop_cpoll():
            SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p; done' % PC.POLL_PROCS)
            subprocess.run(["scp", "-q", "jetson:heteroserve/logs/clockpoll_%s.log" % stamp, os.path.join(d, "clockpoll.log")])

        def stop_logger():
            os.killpg(logger.pid, signal.SIGTERM); logger.wait(timeout=30)

        def guard_tail():
            rc, o = SS.ssh("jetson", "tail -n %d ~/heteroserve/logs/power-guard.log" % (int(time.time() - t_start) + 120))
            open(os.path.join(d, "power_guard_tail.log"), "w").write(o)
            say("guard: %d TRIP lines in the run's window" % sum(1 for l in o.splitlines() if "TRIP" in l))

        def stop_guard():
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        if os.path.exists(STATE):
            safe("serve_pool.sh", stop_serve)
        if stored_ok:
            safe("restore", restore)
        if tpoll:
            safe("throttle poll", stop_tpoll)
        if cpoll_up:
            safe("clock poll", stop_cpoll)
        if logger:
            safe("strip logger", stop_logger)
        safe("guard log", guard_tail)
        if started_guard:
            safe("guard", stop_guard)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


# ------------------------------------------------------------------ score
def measure(d, reg, rows, idle):
    """Per mode: ms per token, gross J per decoded token (pool and Jetson outlet), thermal verdicts, requests. The energy
    windows need the run's strip log, so the tests feed score_rows() directly instead."""
    import full_eval as F
    poll_p = os.path.join(d, "throttle_pi4.log")
    poll = open(poll_p, errors="replace").readlines() if os.path.exists(poll_p) else None
    o = reg["outlets"]
    pool = [o["pi4"], o["jetson"]]
    day = datetime.datetime.fromtimestamp(idle[0]).astimezone().strftime("%Y-%m-%d")
    ina, clk = PC.guard_rows(d, day), PC.poll_rows(d)
    out = {}
    for m in MODES:
        rs = [r for r in rows if r["mode"] == m]
        ms = sum(r["timings"]["predicted_ms"] for r in rs)
        n = sum(r["timings"]["predicted_n"] for r in rs)
        j = jj = 0.0
        th, wins = [], []
        for r in rs:
            a = r["t0"] + r["timings"]["prompt_ms"] / 1000.0
            w = F.window_energy(d, a, r["t1"], idle, pool)
            j += sum(w[x][0] for x in pool); jj += w[o["jetson"]][0]
            wins.append((a, r["t1"]))
            if poll:
                th.append(F.thermal(poll, r["t0"], r["t1"])["verdict"])
        inside = lambda rows_: [x for x in rows_ if any(a <= x[0] <= b for a, b in wins)]
        iw, cw = inside(ina), inside(clk)
        dec_s = sum(b - a for a, b in wins)
        out[m] = {"requests": len(rs), "tokens": n, "T_ms": ms / n if n else None, "E_j": j / n if n else None,
                  "Ej_jetson_j": jj / n if n else None, "thermal": sorted(set(th)),
                  "jetson_ina_w": statistics.fmean(p for _, p in iw) / 1000 if iw else None,
                  "Ej_jetson_ina_j": statistics.fmean(p for _, p in iw) / 1000 * dec_s / n if iw and n else None,
                  "gpu_mhz_mean": statistics.fmean(x[1] for x in cw) / 1e6 if cw else None,
                  "emc_mhz_mean": statistics.fmean(x[4] for x in cw) / 1e6 if cw else None,
                  "by_block": {str(b): round(sum(r["timings"]["predicted_ms"] for r in rs if r["block"] == b) /
                                             sum(r["timings"]["predicted_n"] for r in rs if r["block"] == b), 1)
                               for b in sorted({r["block"] for r in rs})}}
    return out


def score_rows(reg, meas):
    """The registered verdicts from per-mode measurements ({mode: {"requests", "T_ms", "E_j", ...}}). Pure."""
    want = reg["order"].count("locked") * len(reg["prompts"])
    pred, tol = reg["predictions"], reg["tolerance"]
    hyp = {}
    for m in MODES:
        x, p = meas.get(m) or {}, pred[m]
        complete = x.get("requests") == reg["order"].count(m) * len(reg["prompts"])
        for key, k_meas, k_pred in (("PM-T", "T_ms", "T_ms"), ("PM-E", "E_j", "E_j")):
            if not complete or x.get(k_meas) is None:
                hyp["%s-%s" % (key, m)] = {"verdict": "INCOMPLETE"}
                continue
            err = x[k_meas] / p[k_pred] - 1
            hyp["%s-%s" % (key, m)] = {"verdict": "HOLDS" if abs(err) <= tol else "FAILS", "measured": round(x[k_meas], 3),
                                        "predicted": round(p[k_pred], 3), "err_pct": round(100 * err, 2)}
    pk = reg["pick"]
    if not pk["judged"]:
        hyp["PM-PICK"] = {"verdict": "NOT JUDGED", "why": "predicted margin %.2f %% < %.0f %%" % (100 * pk["margin_over_next"],
                                                                                                    100 * reg["pick_margin"])}
    elif any((meas.get(m) or {}).get("E_j") is None for m in MODES):
        hyp["PM-PICK"] = {"verdict": "INCOMPLETE"}
    else:
        least = min(MODES, key=lambda m: meas[m]["E_j"])
        hyp["PM-PICK"] = {"verdict": "HOLDS" if least == pk["least_energy"] else "FAILS", "predicted": pk["least_energy"],
                          "measured": least}
    six = [hyp["%s-%s" % (k, m)]["verdict"] for k in ("PM-T", "PM-E") for m in MODES]
    verdict = "INCOMPLETE" if "INCOMPLETE" in six else ("HOLDS" if all(v == "HOLDS" for v in six) else "FAILS")
    rep = {}
    L = meas.get("locked") or {}
    for m in MODES:
        x = meas.get(m) or {}
        if x.get("E_j") is not None and L.get("E_j") is not None:
            rep[m] = {"dT_ms_measured": round(x["T_ms"] - L["T_ms"], 1), "dT_ms_predicted": round(pred[m]["dT_ms"], 1),
                      "dE_j_measured": round(x["E_j"] - L["E_j"], 3), "dE_j_predicted": round(pred[m]["dE_j"], 3),
                      "dE_pct_measured": round(100 * (x["E_j"] / L["E_j"] - 1), 2),
                      "dT_pct_measured": round(100 * (x["T_ms"] / L["T_ms"] - 1), 2),
                      "jetson_outlet_j_measured": round(x["Ej_jetson_j"], 3) if x.get("Ej_jetson_j") is not None else None,
                      "jetson_outlet_j_emulated": round(pred[m]["Ej_j"], 3)}
    return {"hypotheses": hyp, "upgrade5": verdict, "reported": rep, "requests_per_mode_expected": want}


def score(d, reg_path=None):
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "power_modes_predictions_20261010.json")
    reg = json.load(open(reg_path))
    lines = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    idle = next(l["idle_before"] for l in lines if "idle_before" in l)
    rows = [l for l in lines if l.get("mode") and l.get("timings") and l["timings"].get("predicted_n")]
    meas = measure(d, reg, rows, idle)
    texts = {}
    for r in rows:
        texts.setdefault(r["id"], set()).add(r["content"])
    s = score_rows(reg, meas)
    restored = [l for l in lines if "restored" in l]
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": SS.sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": sorted(p for p, h in reg.get("inputs_sha256", {}).items()
                                                   if os.path.exists(os.path.join(ROOT, p)) and SS.sha(os.path.join(ROOT, p)) != h),
         "measured": meas, **s, "same_text_across_modes": all(len(v) == 1 for v in texts.values()),
         "jetson_restored_equal": bool(restored) and all(l["equal"] for l in restored)}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: s[k] for k in ("hypotheses", "upgrade5", "reported", "same_text_across_modes", "jetson_restored_equal",
                                        "code_changed_since_registration")}, indent=1))
    print("wrote %s" % os.path.relpath(os.path.join(d, "score.json"), ROOT))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "score"):
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1], "--smoke" in a)
    else:
        score(a[1], a[a.index("--reg") + 1] if "--reg" in a else None)
