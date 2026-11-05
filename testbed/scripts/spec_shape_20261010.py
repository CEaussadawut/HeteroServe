#!/usr/bin/env python3
"""The draft's placement under shaping, exactly as registered (register_spec_shape_20261010.py).

    python3 testbed/scripts/spec_shape_20261010.py run testbed/dataset/spec_shape_predictions_20261010.json [--smoke]
    python3 testbed/scripts/spec_shape_20261010.py score testbed/results/spec_shape_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit. It is step 3's runner (spec_serve_20261010.py), plus the
shaping:
- prechecks: no serve_pool.sh state, the Pi 4 unshaped, both model files' sha256 on the Pi 4;
- the Jetson's guard started by its NOTE's procedure and stopped last; the strip logged at 5 Hz and the Pi 4 polled at
  1 Hz for the whole session, with an idle window first and last;
- THE LINKS FIRST, with nothing loaded (measure_link.sh refuses a busy board): for each setting, netshape.sh apply on the
  registered ports (none at s0 and s0r), measure_link.sh on the measurement port, netshape.sh clear. Each log is copied in
  as link_<setting>.log;
- each session of the registered order: the laptop's draft RPC server first for the shaped arm, then serve_pool.sh start
  with the registered plan and the arm's env, and one warm-up request (never scored);
  - the shaped arm, at each setting in order: the setting's shaping applied, one warm-up request of 8 tokens (TCP settles
    to the new rate; never scored), the registered prompts, the shaping cleared;
  - the other arms: the prompts at s0;
  - each server's log is copied back, then serve_pool.sh stop.
The shaping is cleared on every way out.
--smoke: the plumbing only, into spec_shape_smoke_<stamp>/, never scored. It runs the shaped arm alone at s0 and b25, with
one prompt of 8 tokens and idle windows of 10 s.
score: ms per token = the server's sum of predicted_ms / sum of predicted_n over the requests (base and pi4d1 pooled over
their two sessions; lapd1 per setting). Each setting is re-priced at its measured link (the slower direction's
receiver-timed goodput; the mean ping RTT). The registered hypotheses SH-FLIP, SH-CROSS, SH-ACC and SH-SLOPE are scored
as written. A missing request makes the hypotheses that need it INCOMPLETE. Writes <dir>/score.json.
"""
import datetime, glob, hashlib, json, os, re, shutil, signal, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, post, listening, GUARD_PROCS
import register_spec_shape_20261010 as R                    # noqa: E402  speedups, crossover, mbit

STRIP = os.path.join(HERE, "shelly_strip.py")
POLL = os.path.join(HERE, "throttle_poll.sh")
SERVE = os.path.join(HERE, "serve_pool.sh")
NETSHAPE = os.path.join(HERE, "netshape.sh")
MEASURE = os.path.join(HERE, "measure_link.sh")
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def shape(reg, rate):
    """Apply the setting's shaping (rate None: none). Returns the tool's last line."""
    env = dict(os.environ, NETSHAPE_PEER=reg["shaping"]["peer"], NETSHAPE_PORT=reg["shaping"]["ports"])
    if not rate:
        return "unshaped"
    p = subprocess.run(["bash", NETSHAPE, "apply", str(rate), "0"], env=env, capture_output=True, text=True, timeout=120)
    if p.returncode:
        raise RuntimeError("netshape apply %s failed: %s" % (rate, (p.stdout + p.stderr)[-300:]))
    return (p.stdout.strip().splitlines() or [""])[-1]


def unshape():
    subprocess.run(["bash", NETSHAPE, "clear"], capture_output=True, text=True, timeout=120)
    return subprocess.run(["bash", NETSHAPE, "check"], capture_output=True, text=True, timeout=120).returncode == 0


def run(reg_path, smoke=False):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("spec_shape_smoke_%s" if smoke else "spec_shape_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    say("registration %s (%s)%s" % (os.path.relpath(reg_path, ROOT), sha(reg_path)[:16], " SMOKE: never scored" if smoke else ""))
    if os.path.exists(STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if subprocess.run(["bash", NETSHAPE, "check"], capture_output=True).returncode:
        say("FATAL: the Pi 4 is shaped, or did not answer (netshape.sh check)"); sys.exit(1)
    for path, want in reg["models_on_host"].items():
        rc, out = SS.ssh("pi4", "sha256sum %s | cut -d' ' -f1" % path, timeout=900)
        if out.strip() != want:
            say("FATAL: %s on the Pi 4 is %s, registered %s" % (path, out.strip()[:16], want[:16])); sys.exit(1)
        say("model %s sha256 OK on the Pi 4" % path)
    settings = [s for s in reg["settings"] if s["id"] in ("s0", "b25")] if smoke else reg["settings"]
    order = [reg["shaped_arm"]] if smoke else reg["order"]
    prompts = reg["prompts"][:1] if smoke else reg["prompts"]
    n_predict = 8 if smoke else reg["n_predict"]
    idle = (10, 10) if smoke else (reg["idle_before_s"], reg["idle_after_s"])
    out = open(os.path.join(d, "run.jsonl"), "a")
    guard, logger, tpoll, lap = False, None, None, None
    try:
        # ---- the links, with nothing loaded
        for s in settings:
            say("link %s: %s" % (s["id"], shape(reg, s["rate_mbit"])))
            mib = max(16, s["rate_mbit"]) if s["rate_mbit"] else 256
            mark = time.time()
            env = dict(os.environ, PAIR="pi4 laptop", MEASURE_PORT=str(reg["shaping"]["measure_port"]), MIB=str(mib), REPS="3")
            p = subprocess.run(["bash", MEASURE], env=env, capture_output=True, text=True, timeout=1800, cwd=ROOT)
            logs = [f for f in glob.glob(os.path.join(ROOT, "testbed", "results", "link_pi4-laptop_*.log")) if os.path.getmtime(f) >= mark - 1]
            if p.returncode or not logs:
                raise RuntimeError("measure_link.sh for %s failed: %s" % (s["id"], (p.stdout + p.stderr)[-300:]))
            shutil.copy(sorted(logs)[-1], os.path.join(d, "link_%s.log" % s["id"]))
            import link_to_json as LJ
            _, _, rec = LJ.parse(os.path.join(d, "link_%s.log" % s["id"]))
            say("link %s measured: %.2f Mbit/s (slower direction), RTT %.3f ms" % (s["id"], rec["mbps"], rec["rtt_ms"]))
            if not unshape():
                raise RuntimeError("could not clear the shaping after %s" % s["id"])
        # ---- the guard, the strip, the poll
        rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                                 '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
        if g.split() != ["synced", sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
            say("FATAL: guard precheck %r" % g); sys.exit(1)
        rc, n = SS.ssh("jetson", SS.GUARD_PROCS + " | wc -l")
        if n.strip() == "0":
            SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
            say("guard started")
        else:
            say("!!!!! a guard was already running (%s procs): not starting a second; stopped at the end" % n.strip())
        guard = True
        logger = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(d, "strip.csv"), "--hz", "5"],
                                  stdout=open(os.path.join(d, "strip_logger.stdout"), "w"), stderr=subprocess.STDOUT,
                                  start_new_session=True)
        subprocess.run(["scp", "-q", POLL, "pi4:throttle_poll.sh"], check=True)
        rc, p = SS.ssh("pi4", "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_sh_%s.log > /dev/null 2>&1 "
                              "< /dev/null & echo $!" % stamp)
        tpoll = p.strip()
        say("strip logger pid %d, Pi 4 throttle poll pid %s; idle %d s" % (logger.pid, tpoll, idle[0]))
        t_idle0 = time.time(); time.sleep(idle[0]); t_idle1 = time.time()
        out.write(json.dumps({"idle_before": [t_idle0, t_idle1]}) + "\n"); out.flush()
        # ---- the sessions
        for i, arm_id in enumerate(order):
            arm = reg["arms"][arm_id]
            say("===== session %d: %s" % (i, arm_id))
            if arm.get("laptop_rpc"):
                ip, port = arm["laptop_rpc"].split(":")
                lap = subprocess.Popen([reg["laptop_rpc_server"], "-H", ip, "-p", port, "-t", "4"],
                                       stdout=open(os.path.join(d, "rpc_laptop_%d.log" % i), "w"), stderr=subprocess.STDOUT,
                                       start_new_session=True)
                for _ in range(60):
                    if SS.listening(arm["laptop_rpc"]):
                        break
                    time.sleep(0.5)
                if not SS.listening(arm["laptop_rpc"]):
                    say("FATAL: the laptop's RPC server never listened on %s" % arm["laptop_rpc"]); sys.exit(1)
                say("laptop RPC server pid %d on %s" % (lap.pid, arm["laptop_rpc"]))
            env = dict(os.environ, MODEL=reg["model"], CTX=str(reg["ctx"]), PLACE=reg["place"], PLACE_HEAD=reg["place_head"],
                       NODES="pi4 jetson", **arm["env"])
            r = subprocess.run(["bash", SERVE, "start"], env=env, capture_output=True, text=True, timeout=2400)
            open(os.path.join(d, "serve_%d_%s.console" % (i, arm_id)), "w").write(r.stdout + r.stderr)
            if r.returncode or "READY" not in r.stdout:
                say("FATAL: serve_pool.sh start failed (exit %d); see serve_%d_%s.console" % (r.returncode, i, arm_id)); sys.exit(1)
            say("ready: %s" % " | ".join(l for l in r.stdout.splitlines() if l.startswith(("split", "mem", "draft"))))
            SS.post(reg["host_url"] + "/completion", dict(reg["sampling"], prompt=reg["chat_format"] % reg["warmup_prompt"],
                                                           n_predict=16), timeout=3600)
            say("warm-up done")
            for s in (settings if arm_id == reg["shaped_arm"] else [{"id": "s0", "rate_mbit": None}]):
                if arm_id == reg["shaped_arm"]:
                    say("setting %s: %s" % (s["id"], shape(reg, s["rate_mbit"])))
                    SS.post(reg["host_url"] + "/completion", dict(reg["sampling"], prompt=reg["chat_format"] % reg["warmup_prompt"],
                                                                   n_predict=8), timeout=3600)
                for pr in prompts:
                    body = dict(reg["sampling"], prompt=reg["chat_format"] % pr["text"], n_predict=n_predict)
                    t0 = time.time()
                    try:
                        resp = SS.post(reg["host_url"] + "/completion", body, timeout=3600)
                    except Exception as e:
                        say("!!!!! %s %s failed: %s" % (s["id"], pr["id"], e)); continue
                    t1 = time.time()
                    tm = resp.get("timings") or {}
                    out.write(json.dumps({"session": i, "arm": arm_id, "setting": s["id"], "id": pr["id"], "t0": t0, "t1": t1,
                                          "tokens_predicted": resp.get("tokens_predicted"), "timings": tm,
                                          "content": resp.get("content")}, ensure_ascii=False) + "\n"); out.flush()
                    say("%-5s %-4s %-5s %3s tokens, %6.1f ms/token, draft %s accepted %s" % (
                        arm_id, s["id"], pr["id"], resp.get("tokens_predicted"),
                        (tm.get("predicted_ms") or 0) / max(1, tm.get("predicted_n") or 1), tm.get("draft_n"), tm.get("draft_n_accepted")))
                if arm_id == reg["shaped_arm"] and s["rate_mbit"] and not unshape():
                    say("FATAL: could not clear the shaping after %s" % s["id"]); sys.exit(1)
            rc, name = SS.ssh("pi4", "ls -t ~/logs/llama-server_serve_*.log | head -1")
            subprocess.run(["scp", "-q", "pi4:%s" % name.strip(), os.path.join(d, "server_%d_%s.log" % (i, arm_id))])
            r = subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600)
            open(os.path.join(d, "serve_%d_%s.console" % (i, arm_id)), "a").write(r.stdout + r.stderr)
            say("stopped: %s" % " | ".join(l for l in r.stdout.splitlines() if "stop" in l))
            if lap:
                os.killpg(lap.pid, signal.SIGTERM); lap.wait(timeout=30); lap = None
                say("laptop RPC server stopped")
        t_idle0 = time.time(); time.sleep(idle[1]); t_idle1 = time.time()
        out.write(json.dumps({"idle_after": [t_idle0, t_idle1]}) + "\n"); out.flush()
    finally:
        say("shaping cleared and checked: %s" % unshape())
        if os.path.exists(STATE):
            subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600); say("serve_pool.sh stopped (cleanup)")
        if lap:
            os.killpg(lap.pid, signal.SIGTERM); say("laptop RPC server stopped (cleanup)")
        if tpoll:
            SS.ssh("pi4", "kill %s" % tpoll)
            subprocess.run(["scp", "-q", "pi4:logs/throttle_sh_%s.log" % stamp, os.path.join(d, "throttle_pi4.log")])
        if logger:
            os.killpg(logger.pid, signal.SIGTERM); logger.wait(timeout=30)
        if guard:
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
    say("RUN DONE %s" % os.path.relpath(d, ROOT))


# ------------------------------------------------------------------ score
def ms_per_token(rows):
    n = sum(r["timings"]["predicted_n"] for r in rows)
    return sum(r["timings"]["predicted_ms"] for r in rows) / n if n else None


def score_rows(reg, rows, links):
    """The registered hypotheses from request rows and {setting: (goodput bytes/ms, rtt ms)}. Pure, so the tests can feed it."""
    m, th = reg["model_parts"], reg["thresholds"]
    want = len(reg["prompts"])
    sess = lambda a: sorted({r["session"] for r in rows if r["arm"] == a})
    complete = {a: all(sum(1 for r in rows if r["arm"] == a and r["session"] == s) == want for s in sess(a)) and
                len(sess(a)) == reg["order"].count(a) for a in ("base", "pi4d1")}
    base, pi4 = ms_per_token([r for r in rows if r["arm"] == "base"]), ms_per_token([r for r in rows if r["arm"] == "pi4d1"])
    lap, per = {}, {}
    for s in reg["settings"]:
        rs = [r for r in rows if r["arm"] == reg["shaped_arm"] and r["setting"] == s["id"]]
        lap[s["id"]] = ms_per_token(rs) if len(rs) == want else None
        if s["id"] in links:
            bw, rtt = links[s["id"]]
            p = R.speedups(m, bw, rtt)
            per[s["id"]] = {"goodput_mbit": round(R.mbit(bw), 2), "rtt_ms": round(rtt, 3), "pred_lapd1": p["lapd1"],
                            "pred_pi4d1": p["pi4d1"], "pred_gap": p["pi4d1"] / p["lapd1"] - 1,   # unrounded: the verdicts use them
                            "lapd1_ms": round(lap[s["id"]], 2) if lap[s["id"]] else None}
    hyp = {}
    ok = base is not None and pi4 is not None and complete["base"] and complete["pi4d1"]
    # SH-FLIP: pred_gap = pred_pi4d1 / pred_lapd1 - 1, so a negative gap predicts the laptop faster; measured faster = fewer ms
    q = {s: v for s, v in per.items() if abs(v["pred_gap"]) >= th["flip_min"]}
    flips = {s: (("lapd1" if lap[s] < pi4 else "pi4d1") == ("lapd1" if v["pred_gap"] < 0 else "pi4d1"))
             if (ok and lap.get(s)) else None for s, v in q.items()}
    hyp["SH-FLIP"] = {"verdict": "INCOMPLETE" if (None in flips.values() or not ok) else
                      ("HOLDS" if len(flips) >= 2 and all(flips.values()) else "FAILS"), "settings": flips}
    # SH-CROSS: the measured gap lapd1/pi4d1 - 1 (ms), against 1/bw
    pts = sorted(((1.0 / links[s][0], lap[s] / pi4 - 1) for s in per if lap.get(s) and ok), key=lambda t: t[0])
    meas_x = None
    for (x0, g0), (x1, g1) in zip(pts, pts[1:]):
        if g0 < 0 <= g1:
            meas_x = 1.0 / (x0 + (x1 - x0) * (0 - g0) / (g1 - g0))
            break
    rtt0 = statistics.fmean(v[1] for v in links.values()) if links else m["rtt_registered"]
    pred_x = R.crossover(m, rtt0)
    hyp["SH-CROSS"] = {"verdict": "INCOMPLETE" if not ok or len(pts) < len(per) else
                       ("HOLDS" if (meas_x and pred_x and pred_x / th["cross_factor"] <= meas_x <= pred_x * th["cross_factor"])
                        else "FAILS"),
                       "measured_mbit": round(R.mbit(meas_x), 2) if meas_x else None,
                       "predicted_mbit": round(R.mbit(pred_x), 2) if pred_x else None}
    # SH-ACC
    acc = {s: round(100 * (v["pred_lapd1"] / (base / lap[s]) - 1), 2) for s, v in per.items() if lap.get(s) and base}
    hyp["SH-ACC"] = {"verdict": "INCOMPLETE" if (len(acc) < len(reg["settings"]) or not ok) else
                     ("HOLDS" if all(abs(e) <= th["acc_pct"] for e in acc.values()) else "FAILS"), "err_pct": acc}
    # SH-SLOPE
    s0 = "s0"
    xs, ys = [], []
    if lap.get(s0) and s0 in per:
        pred_ms = lambda s: m["T1"] / per[s]["pred_lapd1"]
        for s in per:
            if s not in ("s0", "s0r") and lap.get(s):
                xs.append(pred_ms(s) - pred_ms(s0)); ys.append(lap[s] - lap[s0])
    c = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs) if xs else None
    lo, hi = th["slope_band"]
    hyp["SH-SLOPE"] = {"verdict": "INCOMPLETE" if c is None or len(xs) < len([s for s in reg["settings"] if s["rate_mbit"]]) else
                       ("HOLDS" if lo <= c <= hi else "FAILS"), "c": round(c, 4) if c is not None else None}
    return {"ms_per_token": {"base": base, "pi4d1": pi4, "lapd1": lap}, "settings": per, "hypotheses": hyp,
            "confirmed": hyp["SH-FLIP"]["verdict"] == "HOLDS" and hyp["SH-CROSS"]["verdict"] == "HOLDS",
            "drift_lapd1_s0_s0r_pct": round(100 * (lap["s0r"] / lap["s0"] - 1), 2) if lap.get("s0") and lap.get("s0r") else None}


def score(d, reg_path=None):
    import link_to_json as LJ
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "spec_shape_predictions_20261010.json")
    reg = json.load(open(reg_path))
    rows = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    rows = [r for r in rows if r.get("arm") and r.get("timings") and r["timings"].get("predicted_n")]
    links = {}
    for s in reg["settings"]:
        f = os.path.join(d, "link_%s.log" % s["id"])
        if os.path.exists(f):
            _, _, rec = LJ.parse(f)
            links[s["id"]] = (rec["mbps"] * 125.0, rec["rtt_ms"])          # Mbit/s -> bytes per ms
    s = score_rows(reg, rows, links)
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha(reg_path), "run": os.path.relpath(d, ROOT), **s}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps({k: s[k] for k in ("ms_per_token", "hypotheses", "confirmed", "drift_lapd1_s0_s0r_pct")}, indent=1))
    for sid, v in s["settings"].items():
        print("%-4s %s" % (sid, v))
    print("wrote %s" % os.path.relpath(os.path.join(d, "score.json"), ROOT))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] not in ("run", "score"):
        sys.exit(__doc__)
    if a[0] == "run":
        run(a[1], "--smoke" in a)
    else:
        score(a[1], a[a.index("--reg") + 1] if "--reg" in a else None)
