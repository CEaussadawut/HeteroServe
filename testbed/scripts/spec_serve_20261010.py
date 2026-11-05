#!/usr/bin/env python3
"""Scheduler upgrade 1, step 3: speculative decoding SERVED on the boards, exactly as registered
(register_spec_serve_20261010.py).

    python3 testbed/scripts/spec_serve_20261010.py run testbed/dataset/spec_serve_predictions_20261010.json [--smoke]
    python3 testbed/scripts/spec_serve_20261010.py score testbed/results/spec_serve_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit:
- prechecks: no serve_pool.sh state, the link unshaped (netshape.sh check), both model files' sha256 on the Pi 4;
- the Jetson's guard started by its NOTE's procedure (clock synchronized, the script's sha256 the repo's) and stopped last;
- the strip logged at 5 Hz for the whole session (shelly_strip.py, read-only), with an idle window first and last; the
  Pi 4 polled at 1 Hz (throttle_poll.sh v2) for the whole session;
- each session of the registered order: the draft's laptop RPC server first if the arm has one (the pinned
  ggml-rpc-server, CPU, on the laptop's LAN address), then serve_pool.sh start with the registered plan and the arm's
  SERVE_DRAFT / DRAFT_HOST_MIB / SERVE_RPC_EXTRA; one warm-up request (never scored: the Jetson compiles a new batch
  size's pipelines on first use); the registered prompts in order, greedy, in Qwen3's non-thinking chat format; the
  server's log copied back; serve_pool.sh stop; the laptop server stopped.
--smoke: the two draft arms (sessions 1 and 2), one prompt of 8 tokens each, idle windows of 10 s, into spec_serve_smoke_<stamp>/ (the
plumbing only: never scored).
score: per arm, decode ms per token = sum of the server's predicted_ms / sum of predicted_n over its requests (the base
arm pooled over its sessions; their difference reported as drift); each draft arm's speedup = base / arm, also paired per
prompt; acceptance on the boards from the server's per-check lines; gross J per decoded token over each request's decode
window ([send + prompt_ms, reply]) for the pool's outlets and with the laptop's; the Pi 4's thermal verdict per request.
  SS-BAR    the engine's choice (the arm the registration predicts fastest) is at least 1.30x the base arm
  SS-ACC    every draft arm's measured speedup within 10 % of its prediction
  SS-ORDER  the draft arm predicted faster is measured faster
  SS-MOE    registered as not testable: the MoE does not fit the Pi 4 + Jetson
An arm missing a request makes the hypotheses that need it INCOMPLETE. Writes <dir>/score.json.
"""
import datetime, hashlib, json, os, re, signal, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
STRIP = os.path.join(HERE, "shelly_strip.py")
POLL = os.path.join(HERE, "throttle_poll.sh")
SERVE = os.path.join(HERE, "serve_pool.sh")
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
ACC = re.compile(r"accepted\s+(\d+)/\s*(\d+) draft tokens")
GUARD_PROCS = 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline'


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ssh(node, cmd, timeout=900):
    p = subprocess.run(SSH + [node, cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    return p.returncode, p.stdout


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def listening(addr):
    p = subprocess.run(["ss", "-ltn"], capture_output=True, text=True)
    return (" %s " % addr) in p.stdout


def run(reg_path, smoke=False):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("spec_serve_smoke_%s" if smoke else "spec_serve_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    say("registration %s (%s)%s" % (os.path.relpath(reg_path, ROOT), sha(reg_path)[:16], " SMOKE: never scored" if smoke else ""))
    if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if subprocess.run(["bash", os.path.join(HERE, "netshape.sh"), "check"], capture_output=True).returncode:
        say("FATAL: the Pi 4 is shaped, or did not answer (netshape.sh check)"); sys.exit(1)
    for path, want in reg["models_on_host"].items():
        rc, out = ssh("pi4", "sha256sum %s | cut -d' ' -f1" % path, timeout=900)
        if out.strip() != want:
            say("FATAL: %s on the Pi 4 is %s, registered %s" % (path, out.strip()[:16], want[:16])); sys.exit(1)
        say("model %s sha256 OK on the Pi 4" % path)
    guard, logger, tpoll, lap = False, None, None, None
    sessions = reg["order"][1:3] if smoke else reg["order"]      # the smoke: the two draft arms, the laptop RPC path included
    prompts = reg["prompts"][:1] if smoke else reg["prompts"]
    n_predict = 8 if smoke else reg["n_predict"]
    idle = (10, 10) if smoke else (reg["idle_before_s"], reg["idle_after_s"])
    out = open(os.path.join(d, "run.jsonl"), "a")
    try:
        rc, g = ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                              '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
        if g.split() != ["synced", sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
            say("FATAL: guard precheck %r" % g); sys.exit(1)
        rc, n = ssh("jetson", GUARD_PROCS + " | wc -l")
        if n.strip() == "0":
            ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
            say("guard started")
        else:
            say("!!!!! a guard was already running (%s procs): not starting a second; stopped at the end, as the NOTE keeps it" % n.strip())
        guard = True
        logger = subprocess.Popen([sys.executable, STRIP, "log", "--out", os.path.join(d, "strip.csv"), "--hz", "5"],
                                  stdout=open(os.path.join(d, "strip_logger.stdout"), "w"), stderr=subprocess.STDOUT,
                                  start_new_session=True)
        subprocess.run(["scp", "-q", POLL, "pi4:throttle_poll.sh"], check=True)
        rc, p = ssh("pi4", "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_ss_%s.log > /dev/null 2>&1 "
                           "< /dev/null & echo $!" % stamp)
        tpoll = p.strip()
        say("strip logger pid %d, Pi 4 throttle poll pid %s; idle %d s" % (logger.pid, tpoll, idle[0]))
        t_idle0 = time.time(); time.sleep(idle[0]); t_idle1 = time.time()
        out.write(json.dumps({"idle_before": [t_idle0, t_idle1]}) + "\n"); out.flush()
        for i, arm_id in enumerate(sessions):
            arm = reg["arms"][arm_id]
            say("===== session %d: %s" % (i, arm_id))
            if arm.get("laptop_rpc"):
                ip, port = arm["laptop_rpc"].split(":")
                lap = subprocess.Popen([reg["laptop_rpc_server"], "-H", ip, "-p", port, "-t", "4"],
                                       stdout=open(os.path.join(d, "rpc_laptop_%d.log" % i), "w"), stderr=subprocess.STDOUT,
                                       start_new_session=True)
                for _ in range(60):
                    if listening(arm["laptop_rpc"]):
                        break
                    time.sleep(0.5)
                if not listening(arm["laptop_rpc"]):
                    say("FATAL: the laptop's RPC server never listened on %s" % arm["laptop_rpc"]); sys.exit(1)
                say("laptop RPC server pid %d on %s" % (lap.pid, arm["laptop_rpc"]))
            env = dict(os.environ, MODEL=reg["model"], CTX=str(reg["ctx"]), PLACE=reg["place"], PLACE_HEAD=reg["place_head"],
                       NODES="pi4 jetson", **arm["env"])
            r = subprocess.run(["bash", SERVE, "start"], env=env, capture_output=True, text=True, timeout=2400)
            open(os.path.join(d, "serve_%d_%s.console" % (i, arm_id)), "w").write(r.stdout + r.stderr)
            if r.returncode or "READY" not in r.stdout:
                say("FATAL: serve_pool.sh start failed (exit %d); see serve_%d_%s.console" % (r.returncode, i, arm_id)); sys.exit(1)
            say("ready: %s" % " | ".join(l for l in r.stdout.splitlines() if l.startswith(("split", "mem", "draft"))))
            body = dict(reg["sampling"], prompt=reg["chat_format"] % reg["warmup_prompt"], n_predict=16)
            post(reg["host_url"] + "/completion", body, timeout=3600)
            say("warm-up done")
            for p in prompts:
                body = dict(reg["sampling"], prompt=reg["chat_format"] % p["text"], n_predict=n_predict)
                t0 = time.time()
                try:
                    resp = post(reg["host_url"] + "/completion", body, timeout=3600)
                except Exception as e:
                    say("!!!!! %s failed: %s" % (p["id"], e)); continue
                t1 = time.time()
                tm = resp.get("timings") or {}
                out.write(json.dumps({"session": i, "arm": arm_id, "id": p["id"], "lang": p["lang"], "t0": t0, "t1": t1,
                                      "tokens_predicted": resp.get("tokens_predicted"), "stop_type": resp.get("stop_type"),
                                      "timings": tm, "content": resp.get("content")}, ensure_ascii=False) + "\n"); out.flush()
                say("%-5s %-6s %3s tokens, %6.1f ms/token, draft %s accepted %s" % (arm_id, p["id"], resp.get("tokens_predicted"),
                    (tm.get("predicted_ms") or 0) / max(1, tm.get("predicted_n") or 1), tm.get("draft_n"), tm.get("draft_n_accepted")))
            rc, name = ssh("pi4", "ls -t ~/logs/llama-server_serve_*.log | head -1")
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
        if os.path.exists(os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")):
            subprocess.run(["bash", SERVE, "stop"], capture_output=True, text=True, timeout=600); say("serve_pool.sh stopped (cleanup)")
        if lap:
            os.killpg(lap.pid, signal.SIGTERM); say("laptop RPC server stopped (cleanup)")
        if tpoll:
            ssh("pi4", "kill %s" % tpoll)
            subprocess.run(["scp", "-q", "pi4:logs/throttle_ss_%s.log" % stamp, os.path.join(d, "throttle_pi4.log")])
        if logger:
            os.killpg(logger.pid, signal.SIGTERM); logger.wait(timeout=30)
        if guard:
            rc, o = ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p stopped"; '
                                  'done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (GUARD_PROCS, GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
    say("RUN DONE %s" % os.path.relpath(d, ROOT))


def score(d, reg_path=None):
    import full_eval as F
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "spec_serve_predictions_20261010.json")
    reg = json.load(open(reg_path))
    lines = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    idle = next((l["idle_before"] for l in lines if "idle_before" in l), None)
    rows = [l for l in lines if "arm" in l and l.get("timings")]
    poll = open(os.path.join(d, "throttle_pi4.log"), errors="replace").readlines() if os.path.exists(os.path.join(d, "throttle_pi4.log")) else None
    out = {o: n for n, o in reg["outlets"].items()}
    arms = {}
    for a in reg["arms"]:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            arms[a] = None; continue
        ms = sum(r["timings"]["predicted_ms"] for r in rs) / sum(r["timings"]["predicted_n"] for r in rs)
        per_prompt = {}
        for r in rs:
            per_prompt.setdefault(r["id"], []).append(r["timings"]["predicted_ms"] / r["timings"]["predicted_n"])
        sess = sorted({r["session"] for r in rs})
        by_sess = {str(s): round(sum(r["timings"]["predicted_ms"] for r in rs if r["session"] == s)
                                 / sum(r["timings"]["predicted_n"] for r in rs if r["session"] == s), 1) for s in sess}
        checks = []
        for s in sess:
            p = os.path.join(d, "server_%d_%s.log" % (s, a))
            if os.path.exists(p):
                checks += [(int(x), int(y)) for x, y in ACC.findall(open(p, errors="replace").read())]
        dn = sum(r["timings"].get("draft_n", 0) or 0 for r in rs)
        da = sum(r["timings"].get("draft_n_accepted", 0) or 0 for r in rs)
        en = {"pool_j": 0.0, "with_laptop_j": 0.0, "tokens": 0}
        th = []
        for r in rs:
            t0 = r["t0"] + r["timings"]["prompt_ms"] / 1000.0
            if idle:
                w = F.window_energy(d, t0, r["t1"], idle, list(out))
                pool = sum(w[o][0] for o in out if out[o] in ("pi4", "jetson"))
                en["pool_j"] += pool; en["with_laptop_j"] += pool + sum(w[o][0] for o in out if out[o] == "laptop")
                en["tokens"] += r["timings"]["predicted_n"]
            if poll:
                th.append(F.thermal(poll, r["t0"], r["t1"])["verdict"])
        arms[a] = {"requests": len(rs), "tokens": sum(r["timings"]["predicted_n"] for r in rs), "ms_per_token": round(ms, 1),
                   "by_session": by_sess, "per_prompt_ms": {k: round(statistics.fmean(v), 1) for k, v in per_prompt.items()},
                   "draft_n": dn, "draft_accepted": da, "checks": len(checks),
                   "alpha1_boards": round(sum(1 for x, y in checks if y >= 1 and x >= 1) / max(1, sum(1 for x, y in checks if y >= 1)), 4) if checks else None,
                   "j_per_token_pool": round(en["pool_j"] / en["tokens"], 3) if en["tokens"] else None,
                   "j_per_token_with_laptop": round(en["with_laptop_j"] / en["tokens"], 3) if en["tokens"] else None,
                   "thermal": ("THROTTLED" if "THROTTLED" in th else "UNKNOWN" if "UNKNOWN" in th else "clean") if th else "not polled"}
    base_id = reg["base_arm"]
    want = len(reg["prompts"]) * reg["order"].count(base_id)
    complete = {a: arms[a] is not None and arms[a]["requests"] == len(reg["prompts"]) * reg["order"].count(a) for a in reg["arms"]}
    hyp, sp = {}, {}
    if arms.get(base_id):
        for a, pa in reg["predicted"].items():
            if arms.get(a):
                s = arms[base_id]["ms_per_token"] / arms[a]["ms_per_token"]
                pairs = [arms[base_id]["per_prompt_ms"][k] / v for k, v in arms[a]["per_prompt_ms"].items() if k in arms[base_id]["per_prompt_ms"]]
                sp[a] = {"speedup": round(s, 3), "predicted": pa["speedup"], "err_pct": round(100 * (pa["speedup"] / s - 1), 2),
                         "paired_mean": round(statistics.fmean(pairs), 3), "paired_min": round(min(pairs), 3), "paired_max": round(max(pairs), 3)}
    ch = reg["engine_choice"]
    ok = all(complete.values())
    hyp["SS-BAR"] = {"verdict": ("HOLDS" if sp[ch]["speedup"] >= 1.30 else "FAILS") if ok and ch in sp else "INCOMPLETE",
                     "arm": ch, "speedup": sp.get(ch, {}).get("speedup")}
    hyp["SS-ACC"] = {"verdict": ("HOLDS" if all(abs(sp[a]["predicted"] / sp[a]["speedup"] - 1) <= 0.10 for a in reg["predicted"]) else "FAILS")
                     if ok and all(a in sp for a in reg["predicted"]) else "INCOMPLETE",
                     "err_pct": {a: v["err_pct"] for a, v in sp.items()}}
    pf = max(reg["predicted"], key=lambda a: reg["predicted"][a]["speedup"])
    hyp["SS-ORDER"] = {"verdict": ("HOLDS" if max(sp, key=lambda a: sp[a]["speedup"]) == pf else "FAILS")
                       if ok and all(a in sp for a in reg["predicted"]) else "INCOMPLETE", "predicted_fastest": pf}
    hyp["SS-MOE"] = {"verdict": "NOT TESTABLE", "why": reg["moe_note"]}
    drift = None
    bs = (arms.get(base_id) or {}).get("by_session") or {}
    if len(bs) >= 2:
        v = list(bs.values()); drift = round(100 * (v[-1] / v[0] - 1), 2)
    code = [f for f, h in reg["inputs_sha256"].items() if os.path.exists(os.path.join(HERE, f)) and sha(os.path.join(HERE, f)) != h]
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": code, "complete": complete, "arms": arms, "speedups": sp, "base_drift_pct": drift,
         "hypotheses": hyp}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    return s


def main(argv):
    if len(argv) < 2 or argv[0] not in ("run", "score"):
        sys.exit(__doc__)
    if argv[0] == "run":
        run(argv[1], smoke="--smoke" in argv)
    else:
        s = score(argv[1], argv[argv.index("--reg") + 1] if "--reg" in argv else None)
        print(json.dumps({k: s[k] for k in ("complete", "speedups", "base_drift_pct", "hypotheses", "code_changed_since_registration")},
                         indent=1))
        for a, v in s["arms"].items():
            print(a, json.dumps({k: v[k] for k in v if k != "per_prompt_ms"}) if v else None)
        print("wrote %s" % os.path.relpath(os.path.join(argv[1], "score.json"), ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
