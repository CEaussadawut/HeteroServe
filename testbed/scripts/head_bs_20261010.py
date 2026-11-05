#!/usr/bin/env python3
"""Scheduler upgrade 6: backend sampling on the head board, exactly as registered (register_head_bs_20261010.py).

    python3 testbed/scripts/head_bs_20261010.py run testbed/dataset/head_bs_predictions_20261010.json [--smoke]
    python3 testbed/scripts/head_bs_20261010.py score testbed/results/head_bs_<stamp> [--reg PATH]

run, from the repo root on the laptop, as a systemd user unit:
- prechecks: no serve_pool.sh state; the link unshaped (netshape.sh check); neither board busy; both llama.cpp trees at
  the pin and clean; the model's sha256 on the Pi 4; the Jetson as found (MAXN, governors free);
- THE LINKS FIRST, with nothing loaded (measure_link.sh refuses a busy board): each setting's shaping applied on the
  registered ports, measure_link.sh PAIR="pi4 jetson" on the measurement port, then the shaping cleared;
- the Jetson's guard started by its NOTE's procedure and stopped last. Its clocks are locked as serve_pool.sh locks them
  for a serve (stored, locked, restored, checked equal to the snapshot). The Pi 4 is polled at 1 Hz (throttle_poll.sh);
- ggml-rpc-server on the Jetson and llama-server on the Pi 4, each with serve_pool.sh's command line, the server with
  the registered arguments; stopped by their exact argv, never pkill -f;
- one warm-up request per arm (the first backend-sampled request compiles the Jetson's sampling pipelines);
- each setting in order: its shaping applied, one settle request per arm of 8 tokens, then the prompts in the ABBA arm
  order. The Jetson's RPC socket counters (bytes_acked, bytes_received) are read around each request. Then the shaping is
  cleared. The shaping is cleared on every way out.
--smoke: the plumbing only (s0 and b50, one prompt of 8 tokens per arm, into head_bs_smoke_<stamp>/, never scored).
score: per setting and arm, ms per token = the server's sum of predicted_ms / sum of predicted_n; the saving = off - on,
priced at per_token_factor x V / g (registration b: (n - 1) / n, as the server times n - 1 passes over n tokens; a: 1);
the Jetson's bytes sent per decoded token; each setting re-priced at its measured Jetson -> Pi 4 goodput. HB-BYTES,
HB-SAVE, HB-SLOPE and HB-SAME are scored as written. Writes <dir>/score.json.
"""
import datetime, glob, json, os, re, shutil, signal, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, post, sha, GUARD_PROCS
import power_modes_char_20261010 as PC                      # noqa: E402  STATE_CMD, parse_state, set_mode

TPOLL = os.path.join(HERE, "throttle_poll.sh")
NETSHAPE = os.path.join(HERE, "netshape.sh")
MEASURE = os.path.join(HERE, "measure_link.sh")
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
PIN = "b23701f77d47dad9de834d59ebfcbe25c9e8b46f"
JETSON_IP, PI4_IP, RPC_PORT, HTTP_PORT = "192.168.1.114", "192.168.1.110", 50052, 8080
BUSY = 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%s " $p; done; true'
TREE = "git -C ~/llama.cpp rev-parse HEAD; git -C ~/llama.cpp status --porcelain --untracked-files=no | wc -l"
SOCK = "ss -tinH state established '( sport = :%d )'" % RPC_PORT


def pids(node, binary, needle):
    """PIDs of `binary` whose argv contains `needle` (exact argv match, as serve_pool.sh: never pgrep -f)."""
    rc, o = SS.ssh(node, "for p in $(pgrep -x %s); do tr '\\0' ' ' < /proc/$p/cmdline | grep -qF -- '%s' && echo $p; done" % (binary, needle))
    return o.split()


def jetson_bytes():
    """(bytes_acked, bytes_received) summed over the established connections on the Jetson's RPC port."""
    rc, o = SS.ssh("jetson", SOCK)
    a = sum(int(x) for x in re.findall(r"bytes_acked:(\d+)", o))
    r = sum(int(x) for x in re.findall(r"bytes_received:(\d+)", o))
    return a, r


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run(reg_path, smoke=False):
    reg = json.load(open(reg_path))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("head_bs_smoke_%s" if smoke else "head_bs_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")
    t_start = time.time()

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    settings = [s for s in reg["settings"] if s["id"] in ("s0", "b50")] if smoke else reg["settings"]
    prompts = reg["prompts"][:1] if smoke else reg["prompts"]
    n_predict = 8 if smoke else reg["n_predict"]
    abba = ["off", "on"] if smoke else reg["abba"]
    say("registration %s (%s)%s" % (os.path.relpath(reg_path, ROOT), SS.sha(reg_path)[:16], " SMOKE: never scored" if smoke else ""))
    # ---- prechecks, nothing changed yet
    if os.path.exists(STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if subprocess.run(["bash", NETSHAPE, "check"], capture_output=True).returncode:
        say("FATAL: the Pi 4 is shaped, or did not answer (netshape.sh check)"); sys.exit(1)
    for n in ("pi4", "jetson"):
        rc, b = SS.ssh(n, BUSY)
        if b.strip():
            say("FATAL: %s is busy: %s" % (n, b.strip())); sys.exit(1)
        rc, t = SS.ssh(n, TREE)
        if t.split() != [PIN, "0"]:
            say("FATAL: %s's llama.cpp is %r, not the pin clean" % (n, t)); sys.exit(1)
    rc, out = SS.ssh("pi4", "sha256sum %s | cut -d' ' -f1" % reg["model_file"], timeout=300)
    if out.strip() != reg["model_sha256"]:
        say("FATAL: %s on the Pi 4 is %s, registered %s" % (reg["model_file"], out.strip()[:16], reg["model_sha256"][:16])); sys.exit(1)
    say("model sha256 OK on the Pi 4; both trees at the pin, clean")
    rc, asfound = SS.ssh("jetson", PC.STATE_CMD)
    asfound = asfound.strip()
    k = PC.parse_state(asfound)
    if not (k["pmode"] == "pmode:0000" and k["online"] == "0-3" and k["gpu"].split("-")[0] != k["gpu"].split("-")[1]
            and k["railgate"] == "1" and k["emcov"] == "0" and k["cpuidle"] == "00"):
        say("FATAL: the Jetson is not as found (MAXN, governors free): %s" % asfound); sys.exit(1)
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    sh = reg["shaping"]
    senv = dict(os.environ, NETSHAPE_PEER=sh["peer"], NETSHAPE_PORT=sh["ports"])

    def shape(rate):
        if not rate:
            return "unshaped"
        p = subprocess.run(["bash", NETSHAPE, "apply", str(rate), "0"], env=senv, capture_output=True, text=True, timeout=120)
        if p.returncode:
            raise RuntimeError("netshape apply %s failed: %s" % (rate, (p.stdout + p.stderr)[-300:]))
        return (p.stdout.strip().splitlines() or [""])[-1]

    def unshape():
        subprocess.run(["bash", NETSHAPE, "clear"], capture_output=True, text=True, timeout=120)
        return subprocess.run(["bash", NETSHAPE, "check"], capture_output=True, text=True, timeout=120).returncode == 0
    out = open(os.path.join(d, "run.jsonl"), "a")
    rec = lambda o: (out.write(json.dumps(o, ensure_ascii=False) + "\n"), out.flush())
    stored = "~/heteroserve/run/jetson_clocks_hbs_%s.stored" % stamp
    started_guard, tpoll, stored_ok, rpc_up, srv_up = False, None, False, False, False
    try:
        # ---- the links, with nothing loaded
        import link_to_json as LJ
        for s in settings:
            say("link %s: %s" % (s["id"], shape(s["rate_mbit"])))
            mib = max(16, s["rate_mbit"]) if s["rate_mbit"] else 256
            mark = time.time()
            env = dict(os.environ, PAIR=sh["pair"], MEASURE_PORT=str(sh["measure_port"]), MIB=str(mib), REPS="3")
            p = subprocess.run(["bash", MEASURE], env=env, capture_output=True, text=True, timeout=1800, cwd=ROOT)
            logs = [f for f in glob.glob(os.path.join(ROOT, "testbed", "results", "link_pi4-jetson_*.log")) if os.path.getmtime(f) >= mark - 1]
            if p.returncode or not logs:
                raise RuntimeError("measure_link.sh for %s failed: %s" % (s["id"], (p.stdout + p.stderr)[-300:]))
            shutil.copy(sorted(logs)[-1], os.path.join(d, "link_%s.log" % s["id"]))
            _, _, lk = LJ.parse(os.path.join(d, "link_%s.log" % s["id"]))
            say("link %s measured: jetson->pi4 %.2f Mbit/s, pi4->jetson %.2f, RTT %.3f ms" % (
                s["id"], lk["directions"]["jetson->pi4"]["mbps_mean"], lk["directions"]["pi4->jetson"]["mbps_mean"], lk["rtt_ms"]))
            if not unshape():
                raise RuntimeError("could not clear the shaping after %s" % s["id"])
        # ---- the guard, the clocks, the poll
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
        say("jetson clocks locked, as serve_pool.sh serves (%s)" % state)
        subprocess.run(["scp", "-q", TPOLL, "pi4:throttle_poll.sh"], check=True)
        rc, p = SS.ssh("pi4", "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_hbs_%s.log > /dev/null 2>&1 "
                              "< /dev/null & echo $!" % stamp)
        tpoll = p.strip()
        # ---- the servers, with serve_pool.sh's command lines
        SS.ssh("jetson", "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H %s -p %d -d Vulkan0 "
                         "> ~/logs/rpc-server_hbs_%s.log 2>&1 < /dev/null &" % (JETSON_IP, RPC_PORT, stamp))
        rpc_up = True
        for _ in range(60):
            rc, o = SS.ssh("jetson", "ss -ltnH 'sport = :%d' | wc -l" % RPC_PORT)
            if o.strip() not in ("", "0"):
                break
            time.sleep(1)
        else:
            say("FATAL: the Jetson's rpc-server never listened"); sys.exit(1)
        SS.ssh("pi4", "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/llama-server %s > ~/logs/llama-server_hbs_%s.log 2>&1 "
                      "< /dev/null &" % (reg["server_args"], stamp))
        srv_up = True
        t_load = time.time()
        while True:
            try:
                if get(reg["host_url"] + "/health").get("status") == "ok":
                    break
            except Exception:
                pass
            if time.time() - t_load > 900 or not pids("pi4", "llama-server", "--port %d" % HTTP_PORT):
                say("FATAL: llama-server on the Pi 4 did not come up"); sys.exit(1)
            time.sleep(2)
        say("servers up in %.0f s: %s" % (time.time() - t_load, reg["server_args"]))
        body = lambda arm, text, n: dict(reg["sampling"], prompt=reg["chat_format"] % text, n_predict=n, **reg["arms"][arm])
        for arm in ("off", "on"):
            r = SS.post(reg["host_url"] + "/completion", body(arm, reg["warmup_prompt"], 16), timeout=600)
            say("warm-up %s: %s tokens, %.1f ms/token" % (arm, r.get("tokens_predicted"),
                (r["timings"]["predicted_ms"] / max(1, r["timings"]["predicted_n"]))))
        # ---- the settings
        for s in settings:
            say("===== setting %s: %s" % (s["id"], shape(s["rate_mbit"])))
            for arm in ("off", "on"):
                SS.post(reg["host_url"] + "/completion", body(arm, reg["warmup_prompt"], 8), timeout=600)
            for rep, arm in enumerate(abba):
                for pr in prompts:
                    a0, r0 = jetson_bytes()
                    t0 = time.time()
                    try:
                        resp = SS.post(reg["host_url"] + "/completion", body(arm, pr["text"], n_predict), timeout=900)
                    except Exception as e:
                        say("!!!!! %s %s %s failed: %s" % (s["id"], arm, pr["id"], e)); continue
                    t1 = time.time()
                    a1, r1 = jetson_bytes()
                    tm = resp.get("timings") or {}
                    rec({"setting": s["id"], "arm": arm, "rep": rep, "id": pr["id"], "t0": t0, "t1": t1,
                         "jetson_sent_bytes": a1 - a0, "jetson_received_bytes": r1 - r0, "tokens_predicted": resp.get("tokens_predicted"),
                         "timings": tm, "content": resp.get("content")})
                    say("%-4s %-3s %-5s %3s tokens, %7.2f ms/token, the Jetson sent %.1f KB per token" % (
                        s["id"], arm, pr["id"], resp.get("tokens_predicted"), (tm.get("predicted_ms") or 0) / max(1, tm.get("predicted_n") or 1),
                        (a1 - a0) / 1024.0 / max(1, tm.get("predicted_n") or 1)))
            if s["rate_mbit"] and not unshape():
                say("FATAL: could not clear the shaping after %s" % s["id"]); sys.exit(1)
    finally:
        def safe(what, fn):
            try:
                fn()
            except Exception as e:                          # one failed step must not skip the others
                say("!!!!! cleanup step %s failed: %s" % (what, e))

        def stop_srv():
            for p in pids("pi4", "llama-server", "--port %d" % HTTP_PORT):
                SS.ssh("pi4", "kill %s" % p)
            subprocess.run(["scp", "-q", "pi4:logs/llama-server_hbs_%s.log" % stamp, os.path.join(d, "server.log")])
            say("llama-server on the Pi 4 stopped")

        def stop_rpc():
            for p in pids("jetson", "ggml-rpc-server", "-H %s -p %d " % (JETSON_IP, RPC_PORT)):
                SS.ssh("jetson", "kill %s" % p)
            say("rpc-server on the Jetson stopped")

        def restore():
            ok, final, said = PC.set_mode("unlocked", stored, asfound)
            say("jetson restored: %s -- %s" % (final, "EQUAL to as found" if final == asfound else "!!!!! NOT EQUAL to as found " + asfound))
            rec({"restored": final, "asfound": asfound, "equal": final == asfound})

        def stop_tpoll():
            SS.ssh("pi4", "kill %s" % tpoll)
            subprocess.run(["scp", "-q", "pi4:logs/throttle_hbs_%s.log" % stamp, os.path.join(d, "throttle_pi4.log")])

        def guard_tail():
            rc, o = SS.ssh("jetson", "tail -n %d ~/heteroserve/logs/power-guard.log" % (int(time.time() - t_start) + 120))
            open(os.path.join(d, "power_guard_tail.log"), "w").write(o)
            say("guard: %d TRIP lines in the run's window" % sum(1 for l in o.splitlines() if "TRIP" in l))

        def stop_guard():
            rc, o = SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p && echo "guard pid $p '
                                     'stopped"; done; read -t 2 <> <(:); echo "guard procs: $(%s | wc -l)"' % (SS.GUARD_PROCS, SS.GUARD_PROCS))
            say(o.strip().replace("\n", "; "))
        safe("shaping", lambda: say("shaping cleared and checked: %s" % unshape()))
        if srv_up:
            safe("llama-server", stop_srv)
        if rpc_up:
            safe("rpc-server", stop_rpc)
        if stored_ok:
            safe("restore", restore)
        if tpoll:
            safe("throttle poll", stop_tpoll)
        safe("guard log", guard_tail)
        if started_guard:
            safe("guard", stop_guard)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


# ------------------------------------------------------------------ score
def ms_per_token(rows):
    n = sum(r["timings"]["predicted_n"] for r in rows)
    return sum(r["timings"]["predicted_ms"] for r in rows) / n if n else None


def score_rows(reg, rows, goodput):
    """The registered verdicts from request rows and {setting: Jetson -> Pi 4 goodput in Mbit/s}. Pure, for the tests."""
    V, th = reg["V_bytes"], reg["thresholds"]
    f = reg.get("per_token_factor", 1.0)                    # registration b: n-1 passes over n tokens (a: 1)
    want = reg["abba"].count("off") * len(reg["prompts"])
    per, hyp = {}, {}
    for s in reg["settings"]:
        sid = s["id"]
        off = [r for r in rows if r["setting"] == sid and r["arm"] == "off"]
        on = [r for r in rows if r["setting"] == sid and r["arm"] == "on"]
        complete = len(off) == want and len(on) == want
        mo, mn = ms_per_token(off), ms_per_token(on)
        bo = sum(r["jetson_sent_bytes"] for r in off) / sum(r["timings"]["predicted_n"] for r in off) if off else None
        bn = sum(r["jetson_sent_bytes"] for r in on) / sum(r["timings"]["predicted_n"] for r in on) if on else None
        g = goodput.get(sid)
        pred = f * V / (g * 1e6 / 8 / 1000.0) if g else None
        per[sid] = {"complete": complete, "ms_off": mo, "ms_on": mn, "saving_ms": (mo - mn) if (mo and mn) else None,
                    "goodput_mbit": g, "predicted_saving_ms": pred, "jetson_bytes_per_token_off": bo,
                    "jetson_bytes_per_token_on": bn}
        x = per[sid]
        hyp["HB-SAVE-%s" % sid] = ({"verdict": "INCOMPLETE"} if not complete or pred is None or x["saving_ms"] is None else
                                   {"verdict": "HOLDS" if abs(x["saving_ms"] - pred) <= th["save_ms"] else "FAILS",
                                    "measured_ms": round(x["saving_ms"], 3), "predicted_ms": round(pred, 3),
                                    "err_ms": round(x["saving_ms"] - pred, 3)})
    s0 = per.get("s0", {})
    if not s0.get("complete") or s0.get("jetson_bytes_per_token_off") is None:
        hyp["HB-BYTES"] = {"verdict": "INCOMPLETE"}
    else:
        drop = s0["jetson_bytes_per_token_off"] - s0["jetson_bytes_per_token_on"]
        hyp["HB-BYTES"] = {"verdict": "HOLDS" if drop >= th["bytes_fraction"] * V else "FAILS", "drop_bytes_per_token": round(drop),
                           "V_bytes": V, "fraction_of_V": round(drop / V, 4)}
    pts = [(1.0 / (per[s]["goodput_mbit"] * 1e6 / 8 / 1000.0), per[s]["saving_ms"]) for s in per
           if per[s]["complete"] and per[s]["goodput_mbit"] and per[s]["saving_ms"] is not None]
    if len(pts) == len(reg["settings"]) and len(pts) >= 2:
        n = len(pts); mx = sum(p[0] for p in pts) / n; my = sum(p[1] for p in pts) / n
        slope = sum((x - mx) * (y - my) for x, y in pts) / sum((x - mx) ** 2 for x, _ in pts)
        icpt = my - slope * mx
        hyp["HB-SLOPE"] = {"verdict": "HOLDS" if abs(slope / (f * V) - 1) * 100 <= th["slope_pct"] else "FAILS",
                           "slope_bytes": round(slope), "predicted_slope_bytes": round(f * V),
                           "err_pct": round(100 * (slope / (f * V) - 1), 2), "intercept_ms": round(icpt, 3)}
    else:
        hyp["HB-SLOPE"] = {"verdict": "INCOMPLETE"}
    texts = {}
    for r in rows:
        texts.setdefault((r["setting"], r["id"]), {}).setdefault(r["arm"], set()).add(r["content"])
    same = {"%s/%s" % k: (len(v.get("off", ())) == 1 and v.get("off") == v.get("on")) for k, v in texts.items()}
    hyp["HB-SAME"] = {"verdict": "INCOMPLETE" if not same else ("HOLDS" if all(same.values()) else "FAILS"),
                      "per_prompt": same}
    six = [hyp[k]["verdict"] for k in hyp]
    verdict = "INCOMPLETE" if "INCOMPLETE" in six else ("HOLDS" if all(v == "HOLDS" for v in six) else "FAILS")
    works = hyp["HB-BYTES"]["verdict"] == "HOLDS" and hyp["HB-SAME"]["verdict"] == "HOLDS"
    return {"settings": per, "hypotheses": hyp, "works_over_rpc": works, "upgrade6": verdict}


def score(d, reg_path=None):
    import link_to_json as LJ
    reg_path = reg_path or os.path.join(ROOT, "testbed", "dataset", "head_bs_predictions_20261010.json")
    reg = json.load(open(reg_path))
    lines = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    rows = [r for r in lines if r.get("arm") and r.get("timings") and r["timings"].get("predicted_n")]
    goodput = {}
    for s in reg["settings"]:
        f = os.path.join(d, "link_%s.log" % s["id"])
        if os.path.exists(f):
            _, _, lk = LJ.parse(f)
            goodput[s["id"]] = lk["directions"]["jetson->pi4"]["mbps_mean"]
    s = score_rows(reg, rows, goodput)
    restored = [l for l in lines if "restored" in l]
    s = {"registration": os.path.relpath(reg_path, ROOT), "registration_sha256": SS.sha(reg_path), "run": os.path.relpath(d, ROOT),
         "code_changed_since_registration": sorted(p for p, h in reg.get("inputs_sha256", {}).items()
                                                   if os.path.exists(os.path.join(ROOT, p)) and SS.sha(os.path.join(ROOT, p)) != h),
         **s, "jetson_restored_equal": bool(restored) and all(l["equal"] for l in restored)}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: s[k] for k in ("settings", "hypotheses", "works_over_rpc", "upgrade6", "jetson_restored_equal",
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
