#!/usr/bin/env python3
"""Scheduler upgrade 5, step 1 (2026-10-10): the Jetson's power modes, characterised on the 14B's own 13-layer block,
with the laptop standing in for the Pi 4 as the host. Its numbers feed register_power_modes_20261010.py; nothing here
is scored.

    python3 testbed/scripts/power_modes_char_20261010.py run [--smoke]
    python3 testbed/scripts/power_modes_char_20261010.py analyse testbed/results/pm_char_<stamp>

Why the laptop. In the served plan (Qwen3-14B, Pi 4 27 + Jetson 13, head on the Pi 4) the Jetson waits ~1.55 s while
the Pi 4 computes, then runs its 13 layers (~0.35 s). This run measures how each mode changes that cycle (the wake-up,
the speed, the power) on the same block, but with a different host and a wait it controls. The served plan then stays
an out-of-sample test.

The three modes (the survey's upgrade 5):
  locked    MAXN plus jetson_clocks: CPU 1479 MHz, GPU 921.6 MHz, EMC 1600 MHz, cpuidle and GPU railgating off. Every
            run since 09-30 uses it.
  unlocked  MAXN with the governors free: nvhost_podgov on the GPU, schedutil on the CPU, the EMC scaled by its activity
            monitor, and the GPU railgated after 500 ms idle. This is the "low-power idle".
  5w        nvpmodel 5 W (cores 2-3 offline, the CPU capped at 921.6 MHz and the GPU at 614.4 MHz: the frequency
            table's steps around the mode's 918 and 640 MHz caps), then jetson_clocks, which locks the clocks at those
            maxima.
The modes are switched live with the block loaded (sudo -n, as the guard and serve_pool.sh do). Each switch passes
through the as-found state, and is verified from sysfs afterwards. The as-found state (MAXN, governors free) is stored
first with jetson_clocks --store. On every way out it is restored, then checked equal to the snapshot.

The instrument:
- the Jetson's ggml-rpc-server, exactly as serve_pool.sh starts it (-d Vulkan0, port 50052);
- llama-server on the laptop (the pinned build, CPU only, -t 4) with the served arguments: --rpc <jetson> -ngl 14
  -ot "^output\\.weight=CPU" -c 1024 -np 1 -lm none. The laptop holds layers 0-26 and the head, the Jetson 27-39;
- one "token" is one request of n_predict 1 on a cached prompt, so one forward pass. prompt_n 1 is checked;
- after each reply the client sleeps G seconds. The Jetson's idle gap is then G plus the laptop's part, the same in
  every mode, just as the Pi 4's part is the same in every mode in the served plan;
- each cell opens with one uncached request of the whole prompt (a prefill burst, as every served request begins), then
  64 forward passes, as a served request's 64 decoded tokens. The smoke found the GPU's governor holding whatever
  clock the last work left it at, so a cell must enter from the state a served decode starts in.
Per mode block: 3 warm-up requests, an idle window with the block loaded (the waiting power), then the gaps in order.
The blocks run L U 5 5 U L, and each mode's second block takes the gaps in reverse, so a linear drift cancels.
Logged:
- per request, the send and reply times and the server's timings;
- the strip at 5 Hz (the Jetson's outlet is the measure);
- the Jetson's INA3221, through the guard's log (1 Hz);
- its clocks at 10 Hz (jetson_clock_poll.sh, as root).
--smoke: one block per mode, gaps 0 and 1.3 s, 3 requests each, idle 10 s, into pm_char_smoke_<stamp>/.
The sweep was 6 gaps x 30 requests in the first smoke (16:40); it became 4 gaps x 64 after it, for the reason above.

analyse: per mode and gap, pooled over the mode's blocks:
- the median forward pass (prompt_ms) and the median period between sends;
- the Jetson's mean outlet power over the cell (strip trapezoid over [the third send, last reply + G]: the strip
  refreshes every 1-2 s and lags, so the first two requests are skipped) and its mean INA3221 power over the same;
- the GPU's mean clock and the share of polls that found it railgated.
Per mode, the idle window's power. Each figure is also given per block, so the spread between blocks shows. Writes
<dir>/analysis.json.
"""
import datetime, json, os, signal, statistics, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import spec_serve_20261010 as SS                            # noqa: E402  ssh, post, sha, GUARD_PROCS

STRIP = os.path.join(HERE, "shelly_strip.py")
POLL = os.path.join(HERE, "jetson_clock_poll.sh")
GUARD = os.path.join(HERE, "jetson_power_guard.sh")
SERVE_STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
LLAMA = "/opt/heteroserve/llama.cpp/build/bin/llama-server"
LLAMA_TREE = "/opt/heteroserve/llama.cpp"
PIN = "b23701f77d47dad9de834d59ebfcbe25c9e8b46f"
MODEL = "/opt/heteroserve/models/qwen3-14b-q4km.gguf"
MODEL_SHA = "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"     # testbed/models.md
JETSON_IP, RPC_PORT, HTTP_PORT = "192.168.1.114", 50052, 8091
URL = "http://127.0.0.1:%d" % HTTP_PORT
SERVER_ARGS = ["--rpc", "%s:%d" % (JETSON_IP, RPC_PORT), "-ngl", "14", "-ot", r"^output\.weight=CPU", "-c", "1024", "-np", "1",
               "-t", "4", "-lv", "4", "-lm", "none", "--host", "127.0.0.1", "--port", str(HTTP_PORT)]
OUTLET_JETSON = 2                                           # the strip: pi4 0, laptop 1, jetson 2 (3 is the router)
MODES = ("locked", "unlocked", "5w")
ORDER = ["locked", "unlocked", "5w", "5w", "unlocked", "locked"]
GAPS = [0.0, 1.1, 1.3, 1.5]
N_REQ, N_WARM, IDLE_S = 64, 3, 45
CHAT = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROMPT = CHAT % "Explain how a refrigerator keeps food cold."
BODY = {"prompt": PROMPT, "n_predict": 1, "temperature": 0, "top_k": 1, "cache_prompt": True}
STATE_CMD = ('m=$(cut -d" " -f1 /var/lib/nvpmodel/status); c=/sys/devices/system/cpu/cpu0/cpufreq; '
             'g=/sys/devices/gpu.0/devfreq/57000000.gpu; echo "pmode=$m online=$(cat /sys/devices/system/cpu/online) '
             'cpu=$(cat $c/scaling_min_freq)-$(cat $c/scaling_max_freq) gpu=$(cat $g/min_freq)-$(cat $g/max_freq) '
             'railgate=$(cat /sys/devices/gpu.0/railgate_enable) emcov=$(sudo -n cat /sys/kernel/debug/clk/override.emc/clk_state) '
             'cpuidle=$(cat /sys/devices/system/cpu/cpu0/cpuidle/state*/disable | tr -d "\\n")"')
RPC_PIDS = ("for p in $(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/$p/cmdline | grep -qF -- '-H %s -p %d ' && echo $p; done"
            % (JETSON_IP, RPC_PORT))
POLL_PROCS = 'grep -a -l -s "[j]etson_clock_poll" /proc/[0-9]*/cmdline'


def parse_state(s):
    return dict(x.split("=", 1) for x in s.split())


def mode_ok(mode, state, asfound):
    """True when the sysfs state is the mode's. jetson_clocks (L4T 32.7.6) pins the CPU and GPU minimum to their
    maximum, sets railgate_enable 0, turns the EMC override on and disables cpuidle; 5 W caps the CPU at 921.6 MHz (the
    table's step above 918) and the GPU at 614.4 MHz (the step below 640). Checked by hand 2026-10-10 16:45."""
    k = parse_state(state)
    cpu, gpu = k["cpu"].split("-"), k["gpu"].split("-")
    pinned = k["railgate"] == "0" and k["emcov"] == "1" and k["cpuidle"] == "11"
    if mode == "unlocked":
        return state == asfound
    if mode == "locked":
        return k["pmode"] == "pmode:0000" and k["online"] == "0-3" and cpu == ["1479000"] * 2 and gpu == ["921600000"] * 2 and pinned
    if mode == "5w":
        return (k["pmode"] == "pmode:0001" and k["online"] == "0-1" and cpu[0] == cpu[1] and int(cpu[0]) <= 921600
                and gpu[0] == gpu[1] and int(gpu[0]) <= 640000000 and pinned)
    raise ValueError(mode)


def mode_cmd(mode, stored):
    """The switch, always through the as-found state: MAXN (if the mode is not), then the stored clocks restored."""
    base = ('m=$(cut -d" " -f1 /var/lib/nvpmodel/status); if [ "$m" != pmode:0000 ]; then sudo -n nvpmodel -m 0 </dev/null; fi; '
            'sudo -n jetson_clocks --restore %s' % stored)
    return {"unlocked": base, "locked": base + "; sudo -n jetson_clocks",
            "5w": base + "; sudo -n nvpmodel -m 1 </dev/null; sudo -n jetson_clocks"}[mode]


def set_mode(mode, stored, asfound):
    """(ok, sysfs state line, the switch's own output)."""
    rc, out = SS.ssh("jetson", "{ %s; } 2>&1 | tail -5; echo @@STATE; %s" % (mode_cmd(mode, stored), STATE_CMD), timeout=120)
    said, _, state = out.partition("@@STATE")
    state = state.strip()
    return (mode_ok(mode, state, asfound) if state else False), state, said.strip()


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def run(smoke=False):
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", ("pm_char_smoke_%s" if smoke else "pm_char_%s") % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")
    t_start = time.time()

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    order = list(MODES) if smoke else ORDER
    gaps = [0.0, 1.3] if smoke else GAPS
    n_req, idle_s = (3, 10) if smoke else (N_REQ, IDLE_S)
    say("power-mode characterisation%s: order %s, gaps %s, %d requests per cell, idle %d s" % (
        " SMOKE" if smoke else "", " ".join(order), gaps, n_req, idle_s))
    # ---- prechecks, nothing changed yet
    if os.path.exists(SERVE_STATE):
        say("FATAL: serve_pool.sh state exists; stop it first"); sys.exit(1)
    if (" 127.0.0.1:%d " % HTTP_PORT) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
        say("FATAL: port %d is taken on the laptop" % HTTP_PORT); sys.exit(1)
    head = subprocess.run(["git", "-C", LLAMA_TREE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", LLAMA_TREE, "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout
    if head != PIN or dirty.strip():
        say("FATAL: the laptop's llama.cpp is %s%s, not the pin clean" % (head[:12], " (dirty)" if dirty.strip() else "")); sys.exit(1)
    rc, b = SS.ssh("jetson", 'for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf "%%s " $p; done; '
                             '%s | wc -l; awk "/MemAvailable/{print int(\\$2/1024)}" /proc/meminfo' % POLL_PROCS)
    parts = b.split()
    if len(parts) != 2 or parts[0] != "0" or int(parts[1]) < 3000:
        say("FATAL: the Jetson is busy, polled, or short of memory: %r" % b); sys.exit(1)
    rc, asfound = SS.ssh("jetson", STATE_CMD)
    asfound = asfound.strip()
    k = parse_state(asfound)
    if not (k["pmode"] == "pmode:0000" and k["online"] == "0-3" and k["gpu"].split("-")[0] != k["gpu"].split("-")[1]
            and k["railgate"] == "1" and k["emcov"] == "0" and k["cpuidle"] == "00"):
        say("FATAL: the Jetson is not as found (MAXN, governors free): %s" % asfound); sys.exit(1)
    say("jetson as found: %s; MemAvailable %s MiB" % (asfound, parts[1]))
    got = SS.sha(MODEL)
    if got != MODEL_SHA:
        say("FATAL: %s sha256 %s, pinned %s" % (MODEL, got[:16], MODEL_SHA[:16])); sys.exit(1)
    say("model sha256 OK on the laptop (%s)" % MODEL_SHA[:16])
    rc, g = SS.ssh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum '
                             '~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
    if g.split() != ["synced", SS.sha(GUARD)]:
        say("FATAL: guard precheck %r" % g); sys.exit(1)
    out = open(os.path.join(d, "run.jsonl"), "a")
    rec = lambda o: (out.write(json.dumps(o, ensure_ascii=False) + "\n"), out.flush())
    stored = "~/heteroserve/run/jetson_clocks_pm_%s.stored" % stamp
    started_guard, logger, server, rpc_up, poll_up, stored_ok = False, None, None, False, False, False
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
        subprocess.run(["scp", "-q", POLL, "jetson:heteroserve/tools/jetson_clock_poll.sh"], check=True)
        SS.ssh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_clock_poll.sh ~/heteroserve/logs/clockpoll_%s.log "
                         "</dev/null >/dev/null 2>&1 & echo started" % stamp)
        poll_up = True
        SS.ssh("jetson", "mkdir -p ~/logs; setsid nohup ~/llama.cpp/build/bin/ggml-rpc-server -H %s -p %d -d Vulkan0 "
                         "> ~/logs/rpc-server_pm_%s.log 2>&1 < /dev/null &" % (JETSON_IP, RPC_PORT, stamp))
        rpc_up = True
        for _ in range(60):
            rc, o = SS.ssh("jetson", "ss -ltnH 'sport = :%d' | wc -l" % RPC_PORT)
            if o.strip() not in ("", "0"):
                break
            time.sleep(1)
        else:
            say("FATAL: the Jetson's rpc-server never listened"); sys.exit(1)
        say("strip logger pid %d; clock poll and rpc-server up on the Jetson" % logger.pid)
        t_i0 = time.time(); time.sleep(idle_s); t_i1 = time.time()
        rec({"idle_before": [t_i0, t_i1], "mode": "unlocked", "state": asfound})
        say("idle window before the load (as found): %d s" % idle_s)
        server = subprocess.Popen([LLAMA, "-m", MODEL] + SERVER_ARGS, stdout=open(os.path.join(d, "server.log"), "w"),
                                  stderr=subprocess.STDOUT, start_new_session=True)
        t_load = time.time()
        while True:
            try:
                if get(URL + "/health").get("status") == "ok":
                    break
            except Exception:
                pass
            if server.poll() is not None or time.time() - t_load > 1200:
                say("FATAL: llama-server did not come up (exit %s)" % server.poll()); sys.exit(1)
            time.sleep(2)
        say("llama-server ready on the laptop in %.0f s" % (time.time() - t_load))
        first = SS.post(URL + "/completion", dict(BODY, n_predict=8, cache_prompt=True), timeout=600)
        say("first request (full prompt, compiles the Jetson's pipelines): prompt_n %s" % (first.get("timings") or {}).get("prompt_n"))
        for bi, mode in enumerate(order):
            ok, state, said = set_mode(mode, stored, asfound)
            rec({"block": bi, "mode": mode, "switch": state, "said": said, "ok": ok, "t": time.time()})
            if not ok:
                say("FATAL: block %d: the Jetson is not in %s after the switch: %s" % (bi, mode, state)); sys.exit(1)
            say("===== block %d: %s (%s)" % (bi, mode, state))
            for _ in range(N_WARM):
                SS.post(URL + "/completion", BODY, timeout=600)
            t_i0 = time.time(); time.sleep(idle_s); t_i1 = time.time()
            rec({"block": bi, "mode": mode, "idle": [t_i0, t_i1]})
            seq = gaps if order[:bi].count(mode) % 2 == 0 else list(reversed(gaps))
            for gap in seq:
                # each cell opens as each served request does, with a prefill burst (the prompt, uncached), so the GPU's
                # governor enters every cell from the state a served decode starts in
                t0 = time.time()
                r = SS.post(URL + "/completion", dict(BODY, cache_prompt=False), timeout=600)
                tm = r.get("timings") or {}
                rec({"block": bi, "mode": mode, "gap": gap, "burst": True, "t0": t0, "t1": time.time(), "prompt_n": tm.get("prompt_n"),
                     "prompt_ms": tm.get("prompt_ms")})
                times = []
                for j in range(n_req):
                    t0 = time.time()
                    r = SS.post(URL + "/completion", BODY, timeout=600)
                    t1 = time.time()
                    tm = r.get("timings") or {}
                    rec({"block": bi, "mode": mode, "gap": gap, "k": j, "t0": t0, "t1": t1, "prompt_n": tm.get("prompt_n"),
                         "prompt_ms": tm.get("prompt_ms"), "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms")})
                    times.append(tm.get("prompt_ms") or float("nan"))
                    time.sleep(gap)
                say("block %d %-8s gap %.1f s: forward pass median %.1f ms (min %.1f, max %.1f)" % (
                    bi, mode, gap, statistics.median(times), min(times), max(times)))
        ok, state, said = set_mode("unlocked", stored, asfound)
        say("back to as found: %s (%s)" % (ok, state))
        t_i0 = time.time(); time.sleep(idle_s); t_i1 = time.time()
        rec({"idle_after": [t_i0, t_i1], "mode": "unlocked"})
    finally:
        def safe(what, fn):
            try:
                fn()
            except Exception as e:                          # one failed step must not skip the others
                say("!!!!! cleanup step %s failed: %s" % (what, e))

        def stop_server():
            os.killpg(server.pid, signal.SIGTERM)
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
            say("llama-server stopped")

        def stop_rpc():
            rc, o = SS.ssh("jetson", "for p in $(%s); do kill $p && echo $p; done" % RPC_PIDS)
            say("rpc-server stopped: %s" % o.strip().replace("\n", " "))

        def restore():
            ok, final, said = set_mode("unlocked", stored, asfound)
            say("jetson restored: %s -- %s" % (final, "EQUAL to as found" if final == asfound else "!!!!! NOT EQUAL to as found " + asfound))
            rec({"restored": final, "asfound": asfound, "equal": final == asfound})

        def stop_poll():
            SS.ssh("jetson", 'for p in $(%s | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p; done' % POLL_PROCS)
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
        if server:
            safe("llama-server", stop_server)
        if rpc_up:
            safe("rpc-server", stop_rpc)
        if stored_ok:
            safe("restore", restore)
        if poll_up:
            safe("poll", stop_poll)
        if logger:
            safe("strip logger", stop_logger)
        safe("guard log", guard_tail)
        if started_guard:
            safe("guard", stop_guard)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


# ------------------------------------------------------------------ analyse
def strip_mean(d, t0, t1, idle):
    p = subprocess.run([sys.executable, STRIP, "summarise", os.path.join(d, "strip.csv"), "--window", "%.6f" % t0, "%.6f" % t1,
                        "--idle", "%.6f" % idle[0], "%.6f" % idle[1], "--tokens", "1", "--json"], capture_output=True, text=True)
    o = json.loads(p.stdout)["outlets"][str(OUTLET_JETSON)]
    return {"mean_w": o["mean_w"], "energy_trap_j": o["energy_trap_j"], "window_s": o["window_s"], "fresh_apower_hz": o["fresh_apower_hz"]}


def guard_rows(d, day):
    """[(epoch, P mW)] from the guard log's tail (local time of day, the run's date)."""
    out = []
    p = os.path.join(d, "power_guard_tail.log")
    if not os.path.exists(p):
        return out
    for ln in open(p, errors="replace"):
        f = ln.split()
        if len(f) >= 4 and f[1].startswith("V=") and f[3].startswith("P="):
            t = datetime.datetime.strptime(day + " " + f[0], "%Y-%m-%d %H:%M:%S").astimezone().timestamp()
            out.append((t, int(f[3][2:])))
    return out


def poll_rows(d):
    """[(epoch, gpu Hz, load, cpu kHz, emc Hz, online, runtime)] from the clock poll."""
    p = os.path.join(d, "clockpoll.log")
    if not os.path.exists(p):
        return []
    rows, anchor = [], None
    for ln in open(p, errors="replace"):
        f = ln.split()
        if f and f[0] == "anchor":
            anchor = float(f[1]) - float(f[2])
        elif anchor is not None and len(f) == 7:
            rows.append((anchor + float(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), f[5], f[6]))
    return rows


def p10(xs):
    """The lower decile. Contention on the laptop only ever lengthens a forward pass, so the lower decile estimates the
    undisturbed pass; with the Jetson's clocks fixed its own part hardly varies."""
    return statistics.quantiles(xs, n=10)[0] if len(xs) >= 10 else min(xs)


def within(rows, t0, t1):
    return [r for r in rows if t0 <= r[0] <= t1]


def analyse(d):
    lines = [json.loads(l) for l in open(os.path.join(d, "run.jsonl"))]
    idle0 = next(l["idle_before"] for l in lines if "idle_before" in l)
    allreq = [l for l in lines if "gap" in l and not l.get("burst")]
    day = datetime.datetime.fromtimestamp(idle0[0]).astimezone().strftime("%Y-%m-%d")
    guard, poll = guard_rows(d, day), poll_rows(d)
    bad = [r for r in allreq if r["prompt_n"] != 1]           # not one forward pass: left out of every figure
    reqs = [r for r in allreq if r["prompt_n"] == 1]
    res = {"run": os.path.relpath(d, ROOT), "requests": len(allreq), "prompt_n_not_1": len(bad),
           "prompt_n_not_1_where": sorted({"%s@%.1f block %d k %d (prompt_n %s)" % (r["mode"], r["gap"], r["block"], r["k"], r["prompt_n"])
                                          for r in bad}),
           "cells": {}, "idle": {}, "note": "characterisation, not scored: read by register_power_modes_20261010.py"}

    def power(t0, t1):
        s = strip_mean(d, t0, t1, idle0)
        g = [p for t, p in guard if t0 <= t <= t1]
        pl = within(poll, t0, t1)
        return {"strip_w": s["mean_w"], "strip_j": s["energy_trap_j"], "window_s": s["window_s"],
                "strip_fresh_hz": s["fresh_apower_hz"], "ina_w": statistics.fmean(g) / 1000 if g else None,
                "gpu_mhz_mean": statistics.fmean(r[1] for r in pl) / 1e6 if pl else None,
                "emc_mhz_mean": statistics.fmean(r[4] for r in pl) / 1e6 if pl else None,
                "railgated_share": sum(1 for r in pl if r[6] == "suspended") / len(pl) if pl else None,
                "polls": len(pl)}
    for l in lines:
        if "idle" in l and "block" in l:                    # 5 s after the warm-up are skipped, as the strip lags
            res["idle"].setdefault(l["mode"], []).append(dict(power(l["idle"][0] + 5, l["idle"][1]), block=l["block"]))
    res["idle"]["before_load"] = [power(*idle0)]
    for mode in MODES:
        for gap in sorted({r["gap"] for r in reqs}):
            cell = [r for r in reqs if r["mode"] == mode and r["gap"] == gap]
            if not cell:
                continue
            blocks = []
            for b in sorted({r["block"] for r in cell}):
                rs = sorted((r for r in cell if r["block"] == b), key=lambda r: r["t0"])
                fw = [r["prompt_ms"] for r in rs]
                per = [b2["t0"] - a["t0"] for a, b2 in zip(rs, rs[1:])]
                # the strip refreshes its reading every 1-2 s and lags: the cell's power skips its first two requests
                pw = power(rs[min(2, len(rs) - 1)]["t0"], rs[-1]["t1"] + gap)
                blocks.append(dict(pw, block=b, n=len(rs), forward_ms_median=statistics.median(fw),
                                   forward_ms_mean=statistics.fmean(fw), forward_ms_p10=p10(fw),
                                   period_ms_median=1000 * statistics.median(per) if per else None,
                                   # the power window holds the passes k = 2..n-1, each with its gap: its mean cycle
                                   period_power_ms=1000 * pw["window_s"] / max(1, len(rs) - 2)))
            fw_all = [r["prompt_ms"] for r in cell]
            tot_s = sum(x["window_s"] for x in blocks)
            m = lambda k: (sum(x[k] * x["window_s"] for x in blocks if x[k] is not None) / tot_s
                           if all(x[k] is not None for x in blocks) else None)
            res["cells"]["%s@%.1f" % (mode, gap)] = {
                "mode": mode, "gap_s": gap, "n": len(cell), "forward_ms_median": statistics.median(fw_all),
                "forward_ms_mean": statistics.fmean(fw_all), "forward_ms_p10": p10(fw_all),
                "period_ms_median": statistics.median([x["period_ms_median"] for x in blocks if x["period_ms_median"]]),
                "strip_w": m("strip_w"), "ina_w": m("ina_w"), "gpu_mhz_mean": m("gpu_mhz_mean"), "emc_mhz_mean": m("emc_mhz_mean"),
                "railgated_share": m("railgated_share"),
                "block_spread_forward_pct": (100 * (max(x["forward_ms_median"] for x in blocks) / min(x["forward_ms_median"] for x in blocks) - 1)
                                             if len(blocks) > 1 else None),
                "block_spread_strip_pct": (100 * (max(x["strip_w"] for x in blocks) / min(x["strip_w"] for x in blocks) - 1)
                                           if len(blocks) > 1 and all(x["strip_w"] for x in blocks) else None),
                "blocks": blocks}
    json.dump(res, open(os.path.join(d, "analysis.json"), "w"), indent=1)
    print("prompt_n other than 1: %d of %d requests (left out)%s" % (len(bad), len(allreq), "".join("\n  " + w for w in res["prompt_n_not_1_where"])))
    print("%-14s %5s %9s %8s %9s %8s %8s %8s %8s %7s %7s" % ("cell", "n", "fwd ms", "fwd p10", "period", "strip W", "INA W",
                                                           "GPU MHz", "EMC MHz", "gated", "spread"))
    for k, c in res["cells"].items():
        f = lambda v, fmt: (fmt % v) if v is not None else "-"
        print("%-14s %5d %9.1f %8.1f %9.1f %8s %8s %8s %8s %7s %7s" % (k, c["n"], c["forward_ms_median"], c["forward_ms_p10"],
              c["period_ms_median"] or 0,
              f(c["strip_w"], "%.3f"), f(c["ina_w"], "%.3f"), f(c["gpu_mhz_mean"], "%.0f"), f(c["emc_mhz_mean"], "%.0f"),
              f(c["railgated_share"], "%.2f"), f(c["block_spread_forward_pct"], "%.1f%%")))
    for k, v in res["idle"].items():
        print("idle %-12s %s" % (k, "; ".join("strip %.3f W, INA %s W, gated %s" % (
            x["strip_w"] or 0, "%.3f" % x["ina_w"] if x["ina_w"] else "-",
            "%.2f" % x["railgated_share"] if x["railgated_share"] is not None else "-") for x in v)))
    print("wrote %s" % os.path.relpath(os.path.join(d, "analysis.json"), ROOT))
    return res


def main(argv):
    if argv and argv[0] == "run":
        run(smoke="--smoke" in argv)
    elif len(argv) >= 2 and argv[0] == "analyse":
        analyse(argv[1])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
