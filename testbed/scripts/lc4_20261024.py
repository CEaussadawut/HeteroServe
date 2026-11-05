#!/usr/bin/env python3
"""The long-context study on the four-node cluster (scheduler upgrade 4 again; 2026-10-24), run and scored as registered.

    python3 testbed/scripts/lc4_20261024.py run   testbed/dataset/lc4char_predictions_20261024.json   # step 1
    python3 testbed/scripts/lc4_20261024.py run   testbed/dataset/lc4plan_predictions_20261024.json   # step 2
    python3 testbed/scripts/lc4_20261024.py score testbed/results/lc4<step>_<stamp>

A registration lists "benches": each one llama-bench invocation on one node ("host": laptop, pi5, pi4 or jetson), with its
arguments and, for a pooled bench, the RPC servers it needs ("servers": [{node, ip, port, extra}]); and "order", the
benches' sequence (a list of lists: the benches in an inner list run at the same time, on different nodes that share
nothing). Every bench gets -o jsonl and -r from its own arguments; its raw output is tee'd into the run folder.
Gates before anything starts: the laptop quiet on its charger (as the session scripts), no other llama.cpp anywhere, every
node's tree at the pin and clean, the model's sha256 on every host that loads it (testbed/models.md), the Pis cool. When the
Jetson takes part: its guard started by its NOTE's procedure (or found live and un-tripped), MAXN, and its clocks locked for
the run (stored and restored), as run_pool.sh does. A 1 Hz throttle poll on every Pi that takes part.
score: per bench and depth, ms per decoded token = the mean of the tg samples over n_gen; per (node or plan), the slope
of ms per token against depth by least squares (ms per 1,000 tokens), and the registered hypotheses.
"""
import argparse, datetime, hashlib, json, os, re, shlex, statistics, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=6",
       "-o", "LogLevel=ERROR"]   # 2026-10-24: the laptop's OpenSSH 10.2 warns on stderr (no post-quantum kex) for the Jetson,
                                 # and sh() returns stdout + stderr: lc4char's idle gate read the warning as a running process
LTREE, BTREE = "/opt/heteroserve/llama.cpp", "~/llama.cpp"
APPS = ("zen", "antigravity-ide", "notion-snap-reb")


def sh(node, cmd, timeout=600):
    argv = ["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd]
    p = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    return p.returncode, p.stdout + p.stderr


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def run(regpath):
    reg = json.load(open(regpath))
    if os.access(regpath, os.W_OK):
        raise SystemExit("FATAL: %s is writable (register first)" % regpath)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "%s_%s" % (reg["out_prefix"], stamp))
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## %s: registration %s (%s)" % (reg["out_prefix"], os.path.relpath(regpath, ROOT), sha(regpath)[:16]))
    nodes = sorted({b["host"] for b in reg["benches"].values()} | {s["node"] for b in reg["benches"].values() for s in b.get("servers", [])})
    pin = json.load(open(os.path.join(ROOT, "ThinkPad", "config", "cluster.json")))["llama"]["pin_sha"]
    # ---- gates
    # The laptop's gate waits as the session scripts' quiet() does (2026-10-24: it used to check once and stop, and this
    # step starts ~75 s after the ring-lap study's four-thread benches end): every 30 s, three passes in a row, up to 2 h;
    # then off the charger is fatal, anything else is recorded and the run goes on.
    # 2026-11-04 (the owner: "it's ok with zen open, RAM 8GB is reserved"): QUIET_ALLOW_APPS lets the listed apps stay open,
    # as phase_calib_20261028.laptop_quiet does; unset, every listed app blocks the gate as before. The gate line records them.
    allowed = set(os.environ.get("QUIET_ALLOW_APPS", "").replace(",", " ").split())
    t0, ok = time.time(), 0
    while True:
        ps = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout.split()
        load = float(open("/proc/loadavg").read().split()[0])
        ac = open("/sys/class/power_supply/AC/online").read().strip()
        charging = any(open(p).read().strip() == "Charging" for p in
                       [os.path.join("/sys/class/power_supply", x, "status") for x in os.listdir("/sys/class/power_supply") if x.startswith("BAT")])
        wifi = subprocess.run(["nmcli", "radio", "wifi"], capture_output=True, text=True).stdout.strip()
        apps = sorted({a for a in ps if a in APPS and a not in allowed})
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
        + ("; open and allowed (QUIET_ALLOW_APPS): %s" % sorted({a for a in ps if a in allowed}) if allowed else ""))
    for n in nodes:
        rc, o = sh(n, "for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
        if o.strip():
            say("FATAL: %s runs %s" % (n, o.strip())); sys.exit(1)
        tree = LTREE if n == "laptop" else BTREE
        rc, o = sh(n, "git -C %s rev-parse HEAD; git -C %s status --porcelain --untracked-files=no | wc -l" % (tree, tree))
        if o.split() != [pin, "0"]:
            say("FATAL: %s's tree is %r, not the pin clean" % (n, o.split())); sys.exit(2)
    for b in reg["benches"].values():
        rc, o = sh(b["host"], "sha256sum %s | cut -d' ' -f1" % b["model_path"])
        if o.strip() != b["model_sha256"]:
            say("FATAL: %s on %s is %r, not %s" % (b["model_path"], b["host"], o.strip(), b["model_sha256"])); sys.exit(3)
    say("gates: every node idle and at the pin %s; the model's sha256 on every host" % pin[:12])
    for n in [x for x in nodes if x in ("pi5", "pi4")]:
        for _ in range(40):
            rc, t = sh(n, "vcgencmd measure_temp | tr -dc '0-9.'")
            if float(t or 99) < reg["cold_c"][n]:
                break
            time.sleep(15)
        say("cold %s at %s C (< %s)" % (n, t, reg["cold_c"][n]))
    started, polls, guard_started, locked = [], {}, False, False
    stored = "~/heteroserve/run/jetson_clocks_%s_%s.stored" % (reg["out_prefix"], stamp)
    try:
        if "jetson" in nodes:
            rc, g = sh("jetson", 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l')
            if g.strip() == "0":
                rc, c = sh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
                if c.split() != ["synced", sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
                    say("FATAL: Jetson guard precheck %r" % c.split()); sys.exit(4)
                sh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
                guard_started = True; time.sleep(5)
            rc, g = sh("jetson", 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); G=~/heteroserve/logs/power-guard.log; '
                                 't=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); echo "$m $t"')
            if g.split() != ["pmode:0000", "0"]:
                say("FATAL: Jetson not MAXN with an un-tripped guard: %r" % g); sys.exit(4)
            rc, o = sh("jetson", "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; "
                                 "if [ $(cat $c/scaling_min_freq) = $(cat $c/scaling_max_freq) ] && [ $(cat $g/min_freq) = $(cat $g/max_freq) ]; "
                                 "then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store %s && sudo -n jetson_clocks && echo LOCKED; fi" % stored)
            locked = o.strip() == "LOCKED"
            if o.strip() not in ("LOCKED", "ALREADY_LOCKED"):
                say("FATAL: could not lock the Jetson's clocks: %r" % o); sys.exit(4)
            say("jetson guard %s, clocks %s" % ("started" if guard_started else "found live", o.strip()))
        for n in [x for x in nodes if x in ("pi5", "pi4")]:
            subprocess.run(["scp", "-q", os.path.join(HERE, "throttle_poll.sh"), "%s:throttle_poll.sh" % n], check=True)
            rc, p = sh(n, "mkdir -p ~/logs; setsid nohup bash ~/throttle_poll.sh ~/logs/throttle_%s_%s.log > /dev/null 2>&1 < /dev/null & echo $!" % (reg["out_prefix"], stamp))
            polls[n] = p.strip()
        for label in [x for group in reg["order"] for x in group]:
            for s in reg["benches"][label].get("servers", []):
                key = (s["node"], s["ip"], s["port"])
                if key in started:
                    continue
                tree = LTREE if s["node"] == "laptop" else BTREE
                sh(s["node"], "mkdir -p ~/logs; setsid nohup %s/build/bin/ggml-rpc-server -H %s -p %d %s > ~/logs/rpc-server_%s_%s_%d.log 2>&1 < /dev/null &"
                   % (tree, s["ip"], s["port"], s.get("extra", ""), reg["out_prefix"], stamp, s["port"]))
                started.append(key)
                for _ in range(30):
                    if sh(s["node"], "ss -ltn | grep -q ' %s:%d '" % (s["ip"], s["port"]))[0] == 0:
                        break
                    time.sleep(1)
                else:
                    say("FATAL: server %s never listened" % (key,)); sys.exit(5)
                say("server %s %s:%d listening" % key)
        for gi, group in enumerate(reg["order"], 1):
            threads, results = [], {}

            def one(label):
                b = reg["benches"][label]
                tree = LTREE if b["host"] == "laptop" else BTREE
                cmd = "cd ~ && /usr/bin/timeout %d %s/build/bin/llama-bench -m %s %s -o jsonl; echo EXIT_BENCH=$?" % (
                    b.get("timeout_s", 5400), tree, b["model_path"], b["args"])
                t0 = time.time()
                rc, out = sh(b["host"], cmd, timeout=b.get("timeout_s", 5400) + 120)
                results[label] = (t0, time.time(), out)
            say("group %d: %s" % (gi, " ".join(group)))
            for label in group:
                th = threading.Thread(target=one, args=(label,)); th.start(); threads.append(th)
            for th in threads:
                th.join()
            for label in group:
                t0, t1, out = results[label]
                open(os.path.join(d, "bench_%d_%s.log" % (gi, label)), "w").write(out)
                ok = "EXIT_BENCH=0" in out
                say("##### BENCH group=%d label=%s start=%.3f end=%.3f %s" % (gi, label, t0, t1, "ok" if ok else "FAILED"))
    finally:
        for key in started:
            n, ip, port = key
            sh(n, "for p in $(pgrep -x ggml-rpc-server); do tr '\\0' ' ' < /proc/$p/cmdline | grep -qF -- '-H %s -p %d ' && kill $p; done" % (ip, port))
        for n, p in polls.items():
            sh(n, "kill %s 2>/dev/null" % p)
            subprocess.run(["scp", "-q", "%s:logs/throttle_%s_%s.log" % (n, reg["out_prefix"], stamp), os.path.join(d, "throttle_%s.log" % n)])
        if locked:
            rc, o = sh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); [ \"$m\" = pmode:0000 ] && sudo -n jetson_clocks --restore %s && echo RESTORED" % stored)
            say("jetson clocks %s" % o.strip())
        if guard_started:
            sh("jetson", 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p; done')
            say("jetson guard stopped")
        for n in [x for x in nodes if x in ("pi5", "pi4")]:
            rc, o = sh(n, "vcgencmd get_throttled; vcgencmd measure_temp")
            say("after %s: %s" % (n, o.replace("\n", " ")))
    say("RUN DONE %s" % os.path.relpath(d, ROOT))
    score(d)


def tg_by_depth(path):
    out = {}
    for ln in open(path, errors="replace"):
        if ln.startswith('{"build_commit"') or ln.startswith('{"build'):
            x = json.loads(ln)
            if x.get("n_gen", 0) > 0 and x.get("n_prompt", 0) == 0:
                out[int(x.get("n_depth", 0))] = statistics.fmean(s / 1e6 for s in x["samples_ns"]) / x["n_gen"]
    return out


def slope(points):
    xs, ys = zip(*points)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def score(d):
    text = open(os.path.join(d, "run.log")).read()
    regrel = re.search(r"registration (\S+) \(", text).group(1)
    reg = json.load(open(os.path.join(ROOT, regrel)))
    benches = {}
    for m in re.finditer(r"##### BENCH group=(\d+) label=(\S+) .* (ok|FAILED)$", text, re.M):
        g, label, ok = m.groups()
        tg = tg_by_depth(os.path.join(d, "bench_%s_%s.log" % (g, label))) if ok == "ok" else {}
        benches.setdefault(label, []).append(tg)
    res = {}
    for label, runs in benches.items():
        depths = sorted({k for r in runs for k in r})
        mean = {k: round(statistics.fmean(r[k] for r in runs if k in r), 3) for k in depths}
        pts = [(k, v) for r in runs for k, v in r.items()]
        res[label] = {"runs": len(runs), "ms_per_token_by_depth": mean,
                      "slope_ms_per_1k": round(1000 * slope(pts), 3) if len({k for k, _ in pts}) > 1 else None}
    hyp = {}
    for name, h in reg["hypotheses"].items():
        kind = h["type"]
        try:
            if kind == "slope_within":         # each label's slope within tol of its predicted slope
                err = {l: res[l]["slope_ms_per_1k"] / reg["benches"][l]["predicted"]["slope_ms_per_1k"] - 1 for l in h["labels"]}
                hyp[name] = {"verdict": "HOLDS" if all(abs(e) <= h["tol"] for e in err.values()) else "FAILS",
                             "err_pct": {l: round(100 * e, 1) for l, e in err.items()}}
            elif kind == "faster_at":          # label a faster than b at depth
                a, b, dp = h["a"], h["b"], str(h["depth"])
                va, vb = res[a]["ms_per_token_by_depth"][int(dp)], res[b]["ms_per_token_by_depth"][int(dp)]
                hyp[name] = {"verdict": "HOLDS" if va < vb else "FAILS", a: va, b: vb, "depth": h["depth"]}
            elif kind == "acc_at":             # every label within tol of its prediction at every registered depth
                err = {}
                for l in h["labels"]:
                    for dp, pv in reg["benches"][l]["predicted"]["ms_per_token_by_depth"].items():
                        err["%s@%s" % (l, dp)] = res[l]["ms_per_token_by_depth"][int(dp)] / pv - 1
                hyp[name] = {"verdict": "HOLDS" if all(abs(e) <= h["tol"] for e in err.values()) else "FAILS",
                             "err_pct": {k: round(100 * e, 1) for k, e in err.items()}}
            elif kind == "slope_near":         # one label's slope within tol of a value the hypothesis states (2026-10-24, lc4j:
                v = res[h["label"]]["slope_ms_per_1k"] / h["value"] - 1    # two competing laws registered against one bench)
                hyp[name] = {"verdict": "HOLDS" if abs(v) <= h["tol"] else "FAILS", "slope_ms_per_1k": res[h["label"]]["slope_ms_per_1k"],
                             "value": h["value"], "err_pct": round(100 * v, 1)}
            else:
                raise SystemExit("unknown hypothesis type %s" % kind)
        except (KeyError, TypeError, ZeroDivisionError) as e:
            hyp[name] = {"verdict": "INCOMPLETE", "why": repr(e)}
    s = {"run": os.path.relpath(d, ROOT), "registration": regrel, "registration_sha256": sha(os.path.join(ROOT, regrel)),
         "scored": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
         "code_changed_since_registration": sorted(f for f, hh in reg["inputs_sha256"].items() if f.endswith((".py", ".sh"))
                                                   and sha(os.path.join(ROOT, f)) != hh),
         "benches": res, "hypotheses": hyp}
    json.dump(s, open(os.path.join(d, "score.json"), "w"), indent=1)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "score")); ap.add_argument("path")
    a = ap.parse_args()
    run(a.path) if a.cmd == "run" else score(a.path)
