#!/usr/bin/env python3
"""QR3: time per output token (TPOT) at the 95th percentile, for one user, on the engine's serving plan. From the repo root:

    python3 testbed/scripts/qr3_tpot.py run --model qwen3_14b [--ctx 4096] [--mem <memory file>] [--clocks locked|asfound]
                                          [--reqs 10] [--n 128] [--gen 128] [--probe] [--tag scored]
    python3 testbed/scripts/qr3_tpot.py summary testbed/results/qr3_<tag>_<model>_<stamp>.log

The proposal's QR3 is "TPOT P95 <= 500 ms (MVP) / <= 250 ms (Stretch)", TPOT taken as the decode streams tokens back to the
client. Here:
- serve_pool.sh plans the model with the engine for a --ctx context, gates every board's free memory and the Jetson's guard,
  starts ggml-rpc-server on the workers and llama-server on the host (-np 1 -t 4), and stops exactly what it started.
- Before that, the model file is sha256-checked on its host against testbed/models.md.
- One warm-up request (a 512-token prompt, 32 tokens; logged, never scored), then --reqs requests one after another, each a
  different --n-token prompt (the GPL-3 text from its own offset, then a question) wanting --gen tokens: greedy, ignore_eos,
  cache_prompt false (every request prefills its whole prompt), streamed, with timings_per_token.
- Every token is timestamped twice:
    client  time.perf_counter() on the laptop when its SSE event arrives (what a user on the router LAN sees);
    server  llama-server's timings.predicted_ms (its own clock, stamped right after each token is sampled, in us).
  A TPOT sample is the gap between two consecutive tokens of one request; the first token belongs to TTFT. An event that
  carries more than one token (text held back) gives no sample for the gap it merges; such events are counted.
- Between requests: get_throttled, temperature and MemAvailable on each Pi of the plan; the Jetson's guard line and
  MemAvailable; the laptop's load. --probe adds memprobe v2 on every board (the pilot's memory check). Scored runs go without
  it, so no probe runs while their tokens are timed.
- Everything is written raw to testbed/results/qr3_<tag>_<model>_<stamp>.log; the host's server log is copied beside it.
"""
import argparse, datetime, hashlib, json, os, random, re, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
LAN = {"pi5": "192.168.100.107", "pi4": "192.168.100.106", "jetson": "192.168.100.111"}   # serve_pool.sh's
HTTP_PORT = 8080                                                                           # serve_pool.sh's
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
MODEL_FILE = {  # predict_pool.MODELS key -> (file on its host, sha256 in testbed/models.md)
    "qwen35_9b": ("~/models/qwen3.5-9b-q4_k_m.gguf", "d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43"),
    "nemo12b": ("~/models/mistral-nemo-12b-q4_k_m.gguf", "7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46"),
    "qwen3_14b": ("~/models/qwen3-14b-q4_k_m.gguf", "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"),
    "qwen3moe": ("~/models/qwen3-30b-a3b-q3_k_m.gguf", "e145c9d2f5d11c9583eb099aa75100b7ab943e77d5240c9a2cd936f81c89ef43"),
}
MODEL_HOST = {"qwen3moe": "pi4"}                   # predict_pool.HOST: the board that holds the file
GPL3 = "/usr/share/common-licenses/GPL-3"          # a fixed public text, cut to exact token counts (as pd_twouser.py)
QUESTION = "\n\nSummarise the text above in three short sentences."
OFFSET = 500                                       # request i's text starts at token i * OFFSET of GPL-3
WARM_N, WARM_GEN = 512, 32                         # the warm-up takes the text's last tokens, disjoint from every request
BOOT, BOOT_SEED = 10000, 20260930
GPU = "/sys/devices/57000000.gpu/devfreq/57000000.gpu"
# --clocks locked (the default): the Jetson's clocks are locked as run_pool.sh locks them for every benchmark cell -- the
# condition of the characterisation rows the predictor is built on -- with run_pool.sh's own commands: stored, locked, and
# restored after, only if this run locked them and the power mode is still MAXN. --clocks asfound leaves them to the
# governors (nvhost_podgov on the GPU, schedutil on the CPUs), as serve_pool.sh does for interactive use. The 01:35 pilot, as
# found, decoded the 14B at ~1,050 ms/token against 878.5 predicted; the GPU idled at 76.8 of 921.6 MHz.
CLOCKS = ("c=/sys/devices/system/cpu/cpu0/cpufreq; g=%s; echo cpu=$(cat $c/scaling_min_freq)-$(cat $c/scaling_max_freq)/"
          "$(cat $c/scaling_cur_freq) gpu=$(cat $g/min_freq)-$(cat $g/max_freq)/$(cat $g/cur_freq) $(cut -d' ' -f1 "
          "/var/lib/nvpmodel/status)" % GPU)


class Log:
    def __init__(self, tag, model):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stamp = stamp
        self.path = os.path.join(ROOT, "testbed", "results", "qr3_%s_%s_%s.log" % (tag, model, stamp))
        self.f = open(self.path, "a")

    def __call__(self, tag, obj):
        line = "%s %s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), tag,
                             obj if isinstance(obj, str) else json.dumps(obj))
        self.f.write(line + "\n"); self.f.flush()
        print(line[:240], flush=True)


def ssh(node, cmd, check=True, timeout=900):
    p = subprocess.run(SSH + [node, cmd], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError("ssh %s failed (%d): %s" % (node, p.returncode, p.stderr.strip()))
    return p.stdout


def health(node):
    if node == "jetson":
        # every thermal zone by name: the Nano's PMIC-Die zone reads a fixed 100 C, so a maximum would mislead
        return ssh(node, "echo guard=\\\"$(tail -1 ~/heteroserve/logs/power-guard.log | cut -c1-110)\\\" "
                         "$(for z in /sys/devices/virtual/thermal/thermal_zone*; do printf '%s=%s ' $(cat $z/type) $(cat $z/temp); done)"
                         "gpu_hz=$(cat " + GPU + "/cur_freq) cpu0_khz=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) "
                         "MemAvailable_kB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)", check=False).strip()
    return ssh(node, "echo $(vcgencmd get_throttled) $(vcgencmd measure_temp) MemAvailable_kB=$(awk '/MemAvailable/ "
                     "{print $2}' /proc/meminfo)", check=False).strip()


def read_state():
    st = {}
    for line in open(STATE):
        m = re.match(r"(\w+)=(.*)$", line.strip())
        if m:
            st[m.group(1)] = subprocess.run(["bash", "-c", "printf '%%s' %s" % m.group(2)], capture_output=True, text=True).stdout
    return st


def serve(action, model, ctx, log, mem=None):
    """serve_pool.sh start|stop, its output into the log as it comes. Returns its exit code. mem: a memory file to plan
    against (serve_pool.sh MEMFILE); None keeps its default, memory_at_rest.json."""
    env = dict(os.environ, MODEL=model, CTX=str(ctx))
    if mem:
        env["MEMFILE"] = mem
    p = subprocess.Popen(["bash", os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh"), action], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True)
    for line in p.stdout:
        log("serve_pool", line.rstrip("\n"))
    return p.wait()


def http(host, path, payload=None, timeout=3600):
    req = urllib.request.Request("http://%s:%d%s" % (LAN[host], HTTP_PORT, path),
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def stream(host, ids, gen):
    """One request, streamed. Returns {"events": [[client_s, tokens, predicted_n, predicted_ms, prompt_ms], ...], ...}, times
    relative to the moment the request was sent."""
    payload = {"prompt": ids, "n_predict": gen, "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": False,
               "stream": True, "return_tokens": True, "ignore_eos": True, "timings_per_token": True}
    req = urllib.request.Request("http://%s:%d/completion" % (LAN[host], HTTP_PORT), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    rec = {"prompt_n": len(ids), "n_predict": gen, "sent_wall": time.time(), "events": []}
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            now = time.perf_counter() - t0
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            o = json.loads(line[6:])
            t = o.get("timings") or {}
            rec["events"].append([round(now, 6), o.get("tokens") or [], t.get("predicted_n"), t.get("predicted_ms"),
                                  t.get("prompt_ms")])
            if o.get("stop"):
                rec.update({"stop_type": o.get("stop_type"), "timings": o.get("timings"),
                            "tokens_evaluated": o.get("tokens_evaluated"), "tokens_cached": o.get("tokens_cached")})
                break
    rec["done_s"] = round(time.perf_counter() - t0, 6)
    return rec


def run(a):
    log = Log(a.tag, a.model)
    key = a.model
    path, want = MODEL_FILE[key]
    host = MODEL_HOST.get(key, "pi5")
    raw = open(GPL3, "rb").read()
    me = open(os.path.abspath(__file__), "rb").read()
    sp = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh"), "rb").read()
    log("start", {"what": "QR3 TPOT, one user", "model": key, "ctx": a.ctx, "mem": a.mem, "clocks": a.clocks, "reqs": a.reqs, "n": a.n,
                  "gen": a.gen,
                  "warm": [WARM_N, WARM_GEN], "probe": a.probe, "tag": a.tag,
                  "prompts": "%s (sha256 %s): request i from token i*%d, cut to n tokens, then %r; the warm-up from the end"
                             % (GPL3, hashlib.sha256(raw).hexdigest(), OFFSET, QUESTION),
                  "script_sha256": hashlib.sha256(me).hexdigest(), "serve_pool_sha256": hashlib.sha256(sp).hexdigest(),
                  "laptop_load": os.getloadavg()})
    got = ssh(host, "sha256sum %s | cut -d' ' -f1" % path).strip()
    log("model_sha256", {"host": host, "file": path, "sha256": got, "expected": want, "ok": got == want})
    if got != want:
        raise SystemExit("FATAL: %s on %s is not the pinned bytes" % (path, host))
    # Reading the file dips the host's MemAvailable for a while (the 01:28 pilot's gate read 7,435 MiB right after it; 7,492
    # settled), so serve_pool.sh's gate waits for two readings 5 s apart within 8 MiB, up to 90 s.
    prev = None
    for _ in range(18):
        cur = int(ssh(host, "awk '/MemAvailable/ {print $2}' /proc/meminfo").strip()) // 1024
        if prev is not None and abs(cur - prev) <= 8:
            break
        prev = cur
        time.sleep(5)
    log("settled", {"host": host, "mem_available_mib": cur})
    if serve("start", key, a.ctx, log, a.mem) != 0:
        serve("stop", key, a.ctx, log)
        raise SystemExit("FATAL: serve_pool.sh start failed")
    st = read_state()
    nodes = [st["HOST"]] + st["WORKERS"].split()
    log("served", {"host": st["HOST"], "workers": st["WORKERS"], "args": st["ARGS"], "ctx": st["CTX"], "serve_log": st["LOG"]})
    assert st["HOST"] == host, (st["HOST"], host)
    probes, locked = {}, False
    try:
        if "jetson" in nodes:
            log("jetson clocks before", ssh("jetson", CLOCKS, check=False).strip())
            if a.clocks == "locked":
                r = ssh("jetson", "c=/sys/devices/system/cpu/cpu0/cpufreq; g=%s; if [ $(cat $c/scaling_min_freq) = $(cat $c/scaling_max_freq) ] "
                                  "&& [ $(cat $g/min_freq) = $(cat $g/max_freq) ]; then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && "
                                  "sudo -n jetson_clocks --store ~/heteroserve/run/jetson_clocks_qr3_%s.stored && sudo -n jetson_clocks && "
                                  "echo LOCKED; fi" % (GPU, log.stamp), check=False).strip()
                locked = r == "LOCKED"
                log("jetson clocks", {"lock": r, "now": ssh("jetson", CLOCKS, check=False).strip()})
                if r not in ("LOCKED", "ALREADY_LOCKED"):
                    raise SystemExit("FATAL: could not lock the Jetson's clocks")
        if a.probe:
            for n in nodes:
                script = "memprobe_tegra.sh" if n == "jetson" else "memprobe.sh"
                subprocess.run(["scp", "-q", os.path.join(ROOT, "testbed", "scripts", script), "%s:%s" % (n, script)], check=True)
                pn = "llama-server" if n == host else "ggml-rpc-server"
                pid = ssh(n, "mkdir -p ~/logs; setsid nohup bash ~/%s ~/logs/memprobe_qr3_%s.log %s > /dev/null 2>&1 < /dev/null "
                             "& echo $!" % (script, log.stamp, pn)).strip()
                probes[n] = pid
            log("probes", probes)
        head = http(host, "/tokenize", {"content": "", "add_special": True})["tokens"]
        doc = http(host, "/tokenize", {"content": raw.decode("utf-8"), "add_special": False})["tokens"]
        q = http(host, "/tokenize", {"content": QUESTION, "add_special": False})["tokens"]
        body, wbody = a.n - len(head) - len(q), WARM_N - len(head) - len(q)
        woff = len(doc) - wbody
        assert woff >= (a.reqs - 1) * OFFSET + body, "the warm-up text would overlap a request's (doc %d tokens)" % len(doc)
        prompts = [head + doc[i * OFFSET:i * OFFSET + body] + q for i in range(a.reqs)]
        warm = head + doc[woff:woff + wbody] + q
        assert all(len(p) == a.n for p in prompts) and len(warm) == WARM_N
        log("prompts", {"doc_tokens": len(doc), "head": len(head), "question": len(q), "warm_offset": woff,
                        "sha256": [hashlib.sha256(json.dumps(p).encode()).hexdigest()[:16] for p in prompts]})
        log("health warm", {n: health(n) for n in nodes} | {"laptop_load": os.getloadavg()})
        r = stream(host, warm, WARM_GEN)
        log("warm", r)
        for i in range(a.reqs):
            log("health %d" % i, {n: health(n) for n in nodes} | {"laptop_load": os.getloadavg()})
            r = stream(host, prompts[i], a.gen)
            r["i"] = i
            log("req %d" % i, r)
            if sum(len(e[1]) for e in r["events"]) != a.gen:
                log("WARN", "request %d returned %d tokens, not %d" % (i, sum(len(e[1]) for e in r["events"]), a.gen))
        log("health end", {n: health(n) for n in nodes} | {"laptop_load": os.getloadavg()})
    finally:
        for n, pid in probes.items():
            ssh(n, "kill %s 2>/dev/null" % pid, check=False)
            dst = log.path[:-4] + "_memprobe_%s.log" % n
            subprocess.run(["scp", "-q", "%s:logs/memprobe_qr3_%s.log" % (n, log.stamp), dst], check=False)
            if os.path.exists(dst):
                av = [int(m.group(1)) for m in (re.search(r"avail_kb=(\d+)", l) for l in open(dst)) if m]
                rss = [int(m.group(1)) for m in (re.search(r"VmRSS:(\d+)", l) for l in open(dst)) if m]
                log("memprobe", {"node": n, "samples": len(av), "min_avail_mib": min(av) // 1024 if av else None,
                                 "max_vmrss_mib": max(rss) // 1024 if rss else None})
        m = re.search(r"_(\d{8}_\d{6})\.log$", st.get("LOG", ""))
        if m:   # the server's own log (serve_pool.sh: ~/logs/llama-server_serve_<stamp>.log), beside ours
            subprocess.run(["scp", "-q", "%s:logs/llama-server_serve_%s.log" % (host, m.group(1)), log.path[:-4] + "_server.log"],
                           check=False)
        serve("stop", key, a.ctx, log)
        if locked:
            log("jetson clocks restore", ssh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); if [ \"$m\" = pmode:0000 ]; then "
                                                       "sudo -n jetson_clocks --restore ~/heteroserve/run/jetson_clocks_qr3_%s.stored && "
                                                       "echo restored; else echo \"NOT restored: power mode $m\"; fi" % log.stamp,
                                              check=False).strip() + " | " + ssh("jetson", CLOCKS, check=False).strip())
    log("done", {"log": os.path.relpath(log.path, ROOT)})
    summary(log.path)


def pct(xs, p):
    """The p-th percentile by linear interpolation between order statistics (numpy's default, Hyndman-Fan type 7)."""
    s = sorted(xs)
    if not s:
        return None
    h = (len(s) - 1) * p / 100.0
    lo = int(h)
    return s[lo] + (s[min(lo + 1, len(s) - 1)] - s[lo]) * (h - lo)


def samples(r):
    """(client gaps ms, server gaps ms, merged events, TTFT client ms, TTFT server ms, tokens) for one request."""
    ev = [e for e in r["events"] if e[1]]
    cg, sg, merged = [], [], 0
    for prev, e in zip(ev, ev[1:]):
        if len(e[1]) == 1:
            cg.append(1000 * (e[0] - prev[0]))
        else:
            merged += 1
        if e[2] is not None and prev[2] is not None and e[3] is not None and prev[3] is not None:
            if e[2] - prev[2] == 1:
                sg.append(e[3] - prev[3])
    return cg, sg, merged, (1000 * ev[0][0] if ev else None), (ev[0][4] if ev else None), sum(len(e[1]) for e in ev)


def stats(xs):
    if not xs:
        return {}
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 2), "p50": round(pct(xs, 50), 2), "p95": round(pct(xs, 95), 2),
            "p99": round(pct(xs, 99), 2), "max": round(max(xs), 2), "min": round(min(xs), 2)}


def boot_p95(per_req, B=BOOT, seed=BOOT_SEED):
    """95 % CI of the pooled P95, resampling whole requests (their tokens are not independent)."""
    rng, k, out = random.Random(seed), len(per_req), []
    for _ in range(B):
        pool = []
        for _ in range(k):
            pool.extend(per_req[rng.randrange(k)])
        out.append(pct(pool, 95))
    return [round(pct(out, 2.5), 2), round(pct(out, 97.5), 2)]


def summary(path):
    reqs, meta = [], {}
    for line in open(path):
        parts = line.rstrip("\n").split(" ", 2)
        if len(parts) < 3:
            continue
        tag, body = parts[1], parts[2]
        if tag == "req":
            i, js = body.split(" ", 1)
            reqs.append(json.loads(js))
        elif tag in ("start", "served", "model_sha256"):
            meta[tag] = json.loads(body)
    per_c, per_s, rows = [], [], []
    for r in reqs:
        cg, sg, merged, tc, ts, ntok = samples(r)
        per_c.append(cg); per_s.append(sg)
        rows.append({"i": r.get("i"), "tokens": ntok, "merged_events": merged, "ttft_client_ms": round(tc, 1) if tc else None,
                     "ttft_server_prompt_ms": round(ts, 1) if ts else None,
                     "client": stats(cg), "server": stats(sg)})
    allc = [x for g in per_c for x in g]
    alls = [x for g in per_s for x in g]
    out = {"log": os.path.relpath(path, ROOT), "meta": meta, "requests": rows,
           "pooled": {"client": stats(allc), "server": stats(alls)},
           "per_request_mean_tpot": {"client": stats([statistics.fmean(g) for g in per_c if g]),
                                     "server": stats([statistics.fmean(g) for g in per_s if g])},
           "ttft_client_ms": stats([r["ttft_client_ms"] for r in rows if r["ttft_client_ms"]]),
           "merged_events": sum(r["merged_events"] for r in rows)}
    if len(per_c) >= 2 and allc:
        out["pooled"]["client"]["p95_ci95"] = boot_p95(per_c)
    if len(per_s) >= 2 and alls:
        out["pooled"]["server"]["p95_ci95"] = boot_p95(per_s)
    js = path[:-4] + "_summary.json"
    json.dump(out, open(js, "w"), indent=1)
    for row in rows:
        c, s = row["client"], row["server"]
        print("req %s: %d tokens, %d merged; TTFT %s ms; client mean %s p95 %s max %s | server mean %s p95 %s max %s" % (
            row["i"], row["tokens"], row["merged_events"], row["ttft_client_ms"], c.get("mean"), c.get("p95"), c.get("max"),
            s.get("mean"), s.get("p95"), s.get("max")))
    print("POOLED client %s" % json.dumps(out["pooled"]["client"]))
    print("POOLED server %s" % json.dumps(out["pooled"]["server"]))
    print("wrote", os.path.relpath(js, ROOT))
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True, choices=sorted(MODEL_FILE))
    r.add_argument("--ctx", type=int, default=4096)
    r.add_argument("--reqs", type=int, default=10)
    r.add_argument("--n", type=int, default=128)
    r.add_argument("--gen", type=int, default=128)
    r.add_argument("--probe", action="store_true")
    r.add_argument("--mem", default=None, help="a memory file for serve_pool.sh to plan against (MEMFILE)")
    r.add_argument("--clocks", default="locked", choices=("locked", "asfound"), help="the Jetson's clocks, when it is in the plan")
    r.add_argument("--tag", default="scored")
    s = sub.add_parser("summary")
    s.add_argument("log")
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    else:
        summary(a.log)


if __name__ == "__main__":
    main()
