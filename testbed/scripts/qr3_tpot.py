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
- --nodes "pi5 pi4" restricts the boards the engine plans on (serve_pool.sh NODES).
- --engine prima (2026-09-30, the prima.cpp comparison; testbed/prima.md) serves with prima.cpp instead: ~/prima.cpp (OpenCPIL
  @ 288df6a2 + the qwen3 port) on the Pi 5 as rank 0 (llama-server, the head: it holds the input and output layers) and the
  Pi 4 as rank 1 (llama-cli, a worker), over the wired overlay (ZeroMQ, ports 9000/10000), each loading its own local copy of
  the model (both sha256-checked), --prefetch and -t 4, as its README runs it. With no --prima-lw, its Halda scheduler profiles
  both boards and picks the split, which each rank logs ("Using window size"); --prima-lw "a,b" sets it by hand. Its
  late-2024 server has no timings_per_token and no token ids in the stream, so each SSE event with content counts as one
  token (checked against the final tokens_predicted), and only client-side samples exist.
"""
import argparse, datetime, hashlib, json, os, random, re, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
LAN = {"pi5": "192.168.1.122", "pi4": "192.168.1.110", "jetson": "192.168.1.114",   # serve_pool.sh's (home router, 2026-10-08; Pi 5 10-19)
       "laptop": "192.168.1.108"}                     # the laptop as a node (2026-10-11): it can host the server
HTTP_PORT = 8080                                                                           # serve_pool.sh's
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
MODEL_FILE = {  # predict_pool.MODELS key -> (file on its host, sha256 in testbed/models.md)
    "qwen35_9b": ("~/models/qwen3.5-9b-q4_k_m.gguf", "d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43"),
    "nemo12b": ("~/models/mistral-nemo-12b-q4_k_m.gguf", "7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46"),
    "qwen3_14b": ("~/models/qwen3-14b-q4_k_m.gguf", "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0"),
    "qwen3moe": ("~/models/qwen3-30b-a3b-q3_k_m.gguf", "e145c9d2f5d11c9583eb099aa75100b7ab943e77d5240c9a2cd936f81c89ef43"),
    # 2026-10-11 (user: "Qwen3-30B-A3B at Q6_K"): on the laptop only, ~/models links into its cache
    "qwen3moe_q6k": ("~/models/qwen3-30b-a3b-q6_k.gguf", "b1a3ed3bc584d58d25253c408cbbf894f974dffa2df74e016469289e686ef701"),
    "qwen38_27b": ("~/models/qwen3.8-27b-q3_k_m.gguf", "14865aaef0b6c26a013538782d366443c7298b11d5df9f0fb829d34d1963f869"),
    # 2026-10-20: TinyLlama's other quantisations (predict_pool.MODELS), TheBloke @ 52e7645b, testbed/models.md
    "tinyllama_q8_0": ("~/models/tinyllama-1.1b-q8_0.gguf", "a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59"),
    "tinyllama_q4_0": ("~/models/tinyllama-1.1b-q4_0.gguf", "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"),
    "tinyllama_q3km": ("~/models/tinyllama-1.1b-q3km.gguf", "ec461c4d2b60896152bca3978aa49cd70edad7298715e4222e12af1eefea2125"),
    "tinyllama_q6k": ("~/models/tinyllama-1.1b-q6_k.gguf", "9fb2be7c8bbc0ca358ae88cef6d9e6844b81929eaffa5016a543e6f56b22cb6d"),
    # 2026-10-21: bartowski/Qwen_Qwen3.6-35B-A3B-GGUF @ 5c2410d7 and Qwen/Qwen3-32B-GGUF @ 938a7432, on the laptop (testbed/models.md)
    "qwen36moe_q6k": ("~/models/qwen3.6-35b-a3b-q6_k.gguf", "a97ef79439194d8ec722bc0b6c316f0c41c0b477b35bbae8177c772ecbb0eb91"),
    "qwen3_32b_q8": ("~/models/qwen3-32b-q8_0.gguf", "2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169"),
    # 2026-10-21 (scheduler v2's precision choice): the same repo and revision, on the data drive (~/models links)
    "qwen3_32b_q6k": ("~/models/qwen3-32b-q6_k.gguf", "c4c7c3cb6da260df1fe1d3cfd090a32dc7cc348f1278158be18e301f390d6f6e"),
    "qwen3_32b_q4km": ("~/models/qwen3-32b-q4_k_m.gguf", "efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689"),
}
MODEL_HOST = {"qwen3moe": "pi4", "qwen3moe_q6k": "laptop", "qwen36moe_q6k": "laptop",   # predict_pool.HOST: the node that holds
              "qwen3_32b_q8": "laptop", "qwen3_32b_q6k": "laptop", "qwen3_32b_q4km": "laptop"}   # the file
GPL3 = "/usr/share/common-licenses/GPL-3"          # a fixed public text, cut to exact token counts (as pd_twouser.py)
QUESTION = "\n\nSummarise the text above in three short sentences."
OFFSET = 500                                       # request i's text starts at token i * OFFSET of GPL-3
WARM_N, WARM_GEN = 512, 32                         # the warm-up takes the text's last tokens, disjoint from every request
BOOT, BOOT_SEED = 10000, 20260930
GPU = "/sys/devices/57000000.gpu/devfreq/57000000.gpu"
OVERLAY = {"pi5": "192.168.10.11", "pi4": "192.168.10.12"}                  # the wired overlay (run_pool.sh's WIP)
PRIMA = "~/prima.cpp"                                                    # prima_build.sh's tree on each board
PRIMA_RANKS = ["pi5", "pi4"]                                             # rank 0 = the head, rank 1 = the worker
# Since 2026-09-30 serve_pool.sh locks and restores them itself (CLOCKS, passed through), so the check below finds them
# ALREADY_LOCKED and leaves the restore to serve_pool.sh's stop; it still records the clocks and refuses if they are not locked.
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
    """cmd on node over ssh, or here when the node is this laptop (2026-10-11: a node, with no sshd)."""
    argv = ["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd]
    p = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
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
    if node == "laptop":
        return ssh(node, "echo load=$(cut -d' ' -f1 /proc/loadavg) $(for z in /sys/class/thermal/thermal_zone*; do "
                         "[ \"$(cat $z/type)\" = x86_pkg_temp ] && echo pkg_mC=$(cat $z/temp); done) "
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


def serve(action, model, ctx, log, mem=None, clocks="locked", nodes=None):
    """serve_pool.sh start|stop, its output into the log as it comes. Returns its exit code. mem: a memory file to plan
    against (serve_pool.sh MEMFILE); None keeps its default, memory_at_rest.json. clocks: serve_pool.sh CLOCKS (since
    2026-09-30 it locks the Jetson's clocks itself by default and restores them on stop)."""
    env = dict(os.environ, MODEL=model, CTX=str(ctx), CLOCKS=clocks)
    if mem:
        env["MEMFILE"] = mem
    if nodes:
        env["NODES"] = nodes
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
            toks = o.get("tokens")
            if toks is None and o.get("content") and not o.get("stop"):   # prima.cpp's late-2024 server: no token ids
                toks = [o["content"]]
            rec["events"].append([round(now, 6), toks or [], t.get("predicted_n"), t.get("predicted_ms"),
                                  t.get("prompt_ms"), o.get("content", "")])   # the text too (2026-10-21: A against B's output)
            if o.get("stop"):
                rec.update({"stop_type": o.get("stop_type"), "timings": o.get("timings"),
                            "tokens_evaluated": o.get("tokens_evaluated"), "tokens_cached": o.get("tokens_cached"),
                            "tokens_predicted": o.get("tokens_predicted")})
                break
    rec["done_s"] = round(time.perf_counter() - t0, 6)
    return rec



def prima_pids(node, rank):
    """prima.cpp processes of this rank on this board, found by exact argv (never pkill -f)."""
    return ssh(node, "for p in $(pgrep -x llama-server; pgrep -x llama-cli); do tr '\\0' ' ' 2>/dev/null < /proc/$p/cmdline | "
                     "grep -qF -- '--world %d --rank %d ' && echo $p; done" % (len(PRIMA_RANKS), rank), check=False).split()


def prima_ip(n):
    """The ring's addresses: the wired overlay when every rank is on it (the 2026-09-30 Pi 5 + Pi 4 ring), else the home LAN."""
    return OVERLAY[n] if all(r in OVERLAY for r in PRIMA_RANKS) else LAN[n]


def prima_start(a, log, path, paths=None):
    """The workers (llama-cli) from the last rank down, then rank 0 (llama-server, the head); wait until the server is healthy.
    paths: per-node model files (prima_partial_20261021.py's partial copies), else path everywhere. Returns the log names."""
    for rank, n in enumerate(PRIMA_RANKS):
        busy = ssh(n, "for p in llama-bench llama-batched-b llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null "
                      "&& printf '%s ' $p; done; true").strip()
        if busy:
            raise SystemExit("FATAL: %s is busy: %s" % (n, busy))
    lw = " -lw %s" % a.prima_lw if a.prima_lw else ""
    logs, W = {}, len(PRIMA_RANKS)
    for rank in list(range(W - 1, 0, -1)) + [0]:
        n = PRIMA_RANKS[rank]
        nxt = prima_ip(PRIMA_RANKS[(rank + 1) % W])
        common = "-m %s -t 4 --world %d --rank %d --master %s --next %s --prefetch" % (
            (paths or {}).get(n, path), W, rank, prima_ip(PRIMA_RANKS[0]), nxt)
        if rank == 0:
            cmd = "%s/llama-server %s -c %d%s --host %s --port %d" % (PRIMA, common, a.ctx, lw, LAN[n], HTTP_PORT)
        else:
            cmd = "%s/llama-cli %s" % (PRIMA, common)
        if n == "laptop":   # no system fio there: the user-space build (2026-10-21, the user's OK), for prima.cpp's disk profile
            cmd = "env PATH=$HOME/.local/fio-3.41/bin:$PATH " + cmd
        logs[n] = "prima_qr3_rank%d_%s.log" % (rank, log.stamp)
        log("prima start", {"node": n, "rank": rank, "cmd": cmd})
        # "cd X && cmd &" would background the whole list, whose subshell holds the ssh channel open until cmd exits (the
        # 05:12 smoke hung here, as serve_pool.sh's first start did on 2026-09-28): the cd runs in front, only cmd goes behind.
        ssh(n, "mkdir -p ~/logs; cd %s; setsid nohup %s > ~/logs/%s 2>&1 < /dev/null &" % (PRIMA, cmd, logs[n]), timeout=60)
        time.sleep(3)
    removed = set()                                               # workers Halda dropped (prima.cpp's own decision)
    for i in range(240):                                          # up to 20 min: profiling (fio) and loading on both
        try:
            if http(PRIMA_RANKS[0], "/health", timeout=5).get("status") == "ok":
                break
        except Exception:
            pass
        for rank, n in enumerate(PRIMA_RANKS):
            if rank in removed:
                continue
            if not prima_pids(n, rank):
                tail = ssh(n, "tail -30 ~/logs/%s" % logs[n], check=False)
                # 2026-10-22: a worker Halda gives no layers exits by design ("No layer is assigned to me, exit."; the ring
                # re-forms without it, as the 2026-10-21 20:33 TinyLlama profile showed). Prima.cpp's own decision: recorded,
                # not a crash. Only a worker: rank 0 (the head) never leaves.
                if rank > 0 and "No layer is assigned to me" in tail:
                    removed.add(rank)
                    log("prima removed", {"node": n, "rank": rank, "why": "Halda assigned it no layers (its own decision)"})
                    continue
                raise SystemExit("FATAL: prima.cpp rank %d on %s exited:\n%s" % (rank, n, tail))
        time.sleep(5)
    else:
        raise SystemExit("FATAL: prima.cpp's server not healthy after 20 min")
    for rank, n in enumerate(PRIMA_RANKS):
        log("prima ready", {"node": n, "rank": rank, "pids": prima_pids(n, rank),
                            "window": ssh(n, "grep -a -E 'Using window size|No layer is assigned' ~/logs/%s | tail -2" % logs[n],
                                          check=False).strip()})
    return logs


def prima_stop(log, logs):
    for rank in range(len(PRIMA_RANKS)):
        n = PRIMA_RANKS[rank]
        for p in prima_pids(n, rank):
            ssh(n, "kill %s" % p, check=False)
        for _ in range(20):
            if not prima_pids(n, rank):
                break
            time.sleep(1)
        left = prima_pids(n, rank)
        log("prima stop", {"node": n, "rank": rank, "stopped": not left, "left": left})
        if n in logs:
            src = os.path.expanduser("~/logs/%s" % logs[n]) if n == "laptop" else "%s:logs/%s" % (n, logs[n])
            subprocess.run(["cp" if n == "laptop" else "scp", "-q" if n != "laptop" else "-p", src,
                            log.path[:-4] + "_prima_rank%d.log" % rank], check=False)


def run(a):
    log = Log(a.tag, a.model)
    key = a.model
    path, want = MODEL_FILE[key]
    place = os.environ.get("PLACE", "")             # a fixed plan (serve_pool.sh PLACE): its first node is the host it starts
    host = place.split(",")[0].split(":")[0].strip() if place else MODEL_HOST.get(key, "pi5")   # (2026-10-24: ns4 hosted
    # TinyLlama on the laptop, which MODEL_HOST does not list, and every arm-run failed the served-host check below)
    raw = open(GPL3, "rb").read()
    me = open(os.path.abspath(__file__), "rb").read()
    sp = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh"), "rb").read()
    log("start", {"what": "QR3 TPOT, one user", "engine": a.engine, "nodes": a.nodes, "prima_lw": a.prima_lw, "model": key,
                  "ctx": a.ctx, "mem": a.mem, "clocks": a.clocks, "reqs": a.reqs, "n": a.n, "gen": a.gen,
                  "warm": [WARM_N, WARM_GEN], "probe": a.probe, "tag": a.tag,
                  "prompts": "%s (sha256 %s): request i from token i*%d, cut to n tokens, then %r; the warm-up from the end"
                             % (GPL3, hashlib.sha256(raw).hexdigest(), OFFSET, QUESTION),
                  "script_sha256": hashlib.sha256(me).hexdigest(), "serve_pool_sha256": hashlib.sha256(sp).hexdigest(),
                  "laptop_load": os.getloadavg()})
    if a.engine == "prima":
        PRIMA_RANKS[:] = a.prima_ranks.split()
    copies = json.load(open(a.prima_copies)) if (a.engine == "prima" and a.prima_copies) else None
    paths = {n: e["path"] for n, e in copies["nodes"].items() if n in PRIMA_RANKS} if copies else {}
    for n in (PRIMA_RANKS if a.engine == "prima" else [host]):     # prima.cpp: every rank loads its own copy
        if n in paths:   # a partial copy (prima_partial_20261021.py): every present range against the laptop's verified file
            v = subprocess.run([sys.executable, os.path.join(ROOT, "testbed", "scripts", "prima_partial_20261021.py"), "verify",
                                "--manifest", a.prima_copies, "--node", n], capture_output=True, text=True)
            log("model_partial", {"host": n, "file": paths[n], "verify": v.stdout.strip()[-300:], "ok": v.returncode == 0,
                                  "manifest": os.path.relpath(a.prima_copies, ROOT),
                                  "manifest_sha256": hashlib.sha256(open(a.prima_copies, "rb").read()).hexdigest()})
            if v.returncode:
                raise SystemExit("FATAL: the partial copy on %s does not verify" % n)
            continue
        got = ssh(n, "sha256sum %s | cut -d' ' -f1" % path).strip()
        log("model_sha256", {"host": n, "file": path, "sha256": got, "expected": want, "ok": got == want})
        if got != want:
            raise SystemExit("FATAL: %s on %s is not the pinned bytes" % (path, n))
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
    plogs = {}
    if a.engine == "prima":
        assert host == PRIMA_RANKS[0], "prima.cpp's head (rank 0) must hold the model's file: %s, not %s" % (host, PRIMA_RANKS[0])
        try:
            plogs = prima_start(a, log, path, paths)
            if copies:   # a rank given a layer outside its partial copy would compute with zeros: refuse before timing anything
                ws = []
                for rank, n in enumerate(PRIMA_RANKS):
                    src = os.path.expanduser("~/logs/%s" % plogs[n]) if n == "laptop" else None
                    line = (open(src, errors="replace").read() if src else ssh(n, "cat ~/logs/%s" % plogs[n], check=False))
                    m = re.findall(r"Using window size: (\d+)", line)
                    ws.append(int(m[-1]) if m else 0)
                sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
                import prima_partial_20261021 as PP
                lay = PP.prima_layers(ws, copies["n_layer"])
                bad = {n: sorted(set(lay[r]) - set(copies["nodes"][n]["layers"])) for r, n in enumerate(PRIMA_RANKS) if n in paths}
                bad = {n: v for n, v in bad.items() if v}
                log("prima windows", {"windows": dict(zip(PRIMA_RANKS, ws)), "layers": {n: [lay[r][0], lay[r][-1]] if lay[r] else []
                                                                                      for r, n in enumerate(PRIMA_RANKS)},
                                      "outside_copies": bad})
                if bad or sum(ws) != copies["n_layer"]:
                    raise SystemExit("FATAL: prima.cpp's windows %s give layers outside the partial copies: %s" % (ws, bad))
        except BaseException:
            prima_stop(log, plogs or {n: "prima_qr3_rank%d_%s.log" % (r, log.stamp) for r, n in enumerate(PRIMA_RANKS)})
            raise
        st = {"HOST": host, "WORKERS": " ".join(PRIMA_RANKS[1:]), "ARGS": "prima.cpp", "CTX": str(a.ctx), "LOG": ""}
        log("served", {"host": host, "workers": " ".join(PRIMA_RANKS[1:]), "engine": "prima", "prima_lw": a.prima_lw,
                       "ctx": str(a.ctx)})
    else:
        if serve("start", key, a.ctx, log, a.mem, a.clocks, a.nodes) != 0:
            serve("stop", key, a.ctx, log)
            raise SystemExit("FATAL: serve_pool.sh start failed")
        st = read_state()
        log("served", {"host": st["HOST"], "workers": st["WORKERS"], "args": st["ARGS"], "ctx": st["CTX"], "serve_log": st["LOG"]})
    nodes = [st["HOST"]] + st["WORKERS"].split()
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
                pn = "llama-server" if n == host else ("llama-cli" if a.engine == "prima" else "ggml-rpc-server")
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
            got_n = sum(len(e[1]) for e in r["events"])
            if got_n != a.gen or (r.get("tokens_predicted") is not None and r["tokens_predicted"] != a.gen):
                log("WARN", "request %d: %d token events, tokens_predicted %s, wanted %d" % (i, got_n, r.get("tokens_predicted"), a.gen))
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
        if a.engine == "prima":
            prima_stop(log, plogs)
        else:
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
    r.add_argument("--nodes", default=None, help='the boards the engine may plan on, e.g. "pi5 pi4" (serve_pool.sh NODES)')
    r.add_argument("--engine", default="llama", choices=("llama", "prima"), help="llama.cpp at the pin via serve_pool.sh, or prima.cpp")
    r.add_argument("--prima-lw", default=None, help='prima.cpp: a hand-set split, e.g. "30,10" (default: its Halda scheduler)')
    r.add_argument("--prima-ranks", default="pi5 pi4", help='prima.cpp: the ring, rank 0 (the head, which must hold the file) '
                                                         'first, e.g. "laptop pi5 pi4 jetson" (2026-10-21)')
    r.add_argument("--prima-copies", default=None, help="prima.cpp: a prima_partial_20261021.py manifest; its nodes load their "
                                                        "partial copies, checked range by range, and the windows must stay inside")
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
