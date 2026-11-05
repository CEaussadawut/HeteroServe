#!/usr/bin/env python3
"""Prefill / decode separation, prototyped with llama-server's slot save / restore (QR4, Project 2). From the repo root:

    python3 testbed/scripts/pd_handoff.py check [--node pi5]       # one board: is a restored slot the same as never saving?
    python3 testbed/scripts/pd_handoff.py handoff [--reps 3]       # prefill on the Pi 5, decode on the Pi 4, timed
    python3 testbed/scripts/pd_handoff.py alone --node pi4 [--reps 3]  # one board doing both, the baseline
    python3 testbed/scripts/pd_handoff.py sweep [--lengths 128,256,512,1024] [--reps 3]  # the three arms by prompt length
    python3 testbed/scripts/pd_handoff.py sweep --prefill t580 --decode pi4,pi5 --alone t580   # the laptop as the prefill node
    python3 testbed/scripts/pd_handoff.py summary testbed/results/pd_sweep_<stamp>.log     # tabulate a sweep's log

A prompt is processed on one board, its slot saved to a file (llama-server --slot-save-path; the hybrid qwen35 memory writes
both the attention KV and the recurrent state: llama_memory_hybrid::state_write), and decoding continues from the restored
slot. 'check' does it on one board and compares the continuation, token for token, with a run that never saved. Greedy
decoding (temperature 0) makes the two comparable. Every server response is written raw to
testbed/results/pd_<mode>_<stamp>.log; the server is started here, found by exact argv and stopped here.
Model: Qwen3.5-9B Q4_K_M (testbed/models.md), ~/models/qwen3.5-9b-q4_k_m.gguf on both Pis.
"""
import base64, datetime, hashlib, itertools, json, os, re, statistics, subprocess, sys, time, urllib.request

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
LAN = {"pi5": "192.168.1.122", "pi4": "192.168.1.110", "t580": "127.0.0.1"}   # the home router: the Pi 4 since 2026-10-08, the Pi 5 since 10-19
PORT, CTX = 8080, 4096
MODEL = "~/models/qwen3.5-9b-q4_k_m.gguf"
SLOTS = "~/heteroserve/slots"
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
# The laptop (t580, the control node) can take the prefill side (2026-09-29, "the ThinkPad as a prefill node"): its
# commands run locally rather than over ssh, its llama-server listens on loopback (only this script talks to it), and
# its state leaves from its overlay address. Same pinned revision (b23701f, /opt/heteroserve/llama.cpp), same model bytes.
LOCAL = {"t580"}
BIN = {"pi5": "~/llama.cpp/build/bin", "pi4": "~/llama.cpp/build/bin", "t580": "/opt/heteroserve/llama.cpp/build/bin"}
MODEL_AT = {"pi5": MODEL, "pi4": MODEL, "t580": "/opt/heteroserve/models/qwen3.5-9b-q4_k_m.gguf"}
PROMPT = ("The Raspberry Pi 5 and the Raspberry Pi 4 are small single-board computers. In a heterogeneous cluster, the "
          "prefill phase of a language model processes the whole prompt at once and is limited by compute, while the decode "
          "phase produces one token at a time and is limited by memory bandwidth. Explain, in three short paragraphs, why "
          "placing the two phases on different boards might help, what has to be transferred between them, and what could "
          "go wrong. Begin with the first paragraph now.")
# The sweep: slot files in RAM on both boards, and the state sent straight from the Pi 5 to the Pi 4 over the wired overlay
# the pooled runs use (link.json, pi5-pi4: 938 Mbit/s), with no laptop hop and no ssh encryption on the data path.
OVERLAY = {"pi5": "192.168.10.11", "pi4": "192.168.10.12", "t580": "192.168.10.5"}
TMP_SLOTS, XPORT = "/dev/shm/heteroserve-slots", 50091
GPL3 = "/usr/share/common-licenses/GPL-3"      # a fixed public text, cut to exact token counts
QUESTION = "\n\nSummarise the text above in three short sentences."
# What a slot file of this model should hold, from its GGUF shape, stated before any sweep ran: 32 computed layers
# (block_count 33 less the MTP block), every 4th full attention (full_attention_interval 4), so 24 gated-delta-net layers,
# each a conv state of 3 x 8192 (conv_kernel - 1, by inner_size + 2 x group_count x state_size) and a state of 128 x 4096
# (state_size by inner_size), both f32; and 8 attention layers of K and V at 4 heads x 256, f16, per token. Plus a header.
RECURRENT_BYTES = 24 * (3 * (4096 + 2 * 16 * 128) + 128 * 4096) * 4        # 52,690,944
KV_BYTES_PER_TOKEN = 8 * 2 * 4 * 256 * 2                                  # 32,768
# Runs on the Pis (python3 -c, base64): 'recv' accepts one connection from the sending board only and writes the file;
# 'send' streams it and waits for the receiver's acknowledgement that the whole file is written. Hashed afterwards.
XFER = r'''
import hashlib, os, socket, struct, sys, time
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()
if sys.argv[1] == "recv":
    bind, port, path, peers = sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5].split(",")
    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind((bind, port)); s.listen(1)
    s.settimeout(60)
    print("LISTENING", flush=True)
    while True:
        c, a = s.accept()
        if a[0] in peers:
            break
        c.close()
    t0 = time.perf_counter()
    n = struct.unpack("!Q", c.recv(8, socket.MSG_WAITALL))[0]; got = 0
    with open(path + ".part", "wb") as f:
        while got < n:
            b = c.recv(min(1 << 20, n - got))
            if not b:
                break
            f.write(b); got += len(b)
    os.replace(path + ".part", path); t1 = time.perf_counter()
    c.sendall(b"OK" if got == n else b"NO"); c.close()
    print("RECEIVED %d of %d bytes in %.4f s from %s sha256 %s" % (got, n, t1 - t0, a[0], sha(path)), flush=True)
else:
    host, port, path = sys.argv[2], int(sys.argv[3]), sys.argv[4]
    n = os.path.getsize(path); t0 = time.perf_counter()
    c = socket.create_connection((host, port), timeout=60)
    with open(path, "rb") as f:
        c.sendall(struct.pack("!Q", n)); c.sendfile(f)
    ack = c.recv(2, socket.MSG_WAITALL); t1 = time.perf_counter(); c.close()
    print("SENT %d bytes in %.4f s ack %s sha256 %s" % (n, t1 - t0, ack.decode() or "none", sha(path)), flush=True)
'''


def argv(node, cmd):
    return ["bash", "-c", cmd] if node in LOCAL else SSH + [node, cmd]


def ssh(node, cmd, check=True):
    p = subprocess.run(argv(node, cmd), capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if check and p.returncode:
        raise RuntimeError("ssh %s failed (%d): %s" % (node, p.returncode, p.stderr.strip()))
    return p.stdout


def spid(node):
    return ssh(node, "for p in $(pgrep -x llama-server); do tr '\\0' ' ' < /proc/$p/cmdline | grep -qF -- '--host %s --port %d' "
                     "&& echo $p; done" % (LAN[node], PORT), check=False).split()


class Log:
    def __init__(self, mode):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(ROOT, "testbed", "results", "pd_%s_%s.log" % (mode, stamp))
        self.f = open(self.path, "a")

    def __call__(self, tag, obj):
        line = "%s %s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), tag,
                             obj if isinstance(obj, str) else json.dumps(obj))
        self.f.write(line + "\n"); self.f.flush()
        print(line[:220])


def http(node, path, payload=None, timeout=1800):
    req = urllib.request.Request("http://%s:%d%s" % (LAN[node], PORT, path),
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def start(node, log, slots=SLOTS):
    busy = ssh(node, "for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
    if busy.strip():
        raise SystemExit("FATAL: %s is busy: %s" % (node, busy))
    ssh(node, "mkdir -p %s ~/logs; setsid nohup %s/llama-server -m %s -c %d -np 1 -t 4 --slot-save-path %s "
              "--host %s --port %d > ~/logs/llama-server_pd_%s.log 2>&1 < /dev/null &"
        % (slots, BIN[node], MODEL_AT[node], CTX, slots, LAN[node], PORT, os.path.basename(log.path)[:-4]))
    for _ in range(120):
        try:
            if http(node, "/health", timeout=5).get("status") == "ok":
                log("server", "%s ready (pid %s)" % (node, " ".join(spid(node)))); return
        except Exception:
            pass
        if not spid(node):
            raise SystemExit("FATAL: llama-server on %s exited:\n%s" % (node, ssh(node, "tail -20 ~/logs/llama-server_pd_*.log", False)))
        time.sleep(5)
    raise SystemExit("FATAL: %s not healthy after 10 min" % node)


def stop(node, log):
    for p in spid(node):
        ssh(node, "kill %s" % p, check=False)
    for _ in range(20):
        if not spid(node):
            log("server", "%s stopped" % node); return
        time.sleep(1)
    log("server", "!!!!! %s STILL RUNNING (pid %s)" % (node, " ".join(spid(node))))


def complete(node, prompt, n_predict, log, tag):
    r = http(node, "/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 1,
                                   "cache_prompt": True, "id_slot": 0, "return_tokens": True})
    log(tag, {k: r.get(k) for k in ("content", "tokens", "tokens_cached", "timings", "stop_type")})
    return r


def check(node):
    """The hand-off protocol a recurrent state allows. llama-server matches a new prompt against the slot's cached tokens,
    and when the two are EQUAL it steps back one token to recompute the last logits -- a rollback a recurrent state cannot
    do, so it silently re-processes the whole prompt (2026-09-29 00:34, the first check: prompt_n 98 after the restore). So
    the prefill side also produces the first token t1, and the decode side sends the prompt plus t1: longer than the cache,
    no rollback, only t1 to process."""
    log = Log("check")
    log("start", "check on %s: %s, ctx %d; a restored slot must continue exactly like a run that never saved, processing 1 "
                 "prompt token" % (node, MODEL, CTX))
    start(node, log)
    try:
        ids = http(node, "/tokenize", {"content": PROMPT, "add_special": True})["tokens"]; log("prompt", {"n_tokens": len(ids)})
        ref = complete(node, ids, 33, log, "reference")                  # t1..t33, never saved
        T = ref.get("tokens") or []
        http(node, "/slots/0?action=erase", {}); log("erase", "slot 0 erased")
        pre = complete(node, ids, 1, log, "prefill")                     # the prefill side: the prompt, and t1 from its logits
        t1 = (pre.get("tokens") or [None])[0]
        sv = http(node, "/slots/0?action=save", {"filename": "check.bin"}); log("save", sv)
        size = ssh(node, "stat -c %%s %s/check.bin" % SLOTS).strip(); log("file", "check.bin %s bytes" % size)
        http(node, "/slots/0?action=erase", {}); log("erase", "slot 0 erased")
        rs = http(node, "/slots/0?action=restore", {"filename": "check.bin"}); log("restore", rs)
        cont = complete(node, ids + [t1], 32, log, "continue")            # the decode side: restore, then prompt + t1
        a, b = T[1:33], cont.get("tokens") or []
        same = sum(1 for _ in itertools.takewhile(lambda t: t[0] == t[1], zip(a, b)))
        log("verdict", {"t1_matches_reference": bool(T) and t1 == T[0], "identical": a == b, "matching_prefix_tokens": same,
                        "of": len(a), "continue_prompt_processed": (cont.get("timings") or {}).get("prompt_n"),
                        "state_file_bytes": int(size), "prompt_tokens": len(ids)})
    finally:
        stop(node, log)
    print("log:", os.path.relpath(log.path, ROOT))

def handoff(reps):
    """Prefill on the Pi 5, decode on the Pi 4, `reps` times: the Pi 5 processes the prompt and yields t1, saves the slot;
    the file is streamed Pi 5 -> laptop -> Pi 4 and timed; the Pi 4 restores it and decodes 32 tokens from prompt + t1.
    Once beforehand the Pi 4 does the whole job itself (its own prefill and 33 tokens), for the time a hand-off saves and
    for the tokens to compare against."""
    log = Log("handoff")
    A, B = "pi5", "pi4"
    log("start", "hand-off %s -> %s: %s, ctx %d, %d reps" % (A, B, MODEL, CTX, reps))
    start(A, log); start(B, log)
    try:
        ids = http(A, "/tokenize", {"content": PROMPT, "add_special": True})["tokens"]; log("prompt", {"n_tokens": len(ids)})
        own = complete(B, ids, 33, log, "pi4-alone")                       # the Pi 4 doing prefill and decode itself
        http(B, "/slots/0?action=erase", {})
        for r in range(1, reps + 1):
            http(A, "/slots/0?action=erase", {}); http(B, "/slots/0?action=erase", {})
            pre = complete(A, ids, 1, log, "r%d prefill-%s" % (r, A))
            t1 = (pre.get("tokens") or [None])[0]
            sv = http(A, "/slots/0?action=save", {"filename": "handoff.bin"}); log("r%d save" % r, sv)
            t0 = time.time()
            p = subprocess.run("ssh -o BatchMode=yes %s 'cat %s/handoff.bin' | ssh -o BatchMode=yes %s 'cat > %s/handoff.bin'"
                               % (A, SLOTS, B, SLOTS), shell=True, capture_output=True, text=True)
            dt = time.time() - t0
            size = ssh(B, "stat -c %%s %s/handoff.bin" % SLOTS).strip()
            log("r%d transfer" % r, {"rc": p.returncode, "bytes": int(size), "seconds": round(dt, 3), "MB_per_s": round(int(size) / dt / 1e6, 1)})
            rs = http(B, "/slots/0?action=restore", {"filename": "handoff.bin"}); log("r%d restore" % r, rs)
            cont = complete(B, ids + [t1], 32, log, "r%d decode-%s" % (r, B))
            a, b = (own.get("tokens") or [])[1:33], cont.get("tokens") or []
            same = sum(1 for _ in itertools.takewhile(lambda t: t[0] == t[1], zip(a, b)))
            tp, tc = pre.get("timings") or {}, cont.get("timings") or {}
            log("r%d result" % r, {"t1_matches_pi4_alone": t1 == (own.get("tokens") or [None])[0], "matching_prefix_vs_pi4_alone": same,
                                   "of": len(a), "decode_prompt_processed": tc.get("prompt_n"),
                                   "prefill_ms_pi5": tp.get("prompt_ms"), "save_ms": (sv.get("timings") or {}).get("save_ms"),
                                   "transfer_s": round(dt, 3), "restore_ms": (rs.get("timings") or {}).get("restore_ms"),
                                   "decode_ms_per_token_pi4": tc.get("predicted_per_token_ms"),
                                   "pi4_alone_prefill_ms": (own.get("timings") or {}).get("prompt_ms")})
    finally:
        stop(A, log); stop(B, log)
    print("log:", os.path.relpath(log.path, ROOT))


def alone(node, reps):
    """One board doing prefill and decode itself, `reps` times from an erased slot: the baseline a hand-off is judged against."""
    log = Log("alone")
    log("start", "%s alone: %s, ctx %d, %d reps (erased slot each time)" % (node, MODEL, CTX, reps))
    start(node, log)
    try:
        ids = http(node, "/tokenize", {"content": PROMPT, "add_special": True})["tokens"]; log("prompt", {"n_tokens": len(ids)})
        for r in range(1, reps + 1):
            http(node, "/slots/0?action=erase", {})
            t = complete(node, ids, 33, log, "r%d %s-alone" % (r, node)).get("timings") or {}
            log("r%d result" % r, {"prefill_ms": t.get("prompt_ms"), "prompt_n": t.get("prompt_n"),
                                   "decode_ms_per_token": t.get("predicted_per_token_ms"), "predicted_n": t.get("predicted_n")})
    finally:
        stop(node, log)
    print("log:", os.path.relpath(log.path, ROOT))


def health(node):
    if node in LOCAL:   # no vcgencmd on the laptop: package temperature and the Intel thermal-throttle event counters
        return ssh(node, "echo pkg=$(for z in /sys/class/thermal/thermal_zone*; do [ \"$(cat $z/type)\" = x86_pkg_temp ] && "
                         "echo $(( $(cat $z/temp) / 1000 ))C; done) throttle_events=$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/"
                         "core_throttle_count)/$(cat /sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count) "
                         "MemAvailable_kB=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)", check=False).strip()
    return ssh(node, "echo $(vcgencmd get_throttled) $(vcgencmd measure_temp) MemAvailable_kB=$(awk '/MemAvailable/ "
                     "{print $2}' /proc/meminfo)", check=False).strip()


def prefix(a, b):
    return sum(1 for _ in itertools.takewhile(lambda t: t[0] == t[1], zip(a, b)))


def xfer(src, dst, fname, log, tag):
    """Stream src's slot file straight to dst over the wired overlay: one TCP connection, into dst's tmpfs. The sender
    times it from connect to dst's acknowledgement that the whole file is written; both ends hash the file afterwards,
    outside the timed part. The wall time here also counts two ssh logins, which is orchestration, not the data path."""
    py = "python3 -c \"import base64;exec(base64.b64decode('%s'))\"" % base64.b64encode(XFER.encode()).decode()
    path = "%s/%s" % (TMP_SLOTS, fname)
    t0 = time.time()
    rx = subprocess.Popen(argv(dst, "%s recv %s %d %s %s,%s" % (py, OVERLAY[dst], XPORT, path, OVERLAY[src], LAN[src])),
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = rx.stdout.readline().strip()
    if first != "LISTENING":
        rx.kill()
        raise RuntimeError("receiver on %s did not listen: %r %s" % (dst, first, rx.stderr.read().strip()[-300:]))
    tx = subprocess.run(argv(src, "%s send %s %d %s" % (py, OVERLAY[dst], XPORT, path)), capture_output=True, text=True,
                        stdin=subprocess.DEVNULL,
                        timeout=180)
    out, err = rx.communicate(timeout=180)
    wall = time.time() - t0
    ms = re.search(r"SENT (\d+) bytes in ([\d.]+) s ack (\S+) sha256 (\w+)", tx.stdout)
    mr = re.search(r"RECEIVED (\d+) of (\d+) bytes in ([\d.]+) s from (\S+) sha256 (\w+)", out)
    rec = {"sender": tx.stdout.strip(), "receiver": out.strip(), "rc": [tx.returncode, rx.returncode],
           "stderr": [tx.stderr.strip()[-300:], err.strip()[-300:]], "wall_s_with_ssh": round(wall, 3)}
    ok = (bool(ms and mr) and tx.returncode == 0 and rx.returncode == 0 and ms.group(3) == "OK"
          and ms.group(1) == mr.group(1) == mr.group(2) and ms.group(4) == mr.group(5))
    if ok:
        n, sec = int(ms.group(1)), float(ms.group(2))
        rec.update({"bytes": n, "seconds": sec, "MB_per_s": round(n / sec / 1e6, 1), "receiver_seconds": float(mr.group(3)),
                    "sha256": ms.group(4)})
    log(tag, rec)
    if not ok:
        raise RuntimeError("transfer %s -> %s failed or did not verify" % (src, dst))
    return rec


def arm_alone(node, ids, log, tag):
    http(node, "/slots/0?action=erase", {})
    r = complete(node, ids, 33, log, tag)
    t = r.get("timings") or {}
    return r.get("tokens") or [], {"prefill_ms": t.get("prompt_ms"), "prompt_n": t.get("prompt_n"),
                                   "decode_ms_per_token": t.get("predicted_per_token_ms"), "predicted_n": t.get("predicted_n")}


def arm_handoff(A, B, ids, log, tag):
    http(A, "/slots/0?action=erase", {}); http(B, "/slots/0?action=erase", {})
    pre = complete(A, ids, 1, log, tag + " prefill-" + A)
    t1 = (pre.get("tokens") or [None])[0]
    sv = http(A, "/slots/0?action=save", {"filename": "sweep.bin"}); log(tag + " save", sv)
    x = xfer(A, B, "sweep.bin", log, tag + " transfer")
    rs = http(B, "/slots/0?action=restore", {"filename": "sweep.bin"}); log(tag + " restore", rs)
    cont = complete(B, ids + [t1], 32, log, tag + " decode-" + B)
    tp, tc = pre.get("timings") or {}, cont.get("timings") or {}
    save_ms, restore_ms = (sv.get("timings") or {}).get("save_ms"), (rs.get("timings") or {}).get("restore_ms")
    return [t1] + (cont.get("tokens") or []), {
        "prefill_ms": tp.get("prompt_ms"), "prompt_n": tp.get("prompt_n"), "save_ms": save_ms, "state_bytes": sv.get("n_written"),
        "transfer_s": x["seconds"], "transfer_wall_s_with_ssh": x["wall_s_with_ssh"], "MB_per_s": x["MB_per_s"],
        "restore_ms": restore_ms, "decode_prompt_processed": tc.get("prompt_n"),
        "decode_ms_per_token": tc.get("predicted_per_token_ms"), "predicted_n": tc.get("predicted_n"),
        "handoff_s": round((tp["prompt_ms"] + save_ms + restore_ms) / 1000 + x["seconds"], 3)}


def sweep(lengths, reps, prefill="pi5", decodes=("pi4",), alone=("pi4", "pi5")):
    """Per prompt length and repetition, the arms in turn: each board in `alone` doing its own prefill and 33 tokens,
    then one hand-off per board in `decodes` (the `prefill` board processes the prompt and yields t1, the state is sent
    directly into the decoding board's RAM, and 32 tokens are decoded there). The prompt is the GPL-3 text cut to exactly
    n tokens, then a question. Every result records how far its tokens match each board alone in this run. A failed arm
    is logged and tried once more at the end, as a make-up. The defaults are the 2026-09-29 01:08 sweep exactly (arms
    pi4-alone, pi5-alone, handoff); `--prefill t580 --decode pi4,pi5 --alone t580` is the laptop as the prefill node."""
    log = Log("sweep")
    A = prefill
    nodes = list(dict.fromkeys([A] + list(decodes) + list(alone)))
    raw = open(GPL3, "rb").read()
    me = open(os.path.abspath(__file__), "rb").read()
    arms = {}
    for n_ in alone:
        arms["%s-alone" % n_] = (lambda node: lambda ids, tag: arm_alone(node, ids, log, tag))(n_)
    for d in decodes:
        name = "handoff" if (A, tuple(decodes)) == ("pi5", ("pi4",)) else "handoff-%s-%s" % (A, d)
        arms[name] = (lambda node: lambda ids, tag: arm_handoff(A, node, ids, log, tag))(d)
    log("start", {"what": "prompt-length sweep of the %s -> %s hand-off, against %s alone" % (A, "/".join(decodes), "/".join(alone)),
                  "model": {n_: MODEL_AT[n_] for n_ in nodes}, "ctx": CTX, "lengths": lengths, "reps": reps, "arms": list(arms),
                  "order": "per length, per repetition, the arms in turn",
                  "prompt": "%s (sha256 %s) cut to n tokens, then %r" % (GPL3, hashlib.sha256(raw).hexdigest(), QUESTION),
                  "slots": TMP_SLOTS + " (tmpfs) on every node",
                  "transfer": ", ".join("direct TCP %s -> %s:%d" % (OVERLAY[A], OVERLAY[d], XPORT) for d in decodes) + ", sender-timed",
                  "expected_state_bytes": "%d + %d per token, plus a small header" % (RECURRENT_BYTES, KV_BYTES_PER_TOKEN),
                  "script_sha256": hashlib.sha256(me).hexdigest()})
    for n_ in nodes:
        ssh(n_, "mkdir -p %s" % TMP_SLOTS)
    for n_ in nodes:
        start(n_, log, TMP_SLOTS)
    refs, failed = {}, []

    def run(n, ids, r, arm, tag):
        try:
            toks, res = arms[arm](ids, tag)
        except Exception as e:                                           # logged raw; the arm is tried again at the end
            log(tag + " FAILED", repr(e)); return False
        if arm.endswith("-alone"):
            refs.setdefault(n, {}).setdefault(arm[:-len("-alone")], toks)
        ref = refs.get(n, {})
        res.update({"tokens_n": len(toks), "t1": toks[0] if toks else None})
        for b in alone:
            res["matching_prefix_vs_%s_alone" % b] = prefix(toks, ref[b]) if b in ref else None
        log(tag + " result", res)
        return True

    try:
        text = raw.decode("utf-8")
        head = http(A, "/tokenize", {"content": "", "add_special": True})["tokens"]
        doc = http(A, "/tokenize", {"content": text, "add_special": False})["tokens"]
        q = http(A, "/tokenize", {"content": QUESTION, "add_special": False})["tokens"]
        rec = {"head": head, "doc_tokens": len(doc), "question_tokens": len(q),
               "doc_ids_sha256": hashlib.sha256(json.dumps(doc).encode()).hexdigest()}
        for n_ in nodes[1:]:
            rec["%s_tokenizes_the_same" % n_] = http(n_, "/tokenize", {"content": text, "add_special": False})["tokens"] == doc
        log("tokens", rec)
        prompts = {n: head + doc[:n - len(head) - len(q)] + q for n in lengths}
        assert all(len(v) == n for n, v in prompts.items())
        for n in lengths:
            log("n%d" % n, {"prompt_tokens": n, "health": {n_: health(n_) for n_ in nodes}})
            for r in range(1, reps + 1):
                for arm in arms:
                    if not run(n, prompts[n], r, arm, "n%d r%d %s" % (n, r, arm)):
                        failed.append((n, r, arm))
        for n, r, arm in failed:
            run(n, prompts[n], r, arm, "n%d r%d %s makeup" % (n, r, arm))
        log("end", {"health": {n_: health(n_) for n_ in nodes}, "failed_then_made_up": failed})
    finally:
        for n_ in nodes:
            stop(n_, log)
        stamp = os.path.basename(log.path)[:-4]
        for node in nodes:                                               # keep the last state; do not leave it in RAM
            ssh(node, "mkdir -p ~/heteroserve/slots; [ -f %s/sweep.bin ] && mv %s/sweep.bin ~/heteroserve/slots/%s_last.bin; true"
                % (TMP_SLOTS, TMP_SLOTS, stamp), check=False)
            with open(os.path.join(ROOT, "testbed", "results", "%s_%s_server.log" % (stamp, node)), "w") as f:
                f.write(ssh(node, "cat ~/logs/llama-server_pd_%s.log" % stamp, check=False))   # the name start() gives it
    print("log:", os.path.relpath(log.path, ROOT))


def summary(path):
    """Tabulate a sweep log: per length and arm, the mean of each repetition's figures and their spread (max - min)."""
    res, start_rec = {}, None
    for line in open(path):
        m = re.match(r"(\S+) (n(\d+) r(\d+) (\S+)(?: makeup)?) result (\{.*\})$", line.strip())
        if m:
            res.setdefault((int(m.group(3)), m.group(5)), []).append(json.loads(m.group(6)))
        elif " start {" in line:
            start_rec = json.loads(line.split(" start ", 1)[1])

    def ms(v):
        return "%.2f" % statistics.fmean(v) + (" (%.2f)" % (max(v) - min(v)) if len(v) > 1 else "")
    print("prompt length | arm | reps | prefill s | decode ms/token | hand-off s | state bytes (expected) | transfer s, MB/s "
          "| tokens matching each board alone")
    for (n, arm), rows in sorted(res.items()):
        pre = [x["prefill_ms"] / 1000 for x in rows]
        dec = [x["decode_ms_per_token"] for x in rows]
        line = "%d | %s | %d | %s | %s" % (n, arm, len(rows), ms(pre), ms(dec))
        if arm.startswith("handoff"):
            sb = sorted({x["state_bytes"] for x in rows})
            line += " | %s | %s (%d) | %s, %s" % (ms([x["handoff_s"] for x in rows]), "/".join(map(str, sb)),
                                                  RECURRENT_BYTES + KV_BYTES_PER_TOKEN * n, ms([x["transfer_s"] for x in rows]),
                                                  ms([x["MB_per_s"] for x in rows]))
        else:
            line += " | | |"
        keys = sorted({k for x in rows for k in x if k.startswith("matching_prefix_vs_")})
        line += " | " + "; ".join("%s %s" % (k[len("matching_prefix_vs_"):], "/".join(str(x.get(k)) for x in rows)) for k in keys)
        print(line)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("check", "handoff", "alone", "sweep", "summary"):
        sys.exit(__doc__)
    if sys.argv[1] == "summary":
        summary(sys.argv[2])
    elif sys.argv[1] == "sweep":
        opt = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
        sweep([int(x) for x in opt("--lengths", "128,256,512,1024").split(",")], int(opt("--reps", "3")),
              prefill=opt("--prefill", "pi5"), decodes=tuple(opt("--decode", "pi4").split(",")),
              alone=tuple(opt("--alone", "pi4,pi5").split(",")))
    elif sys.argv[1] == "check":
        check(sys.argv[sys.argv.index("--node") + 1] if "--node" in sys.argv else "pi5")
    elif sys.argv[1] == "alone":
        alone(sys.argv[sys.argv.index("--node") + 1] if "--node" in sys.argv else "pi4",
              int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3)
    else:
        handoff(int(sys.argv[sys.argv.index("--reps") + 1]) if "--reps" in sys.argv else 3)
