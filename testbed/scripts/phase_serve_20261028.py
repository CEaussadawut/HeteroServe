#!/usr/bin/env python3
"""Phase placement for one request: prefill on one node, decode on another, or both on one (2026-10-28; item 3 of the
improvements left, "the prefill-decode hand-off inside the scheduler"). The serving path; phase_plan_20261028.py decides.

    python3 testbed/scripts/phase_serve_20261028.py smoke [--n 1024] [--gen 16]       # one request per plan, checked
    python3 testbed/scripts/phase_serve_20261028.py ask --n 4096 --gen 128 [--objective energy|time] [--plan L|J|H]

Two llama-servers, one per node, each with one slot and --slot-save-path on tmpfs: the laptop (CPU, 4 threads, loopback
only) and the Jetson (Vulkan, every layer on its GPU, flash attention on auto as every Jetson figure), the same model file
(sha256-checked) at the pin. A request is token ids (the GPL-3 text cut to exactly n tokens, then a question, as
pd_handoff.py's sweep), greedy. Plans:
  L  the laptop alone: one /completion of gen tokens
  J  the Jetson alone: the same on the Jetson
  H  the hand-off: the laptop prefills and yields the first token; its slot is saved, streamed straight into the Jetson's
     tmpfs over the router LAN (pd_handoff.py's transfer code: one TCP connection, the receiver acknowledges the whole file,
     both ends hash it after), restored there, and the Jetson decodes the remaining gen - 1 tokens from it.
The first token always comes from the node that prefills, so H's time to first token is L's. Every response is logged raw
(testbed/results/phase_<mode>_<stamp>.log); wall times are the laptop's clock around each HTTP call.
"""
import argparse, base64, datetime, json, os, re, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import pd_handoff as PH                                     # noqa: E402  (its transfer code, prompt text and prefix())
import jetson_proc_20261028 as JP                           # noqa: E402  (the Jetson's guard and clocks)

# The model (2026-11-03, the user: "... Power rows for the other models, for the phase scheduler"): PHASE_MODEL picks one;
# unset, Qwen3-0.6B Q8_0 as every run before (ph1, ph2, the calibration). Qwen3-4B Q4_K_M's context is 2,560 cells: with its
# KV cache (36 layers x 8 KV heads x 128, f16) at 9,216 it would not fit beside its weights in the Jetson's ~3.5 GB.
PHASE_MODELS = {
    "qwen3_0.6b": {"name": "Qwen3-0.6B Q8_0", "sha": "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
                   "laptop": "~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf", "jetson": "~/models/qwen3-0.6b-q8_0.gguf",
                   "ctx": 9216},     # 8,192 prompt tokens + 512 generated + the question, padded
    "qwen3_4b": {"name": "Qwen3-4B Q4_K_M", "sha": "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5",
                 "laptop": "~/.cache/heteroserve-models/Qwen3-4B-Q4_K_M.gguf", "jetson": "~/models/qwen3-4b-q4_k_m.gguf",
                 "ctx": 2560},       # 2,048 prompt tokens + 256 generated + the question, padded
}
PHASE_MODEL = os.environ.get("PHASE_MODEL", "qwen3_0.6b")
MODEL = PHASE_MODELS[PHASE_MODEL]
MODEL_SHA = MODEL["sha"]                                    # testbed/models.md
NODES = {
    "laptop": {"http": "127.0.0.1", "port": 8090, "lan": "192.168.1.108", "bin": "/opt/heteroserve/llama.cpp/build/bin",
               "model": os.path.expanduser(MODEL["laptop"]), "extra": "-t 4"},
    "jetson": {"http": "192.168.1.114", "port": 8090, "lan": "192.168.1.114", "bin": "~/llama.cpp/build/bin",
               "model": MODEL["jetson"], "extra": "-ngl 99"},
}
CTX = MODEL["ctx"]
SLOTS, XPORT = "/dev/shm/heteroserve-slots", 50092
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR"]


def sh(node, cmd, check=True, timeout=600):
    argv = ["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd]
    p = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError("%s: %r failed (%d): %s" % (node, cmd[:80], p.returncode, p.stderr.strip()[-300:]))
    return p.stdout


class Log:
    def __init__(self, mode, path=None):
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = path or os.path.join(ROOT, "testbed", "results", "phase_%s_%s.log" % (mode, stamp))
        self.f = open(self.path, "a")

    def __call__(self, tag, obj):
        line = "%s %s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), tag,
                             obj if isinstance(obj, str) else json.dumps(obj))
        self.f.write(line + "\n"); self.f.flush()
        print(line[:240], flush=True)


def http(node, path, payload=None, timeout=3600):
    c = NODES[node]
    req = urllib.request.Request("http://%s:%d%s" % (c["http"], c["port"], path),
                                 data=None if payload is None else json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if payload is None else "POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def spid(node):
    c = NODES[node]
    return sh(node, "for p in $(pgrep -x llama-server); do tr '\\0' ' ' < /proc/$p/cmdline | grep -qF -- '--host %s --port %d' "
                    "&& echo $p; done; true" % (c["http"], c["port"]), check=False).split()


def start(node, log, tag):
    c = NODES[node]
    busy = sh(node, "for p in llama-bench llama-cli llama-server ggml-rpc-server fa_bench; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
    if busy.strip():
        raise SystemExit("FATAL: %s is busy: %s" % (node, busy))
    got = sh(node, "sha256sum %s | cut -d' ' -f1" % c["model"]).strip()
    if got != MODEL_SHA:
        raise SystemExit("FATAL: %s's model is %r, not %s" % (node, got, MODEL_SHA))
    sh(node, "mkdir -p %s ~/logs; setsid nohup %s/llama-server -m %s -c %d -np 1 %s --slot-save-path %s --host %s --port %d "
             "> ~/logs/llama-server_phase_%s_%s.log 2>&1 < /dev/null &" % (SLOTS, c["bin"], c["model"], CTX, c["extra"], SLOTS,
                                                                           c["http"], c["port"], tag, node))
    for _ in range(120):
        try:
            if http(node, "/health", timeout=5).get("status") == "ok":
                log("server", "%s ready (pid %s), model sha256 %s" % (node, " ".join(spid(node)), got[:16])); return
        except Exception:
            pass
        if not spid(node):
            raise SystemExit("FATAL: llama-server on %s exited:\n%s" % (node, sh(node, "tail -20 ~/logs/llama-server_phase_%s_%s.log" % (tag, node), False)))
        time.sleep(3)
    raise SystemExit("FATAL: %s not healthy after 6 min" % node)


def stop(node, log):
    for p in spid(node):
        sh(node, "kill %s" % p, check=False)
    for _ in range(20):
        if not spid(node):
            log("server", "%s stopped" % node); return
        time.sleep(1)
    log("server", "!!!!! %s STILL RUNNING (pid %s)" % (node, " ".join(spid(node))))


def tokenize(node, text):
    return http(node, "/tokenize", {"content": text})["tokens"]


def prompt_ids(node, n):
    """The GPL-3 text cut to exactly n tokens, then the question (pd_handoff.py's sweep prompt)."""
    body = tokenize(node, open(PH.GPL3).read() * 4)
    if len(body) < n:
        raise SystemExit("the text has only %d tokens" % len(body))
    return body[:n] + tokenize(node, PH.QUESTION)


def complete(node, ids, n_predict, log, tag):
    t0 = time.time()
    r = http(node, "/completion", {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 1,
                                   "cache_prompt": True, "id_slot": 0, "return_tokens": True})
    wall = time.time() - t0
    log(tag, {"wall_s": round(wall, 4), **{k: r.get(k) for k in ("tokens", "tokens_cached", "timings", "stop_type")}})
    return r, wall


def xfer(src, dst, fname, log, tag):
    """src's slot file, streamed straight into dst's tmpfs (pd_handoff.py's XFER: the receiver accepts only src's address)."""
    py = "python3 -c \"import base64;exec(base64.b64decode('%s'))\"" % base64.b64encode(PH.XFER.encode()).decode()
    path = "%s/%s" % (SLOTS, fname)
    argv = lambda node, cmd: ["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd]
    t0 = time.time()
    rx = subprocess.Popen(argv(dst, "%s recv %s %d %s %s" % (py, NODES[dst]["lan"], XPORT, path, NODES[src]["lan"])),
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    first = rx.stdout.readline().strip()
    if first != "LISTENING":
        rx.kill()
        raise RuntimeError("receiver on %s did not listen: %r %s" % (dst, first, rx.stderr.read().strip()[-300:]))
    tx = subprocess.run(argv(src, "%s send %s %d %s" % (py, NODES[dst]["lan"], XPORT, path)), capture_output=True, text=True,
                        stdin=subprocess.DEVNULL, timeout=600)
    out, err = rx.communicate(timeout=600)
    wall = time.time() - t0
    ms = re.search(r"SENT (\d+) bytes in ([\d.]+) s ack (\S+) sha256 (\w+)", tx.stdout)
    mr = re.search(r"RECEIVED (\d+) of (\d+) bytes in ([\d.]+) s from (\S+) sha256 (\w+)", out)
    ok = (bool(ms and mr) and tx.returncode == 0 and rx.returncode == 0 and ms.group(3) == "OK"
          and ms.group(1) == mr.group(1) == mr.group(2) and ms.group(4) == mr.group(5))
    rec = {"ok": ok, "wall_s_with_ssh": round(wall, 3), "sender": tx.stdout.strip()[-200:], "receiver": out.strip()[-200:],
           "stderr": [tx.stderr.strip()[-200:], err.strip()[-200:]]}
    if ok:
        rec.update({"bytes": int(ms.group(1)), "seconds": float(ms.group(2)), "MB_per_s": round(int(ms.group(1)) / float(ms.group(2)) / 1e6, 1)})
    log(tag, rec)
    if not ok:
        raise RuntimeError("transfer %s -> %s failed or did not verify" % (src, dst))
    return rec


def meminfo(node):
    """MemAvailable, MemFree, Shmem and SwapFree of a node, in bytes (2026-11-04, hm1: the hand-off's memory check)."""
    out = sh(node, "grep -E '^(MemAvailable|MemFree|Shmem|SwapFree):' /proc/meminfo", check=False)
    return {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in out.splitlines() if ln.strip()}


def run(plan, ids, gen, log, tag, guard_headroom=None):
    """One request by plan L, J or H. Returns (tokens, record). Wall-clock marks are the laptop's time.time().
    guard_headroom (2026-11-04, hm1; bytes): H's guard. Once the slot file has landed on the Jetson, its MemAvailable is read;
    below the headroom, the file is deleted before any restore and the laptop, which still holds the prompt in its slot,
    decodes the rest (the record's "guard" says so, and its plan reads "H->L"). Above it, the restore goes ahead and the file
    is deleted right after, so the decode does not carry it. None: no guard and the file kept until the next hand-off, as
    every run before."""
    rec = {"plan": plan, "prompt_n": len(ids), "gen": gen}
    t_start = time.time()
    if plan in ("L", "J"):
        node = "laptop" if plan == "L" else "jetson"
        http(node, "/slots/0?action=erase", {})
        t_start = time.time()
        r, wall = complete(node, ids, gen, log, tag + " " + plan)
        t = r.get("timings") or {}
        rec.update({"t_start": t_start, "t_end": t_start + wall, "wall_s": wall, "prefill_ms": t.get("prompt_ms"),
                    "decode_ms_per_token": t.get("predicted_per_token_ms"), "predicted_n": t.get("predicted_n"),
                    "ttft_s": (t.get("prompt_ms") or 0) / 1000})     # the first token is sampled from the prompt's logits
        return r.get("tokens") or [], rec
    http("laptop", "/slots/0?action=erase", {}); http("jetson", "/slots/0?action=erase", {})
    sh("jetson", "rm -f %s/phase.bin" % SLOTS, check=False); sh("laptop", "rm -f %s/phase.bin" % SLOTS, check=False)
    t_start = time.time()
    pre, w1 = complete("laptop", ids, 1, log, tag + " H prefill-laptop")
    t_first = time.time()
    t1 = (pre.get("tokens") or [None])[0]
    ts = time.time(); sv = http("laptop", "/slots/0?action=save", {"filename": "phase.bin"}); w_save = time.time() - ts
    log(tag + " H save", {"wall_s": round(w_save, 4), **sv})
    x = xfer("laptop", "jetson", "phase.bin", log, tag + " H transfer")
    guard = None
    if guard_headroom is not None:
        m = meminfo("jetson")
        guard = {"headroom": guard_headroom, "after_landing": m, "refused": m.get("MemAvailable", 0) < guard_headroom}
        log(tag + " H guard", guard)
        if guard["refused"]:
            sh("jetson", "rm -f %s/phase.bin" % SLOTS, check=False)
            cont, w2 = complete("laptop", ids + [t1], gen - 1, log, tag + " H->L decode-laptop")
            t_end = time.time()
            tp, tc = pre.get("timings") or {}, cont.get("timings") or {}
            rec.update({"plan": "H->L", "guard": guard, "t_start": t_start, "t_first": t_first, "t_end": t_end,
                        "wall_s": t_end - t_start, "prefill_ms": tp.get("prompt_ms"), "ttft_s": w1, "save_wall_s": w_save,
                        "state_bytes": sv.get("n_written"), "transfer_s": x["seconds"],
                        "decode_ms_per_token": tc.get("predicted_per_token_ms"), "predicted_n": (tc.get("predicted_n") or 0) + 1})
            return [t1] + (cont.get("tokens") or []), rec
    tr = time.time(); rs = http("jetson", "/slots/0?action=restore", {"filename": "phase.bin"}); w_rest = time.time() - tr
    log(tag + " H restore", {"wall_s": round(w_rest, 4), **rs})
    if guard is not None:
        sh("jetson", "rm -f %s/phase.bin" % SLOTS, check=False)
        guard["after_restore"] = meminfo("jetson")
    cont, w2 = complete("jetson", ids + [t1], gen - 1, log, tag + " H decode-jetson")
    t_end = time.time()
    if guard is not None:
        rec["guard"] = guard
    tp, tc = pre.get("timings") or {}, cont.get("timings") or {}
    rec.update({"t_start": t_start, "t_first": t_first, "t_end": t_end, "wall_s": t_end - t_start,
                "prefill_ms": tp.get("prompt_ms"), "ttft_s": w1, "save_ms": (sv.get("timings") or {}).get("save_ms"),
                "save_wall_s": w_save, "state_bytes": sv.get("n_written"), "transfer_s": x["seconds"],
                "transfer_wall_s_with_ssh": x["wall_s_with_ssh"], "MB_per_s": x["MB_per_s"],
                "restore_ms": (rs.get("timings") or {}).get("restore_ms"), "restore_wall_s": w_rest,
                "decode_prompt_processed": tc.get("prompt_n"), "decode_ms_per_token": tc.get("predicted_per_token_ms"),
                "predicted_n": (tc.get("predicted_n") or 0) + 1, "handoff_wall_s": t_end - t_first - w2})
    return [t1] + (cont.get("tokens") or []), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("smoke", "ask"))
    ap.add_argument("--n", type=int, default=1024); ap.add_argument("--gen", type=int, default=16)
    ap.add_argument("--plan", choices=("L", "J", "H")); ap.add_argument("--objective", choices=("energy", "time"), default="energy")
    a = ap.parse_args()
    log = Log(a.cmd)
    st = JP.up(lambda s: log("jetson", s), os.path.basename(log.path)[:-4])
    try:
        start("laptop", log, os.path.basename(log.path)[:-4]); start("jetson", log, os.path.basename(log.path)[:-4])
        ids = prompt_ids("laptop", a.n)
        if prompt_ids("jetson", a.n) != ids:
            raise SystemExit("FATAL: the two servers tokenize the prompt differently")
        if a.cmd == "smoke":
            out = {}
            for plan in ("L", "J", "H"):
                toks, rec = run(plan, ids, a.gen, log, "smoke n%d" % a.n)
                out[plan] = toks
                log("smoke result", rec)
            log("smoke match", {"H_vs_L": PH.prefix(out["H"], out["L"]), "H_vs_J": PH.prefix(out["H"], out["J"]),
                                "J_vs_L": PH.prefix(out["J"], out["L"]), "gen": a.gen})
        else:
            import phase_plan_20261028 as PP
            plan = a.plan
            if not plan:
                import phase_power_20261028 as W
                rows = W.load()["rows"] if os.path.exists(W.OUT) else None       # ph2: the per-model power rows
                avail = meminfo("jetson").get("MemAvailable")                  # hm1: the hand-off's memory check
                r = PP.choose(a.n, a.gen, a.objective, rows=rows, receiver_avail=avail)
                plan = r["pick"]
                log("plan", {"pick": plan, "objective": a.objective, "power": "rows" if rows else "table",
                             "handoff_memory": r["plans"]["H"].get("memory")})
            toks, rec = run(plan, ids, a.gen, log, "ask n%d" % a.n, guard_headroom=PP.HEADROOM_BYTES)
            log("ask result", rec)
            print(http("laptop", "/detokenize", {"tokens": toks}).get("content", ""))
    finally:
        stop("laptop", log); stop("jetson", log)
        JP.down(lambda s: log("jetson", s), st)


if __name__ == "__main__":
    main()
