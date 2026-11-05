#!/usr/bin/env python3
"""Two users at once: does the prefill/decode hand-off earn its place? (QR4; remaining work item 6.) From the repo root:

    python3 testbed/scripts/pd_twouser.py run [--reps 3] [--n 128] [--gen 32] [--arms shared,handoff,static] [--stagger 0]
    python3 testbed/scripts/pd_twouser.py summary testbed/results/pd_twouser_<stamp>.log

Two users each send a different prompt of exactly n tokens (the GPL-3 text from two offsets, then the same question) and want
`gen` tokens back, greedy. User 1 arrives at t = 0 and user 2 at t = stagger (0: together). The arms, all Qwen3.5-9B Q4_K_M:
  shared   the Pi 5 alone serves both (llama-server -np 2, continuous batching); the Pi 4 idles
  handoff  the Pi 5 prefills user 2 first and yields t1, saves the slot and sends the state straight into the Pi 4's RAM
           (pd_handoff.xfer: one TCP connection over the wired overlay, sha256 both ends); the Pi 4 restores it and decodes user
           2 (prompt + t1: the protocol a recurrent state allows). The Pi 5 then serves user 1 whole.
  static   user 1 on the Pi 5, user 2 on the Pi 4, each board doing its user's prefill and decode
  shared_nr  as shared, with the Pi 5's server started --no-repack: llama-batched-bench measured its repacked kernels cost a
           second sequence 90 % of a decode step and only 28 % without them (batched_decode_20260929_211529), at 1.55x slower
           prefill. The Pi 5's server is restarted for this arm and back after it; each restart is logged.
Every token is streamed and timestamped here, on the laptop, against the arrival time of its user: TTFT = first token - arrival,
TPOT = the mean gap between later tokens, done = last token - arrival, and the makespan. Both boards run -np 2 (-c 2 x 4096) in
every arm, so no server restarts between arms (but shared_nr's): a slot state records its server's KV stream count, and a 1-slot
server refuses one saved by a 2-slot server ("n_stream mismatch", the smoke run of 2026-09-29 21:47). The static arm is the check
that a 2-slot server serves one user as the sweep's 1-slot ones did. Arms are interleaved per repetition (A B C, A B C, ...). The
servers are started here, found by exact argv and stopped here; every response is written raw to
testbed/results/pd_twouser_<stamp>.log, with get_throttled and temperatures per repetition.
"""
import datetime, hashlib, json, os, statistics, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pd_handoff as H  # noqa: E402

ROOT = H.ROOT
NP = {"pi5": 2, "pi4": 2}


def start(node, log, n_parallel, extra=""):
    # llama-batched-b: the kernel keeps 15 characters of a process name, so pgrep -x never matches "llama-batched-bench"
    busy = H.ssh(node, "for p in llama-bench llama-batched-b llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null "
                       "&& printf '%s ' $p; done; true")
    if busy.strip():
        raise SystemExit("FATAL: %s is busy: %s" % (node, busy))
    H.ssh(node, "mkdir -p %s ~/logs; setsid nohup %s/llama-server -m %s -c %d -np %d -t 4 %s--slot-save-path %s "
                "--host %s --port %d >> ~/logs/llama-server_pd2_%s.log 2>&1 < /dev/null &"
          % (H.TMP_SLOTS, H.BIN[node], H.MODEL_AT[node], H.CTX * n_parallel, n_parallel, extra + " " if extra else "",
             H.TMP_SLOTS, H.LAN[node], H.PORT, os.path.basename(log.path)[:-4]))
    for _ in range(120):
        try:
            if H.http(node, "/health", timeout=5).get("status") == "ok":
                log("server", "%s ready, -np %d -c %d %s(pid %s)" % (node, n_parallel, H.CTX * n_parallel, extra + " " if extra else "",
                                                                  " ".join(H.spid(node))))
                return
        except Exception:
            pass
        if not H.spid(node):
            raise SystemExit("FATAL: llama-server on %s exited" % node)
        time.sleep(5)
    raise SystemExit("FATAL: %s not healthy after 10 min" % node)


def stream(node, ids, n_predict, slot, origin, rec, cache=False):
    """POST /completion with stream: every token timestamped against origin. rec gets sent/tokens/times/timings. cache=True
    only for the hand-off's continuation, which must reuse the slot it restored (prompt + t1: only t1 is processed)."""
    # ignore_eos: every user gets exactly n_predict tokens, so every arm does the work the registration priced. Without it user 1's
    # prompt ends its answer at 61 tokens (the 2026-09-29 22:40 run at --gen 128, stopped for it); at --gen 32 nobody reached it.
    payload = {"prompt": ids, "n_predict": n_predict, "temperature": 0.0, "top_k": 1, "seed": 1, "cache_prompt": cache,
               "id_slot": slot, "stream": True, "return_tokens": True, "ignore_eos": True}
    req = urllib.request.Request("http://%s:%d/completion" % (H.LAN[node], H.PORT), data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    rec.update({"node": node, "slot": slot, "prompt_n": len(ids), "n_predict": n_predict, "sent": time.time() - origin,
                "tokens": [], "times": []})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            o = json.loads(line[6:])
            now = time.time() - origin
            for t in o.get("tokens") or []:
                rec["tokens"].append(t); rec["times"].append(round(now, 4))
            if o.get("stop"):
                rec.update({"timings": o.get("timings"), "stop_type": o.get("stop_type"), "id_slot": o.get("id_slot"),
                            "tokens_evaluated": o.get("tokens_evaluated")})
                break
    return rec


def erase(node, slots):
    for s in slots:
        H.http(node, "/slots/%d?action=erase" % s, {})


def per_user(arrival, times):
    if not times:
        return {"ttft_s": None, "tpot_ms": None, "done_s": None, "n": 0}
    gaps = [b - a for a, b in zip(times, times[1:])]
    return {"ttft_s": round(times[0] - arrival, 3), "tpot_ms": round(1000 * statistics.fmean(gaps), 1) if gaps else None,
            "tpot_max_ms": round(1000 * max(gaps), 1) if gaps else None, "done_s": round(times[-1] - arrival, 3), "n": len(times)}


def arm_shared(p1, p2, gen, stagger, log, tag):
    erase("pi5", [0, 1])
    origin = time.time()
    r1, r2 = {}, {}
    t1 = threading.Thread(target=stream, args=("pi5", p1, gen, 0, origin, r1))
    t1.start()
    time.sleep(stagger)
    t2 = threading.Thread(target=stream, args=("pi5", p2, gen, 1, origin, r2))
    t2.start(); t1.join(); t2.join()
    log(tag + " user1", r1); log(tag + " user2", r2)
    return per_user(0.0, r1["times"]), per_user(stagger, r2["times"]), {}


def arm_static(p1, p2, gen, stagger, log, tag):
    erase("pi5", [0, 1]); erase("pi4", [0])
    origin = time.time()
    r1, r2 = {}, {}
    t1 = threading.Thread(target=stream, args=("pi5", p1, gen, 0, origin, r1))
    t1.start()
    time.sleep(stagger)
    t2 = threading.Thread(target=stream, args=("pi4", p2, gen, 0, origin, r2))
    t2.start(); t1.join(); t2.join()
    log(tag + " user1", r1); log(tag + " user2", r2)
    return per_user(0.0, r1["times"]), per_user(stagger, r2["times"]), {}


def arm_handoff(p1, p2, gen, stagger, log, tag):
    """User 2's prompt is prefilled on the Pi 5 FIRST (it has the long decode ahead, on the slow board); its state goes to the
    Pi 4 while the Pi 5 serves user 1 whole. With a stagger, user 1 (t = 0) is already on the Pi 5 when user 2 arrives, and
    user 2's prefill shares the Pi 5 with user 1's decode (slot 1), as it would in service."""
    erase("pi5", [0, 1]); erase("pi4", [0])
    origin = time.time()
    r1, pre, dec, info = {}, {}, {}, {}
    saved = threading.Event()

    def user2():
        try:
            time.sleep(stagger)
            stream("pi5", p2, 1, 1, origin, pre)                   # prefill + t1 on the Pi 5, slot 1
            sv = H.http("pi5", "/slots/1?action=save", {"filename": "twouser.bin"})
            info["save"] = sv; saved.set()
            info["transfer"] = H.xfer("pi5", "pi4", "twouser.bin", log, tag + " transfer")
            info["restore"] = H.http("pi4", "/slots/0?action=restore", {"filename": "twouser.bin"})
            stream("pi4", p2 + pre["tokens"][:1], gen - 1, 0, origin, dec, cache=True)
        except Exception as e:                                     # logged, and user 1 is never left waiting
            info["error"] = repr(e)
        finally:
            saved.set()

    def user1():
        if stagger == 0:
            saved.wait()                                           # together: the Pi 5 prefills user 2 first
        stream("pi5", p1, gen, 0, origin, r1)

    a, b = threading.Thread(target=user2), threading.Thread(target=user1)
    a.start(); b.start(); a.join(); b.join()
    log(tag + " user1", r1); log(tag + " user2 prefill", pre); log(tag + " user2 decode", dec)
    log(tag + " handoff", {k: v for k, v in info.items() if k != "transfer"})
    if "error" in info:
        raise RuntimeError("user 2's hand-off failed: %s" % info["error"])
    u2 = per_user(stagger, pre.get("times", [])[:1] + dec.get("times", []))
    extra = {"u2_prefill_ms": (pre.get("timings") or {}).get("prompt_ms"), "u2_decode_prompt_processed": (dec.get("timings") or {}).get("prompt_n"),
             "state_bytes": (info.get("save") or {}).get("n_written"), "transfer_s": (info.get("transfer") or {}).get("seconds"),
             "restored": (info.get("restore") or {}).get("n_restored")}
    return per_user(0.0, r1["times"]), u2, extra


ARMS = {"shared": arm_shared, "handoff": arm_handoff, "static": arm_static, "shared_nr": arm_shared}
PI5_MODE = {"shared_nr": "--no-repack"}          # every other arm runs the Pi 5's server as the sweep did (repacked)


def run(reps, n, gen, arms, stagger):
    log = H.Log("twouser")
    raw = open(H.GPL3, "rb").read()
    me = open(os.path.abspath(__file__), "rb").read()
    log("start", {"what": "two users at once, Qwen3.5-9B: %s" % ", ".join(arms), "n": n, "gen": gen, "reps": reps, "stagger_s": stagger,
                  "servers": {nd: "-np %d -c %d" % (NP[nd], H.CTX * NP[nd]) for nd in ("pi5", "pi4")},
                  "prompts": "%s (sha256 %s): user 1 from token 0, user 2 from token 2000, each cut to n tokens, then %r"
                             % (H.GPL3, hashlib.sha256(raw).hexdigest(), H.QUESTION),
                  "script_sha256": hashlib.sha256(me).hexdigest(), "pd_handoff_sha256": hashlib.sha256(open(H.__file__, "rb").read()).hexdigest()})
    for node in ("pi5", "pi4"):
        H.ssh(node, "mkdir -p %s" % H.TMP_SLOTS)
    try:
        for node in ("pi5", "pi4"):
            start(node, log, NP[node])
        head = H.http("pi5", "/tokenize", {"content": "", "add_special": True})["tokens"]
        doc = H.http("pi5", "/tokenize", {"content": raw.decode("utf-8"), "add_special": False})["tokens"]
        q = H.http("pi5", "/tokenize", {"content": H.QUESTION, "add_special": False})["tokens"]
        body = n - len(head) - len(q)
        p1, p2 = head + doc[:body] + q, head + doc[2000:2000 + body] + q
        assert len(p1) == len(p2) == n
        log("prompts", {"n": n, "p1_sha256": hashlib.sha256(json.dumps(p1).encode()).hexdigest(),
                        "p2_sha256": hashlib.sha256(json.dumps(p2).encode()).hexdigest(),
                        "pi4_tokenizes_the_same": H.http("pi4", "/tokenize", {"content": raw.decode("utf-8"), "add_special": False})["tokens"] == doc})
        mode = ""
        for r in range(1, reps + 1):
            log("rep %d health" % r, {nd: H.health(nd) for nd in ("pi5", "pi4")})
            for arm in arms:
                tag = "r%d %s" % (r, arm)
                if PI5_MODE.get(arm, "") != mode:                  # the Pi 5's server, restarted in the mode this arm needs
                    H.stop("pi5", log)
                    mode = PI5_MODE.get(arm, "")
                    start("pi5", log, NP["pi5"], mode)
                try:
                    u1, u2, extra = ARMS[arm](p1, p2, gen, stagger, log, tag)
                except Exception as e:
                    log(tag + " FAILED", repr(e)); continue
                done = [u["done_s"] for u in (u1, u2) if u["done_s"] is not None]
                log(tag + " result", {"user1": u1, "user2": u2, "makespan_s": round(max(done), 3) if len(done) == 2 else None, **extra})
        log("end", {"health": {nd: H.health(nd) for nd in ("pi5", "pi4")}})
    finally:
        for node in ("pi5", "pi4"):
            H.stop(node, log)
            with open(os.path.join(ROOT, "testbed", "results", "%s_%s_server.log" % (os.path.basename(log.path)[:-4], node)), "w") as f:
                f.write(H.ssh(node, "cat ~/logs/llama-server_pd2_%s.log" % os.path.basename(log.path)[:-4], check=False))
            H.ssh(node, "rm -f %s/twouser.bin" % H.TMP_SLOTS, check=False)
    print("log:", os.path.relpath(log.path, ROOT))


def summary(path):
    res = {}
    for line in open(path):
        parts = line.split(" ", 3)
        if len(parts) == 4 and parts[3].startswith("result "):
            arm = parts[2]
            res.setdefault(arm, []).append(json.loads(parts[3][len("result "):]))
    for arm, rs in res.items():
        def m(f):
            v = [f(x) for x in rs if f(x) is not None]
            return "%.1f" % statistics.fmean(v) + (" (%.1f-%.1f)" % (min(v), max(v)) if len(v) > 1 else "") if v else "n/a"
        print("%-8s n=%d  user1 TTFT %s s, TPOT %s ms, done %s s | user2 TTFT %s s, TPOT %s ms, done %s s | makespan %s s"
              % (arm, len(rs), m(lambda x: x["user1"]["ttft_s"]), m(lambda x: x["user1"]["tpot_ms"]), m(lambda x: x["user1"]["done_s"]),
                 m(lambda x: x["user2"]["ttft_s"]), m(lambda x: x["user2"]["tpot_ms"]), m(lambda x: x["user2"]["done_s"]),
                 m(lambda x: x["makespan_s"])))


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: a[a.index(k) + 1] if k in a else d
    if not a or a[0] not in ("run", "summary"):
        sys.exit(__doc__)
    if a[0] == "summary":
        summary(a[1])
    else:
        run(int(opt("--reps", "3")), int(opt("--n", "128")), int(opt("--gen", "32")), opt("--arms", "shared,handoff,static").split(","),
            float(opt("--stagger", "0")))
