#!/usr/bin/env python3
"""The laptop's worker rows at every quantisation the planner prices (2026-10-11, the laptop as a node). Nothing here is
scored; laptop_worker_c_20261011.py turns the run into testbed/dataset/laptop_worker_20261011c.json.

    python3 testbed/scripts/laptop_worker_quants_20261011.py

The cell is laptop_worker_remote_20261011.py's: llama-bench on the Pi 4, every layer and the head on the laptop's worker
(ggml-rpc-server -H 192.168.1.108 -p 50053 -t 4), over the link, so no host polls on the worker's cores. TinyLlama at Q4_K_M
(again, beside the b row), Q3_K_M, Q4_0, Q8_0 and Q6_K, -p 512 -n 128 -r 3, three invocations each, in a rotated order so a
drift through the session does not fall on one quantisation. The Pi 4's own files, sha256 checked there. The laptop's state
(load, clock, memory) around every run, and its 1 Hz poll (laptop_poll.sh) for the whole session. Writes
testbed/results/laptop_worker_quants_<stamp>/.
"""
import datetime, json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import spec_serve_20261010 as SS                            # noqa: E402
import longctx_20261011 as LC                               # noqa: E402  laptop_state, LAPTOP_RPC, LAPTOP_ARGS, BUSY, TREE

MODELS = {"Q4_K_M": ("~/models/tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0"),
          "Q3_K_M": ("~/models/tinyllama-1.1b-q3km.gguf", "ec461c4d2b60896152bca3978aa49cd70edad7298715e4222e12af1eefea2125"),
          "Q4_0": ("~/models/tinyllama-1.1b-q4_0.gguf", "da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556"),
          "Q8_0": ("~/models/tinyllama-1.1b-q8_0.gguf", "a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59"),
          "Q6_K": ("~/models/tinyllama-1.1b-q6_k.gguf", "9fb2be7c8bbc0ca358ae88cef6d9e6844b81929eaffa5016a543e6f56b22cb6d")}
ARGS = "-p 512 -n 128 -r 3"


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "laptop_worker_quants_%s" % stamp)
    os.makedirs(d)
    prog = open(os.path.join(d, "progress.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); prog.write(line + "\n"); prog.flush()
    if os.path.exists(LC.STATE):
        say("FATAL: serve_pool.sh state exists"); sys.exit(1)
    rc, b = SS.ssh("pi4", LC.BUSY)
    if b.strip():
        say("FATAL: the Pi 4 is busy: %s" % b.strip()); sys.exit(1)
    rc, t = SS.ssh("pi4", LC.TREE)
    if t.split() != [LC.PIN, "0"]:
        say("FATAL: the Pi 4's llama.cpp is %r" % t); sys.exit(1)
    for qn, (m, want) in MODELS.items():
        rc, o = SS.ssh("pi4", "sha256sum %s | cut -d' ' -f1" % m)
        if o.strip() != want:
            say("FATAL: %s on the Pi 4 is %s" % (m, o.strip()[:16])); sys.exit(1)
    if (" %s:%d " % (LC.LAPTOP_IP, LC.LAPTOP_PORT)) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
        say("FATAL: the laptop's port is taken"); sys.exit(1)
    poll = subprocess.Popen(["bash", os.path.join(HERE, "laptop_poll.sh"), os.path.join(d, "laptop_poll.log")],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    say("prechecks passed (models sha256 OK on the Pi 4); the laptop %s" % LC.laptop_state())
    q = list(MODELS)
    order = [q[i % len(q):] + q[:i % len(q)] for i in range(3)]
    json.dump({"order": order, "args": ARGS, "models": MODELS}, open(os.path.join(d, "plan.json"), "w"), indent=1)
    lap = None
    try:
        for inv, quants in enumerate(order, 1):
            for qn in quants:
                m = MODELS[qn][0]
                lap = subprocess.Popen([LC.LAPTOP_RPC] + LC.LAPTOP_ARGS, stdout=open(os.path.join(d, "rpc_%s_%d.log" % (qn, inv)), "w"),
                                       stderr=subprocess.STDOUT, start_new_session=True)
                for _ in range(30):
                    if (" %s:%d " % (LC.LAPTOP_IP, LC.LAPTOP_PORT)) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
                        break
                    time.sleep(1)
                say("%s %d: worker up; the laptop %s" % (qn, inv, LC.laptop_state()))
                log = "~/logs/lwq_%s_%s_%d.log" % (stamp, qn, inv)
                rc, o = SS.ssh("pi4", "~/llama.cpp/build/bin/llama-bench -m %s -rpc %s:%d -ngl 99 -t 4 %s -oe jsonl 2>&1 | tee %s > /dev/null; "
                                      "echo EXIT_BENCH=${PIPESTATUS[0]}" % (m, LC.LAPTOP_IP, LC.LAPTOP_PORT, ARGS, log), timeout=3600)
                subprocess.run(["scp", "-q", "pi4:%s" % log.replace("~/", ""), os.path.join(d, "%s.%d.log" % (qn, inv))])
                rows = [json.loads(l) for l in open(os.path.join(d, "%s.%d.log" % (qn, inv)), errors="replace") if l.startswith("{")]
                say("%s %d: %s; %s; the laptop %s" % (qn, inv, o.strip().splitlines()[-1] if o.strip() else "no exit line",
                    ", ".join("p%d n%d %.2f ms/token" % (r["n_prompt"], r["n_gen"], 1000.0 / r["avg_ts"]) for r in rows),
                    LC.laptop_state()))
                os.killpg(lap.pid, signal.SIGTERM); lap.wait(timeout=60); lap = None
    finally:
        if lap:
            os.killpg(lap.pid, signal.SIGTERM)
        try:
            os.killpg(poll.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
