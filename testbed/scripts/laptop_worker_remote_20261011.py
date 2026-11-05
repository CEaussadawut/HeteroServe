#!/usr/bin/env python3
"""The laptop as an RPC worker, characterised with a REMOTE host (2026-10-11, 00:30). Supersedes the loopback cell
(laptop_worker_char_20261010.sh). Nothing here is scored.

    python3 testbed/scripts/laptop_worker_remote_20261011.py

Why again: the long-context smoke (testbed/results/longctx_smoke_20261011_001353, plumbing only) decoded the plan at
668 ms per token, against 892 predicted. Back-computed, the laptop's 26 blocks ran about 2x faster than its loopback row
priced them. The loopback cell ran llama-bench and the worker on the same 4-core laptop. The laptop's build waits by
polling (llama-bench "poll": 50, ggml's thread pool; the boards' builds use OpenMP), so the host's 4 threads spun
against the worker's 4. A worker row has to be measured without a host beside it.

The cell: llama-bench on the Pi 4 (the host the plan uses), every layer and the head on the laptop's worker
(ggml-rpc-server -H 192.168.1.108 -p 50053 -t 4, as the plan runs it), over the measured link. Three invocations of each:
  1. TinyLlama Q4_K_M, -p 512 -n 128 -r 3: the worker row. Each token also pays one round trip and the logits back
     (32,000 x 4 bytes); the network time at the measured link is reported, and the row is also given net of it;
  2. Qwen3-0.6B Q8_0, -p 0 -n 32 -d 0,1024,2048,4096 -r 3: the attention slope. The network part is the same at every
     depth, so the slope carries none of it.
The models are the Pi 4's own files, sha256 checked there. The laptop's load is logged around each run. Writes
testbed/results/laptop_worker_remote_<stamp>/.
"""
import datetime, json, os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import spec_serve_20261010 as SS                            # noqa: E402
import longctx_20261011 as LC                               # noqa: E402  laptop_state, LAPTOP_RPC, LAPTOP_ARGS

MODELS = {"tiny_q4km": ("~/models/tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0",
                        "-p 512 -n 128 -r 3"),
          "q06_depth": ("~/models/qwen3-0.6b-q8_0.gguf", "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031",
                        "-p 0 -n 32 -d 0,1024,2048,4096 -r 3")}


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "laptop_worker_remote_%s" % stamp)
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
    for tag, (m, want, _) in MODELS.items():
        rc, o = SS.ssh("pi4", "sha256sum %s | cut -d' ' -f1" % m)
        if o.strip() != want:
            say("FATAL: %s on the Pi 4 is %s" % (m, o.strip()[:16])); sys.exit(1)
    if (" %s:%d " % (LC.LAPTOP_IP, LC.LAPTOP_PORT)) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
        say("FATAL: the laptop's port is taken"); sys.exit(1)
    say("prechecks passed; the laptop %s" % LC.laptop_state())
    lap = None
    try:
        for tag, (m, _, args) in MODELS.items():
            for inv in (1, 2, 3):
                lap = subprocess.Popen([LC.LAPTOP_RPC] + LC.LAPTOP_ARGS, stdout=open(os.path.join(d, "rpc_%s_%d.log" % (tag, inv)), "w"),
                                       stderr=subprocess.STDOUT, start_new_session=True)
                for _ in range(30):
                    if (" %s:%d " % (LC.LAPTOP_IP, LC.LAPTOP_PORT)) in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout:
                        break
                    time.sleep(1)
                say("%s %d: worker up; the laptop %s" % (tag, inv, LC.laptop_state()))
                log = "~/logs/lwr_%s_%s_%d.log" % (stamp, tag, inv)
                rc, o = SS.ssh("pi4", "~/llama.cpp/build/bin/llama-bench -m %s -rpc %s:%d -ngl 99 -t 4 %s -oe jsonl 2>&1 | tee %s > /dev/null; "
                                      "echo EXIT_BENCH=${PIPESTATUS[0]}" % (m, LC.LAPTOP_IP, LC.LAPTOP_PORT, args, log), timeout=3600)
                subprocess.run(["scp", "-q", "pi4:%s" % log.replace("~/", ""), os.path.join(d, "%s.%d.log" % (tag, inv))])
                rows = [json.loads(l) for l in open(os.path.join(d, "%s.%d.log" % (tag, inv)), errors="replace") if l.startswith("{")]
                say("%s %d: %s; %s; the laptop %s" % (tag, inv, o.strip().splitlines()[-1] if o.strip() else "no exit line",
                    ", ".join("p%d n%d d%d %.2f ms/token" % (r["n_prompt"], r["n_gen"], r["n_depth"], 1000.0 / r["avg_ts"]) for r in rows),
                    LC.laptop_state()))
                os.killpg(lap.pid, signal.SIGTERM); lap.wait(timeout=60); lap = None
    finally:
        if lap:
            os.killpg(lap.pid, signal.SIGTERM)
        say("RUN DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
