#!/usr/bin/env python3
"""Exploration for the scheduler-improvement review (2026-11-01; the user: "dig deeper and find ways to improve our
scheduler"). Not a registered test: these llama-bench runs size three candidate improvements before any is registered.

    python3 testbed/scripts/explore_sched_20261101.py [--only A,B,C]

A  Qwen3-14B Q4_K_M on the laptop (-t 4 as serving, -lm none): a batch of 1, 2 and 4 tokens (pp1, pp2, pp4: what checking
   k drafted tokens costs in speculative decoding) against one decoded token (tg32), at depth 0 and at 1,024 tokens of
   context; and 4 against 8 threads for a 128-token prefill and decode (the i5-8350U has 4 cores and 8 threads; serving
   uses 4 everywhere).
B  Qwen3-0.6B Q8_0 on the laptop, flash attention on: the KV cache in f16 against q8_0 (-ctk/-ctv), decode at depth 0 and
   4,096: does a quantized cache cut the CPU's attention cost per token of context?
C  The same on the Jetson's GPU (Vulkan, every layer, the Jetson's procedure: jetson_proc_20261028.py), at depths 1,024 and
   4,096.
Every run: llama-bench -r 3 -o jsonl, raw output tee'd to testbed/results/explsched_<stamp>/; the laptop's load and apps are
logged before each laptop run. Gates: no llama.cpp running on the node, the tree at the pin, each model's sha256.
"""
import argparse, datetime, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lc4_20261024 as L                                     # noqa: E402
import jetson_proc_20261028 as JP                            # noqa: E402

LBIN = "/opt/heteroserve/llama.cpp/build/bin/llama-bench"
M14 = (os.path.expanduser("~/.cache/heteroserve-models/Qwen3-14B-Q4_K_M.gguf"), "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0")
M06 = (os.path.expanduser("~/.cache/heteroserve-models/Qwen3-0.6B-Q8_0.gguf"), "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031")
J06 = ("~/models/qwen3-0.6b-q8_0.gguf", M06[1])
RUNS = {
    "A": [("laptop", "a_verify_d0", M14, "-t 4 -lm none -p 1,2,4 -n 0 -r 3"),
          ("laptop", "a_tg_d0", M14, "-t 4 -lm none -p 0 -n 32 -r 3"),
          ("laptop", "a_verify_d1024", M14, "-t 4 -lm none -p 1,2 -n 32 -d 1024 -r 3"),
          ("laptop", "a_threads", M14, "-t 4,8 -lm none -p 128 -n 16 -r 3")],
    "B": [("laptop", "b_kv_f16", M06, "-t 4 -lm none -fa 1 -ctk f16 -ctv f16 -p 0 -n 32 -d 0,4096 -r 3"),
          ("laptop", "b_kv_q8", M06, "-t 4 -lm none -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 32 -d 0,4096 -r 3")],
    "C": [("jetson", "c_kv_f16", J06, "-ngl 99 -fa 1 -ctk f16 -ctv f16 -p 0 -n 32 -d 1024,4096 -r 3"),
          ("jetson", "c_kv_q8", J06, "-ngl 99 -fa 1 -ctk q8_0 -ctv q8_0 -p 0 -n 32 -d 1024,4096 -r 3")],
}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="A,B,C")
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "explsched_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## explsched: scheduler-improvement exploration (not registered)")
    pin = json.load(open(os.path.join(ROOT, "ThinkPad", "config", "cluster.json")))["llama"]["pin_sha"]
    groups = [g for g in a.only.split(",") if g in RUNS]
    nodes = {r[0] for g in groups for r in RUNS[g]}
    for n in nodes:
        rc, o = L.sh(n, "for p in llama-bench llama-cli llama-server ggml-rpc-server fa_bench; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
        if o.strip():
            say("FATAL: %s runs %s" % (n, o.strip())); sys.exit(1)
        tree = L.LTREE if n == "laptop" else L.BTREE
        rc, o = L.sh(n, "git -C %s rev-parse HEAD; git -C %s status --porcelain --untracked-files=no | wc -l" % (tree, tree))
        if o.split() != [pin, "0"]:
            say("FATAL: %s's tree is %r, not the pin clean" % (n, o.split())); sys.exit(2)
    for g in groups:
        for node, label, (path, want), args in RUNS[g]:
            rc, o = L.sh(node, "sha256sum %s | cut -d' ' -f1" % path)
            if o.strip() != want:
                say("FATAL: %s on %s is %r" % (path, node, o.strip())); sys.exit(3)
    say("gates: idle, at the pin %s, every model's sha256" % pin[:12])
    st = JP.up(say, "explsched_%s" % stamp) if "jetson" in nodes else None
    try:
        for g in groups:
            for node, label, (path, _), args in RUNS[g]:
                if node == "laptop":
                    load = open("/proc/loadavg").read().split()[0]
                    ps = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True).stdout.split()
                    say("laptop before %s: load %s, apps %s" % (label, load, sorted({x for x in ps if x in L.APPS})))
                    cmd = "%s -m %s %s -o jsonl; echo EXIT_BENCH=$?" % (LBIN, path, args)
                else:
                    cmd = "cd ~ && /usr/bin/timeout 2400 %s/build/bin/llama-bench -m %s %s -o jsonl; echo EXIT_BENCH=$?" % (L.BTREE, path, args)
                t0 = time.time()
                rc, out = L.sh(node, cmd, timeout=3000)
                open(os.path.join(d, "bench_%s.log" % label), "w").write(out)
                say("##### BENCH %s %s %.0f s %s" % (node, label, time.time() - t0, "ok" if "EXIT_BENCH=0" in out else "FAILED"))
    finally:
        if st:
            JP.down(say, st)
    say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
