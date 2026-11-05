#!/usr/bin/env python3
"""The attention op's own cost on the Jetson, per attention shape and cache length, with no model loaded (lc4n, 2026-10-28;
item 2, a long-context law for the Jetson).

    python3 testbed/scripts/lc4n_fa_20261028.py [--reps 30] [--tag lc4nfa]

The exploration (lc4n_explore_20261028.py, testbed/results/lc4nx_*) found every op of a decoded token on the Jetson
constant in the cache's length except FLASH_ATTN_EXT, whose cost changes path where ggml-vulkan's code says it does. This
times that op alone: fa_bench_20261028.cpp, compiled on the Jetson against its pinned tree (instrumentation; the tree is
not changed), for each attention shape in SHAPES at each cache length in KVS, one graph of <layers> ops per decoded token.
The shapes are the three models the Jetson can hold alone (the lc4n test) and the attention shapes of every model the
planner places (testbed/dataset/model_shapes.json), for the planner's table. Writes testbed/results/<tag>_<stamp>/fa.json
({shape label: {n_kv: us per graph}}) and the raw output. The Jetson procedure is lc4_20261024.py's: no llama.cpp running,
the tree at the pin and clean, the guard started (or found live and un-tripped), MAXN, the clocks locked and restored.
"""
import argparse, datetime, json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lc4_20261024 as L                                     # noqa: E402

SRC = "fa_bench_20261028.cpp"
SHAPES = [   # (label, head_dim, n_head, n_head_kv, layers in one graph)
    ("tinyllama", 64, 32, 4, 22),          # TinyLlama 1.1B: the lc4n test, every quantisation
    ("qwen3_0.6b", 128, 16, 8, 28),        # the lc4n test
    ("qwen3_4b", 128, 32, 8, 36),          # the lc4n test; also Mistral-7B and Mistral-Nemo 12B's shape
    ("qwen3_14b", 128, 40, 8, 40),         # Qwen3-14B (and Qwen2.5-14B)
    ("qwen3_32b", 128, 64, 8, 64),         # Qwen3-32B
    ("qwen3_30b_a3b", 128, 32, 4, 48),     # Qwen3-30B-A3B
    ("qwen3.5_9b", 256, 16, 4, 8),         # Qwen3.5-9B: 8 attention layers of 32 (every 4th; the rest recurrent)
    ("qwen3.6_35b_a3b", 256, 16, 2, 10),   # Qwen3.6-35B-A3B: 10 attention layers of 40
    ("qwen3.8_27b", 256, 24, 4, 16),       # Qwen3.8-27B: 16 attention layers of 64
]
KVS = [256, 512, 768, 1024, 1280, 1536, 2304, 4352]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30); ap.add_argument("--tag", default="lc4nfa")
    ap.add_argument("--only", default="", help="comma-separated shape labels (default: all)")
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "%s_%s" % (a.tag, stamp))
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## %s: fa_bench on the Jetson (%s %s)" % (a.tag, SRC, L.sha(os.path.join(HERE, SRC))[:16]))
    pin = json.load(open(os.path.join(ROOT, "ThinkPad", "config", "cluster.json")))["llama"]["pin_sha"]
    rc, o = L.sh("jetson", "for p in llama-bench llama-cli llama-server ggml-rpc-server fa_bench; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
    if o.strip():
        say("FATAL: the Jetson runs %s" % o.strip()); sys.exit(1)
    rc, o = L.sh("jetson", "git -C %s rev-parse HEAD; git -C %s status --porcelain --untracked-files=no | wc -l" % (L.BTREE, L.BTREE))
    if o.split() != [pin, "0"]:
        say("FATAL: the Jetson's tree is %r, not the pin clean" % o.split()); sys.exit(2)
    import subprocess
    subprocess.run(["scp", "-q", "-o", "LogLevel=ERROR", os.path.join(HERE, SRC), "jetson:heteroserve/tools/%s" % SRC], check=True)
    rc, o = L.sh("jetson", "cd ~/heteroserve/tools && sha256sum %s | cut -d' ' -f1 && g++ -O2 -std=c++17 %s -I ~/llama.cpp/ggml/include "
                           "-L ~/llama.cpp/build/bin -lggml -lggml-base -lggml-vulkan -Wl,-rpath,$HOME/llama.cpp/build/bin -o fa_bench "
                           "&& echo BUILT" % (SRC, SRC), timeout=900)
    if "BUILT" not in o or L.sha(os.path.join(HERE, SRC)) not in o:
        say("FATAL: fa_bench did not build: %r" % o[-2000:]); sys.exit(3)
    say("fa_bench built on the Jetson from %s" % L.sha(os.path.join(HERE, SRC))[:16])
    guard_started, locked = False, False
    stored = "~/heteroserve/run/jetson_clocks_%s_%s.stored" % (a.tag, stamp)
    res = {}
    try:
        rc, g = L.sh("jetson", 'grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | wc -l')
        if g.strip() == "0":
            rc, c = L.sh("jetson", 'timedatectl | grep -q "synchronized: yes" && echo synced; sha256sum ~/heteroserve/tools/jetson_power_guard.sh | cut -d" " -f1')
            if c.split() != ["synced", L.sha(os.path.join(HERE, "jetson_power_guard.sh"))]:
                say("FATAL: Jetson guard precheck %r" % c.split()); sys.exit(4)
            L.sh("jetson", "sudo -n setsid nohup bash ~/heteroserve/tools/jetson_power_guard.sh </dev/null >/dev/null 2>&1 & echo started")
            guard_started = True; time.sleep(5)
        rc, g = L.sh("jetson", 'm=$(cut -d" " -f1 /var/lib/nvpmodel/status); G=~/heteroserve/logs/power-guard.log; '
                               't=$(awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G); echo "$m $t"')
        if g.split() != ["pmode:0000", "0"]:
            say("FATAL: Jetson not MAXN with an un-tripped guard: %r" % g); sys.exit(4)
        rc, o = L.sh("jetson", "c=/sys/devices/system/cpu/cpu0/cpufreq; g=/sys/devices/57000000.gpu/devfreq/57000000.gpu; "
                               "if [ $(cat $c/scaling_min_freq) = $(cat $c/scaling_max_freq) ] && [ $(cat $g/min_freq) = $(cat $g/max_freq) ]; "
                               "then echo ALREADY_LOCKED; else mkdir -p ~/heteroserve/run && sudo -n jetson_clocks --store %s && sudo -n jetson_clocks && echo LOCKED; fi" % stored)
        locked = o.strip() == "LOCKED"
        if o.strip() not in ("LOCKED", "ALREADY_LOCKED"):
            say("FATAL: could not lock the Jetson's clocks: %r" % o); sys.exit(4)
        say("jetson guard %s, clocks %s" % ("started" if guard_started else "found live", o.strip()))
        only = set(x for x in a.only.split(",") if x)
        for label, hd, nh, nkvh, layers in SHAPES:
            if only and label not in only:
                continue
            cmd = "cd ~ && /usr/bin/timeout 900 ~/heteroserve/tools/fa_bench %d %d %d %d %d %s; echo EXIT_FA=$?" % (
                hd, nh, nkvh, layers, a.reps, ",".join(str(k) for k in KVS))
            t0 = time.time()
            rc, out = L.sh("jetson", cmd, timeout=1000)
            open(os.path.join(d, "fa_%s.log" % label), "w").write(out)
            got = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"^n_kv (\d+) us_per_graph ([0-9.]+)", out, re.M)}
            res[label] = {"head_dim": hd, "n_head": nh, "n_head_kv": nkvh, "layers": layers, "reps": a.reps, "us_per_graph": got}
            say("##### FA %s %.0f s %s: %s" % (label, time.time() - t0, "ok" if "EXIT_FA=0" in out else "FAILED",
                                               {k: round(v) for k, v in got.items()}))
    finally:
        if locked:
            rc, o = L.sh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); [ \"$m\" = pmode:0000 ] && sudo -n jetson_clocks --restore %s && echo RESTORED" % stored)
            say("jetson clocks %s" % o.strip())
        if guard_started:
            L.sh("jetson", 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p; done')
            say("jetson guard stopped")
        rc, o = L.sh("jetson", 'G=~/heteroserve/logs/power-guard.log; awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G')
        say("jetson guard trips this run: %s" % o.strip())
        json.dump({"made": "lc4n_fa_20261028.py", "source": SRC, "source_sha256": L.sha(os.path.join(HERE, SRC)),
                   "stamp": stamp, "kvs": KVS, "shapes": res}, open(os.path.join(d, "fa.json"), "w"), indent=1)
    say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
