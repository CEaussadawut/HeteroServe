#!/usr/bin/env python3
"""Exploration for lc4n (2026-10-28; item 2, a long-context law for the Jetson): which ops carry the Jetson's depth cost.

    python3 testbed/scripts/lc4n_explore_20261028.py [--n 8]

The Jetson decodes on Vulkan with flash attention on auto. ggml-vulkan at the pin has its own per-op GPU timer
(GGML_VK_PERF_LOGGER: a timestamp query around every op, printed per graph as "<op and shapes>: <count> x <us> us").
This runs llama-bench decode on the Jetson alone, with the timer on, at a few depths per model, to split each decoded
token's time into ops and see how each grows with the cache. The code says (ggml-vulkan.cpp, ggml_vk_flash_attn and
get_fa_tuning_params_scalar) that TinyLlama's attention changes path at 1,024 cache cells (below: one workgroup per query
head; from 1,024: one per KV head with its 8 query heads together), and the Qwen3 models' does not; the depths are chosen
to see it. NOT A REGISTERED MEASUREMENT: the timer ends the command buffer and waits after every graph and puts a timestamp
between every two ops, so the decode times it prints are not reported; only the per-op split is used, to design lc4n.
Raw output is tee'd into testbed/results/lc4nx_<stamp>/ (one file per bench, stdout and stderr).
Gates and the Jetson procedure as lc4_20261024.py: no llama.cpp running, the tree at the pin and clean, each model's
sha256 (testbed/models.md), the guard started by its NOTE's procedure (or found live and un-tripped), MAXN, the clocks
locked for the run and restored after.
"""
import argparse, datetime, json, os, re, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lc4_20261024 as L                                     # noqa: E402

BENCHES = [   # (label, model path on the Jetson, sha256 from testbed/models.md, extra args, depths)
    ("tq4km", "~/models/tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0", "",
     "0,512,768,2048"),
    ("q06", "~/models/qwen3-0.6b-q8_0.gguf", "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031", "",
     "512,768,2048"),
    ("q4b", "~/models/qwen3-4b-q4_k_m.gguf", "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", "-ub 128",
     "256,768"),
]


def parse(text):
    """Every decode graph's per-op times: [(n_kv, {op name: total us})], a decode graph being one whose attention has one
    query row. The op name keeps the shapes the timer prints, minus the cache length, so the same op lines up across depths."""
    out = []
    for blk in text.split("Vulkan Timings:")[1:]:
        body = blk.split("Total time:")[0]
        ops, n_kv, decode = {}, None, False
        for ln in body.strip().splitlines():
            m = re.match(r"^(.*): (\d+) x ([0-9.]+) us = ([0-9.]+) us", ln.strip())
            if not m:
                continue
            name, total = m.group(1), float(m.group(4))
            fa = re.search(r"FLASH_ATTN_EXT dst\((\d+),(\d+),(\d+),(\d+)\),\s+q\((\d+),(\d+),(\d+),(\d+)\),\s+k\((\d+),(\d+),(\d+),(\d+)\)", name)
            if fa:
                n_kv, decode = int(fa.group(10)), int(fa.group(6)) == 1
                name = "FLASH_ATTN_EXT hd=%s q_heads=%s kv_heads=%s" % (fa.group(5), fa.group(7), fa.group(11))
            ops[name] = ops.get(name, 0.0) + total
        if decode and n_kv:
            out.append((n_kv, ops))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "lc4nx_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## lc4nx: per-op timer on the Jetson (exploration, not registered)")
    pin = json.load(open(os.path.join(ROOT, "ThinkPad", "config", "cluster.json")))["llama"]["pin_sha"]
    rc, o = L.sh("jetson", "for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
    if o.strip():
        say("FATAL: the Jetson runs %s" % o.strip()); sys.exit(1)
    rc, o = L.sh("jetson", "git -C %s rev-parse HEAD; git -C %s status --porcelain --untracked-files=no | wc -l" % (L.BTREE, L.BTREE))
    if o.split() != [pin, "0"]:
        say("FATAL: the Jetson's tree is %r, not the pin clean" % o.split()); sys.exit(2)
    for label, path, want, _, _ in BENCHES:
        rc, o = L.sh("jetson", "sha256sum %s | cut -d' ' -f1" % path)
        if o.strip() != want:
            say("FATAL: %s is %r, not %s" % (path, o.strip(), want)); sys.exit(3)
    say("gates: the Jetson idle and at the pin %s; every model's sha256" % pin[:12])
    guard_started, locked = False, False
    stored = "~/heteroserve/run/jetson_clocks_lc4nx_%s.stored" % stamp
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
        for label, path, _, extra, depths in BENCHES:
            cmd = ("cd ~ && GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 /usr/bin/timeout 1200 %s/build/bin/llama-bench "
                   "-m %s -ngl 99 %s -p 0 -n %d -d %s -r 1 -o jsonl; echo EXIT_BENCH=$?" % (L.BTREE, path, extra, a.n, depths))
            t0 = time.time()
            rc, out = L.sh("jetson", cmd, timeout=1500)
            open(os.path.join(d, "bench_%s.log" % label), "w").write(out)
            say("##### BENCH %s %.0f s %s" % (label, time.time() - t0, "ok" if "EXIT_BENCH=0" in out else "FAILED"))
    finally:
        if locked:
            rc, o = L.sh("jetson", "m=$(cut -d' ' -f1 /var/lib/nvpmodel/status); [ \"$m\" = pmode:0000 ] && sudo -n jetson_clocks --restore %s && echo RESTORED" % stored)
            say("jetson clocks %s" % o.strip())
        if guard_started:
            L.sh("jetson", 'for p in $(grep -a -l -s "[j]etson_power_guard" /proc/[0-9]*/cmdline | sed "s|/proc/||; s|/cmdline||"); do sudo -n kill $p; done')
            say("jetson guard stopped")
        rc, o = L.sh("jetson", 'G=~/heteroserve/logs/power-guard.log; awk "/guard start/ {t=0} /TRIP/ {t++} END {print t+0}" $G')
        say("jetson guard trips this run: %s" % o.strip())
    summary = {}
    for label, *_ in BENCHES:
        p = os.path.join(d, "bench_%s.log" % label)
        if not os.path.exists(p):
            continue
        by = {}
        for n_kv, ops in parse(open(p, errors="replace").read()):
            by.setdefault(n_kv, []).append(ops)
        summary[label] = {}
        for n_kv, graphs in sorted(by.items()):
            names = sorted({k for g in graphs for k in g})
            med = {k: round(statistics.median(g.get(k, 0.0) for g in graphs), 1) for k in names}
            summary[label][n_kv] = {"graphs": len(graphs), "total_us": round(statistics.median(sum(g.values()) for g in graphs), 1),
                                    "ops_us": med}
            fa = {k: v for k, v in med.items() if k.startswith("FLASH_ATTN_EXT")}
            say("%s n_kv %d: %d decode graphs, median total %.1f us; attention %s" % (
                label, n_kv, len(graphs), summary[label][n_kv]["total_us"], fa))
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"), indent=1)
    say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
