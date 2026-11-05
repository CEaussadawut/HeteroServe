#!/usr/bin/env python3
"""Why lc4n's Qwen3-4B slope missed fa_bench by 10.2 % (2026-10-28; exploration after the data, not registered).

    python3 testbed/scripts/lc4n_q4_explore_20261028.py

lc4n measured Qwen3-4B Q4_K_M's decode slope over depths 256-768 at 18.31 ms per 1,000 tokens. fa_bench predicted 16.62
for the attention op; the Vulkan per-op timer had the op at 17.2 in the model (lc4nx). So roughly 4 % sits between fa_bench
and the op in the model, and 6 % outside the op. Two runs of lc4n's bench on the Jetson alone, each with a 1 Hz monitor
of the board's memory (/proc/meminfo: MemAvailable, Cached, SwapFree, zram) and paging counters (/proc/vmstat: pswpin,
pswpout, pgmajfault, pgscan_kswapd, pgsteal_kswapd):
  A  llama-bench -ngl 99 -ub 128 -p 0 -n 32 -d 256,512,768 -r 3, plain: the wall time per token, as lc4n
  B  the same with GGML_VK_PERF_LOGGER=1: the GPU time of every op of every decoded token
If the wall slope (A) exceeds the slope of the GPU ops' sum (B), the extra is outside the GPU (CPU work, waits); if the
paging counters climb with depth, memory pressure is the suspect. TinyLlama Q4_K_M runs the same as a control (no
pressure: 0.67 GB of weights). Raw output in testbed/results/lc4nq_<stamp>/. The Jetson's procedure: jetson_proc_20261028.
"""
import datetime, json, os, re, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import lc4_20261024 as L                                     # noqa: E402
import jetson_proc_20261028 as JP                            # noqa: E402
import lc4n_explore_20261028 as X                            # noqa: E402  (the per-op timer's parser)

MODELS = [("q4", "~/models/qwen3-4b-q4_k_m.gguf", "7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5", "-ub 128"),
          ("tq", "~/models/tinyllama-1.1b-q4km.gguf", "9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0", "")]
DEPTHS = "256,512,768"
MON = ("while true; do echo T \\$(date +%s.%N); grep -E '^(MemAvailable|Cached|SwapFree|SwapTotal):' /proc/meminfo; "   # \$: the
       "grep -E '^(pswpin|pswpout|pgmajfault|pgscan_kswapd|pgsteal_kswapd) ' /proc/vmstat; sleep 1; done")             # inner bash expands it


def main():
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    d = os.path.join(ROOT, "testbed", "results", "lc4nq_%s" % stamp)
    os.makedirs(d)
    log = open(os.path.join(d, "run.log"), "a")

    def say(s):
        line = "%s %s" % (datetime.datetime.now().astimezone().isoformat(timespec="seconds"), s)
        print(line, flush=True); log.write(line + "\n"); log.flush()
    say("########## lc4nq: Qwen3-4B's extra depth cost (exploration, not registered)")
    pin = json.load(open(os.path.join(ROOT, "ThinkPad", "config", "cluster.json")))["llama"]["pin_sha"]
    rc, o = L.sh("jetson", "for p in llama-bench llama-cli llama-server ggml-rpc-server fa_bench; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true")
    if o.strip():
        say("FATAL: the Jetson runs %s" % o.strip()); sys.exit(1)
    rc, o = L.sh("jetson", "git -C %s rev-parse HEAD; git -C %s status --porcelain --untracked-files=no | wc -l" % (L.BTREE, L.BTREE))
    if o.split() != [pin, "0"]:
        say("FATAL: the Jetson's tree is %r, not the pin clean" % o.split()); sys.exit(2)
    for label, path, want, _ in MODELS:
        rc, o = L.sh("jetson", "sha256sum %s | cut -d' ' -f1" % path)
        if o.strip() != want:
            say("FATAL: %s is %r, not %s" % (path, o.strip(), want)); sys.exit(3)
    st = JP.up(say, "lc4nq_%s" % stamp)
    try:
        for label, path, _, extra in MODELS:
            for mode, env in (("plain", ""), ("timer", "GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 ")):
                mlog = "~/logs/lc4nq_mon_%s_%s_%s.log" % (stamp, label, mode)
                L.sh("jetson", "mkdir -p ~/logs; setsid nohup bash -c \"%s\" > %s 2>&1 < /dev/null & echo $! > ~/logs/lc4nq_mon.pid" % (MON, mlog))
                cmd = ("cd ~ && %s/usr/bin/timeout 1500 %s/build/bin/llama-bench -m %s -ngl 99 %s -p 0 -n 32 -d %s -r 3 -o jsonl; "
                       "echo EXIT_BENCH=$?" % (env, L.BTREE, path, extra, DEPTHS))
                t0 = time.time()
                rc, out = L.sh("jetson", cmd, timeout=1700)
                L.sh("jetson", "kill $(cat ~/logs/lc4nq_mon.pid) 2>/dev/null; true")
                open(os.path.join(d, "bench_%s_%s.log" % (label, mode)), "w").write(out)
                subprocess.run(["scp", "-q", "-o", "LogLevel=ERROR", "jetson:%s" % mlog.replace("~/", ""), os.path.join(d, "mon_%s_%s.log" % (label, mode))])
                say("##### BENCH %s %s %.0f s %s" % (label, mode, time.time() - t0, "ok" if "EXIT_BENCH=0" in out else "FAILED"))
    finally:
        JP.down(say, st)
    # ---- summary
    out = {}
    for label, *_ in MODELS:
        plain = open(os.path.join(d, "bench_%s_plain.log" % label), errors="replace").read()
        wall = {}
        for ln in plain.splitlines():
            if ln.startswith('{"build'):
                x = json.loads(ln)
                wall[int(x["n_depth"])] = statistics.fmean(s / 1e6 for s in x["samples_ns"]) / x["n_gen"]
        by = {}
        for n_kv, ops in X.parse(open(os.path.join(d, "bench_%s_timer.log" % label), errors="replace").read()):
            by.setdefault(n_kv, []).append(ops)
        gpu = {k: statistics.median(sum(g.values()) for g in v) / 1000.0 for k, v in by.items()}
        fa = {k: statistics.median(sum(x for n, x in g.items() if n.startswith("FLASH")) for g in v) / 1000.0 for k, v in by.items()}
        depths = sorted(wall)
        slope = lambda pts: 1000 * L.slope(pts)
        r = {"wall_ms_by_depth": {str(k): round(v, 3) for k, v in sorted(wall.items())},
             "gpu_ops_ms_by_n_kv": {str(k): round(v, 3) for k, v in sorted(gpu.items())},
             "fa_ms_by_n_kv": {str(k): round(v, 3) for k, v in sorted(fa.items())},
             "wall_slope_ms_per_1k": round(slope([(k, wall[k]) for k in depths]), 3)}
        kvs = [k for k in sorted(gpu) if k >= 512]
        if len(kvs) >= 2:
            r["gpu_slope_ms_per_1k"] = round(slope([(k, gpu[k]) for k in kvs]), 3)
            r["fa_slope_ms_per_1k"] = round(slope([(k, fa[k]) for k in kvs]), 3)
        mon = {}
        for mode in ("plain", "timer"):
            vals = {}
            for ln in open(os.path.join(d, "mon_%s_%s.log" % (label, mode)), errors="replace"):
                m = re.match(r"^(\w+):?\s+(\d+)", ln)
                if m:
                    vals.setdefault(m.group(1), []).append(int(m.group(2)))
            mon[mode] = {k: {"first": v[0], "last": v[-1], "min": min(v), "max": max(v)} for k, v in vals.items() if v}
        r["memory"] = mon
        out[label] = r
        say("%s: wall slope %s, GPU-ops slope %s, attention slope %s ms per 1,000 tokens; MemAvailable min %s kB; "
            "pswpout %s -> %s; pgmajfault %s -> %s" % (label, r["wall_slope_ms_per_1k"], r.get("gpu_slope_ms_per_1k"),
                                                       r.get("fa_slope_ms_per_1k"), mon["plain"].get("MemAvailable", {}).get("min"),
                                                       mon["plain"].get("pswpout", {}).get("first"), mon["plain"].get("pswpout", {}).get("last"),
                                                       mon["plain"].get("pgmajfault", {}).get("first"), mon["plain"].get("pgmajfault", {}).get("last")))
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    say("DONE %s" % os.path.relpath(d, ROOT))


if __name__ == "__main__":
    main()
