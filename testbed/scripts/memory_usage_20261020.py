#!/usr/bin/env python3
"""Memory per board for every served placement (the advisor, 2026-10-16: "resource (memory usages)"). Laptop only; reads
the run directories of full_eval.py.

    python3 testbed/scripts/memory_usage_20261020.py [RUN_DIR ...]      # default: every testbed/results/full_eval_*_2026* run

Per (run, placement, board), from round 1's serve log (serve_pool.sh's output; the rounds load the same plan):
  available   MemAvailable when serve_pool.sh's gate read it, MiB
  planned     the plan's need with the board's reserve, as the gate printed it (the engine's validated accounting)
  model / kv / compute   llama.cpp's own buffers on that board's device(s), MiB (load_tensors, llama_kv_cache, sched_reserve);
              a host's CPU buffers are the host's, an RPC<n>[ip:port] device is the board at that address
  min_avail   the lowest MemAvailable the board's 1 Hz poll saw while the placement served (throttle_<node>.log lines with
              avail_mib=, the poll since 2026-10-20; the laptop's laptop_poll.log mem field when present), if recorded
Writes testbed/results/memory_usage_<stamp>/{memory.json, report.txt}.
"""
import datetime, glob, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(ROOT, "testbed", "results")
ADDR = {"192.168.1.114": "jetson", "192.168.1.109": "jetson", "192.168.100.111": "jetson", "192.168.10.11": "pi5",
        "192.168.10.12": "pi4", "192.168.1.108": "laptop", "192.168.1.110": "pi4", "192.168.1.122": "pi5"}
GATE = re.compile(r"^gate\s+(\S+): MemAvailable (\d+) MiB, the plan needs (\d+) MiB")
BUF = re.compile(r"(load_tensors|llama_kv_cache|sched_reserve):\s+(CPU(?:_Mapped)?|RPC\d+\[([\d.]+):\d+\]|Vulkan\d+|CUDA\d+)\s+"
                 r"(model|KV|compute) buffer size =\s+([\d.]+) MiB")


def parse_serve(path, host):
    out = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        g = GATE.search(ln.strip())
        if g:
            out.setdefault(g.group(1), {})["available"] = int(g.group(2))
            out[g.group(1)]["planned"] = int(g.group(3))
        b = BUF.search(ln)
        if b:
            dev, ip, kind, mib = b.group(2), b.group(3), b.group(4).lower(), float(b.group(5))
            node = ADDR.get(ip, ip) if ip else host
            d = out.setdefault(node, {})
            d[kind] = round(d.get(kind, 0.0) + mib, 2)
    return out


def poll_min(run, node, t0, t1):
    """The lowest MemAvailable in [t0, t1]: a Pi's throttle_<node>.log (avail_mib=, the poll since 2026-10-20), or the
    laptop's laptop_poll.log (t=<epoch> ... avail=<MiB>, since 2026-10-11)."""
    if node == "laptop":
        p, ts, key = os.path.join(run, "laptop_poll.log"), None, "avail"
    else:
        p, key = os.path.join(run, "throttle_%s.log" % node), "avail_mib"
    if not os.path.exists(p):
        return None
    vals = []
    for ln in open(p, encoding="utf-8", errors="replace"):
        tok = ln.split()
        if not tok:
            continue
        kv = dict(x.split("=", 1) for x in tok if "=" in x)
        try:
            t = float(kv["t"]) if node == "laptop" else float(tok[0])
        except (KeyError, ValueError):
            continue
        if t0 <= t <= t1 and kv.get(key, "NA").isdigit():
            vals.append(int(kv[key]))
    return min(vals) if vals else None


def main():
    runs = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(RES, "full_eval_*_2026*")) if os.path.isdir(d) and "smoke" not in d)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(RES, "memory_usage_%s" % stamp)
    os.makedirs(out)
    res = {"made": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "runs": {}}
    L = ["Memory per board for every served placement (MiB): available at the gate / planned need with reserve / llama.cpp's "
         "model + KV + compute buffers / lowest MemAvailable while serving (when the poll records it)", ""]
    for run in runs:
        rows = [json.loads(l) for l in open(os.path.join(run, "run.jsonl"))]
        start = next(r for r in rows if r["kind"] == "start")["data"]
        reg = json.load(open(os.path.join(ROOT, start["registration"])))
        windows = {}
        for r in rows:
            if r["kind"] in ("serve start", "serve stop"):
                k = (r["data"]["round"], r["data"]["placement"])
                windows.setdefault(k, [None, None])[0 if r["kind"] == "serve start" else 1] = r["t"]
        rr = {}
        for lab, c in reg["configs"].items():
            logs = sorted(glob.glob(os.path.join(run, "serve_01_%s.log" % lab)))
            if not logs:
                continue
            per = parse_serve(logs[0], c["host"])
            for node in per:
                w = windows.get((1, lab))
                per[node]["min_avail"] = poll_min(run, node, int(w[0]), int(w[1]) + 1) if w and w[0] and w[1] else None
            rr[lab] = {"assign": c["assign"], "head": c["head"], "host": c["host"], "boards": per}
            L.append("%s  %s (%s, head %s)" % (os.path.basename(run), lab, " + ".join("%s %d" % tuple(x) for x in c["assign"]), c["head"]))
            for node, v in per.items():
                L.append("    %-7s available %6s  planned %6s  model %8s  kv %7s  compute %7s  min while serving %s" % (
                    node, v.get("available", "-"), v.get("planned", "-"), v.get("model", "-"), v.get("kv", "-"),
                    v.get("compute", "-"), v.get("min_avail") if v.get("min_avail") is not None else "not polled"))
        res["runs"][os.path.relpath(run, ROOT)] = rr
    json.dump(res, open(os.path.join(out, "memory.json"), "w"), indent=1)
    open(os.path.join(out, "report.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
