#!/usr/bin/env python3
"""A dated snapshot of each board's free memory, in memory_at_rest.json's format, so that a serving start can be planned
against the boards as they are now (serve_pool.sh MEMFILE=<this file>). From the repo root:

    python3 testbed/scripts/memory_snapshot.py [--reads 10]

Why (2026-09-30): the QR3 pilot's serving plan for Qwen3-14B at a 4096-token context was refused by serve_pool.sh's gate. The
Pi 5 had 7,435 MiB available, against the 7,619 of memory_at_rest.json (read 2026-09-27 ~19:05, before the Pis' Wi-Fi was
connected), and the plan's margin was 89 MiB. The at-rest file stays as it is: every registration before this one hashes it.
Each board: MemTotal once, MemAvailable --reads times 1 s apart. mem_available_mib is the LOWEST reading, so the planning
figure is never the lucky one; the range is kept beside it. It refuses if any llama.cpp process runs on a board (not at rest).
Written to testbed/dataset/memory_snapshot_<stamp>.json, read-only.
"""
import argparse, datetime, json, os, stat, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
BOARDS = ("pi5", "pi4", "jetson", "laptop")   # the laptop is a node since 2026-10-11 (read here: no sshd)


def ssh(node, cmd):
    argv = ["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd]
    p = subprocess.run(argv, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    if p.returncode:
        raise SystemExit("FATAL: ssh %s failed (%d): %s" % (node, p.returncode, p.stderr.strip()))
    return p.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", type=int, default=10)
    a = ap.parse_args()
    now = datetime.datetime.now().astimezone()
    out = {"_source": "MemAvailable from /proc/meminfo, lowest of %d reads 1 s apart per board, read over ssh %s by "
                      "testbed/scripts/memory_snapshot.py; boards idle (no llama.cpp process), as found (desktop session, Wi-Fi as "
                      "configured, page cache as left by earlier runs)" % (a.reads, now.isoformat(timespec="seconds"))}
    for n in BOARDS:
        busy = ssh(n, "for p in llama-bench llama-batched-b llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null "
                      "&& printf '%s ' $p; done; true").strip()
        if busy:
            raise SystemExit("FATAL: %s is not at rest: %s" % (n, busy))
        r = ssh(n, "awk '/MemTotal/ {print $2}' /proc/meminfo; for i in $(seq %d); do awk '/MemAvailable/ {print $2}' /proc/meminfo; "
                   "sleep 1; done; cut -d' ' -f1-3 /proc/loadavg; ip -4 -o addr show | awk '{print $2\"=\"$4}' | tr '\\n' ' '"
                % a.reads).split("\n")
        total, av = int(r[0]), [int(x) for x in r[1:1 + a.reads]]
        out[n] = {"mem_total_mib": total // 1024, "mem_available_mib": min(av) // 1024,
                  "mem_available_range_mib": [min(av) // 1024, max(av) // 1024], "reads": a.reads,
                  "load": r[1 + a.reads].strip(), "interfaces": r[2 + a.reads].strip()}
        print(n, out[n])
    path = os.path.join(ROOT, "testbed", "dataset", "memory_snapshot_%s.json" % now.strftime("%Y%m%d_%H%M%S"))
    with open(path, "w") as f:
        json.dump(out, f, indent=1); f.write("\n")
    os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
