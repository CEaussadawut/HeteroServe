#!/usr/bin/env python3
"""Turn measure_link.sh logs into testbed/dataset/link.json. Stdlib only.

    python3 testbed/scripts/link_to_json.py LOG [LOG ...] [--out testbed/dataset/link.json]

One log per pair of boards. Per direction: each transfer's Mbit/s (bytes x 8 / dd's seconds on the sender, SI mega;
or, where the log has RECV lines -- measure_link.sh from 2026-10-09 -- the receiver's: (bytes - its first read) x 8 /
its seconds from the first byte to EOF, with the sender's figures kept beside them as sender_mbps_each), then
min / mean / max over the transfers; RTT from ping each way. A pair's figure is the slower direction's mean
throughput and the mean of the two directions' average RTT.

The output carries every pair under "pairs", and per-node figures under "nodes" in the schema
ThinkPad/control/profiles.link_survey reads. A pooled run's traffic all passes through the host (llama.cpp's RPC
backend copies between devices via the host). "nodes" is the view of a pool the Pi 5 hosts, built from the pairs
with the Pi 5 in them: each other board gets its pair with the Pi 5, and the Pi 5 gets its best pair --
placement.hop_ms takes the slower endpoint, which then reproduces each pair's own figure. A pair without the Pi 5
(Pi 4 <-> Jetson, for a pool the Pi 4 hosts) is under "pairs" only; predict_pool.py builds each host's view from
"pairs".

Refuses a log that did not finish, has fewer than 3 transfers in a direction, has RECV lines that do not match its
transfers one for one (a receiver that failed, or got fewer bytes than were sent), was taken while a board was busy
or the port was already in use, or where a board had more than one interface on the measured network.
"""
import hashlib, json, os, re, statistics, sys


def parse(log):
    text = open(log, encoding="utf-8", errors="replace").read()
    if "LINK DONE" not in text:
        sys.exit("refused: %s did not finish (no LINK DONE)" % log)
    m = re.search(r"^########## (\S+) link (\S+) <-> (\S+),", text, re.M)
    if not m:
        sys.exit("refused: %s has no header" % log)
    when, a, b = m.groups()
    for s in re.finditer(r"^STATE (\S+) .* busy=(\d+) listening_\d+=(\d+)(?: ifaces_on_net=(\d+))?", text, re.M):
        if s.group(2) != "0" or s.group(3) != "0" or (s.group(4) not in (None, "1")):
            sys.exit("refused: %s: %s was busy, its port was taken, or it had %s interfaces on the net"
                     % (log, s.group(1), s.group(4)))
    xfer, sent = {}, {}
    for x in re.finditer(r"^XFER (\S+) (\d+) bytes .* copied, ([\d.]+) s", text, re.M):
        xfer.setdefault(x.group(1), []).append(int(x.group(2)) * 8 / float(x.group(3)) / 1e6)
        sent.setdefault(x.group(1), []).append(int(x.group(2)))
    recv, got = {}, {}
    for x in re.finditer(r"^RECV (\S+) (\d+) bytes, first (\d+), ([\d.]+) s from the first byte to EOF", text, re.M):
        recv.setdefault(x.group(1), []).append((int(x.group(2)) - int(x.group(3))) * 8 / float(x.group(4)) / 1e6)
        got.setdefault(x.group(1), []).append(int(x.group(2)))
    if re.search(r"^RECV ", text, re.M) and got != sent:
        sys.exit("refused: %s: the receivers' lines do not match the transfers one for one: sent %s, received %s"
                 % (log, sent, got))
    rtt, cur = {}, None
    for ln in text.splitlines():
        p = re.match(r"PING (\S+)", ln)
        if p:
            cur = p.group(1); continue
        r = re.search(r"rtt min/avg/max/mdev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", ln)
        if r and cur:
            rtt[cur] = dict(zip(("min", "avg", "max", "mdev"), map(float, r.groups())))
    if len(xfer) != 2 or min(len(v) for v in xfer.values()) < 3 or len(rtt) != 2:
        sys.exit("refused: %s: need >= 3 transfers and one ping each way, got %s / %s"
                 % (log, {k: len(v) for k, v in xfer.items()}, sorted(rtt)))
    dirs = {d: {"mbps_each": [round(x, 1) for x in v], "mbps_mean": round(statistics.fmean(v), 1), "rtt_ms": rtt[d]}
            for d, v in sorted((recv or xfer).items())}
    for d in dirs if recv else ():                       # receiver-timed; a sender-timed log reads exactly as before
        dirs[d]["sender_mbps_each"] = [round(x, 1) for x in xfer[d]]
    return a, b, {**({"timed_at": "receiver"} if recv else {}), "measured": when, "log": os.path.relpath(log),
                  "log_sha256": hashlib.sha256(open(log, "rb").read()).hexdigest(),
                  "mbps": round(min(d["mbps_mean"] for d in dirs.values()), 1),
                  "rtt_ms": round(statistics.fmean(d["rtt_ms"]["avg"] for d in dirs.values()), 3),
                  "directions": dirs}


def main(argv):
    out = "testbed/dataset/link.json"
    if "--out" in argv:
        i = argv.index("--out"); out = argv[i + 1]; del argv[i:i + 2]
    if not argv:
        sys.exit(__doc__)
    pairs = {}
    for log in argv:
        a, b, rec = parse(log)
        pairs["%s-%s" % (a, b)] = rec
    def entry(rec, key):
        return {"mbps": rec["mbps"], "rtt_ms": rec["rtt_ms"], "tier": "gbe", "iface": "eth0",
                "measured_utc": rec["measured"], "warnings": [], "pair": key}
    nodes = {}
    for key, rec in pairs.items():
        a, b = key.split("-")
        if "pi5" not in (a, b):
            continue                  # another host's pair: under "pairs" only
        nodes[b if a == "pi5" else a] = entry(rec, key)
        if "pi5" not in nodes or rec["mbps"] > nodes["pi5"]["mbps"]:
            nodes["pi5"] = entry(rec, key)
    res = {"_source": "measure_link.sh: raw TCP (OpenBSD nc, 1 GiB per transfer, sender-side dd; timed at the receiver "
                      "where a pair has timed_at: receiver, 2026-10-09 on) and ping, on the "
                      "path each pair's pooled runs use (Pi 5 <-> Pi 4: the 192.168.10.x overlay; Pi 5 <-> Jetson and "
                      "Pi 4 <-> Jetson: the router LAN, eth0 only), all across the router's switch",
           "pairs": pairs, "nodes": nodes}
    json.dump(res, open(out, "w"), indent=1)
    for key, rec in pairs.items():
        print("%-11s %6.1f Mbit/s (slower direction's mean)  RTT %.3f ms   %s"
              % (key, rec["mbps"], rec["rtt_ms"], {d: v["mbps_mean"] for d, v in rec["directions"].items()}))
    print("wrote %s; nodes: %s" % (out, {n: (v["mbps"], v["rtt_ms"]) for n, v in nodes.items()}))


if __name__ == "__main__":
    main(sys.argv[1:])
