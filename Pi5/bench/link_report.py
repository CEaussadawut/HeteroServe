#!/usr/bin/env python3
"""Fold iperf3 and ping output into the link_survey.json the placement engine reads.

ThinkPad/control/profiles.py consumes exactly this shape:

    {"nodes": {"<node_id>": {"mbps": <float>, "rtt_ms": <float>}}}

and feeds it into DeviceProfile.link_mbps / rtt_ms, which are the only measured inputs to
C -- the communication term in T = sum_i n_i*c_i + C. Everything else this file writes is
detail that profiles.py ignores by construction (it reads those two keys and nothing
else), kept because a number without its conditions is not reproducible.

Two choices in here are worth stating plainly rather than leaving in the arithmetic:

1. `mbps` for the contract is the **slower of the two directions**, not their mean. A hop
   costs what its bottleneck costs, and the Pi 5's 2.5GbE is a USB 3.0 adapter, which is
   not symmetric. Averaging would flatter the link in exactly the case the study cares
   about. Both directions are recorded separately so the asymmetry stays visible.

2. A link with packet loss or TCP retransmits is reported with a `warnings` list rather
   than silently averaged in. Loss on a wired LAN segment means something is wrong with
   the switch, the cable or the adapter, and a placement computed from that number is
   measuring a fault, not a topology.

Usage (driven by scripts/11_link_survey.sh, not normally by hand):
    link_report.py --accumulate FILE --ping P.txt --forward F.json [--reverse R.json]
    link_report.py --finish FILE --out link_survey.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RTT_RE = re.compile(
    r"rtt min/avg/max/mdev\s*=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)")
LOSS_RE = re.compile(r"([\d.]+)%\s*packet loss")
SENT_RE = re.compile(r"(\d+)\s+packets transmitted,\s*(\d+)\s*received")


def parse_ping(text: str) -> dict:
    out = {"rtt_min_ms": None, "rtt_ms": None, "rtt_max_ms": None,
           "rtt_mdev_ms": None, "loss_pct": None, "packets": None}
    m = RTT_RE.search(text)
    if m:
        out["rtt_min_ms"] = float(m.group(1))
        out["rtt_ms"] = float(m.group(2))
        out["rtt_max_ms"] = float(m.group(3))
        out["rtt_mdev_ms"] = float(m.group(4))
    m = LOSS_RE.search(text)
    if m:
        out["loss_pct"] = float(m.group(1))
    m = SENT_RE.search(text)
    if m:
        out["packets"] = int(m.group(1))
    return out


def parse_iperf(path) -> dict:
    """Goodput actually delivered, in Mbit/s, plus the health signals around it."""
    p = Path(path)
    if not p.is_file() or not p.read_text(encoding="utf-8").strip():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    if data.get("error"):
        return {"error": data["error"]}
    end = data.get("end", {})
    # sum_received is what arrived, in both normal and -R runs, so it is the same
    # quantity in both directions: delivered goodput, not offered load.
    recv = end.get("sum_received") or {}
    sent = end.get("sum_sent") or {}
    bps = recv.get("bits_per_second") or sent.get("bits_per_second")
    return {
        "mbps": round(bps / 1e6, 2) if bps else None,
        "retransmits": sent.get("retransmits"),
        "seconds": round(recv.get("seconds") or sent.get("seconds") or 0.0, 2),
        "bytes": recv.get("bytes") or sent.get("bytes"),
        "iperf_version": data.get("start", {}).get("version"),
        "protocol": (data.get("start", {}).get("test_start") or {}).get("protocol"),
    }


def accumulate(args) -> int:
    store = Path(args.accumulate)
    rows = json.loads(store.read_text(encoding="utf-8")) if store.is_file() else []

    ping = parse_ping(Path(args.ping).read_text(encoding="utf-8")) if args.ping else {}
    fwd = parse_iperf(args.forward) if args.forward else {}
    rev = parse_iperf(args.reverse) if args.reverse else {}

    warnings = []
    # `if ping.get("loss_pct")` conflated 0.0 (measured, clean) with None (the summary did
    # not parse -- a non-English locale renders "packet loss" differently, and the ping in
    # 11_link_survey.sh inherits the node's locale). A link whose loss is unknown was
    # published as a clean link with an empty warnings list.
    if ping.get("loss_pct") is None:
        if args.ping:
            warnings.append("ping summary not parsed -- packet loss unknown")
    elif ping["loss_pct"] > 0:
        warnings.append("%.1f%% ICMP packet loss" % ping["loss_pct"])
    for label, leg in (("forward", fwd), ("reverse", rev)):
        if leg.get("error"):
            warnings.append("%s: %s" % (label, leg["error"]))
        rt = leg.get("retransmits")
        if rt:
            warnings.append("%s: %d TCP retransmits" % (label, rt))

    speeds = [leg["mbps"] for leg in (fwd, rev) if leg.get("mbps")]
    # The bottleneck, not the average -- see the module docstring.
    mbps = min(speeds) if speeds else None

    # ...but it is only a bottleneck if both directions were actually measured. A failed
    # reverse leg leaves 11_link_survey.sh writing an empty rev.json, which parses to {},
    # which drops out of `speeds` -- so the forward number would be published as "the
    # slower of the two directions" with nothing saying only one was taken. The USB 3.0
    # 2.5GbE adapter is not symmetric, which is exactly why both legs are run.
    if mbps and len(speeds) < 2:
        absent = [label for label, leg in (("forward", fwd), ("reverse", rev))
                  if not leg.get("mbps")]
        warnings.append(
            "only one direction measured (%s leg produced no result); mbps is that single "
            "direction, not a bottleneck" % "+".join(absent))

    if mbps and os.environ.get("LINK_SPEED", "").isdigit():
        nominal = int(os.environ["LINK_SPEED"])
        if nominal and mbps < 0.5 * nominal:
            warnings.append(
                "delivered %.0f Mb/s against a %d Mb/s negotiated link (<50%%)" % (mbps, nominal))

    rows.append({
        "peer": os.environ.get("PEER_ID", ""),
        "peer_host": os.environ.get("PEER_HOST", ""),
        "peer_role": os.environ.get("PEER_ROLE", ""),
        "iface": os.environ.get("IFACE", ""),
        "tier": os.environ.get("TIER", ""),
        "negotiated_mbps": int(os.environ["LINK_SPEED"])
        if os.environ.get("LINK_SPEED", "").isdigit() else None,
        "mtu": int(os.environ["MTU"]) if os.environ.get("MTU", "").isdigit() else None,
        "mbps": mbps,
        "mbps_forward": fwd.get("mbps"),
        "mbps_reverse": rev.get("mbps"),
        "rtt_ms": ping.get("rtt_ms"),
        "rtt_min_ms": ping.get("rtt_min_ms"),
        "rtt_max_ms": ping.get("rtt_max_ms"),
        "rtt_mdev_ms": ping.get("rtt_mdev_ms"),
        "loss_pct": ping.get("loss_pct"),
        "icmp_packets": ping.get("packets"),
        "retransmits_forward": fwd.get("retransmits"),
        "retransmits_reverse": rev.get("retransmits"),
        "iperf_version": fwd.get("iperf_version") or rev.get("iperf_version"),
        "warnings": warnings,
    })
    store.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    label = "%.0f Mb/s" % mbps if mbps else "no bandwidth"
    rtt = "%.3f ms" % ping["rtt_ms"] if ping.get("rtt_ms") else "no rtt"
    print("    %s  %s  %s%s" % (os.environ.get("PEER_ID", "?"), label, rtt,
                                ("  [%s]" % "; ".join(warnings)) if warnings else ""))
    return 0


def finish(args) -> int:
    rows = json.loads(Path(args.finish).read_text(encoding="utf-8"))
    node_id = os.environ.get("NODE_ID", "")
    orch = os.environ.get("ORCH_ID", "")

    # profiles.py keys this node's link by the node's own id, and the link it means is
    # the one to the control node. Peer-to-peer rows stay in "links" for the pipeline
    # study; they are not what DeviceProfile.link_mbps describes.
    to_orch = [r for r in rows if r["peer"] == orch and r.get("mbps")]
    nodes = {}
    if to_orch:
        best = max(to_orch, key=lambda r: r["mbps"])  # the tier actually in use
        nodes[node_id] = {
            "mbps": best["mbps"],
            "rtt_ms": best["rtt_ms"],
            "measured_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "tier": best["tier"],
            "iface": best["iface"],
            "peer": best["peer"],
            "mbps_forward": best["mbps_forward"],
            "mbps_reverse": best["mbps_reverse"],
            "warnings": best["warnings"],
        }

    doc = {
        "schema": "heteroserve.link_survey/1",
        "source_node": node_id,
        "run_id": os.environ.get("RUN_ID", ""),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "nodes": nodes,
        "links": rows,
        "provenance": {
            "soc_temp_start_c": os.environ.get("TEMP_START", ""),
            "soc_temp_end_c": os.environ.get("TEMP_END", ""),
            # Recorded, not asserted. 11_link_survey.sh refuses to run under netem and
            # passes what it actually saw on the interface; a hand-run report that did not
            # make that check says so rather than inheriting the claim.
            "qdisc_observed": os.environ.get("QDISC_OBSERVED", "not recorded by the caller"),
        },
    }
    Path(args.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")

    if not nodes:
        # Only an error when the orchestrator was in scope. `--peer pi4` deliberately
        # surveys a pipeline hop and has no orchestrator row; failing there stopped the
        # caller before it could report where the file went.
        if os.environ.get("EXPECT_ORCH", "1") == "0":
            print("note: no orchestrator link in this run, so the file carries no \"nodes\" "
                  "entry. Survey the orchestrator before computing a placement.",
                  file=sys.stderr)
        else:
            print("WARNING: no usable link to the orchestrator (%r) was measured, so this "
                  "file carries no \"nodes\" entry and profiles.py will fall back to its "
                  "1000 Mb/s default. Survey the orchestrator before computing a placement."
                  % (orch or "unset"), file=sys.stderr)
            return 1
    flagged = [r for r in rows if r.get("warnings")]
    if flagged:
        print("WARNING: %d link(s) measured with warnings:" % len(flagged), file=sys.stderr)
        for r in flagged:
            print("  %s: %s" % (r["peer"], "; ".join(r["warnings"])), file=sys.stderr)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--accumulate")
    ap.add_argument("--ping")
    ap.add_argument("--forward")
    ap.add_argument("--reverse")
    ap.add_argument("--finish")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.accumulate:
        return accumulate(args)
    if args.finish:
        if not args.out:
            ap.error("--finish needs --out")
        return finish(args)
    ap.error("one of --accumulate or --finish is required")


if __name__ == "__main__":
    raise SystemExit(main())
