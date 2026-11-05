#!/usr/bin/env python3
"""HeteroServe Pi 5 node agent.

A deliberately small HTTP service the orchestrator on the control node polls
before it commits a placement. Step 1 of the serving pipeline in the proposal
(Appendix B.8) is "verify readiness of every node: rpc-server status, free memory,
network link, power sensors" -- this is the node half of that check.

Read-only by design apart from the run tag and the phase-marker log, neither of
which changes how the node computes anything. It does not start, stop or
reconfigure anything: a node that can be reconfigured mid-sweep over HTTP is a
node whose measurements cannot be trusted. That is why ThinkPad/control/nodes.py's
POST /rpc/start and /rpc/stop are deliberately NOT served here -- see README.

    GET /health     readiness verdict; 200 when ready, 503 when blocked
    GET /ready      identical to /health (kept: 06_verify_node.sh polls it)
    GET /info       static node identity and capability card
    GET /stat       one live telemetry sample
    GET /metrics    identical to /stat (the name the control node uses)
    GET /events?n=  tail of the phase-marker log
    POST /run-tag   {"tag": "..."} -- record the current harness cell id
    POST /mark      {"event": "...", "note": "..."} -- append a phase marker

/health used to answer 200 with a liveness body carrying no `ready` and no
`blocking`. ThinkPad/control/nodes.py:139-145 reads exactly those two keys, so this
node contributed no blocking reasons however throttled, hot or swapped it was, and
the cluster preflight gate declared READY regardless. The readiness logic was
already here -- it was just served at a path the control node never calls.
"""
from __future__ import annotations

import json
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envfile  # noqa: E402
import telemetry  # noqa: E402

ENV = envfile.load()
STARTED = time.time()
STATE = {"run_tag": "", "run_tag_set_utc": ""}
LOCK = threading.Lock()


def _port_open(host, port, timeout=0.4):
    target = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    try:
        with socket.create_connection((target, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _iface(name):
    base = Path("/sys/class/net") / name
    if not base.exists():
        return {"present": False}
    def rd(f, default="?"):
        try:
            return (base / f).read_text().strip()
        except OSError:
            return default
    return {
        "present": True,
        "operstate": rd("operstate"),
        "speed_mbit": rd("speed", "-"),
        "carrier": rd("carrier", "0"),
    }


def info():
    return {
        "node_id": ENV["NODE_ID"],
        "role": ENV.get("NODE_ROLE", ""),
        "desc": ENV.get("NODE_DESC", ""),
        "soc": ENV.get("NODE_SOC", ""),
        "cpu": ENV.get("NODE_CPU", ""),
        "cores": envfile.get_int(ENV, "NODE_CORES", 4),
        "arch": ENV.get("NODE_ARCH", ""),
        "ram_mb": envfile.get_int(ENV, "NODE_RAM_MB", 0),
        "usable_mb": envfile.get_int(ENV, "NODE_USABLE_MB", 0),
        # What the placement engine is allowed to assign. Always below usable_mb.
        "rpc_mem_mb": envfile.get_int(ENV, "RPC_MEM_MB", 0),
        "rpc_endpoint": "%s:%s" % (ENV.get("ETH_ADDR", ""), ENV.get("RPC_PORT", "")),
        "llama_cpp_sha": ENV.get("LLAMA_PIN_SHA", ""),
        "arch_flags": ENV.get("LLAMA_ARCH_FLAGS", ""),
        "threads": envfile.get_int(ENV, "LLAMA_THREADS", 4),
        "ina_channel": envfile.get_int(ENV, "INA_CHANNEL", 0),
        "agent_started_utc": datetime.fromtimestamp(STARTED, timezone.utc).isoformat(
            timespec="seconds"
        ),
        # The flat keys above are this node's own card. The two nested blocks below are
        # what the control node actually reads, and it reads them with `.get(...) or {}`
        # so a missing block is silent:
        #   nodes.py:188      (info(node).get("software") or {}).get("llama_pin_sha")
        #                     -> the cluster-wide revision gate. Absent = None = the
        #                        blocking pin check quietly degrades to a warning, and the
        #                        whole point of pinning one SHA across nodes is lost.
        #   profiles.py:154-156  inv.get("memory", {}).get("usable_budget_mb")
        #                     -> absent = fall back to cluster.json's configured figure,
        #                        i.e. a number from a config file where a measured one was
        #                        expected.
        "software": {
            "llama_pin_sha": ENV.get("LLAMA_PIN_SHA", ""),
            "arch_flags": ENV.get("LLAMA_ARCH_FLAGS", ""),
            "threads": envfile.get_int(ENV, "LLAMA_THREADS", 4),
        },
        "memory": {
            # RPC_MEM_MB, not NODE_USABLE_MB: advertising less than the node has is
            # deliberate, so the placement engine never hands it a partition that OOMs
            # mid-run. This is the budget the memory constraint mem_i <= M_i is about.
            "usable_budget_mb": envfile.get_int(ENV, "RPC_MEM_MB", 0),
            "node_usable_mb": envfile.get_int(ENV, "NODE_USABLE_MB", 0),
            "total_mb": envfile.get_int(ENV, "NODE_RAM_MB", 0),
        },
    }


def readiness():
    """Every gate that, if failed, would silently produce an unusable measurement."""
    s = telemetry.sample(ENV["NODE_ID"])
    gates = {}

    gates["rpc_server_listening"] = _port_open(
        ENV.get("RPC_HOST", "0.0.0.0"), ENV.get("RPC_PORT", "50052")
    )
    gates["governor_pinned"] = s["governor"] == ENV.get("CPU_GOVERNOR", "performance")
    gates["no_swap_in_use"] = s["swap_used_mb"] == 0

    # Fail closed, and keep "unknown" separate from "clean". telemetry.read_throttled
    # returns -1 when vcgencmd is missing, off the unit's PATH, or times out, which
    # telemetry renders as "" -- and `in (0, "")` read that as NOT THROTTLED. A node whose
    # firmware word cannot be read would have passed the readiness gate and gone on to
    # produce measurements nobody could validate. Pi 4 states the same rule in its own
    # parser: "fail closed: never assume 'not throttled'".
    gates["throttle_word_readable"] = s["throttled_now"] != ""
    gates["not_throttled"] = s["throttled_now"] == 0
    gates["no_under_voltage"] = s["under_voltage_now"] == 0

    # A thermal zone that cannot be read yields 0.0 (scripts/lib/common.sh's `echo 0`
    # fallback, and -0.0 once it round-trips through a float), which is not a cold board
    # and must not pass a gate whose whole job is refusing to start warm.
    start_max = float(ENV.get("THERMAL_START_MAX_C", 55))
    t = s["soc_temp_c"]
    gates["cool_enough_to_start"] = (
        isinstance(t, (int, float)) and 0.0 < t <= start_max
    )

    # The node must still be able to honour the memory it advertises.
    gates["advertised_memory_available"] = (
        s["mem_avail_mb"] >= envfile.get_int(ENV, "RPC_MEM_MB", 0)
        or s["rpc_rss_mb"] > 0  # already loaded: the memory is in use, not missing
    )

    link = _iface(ENV.get("ETH_IFACE", "eth0"))
    gates["wired_link_up"] = link.get("operstate") == "up"

    return gates, s, link


# Phase markers live beside the telemetry they annotate. The control node POSTs
# run_begin/run_end here and later reads them back through /events; without them
# ThinkPad/control/analysis.py finds no window for this node, its mean active power
# comes out None, and profiles.build_devices refuses a Pi 5 profile with no clue why.
EVENTS = Path(ENV.get("DATA_DIR", "/opt/heteroserve/data")) / "logs" / "events.jsonl"

# Gate name -> the sentence the control node prints in its preflight report. A bare
# gate key like "no_swap_in_use" is not much use to someone reading a refusal.
_BLOCKING_TEXT = {
    "governor_pinned": "cpu governor is not pinned to performance",
    "no_swap_in_use": "swap is in use -- a run that swaps is a corrupted measurement",
    "not_throttled": "SoC reports thermal/voltage throttling now",
    "no_under_voltage": "SoC reports under-voltage now",
    "cool_enough_to_start": "SoC is above the start-temperature gate",
    "advertised_memory_available": "less free memory than this node advertises as RPC_MEM_MB",
    "wired_link_up": "wired link is down",
    "rpc_server_listening": "rpc-server is not accepting connections",
    "throttle_word_readable": "cannot read the firmware throttle word -- vcgencmd missing, "
                             "off the unit's PATH, or timing out",
}


def health():
    """The readiness verdict, in the shape ThinkPad/control/nodes.py consumes.

    `blocking` is the load-bearing key: anything in it means do not record
    measurements from this node, and the control node is expected to refuse the run
    rather than quietly produce a bad row. `warnings` never blocks.
    """
    gates, s, link = readiness()
    blocking = [_BLOCKING_TEXT.get(k, k) for k, v in gates.items() if not v]
    warnings = []
    if not ENV.get("LLAMA_PIN_SHA"):
        warnings.append("llama.cpp revision not pinned in config/node.env")
    return {
        "node_id": ENV["NODE_ID"],
        "ready": not blocking,
        "blocking": blocking,
        "warnings": warnings,
        "failed": [k for k, v in gates.items() if not v],
        "gates": gates,
        "sample": s,
        # nodes.py:161 reads (h.get("metrics") or {}).get("soc_temp_c") when it builds the
        # cluster preflight report. Under "sample" alone it printed temp=None for this node.
        "metrics": s,
        "wired_link": link,
        "uptime_s": round(time.time() - STARTED, 1),
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def append_event(event, note=""):
    """One phase marker, in the shape ThinkPad/control/analysis.py actually reads.

    The key names are not cosmetic. windows_from_events does

        node, event, ts = rec.get("node"), rec.get("event", ""), rec.get("ts")
        if not (node and ts):
            continue

    so a record keyed `node_id`/`utc` is skipped without a message -- this node would
    contribute no windows at all, its mean active power would come out None, and
    profiles.build_devices would refuse a Pi 5 profile blaming "unmeasured inputs". `ts`
    must also be an epoch float, matching Pi4/node/node_agent.py and the DAQ's own time
    base; an ISO-8601 string is truthy and would pass the guard above, then be compared
    against float power-sample timestamps.

    temp_c and throttled mirror Pi 4's record so a marker carries the conditions it was
    taken under. The extra keys (utc, monotonic_s, run_tag) are this node's own and are
    ignored by the control node -- monotonic_s survives an NTP step mid-run, which the
    wall clock does not.
    """
    sample = {}
    try:
        sample = telemetry.sample(ENV["NODE_ID"]) or {}
    except Exception:                                    # never let telemetry break a mark
        pass
    row = {
        # --- the contract the control node reads ---
        "ts": time.time(),
        "node": ENV["NODE_ID"],
        "event": str(event),
        "note": str(note),
        "temp_c": sample.get("soc_temp_c", ""),
        "throttled": sample.get("throttled_hex", ""),
        # --- this node's own provenance ---
        "node_id": ENV["NODE_ID"],
        "utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "monotonic_s": round(time.monotonic(), 6),
    }
    with LOCK:
        row["run_tag"] = STATE.get("run_tag", "")
        EVENTS.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
    return row


def tail_events(n=50):
    try:
        lines = EVENTS.read_text(encoding="utf-8").splitlines()[-n:]
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "HeteroServeAgent/1.0"

    def log_message(self, fmt, *args):  # keep the journal readable
        pass

    def _send(self, code, payload):
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        split = self.path.split("?", 1)
        path = split[0].rstrip("/") or "/"
        query = parse_qs(split[1]) if len(split) > 1 else {}
        # /health and /ready are the same verdict. The control node polls /health;
        # 06_verify_node.sh polls /ready. Serving one of them a body without `ready`
        # is how this node used to pass the cluster preflight gate unchecked.
        if path in ("/", "/health", "/ready"):
            h = health()
            self._send(200 if h["ready"] else 503, h)
        elif path == "/info":
            self._send(200, info())
        elif path in ("/stat", "/metrics"):
            with LOCK:
                tag = dict(STATE)
            self._send(200, {"sample": telemetry.sample(ENV["NODE_ID"]), "run": tag})
        elif path == "/events":
            try:
                n = int(query.get("n", ["50"])[0])
            except ValueError:
                n = 50
            self._send(200, tail_events(max(1, min(n, 5000))))
        else:
            self._send(404, {"error": "no such endpoint", "path": path,
                             "routes": ["/health", "/ready", "/info", "/stat",
                                        "/metrics", "/events"]})

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        # /rpc/start and /rpc/stop are intentionally absent -- see the module docstring.
        # They are named here so a 404 reads as a decision rather than an oversight.
        if path not in ("/run-tag", "/mark"):
            self._send(404, {"error": "no such endpoint", "path": path,
                             "routes": ["/run-tag", "/mark"],
                             "note": "this agent does not start or stop services"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "expected a JSON body"})
            return
        # `5`, `"x"`, `[1,2]` and `null` are all valid JSON that json.loads accepts, and
        # .get() on any of them raises AttributeError -- which the clause above does not
        # catch, so the handler died and the client got no response at all.
        if not isinstance(body, dict):
            self._send(400, {"error": "expected a JSON object", "got": type(body).__name__})
            return
        tag = body.get("tag", "")

        if path == "/mark":
            self._send(200, append_event(body.get("event", "mark"), body.get("note", "")))
            return
        with LOCK:
            STATE["run_tag"] = str(tag)
            STATE["run_tag_set_utc"] = datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            )
            snapshot = dict(STATE)
        self._send(200, snapshot)


def main():
    port = envfile.get_int(ENV, "AGENT_PORT", 9410)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    sys.stderr.write("agent for %s listening on :%d\n" % (ENV["NODE_ID"], port))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
