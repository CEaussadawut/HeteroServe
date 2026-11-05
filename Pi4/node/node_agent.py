#!/usr/bin/env python3
"""HeteroServe node agent - the control plane's handle on this Pi 4.

The cluster manager on the T580 needs three things from every worker: what the node is,
whether it is fit to take part in a run right now, and the ability to start/stop the
rpc-server without an interactive ssh session. That is all this serves.

Standard library only, single-threaded, no auth: the testbed is a private wired segment
behind a managed switch. Do not put it on a routable network.

    python3 node/node_agent.py             # listen on $AGENT_PORT

Endpoints
    GET  /info      full device inventory (device_info.py)
    GET  /health    readiness summary, HTTP 200 ready / 503 not ready
    GET  /metrics   cheap live sample: temp, clock, throttled word, load, free RAM
    GET  /events?n= tail of the phase-marker log
    POST /rpc/start start rpc-server via systemd
    POST /rpc/stop  stop it
    POST /mark      body {"event":..., "note":...} - inject a phase marker from the
                    control node so a cluster-wide run is bracketed on every node at once
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import device_info  # noqa: E402
import envfile  # noqa: E402

ENV = envfile.load()
ROOT = HERE.parent
DATA_DIR = Path(ENV.get("DATA_DIR", "/opt/heteroserve/data"))
EVENTS = DATA_DIR / "logs" / "events.jsonl"


def sh(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except Exception as exc:  # noqa: BLE001
        return 127, "", str(exc)


def metrics():
    thermal = device_info.thermal()
    mem = device_info.memory()
    _, clock, _ = sh(["vcgencmd", "measure_clock", "arm"])
    mhz = None
    if "=" in clock:
        try:
            mhz = int(clock.split("=")[1]) // 1_000_000
        except ValueError:
            mhz = None
    try:
        load1, load5, load15 = (float(x) for x in Path("/proc/loadavg").read_text().split()[:3])
    except Exception:
        load1 = load5 = load15 = None
    return {
        "ts": time.time(),
        "node": ENV.get("NODE_ID"),
        "soc_temp_c": thermal["soc_temp_c"],
        "throttled_word": thermal["throttled_word"],
        "arm_mhz": mhz,
        "governor": device_info.cpu()["governor"],
        "mem_available_mb": mem["available_mb"],
        "swap_total_mb": mem["swap_total_mb"],
        "load": [load1, load5, load15],
        "rpc_active": sh(["systemctl", "is-active", "--quiet", "heteroserve-rpc"])[0] == 0,
        # `is-active` cannot distinguish "serving steadily" from "crash-looping every
        # 30 seconds" -- both read active. NRestarts is the only cheap signal that
        # separates them, and nothing recorded it before, so a slow restart loop was
        # invisible to the control node and to the dataset.
        "rpc_restarts": _unit_restarts("heteroserve-rpc"),
    }


def _unit_restarts(unit):
    """systemd's restart counter for a unit, or None if it cannot be read."""
    rc, out, _ = sh(["systemctl", "show", "-p", "NRestarts", "--value", unit])
    if rc != 0:
        return None
    try:
        return int(out.strip())
    except (TypeError, ValueError):
        return None


# The rpc server target was renamed rpc-server -> ggml-rpc-server upstream in
# c299a92 (2026-06-27), and the pinned SHA is chosen on the Jetson, so the name
# is not knowable here. Mirror bin_rpc in scripts/lib.sh: newest name first.
# (llama-rpc-server is a last resort; note it truncates to "llama-rpc-serve" in
# /proc/<pid>/comm, which breaks any comm-based process match.)
RPC_BIN_CANDIDATES = ("ggml-rpc-server", "rpc-server", "llama-rpc-server")


try:
    RPC_PORT = int(ENV.get("RPC_PORT", "") or 50052)
except ValueError:
    RPC_PORT = 50052


def rpc_binary():
    """Path to whichever rpc server binary this build produced, or None."""
    bindir = ENV.get("LLAMA_BIN", "")
    if not bindir:
        return None
    for name in RPC_BIN_CANDIDATES:
        candidate = Path(bindir) / name
        if candidate.exists():
            return candidate
    return None


def rpc_listening(port=None, attempts=12, delay=0.25):
    """True once something accepts a TCP connection on the rpc port.

    The port is the only honest liveness signal for this unit. `systemctl start` returns
    when the process is forked, and upstream rpc-server exits 0 on a bad flag, so neither
    the exit code nor the unit's ActiveState distinguishes "serving" from "printed usage
    and quit". Polls briefly because binding is not instantaneous after fork.
    """
    port = RPC_PORT if port is None else port
    for _ in range(max(1, attempts)):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return True
        except OSError:
            pass
        finally:
            sock.close()
        time.sleep(delay)
    return False


def _net_tier_declared_shaped():
    """(declared_shaped, raw_declaration) from the tier state file scripts/lib.sh writes."""
    path = DATA_DIR / "inventory" / "net_tier.json"
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None
    return bool(rec.get("shaped")), rec


def _measurement_preconditions():
    """Yield (name, detail) for every unmet precondition, mirroring 99_healthcheck.sh.

    Kept as a generator of failures rather than a list of booleans so the caller gets the
    reason, not just a verdict -- the control node prints `blocking` verbatim to whoever
    aborted the run, and "not ready" with no cause is how an operator starts guessing.
    """
    eth = ENV.get("ETH_IFACE", "eth0")

    # Shaping: NOT "any netem is bad". The interconnect sweep shapes the link on purpose,
    # so the gate compares the live qdisc against the tier that was declared. Undeclared
    # shaping silently degrades this run and every later one; a declared tier that has
    # vanished mislabels the row with a link it is not running.
    qdisc = (device_info.network().get(eth) or {}).get("qdisc", "") or ""
    live_shaped = ("netem" in qdisc) or ("htb" in qdisc)
    declared_shaped, decl = _net_tier_declared_shaped()
    if live_shaped != declared_shaped:
        yield ("tc shaping does not match the declared tier",
               "live=%s declared=%s (%s); re-declare with scripts/05_net_tier.sh"
               % ("shaped" if live_shaped else "unshaped",
                  "shaped" if declared_shaped else "unshaped",
                  decl if decl else "no declaration on file"))

    # systemd accepts MemoryMax= whether or not the kernel exposes the memory controller.
    # Raspberry Pi OS ships it off unless cgroup_enable=memory is in /boot/cmdline.txt, and
    # without it heteroserve-rpc.service's ceiling enforces nothing on a swapless board.
    v2 = Path("/sys/fs/cgroup/cgroup.controllers")
    try:
        has_cgroup_mem = ("memory" in v2.read_text(encoding="utf-8").split()) if v2.exists() \
            else Path("/sys/fs/cgroup/memory").is_dir()
    except OSError:
        has_cgroup_mem = False
    if not has_cgroup_mem:
        yield ("memory cgroup controller absent",
               "MemoryMax in heteroserve-rpc.service is silently inert; add "
               "'cgroup_enable=memory cgroup_memory=1' to /boot/cmdline.txt and reboot")

    avail = device_info.memory().get("available_mb") or 0
    if avail <= 3000:
        yield ("free memory %s MB is at or below 3000" % avail,
               "something is holding RAM; the placement budget assumes it is free")

    bindir = Path(ENV.get("LLAMA_BIN", "") or ".")
    bench = bindir / "llama-bench"
    if not bench.exists():
        yield ("llama-bench not built", "run scripts/02_build_llama_cpp.sh")
    else:
        # Without --progress the energy window falls back to the whole llama-bench
        # process, which overstates joules-per-token by tens of percent -- and the CSV
        # still looks self-consistent, so nothing downstream would reveal it.
        rc, out, err = sh([str(bench), "--help"], timeout=15)
        if "--progress" not in (out + err):
            yield ("llama-bench has no --progress",
                   "energy windows would silently fall back to whole-process and would not "
                   "be comparable with the rest of the dataset")

    missing_tooling = [n for n in ("stamp_lines.py", "bench_windows.py")
                       if not (ROOT / "node" / n).exists()]
    if missing_tooling:
        yield ("window tooling missing: " + ", ".join(missing_tooling),
               "rsync the node/ folder again")

    manifest = DATA_DIR / "inventory" / "build_manifest.json"
    if not manifest.exists():
        yield ("build manifest absent",
               "rebuild to regenerate; without it the rows are not reproducible")
    else:
        # A pin that does not describe the binary on disk is worse than no pin: the row
        # would record a revision that never ran. Edit the pin without rebuilding, or
        # rebuild without updating the pin, and nothing else would notice.
        pin = ENV.get("LLAMA_PIN_SHA") or ""
        try:
            built = (json.loads(manifest.read_text(encoding="utf-8"))
                     .get("llama_cpp", {}).get("commit") or "")
        except (OSError, json.JSONDecodeError, AttributeError):
            built = ""
        if pin and built and pin != built:
            yield ("pinned revision does not match the built commit",
                   "config/node.env pins %s but the build manifest records %s -- rebuild, "
                   "or correct the pin" % (pin[:12], built[:12]))


def health():
    """Same contract as scripts/99_healthcheck.sh, in a form the manager can consume.

    Anything in `blocking` means: do not record measurements from this node. The control
    node is expected to refuse to start a run rather than quietly produce a bad row.
    """
    m = metrics()
    thermal = device_info.thermal()
    mem = device_info.memory()
    blocking, warnings = [], []

    if thermal["throttled_word"] not in ("0x0", "unavailable"):
        decoded = thermal["throttled_decoded"] or {}
        now = [k for k, v in decoded.items() if v and k.endswith("_now")]
        (blocking if now else warnings).append("throttled=" + thermal["throttled_word"])
    if m["governor"] != "performance":
        blocking.append("governor=" + str(m["governor"]))
    if mem["swap_total_mb"] > 0:
        blocking.append("swap enabled (" + str(mem["swap_total_mb"]) + " MB)")
    # Fail CLOSED on an unreadable sensor. `(x or 0) >= 70` read None as 0 degrees and
    # reported the node ready -- a dead thermal zone looked exactly like a cold board.
    # Thermal state is a precondition for a valid measurement, so not knowing it blocks.
    if thermal["soc_temp_c"] is None:
        blocking.append("soc_temp_c unreadable -- thermal state cannot be verified")
    elif thermal["soc_temp_c"] >= 70:
        blocking.append("soc_temp_c=" + str(thermal["soc_temp_c"]))
    if not ENV.get("LLAMA_PIN_SHA"):
        # Deliberately a WARNING here while scripts/99_healthcheck.sh makes it a hard gate.
        # The two disagree on purpose: the first bring-up build has to happen before a SHA
        # exists, and /health is polled from bring-up onwards. Do not "fix" this to match
        # the shell gate without deciding which side wins -- see the test that pins it,
        # tests/test_node_agent.py::test_an_unpinned_llama_revision_is_reported.
        warnings.append("llama.cpp revision not pinned in config/node.env -- rows recorded "
                        "now cannot be tied to a revision")
    if rpc_binary() is None:
        blocking.append(
            "rpc server not built (looked for %s in %s)"
            % (", ".join(RPC_BIN_CANDIDATES), ENV.get("LLAMA_BIN") or "<LLAMA_BIN unset>")
        )

    # ---- checks that scripts/99_healthcheck.sh had and this gate did not -------------
    # THIS is the gate that runs before a cluster run: the control node calls /health,
    # nobody SSHes in to run `make health` first. Anything the shell gate blocks on that
    # is missing here is, in practice, not enforced at all.
    for name, detail in _measurement_preconditions():
        blocking.append("%s -- %s" % (name, detail))

    rc, out, _ = sh(["chronyc", "tracking"])
    offset = None
    if rc == 0:
        for line in out.splitlines():
            if line.startswith("System time"):
                try:
                    offset = float(line.split()[3])
                except (IndexError, ValueError):
                    offset = None
        if "Normal" not in out:
            blocking.append("clock not synced to control node")
    else:
        blocking.append("chrony unavailable - energy/event alignment cannot be trusted")
    if offset is not None and offset > 0.005:
        blocking.append("clock offset " + str(offset) + " s exceeds 5 ms")

    return {
        "node": ENV.get("NODE_ID"),
        "role": ENV.get("NODE_ROLE"),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "ready": not blocking,
        "blocking": blocking,
        "warnings": warnings,
        "clock_offset_s": offset,
        "metrics": m,
    }


def append_event(event, note=""):
    EVENTS.parent.mkdir(parents=True, exist_ok=True)
    thermal = device_info.thermal()
    rec = {
        "ts": time.time(),
        "node": ENV.get("NODE_ID"),
        "event": event,
        "note": note,
        "temp_c": thermal["soc_temp_c"],
        "throttled": thermal["throttled_word"],
    }
    with EVENTS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def tail_events(n=50):
    if not EVENTS.exists():
        return []
    lines = EVENTS.read_text(encoding="utf-8").splitlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = "HeteroServeNodeAgent/1"

    def _send(self, payload, status=200):
        body = json.dumps(payload, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # quieter than the default stderr spam
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def do_GET(self):
        route = urlparse(self.path)
        query = parse_qs(route.query)
        if route.path == "/info":
            self._send(device_info.collect())
        elif route.path == "/health":
            h = health()
            self._send(h, 200 if h["ready"] else 503)
        elif route.path == "/metrics":
            self._send(metrics())
        elif route.path == "/events":
            self._send(tail_events(int(query.get("n", ["50"])[0])))
        else:
            self._send({"error": "not found", "routes": ["/info", "/health", "/metrics", "/events"]}, 404)

    def do_POST(self):
        route = urlparse(self.path)
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            self._send({"error": "body is not JSON"}, 400)
            return

        if route.path == "/rpc/start":
            rc, out, err = sh(["sudo", "-n", "systemctl", "start", "heteroserve-rpc"])
            # `systemctl start` on a Type=simple unit returns as soon as the process is
            # FORKED -- it does not wait to see whether it survives. Upstream rpc-server
            # exits 0 on an unknown argument, so a bad invocation gives rc=0 here with
            # nothing listening, and the control node would place layers on a worker that
            # is not there. The port is the only honest liveness signal, so check it.
            listening = rpc_listening() if rc == 0 else False
            append_event("rpc_start", "via agent, listening=%s" % listening)
            self._send({"rc": rc, "stdout": out, "stderr": err,
                        "listening": listening, "port": RPC_PORT},
                       200 if (rc == 0 and listening) else 500)
        elif route.path == "/rpc/stop":
            rc, out, err = sh(["sudo", "-n", "systemctl", "stop", "heteroserve-rpc"])
            append_event("rpc_stop", "via agent")
            self._send({"rc": rc, "stdout": out, "stderr": err}, 200 if rc == 0 else 500)
        elif route.path == "/mark":
            self._send(append_event(body.get("event", "mark"), body.get("note", "")))
        else:
            self._send({"error": "not found", "routes": ["/rpc/start", "/rpc/stop", "/mark"]}, 404)


def main():
    port = int(ENV.get("AGENT_PORT", 9410))
    DATA_DIR.joinpath("logs").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print("node agent for " + str(ENV.get("NODE_ID")) + " listening on :" + str(port), file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
