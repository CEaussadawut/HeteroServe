"""Client for the per-node agents, and the cluster-wide readiness gate.

Scope item 1 of the proposal: device inventory, node roles, and "verify readiness before
every experimental run". The agents (node/node_agent.py on each worker) do the local
checking; this module fans out, aggregates, and decides whether a run may start.
"""
from __future__ import annotations

import concurrent.futures
import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone


class NodeError(Exception):
    pass


# ---------------------------------------------------------------- agent calls
def _get(url: str, timeout: float = 10.0):
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        # /health answers 503 with a full body when the node is not ready; that body is
        # the useful part, so an HTTPError here is data rather than a failure.
        try:
            return exc.code, json.loads(exc.read().decode())
        except Exception:
            return exc.code, {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        raise NodeError("%s: %s" % (url, exc))


def _post(url: str, payload: dict | None = None, timeout: float = 20.0):
    body = json.dumps(payload or {}).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except Exception as exc:  # noqa: BLE001
        raise NodeError("%s: %s" % (url, exc))


def info(node):
    return _get(node.agent_url + "/info")[1]


def health(node):
    status, body = _get(node.agent_url + "/health")
    body["http_status"] = status
    return body


def metrics(node):
    return _get(node.agent_url + "/metrics", timeout=5)[1]


def mark(node, event: str, note: str = ""):
    return _post(node.agent_url + "/mark", {"event": event, "note": note})[1]


def rpc_start(node):
    return _post(node.agent_url + "/rpc/start")[1]


def rpc_stop(node):
    return _post(node.agent_url + "/rpc/stop")[1]


# ------------------------------------------------------------------- fan-out
def fanout(nodes, fn, workers: int = 8):
    """Call fn(node) on every node in parallel. Errors are captured, never raised.

    A cluster check must report on all four nodes even when one is unplugged; an exception
    from the first unreachable node would hide the state of the rest.
    """
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, n): n for n in nodes}
        for fut in concurrent.futures.as_completed(futures):
            node = futures[fut]
            try:
                out[node.id] = {"ok": True, "data": fut.result()}
            except Exception as exc:  # noqa: BLE001
                out[node.id] = {"ok": False, "error": str(exc)}
    return out


def mark_all(nodes, event: str, note: str = ""):
    """Bracket a distributed run with one event boundary on every node at once.

    The marks are issued in parallel, but they are still N separate HTTP calls, so they
    land within a few milliseconds of each other rather than simultaneously. That is fine
    for phase attribution at a ~28 Hz sample rate; it would not be for anything finer.
    """
    return fanout(nodes, lambda n: mark(n, event, note))


# ------------------------------------------------------------------ ssh helper
def ssh(node, command: str, timeout: int = 60, check: bool = False):
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", node.ssh_target, command]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise NodeError("ssh %s: %s" % (node.id, proc.stderr.strip()))
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


# --------------------------------------------------------------- readiness gate
# Pi 4's agent blocks its own readiness at this figure (Pi4/node/node_agent.py:299,
# "clock offset ... exceeds 5 ms") and its healthcheck repeats it
# (Pi4/scripts/99_healthcheck.sh:39). Adopted rather than re-chosen: two thresholds for one
# quantity is how a cluster ends up disagreeing with itself about whether it is ready.
MAX_CLOCK_OFFSET_S = 0.005


def preflight(cluster, nodes=None, require_pin: bool = True,
              require_clock: bool = True) -> dict:
    """Decide whether the cluster may be measured right now.

    Returns a report; `ready` False means: do not start the run. The harness refuses to
    proceed on a False, and that refusal is the point - a run that starts on a skewed
    clock or a throttled board produces numbers that look fine and are wrong.
    """
    nodes = nodes if nodes is not None else [n for n in cluster.workers() if n.agent_port]
    report = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "nodes": {},
        "blocking": [],
        "warnings": [],
    }

    results = fanout(nodes, health)
    for node in nodes:
        entry = results.get(node.id, {"ok": False, "error": "no result"})
        if not entry["ok"]:
            report["nodes"][node.id] = {"reachable": False, "error": entry["error"]}
            report["blocking"].append("%s: agent unreachable (%s)" % (node.id, entry["error"]))
            continue
        h = entry["data"]
        # A body that is not the readiness contract is not evidence of readiness. The gate
        # used to read `h.get("blocking", [])`, so ANY parseable JSON that lacked that key -
        # a proxy page, an error body, a drifted key name, a half-deployed agent - supplied
        # zero blocking reasons and the node sailed through. Absence of stated problems is
        # not the same as a statement that there are none.
        contract = (isinstance(h, dict) and "ready" in h
                    and isinstance(h.get("blocking"), list)
                    and isinstance(h.get("warnings", []), list))
        if not contract:
            report["nodes"][node.id] = {"reachable": True, "ready": False,
                                        "error": "health body is not the readiness contract",
                                        "body_keys": sorted(h)[:12] if isinstance(h, dict) else None}
            report["blocking"].append(
                "%s: /health did not answer the readiness contract (need `ready` and a list "
                "`blocking`); got keys %s"
                % (node.id, sorted(h)[:12] if isinstance(h, dict) else type(h).__name__))
            continue

        report["nodes"][node.id] = {
            "reachable": True,
            "ready": h.get("ready"),
            "blocking": h.get("blocking", []),
            "warnings": h.get("warnings", []),
            "clock_offset_s": h.get("clock_offset_s"),
            "temp_c": (h.get("metrics") or {}).get("soc_temp_c"),
        }
        # Honour the node's own verdict even when it lists no reasons. `ready` was recorded
        # here but never consulted - the gate's decision came only from `blocking` - so a
        # node answering {"ready": false, "blocking": []} was treated as ready.
        if not h.get("ready") and not h.get("blocking"):
            report["blocking"].append(
                "%s: reports ready=%r but gave no reason - treating as not ready"
                % (node.id, h.get("ready")))
        for reason in h.get("blocking", []):
            report["blocking"].append("%s: %s" % (node.id, reason))
        for reason in h.get("warnings", []):
            report["warnings"].append("%s: %s" % (node.id, reason))

    # Clock sync, gated rather than merely displayed. This function's own docstring says a
    # run that starts on a skewed clock "produces numbers that look fine and are wrong", and
    # `clock_offset_s` was recorded above and then read by nothing but format_preflight's
    # display column - so the gate printed the evidence and never acted on it.
    #
    # It matters because node event markers drive analysis.windows_from_events, which builds
    # the idle windows behind mean_power_w_idle and therefore delta_power_w, the marginal
    # energy figure. A skewed clock averages the wrong samples and flags nothing.
    #
    # A missing offset BLOCKS, for the same reason an unreported llama pin does below:
    # "unverified" must not read as "synced". That is not hypothetical - Pi 4's agent
    # supplies clock_offset_s and gates on chrony, while the Pi 5's readiness() has nine
    # gates and not one concerns the clock (Pi5/node/agent.py:126-165), so the Pi 5 can
    # currently answer ready=true with an arbitrarily skewed clock. `heteroctl preflight
    # --no-clock-check` is the deliberate escape hatch.
    if require_clock:
        for node_id, entry in sorted(report["nodes"].items()):
            if not entry.get("reachable"):
                continue
            offset = entry.get("clock_offset_s")
            if offset is None:
                report["blocking"].append(
                    "%s: will not report its clock offset, so alignment of its event "
                    "markers with the power trace cannot be verified" % node_id)
            elif abs(offset) > MAX_CLOCK_OFFSET_S:
                report["blocking"].append(
                    "%s: clock offset %s s exceeds %s s - phase attribution would be wrong"
                    % (node_id, offset, MAX_CLOCK_OFFSET_S))

    # One pinned llama.cpp revision across the whole cluster. Rows built from two
    # revisions cannot be compared with each other, so this is blocking, not advisory.
    pin = (cluster.llama or {}).get("pin_sha", "")
    if require_pin:
        if not pin:
            report["blocking"].append("cluster: llama.pin_sha is empty in cluster.json")
        else:
            for node in nodes:
                entry = results.get(node.id)
                if not entry or not entry["ok"]:
                    continue
                # The agent reports its own pin via /info; cheap enough to fetch once here.
                try:
                    node_pin = (info(node).get("software") or {}).get("llama_pin_sha")
                except NodeError:
                    node_pin = None
                if node_pin and node_pin != pin:
                    report["blocking"].append(
                        "%s: built %s but the cluster pins %s" % (node.id, node_pin[:12], pin[:12]))
                elif not node_pin:
                    # Blocking, not advisory. The comment above says rows built from two
                    # revisions cannot be compared; a node that will not say which revision
                    # it built leaves that unverifiable, and "unverified" must not read as
                    # "agrees". These nodes were selected for having an agent_port, so not
                    # answering is a real failure rather than an absent capability.
                    # `heteroctl preflight --no-pin-check` is the deliberate escape hatch.
                    report["blocking"].append(
                        "%s: will not report its built revision, so agreement with the "
                        "cluster pin cannot be verified" % node.id)

    report["ready"] = not report["blocking"]
    return report


def format_preflight(report: dict) -> str:
    lines = ["preflight @ %s" % report["checked_at_utc"]]
    for node_id, entry in sorted(report["nodes"].items()):
        if not entry.get("reachable"):
            lines.append("  FAIL %-13s unreachable: %s" % (node_id, entry.get("error")))
            continue
        state = "READY" if entry.get("ready") else "NOT READY"
        lines.append("  %-9s %-13s offset=%-10s temp=%s" %
                     (state, node_id, entry.get("clock_offset_s"), entry.get("temp_c")))
        for reason in entry.get("blocking", []):
            lines.append("            - %s" % reason)
    if report.get("warnings"):
        lines.append("  warnings:")
        lines.extend("    ! %s" % w for w in report["warnings"])
    lines.append("")
    lines.append("  RESULT: %s" % ("READY" if report["ready"] else "BLOCKED"))
    for reason in report.get("blocking", []):
        lines.append("    x %s" % reason)
    return "\n".join(lines)


def inventory(cluster, nodes=None) -> dict:
    """Collect every node's device_info into one document.

    This is the device-characterisation table's static half; the measured half comes from
    the benchmark sweeps.
    """
    nodes = nodes if nodes is not None else [n for n in cluster.nodes if n.agent_port and n.enabled]
    results = fanout(nodes, info)
    return {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "cluster": cluster.name,
        "nodes": {nid: (r["data"] if r["ok"] else {"error": r["error"]}) for nid, r in results.items()},
    }
