"""Link-tier control across the whole cluster.

Scope item 6: measure across two real link tiers (Wi-Fi, GbE) and use tc/netem to sweep
intermediate speeds, to find the threshold where distributing work starts to pay off.

Two rules this module enforces, because both are easy to get wrong and neither announces
itself in the data:

  1. netem shapes egress only. Shaping one side of a link and measuring it as if it were
     symmetric silently halves the modelled cost of every hop. Every tier change is
     therefore applied to all participating nodes, and to the control node too when it
     serves.
  2. Shaping outlives the experiment that set it. A tier left applied quietly corrupts the
     *next* run, so `clear` is idempotent, `verify` reports what is actually applied, and
     the preflight check looks for leftovers.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from . import nodes as nodeclient


class NetemError(Exception):
    pass


def _targets(cluster, node_ids):
    """Every node a tier change must reach.

    Deliberately NOT filtered by `ssh_user`. That filter silently dropped the control node,
    because `config/cluster.json` gives t580 `"ssh_user": null` -- and `config.workers()`
    includes the control node precisely when `serves_rpc` is set, which is how the
    prefill-on-the-T580 disaggregation experiment is expressed. Dropping it there left the
    control node's egress unshaped while the tier claimed otherwise, so rule 1 in this
    module's docstring ("applied to all participating nodes, and to the control node too
    when it serves") was stated and then not done. The control node is shaped locally
    instead; see `_run`.
    """
    if node_ids:
        return [cluster.node(n) for n in node_ids]
    return list(cluster.workers())


def _is_local(cluster, node) -> bool:
    return node.id == cluster.control_id


def _run(cluster, node, cmd, timeout):
    """Run a shaping command on `node` -- over ssh, or locally for the control node."""
    if _is_local(cluster, node):
        try:
            done = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                  timeout=timeout)
            return done.returncode, done.stdout, done.stderr
        except subprocess.TimeoutExpired:
            return 124, "", "timed out after %ss" % timeout
        except OSError as exc:
            return 126, "", str(exc)
    if not node.ssh_user:
        return 126, "", ("no ssh_user in cluster.json and this is not the control node, so "
                         "there is no way to reach it")
    return nodeclient.ssh(node, cmd, timeout=timeout)


def _explain(node, rc, err) -> str:
    """A human reason for a failed shaping command, naming the likely cause."""
    if rc == 127 or "No such file" in (err or ""):
        return ("%s: %s/scripts/05_net_tier.sh is missing. Only Pi4/ ships that script; "
                "Pi5/ has 05_network_setup.sh with a different argument convention, and "
                "the Jetson has neither." % (node.id, node.script_dir))
    return "%s: shaping command failed rc=%s %s" % (node.id, rc, (err or "").strip()[:160])


def apply_tier(cluster, tier_id: str, node_ids=None, strict: bool = True) -> dict:
    """Put every relevant node onto one link tier.

    Refuses a partial application by default: see the NetemError below.
    """
    tier = cluster.link_tiers.get(tier_id)
    if not tier:
        raise NetemError("unknown link tier '%s' - known: %s"
                         % (tier_id, ", ".join(sorted(cluster.link_tiers))))

    targets = _targets(cluster, node_ids)

    results = {}
    failed = []
    for node in targets:
        script = "%s/scripts/05_net_tier.sh" % node.script_dir
        if tier["kind"] == "real":
            cmd = "sudo %s %s" % (script, "wifi" if tier_id == "wifi" else "gbe")
        else:
            cmd = "sudo %s shape %s %s" % (script, tier["nominal_mbps"], tier.get("delay_ms", 0))
        rc, out, err = _run(cluster, node, cmd, timeout=60)
        results[node.id] = {"rc": rc, "stdout": out, "stderr": err}
        if rc == 0:
            nodeclient_safe_mark(node, "net_tier", tier_id)
        else:
            failed.append(_explain(node, rc, err))

    report = {"tier": tier_id, "spec": tier, "nodes": results,
              "applied": [n for n, r in results.items() if r["rc"] == 0],
              "failed": failed, "ok": not failed}
    if failed and strict:
        # A partially applied tier is the worst outcome available here: the run still
        # records `tier=shaped-10`, so the condition is MISLABELLED rather than missing,
        # and a gbe-vs-shaped-10 comparison becomes two unshaped conditions under
        # different names. The rc was previously stored and even inspected, yet
        # `apply_tier` returned normally and the CLI exited 0.
        raise NetemError(
            "tier '%s' applied to %d of %d nodes; refusing a partially shaped cluster "
            "because the runs would still be labelled '%s'.\n  %s\n"
            "Pass strict=False (heteroctl tier --allow-partial) to proceed deliberately."
            % (tier_id, len(report["applied"]), len(targets), tier_id, "\n  ".join(failed)))
    return report


def nodeclient_safe_mark(node, event, note):
    """Marking is best-effort: a failed marker must not abort a tier change."""
    if not node.agent_port:
        return
    try:
        nodeclient.mark(node, event, note)
    except Exception:  # noqa: BLE001
        pass


def clear(cluster, node_ids=None) -> dict:
    targets = _targets(cluster, node_ids)
    out = {}
    for node in targets:
        rc, stdout, stderr = _run(
            cluster, node, "sudo %s/scripts/05_net_tier.sh clear" % node.script_dir, timeout=30)
        out[node.id] = {"rc": rc, "stdout": stdout, "stderr": stderr}
    return out


def verify(cluster, node_ids=None) -> dict:
    """Report the qdisc actually in force on each node - the ground truth, not the intent."""
    targets = _targets(cluster, node_ids)
    out = {}
    for node in targets:
        rc, stdout, _ = _run(cluster, node, "tc qdisc show dev eth0", timeout=20)
        shaped = "netem" in stdout or "htb" in stdout
        out[node.id] = {"qdisc": stdout, "shaped": shaped, "rc": rc}
    return out


# ------------------------------------------------------------- link survey
def survey(cluster, node_ids=None, seconds=10, iperf_server=None) -> dict:
    """Measure real bandwidth and RTT to every node - the rule's link inputs.

    Run this after every tier change. The rule must be fed what the link *is*, not what the
    tier was meant to be: a shaped 100 Mbit tier that negotiated 100 Mbit half-duplex is a
    different link, and only a measurement can tell.
    """
    # Unlike apply_tier/clear/verify, the control node is excluded here ON PURPOSE and not
    # as a side effect of the ssh_user filter: it is the iperf3 SERVER, so measuring it
    # against itself is meaningless. Everything else is measured.
    #
    # CAVEAT worth knowing when reading link_mbps: iperf3 runs worker -> control, which is
    # the direction netem shapes (egress). If the control node's own egress is ever left
    # unshaped, the recorded figure describes the shaped leg only, while placement.py
    # applies it to both legs of the hop.
    targets = [cluster.node(n) for n in node_ids] if node_ids else \
              [n for n in cluster.workers() if n.id != cluster.control_id]
    control_host = cluster.control.host
    out = {"nodes": {}, "iperf_server": iperf_server or control_host}

    for node in targets:
        entry = {}
        rc, stdout, _ = nodeclient.ssh(
            node, "ping -c 20 -i 0.2 -q %s" % control_host, timeout=60)
        m = re.search(r"= [\d.]+/([\d.]+)/", stdout)
        if m:
            entry["rtt_ms"] = float(m.group(1))

        rc, stdout, err = nodeclient.ssh(
            node, "iperf3 -c %s -t %d -J" % (out["iperf_server"], seconds), timeout=seconds + 30)
        if rc == 0 and stdout:
            try:
                data = json.loads(stdout)
                entry["mbps"] = round(data["end"]["sum_received"]["bits_per_second"] / 1e6, 2)
            except (json.JSONDecodeError, KeyError):
                entry["iperf_error"] = "unparseable iperf3 output"
        else:
            entry["iperf_error"] = err or "iperf3 failed - is `iperf3 -s` running on the control node?"
        out["nodes"][node.id] = entry
    return out


def save_survey(survey_result: dict, path):
    Path(path).write_text(json.dumps(survey_result, indent=2), encoding="utf-8")
    return path


def local_iperf_server(start=True):
    """Start/stop iperf3 -s on the control node (the survey needs a peer to talk to)."""
    if start:
        return subprocess.Popen(["iperf3", "-s", "-D"], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "iperf3 -s"], capture_output=True)
    return None
