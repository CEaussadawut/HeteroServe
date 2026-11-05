"""Load and validate config/cluster.json.

One loader, used by every other module, so a typo in the cluster file is caught once at
startup rather than halfway through a four-hour sweep.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "cluster.json"


class ConfigError(Exception):
    pass


class Node:
    def __init__(self, raw: dict):
        self.id = raw["id"]
        self.role = raw.get("role", "worker")
        self.description = raw.get("description", "")
        self.host = raw["host"]
        self.ssh_user = raw.get("ssh_user")
        self.agent_port = raw.get("agent_port")
        self.rpc_port = raw.get("rpc_port", 50052)
        self.ina_channel = raw.get("ina_channel")
        self.enabled = bool(raw.get("enabled", True))
        self.serves_rpc = bool(raw.get("serves_rpc", False))
        self.usable_mem_mb = int(raw.get("usable_mem_mb", 0))
        # Measured MemAvailable overhead by role ({"host": MB, "worker": MB}; a role left out is
        # not measured) and whether this node, as an RPC worker, pays the largest tensor it
        # receives a second time -- see placement.memory_cap.
        overhead = raw.get("memory_overhead_mb") or {}
        self.host_overhead_mb = overhead.get("host")
        self.worker_overhead_mb = overhead.get("worker")
        self.worker_repays_received = bool(raw.get("worker_repays_received_tensor", False))
        # MiB this node must keep free beyond what it is charged (placement.DeviceProfile.memory_reserve_mb).
        self.memory_reserve_mb = float(raw.get("memory_reserve_mb") or 0)
        # The most model-buffer MiB this node's backend has been shown to hold (placement.DeviceProfile.max_weights_mb).
        self.max_weights_mb = float(raw["max_weights_mb"]) if raw.get("max_weights_mb") is not None else None
        # This node, as the host, serves its blocks from the page cache (placement.DeviceProfile.host_weights_mapped).
        self.host_weights_mapped = bool(raw.get("host_weights_mapped", False))
        self.power_idle_w = raw.get("power_idle_w")
        # Where this node's own scripts live. Each node folder (Pi4/, Jetson/, ...)
        # is deployed to its own path, so the control node cannot assume one name.
        self.script_dir = raw.get("script_dir", "/opt/heteroserve/node")
        self.notes = raw.get("notes", "")

    # -- derived addresses ---------------------------------------------------
    @property
    def rpc_endpoint(self) -> str:
        return "%s:%d" % (self.host, self.rpc_port)

    @property
    def agent_url(self) -> str:
        if not self.agent_port:
            raise ConfigError("node %s has no agent_port" % self.id)
        return "http://%s:%d" % (self.host, self.agent_port)

    @property
    def ssh_target(self) -> str:
        if not self.ssh_user:
            raise ConfigError("node %s has no ssh_user" % self.id)
        return "%s@%s" % (self.ssh_user, self.host)

    @property
    def is_control(self) -> bool:
        return self.role == "control"

    def __repr__(self):
        return "<Node %s %s %s>" % (self.id, self.role, self.host)


class Cluster:
    def __init__(self, raw: dict, path: Path):
        self.path = path
        self.raw = raw
        self.name = raw["cluster"]["name"]
        self.control_id = raw["cluster"]["control_node"]
        self.ntp_server = raw["cluster"].get("ntp_server")
        self.nodes = [Node(n) for n in raw["nodes"]]
        self.link_tiers = {t["id"]: t for t in raw.get("link_tiers", [])}
        self.daq = raw.get("daq", {})
        self.paths = raw.get("paths", {})
        self.llama = raw.get("llama", {})
        self._validate()

    # -- lookups -------------------------------------------------------------
    def node(self, node_id: str) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise ConfigError("unknown node id: %s" % node_id)

    @property
    def control(self) -> Node:
        return self.node(self.control_id)

    def workers(self, include_disabled: bool = False):
        """Nodes that can serve layers over RPC.

        The control node is included only when serves_rpc is set, which is how the
        prefill-on-the-T580 disaggregation experiment is expressed.
        """
        return [n for n in self.nodes if n.serves_rpc and (n.enabled or include_disabled)]

    def measured(self):
        """Nodes on a DAQ channel - the ones energy figures exist for."""
        return [n for n in self.nodes if n.ina_channel]

    def node_for_channel(self, channel: int):
        for n in self.nodes:
            if n.ina_channel == channel:
                return n
        return None

    # -- validation ----------------------------------------------------------
    def _validate(self):
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ConfigError("duplicate node ids in %s" % self.path)
        if self.control_id not in ids:
            raise ConfigError("control_node '%s' is not in the node list" % self.control_id)

        channels = [n.ina_channel for n in self.nodes if n.ina_channel]
        if len(channels) != len(set(channels)):
            raise ConfigError("two nodes claim the same INA226 channel - the energy join would "
                              "silently attribute one node's power to another")

        # The DAQ table and the node table must agree, in both directions. A mismatch here
        # produces plausible-looking numbers attributed to the wrong device, which is the
        # worst kind of error this project can make.
        for ch, spec in self.daq.get("channels", {}).items():
            node_id = spec.get("node")
            if node_id not in ids:
                raise ConfigError("daq channel %s maps to unknown node '%s'" % (ch, node_id))
            declared = self.node(node_id).ina_channel
            if declared != int(ch):
                raise ConfigError(
                    "daq channel %s says node '%s', but that node declares ina_channel=%s"
                    % (ch, node_id, declared))

        for n in self.nodes:
            if n.serves_rpc and n.enabled and not n.usable_mem_mb:
                raise ConfigError("node %s serves rpc but has no usable_mem_mb - the placement "
                                  "engine cannot respect a memory budget it does not know" % n.id)

    def summary(self) -> str:
        lines = ["%s  (%d nodes, %d serving)" % (self.name, len(self.nodes), len(self.workers()))]
        for n in self.nodes:
            state = "on " if n.enabled else "OFF"
            ch = ("ch%d" % n.ina_channel) if n.ina_channel else "  -"
            lines.append("  [%s] %-13s %-16s %-15s %s %6s MB" %
                         (state, n.id, n.role, n.host, ch, n.usable_mem_mb or "-"))
        return "\n".join(lines)


def load(path=None) -> Cluster:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists():
        raise ConfigError("cluster config not found: %s" % path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError("%s is not valid JSON: %s" % (path, exc))
    return Cluster(raw, path)
