"""The heterogeneity-aware placement rule, and the baselines it is measured against.

Scope item 3: "Develop a rule for distributing model layers across nodes based on each
node's compute capability, memory bandwidth and link speed. Compare against an even split
across all nodes, and study whether a markedly slower node should be given reduced load or
dropped from the cluster entirely."

The rule here is deliberately simple and fully explainable. Its inputs are all *measured*:
per-layer compute time and active/idle power come from the Phase 2 characterisation sweeps,
link bandwidth and RTT from the iperf3/ping sweeps. Nothing is taken from a datasheet.

    minimise   T + lambda * E
    subject to mem_i(x) <= M_i   for every node i

with, for one token of decode through a layer-pipelined cluster:

    T = sum_i n_i * c_i + C                     C = comm time across the hops
    E = sum_i [ n_i*c_i*P_act_i + (T - n_i*c_i)*P_idle_i ]

The second term of E is what makes this interesting: a node that holds few layers still
burns idle power for the whole token period, so adding a slow node taxes every other node's
energy as well as its own. That is the formal statement of the straggler question.

Substituting T into E makes the objective linear in the layer counts:

    J = sum_i n_i * c_i * [1 + lambda*(dP_i + P_idle_S)] + C * [1 + lambda*P_idle_S]

where dP_i = P_act_i - P_idle_i and P_idle_S is the summed idle power of the participating
set S. For a fixed participating set and pipeline order, the optimum is therefore reached
by filling layers into the smallest weights first, subject to each node's memory cap - no
search needed. The only combinatorial part is the choice of S and its order, and with at
most five nodes that is enumerated exhaustively, so the result is exact rather than greedy.

Complexity: O(2^N * N! * N log N) in the worst case, which for N <= 6 is a few thousand
cheap operations, done once at cluster assembly. The proposal quotes O(N*L); that figure
describes the layer fill, and this file is the honest version.
"""
from __future__ import annotations

import itertools
from typing import Optional
import math
from dataclasses import dataclass, field, replace


MB = 1024 * 1024


@dataclass
class DeviceProfile:
    """Everything the rule knows about one node. All of it measured."""
    node_id: str
    ms_per_layer_decode: float      # measured: isolated decode ms/token / n_layers
    ms_per_layer_prefill: float     # measured: isolated prefill ms/token / n_layers
    # measured: MemAvailable at rest. The runtime's own headroom is ModelProfile.runtime_overhead_mb,
    # so it is not subtracted here as well (that convention reproduces the 14B boundary).
    usable_mem_mb: float
    # Power is OPTIONAL. None means "not measured" -- the Pis have no power figure until the INA226
    # rig exists -- and never a stand-in number: 0 W would make a node look free to run, and
    # joules_per_token would come out as a clean-looking 0. With None the rule still plans for
    # latency (energy_weight = 0 never reads power), energy figures report None, and anything
    # that needs power (energy_weight > 0, the Pareto front) refuses instead of guessing.
    power_active_w: Optional[float] = None   # measured: mean rail power during a decode run
    power_idle_w: Optional[float] = None     # measured: mean rail power at rest, same rail
    link_mbps: float = 1000.0       # measured: iperf3 to the control node
    rtt_ms: float = 0.3             # measured: ping
    is_straggler: bool = False
    notes: str = ""
    # Per-layer cost when this node serves as an RPC worker (ggml-rpc-server) instead of running
    # llama.cpp itself. The two differ where the host path uses kernels the worker path does not:
    # ggml-rpc allocates the default CPU buffer type, so a worker gets no repacked weight layouts
    # (the Pi 5's dotprod kernels). None = the same as the host figures -- right for the Pi 4 and
    # the Jetson's CPU, which never repack, and measured within 2.5 % for the Jetson's Vulkan.
    ms_per_layer_decode_worker: Optional[float] = None
    ms_per_layer_prefill_worker: Optional[float] = None
    # Memory by role, measured (testbed/benchmarks.md, "Memory accounting, validated out of
    # sample", 2026-09-28): the MemAvailable a node loses beyond its blocks and a head it holds,
    # as a host (it runs llama.cpp itself; the buffer for the largest tensor it uploads is
    # charged on top) and as an RPC worker. None = not measured for that role, and the older
    # stand-in applies there (ModelProfile.runtime_overhead_mb, plus the embedding on a host).
    host_overhead_mb: Optional[float] = None
    worker_overhead_mb: Optional[float] = None
    # A worker that pays the largest tensor it receives a second time: the Jetson's Vulkan0
    # worker (+643 MiB over its blocks and head, holding the 14B's 609 MiB head). CPU workers do not.
    worker_repays_received: bool = False
    # MiB this node must keep free on top of everything it is charged, because its allocator fails
    # before MemAvailable reaches zero. The Jetson's Vulkan0 worker refused 108 MiB with about 290 MiB
    # available (the MoE cell, testbed/benchmarks.md 2026-09-28), and every Jetson cell has run with
    # 450 MiB or more left. It is headroom, not usage: memory_cap subtracts it and memory_need_mb does
    # not, so a cell's need stays what the memory probe measures. 0 = none observed.
    memory_reserve_mb: float = 0.0
    # The most MiB of weights -- its blocks plus a head it holds -- this node may be given, whatever its memory allows:
    # the largest share its backend has been shown to hold. The Jetson's Vulkan0 worker ran every share up to a
    # 2,829 MiB model buffer and failed above it -- SIGSEGV in NVIDIA's Tegra Vulkan driver at 2,922 MiB, an allocation
    # error at 3,112 -- while a passing share ran with 423 MiB MemAvailable, so no memory reserve explains it (registered
    # rule GPU_CAP, testbed/dataset/jetson_graph_limit_20260929_addendum.json; cluster.json max_weights_mb). memory_cap
    # applies it in every role, against the model's average block as it prices memory. None = no such limit observed.
    max_weights_mb: Optional[float] = None
    # A host that serves its own blocks from the page cache, mmap'd and not repacked: the Pi 4. Its steady charge is its
    # blocks + a head it holds + host_overhead_mb, without the largest-upload copy. That copy is anonymous memory that
    # lives only while it loads, and it can only evict the mapped blocks, which the first tokens then read back (the
    # cold first prefill repetition). memory_cap checks it apart instead, against the memory that can really run out:
    # host_overhead_mb + the copy must fit the node's budget on their own. Registered rule ACCOUNT
    # (testbed/dataset/pool_predictions_20260928c.json): the Pi 4 hosted 46 blocks of Qwen2.5-14B at normal speed and
    # thrashed 17.6x slower at 47, where blocks + overhead pass MemAvailable -- the cap this charge gives, 46 not 42.
    host_weights_mapped: bool = False
    # Unit costs (2026-09-29): ms per byte read per decoded token, and ms per block parameter per prompt token, from a
    # profiled model whose geometry was known (profiles.build_devices). profiles.for_model turns them into the per-layer
    # figures of the model being planned. Without them ms_per_layer_* belong to the profiled model only: heteroctl once
    # priced a 14B's layers as TinyLlama's and predicted it 85 % too fast.
    ms_per_byte_decode: Optional[float] = None
    ms_per_param_prefill: Optional[float] = None
    profiled_on: str = ""
    # Mean power while PREFILLING, where it differs from decoding (the strip, 2026-10-05, TinyLlama Q4_K_M at the wall:
    # the Jetson's GPU 8.9 W prefilling against 11.2 W decoding, the Pi 4 7.5 against 6.3). None = power_active_w for
    # both phases, as every profile before 2026-10-08.
    power_active_prefill_w: Optional[float] = None

    def active_w(self, phase="decode") -> Optional[float]:
        """Mean power while computing in this phase ("prefill" or "decode")."""
        if phase == "prefill" and self.power_active_prefill_w is not None:
            return self.power_active_prefill_w
        return self.power_active_w

    def delta_w(self, phase="decode") -> Optional[float]:
        """Power above idle while computing in this phase; None without a power figure."""
        if not self.has_power:
            return None
        return max(0.0, self.active_w(phase) - self.power_idle_w)

    def layer_ms(self, phase: str, as_worker: bool) -> float:
        """ms per layer in this phase, in this role."""
        if phase == "decode":
            w, h = self.ms_per_layer_decode_worker, self.ms_per_layer_decode
        else:
            w, h = self.ms_per_layer_prefill_worker, self.ms_per_layer_prefill
        return w if (as_worker and w is not None) else h

    @property
    def has_power(self) -> bool:
        return self.power_active_w is not None and self.power_idle_w is not None

    @property
    def delta_power_w(self) -> Optional[float]:
        if not self.has_power:
            return None
        return max(0.0, self.power_active_w - self.power_idle_w)


def _unpowered(devices, node_ids=None) -> list:
    """Node ids (of node_ids, or of all devices) that carry no measured power figure."""
    ids = node_ids if node_ids is not None else list(devices)
    return [nid for nid in ids if not devices[nid].has_power]


def _require_power(devices, node_ids, what):
    missing = _unpowered(devices, node_ids)
    if missing:
        raise ValueError(
            "%s needs measured power for every node, and %s has none (power is None = not "
            "measured, e.g. before the INA226 rig exists). Plan with energy_weight=0, which "
            "optimises latency only and never reads power." % (what, ", ".join(sorted(missing))))


@dataclass
class ModelProfile:
    """Model shape. n_layers and the per-layer byte costs decide the memory constraint."""
    name: str
    n_layers: int
    bytes_per_layer: float          # weights only, at this quantisation
    kv_bytes_per_layer_per_token: float
    activation_bytes: float         # transferred at each pipeline boundary, per token
    context_len: int = 2048
    runtime_overhead_mb: float = 350.0   # llama.cpp + OS working set on a worker
    # Two tensors are not per-layer; one node each holds them (llama.cpp at the pin,
    # llama-model.cpp load_tensors; the gguf.shape fields of the same names):
    #   head_bytes  the output layer (output.weight + output_norm; the token embedding when
    #               the two are tied). Any offload (-ngl >= 1) sends it to the LAST device of
    #               the split -- -ngl N offloads the output layer first, then N-1 repeating
    #               layers -- so the pipeline's LAST node pays, in every launch used here.
    #   embd_bytes  the token embedding, always on the CPU of the process that loads the
    #               model. The harness runs llama-cli on the control node, which holds no
    #               layers, so no participant pays. A node that runs llama.cpp itself and keeps
    #               the first layers on its CPU (run_14b_pooled.sh: the Pi 5) is the pipeline's
    #               FIRST node and pays: set host_in_pipeline.
    # Both 0 (the default) is the older accounting, where bytes_per_layer = file size /
    # n_layers folds them into every layer. On the 14B the new accounting reproduces the Pi
    # 5's measured boundary with the default 350 MB overhead: 43 blocks run and 45 are
    # OOM-killed (testbed/dataset/placement_validation.json).
    # Probed since (2026-09-27, run_pool.sh's 1 Hz memory probe): that match holds for a
    # different reason. The mmap'd embedding is NOT resident -- ~30 MiB of the file stays
    # mapped at steady state. What sets the boundary is a load-time transient: ggml-rpc's
    # serialize_set_tensor copies each tensor it uploads whole into a fresh buffer, so the
    # host peaks at its steady footprint (blocks + KV + compute + ~60 MiB) plus the LARGEST
    # tensor it sends -- the 609 MiB output head for the 14B (+557 MiB measured), 133 MiB for
    # the 7B (+207). embd_bytes + 350 MB stands in for that on the 14B; on the 7B it is ~290
    # MiB conservative. Where a node's role overheads are measured, memory_cap uses the board-aware
    # accounting that replaced this stand-in (validated out of sample 2026-09-28) instead.
    head_bytes: float = 0.0
    embd_bytes: float = 0.0
    host_in_pipeline: bool = False
    # The largest single tensor inside one block (gguf.shape). It sets the upload buffer of a host
    # whose head stays local, and what a received-tensor-repaying worker pays again when it does
    # not hold the head. 0 = unknown, and only the head is charged there.
    max_block_tensor_bytes: float = 0.0
    # A mixture-of-experts model reads only its active experts per decoded token (gguf.shape): the per-layer
    # decode cost the caller supplies is then per ACTIVE bytes, and the head is priced on that same basis.
    # Memory still counts every expert (bytes_per_layer). 0 = a dense model: active = bytes_per_layer.
    active_bytes_per_layer: float = 0.0
    # Block parameters, for pricing prefill per parameter (profiles.for_model); a mixture of experts computes its prompt
    # with its active experts. 0 = unknown, and prefill keeps the profiled per-layer figure.
    params_per_layer: float = 0.0
    active_params_per_layer: float = 0.0
    # Each block's real weight bytes, block 0 first (gguf.shape layer_bytes, from the tensor table). Used only by
    # plan_optimal(fill="exact"), which checks each node's actual contiguous run of blocks instead of a count against
    # the average block. None = unknown; the average fill needs only bytes_per_layer.
    layer_bytes: Optional[tuple] = None
    # The host reads its weights into its own memory (llama.cpp -lm none) instead of mapping the file: serve_pool.sh
    # since 2026-10-08, and registrations that pass load_mode "none". Every tensor it keeps is then anonymous memory,
    # the token embedding included, so a host in a measured role pays embd_bytes on top of its charge. Mapped, only
    # the rows a prompt touches are resident (~30 MiB, 2026-09-27), and host_overhead_mb was fitted that way. Found on
    # 2026-10-09: the Pi 4 hosting Qwen3-14B logged CPU buffers 427 MiB above its charge in all three placements,
    # 417 of them the embedding, and the one placement charged within that gap of its MemAvailable was OOM-killed.
    # False = the mapped launch that every registration on file priced, so each of them reproduces.
    host_loads_weights: bool = False

    def per_layer_footprint_bytes(self) -> float:
        """Weights plus the KV cache that layer will hold at full context."""
        return self.bytes_per_layer + self.kv_bytes_per_layer_per_token * self.context_len


@dataclass
class Placement:
    """One assignment of layers to an ordered pipeline of nodes."""
    strategy: str
    order: list                      # node ids, in pipeline order
    layers: dict                     # node_id -> layer count
    model: str
    predicted: dict = field(default_factory=dict)
    feasible: bool = True
    reason: str = ""
    # The node holding the output head; None = the last participant, where llama.cpp's -ngl puts it. The host, when
    # the plan keeps the head there as prima.cpp's Halda does (at the pin: -ot "^output\.weight=CPU", anchored).
    head_node: Optional[str] = None

    @property
    def participants(self) -> list:
        return [nid for nid in self.order if self.layers.get(nid, 0) > 0]

    def tensor_split(self) -> list:
        """llama.cpp --tensor-split proportions, in the order of the --rpc list."""
        total = sum(self.layers.get(nid, 0) for nid in self.participants)
        if not total:
            return []
        return [round(self.layers[nid] / total, 6) for nid in self.participants]

    def llama_args(self, cluster, model_path: str, n_gpu_layers: int = 99) -> list:
        """The actual command-line flags that realise this placement.

        A placement that cannot be expressed as a llama.cpp invocation is a paper exercise,
        so the rule emits the flags rather than a diagram.

        These are the flags for the harness's launch: llama-cli on the control node, every
        participant an RPC device (ModelProfile.host_in_pipeline False). A plan made with
        host_in_pipeline True is launched instead on its first node, as run_14b_pooled.sh does:
        -rpc <the other nodes> -ngl <n_layers + 1 - that node's layers>, since -ngl counts the
        output layer as one of the layers it offloads.
        """
        if self.head_node:
            # Here every participant is an RPC device and llama.cpp runs on the control node, so no participant is
            # "the host": -ot would put the head on the control node, not where this plan priced it. A head-on-host
            # plan is launched on its first node (run_pool.sh / serve_pool.sh, predict_pool.bench_args(head_at=0)).
            raise ValueError("plan keeps the output head on %s: launch it on its first node with -ot, not with "
                             "llama_args" % self.head_node)
        endpoints = [cluster.node(nid).rpc_endpoint for nid in self.participants]
        split = ",".join(str(x) for x in self.tensor_split())
        return [
            "-m", model_path,
            "--rpc", ",".join(endpoints),
            "-ngl", str(n_gpu_layers),
            "--tensor-split", split,
        ]

    def describe(self) -> str:
        lines = ["%s  [%s]" % (self.strategy, self.model)]
        if not self.feasible:
            return "\n".join(lines + ["  INFEASIBLE: " + self.reason])
        total = sum(self.layers.values())
        for nid in self.order:
            n = self.layers.get(nid, 0)
            bar = "#" * max(0, int(30 * n / total)) if total else ""
            lines.append("  %-13s %3d layers  %-31s" % (nid, n, bar))
        if self.head_node:
            lines.append("  output head on %s (the host)" % self.head_node)
        p = self.predicted
        # A latency-only plan (power None until the INA226 rig exists) leaves the energy figures
        # None and says why in "energy"; formatting that None with %.3f crashed `heteroctl plan`
        # and `run` on every such profile (2026-09-29, the first bring-up on the real cluster).
        if p.get("joules_per_token") is None or p.get("mean_power_w") is None:
            energy = p.get("energy") or "energy not predicted"
        else:
            energy = "%.3f J/token, %.2f W mean" % (p["joules_per_token"], p["mean_power_w"])
        lines.append("  predicted: %.1f ms/token (%.2f tok/s), %s" %
                     (p.get("ms_per_token", 0), p.get("tokens_per_s", 0), energy))
        return "\n".join(lines)


# --------------------------------------------------------------------- costing
def hop_ms(src: DeviceProfile, dst: DeviceProfile, model: ModelProfile) -> float:
    """Time to move one token's activations across one pipeline boundary.

    Bandwidth is the slower of the two endpoints; a fast node behind a shaped link is as
    slow as the shaping. RTT is counted once per hop, which is the right model for a
    synchronous hand-off and the wrong one for a pipelined batch - single-stream decode is
    synchronous, so this holds for the decode case the project measures.
    """
    bw_bytes_s = min(src.link_mbps, dst.link_mbps) * 1e6 / 8.0
    if bw_bytes_s <= 0:
        return math.inf
    transfer = model.activation_bytes / bw_bytes_s * 1000.0
    return transfer + max(src.rtt_ms, dst.rtt_ms)


def memory_cap(dev: DeviceProfile, model: ModelProfile, holds_head: bool = False,
               holds_embd: bool = False, role: Optional[str] = None, uploads: bool = False) -> int:
    """How many layers this node can hold without swapping or OOMing, after the output head
    and the token embedding if it holds them (see ModelProfile.head_bytes).

    role None, or a role whose overhead is not measured on this node, keeps the stand-in:
    runtime_overhead_mb per node, plus the head and the embedding on whoever holds them.
    With role 'host' or 'worker' measured (DeviceProfile.host_overhead_mb / worker_overhead_mb),
    the board-aware accounting validated out of sample on 2026-09-28 applies:
      host    blocks + a head it holds + host_overhead_mb + the largest tensor it uploads when it
              has workers (uploads): the head, which goes to the last node, or else the largest
              block tensor. Mapped, the token embedding is not resident and is not charged; a host
              that reads its weights into memory (ModelProfile.host_loads_weights) pays it.
      worker  blocks + a head it holds + worker_overhead_mb, plus the largest tensor it receives
              (the head if it holds it, else the largest block tensor) once more if it repays
              received tensors (DeviceProfile.worker_repays_received).
    A host whose weights stay mmap'd (the Pi 4 does not repack) loses no MemAvailable for its
    blocks, but they are read on every token and must stay resident in page cache, so they are
    charged all the same: its measured overhead is small, not its footprint. Such a host
    (DeviceProfile.host_weights_mapped) is not charged the upload copy on top of its blocks: the copy
    only lives while it loads, so it must fit beside host_overhead_mb alone, or the cap is 0. That
    holds when it reads its weights into memory too, where the copy is held twice (_copy_fits).
    role 'any' is the upper bound that prunes node subsets before an order is fixed: the
    smallest overhead the node could pay in any role, and nothing else charged.
    In every role, a node's memory_reserve_mb stays free on top of what it is charged, and its
    max_weights_mb, if it has one, caps its blocks plus a head it holds, whatever its memory allows."""
    if role == "host" and uploads and dev.host_weights_mapped and dev.host_overhead_mb is not None:
        if not _copy_fits(dev, model, holds_head):
            return 0                  # the load-time copy cannot fit beside the host's own overhead
    budget = ((dev.usable_mem_mb - dev.memory_reserve_mb) * MB
              - charged_bytes(dev, model, holds_head, holds_embd, role, uploads))
    if budget <= 0:
        return 0
    cap = int(budget // model.per_layer_footprint_bytes())
    if dev.max_weights_mb is not None:
        weights = dev.max_weights_mb * MB - (model.head_bytes if holds_head else 0.0)
        cap = min(cap, max(0, int(weights // model.bytes_per_layer)))
    return cap


def charged_bytes(dev: DeviceProfile, model: ModelProfile, holds_head: bool = False,
                  holds_embd: bool = False, role: Optional[str] = None, uploads: bool = False) -> float:
    """Everything memory_cap charges a node apart from its layers, by the rules in its docstring:
    the runtime overhead in its role, a head it holds, the embedding (the stand-in, or a host that
    loads its weights), and the upload or received-tensor buffer (board-aware only). The one
    source of both memory_cap and memory_need_mb, so a plan and the memory gate it is run against
    cannot disagree."""
    head = model.head_bytes if holds_head else 0.0
    if role == "any":
        measured = [x for x in (dev.host_overhead_mb, dev.worker_overhead_mb) if x is not None]
        return min([model.runtime_overhead_mb] + measured) * MB
    if role == "host" and dev.host_overhead_mb is not None:
        upload = max(0.0 if holds_head else model.head_bytes, model.max_block_tensor_bytes) if uploads else 0.0
        if dev.host_weights_mapped:
            upload = 0.0              # checked apart, in memory_cap: see DeviceProfile.host_weights_mapped
        embd = model.embd_bytes if (holds_embd and model.host_loads_weights) else 0.0   # see host_loads_weights
        return dev.host_overhead_mb * MB + head + upload + embd
    if role == "worker" and dev.worker_overhead_mb is not None:
        again = max(head, model.max_block_tensor_bytes) if dev.worker_repays_received else 0.0
        return dev.worker_overhead_mb * MB + head + again
    return model.runtime_overhead_mb * MB + head + (model.embd_bytes if holds_embd else 0.0)


def _copy_fits(dev: DeviceProfile, model: ModelProfile, holds_head: bool) -> bool:
    """The load-time check of a host whose upload copy is not charged with its blocks (DeviceProfile.host_weights_mapped):
    the copy of the largest tensor it uploads must fit beside host_overhead_mb alone. Mapped, the copy is RPC's serialized
    message (ggml-rpc.cpp serialize_set_tensor). Reading its weights into memory (ModelProfile.host_loads_weights), the
    host holds the tensor twice while it sends it -- the loader's staging buffer (llama-model-loader.cpp:1739 at the pin)
    and that message -- both freed before the next tensor, and before the KV cache and compute buffers exist. "Alone"
    rests on the file order: each of the seven model files on the Pi 4 stores output.weight first and the token
    embedding before block 0 (2026-10-09), so the head's copy comes and goes before any block is read."""
    copy = max(0.0 if holds_head else model.head_bytes, model.max_block_tensor_bytes)
    copies = 2 if model.host_loads_weights else 1
    return (dev.usable_mem_mb - dev.memory_reserve_mb) * MB >= dev.host_overhead_mb * MB + copies * copy


def memory_need_mb(dev: DeviceProfile, model: ModelProfile, block_bytes: float, n_blocks: int,
                   holds_head: bool = False, holds_embd: bool = False, role: Optional[str] = None,
                   uploads: bool = False) -> float:
    """MiB a node needs to hold these blocks -- block_bytes is their real size from the tensor
    table -- with their KV cache, in its role: what a pre-registered cell is gated against."""
    kv = model.kv_bytes_per_layer_per_token * model.context_len * n_blocks
    return (block_bytes + kv + charged_bytes(dev, model, holds_head, holds_embd, role, uploads)) / MB


def _head_layers(model: ModelProfile, phase: str) -> float:
    """The output head's work, in layers, on the node that holds it. Decode reads it for every
    token, and decode time follows bytes read (testbed/dataset, within 4 % on the Pis), so it
    costs head_bytes / bytes_per_layer layers there. Prefill computes logits for the last prompt
    token only, so its share per prompt token is negligible and it is not charged."""
    per_layer = model.active_bytes_per_layer or model.bytes_per_layer   # the bytes one layer reads per token
    if phase != "decode" or not model.head_bytes or not per_layer:
        return 0.0
    return model.head_bytes / per_layer


def _node_ms(participants, devices, model, phase) -> dict:
    """ms per layer for each participant in its role: the first node runs llama.cpp itself when
    model.host_in_pipeline, and every other participant is an RPC worker."""
    host = participants[0] if (participants and model.host_in_pipeline) else None
    return {nid: devices[nid].layer_ms(phase, as_worker=(nid != host)) for nid in participants}


def _caps(participants, devices, model, head_node=None) -> dict:
    """Memory caps for an ordered pipeline: the last node holds the output head (head_node moves it,
    e.g. onto the host as prima.cpp's Halda places it), and the first the token embedding when it
    is the process that loads the model. That first node is then the host, uploading to the others
    if there are any; every other participant is an RPC worker -- the roles memory_cap's
    board-aware accounting needs."""
    first, last = participants[0], participants[-1]
    head_at = head_node if head_node in participants else last
    host = first if model.host_in_pipeline else None
    return {nid: memory_cap(devices[nid], model, holds_head=(nid == head_at),
                            holds_embd=(model.host_in_pipeline and nid == first),
                            role=("host" if nid == host else "worker"),
                            uploads=(nid == host and len(participants) > 1))
            for nid in participants}


def evaluate(order, layers, devices, model, phase="decode", head_node=None, idle_on=()) -> dict:
    """Predicted latency and energy for one assignment. Same cost model as the objective.

    phase is "decode", "prefill", or a request workload {"prefill": prompt tokens, "decode": generated
    tokens} (see evaluate_request). head_node puts the output head on that participant instead of the
    last one. idle_on: boards outside the plan that stay powered, charged their idle power for the whole token
    (plan_optimal(unused="idle")); a board a plan leaves out is otherwise assumed switched off."""
    if isinstance(phase, dict):
        return evaluate_request(order, layers, devices, model, phase, head_node, idle_on)
    participants = [nid for nid in order if layers.get(nid, 0) > 0]
    c = _node_ms(participants, devices, model, phase)
    head = _head_layers(model, phase)
    head_at = head_node if head_node in participants else participants[-1]
    work = {nid: layers[nid] + (head if nid == head_at else 0.0) for nid in participants}

    compute_ms = sum(work[nid] * c[nid] for nid in participants)
    comm_ms = 0.0
    for a, b in zip(participants, participants[1:]):
        comm_ms += hop_ms(devices[a], devices[b], model)

    total_ms = compute_ms + comm_ms
    if total_ms <= 0:
        return {"ms_per_token": 0, "tokens_per_s": 0, "joules_per_token": 0, "mean_power_w": 0}

    out = {
        "ms_per_token": round(total_ms, 3),
        "tokens_per_s": round(1000.0 / total_ms, 4),
        "compute_ms": round(compute_ms, 3),
        "comm_ms": round(comm_ms, 3),
        "comm_fraction": round(comm_ms / total_ms, 4),
        "participants": participants,
        "phase": phase,
    }
    idle_on = [nid for nid in idle_on if nid not in participants]
    unpowered = _unpowered(devices, participants + idle_on)
    if unpowered:
        # Not measured, so not computed: a partial sum over the powered nodes would look like
        # the whole cluster's energy while omitting some of it.
        out.update(joules_per_token=None, mean_power_w=None, wh_per_1k_tokens=None,
                   energy="not measured: no power figure for " + ", ".join(sorted(unpowered)))
        return out
    energy_j = 0.0
    for nid in participants:
        dev = devices[nid]
        busy_s = work[nid] * c[nid] / 1000.0
        idle_s = max(0.0, total_ms / 1000.0 - busy_s)
        energy_j += busy_s * dev.active_w(phase) + idle_s * dev.power_idle_w
    for nid in idle_on:
        energy_j += total_ms / 1000.0 * devices[nid].power_idle_w
    out.update(joules_per_token=round(energy_j, 4),
               mean_power_w=round(energy_j / (total_ms / 1000.0), 3),
               wh_per_1k_tokens=round(energy_j * 1000 / 3600.0, 4))
    return out


def evaluate_request(order, layers, devices, model, workload, head_node=None, idle_on=()) -> dict:
    """One request of workload {"prefill": P, "decode": G}: P prompt tokens, then G generated ones, each phase at
    its own per-token cost (the characterisation rows price the phases separately).

    request_ms = P x prefill ms/token + G x decode ms/token
    ttft_ms    = P x prefill ms/token (the first token's logits come out of the prompt pass)
    ms_per_token / tokens_per_s are the decode figures, the time per output token (TPOT); joules_per_request,
    when every participant has a power figure, is the sum over the phases."""
    per = {ph: evaluate(order, layers, devices, model, ph, head_node, idle_on) for ph in workload}
    request_ms = sum(n * per[ph]["ms_per_token"] for ph, n in workload.items())
    main = per.get("decode") or next(iter(per.values()))
    out = {"workload": dict(workload), "request_ms": round(request_ms, 1),
           "ttft_ms": round(workload.get("prefill", 0) * per["prefill"]["ms_per_token"], 1) if "prefill" in per else None,
           "ms_per_token": main["ms_per_token"], "tokens_per_s": main["tokens_per_s"],
           "participants": main.get("participants", []), "phase": "request", "per_phase": per}
    joules = [per[ph].get("joules_per_token") for ph in workload]
    if all(j is not None for j in joules):
        out["joules_per_request"] = round(sum(n * j for n, j in zip(workload.values(), joules)), 3)
        out["joules_per_token"] = main.get("joules_per_token")
        out["mean_power_w"] = main.get("mean_power_w")
    else:
        out.update(joules_per_token=None, mean_power_w=None,
                   energy=next((per[ph].get("energy") for ph in workload if per[ph].get("energy")), "not measured"))
    return out


def with_headroom(devices, keep_free_mb) -> dict:
    """The same devices, each told to keep keep_free_mb more MiB free than it already must (memory_reserve_mb): a
    number for every node, or a dict by node id. prima.cpp designs for devices that stay usable while they serve;
    planning under the same headroom compares the two schedulers at equal memory stress and prices the headroom."""
    extra = keep_free_mb if isinstance(keep_free_mb, dict) else {nid: keep_free_mb for nid in devices}
    return {nid: replace(d, memory_reserve_mb=d.memory_reserve_mb + float(extra.get(nid, 0.0)))
            for nid, d in devices.items()}


# -------------------------------------------------------------------- baselines
def plan_even(devices, model, node_order=None, phase="decode") -> Placement:
    """Even split across every available node - the baseline the rule must beat.

    Deliberately naive: it ignores speed and memory entirely, exactly as a user would if
    they had no characterisation data. Where it is infeasible on memory, that fact is the
    result, so it is reported rather than repaired.
    """
    order = list(node_order or devices.keys())
    L = model.n_layers
    base, extra = divmod(L, len(order))
    layers = {nid: base + (1 if i < extra else 0) for i, nid in enumerate(order)}

    caps = _caps([nid for nid in order if layers[nid] > 0], devices, model)
    over = [nid for nid in caps if layers[nid] > caps[nid]]
    placement = Placement("even-split", order, layers, model.name)
    if over:
        placement.feasible = False
        placement.reason = "exceeds memory on: " + ", ".join(
            "%s (%d > %d layers)" % (nid, layers[nid], caps[nid]) for nid in over)
    placement.predicted = evaluate(order, layers, devices, model, phase)
    return placement


def plan_single(devices, model, node_id, phase="decode") -> Placement:
    """Everything on one node - the "just buy more RAM" alternative from the decision matrix."""
    layers = {node_id: model.n_layers}
    placement = Placement("single-node:" + node_id, [node_id], layers, model.name)
    cap = memory_cap(devices[node_id], model, holds_head=True, holds_embd=model.host_in_pipeline,
                     role=("host" if model.host_in_pipeline else "worker"))
    if model.n_layers > cap:
        placement.feasible = False
        placement.reason = "%s holds at most %d of %d layers" % (node_id, cap, model.n_layers)
    placement.predicted = evaluate([node_id], layers, devices, model, phase)
    return placement


# ------------------------------------------------------------------- the rule
# The exact fill packs each board to its memory accounting's edge (the plans it changes kept 4-37 MiB), so by default it
# leaves this much free on every board, on top of the board's own memory_reserve_mb. The figure is evidence-based: across
# the 83 board-cells measured since the board-aware accounting (2026-09-28 to 10-07, every pool_results_*.json with a
# memory probe), the accounting under-estimated a board's MemAvailable drop 6 times, by at most 133 MiB
# (pool_results_20260928c.json, p4.qwen14b.k46_forced, the Pi 5; then 117, 77, 76). 256 MiB is about twice that.
EXACT_FILL_HEADROOM_MB = 256.0


def plan_optimal(devices, model, phase="decode", energy_weight=0.0,
                 max_permutations=5000, force_include=None, host_candidates=None,
                 head_placement="best", fill="auto", headroom_mb=None, unused="off") -> Placement:
    """The heterogeneity-aware rule.

    energy_weight (lambda) is in seconds per joule: 0 optimises pure latency, and a larger
    value trades latency for energy. Sweeping it traces the Pareto front the proposal asks
    for, which is why it is a parameter rather than a constant.

    force_include pins nodes into the participating set - used by the straggler study to
    ask "what does the best plan look like if the Pi 4 must be used?" and compare it with
    the unconstrained answer.

    host_candidates, with model.host_in_pipeline, names the nodes that may run llama.cpp
    itself -- the ones that hold the model file (the 14B lives on the Pi 5 alone). None
    allows any. Without host_in_pipeline the host is the control node and this is unused.

    phase may also be a request workload {"prefill": P, "decode": G}: the rule then minimises the
    whole request, P x prefill + G x decode (evaluate_request), rather than one phase's per-token
    time. The objective stays linear in the layer counts, so the fill below is still exact.

    head_placement: where the output head goes. "best" (the default since 2026-10-08, user's decision) tries both
    places for every order and keeps the cheaper: measured on the boards 2026-10-07, Qwen3-14B on Pi 4 + Jetson ran
    5.4 % faster in decode and 13.8 % in prefill with the head on the host, while on TinyLlama the head stays last, where
    Halda's head-on-host layout measured 6.4 % slower (101.7 against 95.6 ms). "last" puts it on the last participant,
    where llama.cpp's -ngl puts it: every plan made before 2026-10-08. "host" keeps it on the first participant, the
    host, which is what prima.cpp's Halda does; at the pin, -ot "^output\\.weight=CPU" (anchored: llama.cpp
    regex_searches tensor names). The head is read on every decoded token (for Qwen3-14B, 638 MB), so on a slow last
    board it costs more than one layer's worth: on the Pi 4, about three times what it costs on a Pi 5
    host. Only with model.host_in_pipeline and two or more participants; otherwise it is "last".

    fill: how a fixed order's layers are counted out. "exact" checks each node's actual contiguous run of blocks
    (model.layer_bytes) against its budget, by a dynamic programme over the layer boundaries: where blocks differ in size
    -- a hybrid's recurrent and attention blocks, a K-quant's mixed blocks -- the average over- or under-counts what a
    given node can hold (it over-committed in 296 of 3,200 random unequal models, 2026-10-07). "average" caps each node
    at its budget over the AVERAGE block, then fills greedily: every plan registered before 2026-10-07. "auto" (the
    default since 2026-10-07, user's decision) is "exact" when the model carries its block sizes, else "average". Same
    charges, same objective.

    unused: what a board the plan leaves out does, for the energy term (energy_weight > 0) and the predicted joules.
    "off" (the default, and every plan before 2026-10-08): it is switched off and costs nothing. "idle": it stays
    powered and is charged its idle power for every token -- on the boards here 2.3-3.1 W, which is what makes a plan
    that drops a board look cheaper than it is unless the board is really switched off (the strip can; prima.cpp and
    Halda never do).

    headroom_mb: MiB every board keeps free on top of its own memory_reserve_mb (with_headroom). None (the default) is
    EXACT_FILL_HEADROOM_MB with the exact fill and 0 with the average fill, so a model without block sizes plans exactly
    as before. 0 turns it off.
    """
    if head_placement not in ("last", "host", "best"):
        raise ValueError("head_placement must be 'last', 'host' or 'best', not %r" % (head_placement,))
    if fill not in ("auto", "average", "exact"):
        raise ValueError("fill must be 'auto', 'average' or 'exact', not %r" % (fill,))
    if unused not in ("off", "idle"):
        raise ValueError("unused must be 'off' or 'idle', not %r" % (unused,))
    if fill == "auto":
        fill = "exact" if model.layer_bytes is not None else "average"
    if fill == "exact" and (not model.layer_bytes or len(model.layer_bytes) != model.n_layers):
        raise ValueError("fill='exact' needs model.layer_bytes, one entry per layer (%d)" % model.n_layers)
    if headroom_mb is None:
        headroom_mb = EXACT_FILL_HEADROOM_MB if fill == "exact" else 0.0
    if headroom_mb:
        devices = with_headroom(devices, headroom_mb)
    node_ids = list(devices.keys())
    force_include = set(force_include or [])
    if energy_weight > 0:
        _require_power(devices, node_ids, "energy-weighted planning (energy_weight=%g)" % energy_weight)
    best = None
    checked = 0

    for size in range(1, len(node_ids) + 1):
        for subset in itertools.combinations(node_ids, size):
            if force_include and not force_include.issubset(subset):
                continue
            # Without the head, the embedding and any role these caps are an upper bound, so a
            # subset that fails here fails in every order; those are charged per order below, to
            # the nodes that hold them, in the role each has there.
            # (The exact fill skips this average-block pruning: a node that takes the small blocks can hold more
            # than its budget over the average block, so it would not be an upper bound there.)
            caps = {nid: memory_cap(devices[nid], model, role="any") for nid in subset}
            if fill == "average" and sum(caps.values()) < model.n_layers:
                continue                      # cannot hold the model at all
            if fill == "average" and any(caps[nid] < 1 for nid in subset):
                continue                      # a participant that can hold no layer is not one

            # Only the energy term reads power; at energy_weight == 0 it is not needed.
            idle_sum = (sum(devices[nid].power_idle_w for nid in (node_ids if unused == "idle" else subset))
                        if energy_weight > 0 else 0.0)
            for order in itertools.permutations(subset):
                checked += 1
                if checked > max_permutations:
                    break
                if model.host_in_pipeline and host_candidates is not None and order[0] not in host_candidates:
                    continue
                heads = [None]
                if model.host_in_pipeline and len(order) > 1 and head_placement != "last":
                    heads = [order[0]] if head_placement == "host" else [None, order[0]]
                for hn in heads:
                    if fill == "exact":
                        layers, cost = _fill_exact(order, devices, model, phase, energy_weight, idle_sum,
                                                   head_node=hn)
                        if layers is None:
                            continue
                        if best is None or cost < best[0]:
                            best = (cost, list(order), layers, hn)
                        continue
                    order_caps = _caps(list(order), devices, model, head_node=hn)
                    if any(c < 1 for c in order_caps.values()):
                        continue
                    layers, cost = _fill(order, order_caps, devices, model, phase, energy_weight, idle_sum,
                                         head_node=hn)
                    if layers is None:
                        continue
                    if best is None or cost < best[0]:
                        best = (cost, list(order), layers, hn)

    if best is None:
        placement = Placement("heteroserve-rule", node_ids, {}, model.name)
        placement.feasible = False
        # State the arithmetic that actually decided, not the raw memory sum. `memory_cap`
        # subtracts `runtime_overhead_mb` PER NODE, so the raw sum is always larger than
        # what the cluster can really hold -- and quoting it alone produced refusals whose
        # own numbers said the model should have fitted ("needs 11.6 GB, cluster offers
        # 12.0 GB usable"). That reads as a bug in the refusal rather than a result, and
        # this refusal IS a headline result: see config/experiments/memory_wall_14b.json.
        per_layer_mb = model.per_layer_footprint_bytes() / MB
        caps = {nid: memory_cap(d, model, role="any") for nid, d in devices.items()}
        holdable = sum(caps.values())
        raw_gb = sum(d.usable_mem_mb for d in devices.values()) / 1024
        # The overhead each node pays in its cheapest role: the stand-in unless measured.
        overheads = {nid: min([model.runtime_overhead_mb] + [x for x in (d.host_overhead_mb, d.worker_overhead_mb)
                                                             if x is not None])
                     for nid, d in devices.items()}
        net_gb = sum(max(0.0, d.usable_mem_mb - overheads[nid] - d.memory_reserve_mb)
                     for nid, d in devices.items()) / 1024
        if set(overheads.values()) == {model.runtime_overhead_mb}:
            overhead_text = "%.0f MB of runtime overhead" % model.runtime_overhead_mb
        else:
            overhead_text = "its runtime overhead (%s)" % ", ".join(
                "%s %.0f MB" % (n, o) for n, o in sorted(overheads.items()))
        reserves = ["%s %.0f MB" % (n, d.memory_reserve_mb) for n, d in sorted(devices.items()) if d.memory_reserve_mb]
        if reserves:
            overhead_text += " and the reserve it must keep free (%s)" % ", ".join(reserves)
        placement.reason = (
            "no subset of the cluster can hold %s: it needs %d layers x %.1f MB = %.1f GB, "
            "and the cluster can hold %d layers (%.1f GB). The raw memory sum is %.1f GB, "
            "but each participating node also pays %s, leaving "
            "%.1f GB for weights. Per-node capacity in layers: %s."
            % (model.name, model.n_layers, per_layer_mb,
               model.n_layers * per_layer_mb / 1024,
               holdable, holdable * per_layer_mb / 1024,
               raw_gb, overhead_text, net_gb,
               ", ".join("%s=%d" % (n, c) for n, c in sorted(caps.items()))))
        if model.head_bytes or (model.embd_bytes and model.host_in_pipeline):
            placement.reason += (" The pipeline's last node also holds the %.0f MB output head%s, "
                                 "so holds fewer layers than that."
                                 % (model.head_bytes / MB,
                                    " and its first the %.0f MB token embedding" % (model.embd_bytes / MB)
                                    if model.host_in_pipeline and model.embd_bytes else ""))
        return placement

    _, order, layers, head_node = best
    placement = Placement("heteroserve-rule", order, layers, model.name, head_node=head_node)
    placement.predicted = evaluate(order, layers, devices, model, phase, head_node=head_node,
                                   idle_on=[n for n in node_ids if n not in order] if unused == "idle" else ())
    placement.predicted["energy_weight"] = energy_weight
    return placement


def _fill(order, caps, devices, model, phase, energy_weight, idle_sum, head_node=None):
    """Optimal layer counts for a fixed ordered subset.

    The objective is linear in the layer counts once the participating set is fixed, so
    the optimum is: give every participant one layer (it must hold at least one to be a
    participant), then hand out the remaining layers to the cheapest weight still under
    its cap. Ties go to the earlier node, which keeps the result deterministic.
    """
    if sum(caps[nid] for nid in order) < model.n_layers:
        return None, math.inf

    # One phase, or a request workload {"prefill": P, "decode": G}: tokens per phase weight each phase's cost.
    mix = phase if isinstance(phase, dict) else {phase: 1.0}
    c_by = {ph: _node_ms(list(order), devices, model, ph) for ph in mix}
    c = {nid: sum(n * c_by[ph][nid] for ph, n in mix.items()) for nid in order}
    weights = {}
    for nid in order:
        dev = devices[nid]
        if energy_weight > 0:                     # each phase's time at that phase's power above idle
            weights[nid] = sum(n * c_by[ph][nid] * (1.0 + energy_weight * (dev.delta_w(ph) + idle_sum))
                               for ph, n in mix.items())
        else:
            weights[nid] = c[nid]                  # latency only: power is never read

    layers = {nid: 1 for nid in order}
    remaining = model.n_layers - len(order)
    if remaining < 0:
        return None, math.inf

    for nid in sorted(order, key=lambda n: (weights[n], order.index(n))):
        if remaining <= 0:
            break
        take = min(remaining, caps[nid] - 1)
        if take > 0:
            layers[nid] += take
            remaining -= take
    if remaining > 0:
        return None, math.inf

    # Drop any node left holding nothing useful, then cost the result.
    comm_ms = 0.0
    participants = [nid for nid in order if layers[nid] > 0]
    for a, b in zip(participants, participants[1:]):
        comm_ms += hop_ms(devices[a], devices[b], model)

    last = head_node if head_node in participants else participants[-1]   # the node holding the head
    factor = weights[last] / c[last] if c[last] else 1.0          # that node's energy factor
    cost = sum(layers[nid] * weights[nid] for nid in participants)
    cost += sum(n * _head_layers(model, ph) * c_by[ph][last] for ph, n in mix.items()) * factor   # the head
    cost += comm_ms * sum(mix.values()) * (1.0 + energy_weight * idle_sum)
    return layers, cost


def _fill_exact(order, devices, model, phase, energy_weight, idle_sum, head_node=None):
    """_fill with each node's ACTUAL blocks checked, not a count against the average block (plan_optimal fill="exact").

    llama.cpp gives each device a contiguous run of blocks, in pipeline order, so the choice is the k-1 boundaries
    between them. Node i can hold the run [a, b) when its blocks' real bytes (model.layer_bytes) plus their KV cache fit
    the budget memory_cap would give it -- the same charged_bytes in the same role, its memory_reserve_mb, and its
    max_weights_mb cap with a head it holds -- and every participant holds at least one block. The objective is
    _fill's (linear in the counts), minimised over the boundaries by a dynamic programme: O(k x L^2), trivial at k <= 5,
    L <= 100. Ties keep more blocks on the earlier node, as _fill does."""
    order = list(order)
    k, L = len(order), model.n_layers
    if k > L:
        return None, math.inf
    mix = phase if isinstance(phase, dict) else {phase: 1.0}
    c_by = {ph: _node_ms(order, devices, model, ph) for ph in mix}
    c = {nid: sum(n * c_by[ph][nid] for ph, n in mix.items()) for nid in order}
    weights = {nid: sum(n * c_by[ph][nid] * (1.0 + energy_weight * (devices[nid].delta_w(ph) + idle_sum))
                        for ph, n in mix.items()) if energy_weight > 0 else c[nid] for nid in order}
    head_at = head_node if head_node in order else order[-1]
    host = order[0] if model.host_in_pipeline else None
    avail, wcap = [], []
    for nid in order:
        dev = devices[nid]
        holds_head, holds_embd = nid == head_at, model.host_in_pipeline and nid == order[0]
        role, uploads = ("host" if nid == host else "worker"), (nid == host and k > 1)
        if role == "host" and uploads and dev.host_weights_mapped and dev.host_overhead_mb is not None:
            if not _copy_fits(dev, model, holds_head):
                return None, math.inf         # memory_cap's 0: the load-time copy cannot fit beside the overhead
        avail.append((dev.usable_mem_mb - dev.memory_reserve_mb) * MB
                     - charged_bytes(dev, model, holds_head, holds_embd, role, uploads))
        wcap.append(dev.max_weights_mb * MB - (model.head_bytes if holds_head else 0.0)
                    if dev.max_weights_mb is not None else math.inf)
    kv = model.kv_bytes_per_layer_per_token * model.context_len
    F, W = [0.0], [0.0]                       # prefix sums: footprint (weights + KV) and weights alone
    for b in model.layer_bytes:
        F.append(F[-1] + b + kv)
        W.append(W[-1] + b)
    inf = math.inf
    f = [[inf] * (L + 1) for _ in range(k + 1)]
    cut = [[-1] * (L + 1) for _ in range(k + 1)]
    f[0][0] = 0.0
    for i in range(1, k + 1):                 # node order[i-1] takes [a, b); nodes after it need >= 1 block each
        w = weights[order[i - 1]]
        for b in range(i, L - (k - i) + 1):
            for a in range(b - 1, i - 2, -1):     # descending a: on a tie the earlier nodes keep more blocks
                if f[i - 1][a] == inf:
                    continue
                if F[b] - F[a] > avail[i - 1] or W[b] - W[a] > wcap[i - 1]:
                    break                     # a longer run [a', b) with a' < a only weighs more
                v = f[i - 1][a] + (b - a) * w
                if v < f[i][b]:
                    f[i][b], cut[i][b] = v, a
    if f[k][L] == inf:
        return None, math.inf
    layers, b = {}, L
    for i in range(k, 0, -1):
        a = cut[i][b]
        layers[order[i - 1]] = b - a
        b = a
    comm_ms = sum(hop_ms(devices[x], devices[y], model) for x, y in zip(order, order[1:]))
    factor = weights[head_at] / c[head_at] if c[head_at] else 1.0
    cost = f[k][L]
    cost += sum(n * _head_layers(model, ph) * c_by[ph][head_at] for ph, n in mix.items()) * factor
    cost += comm_ms * sum(mix.values()) * (1.0 + energy_weight * idle_sum)
    return layers, cost


# ------------------------------------------------------------------ comparison
def compare(devices, model, phase="decode", energy_weight=0.0) -> dict:
    """Everything the placement chapter needs in one call.

    Includes the straggler contrast explicitly: the best plan overall, versus the best
    plan that is forced to use the straggler. The difference between the two IS the answer
    to QR8, expressed as a number instead of an opinion.
    """
    rule = plan_optimal(devices, model, phase, energy_weight)
    even = plan_even(devices, model, phase=phase)

    stragglers = [nid for nid, d in devices.items() if d.is_straggler]
    forced = plan_optimal(devices, model, phase, energy_weight,
                          force_include=stragglers) if stragglers else None
    without = None
    if stragglers:
        remaining = {nid: d for nid, d in devices.items() if nid not in stragglers}
        if remaining:
            without = plan_optimal(remaining, model, phase, energy_weight)

    singles = {nid: plan_single(devices, model, nid, phase) for nid in devices}

    out = {
        "model": model.name,
        "phase": phase,
        "energy_weight": energy_weight,
        "rule": rule,
        "even": even,
        "with_straggler": forced,
        "without_straggler": without,
        "singles": singles,
    }

    if rule.feasible and even.feasible and even.predicted.get("ms_per_token"):
        out["tpot_improvement_pct"] = round(
            100.0 * (even.predicted["ms_per_token"] - rule.predicted["ms_per_token"])
            / even.predicted["ms_per_token"], 2)
        if even.predicted.get("joules_per_token") and rule.predicted.get("joules_per_token") is not None:
            out["energy_improvement_pct"] = round(
                100.0 * (even.predicted["joules_per_token"] - rule.predicted["joules_per_token"])
                / even.predicted["joules_per_token"], 2)

    if forced and without and forced.feasible and without.feasible:
        out["straggler_cost_pct"] = round(
            100.0 * (forced.predicted["ms_per_token"] - without.predicted["ms_per_token"])
            / without.predicted["ms_per_token"], 2)
        if (without.predicted.get("joules_per_token")
                and forced.predicted.get("joules_per_token") is not None):
            out["straggler_energy_pct"] = round(
                100.0 * (forced.predicted["joules_per_token"] - without.predicted["joules_per_token"])
                / without.predicted["joules_per_token"], 2)
    return out


def pareto(devices, model, phase="decode", weights=None) -> list:
    """Sweep lambda to trace the latency/energy Pareto front across placements."""
    weights = weights or [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    _require_power(devices, list(devices), "a latency/energy Pareto front")
    front = []
    seen = set()
    for lam in weights:
        p = plan_optimal(devices, model, phase, energy_weight=lam)
        if not p.feasible:
            continue
        key = tuple(sorted(p.layers.items()))
        if key in seen:
            continue
        seen.add(key)
        front.append({
            "energy_weight": lam,
            "order": p.order,
            "layers": p.layers,
            "ms_per_token": p.predicted["ms_per_token"],
            "tokens_per_s": p.predicted["tokens_per_s"],
            "joules_per_token": p.predicted["joules_per_token"],
            "mean_power_w": p.predicted["mean_power_w"],
        })
    return front
