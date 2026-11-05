"""Turn measured data into the profiles the placement rule consumes.

The rule is only as good as its inputs, and the whole point of the project is that those
inputs are measured rather than assumed. This module is the boundary between the two:

    bench results.csv  (throughput per node)   ->  ms_per_layer_decode / _prefill
    power.csv          (rail power per node)   ->  power_active_w / power_idle_w
    link_survey.json   (iperf3 + ping)         ->  link_mbps / rtt_ms
    device_info.json   (agent inventory)       ->  usable_mem_mb

Anything that cannot be derived from a measurement is left as None and the loader says so,
rather than substituting a plausible default. A placement computed from invented numbers
would look exactly like one computed from measured numbers, which is why this refuses.
"""
from __future__ import annotations

import csv
import json
import sys
import statistics
from dataclasses import replace
from pathlib import Path

from . import analysis, gguf
from .placement import DeviceProfile, ModelProfile


class ProfileError(Exception):
    pass


# ----------------------------------------------------------------- model side
def model_profile(model_path, context_len: int = 2048, overrides: dict | None = None) -> ModelProfile:
    """Read the model's shape from its GGUF header."""
    shape = gguf.shape(model_path)
    shape.update(overrides or {})
    return ModelProfile(
        name=shape["name"],
        n_layers=shape["n_layers"],
        bytes_per_layer=shape["bytes_per_layer"],
        kv_bytes_per_layer_per_token=shape["kv_bytes_per_layer_per_token"],
        activation_bytes=shape["activation_bytes"],
        context_len=context_len,
        head_bytes=float(shape.get("head_bytes") or 0.0),
        embd_bytes=float(shape.get("embd_bytes") or 0.0),
        max_block_tensor_bytes=float(shape.get("max_block_tensor_bytes") or 0.0),
        active_bytes_per_layer=float(shape.get("active_bytes_per_layer") or 0.0),
        params_per_layer=float(shape.get("params_per_layer") or 0.0),
        active_params_per_layer=float(shape.get("active_params_per_layer") or 0.0),
        # real block sizes from the tensor table: plan_optimal's default (fill="auto") then uses the exact fill
        layer_bytes=tuple(float(b) for b in shape["layer_bytes"]) if shape.get("layer_bytes") else None,
    )


def model_profile_from_dict(spec: dict) -> ModelProfile:
    """Build a profile from config/models.json - for planning before a model is downloaded."""
    required = ("name", "n_layers", "bytes_per_layer", "kv_bytes_per_layer_per_token", "activation_bytes")
    missing = [k for k in required if k not in spec]
    if missing:
        raise ProfileError("model spec %s is missing: %s" % (spec.get("name", "?"), ", ".join(missing)))
    return ModelProfile(
        name=spec["name"],
        n_layers=int(spec["n_layers"]),
        bytes_per_layer=float(spec["bytes_per_layer"]),
        kv_bytes_per_layer_per_token=float(spec["kv_bytes_per_layer_per_token"]),
        activation_bytes=float(spec["activation_bytes"]),
        context_len=int(spec.get("context_len", 2048)),
        runtime_overhead_mb=float(spec.get("runtime_overhead_mb", 350.0)),
        head_bytes=float(spec.get("head_bytes", 0.0)),
        embd_bytes=float(spec.get("embd_bytes", 0.0)),
        host_in_pipeline=bool(spec.get("host_in_pipeline", False)),
        max_block_tensor_bytes=float(spec.get("max_block_tensor_bytes", 0.0)),
        active_bytes_per_layer=float(spec.get("active_bytes_per_layer", 0.0)),
        params_per_layer=float(spec.get("params_per_layer", 0.0)),
        active_params_per_layer=float(spec.get("active_params_per_layer", 0.0)),
    )


# ---------------------------------------------------------------- device side
def _rows(csv_path) -> list:
    with Path(csv_path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def model_geometry(model_dir, names) -> dict:
    """{model file name: (bytes a decoded token reads, block parameters)} from the GGUF headers in model_dir, for the
    profiled models. Bytes read = every block + the output head; prefill computes the blocks. A model whose file is not
    there is left out, with a warning, and its rows then give per-layer figures only."""
    out = {}
    for name in names:
        path = Path(model_dir) / name
        if not path.is_file():
            print("warning: %s is not in %s, so no per-byte cost comes from its rows (per-layer figures only, which belong to that model alone)" % (name, model_dir), file=sys.stderr)
            continue
        s = gguf.shape(path)
        blocks = sum(s["layer_bytes"]) if s.get("layer_bytes") else s["bytes_per_layer"] * s["n_layers"]
        out[name] = (float(blocks + (s.get("head_bytes") or 0)), float(s["params_per_layer"] * s["n_layers"]))
    return out


def per_layer_times(results_csv, n_layers_by_model: dict, threads=None, geometry=None) -> dict:
    """ms per layer, per node, per phase, from the isolated characterisation rows.

    A node's isolated decode run holds the whole model, so ms_per_token / n_layers is that
    node's per-layer cost. Rows carrying validity flags (throttled, no window) are dropped:
    a flagged row is not a measurement, and averaging it in would quietly poison the rule.

    Where a node was measured on several small models, the median across models is used.
    Per-layer cost is not perfectly model-independent - attention width differs - but on
    this device class the spread is small compared with the between-node spread the rule
    is actually deciding on.

    Thread count is NOT pooled that way. The sweep in docs/05 covers 1, 2 and 4 threads
    while the cluster run uses llama.cpp's default (all cores), so a median taken across
    the sweep returns roughly the 2-thread cost - inflated 1.35-1.95x per node - and that
    goes straight into predicted tokens_per_s and joules_per_token, a named deliverable,
    with no flag. docs/06_troubleshooting.md names this as the first thing to check when
    predictions miss by a consistent margin. Pass `threads=` to pick the operating point;
    a CSV spanning several thread counts with no choice made is refused rather than
    silently averaged, per this module's "no plausible defaults" contract.
    """
    acc, unit = {}, {}
    seen_threads = {}
    dropped = 0
    unmapped = {}
    for row in _rows(results_csv):
        # Which flags invalidate WHICH column is not one question, and this used to collapse
        # it: ANY flag removed the row from the placement rule. But the power-trace flags
        # (gap:, late-start:, too-few-samples:, uncalibrated-channel, no-window:,
        # energy-span:, low-duty:) say the ENERGY join is untrustworthy -- the tokens_per_s
        # from that same run is a perfectly good benchmark. analysis.characterisation_table
        # has always drawn that line (its `unthrottled` filter); the placement rule did not,
        # so the report table and Rule A silently disagreed about which runs counted.
        #
        # The practical effect was severe: with no calibration file on disk, join_run stamps
        # `uncalibrated-channel` on every joined row of every node, so EVERY row was dropped
        # here and build_devices raised "unmeasured inputs" for nodes whose throughput data
        # was complete and correct.
        if analysis.invalidates_throughput(row.get("flags")):
            dropped += 1
            continue
        node = row["node_id"]
        phase = row["phase"]
        model = row["model"]
        n_layers = n_layers_by_model.get(model)
        if not n_layers:
            # The --layers map is hand-written and nothing derives it from the nodes'
            # models.txt or their GGUF headers, so a model the sweep measured but the map
            # does not name is entirely plausible -- and this used to skip it with no
            # counter and no message. A map written for one node's four models silently
            # discarded half of another node's cells, and the profile still built, so
            # nothing looked wrong.
            unmapped.setdefault(model, 0)
            unmapped[model] += 1
            continue
        if threads is not None and int(row.get("n_threads") or 0) != int(threads):
            continue
        ms = _float(row.get("ms_per_token"))
        if not ms:
            continue
        acc.setdefault(node, {}).setdefault(phase, []).append(ms / n_layers)
        if geometry and model in geometry:   # unit costs: decode per byte read, prefill per block parameter
            unit.setdefault(node, {}).setdefault(phase, []).append(ms / geometry[model][0 if phase == "tg" else 1])
        seen_threads.setdefault(node, set()).add(int(row.get("n_threads") or 0))

    out = {}
    for node, phases in acc.items():
        threads_seen = sorted(seen_threads.get(node, set()))
        if threads is None and len(threads_seen) > 1:
            raise ProfileError(
                "%s: the results span thread counts %s. c_i must be taken at the thread "
                "count the cluster run actually uses, not a median across the sweep - "
                "pooling inflates it and the error lands in predicted tokens_per_s and "
                "joules_per_token with no flag. Pass threads=<n> "
                "(heteroctl profiles --threads <n>)." % (node, threads_seen))
        out[node] = {
            "n_threads": threads_seen,
            "decode": round(statistics.median(phases["tg"]), 4) if phases.get("tg") else None,
            "prefill": round(statistics.median(phases["pp"]), 4) if phases.get("pp") else None,
            "samples": {p: len(v) for p, v in phases.items()},
            "decode_per_byte": statistics.median(unit[node]["tg"]) if unit.get(node, {}).get("tg") else None,
            "prefill_per_param": statistics.median(unit[node]["pp"]) if unit.get(node, {}).get("pp") else None,
        }
    out["_dropped_flagged_rows"] = dropped
    if unmapped:
        print("warning: %d result row(s) ignored because no layer count was supplied for "
              "%s. Add them to the --layers map, or their cells contribute nothing to the "
              "placement rule."
              % (sum(unmapped.values()), ", ".join(sorted(unmapped))), file=sys.stderr)
    return out


def power_levels(power_summary_json) -> dict:
    """Active and idle rail power per node, from the energy collector's summary."""
    data = json.loads(Path(power_summary_json).read_text(encoding="utf-8"))
    out = {}
    for node, entry in data.get("nodes", {}).items():
        out[node] = {
            "active_w": _float(entry.get("mean_power_w_active")),
            "idle_w": _float(entry.get("mean_power_w_idle")),
        }
    return out


def link_survey(path) -> dict:
    """Link bandwidth and RTT per node, with the conditions they were measured under.

    The conditions used to be projected away here. A survey records which interface tier it
    ran over and whether the link was lossy or retransmitting, and all of it was dropped one
    line after being read -- so a figure taken on the 2.5GbE tier, or across a link that was
    retransmitting heavily, entered the placement rule indistinguishable from a clean one.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {node: {"mbps": _float(e.get("mbps")), "rtt_ms": _float(e.get("rtt_ms")),
                   "tier": e.get("tier", ""), "iface": e.get("iface", ""),
                   "measured_utc": e.get("measured_utc", ""),
                   "warnings": list(e.get("warnings") or [])}
            for node, e in data.get("nodes", {}).items()}


def build_devices(cluster, results_csv=None, power_summary=None, link_json=None,
                  inventory_json=None, n_layers_by_model=None, threads=None,
                  strict: bool = True, model_dir=None) -> dict:
    """Assemble DeviceProfiles for every serving node.

    strict=True (the default, and what the harness uses) refuses to return a profile that
    rests on anything unmeasured. Pass strict=False only for planning sketches, and label
    the output as such wherever it appears.
    """
    geometry = model_geometry(model_dir, n_layers_by_model or {}) if (results_csv and model_dir) else {}
    times = per_layer_times(results_csv, n_layers_by_model or {},
                            threads=threads, geometry=geometry) if results_csv else {}
    dropped_rows = times.pop("_dropped_flagged_rows", 0) if times else 0
    powers = power_levels(power_summary) if power_summary else {}
    links = link_survey(link_json) if link_json else {}
    inventory = json.loads(Path(inventory_json).read_text(encoding="utf-8")) if inventory_json else {}

    devices = {}
    missing = []
    for node in cluster.workers():
        nid = node.id
        t = times.get(nid, {})
        p = powers.get(nid, {})
        link = links.get(nid, {})

        # Memory budget (user decision 2026-09-27): MemAvailable measured at rest -- cluster.json's
        # usable_mem_mb, whose provenance is in that node's notes -- and never more than the node
        # reports free right now (the inventory's live available_mb, when its agent sends one).
        # The engine's runtime_overhead_mb is the only headroom on top, and with exactly that it
        # reproduces the measured 14B boundary (testbed/dataset/placement_validation.json). The
        # agents' usable_budget_mb is no longer planned against: it is deliberate slack below
        # MemAvailable (Pi 4 NODE_USABLE_MB 2600, Pi 5 RPC_MEM_MB 6144), and with it the rule
        # refused the split that measured fastest and called a split that ran infeasible.
        usable = node.usable_mem_mb
        inv = (inventory.get("nodes") or {}).get(nid) or {}
        live = (inv.get("memory") or {}).get("available_mb")
        if live and live > 0:
            usable = min(usable, live)

        # link_mbps/rtt_ms belong here for the same reason as everything else: this module
        # refuses to invent inputs. They were previously defaulted inline (`or 1000.0`,
        # `or 0.3`), which is exactly the "plausible default" the module docstring forbids
        # -- an unmeasured link silently became 1 Gbit, understating the communication term
        # the study is measuring by more than 10x on a shaped tier, with nothing to show it.
        # Note `or` also swallowed a genuine measured 0; a 0 Mbps link means unreachable,
        # which is not a measurement to plan from either, so it counts as absent.
        link_mbps = link.get("mbps")
        rtt_ms = link.get("rtt_ms")
        needed = {
            "ms_per_layer_decode": t.get("decode"),
            "ms_per_layer_prefill": t.get("prefill"),
            "power_active_w": p.get("active_w"),
            "power_idle_w": p.get("idle_w") if p.get("idle_w") is not None else node.power_idle_w,
            "link_mbps": link_mbps if (link_mbps or 0) > 0 else None,
            "rtt_ms": rtt_ms if rtt_ms is not None and rtt_ms >= 0 else None,
        }
        # Power is not a strict requirement: a node with measured throughput and link but no
        # power figure builds as a LATENCY-ONLY profile (power None, labelled in notes). The
        # rule plans latency from it, and every energy figure reports None instead of the
        # 0 W that used to be substituted below -- a 0 made joules_per_token read as a clean
        # measured 0. Energy-weighted planning refuses such profiles (placement._require_power).
        optional = ("power_active_w", "power_idle_w")
        absent = [k for k, v in needed.items() if v is None and k not in optional]
        unpowered = [k for k in optional if needed[k] is None]
        if absent:
            missing.append("%s: %s" % (nid, ", ".join(absent)))
            if strict:
                continue

        devices[nid] = DeviceProfile(
            node_id=nid,
            ms_per_layer_decode=needed["ms_per_layer_decode"] or 0.0,
            ms_per_layer_prefill=needed["ms_per_layer_prefill"] or 0.0,
            usable_mem_mb=usable,
            power_active_w=needed["power_active_w"],     # None = not measured, never 0.0
            power_idle_w=needed["power_idle_w"],
            # Reached only in sketch mode (strict=False); strict refuses above. The
            # sketch values are deliberately the dataclass defaults, and a sketch must be
            # labelled as one by its caller.
            link_mbps=needed["link_mbps"] if needed["link_mbps"] is not None else 1000.0,
            rtt_ms=needed["rtt_ms"] if needed["rtt_ms"] is not None else 0.3,
            is_straggler=(node.role == "worker-straggler"),
            notes=(node.description + ("; " if node.description else "")
                   + "power not measured: latency-only profile") if unpowered else node.description,
            host_overhead_mb=getattr(node, "host_overhead_mb", None),
            worker_overhead_mb=getattr(node, "worker_overhead_mb", None),
            worker_repays_received=bool(getattr(node, "worker_repays_received", False)),
            memory_reserve_mb=float(getattr(node, "memory_reserve_mb", 0.0) or 0.0),
            max_weights_mb=getattr(node, "max_weights_mb", None),
            host_weights_mapped=bool(getattr(node, "host_weights_mapped", False)),
            ms_per_byte_decode=t.get("decode_per_byte"),
            ms_per_param_prefill=t.get("prefill_per_param"),
            profiled_on=",".join(sorted(geometry)),
        )

    for nid, link in sorted(links.items()):
        if link.get("warnings"):
            print("warning: %s's link measurement carries %s -- it is being used for the "
                  "communication term anyway" % (nid, "; ".join(link["warnings"])),
                  file=sys.stderr)
    tiers = {l.get("tier") for l in links.values() if l.get("tier")}
    if len(tiers) > 1:
        print("warning: link measurements span more than one interface tier (%s). The "
              "communication term assumes one fabric." % ", ".join(sorted(tiers)),
              file=sys.stderr)
    if dropped_rows:
        # A silent exclusion count is how a half-empty profile comes to look complete. This
        # number was computed by per_layer_times and then discarded by every caller.
        print("note: %d result row(s) excluded from per-layer times because their flags "
              "invalidate throughput (throttled / throttle-unknown)" % dropped_rows,
              file=sys.stderr)
    if missing and strict:
        raise ProfileError(
            "unmeasured inputs, so no placement can be computed:\n  " + "\n  ".join(missing) +
            "\n\nRun the Phase 2 characterisation sweep and the energy join first, or pass "
            "strict=False for a planning sketch (and label it as one).")
    return devices


def save_devices(devices: dict, path):
    """Persist the profile table - it is a deliverable in its own right."""
    out = {nid: {
        "node_id": d.node_id,
        "ms_per_layer_decode": d.ms_per_layer_decode,
        "ms_per_layer_prefill": d.ms_per_layer_prefill,
        "usable_mem_mb": d.usable_mem_mb,
        "power_active_w": d.power_active_w,
        "power_idle_w": d.power_idle_w,
        "link_mbps": d.link_mbps,
        "rtt_ms": d.rtt_ms,
        "is_straggler": d.is_straggler,
        "host_overhead_mb": d.host_overhead_mb,
        "worker_overhead_mb": d.worker_overhead_mb,
        "worker_repays_received": d.worker_repays_received,
        "memory_reserve_mb": d.memory_reserve_mb,
        "max_weights_mb": d.max_weights_mb,
        "host_weights_mapped": d.host_weights_mapped,
        "ms_per_byte_decode": d.ms_per_byte_decode,
        "ms_per_param_prefill": d.ms_per_param_prefill,
        "profiled_on": d.profiled_on,
    } for nid, d in devices.items()}
    Path(path).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def for_model(devices: dict, model) -> dict:
    """The profiles, priced for `model`: decode per byte a token reads (a mixture of experts' active experts), prefill
    per block parameter -- the testbed predictor's convention (testbed/scripts/predict_pool.py unit()). With the head
    the engine adds on the last node (placement._head_layers) this reproduces the profiled model's own ms/token
    exactly. A profile without unit costs (an older profile file) comes back unchanged."""
    bpl = model.active_bytes_per_layer or model.bytes_per_layer
    ppl = model.active_params_per_layer or model.params_per_layer
    out = {}
    for nid, d in devices.items():
        dec = d.ms_per_byte_decode * bpl if (d.ms_per_byte_decode and bpl) else d.ms_per_layer_decode
        pre = d.ms_per_param_prefill * ppl if (d.ms_per_param_prefill and ppl) else d.ms_per_layer_prefill
        out[nid] = replace(d, ms_per_layer_decode=dec, ms_per_layer_prefill=pre)
    return out


def load_devices(path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {nid: DeviceProfile(**spec) for nid, spec in raw.items()}
