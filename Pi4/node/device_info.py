#!/usr/bin/env python3
"""Device inventory for the Pi 4 node.

Scope item 1 of the proposal: "Record each node's characteristics: processor architecture,
supported vector instruction sets, usable memory, accelerator type, and network link speed."

Emits one JSON object. The control node's cluster manager consumes it; the placement rule
is computed from these fields together with the measured benchmark rows.

    python3 node/device_info.py            # pretty JSON to stdout
    python3 node/device_info.py --save     # also write $DATA_DIR/inventory/device_info.json
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import envfile  # noqa: E402

ENV = envfile.load()

# Cortex-A72 exposes these in /proc/cpuinfo "Features". What matters for llama.cpp on this
# board is what is ABSENT: asimddp (dotprod), which the Pi 5's A76 does have, and i8mm,
# which neither chip has (i8mm is ARMv8.6, Cortex-A710 and later; the A76 is ARMv8.2).
# Their absence is the main reason the Pi 4 is the designated straggler, and it has to be
# recorded rather than assumed, because it decides which ggml quant kernels are eligible.
VECTOR_FLAGS = [
    "asimd", "asimdhp", "asimddp", "asimdrdm", "asimdfhm",
    "i8mm", "bf16", "sve", "sve2", "crc32", "aes", "sha1", "sha2",
]


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def _read(path, default=""):
    try:
        return Path(path).read_text().strip()
    except Exception:
        return default


def decode_throttled(word):
    """Turn the Raspberry Pi throttled word into named booleans.

    A non-zero value means the board was under-volted or clock-capped, which silently
    changes the result of any benchmark running at the time. Storing the decoded form
    keeps the dataset self-explanatory months later.
    """
    try:
        value = int(word, 16)
    except Exception:
        return None
    bits = {
        0: "under_voltage_now",
        1: "arm_freq_capped_now",
        2: "throttled_now",
        3: "soft_temp_limit_now",
        16: "under_voltage_since_boot",
        17: "arm_freq_capped_since_boot",
        18: "throttled_since_boot",
        19: "soft_temp_limit_since_boot",
    }
    return {name: bool(value & (1 << bit)) for bit, name in bits.items()}


def cpu():
    info = _read("/proc/cpuinfo")
    feats = set()
    match = re.search(r"^Features\s*:\s*(.+)$", info, re.M)
    if match:
        feats = set(match.group(1).split())
    freqs = {}
    for name in ("cpuinfo_max_freq", "cpuinfo_min_freq", "scaling_cur_freq"):
        value = _read("/sys/devices/system/cpu/cpu0/cpufreq/" + name)
        if value.isdigit():
            freqs[name.replace("_freq", "_mhz")] = int(value) // 1000
    return {
        "model": ENV.get("NODE_CPU", "unknown"),
        "soc": ENV.get("NODE_SOC", ""),
        "board": _read("/proc/device-tree/model").rstrip("\x00"),
        "arch": platform.machine(),
        "cores": os.cpu_count(),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor", "unknown"),
        "freq": freqs,
        "vector_isa": {flag: (flag in feats) for flag in VECTOR_FLAGS},
        "all_features": sorted(feats),
        "kernel_eligibility": {
            "dotprod": "absent on Cortex-A72 - Q4_K_M/Q8_0 take the plain NEON path",
            "i8mm": "absent on Cortex-A72 - and on the Pi 5 A76 too (ARMv8.6 feature)",
            "fp16_arith": "asimdhp" in feats,
        },
    }


def memory():
    mem = {}
    for line in _read("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        try:
            mem[key] = int(value.split()[0]) // 1024  # MB
        except (ValueError, IndexError):
            continue
    return {
        "total_mb": mem.get("MemTotal", 0),
        "available_mb": mem.get("MemAvailable", 0),
        "usable_budget_mb": int(ENV.get("NODE_USABLE_MB", 0)),
        # Advertised only on a build that still has --mem (removed upstream 41386cf);
        # scripts/lib.sh rpc_extra_args() passes it when present and warns when not. On a
        # current build the server reports real free memory instead, so this records the
        # INTENDED headroom next to usable_budget_mb, which is what is really planned against.
        "rpc_advertised_mb": int(ENV.get("RPC_MEM_MB", 0)),
        "swap_total_mb": mem.get("SwapTotal", 0),
        # Single-channel LPDDR4-3200 on the BCM2711. Decode is memory-bandwidth-bound, so
        # this figure - not the core count - is what predicts this node's decode rate.
        "theoretical_bw_gbps": 6.4,
    }


def accelerator():
    # Stated explicitly rather than omitted: "no usable accelerator" is itself an input to
    # the placement rule (contrast the Jetson Nano's 128-core Maxwell GPU).
    return {
        "type": "none",
        "gpu": "VideoCore VI (no general-compute path used by llama.cpp)",
        "npu": None,
        "usable_for_inference": False,
    }


def network():
    out = {}
    for iface in (ENV.get("ETH_IFACE", "eth0"), ENV.get("WLAN_IFACE", "wlan0")):
        base = "/sys/class/net/" + iface
        if not Path(base).exists():
            continue
        speed = _read(base + "/speed", "")
        out[iface] = {
            "state": _read(base + "/operstate", "unknown"),
            "speed_mbps": int(speed) if speed.lstrip("-").isdigit() and int(speed) > 0 else None,
            "mac": _read(base + "/address"),
            "mtu": int(_read(base + "/mtu", "0") or 0),
            "addresses": re.findall(r"inet (\d+\.\d+\.\d+\.\d+)", _run(["ip", "-4", "addr", "show", iface])),
            "qdisc": _run(["tc", "qdisc", "show", "dev", iface]),
        }
    return out


def thermal():
    raw = _read("/sys/class/thermal/thermal_zone0/temp", "")
    word = _run(["vcgencmd", "get_throttled"]).replace("throttled=", "") or "unavailable"
    return {
        "soc_temp_c": round(int(raw) / 1000, 1) if raw.isdigit() else None,
        "throttled_word": word,
        "throttled_decoded": decode_throttled(word),
    }


def power_channel():
    return {
        "ina226_channel": int(ENV.get("INA_CHANNEL", 0)),
        "i2c_address": ENV.get("INA_I2C_ADDR"),
        "shunt_ohms": float(ENV.get("INA_SHUNT_OHMS", 0.01)),
        "rail": "5V USB-C input (high-side shunt)",
        "psu": ENV.get("PSU_MODEL"),
        "inside_measurement_boundary": ["SoC", "LPDDR4", "microSD", "GbE PHY", "case fan"],
        # The fan sits inside the boundary on purpose: the honest energy-per-token figure
        # for this node is what the node actually costs to run, cooling included.
        "caveat": "case fan is measured, not subtracted - report it as part of node power",
    }


def software():
    data_dir = Path(ENV.get("DATA_DIR", "/opt/heteroserve/data"))
    manifest = data_dir / "inventory" / "build_manifest.json"
    build = None
    if manifest.exists():
        try:
            build = json.loads(manifest.read_text())
        except json.JSONDecodeError:
            build = {"error": "build_manifest.json is not valid JSON"}
    pretty = ""
    for line in _read("/etc/os-release").splitlines():
        if line.startswith("PRETTY_NAME="):
            pretty = line.split("=", 1)[1].strip('"')
    return {
        "os": pretty,
        "kernel": platform.release(),
        "python": platform.python_version(),
        "llama_cpp_build": build,
        "llama_pin_sha": ENV.get("LLAMA_PIN_SHA") or None,
    }


def collect():
    data_dir = ENV.get("DATA_DIR", "/opt/heteroserve/data")
    free_mb = shutil.disk_usage(data_dir if Path(data_dir).exists() else "/").free // (1024 * 1024)
    return {
        "schema": "heteroserve.device_info/1",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": {
            "id": ENV.get("NODE_ID"),
            "role": ENV.get("NODE_ROLE"),
            "description": ENV.get("NODE_DESC"),
            "hostname": socket.gethostname(),
        },
        "cpu": cpu(),
        "memory": memory(),
        "accelerator": accelerator(),
        "network": network(),
        "thermal": thermal(),
        "power": power_channel(),
        "software": software(),
        "storage": {"data_volume_free_mb": free_mb},
    }


def main():
    data = collect()
    print(json.dumps(data, indent=2))
    if "--save" in sys.argv:
        out = Path(ENV.get("DATA_DIR", "/opt/heteroserve/data")) / "inventory" / "device_info.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, indent=2))
        print("saved -> " + str(out), file=sys.stderr)


if __name__ == "__main__":
    main()
