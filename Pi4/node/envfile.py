"""Minimal parser for config/node.env.

Deliberately dependency-free: the Pi 4 is the memory-tightest node in the cluster and
nothing here is worth a pip install or a venv. The same file is sourced by bash.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path=None) -> dict:
    path = Path(path) if path else ROOT / "config" / "node.env"
    env = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key.strip()] = value
    # A real shell environment wins, so a single run can be overridden without editing
    # the file (e.g. RPC_MEM_MB=1800 scripts/04_rpc_server.sh).
    env.update({k: v for k, v in os.environ.items() if k in env})
    return env
