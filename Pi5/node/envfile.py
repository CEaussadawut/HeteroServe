"""Parse config/node.env.

node.env is deliberately shell-sourceable so the bash scripts can just `. ` it;
this reader gives the Python services the same view without spawning a shell.
Supported: KEY=value, optional quotes, # comments, blank lines. Nothing else --
no command substitution, no variable expansion, by design.
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_PATHS = (
    Path("/etc/heteroserve/node.env"),
    Path(__file__).resolve().parent.parent / "config" / "node.env",
)


def load(path=None):
    candidates = [Path(path)] if path else list(DEFAULT_PATHS)
    for p in candidates:
        if p.is_file():
            return _parse(p)
    raise FileNotFoundError("no node.env found in %s" % [str(c) for c in candidates])


def _parse(p):
    env = {}
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            raise ValueError("%s:%d: not KEY=value: %r" % (p, lineno, raw))
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        env[key] = val
    return env


def get_int(env, key, default=None):
    if key not in env or env[key] == "":
        if default is None:
            raise KeyError("%s missing from node.env and has no default" % key)
        return default
    return int(env[key])


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(load(sys.argv[1] if len(sys.argv) > 1 else None), indent=2))
