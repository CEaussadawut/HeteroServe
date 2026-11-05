#!/usr/bin/env python3
"""Serve with the idle Pi 4 switched off at its outlet: the switch-off automation, an opt-in (2026-10-24; the user: "continue
what's left for project 2"). so4 (testbed/results/so4_20261024_151550) measured that cutting the outlet of a board the plan
leaves out saves 3.41 J per token (5.7 %) at unchanged speed, that a clean shutdown alone saves little (the halted Pi 4 still
drew 1.7-1.9 W), and that the board takes about a minute to come back. This wraps serve_pool.sh, which it does not change:

    python3 testbed/scripts/serve_switch_20261024.py start  [--dry-run] [--assume-pi4 on|off]
    python3 testbed/scripts/serve_switch_20261024.py stop   [--dry-run] [--assume-pi4 on|off] [--power-on-after]
    python3 testbed/scripts/serve_switch_20261024.py status

with serve_pool.sh's own environment (MODEL, CTX, MEMFILE, NODES, PLACE, PLACE_HEAD, SERVE_HOST, SERVE_DRAFT, ...).
start  1. plans as serve_pool.sh plans (the same predict_pool call on the same environment), to know whether the Pi 4 is in
          the plan; refuses when serve_pool.sh's state exists (something is served already, a measurement perhaps);
       2. the Pi 4 in the plan and its outlet off: strip_switch_20261024.py on-node pi4 (about a minute; a failure stops here);
       3. serve_pool.sh start (its exit code is this tool's);
       4. reads what was served (serve_pool.sh's state: HOST, WORKERS). The Pi 4 not in it, its outlet on, and no llama.cpp
          process on it: strip_switch_20261024.py off-node pi4 (a clean poweroff, 25 s of silence, then the outlet).
stop   serve_pool.sh stop. The Pi 4 is left as it is; --power-on-after switches it back on (so the next plan reads it live).
status serve_pool.sh status, and the Pi 4's outlet.
Only the Pi 4 is ever switched: this tool names no other board, and strip_switch_20261024.py refuses any other board and the
laptop's outlet (strip.json's do_not_switch_off) on its own.
--dry-run prints the plan and every step, and runs none of them; it reads the outlet once (strip_switch_20261024.py status),
unless --assume-pi4 gives its state, and then contacts nothing at all.
NOT RUN LIVE: written and dry-run tested on 2026-10-24 (testbed/tests/test_serve_switch.py). Switching the outlet from this tool
waits for the user's yes; until then the outlet is switched only as so4 did, by hand, with strip_switch_20261024.py.
"""
import argparse, json, os, shlex, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
STATE = os.path.expanduser("~/.cache/heteroserve-wf/serve_pool.state")
BOARD = "pi4"                       # the one board this tool switches (strip_switch_20261024.ALLOWED)
SWITCH = [sys.executable, os.path.join(HERE, "strip_switch_20261024.py")]
SERVE = ["bash", os.path.join(HERE, "serve_pool.sh")]
LOG = os.path.join(ROOT, "testbed", "results", "serve_switch.jsonl")


def planned_members(env):
    """The boards serve_pool.sh's start would plan on, from the same call on the same environment (serve_pool.sh's PLANNED)."""
    import predict_pool as P
    key, ctx = env.get("MODEL") or "qwen3moe", int(env.get("CTX") or 4096)
    ds = os.path.join(ROOT, "testbed", "dataset")
    memfile = env.get("MEMFILE") or os.path.join(ds, "memory_at_rest.json")
    mem = json.load(open(memfile if os.path.isabs(memfile) else os.path.join(ROOT, memfile)))
    place = env.get("PLACE")
    if place:
        r = P.plan_fixed(key, [(n, int(k)) for n, k in (x.split(":") for x in place.split(","))],
                         env.get("PLACE_HEAD") or "last", ctx, mem, load_mode="none")
    else:
        # NODES unset: serve_pool.sh's own default (2026-11-01, serve_defaults.sh default_nodes): the laptop too for a person
        # at a terminal, as serve_pool.sh then is (it inherits this tool's stdout)
        nodes = env.get("NODES") or ("laptop pi5 pi4 jetson" if sys.stdout.isatty() else "pi5 pi4 jetson")
        r = P.plan_host_choice(key, nodes.split(), ctx, mem, host=env.get("SERVE_HOST") or None, load_mode="none")
    return [n for n, _ in r["assign"]], r


def served_members(path=STATE):
    """HOST and WORKERS from serve_pool.sh's state file (printf %q values), or None when nothing is served."""
    if not os.path.exists(path):
        return None
    kv = {}
    with open(path) as f:
        for line in f:
            k, _, v = line.rstrip("\n").partition("=")
            if k in ("HOST", "WORKERS"):    # only these: ARGS and the rest carry quoting this tool does not need to read
                kv[k] = " ".join(shlex.split(v)) if v else ""
    if not kv.get("HOST"):
        raise SystemExit("cannot read HOST from %s" % path)
    return [kv["HOST"]] + kv.get("WORKERS", "").split()


def before_start(planned, outlet_on):
    """The steps before serve_pool.sh start: the board powered up only when the plan needs it and its outlet is off."""
    return [("on-node", BOARD)] if BOARD in planned and outlet_on is False else []


def after_start(served, outlet_on, busy):
    """The steps after a successful start: the board switched off only when the served plan leaves it out, its outlet is on,
    and nothing of llama.cpp runs on it."""
    if served is None or BOARD in served or outlet_on is False:
        return []
    if busy:
        return [("leave-on", "%s runs %s" % (BOARD, busy))]
    return [("off-node", BOARD)]


def outlet_on(assume=None):
    if assume:
        return assume == "on"
    p = subprocess.run(SWITCH + ["status", BOARD], capture_output=True, text=True, timeout=60)
    try:                            # strip_switch_20261024.py status prints {"output", "apower", "source", "outlet", "board"}
        d = json.loads(p.stdout.strip().splitlines()[-1])
        if d.get("board") != BOARD or not isinstance(d.get("output"), bool):
            raise ValueError(d)
        return d["output"]
    except (ValueError, IndexError, KeyError, TypeError, AttributeError):
        raise SystemExit("cannot read the %s outlet: %s" % (BOARD, (p.stdout + p.stderr).strip()[-300:]))


def busy_on_board():
    p = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "LogLevel=ERROR", BOARD,
                        "for p in llama-bench llama-cli llama-server ggml-rpc-server; do pgrep -x $p >/dev/null && printf '%s ' $p; done; true"],
                       capture_output=True, text=True, timeout=30)
    return p.stdout.strip() if p.returncode == 0 else "(ssh exit %d)" % p.returncode


def step(kind, arg, dry, log=LOG):
    print("%s %s %s" % ("DRY-RUN would run:" if dry else "running:", kind, arg), flush=True)
    if dry:
        return 0
    if kind in ("on-node", "off-node"):
        return subprocess.run(SWITCH + [kind, arg, "--log", log]).returncode
    if kind == "serve":
        return subprocess.run(SERVE + [arg], cwd=ROOT).returncode
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("start", "stop", "status"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--assume-pi4", choices=("on", "off"))
    ap.add_argument("--power-on-after", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ)
    if a.cmd == "status":
        subprocess.run(SERVE + ["status"], cwd=ROOT)
        print("%s outlet: %s" % (BOARD, "on" if outlet_on(a.assume_pi4) else "off"))
        return 0
    if a.cmd == "stop":
        rc = step("serve", "stop", a.dry_run)
        if a.power_on_after and not outlet_on(a.assume_pi4):
            rc = step("on-node", BOARD, a.dry_run) or rc
        return rc
    if served_members() is not None:
        raise SystemExit("refused: serve_pool.sh state exists (%s): something is served already" % STATE)
    planned, r = planned_members(env)
    on = outlet_on(a.assume_pi4)
    print("plan: %s (head %s), predicted %.1f ms a token; %s outlet %s" % (
        " + ".join("%s %d" % x for x in r["assign"]), "on the host" if r["head_at"] == 0 else "last", r["decode_ms"],
        BOARD, "on" if on else "off"))
    for kind, arg in before_start(planned, on):
        if step(kind, arg, a.dry_run):
            raise SystemExit("stopped: %s %s failed; nothing was started" % (kind, arg))
        on = True
    rc = step("serve", "start", a.dry_run)
    if rc:
        return rc
    served = planned if a.dry_run else served_members()
    if a.dry_run:
        print("(dry run: the served plan is taken to be the planned one, and %s idle)" % BOARD)
    elif served != planned:
        print("note: serve_pool.sh served %s, not the %s planned here; the switching follows what was served" % (served, planned))
    busy = "" if a.dry_run or served is None or BOARD in served or not on else busy_on_board()
    for kind, arg in after_start(served, on, busy):
        if kind == "leave-on":
            print("left on: %s" % arg)
        elif step(kind, arg, a.dry_run):
            print("warning: %s %s failed; the model is served, the board's outlet is as the log says" % (kind, arg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
