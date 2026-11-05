#!/usr/bin/env python3
"""Interactive serving's plan (2026-11-01, the scheduler review's items 4 and 6; the user: "do item 1 to 4 then 6-7").
serve_pool.sh imports this for a person at a terminal (SERVE_PLANNER v4); a harness keeps planning as before.

The review found that plans flipped with small memory changes: over 25.6-28.2 GB of the laptop's free memory the 32B Q6_K plan
changed 13 times, every 128-256 MiB, between plans 0-1.6 % apart, and five sessions ran five different plans. Serving planned
against a snapshot (memory_at_rest.json, 20 Oct) but gated on live memory, which is how spec4c was refused. So:
  live_memory  plan on each node's MemAvailable now (the lowest of a few reads, as memory_snapshot.py); a board that does not
               answer -- switched off -- gets its figure from the newest memory snapshot, and the gate checks it live once it
               is powered up
  on_grid      every figure rounded DOWN to a 512 MiB grid, so a plan changes only when memory crosses a grid line
  choose       the plan served last for the same model, context and boards is kept while it still fits (every board's margin
               at least the exact fill's headroom) and is predicted no more than KEEP_PCT slower than the new pick
  recall / remember   that last plan, in ~/.cache/heteroserve-wf/serve_plan_<model>.json, written once a serve is READY
The depth a plan is priced at is PLAN_DEPTH, by default 0 (plan_depth: registered as half the context, v4d found the laptop's
depth law 34.5 % high on the default model at 4,096 tokens and the plan it chose 4.0 % slower).
"""
import glob, json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [HERE, os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                    # noqa: E402

GRID_MIB = 512
KEEP_PCT = 2.0
STATE_DIR = os.path.expanduser("~/.cache/heteroserve-wf")
SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "LogLevel=ERROR"]


def newest_snapshot(ds=None):
    """The newest testbed/dataset/memory_snapshot_*.json, else memory_at_rest.json."""
    ds = ds or os.path.join(ROOT, "testbed", "dataset")
    fs = sorted(glob.glob(os.path.join(ds, "memory_snapshot_*.json")))
    return fs[-1] if fs else os.path.join(ds, "memory_at_rest.json")


def read_available(node, reads=3, run=subprocess.run):
    """The lowest of `reads` MemAvailable readings (MiB) 1 s apart on node, or None if it does not answer."""
    cmd = "for i in $(seq %d); do awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo; sleep 1; done" % reads
    try:
        p = run(["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd], capture_output=True, text=True,
                stdin=subprocess.DEVNULL, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    vals = [int(x) for x in p.stdout.split() if x.isdigit()]
    return min(vals) if p.returncode == 0 and vals else None


def live_memory(nodes, fallback, reads=3, read=read_available):
    """({node: {"mem_available_mib", ...}} as memory_at_rest.json, {node: "live" | "snapshot (no answer)"}): each node's
    figure now, or the fallback file's for a board that does not answer."""
    mem, src = {}, {}
    for n in nodes:
        v = read(n, reads)
        if v is None:
            if n not in fallback:
                raise ValueError("%s does not answer and %s has no figure for it" % (n, fallback.get("_source", "the snapshot")))
            mem[n], src[n] = dict(fallback[n]), "snapshot (no answer)"
        else:
            mem[n], src[n] = dict(fallback.get(n, {}), mem_available_mib=v), "live"
    return mem, src


def on_grid(mem, grid=GRID_MIB):
    """The same figures rounded down to a multiple of grid MiB (the raw figure kept as mem_available_raw_mib)."""
    out = {}
    for n, v in mem.items():
        if not isinstance(v, dict) or "mem_available_mib" not in v:
            out[n] = v
            continue
        raw = v.get("mem_available_raw_mib", v["mem_available_mib"])
        out[n] = dict(v, mem_available_raw_mib=raw, mem_available_mib=int(raw // grid * grid))
    return out


def state_path(key):
    return os.path.join(STATE_DIR, "serve_plan_%s.json" % key)


def recall(key, ctx, nodes, path=None):
    """The plan last served for this model, context and set of boards (remember()), or None."""
    p = path or state_path(key)
    try:
        r = json.load(open(p))
    except (OSError, ValueError):
        return None
    if r.get("key") != key or r.get("ctx") != ctx or sorted(r.get("nodes", [])) != sorted(nodes):
        return None
    return r


def remember(rec, path=None):
    p = path or state_path(rec["key"])
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + ".tmp", "w") as f:
        json.dump(dict(rec, served=time.strftime("%Y-%m-%dT%H:%M:%S%z")), f, indent=1)
    os.replace(p + ".tmp", p)


def record(key, ctx, nodes, depth, r):
    """What remember() keeps of a plan: enough to price it again (plan_fixed) and to compare it."""
    return {"key": key, "ctx": ctx, "nodes": list(nodes), "depth": depth, "assign": [list(x) for x in r["assign"]],
            "head": "host" if r["head_at"] == 0 else "last", "decode_ms": round(r["decode_ms"], 1)}


def fits(r, mem, headroom=None):
    """Every board of plan r keeps at least the exact fill's headroom (placement.EXACT_FILL_HEADROOM_MB) beyond its need and
    reserve, the same budget plan_optimal fills to."""
    from control import placement as PL
    h = PL.EXACT_FILL_HEADROOM_MB if headroom is None else headroom
    m = P.memory(r["assign"], r["q"], mem, r["devices"], r["model"], head_at=r["head_at"])
    return all(v["margin_mib"] >= h for v in m.values()), m


def choose(key, nodes, ctx, mem, depth, last=None, keep_pct=KEEP_PCT, host=None, **planner):
    """(plan, why): plan_host_choice's pick at this depth, or the last served plan when it still fits and is predicted no
    more than keep_pct slower (why says which, with both figures)."""
    new = P.plan_host_choice(key, nodes, ctx, mem, host=host, load_mode="none", depth=depth, **planner)
    if not last:
        return new, "planned (no plan served before for %s at ctx %d on these boards)" % (key, ctx)
    if [list(x) for x in new["assign"]] == last["assign"] and ("host" if new["head_at"] == 0 else "last") == last["head"]:
        return new, "planned: the same plan as served last (%s)" % last.get("served", "?")
    if host and last["assign"][0][0] != host:
        return new, "planned: the last plan's host is not SERVE_HOST"
    try:
        old = P.plan_fixed(key, [tuple(x) for x in last["assign"]], last["head"], ctx, mem, load_mode="none", depth=depth,
                           **planner)
    except (ValueError, KeyError) as e:
        return new, "planned: the last plan cannot be priced now (%s)" % str(e)[:120]
    ok, m = fits(old, mem)
    if not ok:
        return new, "planned: the last plan no longer fits (margins %s)" % {n: v["margin_mib"] for n, v in m.items()}
    if old["decode_ms"] <= new["decode_ms"] * (1 + keep_pct / 100.0):
        old["considered"] = new["considered"] + [("kept", "the last plan, %.1f ms against the new pick's %.1f" % (
            old["decode_ms"], new["decode_ms"]))]
        return old, "kept the plan served last (%s): %.1f ms a token against the new pick's %.1f (%s), within %.0f %%" % (
            last.get("served", "?"), old["decode_ms"], new["decode_ms"],
            " + ".join("%s %d" % tuple(x) for x in new["assign"]), keep_pct)
    return new, "planned: the new pick is %.1f %% faster than the last plan (%.1f against %.1f ms)" % (
        100 * (1 - new["decode_ms"] / old["decode_ms"]), new["decode_ms"], old["decode_ms"])


def plan_depth(ctx, env=os.environ):
    """PLAN_DEPTH if given, else 0: depth is priced only when asked. Registered as half the context (v4d), but on serve_pool.sh's
    default model at 4,096 tokens the laptop's query-head law over-priced its attention by 34.5 % (70.5 ms per 1,000 tokens
    measured against 107.5), and the plan it chose was 4.0 % slower than the plan without the term (v4d: D4-LONG, D4-ACC and
    D4-LAW fail; testbed/results/v4d_20261101_071003). At 8,192 tokens lc4planb's depth-priced plan was 1.81x faster, so a long
    session may still ask for it (PLAN_DEPTH=8192); ctx is kept for that use."""
    v = env.get("PLAN_DEPTH")
    return int(v) if v not in (None, "") else 0
