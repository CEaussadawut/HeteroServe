#!/usr/bin/env python3
"""Which board holds which model file: a dated snapshot of every board's ~/models, for the planner's host choice.

    python3 testbed/scripts/model_inventory.py [--nodes pi5,pi4,jetson,laptop] [--out FILE]

Lists ~/models on each board over ssh (names and sizes; nothing is read or hashed: a runner checks a model's sha256 on
its host before any reported figure, testbed/models.md) and writes testbed/dataset/model_inventory_<stamp>.json:
    {"taken": ISO, "boards": {node: {"reachable": bool, "files": {name: bytes}}}, "holders": {model key: [nodes]}}
A board holds a model when the file predict_pool.MODELS names is there. A board that does not answer holds nothing. Why
(2026-10-08): the engine let only one board host each model (predict_pool.HOST, the Pi 5 by default), so TinyLlama was always
hosted on the Pi 4. Planned on the Jetson, which holds it, the plan measured 9.8 % faster and 6.7 % lower in energy.
"""
import datetime, json, os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                    # noqa: E402

SSH = ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]


def board_files(node):
    """{name: bytes} of ~/models on node, or None if the board does not answer."""
    try:
        cmd = ("cd ~/models 2>/dev/null && for f in *; do [ -f \"$f\" ] && printf '%s %s\\n' "
               "\"$(stat -L -c %s \"$f\")\" \"$f\"; done; true")
        # the laptop (a node since 2026-10-11) is listed here: it has no sshd, and its ~/models holds links into its cache
        r = subprocess.run((["bash", "-c", cmd] if node == "laptop" else SSH + [node, cmd]),
                           capture_output=True, text=True, timeout=30)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        return None
    out = {}
    for ln in r.stdout.splitlines():
        size, _, name = ln.partition(" ")
        if size.isdigit() and name:
            out[name] = int(size)
    return out


def holders(boards):
    """{model key: [nodes holding its file]} for every predict_pool.MODELS key."""
    res = {}
    for key, (_, _, path) in P.MODELS.items():
        name = os.path.basename(path)
        res[key] = sorted(n for n, b in boards.items() if b["reachable"] and name in b["files"])
    return res


def main(argv):
    # 2026-11-01: the laptop by default too (a node since 2026-10-11). Without it the newest inventory listed no laptop file,
    # and predict_pool.holders() would have dropped the laptop as a host of every model (caught before use; trashed).
    nodes = argv[argv.index("--nodes") + 1].split(",") if "--nodes" in argv else ["pi5", "pi4", "jetson", "laptop"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join(ROOT, "testbed", "dataset", "model_inventory_%s.json" % stamp)
    boards = {}
    for n in nodes:
        f = board_files(n)
        boards[n] = {"reachable": f is not None, "files": f or {}}
    res = {"taken": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "boards": boards, "holders": holders(boards)}
    json.dump(res, open(out, "w"), indent=1)
    for n, b in boards.items():
        print("%-7s %s" % (n, ("%d files" % len(b["files"])) if b["reachable"] else "NOT REACHABLE"))
    for k, v in res["holders"].items():
        print("  %-11s %s" % (k, ", ".join(v) or "-"))
    print("wrote %s" % os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main(sys.argv[1:])
