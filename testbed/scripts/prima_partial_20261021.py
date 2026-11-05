#!/usr/bin/env python3
"""Partial model copies for prima.cpp on the boards (user, 2026-10-21 ~20:45: "Partial copies").

prima.cpp loads the model from a local file on every rank, but a rank reads only the tensors of its own layers (src/llama.cpp
this_layer_is_mine: windows run from rank 1, the head rank 0 last in each cycle, which also holds token_embd and output). The
Pis have 9.1 and 13 GB free and Qwen3-32B Q8_0 is 34.8 GB, so each board gets a SPARSE file of the full size: the GGUF header
plus the byte ranges of the layers it may be given, holes elsewhere (zeros, no disk). The Jetson gets one too, sparing its
SD card (three storage hangs in October) a 35 GB write. The laptop, the head, keeps the full verified file.

This is an exception to the whole-file sha256 rule (testbed/models.md), so the check is range by range: every present range's
sha256 is computed here from the laptop's verified file and on the board from its copy, and must match; a manifest records
them. A rank that is given a layer outside its copy would compute with zeros, so qr3_tpot.py refuses to run when the windows
prima.cpp logs fall outside the manifest's layers, and the arm's greedy output is checked against llama.cpp's.

    python3 testbed/scripts/prima_partial_20261021.py plan   --model PATH [--windows w0,w1,w2,w3] [--margin 2]
    python3 testbed/scripts/prima_partial_20261021.py copy   --model PATH --node NODE --layers 0-13 [--dest ~/models/X]
    python3 testbed/scripts/prima_partial_20261021.py verify --manifest FILE --node NODE
    (copy with --layers none writes the header only: enough for prima.cpp's profiler and Halda, which read shapes, not data)

The ring is laptop (rank 0) -> pi5 (1) -> pi4 (2) -> jetson (3). Writes testbed/dataset/prima_copies_<stem>.json (one entry
per node, replaced on a later copy of that node). Raw output tee'd by the caller; this prints every range it writes.
"""
import argparse, datetime, hashlib, json, os, struct, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
from control.gguf import _Reader                          # noqa: E402
RING = ["laptop", "pi5", "pi4", "jetson"]
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8"]
CHUNK = 8 << 20


def table(path):
    """(data_start, file_size, {tensor: (start, end)}) with absolute byte ranges, from the GGUF header alone."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        assert r.raw(4) == b"GGUF", "not a GGUF file"
        version, n_t, n_kv = struct.unpack("<I", r.raw(4))[0], struct.unpack("<Q", r.raw(8))[0], struct.unpack("<Q", r.raw(8))[0]
        kv = {}
        for _ in range(n_kv):
            k = r.string(); kv[k] = r.value(struct.unpack("<I", r.raw(4))[0])
        infos = []
        for _ in range(n_t):
            name = r.string(); nd = struct.unpack("<I", r.raw(4))[0]
            r.raw(8 * nd); r.raw(4); infos.append((name, struct.unpack("<Q", r.raw(8))[0]))
        align = int(kv.get("general.alignment", 32))
        data_start = -(-fh.tell() // align) * align
    size = os.path.getsize(path)
    by = sorted(infos, key=lambda t: t[1])
    ends = [o for _, o in by[1:]] + [size - data_start]
    return data_start, size, {n: (data_start + o, data_start + e) for (n, o), e in zip(by, ends)}, kv


def layer_set(spec, n_layer):
    if spec in ("none", ""):
        return []
    out = set()
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    assert all(0 <= i < n_layer for i in out), "layer outside 0..%d" % (n_layer - 1)
    return sorted(out)


def ranges_for(tab, data_start, layers):
    """The header, then each layer's tensors, merged into contiguous ranges."""
    rs = [(0, data_start)]
    want = {"blk.%d." % i for i in layers}
    for n, (s, e) in sorted(tab.items(), key=lambda t: t[1][0]):
        if any(n.startswith(p) for p in want):
            rs.append((s, e))
    merged = []
    for s, e in sorted(rs):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def sha_ranges(path, rs):
    out = []
    with open(path, "rb") as f:
        for s, e in rs:
            h = hashlib.sha256(); f.seek(s); left = e - s
            while left:
                b = f.read(min(CHUNK, left)); h.update(b); left -= len(b)
            out.append(h.hexdigest())
    return out


def prima_layers(windows, n_layer):
    """The same mapping, written as prima.cpp's loop: for each layer, walk ranks 1, 2, ..., 0, 1, ... by window."""
    out = {r: [] for r in range(len(windows))}
    for lid in range(n_layer):
        cum, rank = 0, 1 % len(windows)
        while True:
            cum += windows[rank]
            if lid < cum:
                out[rank].append(lid); break
            rank = (rank + 1) % len(windows)
    return out


RECV = ("import sys, os, struct\n"
        "p, size = sys.argv[1], int(sys.argv[2])\n"
        "fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644); os.ftruncate(fd, size)\n"
        "inp = sys.stdin.buffer\n"
        "while True:\n"
        "    h = inp.read(16)\n"
        "    if len(h) < 16: break\n"
        "    off, n = struct.unpack('<QQ', h)\n"
        "    while n:\n"
        "        b = inp.read(min(n, 8 << 20))\n"
        "        if not b: raise SystemExit('short stream')\n"
        "        os.pwrite(fd, b, off); off += len(b); n -= len(b)\n"
        "os.fsync(fd); os.close(fd); print('received', p)\n")

SUMS = ("import sys, json, hashlib\n"
        "p, rs = sys.argv[1], json.loads(sys.argv[2]); out = []\n"
        "with open(p, 'rb') as f:\n"
        "    for s, e in rs:\n"
        "        h = hashlib.sha256(); f.seek(s); left = e - s\n"
        "        while left:\n"
        "            b = f.read(min(8 << 20, left)); h.update(b); left -= len(b)\n"
        "        out.append(h.hexdigest())\n"
        "import os; st = os.stat(p); print(json.dumps({'sha': out, 'size': st.st_size, 'blocks_mib': st.st_blocks * 512 >> 20}))\n")


def remote_py(node, code, args, stdin=None):
    cmd = "python3 -c %s %s" % (sh(code), " ".join(sh(a) for a in args))
    return subprocess.run(SSH + [node, cmd], input=stdin, capture_output=True, check=False)


def sh(s):
    return "'" + s.replace("'", "'\\''") + "'"


def manifest_path(model):
    return os.path.join(ROOT, "testbed", "dataset", "prima_copies_%s.json" % os.path.basename(model).rsplit(".", 1)[0])


def cmd_plan(a):
    data_start, size, tab, kv = table(a.model)
    arch = kv.get("general.architecture"); n_layer = int(kv["%s.block_count" % arch])
    print("model %s: %s, %d layers, %d tensors, header %d B, file %d B" % (a.model, arch, n_layer, len(tab), data_start, size))
    if a.windows:
        w = [int(x) for x in a.windows.split(",")]
        for r, ls in prima_layers(w, n_layer).items():
            lo = max(0, min(ls) - a.margin) if ls else None
            hi = min(n_layer - 1, max(ls) + a.margin) if ls else None
            rs = ranges_for(tab, data_start, list(range(lo, hi + 1))) if ls else []
            print("rank %d %-6s window %2d -> layers %s; with margin %d: %s-%s, %.2f GiB" % (
                r, RING[r] if r < len(RING) else "?", w[r], "%d-%d" % (min(ls), max(ls)) if ls else "none", a.margin, lo, hi,
                sum(e - s for s, e in rs) / 2 ** 30))


def cmd_copy(a):
    data_start, size, tab, kv = table(a.model)
    arch = kv.get("general.architecture"); n_layer = int(kv["%s.block_count" % arch])
    layers = layer_set(a.layers, n_layer)
    rs = ranges_for(tab, data_start, layers)
    want = sha_ranges(a.model, rs)
    dest = a.dest or "~/models/%s.prima-partial.gguf" % os.path.basename(a.model).rsplit(".", 1)[0]
    print("%s copy %s -> %s:%s, layers %s, %d ranges, %.2f GiB of %.2f" % (datetime.datetime.now().isoformat(timespec="seconds"),
          a.model, a.node, dest, a.layers, len(rs), sum(e - s for s, e in rs) / 2 ** 30, size / 2 ** 30), flush=True)
    rdest = subprocess.run(SSH + [a.node, "echo %s" % dest], capture_output=True, text=True, check=True).stdout.strip()
    tmp = rdest + ".part"
    p = subprocess.Popen(SSH + [a.node, "python3 -c %s %s %d" % (sh(RECV), sh(tmp), size)], stdin=subprocess.PIPE)
    with open(a.model, "rb") as f:
        for s, e in rs:
            p.stdin.write(struct.pack("<QQ", s, e - s)); f.seek(s); left = e - s
            while left:
                b = f.read(min(CHUNK, left)); p.stdin.write(b); left -= len(b)
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("FATAL: the copy to %s failed" % a.node)
    r = remote_py(a.node, SUMS, [tmp, json.dumps(rs)])
    got = json.loads(r.stdout.decode()) if r.returncode == 0 else None
    if not got or got["sha"] != want or got["size"] != size:
        raise SystemExit("FATAL: %s's copy does not match (%s)" % (a.node, r.stderr.decode()[-300:] if not got else "sha/size"))
    subprocess.run(SSH + [a.node, "mv %s %s" % (sh(tmp), sh(rdest))], check=True)
    m = json.load(open(manifest_path(a.model))) if os.path.exists(manifest_path(a.model)) else {
        "what": "prima.cpp partial copies (prima_partial_20261021.py): per node, the present byte ranges and their sha256, "
                "from the laptop's verified file", "source": os.path.realpath(a.model), "size": size, "data_start": data_start,
        "n_layer": n_layer, "nodes": {}}
    m["nodes"][a.node] = {"path": dest, "layers": layers, "ranges": rs, "sha256": want, "copied": datetime.datetime.now()
                          .astimezone().isoformat(timespec="seconds"), "disk_mib": got["blocks_mib"]}
    json.dump(m, open(manifest_path(a.model), "w"), indent=1)
    print("verified %s: %d ranges match; on disk %d MiB; manifest %s" % (a.node, len(rs), got["blocks_mib"],
          os.path.relpath(manifest_path(a.model), ROOT)))


def cmd_verify(a):
    m = json.load(open(a.manifest)); e = m["nodes"][a.node]
    path = subprocess.run(SSH + [a.node, "echo %s" % e["path"]], capture_output=True, text=True, check=True).stdout.strip()
    r = remote_py(a.node, SUMS, [path, json.dumps(e["ranges"])])
    got = json.loads(r.stdout.decode()) if r.returncode == 0 else None
    ok = bool(got) and got["sha"] == e["sha256"] and got["size"] == m["size"]
    print(json.dumps({"node": a.node, "path": e["path"], "ok": ok, "layers": "%s" % (e["layers"][:1] + e["layers"][-1:]),
                      "ranges": len(e["ranges"])}))
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--model", required=True); p.add_argument("--windows"); p.add_argument("--margin", type=int, default=2)
    c = sub.add_parser("copy"); c.add_argument("--model", required=True); c.add_argument("--node", required=True)
    c.add_argument("--layers", required=True); c.add_argument("--dest")
    v = sub.add_parser("verify"); v.add_argument("--manifest", required=True); v.add_argument("--node", required=True)
    a = ap.parse_args()
    {"plan": cmd_plan, "copy": cmd_copy, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
