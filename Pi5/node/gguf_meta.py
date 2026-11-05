#!/usr/bin/env python3
"""Read a GGUF header and report the model geometry the Phase 2 sweep needs.

Why this exists
---------------
The characterisation dataset's most important derived column is c_per_layer_ms -- the
c_i in Rule A of proposal Appendix B.6:

    L_i = round( L_total * (1/c_i) / sum_j (1/c_j) )   subject to mem_i <= M_i

c_i is TPOT divided by the model's layer count, and it is what makes a measurement on
this node comparable with a measurement on a node running a different model. It is the
whole reason the isolated sweep exists.

llama-bench does NOT emit a layer count. Its JSON field list (test::get_fields() in
tools/llama-bench/llama-bench.cpp) is fixed and contains model_size, model_n_params,
model_type and n_gpu_layers -- but no total block count. n_gpu_layers is the offload
count and is 0 on a CPU-only worker like this one, so it is not a substitute.

So the layer count is read here, from the GGUF header on the node at sweep time, and
written into cells.csv by scripts/10_isolated_bench.sh. Only the metadata block is
parsed; tensor data is never touched, so this is fast even on the 4.7 GB 7B rung.

This is deliberately a standalone copy of the same logic as the control node's
ThinkPad/control/gguf.py: a worker node folder is deployed on its own and cannot import
from the orchestrator. The GGUF key names are the contract between them -- if one is
changed, change both.

Format: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count, then kv pairs.

Usage:
    gguf_meta.py MODEL.gguf                 -> JSON on stdout
    gguf_meta.py MODEL.gguf --field n_layers -> a single bare value (for shell use)
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

# GGUF value type tags, in order.
(UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL,
 STRING, ARRAY, UINT64, INT64, FLOAT64) = range(13)

_SCALAR = {
    UINT8: ("<B", 1), INT8: ("<b", 1),
    UINT16: ("<H", 2), INT16: ("<h", 2),
    UINT32: ("<I", 4), INT32: ("<i", 4),
    FLOAT32: ("<f", 4), BOOL: ("<?", 1),
    UINT64: ("<Q", 8), INT64: ("<q", 8), FLOAT64: ("<d", 8),
}


class GGUFError(Exception):
    pass


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        data = self.fh.read(n)
        if len(data) != n:
            raise GGUFError("unexpected end of file")
        return data

    def scalar(self, tag: int):
        fmt, size = _SCALAR[tag]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        n = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(n).decode("utf-8", errors="replace")

    def value(self, tag: int):
        if tag in _SCALAR:
            return self.scalar(tag)
        if tag == STRING:
            return self.string()
        if tag == ARRAY:
            item_tag = struct.unpack("<I", self.raw(4))[0]
            count = struct.unpack("<Q", self.raw(8))[0]
            # Token vocabularies run to 128k strings and are of no interest here; read
            # them so the stream stays aligned, but do not keep them.
            keep = count <= 4096
            out = []
            for _ in range(count):
                v = self.value(item_tag)
                if keep:
                    out.append(v)
            return out if keep else "<array of %d>" % count
        raise GGUFError("unknown GGUF value type: %d" % tag)


def read_metadata(path) -> dict:
    path = Path(path)
    with path.open("rb") as fh:
        r = _Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise GGUFError("%s is not a GGUF file (magic=%r)" % (path.name, magic))
        version = struct.unpack("<I", r.raw(4))[0]
        tensor_count = struct.unpack("<Q", r.raw(8))[0]
        kv_count = struct.unpack("<Q", r.raw(8))[0]
        kv = {}
        for _ in range(kv_count):
            key = r.string()
            tag = struct.unpack("<I", r.raw(4))[0]
            kv[key] = r.value(tag)
    kv["_gguf_version"] = version
    kv["_tensor_count"] = tensor_count
    kv["_file_size"] = path.stat().st_size
    return kv


def shape(path) -> dict:
    """The subset the sweep and the placement rule use, normalised across architectures."""
    kv = read_metadata(path)
    arch = kv.get("general.architecture", "llama")

    def get(suffix, default=None):
        return kv.get("%s.%s" % (arch, suffix), default)

    n_layers = get("block_count")
    # Loud, not silent, and the test is "a positive integer" rather than "truthy". A
    # missing layer count means c_per_layer_ms cannot be computed and the row is useless
    # to the placement rule; a ZERO or NEGATIVE one is worse than missing, because it
    # parses. Rule A divides layers in proportion to 1/c_i, so zero gives a
    # ZeroDivisionError far from the cause (or an infinite share), and a negative count
    # inverts the allocation while every column still looks well-formed.
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers <= 0:
        raise GGUFError(
            "%s: %s.block_count is %r, expected a positive integer -- a layer count that "
            "is missing, zero or negative cannot produce a usable c_per_layer_ms"
            % (Path(path).name, arch, n_layers))

    n_embd = get("embedding_length")
    n_head = get("attention.head_count")
    n_head_kv = get("attention.head_count_kv", n_head)
    head_dim = get("attention.key_length")
    if not head_dim and n_embd and n_head:
        head_dim = n_embd // n_head

    return {
        "name": Path(path).stem,
        "architecture": arch,
        "n_layers": int(n_layers),
        "n_embd": int(n_embd) if n_embd else None,
        "n_head": int(n_head) if n_head else None,
        "n_head_kv": int(n_head_kv) if n_head_kv else None,
        "head_dim": int(head_dim) if head_dim else None,
        "context_train": get("context_length"),
        "file_size": kv["_file_size"],
        "quant": kv.get("general.file_type"),
    }


def main(argv) -> int:
    import json

    args = [a for a in argv[1:] if a != "--"]
    if not args:
        print(__doc__.strip().splitlines()[-3].strip(), file=sys.stderr)
        print("usage: gguf_meta.py MODEL.gguf [--field NAME]", file=sys.stderr)
        return 2

    field = None
    if "--field" in args:
        i = args.index("--field")
        try:
            field = args[i + 1]
        except IndexError:
            print("--field needs a name", file=sys.stderr)
            return 2
        del args[i:i + 2]

    rc = 0
    for path in args:
        try:
            info = shape(path)
        except (GGUFError, OSError) as exc:
            print("gguf_meta: %s" % exc, file=sys.stderr)
            rc = 1
            continue
        if field:
            value = info.get(field)
            if value is None:
                print("gguf_meta: no field %r for %s" % (field, path), file=sys.stderr)
                rc = 1
                continue
            print(value)
        else:
            print(json.dumps(info, indent=2, default=str))
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
