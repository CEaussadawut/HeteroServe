"""Minimal GGUF metadata reader.

The placement rule needs a model's layer count, hidden size and KV-head geometry to work
out per-layer memory and activation sizes. Those numbers are already in the GGUF header,
so read them rather than typing them into a config file and hoping they were right - a
wrong n_layers silently produces a wrong memory constraint and a wrong per-layer time.

Only the header is parsed (key/value pairs, then the tensor-info table); tensor data is
never read, so this is fast even on a 9 GB file.

Format reference: magic "GGUF", u32 version, u64 tensor_count, u64 kv_count, then the
key/value pairs, then one info per tensor (name, u32 n_dims, u64 dims[n_dims], u32 ggml
type, u64 data offset), then the data section at the next multiple of general.alignment.
"""
from __future__ import annotations

import re
import struct
from pathlib import Path

# GGUF value type tags
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

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
            # Token vocabularies run to 128k strings and are of no interest here; read them
            # so the stream stays aligned, but do not keep them.
            keep = count <= 4096
            out = []
            for _ in range(count):
                v = self.value(item_tag)
                if keep:
                    out.append(v)
            return out if keep else "<array of %d>" % count
        raise GGUFError("unknown GGUF value type: %d" % tag)


def read_metadata(path, with_tensors: bool = False) -> dict:
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
        if with_tensors:
            kv["_tensors"] = _read_tensor_infos(r, fh, kv, tensor_count, path.stat().st_size)

    kv["_gguf_version"] = version
    kv["_tensor_count"] = tensor_count
    kv["_file_size"] = path.stat().st_size
    kv["_path"] = str(path)
    return kv


def _read_tensor_infos(r, fh, kv, count, file_size):
    """{tensor name: (n_elements, n_bytes)} from the tensor-info table, or None without one.

    A tensor's size is the distance from its data offset to the next tensor's (the last one
    runs to the end of the file), so no per-quant block layout is needed. The distance
    includes the alignment padding, under general.alignment bytes per tensor. A header-only
    file (the test fixtures) has no table, and anything that does not parse as one -- a name
    or rank out of range, an offset outside the data section -- returns None, never a
    half-read table.
    """
    infos = []
    try:
        for _ in range(count):
            n = struct.unpack("<Q", r.raw(8))[0]
            if not 0 < n < 4096:
                return None
            name = r.raw(n).decode("utf-8", errors="replace")
            n_dims = struct.unpack("<I", r.raw(4))[0]
            if not 0 < n_dims <= 8:
                return None
            dims = struct.unpack("<%dQ" % n_dims, r.raw(8 * n_dims))
            r.raw(4)                                    # ggml type: the offsets already give the size
            offset = struct.unpack("<Q", r.raw(8))[0]
            infos.append((name, dims, offset))
    except GGUFError:
        return None
    align = int(kv.get("general.alignment", 32))
    data_len = file_size - (-(-fh.tell() // align) * align)
    by_offset = sorted(infos, key=lambda t: t[2])
    ends = [t[2] for t in by_offset[1:]] + [data_len]
    table = {}
    for (name, dims, offset), end in zip(by_offset, ends):
        if not 0 <= offset <= end <= data_len:
            return None
        n_elements = 1
        for d in dims:
            n_elements *= d
        table[name] = (n_elements, end - offset)
    return table


def tensor_table(path):
    """{tensor name: (n_elements, n_bytes)} over every shard of the model, or None if any
    shard has no readable table."""
    kv = read_metadata(path, with_tensors=True)
    n_shards = kv.get("split.count") or kv.get("general.split.count")
    if not (n_shards and int(n_shards) > 1):
        return kv["_tensors"]
    table = {}
    for shard in _shard_paths(path, int(n_shards)):
        part = read_metadata(shard, with_tensors=True)["_tensors"]
        if part is None:
            return None
        table.update(part)
    return table


def _split_total_bytes(path, n_shards: int) -> int:
    """Total bytes across every shard of a split GGUF."""
    return sum(shard.stat().st_size for shard in _shard_paths(path, n_shards))


def _shard_paths(path, n_shards: int) -> list:
    """Every shard of a split GGUF, in order.

    llama.cpp names shards `<stem>-00001-of-000NN.gguf`. Every shard must be present: a
    partial set would silently under-count the model, which is the exact failure this
    guards against, so a missing shard is an error rather than a best effort.
    """
    p = Path(path)
    match = re.search(r"^(?P<stem>.+)-(?P<no>\d{5})-of-(?P<count>\d{5})$", p.stem)
    if not match:
        raise GGUFError(
            "%s: header says split.count=%d but the filename is not the "
            "<name>-00001-of-%05d.gguf form, so the other shards cannot be located"
            % (p.name, n_shards, n_shards))
    stem = match.group("stem")
    shards = []
    for i in range(1, n_shards + 1):
        shard = p.with_name("%s-%05d-of-%05d%s" % (stem, i, n_shards, p.suffix))
        if not shard.is_file():
            raise GGUFError(
                "%s: split model is missing shard %d of %d (%s). Planning from a partial "
                "set would under-count the model and can report an infeasible placement as "
                "feasible." % (p.name, i, n_shards, shard.name))
        shards.append(shard)
    return shards


def shape(path) -> dict:
    """The subset the placement rule actually uses, normalised across architectures."""
    kv = read_metadata(path)
    arch = kv.get("general.architecture", "llama")

    def get(suffix, default=None):
        return kv.get("%s.%s" % (arch, suffix), default)

    n_layers = get("block_count")
    n_embd = get("embedding_length")
    n_head = get("attention.head_count")
    n_head_kv = get("attention.head_count_kv", n_head)
    head_dim = get("attention.key_length")
    if not head_dim and n_embd and n_head:
        head_dim = n_embd // n_head

    if not n_layers:
        raise GGUFError("%s: no %s.block_count in the header" % (Path(path).name, arch))

    # A SPLIT model: `_file_size` is one shard, so dividing it by n_layers would halve a
    # 2-shard 14B's bytes_per_layer, double the computed memory_cap and can flip a node from
    # infeasible to feasible on the 14B memory-wall experiment - i.e. invert the stated
    # result. Sum the shards when they are all present; refuse when they are not, rather
    # than planning from half a model.
    file_size = kv["_file_size"]
    n_shards = kv.get("split.count") or kv.get("general.split.count")
    if n_shards and int(n_shards) > 1:
        file_size = _split_total_bytes(path, int(n_shards))

    # Weights per layer come from the tensor table: the sum of each blk.N.* layer's tensors.
    # The output head (output.weight + output_norm; the token embedding itself when the two
    # are tied) is reported apart, because one node holds it, not every layer -- see
    # placement.ModelProfile.head_bytes. file_size / n_layers, the older estimate, folds the
    # head and the embedding into every layer: it is 13.7% high for Qwen2.5-14B Q4_K_M (178.6
    # against 157.1 MiB), and with it the placement rule capped the Pi 5 at 40 layers where
    # 43 ran -- and ran fastest (2026-09-27, testbed/dataset/placement_validation.json). It
    # remains only as the fallback for a file with no readable table, and says so in
    # bytes_per_layer_source.
    table = tensor_table(path)
    layers = {}
    experts = {}          # a mixture-of-experts model's routed-expert tensors, per layer
    max_block_tensor = 0
    for name, (n_el, n_bytes) in (table or {}).items():
        m = re.match(r"blk\.(\d+)\.", name)
        if m:
            acc = layers.setdefault(int(m.group(1)), [0, 0])
            acc[0] += n_bytes
            acc[1] += n_el
            max_block_tensor = max(max_block_tensor, n_bytes)
            # Routed experts (ffn_{gate,up,down}_exps: every expert of the layer in one tensor). The router
            # (ffn_gate_inp) and any shared expert (ffn_*_shexp) run for every token, so they are not scaled.
            if re.match(r"blk\.\d+\.ffn_(gate|up|down)_exps\.", name):
                e = experts.setdefault(int(m.group(1)), [0, 0])
                e[0] += n_bytes
                e[1] += n_el
    by_table = sorted(layers) == list(range(int(n_layers)))
    if by_table:
        layer_bytes = [layers[i][0] for i in range(int(n_layers))]
        layer_params = [layers[i][1] for i in range(int(n_layers))]
        head = [n for n in table if n.startswith("output")]
        tied = "output.weight" not in table and "token_embd.weight" in table
        if tied:
            head.append("token_embd.weight")
        embd = table.get("token_embd.weight", (0, 0))
        bytes_per_layer = sum(layer_bytes) / float(n_layers)
        table_fields = {
            "bytes_per_layer_source": "tensor table",
            "layer_bytes": layer_bytes,
            "params_per_layer": sum(layer_params) / float(n_layers),
            "head_bytes": sum(table[n][1] for n in head),
            "head_params": sum(table[n][0] for n in head),
            "head_tied_to_embedding": tied,
            "embd_bytes": embd[1],
            "embd_params": embd[0],
            # The largest single tensor inside one block. ggml-rpc moves a tensor whole, so a
            # host's upload buffer, and a worker that pays what it receives a second time (the
            # Jetson's Vulkan worker), peak by the largest tensor sent -- this one when the head
            # does not travel (placement.ModelProfile.max_block_tensor_bytes).
            "max_block_tensor_bytes": max_block_tensor,
            "n_params": sum(t[0] for t in table.values()),
        }
        # Mixture of experts: a decoded token reads only expert_used_count of expert_count routed experts, so
        # its bytes read (and its compute) per layer are the layer minus the unused share of the expert tensors.
        # Memory still holds every expert: layer_bytes / bytes_per_layer are unchanged.
        n_expert, n_used = get("expert_count"), get("expert_used_count")
        if n_expert and n_used and experts:
            unused = 1.0 - float(n_used) / float(n_expert)
            active_bytes = [layer_bytes[i] - experts.get(i, (0, 0))[0] * unused for i in range(int(n_layers))]
            active_params = [layer_params[i] - experts.get(i, (0, 0))[1] * unused for i in range(int(n_layers))]
            table_fields.update({
                "expert_count": int(n_expert),
                "expert_used_count": int(n_used),
                "layer_active_bytes": active_bytes,
                "active_bytes_per_layer": sum(active_bytes) / float(n_layers),
                "active_params_per_layer": sum(active_params) / float(n_layers),
            })
    else:
        bytes_per_layer = file_size / float(n_layers)
        table_fields = {
            "bytes_per_layer_source": "file size / n_layers (approximation: no readable tensor "
                                      "table; folds the output head and embedding into every layer)",
            "head_bytes": 0.0,       # already folded into bytes_per_layer
        }

    # KV cache: 2 (K and V) * n_head_kv * head_dim * bytes_per_element, per layer per token.
    # f16 cache is llama.cpp's default; -ctk/-ctv q8_0 halves it and must be recorded if used.
    kv_bytes = 2 * (n_head_kv or 1) * (head_dim or 1) * 2

    return {
        "name": Path(path).stem,
        "architecture": arch,
        "n_layers": int(n_layers),
        "n_embd": int(n_embd) if n_embd else None,
        "n_head": int(n_head) if n_head else None,
        "n_head_kv": int(n_head_kv) if n_head_kv else None,
        "head_dim": int(head_dim) if head_dim else None,
        "context_train": get("context_length"),
        "file_size": file_size,
        "bytes_per_layer": bytes_per_layer,
        "kv_bytes_per_layer_per_token": kv_bytes,
        # One hidden state crosses each pipeline boundary per token, in f32 on the wire.
        "activation_bytes": (int(n_embd) * 4) if n_embd else 16384,
        "quant": kv.get("general.file_type"),
        "path": str(path),
        **table_fields,
    }


if __name__ == "__main__":
    import json
    import sys

    for arg in sys.argv[1:]:
        print(json.dumps(shape(arg), indent=2, default=str))
