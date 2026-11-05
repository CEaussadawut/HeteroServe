"""Tests for node/gguf_meta.py -- the only source of the layer count.

llama-bench does not report how many layers a model has, so `c_per_layer_ms` (the c_i in
Rule A) exists only because this module can read `<arch>.block_count` out of a GGUF
header. If it returns a wrong number the dataset is wrong in a way nothing downstream can
detect; if it returns nothing the sweep must stop rather than record a row Rule A cannot
use. Both behaviours are pinned here.

The fixtures are synthetic GGUF headers assembled byte by byte. That is deliberate: the
real models are 0.4-4.7 GB and cannot be checked in, and a mock of the parser would test
nothing. These exercise the actual struct-unpacking path against the actual format.
"""
from __future__ import annotations

import importlib.util
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GGUF_META = ROOT / "node" / "gguf_meta.py"


def _load():
    spec = importlib.util.spec_from_file_location("gguf_meta", GGUF_META)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gm = _load()

UINT32, FLOAT32, STRING, ARRAY, UINT64 = 4, 6, 8, 9, 10


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, tag: int, payload: bytes) -> bytes:
    return _str(key) + struct.pack("<I", tag) + payload


def u32(key, value):
    return _kv(key, UINT32, struct.pack("<I", value))


def string(key, value):
    return _kv(key, STRING, _str(value))


def str_array(key, items):
    """A token vocabulary: the thing that makes naive GGUF parsers lose stream alignment."""
    payload = struct.pack("<I", STRING) + struct.pack("<Q", len(items))
    payload += b"".join(_str(i) for i in items)
    return _kv(key, ARRAY, payload)


# The full tag table, mirroring gguf_meta's own _SCALAR map. Every one of these appears in
# a real GGUF header BEFORE the architecture keys -- UINT64 (split.tensors.count), FLOAT32
# (rope.freq_base, layer_norm_rms_epsilon), BOOL (add_bos_token), INT32/FLOAT32 arrays
# (tokenizer.ggml.token_type, .scores) -- so a wrong width for any of them advances the file
# pointer wrongly and every subsequent key, block_count included, is read from garbage.
SCALAR_TAGS = {
    "UINT8": (0, "<B", 200), "INT8": (1, "<b", -42),
    "UINT16": (2, "<H", 65000), "INT16": (3, "<h", -3000),
    "UINT32": (4, "<I", 4000000000), "INT32": (5, "<i", -123456),
    "FLOAT32": (6, "<f", 0.5), "BOOL": (7, "<?", True),
    "UINT64": (10, "<Q", 2**40), "INT64": (11, "<q", -(2**40)),
    "FLOAT64": (12, "<d", 1.5),
}


def scalar(key, name):
    tag, fmt, value = SCALAR_TAGS[name]
    return _kv(key, tag, struct.pack(fmt, value)), value


def typed_array(key, item_name, items):
    tag, fmt, _ = SCALAR_TAGS[item_name]
    payload = struct.pack("<I", tag) + struct.pack("<Q", len(items))
    payload += b"".join(struct.pack(fmt, v) for v in items)
    return _kv(key, ARRAY, payload)


def build_gguf(path, kvs, *, magic=b"GGUF", version=3, tensor_count=291):
    body = b"".join(kvs)
    head = magic + struct.pack("<I", version) + struct.pack("<Q", tensor_count) \
        + struct.pack("<Q", len(kvs))
    Path(path).write_bytes(head + body)
    return Path(path)


def llama_like(arch="llama", *, block_count=22, embedding_length=2048,
               head_count=32, head_count_kv=4, extra=()):
    kvs = [string("general.architecture", arch),
           string("general.name", "test-model"),
           u32("general.file_type", 15)]
    if block_count is not None:
        kvs.append(u32("%s.block_count" % arch, block_count))
    kvs += [u32("%s.embedding_length" % arch, embedding_length),
            u32("%s.attention.head_count" % arch, head_count),
            u32("%s.attention.head_count_kv" % arch, head_count_kv),
            u32("%s.context_length" % arch, 2048)]
    kvs += list(extra)
    return kvs


class GGUFCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, kvs, name="m.gguf", **kw):
        return build_gguf(self.tmp / name, kvs, **kw)


class TestLayerCount(GGUFCase):
    def test_reads_block_count(self):
        """TinyLlama-1.1B has 22 layers -- the value this was validated against."""
        p = self.write(llama_like("llama", block_count=22))
        self.assertEqual(gm.shape(p)["n_layers"], 22)

    def test_key_is_architecture_relative(self):
        """The key is `<arch>.block_count`, so a Qwen model's count lives under `qwen2.`
        and a parser hard-coded to `llama.` would read nothing. Qwen2.5-0.5B has 24."""
        p = self.write(llama_like("qwen2", block_count=24))
        info = gm.shape(p)
        self.assertEqual(info["architecture"], "qwen2")
        self.assertEqual(info["n_layers"], 24)

    def test_missing_block_count_raises(self):
        """Loud, not silent. A missing layer count must stop the sweep -- a row without
        c_per_layer_ms is invisible to Rule A while still looking like data."""
        p = self.write(llama_like(block_count=None))
        with self.assertRaises(gm.GGUFError) as cm:
            gm.shape(p)
        self.assertIn("block_count", str(cm.exception))

    def test_layer_count_is_an_int_not_a_string(self):
        p = self.write(llama_like(block_count=32))
        self.assertIsInstance(gm.shape(p)["n_layers"], int)


class TestEveryValueTypeKeepsTheStreamAligned(GGUFCase):
    """Only UINT32 was ever decoded by any fixture, so twelve of the thirteen width entries
    in gguf_meta's _SCALAR table had no witness at all: eight separate width mutations
    survived the whole suite. A wrong width does not raise -- it silently advances the file
    pointer by the wrong number of bytes, and every key after it, including block_count, is
    read from the wrong offset. That yields a plausible wrong layer count, which becomes a
    plausible wrong c_i, which Rule A divides layers by."""

    def test_each_scalar_type_decodes_and_leaves_block_count_readable(self):
        for name in SCALAR_TAGS:
            with self.subTest(type=name):
                kv, expected = scalar("test.value", name)
                kvs = [string("general.architecture", "llama"), kv,
                       u32("llama.block_count", 22),
                       u32("llama.embedding_length", 2048),
                       u32("llama.attention.head_count", 32)]
                meta = gm.read_metadata(self.write(kvs, name="%s.gguf" % name))
                got = meta["test.value"]
                if name.startswith("FLOAT"):
                    self.assertAlmostEqual(got, expected, places=5)
                else:
                    self.assertEqual(got, expected)
                # the real assertion: the key AFTER it is still readable
                self.assertEqual(meta["llama.block_count"], 22,
                                 "a %s value desynchronised the stream" % name)

    def test_each_array_element_type_keeps_the_stream_aligned(self):
        """Array dispatch was only ever exercised for STRING. token_type is an INT32 array
        and scores is a FLOAT32 array in every real tokenizer block."""
        for name in ("UINT8", "INT32", "FLOAT32", "UINT64", "BOOL"):
            with self.subTest(item=name):
                kvs = [string("general.architecture", "llama"),
                       typed_array("test.arr", name, [SCALAR_TAGS[name][2]] * 5),
                       u32("llama.block_count", 22)]
                meta = gm.read_metadata(self.write(kvs, name="arr_%s.gguf" % name))
                self.assertEqual(meta["llama.block_count"], 22,
                                 "a %s array desynchronised the stream" % name)

    def test_a_bool_decodes_as_a_bool_not_as_an_integer(self):
        """BOOL is `("<?", 1)`; `("<B", 1)` is the same WIDTH, so every alignment check in
        this file passes with it and the stream never desynchronises. The difference is the
        decoded TYPE -- and `1 == True` in Python, so an equality assertion cannot see it
        either. Only an identity/type check can. It reaches the JSON this tool prints, where
        `true` becomes `1` and any consumer testing `is True` silently stops matching."""
        kvs = [string("general.architecture", "llama"),
               scalar("tokenizer.ggml.add_bos_token", "BOOL")[0],
               u32("llama.block_count", 22)]
        meta = gm.read_metadata(self.write(kvs, name="booltype.gguf"))
        v = meta["tokenizer.ggml.add_bos_token"]
        self.assertIsInstance(v, bool, "a GGUF BOOL decoded as %r (%s), not a bool"
                              % (v, type(v).__name__))

    def test_a_realistic_header_ordering_still_finds_the_layer_count(self):
        """The shape a real GGUF actually has: mixed scalar types and a large token
        vocabulary BEFORE the architecture keys."""
        kvs = [string("general.architecture", "llama"),
               scalar("split.tensors.count", "UINT64")[0],
               scalar("llama.rope.freq_base", "FLOAT32")[0],
               scalar("tokenizer.ggml.add_bos_token", "BOOL")[0],
               typed_array("tokenizer.ggml.token_type", "INT32", [1] * 3000),
               str_array("tokenizer.ggml.tokens", ["t%d" % i for i in range(3000)]),
               typed_array("tokenizer.ggml.scores", "FLOAT32", [0.5] * 3000),
               u32("llama.block_count", 22),
               u32("llama.embedding_length", 2048),
               u32("llama.attention.head_count", 32),
               u32("llama.attention.head_count_kv", 4)]
        info = gm.shape(self.write(kvs, name="real.gguf"))
        self.assertEqual(info["n_layers"], 22)
        self.assertEqual(info["n_head"], 32)
        self.assertEqual(info["n_head_kv"], 4)


class TestTruncationIsDetected(GGUFCase):
    """The short-read guard in `_Reader.raw` is the only truncation check in the node folder,
    and no test reached it: the empty-file test trips the magic check and the half-file test
    trips the missing-block_count check, both from different sites. Replacing the guard with
    zero-padding survived the whole suite -- and `03_fetch_models.sh` skips re-download on
    `[ -f ]` and records the sha of whatever bytes are on disk, so a short GGUF is not caught
    upstream either."""

    def _full(self):
        return [string("general.architecture", "llama"),
                u32("llama.block_count", 22),
                u32("llama.embedding_length", 2048),
                u32("llama.attention.head_count", 32),
                u32("llama.context_length", 2048)]

    def test_a_header_cut_mid_value_raises(self):
        p = self.write(self._full(), name="cut.gguf")
        data = p.read_bytes()
        p.write_bytes(data[:-2])          # lands inside the final u32's payload
        with self.assertRaises(gm.GGUFError):
            gm.read_metadata(p)

    def test_a_header_cut_mid_key_string_raises(self):
        p = self.write(self._full(), name="cutkey.gguf")
        data = p.read_bytes()
        p.write_bytes(data[:len(data) - 14])
        with self.assertRaises(gm.GGUFError):
            gm.read_metadata(p)

    def test_truncation_is_never_silently_padded(self):
        """Walk every cut point after the header preamble and require each to raise rather
        than return a value. A zero-padding mutant returns n_layers=22 from a file that has
        been cut in half."""
        p = self.write(self._full(), name="walk.gguf")
        data = p.read_bytes()
        for cut in range(24, len(data), 3):
            with self.subTest(cut=cut):
                p.write_bytes(data[:cut])
                try:
                    gm.read_metadata(p)
                except gm.GGUFError:
                    continue
                self.fail("a file truncated at %d bytes parsed without error" % cut)


class TestStreamAlignment(GGUFCase):
    """Every real GGUF carries a token vocabulary of 32k-152k strings before or between
    the keys that matter. The parser reads them to stay aligned and then discards them.
    If that walk is wrong by a single byte, every subsequent key is garbage -- including
    block_count. This is the failure mode most likely to produce a plausible wrong
    number rather than an error."""

    def test_keys_after_a_large_vocab_still_parse(self):
        vocab = str_array("tokenizer.ggml.tokens", ["tok%d" % i for i in range(9000)])
        kvs = [string("general.architecture", "llama"), vocab,
               u32("llama.block_count", 22), u32("llama.embedding_length", 2048),
               u32("llama.attention.head_count", 32)]
        info = gm.shape(self.write(kvs))
        self.assertEqual(info["n_layers"], 22)
        self.assertEqual(info["n_embd"], 2048)

    def test_large_arrays_are_not_retained(self):
        """A 9000-entry vocabulary is read for alignment only; holding it would make the
        sweep's per-model probe cost scale with vocabulary size for no reason."""
        vocab = str_array("tokenizer.ggml.tokens", ["tok%d" % i for i in range(9000)])
        kvs = [string("general.architecture", "llama"), vocab,
               u32("llama.block_count", 22)]
        kv = gm.read_metadata(self.write(kvs))
        self.assertIsInstance(kv["tokenizer.ggml.tokens"], str)
        self.assertIn("9000", kv["tokenizer.ggml.tokens"])

    def test_small_arrays_are_kept(self):
        kvs = [string("general.architecture", "llama"), u32("llama.block_count", 22),
               str_array("tokenizer.ggml.merges", ["a b", "c d"])]
        kv = gm.read_metadata(self.write(kvs))
        self.assertEqual(kv["tokenizer.ggml.merges"], ["a b", "c d"])


class TestGeometry(GGUFCase):
    def test_head_dim_is_derived_when_absent(self):
        p = self.write(llama_like(embedding_length=2048, head_count=32))
        self.assertEqual(gm.shape(p)["head_dim"], 64)

    def test_explicit_key_length_wins(self):
        kvs = llama_like(embedding_length=2048, head_count=32,
                         extra=[u32("llama.attention.key_length", 128)])
        self.assertEqual(gm.shape(self.write(kvs))["head_dim"], 128)

    def test_head_count_kv_defaults_to_head_count(self):
        """Multi-head attention, where there is no separate KV head count. Getting this
        wrong misstates the KV-cache size, which is what the memory bound in Rule A
        (mem_i <= M_i) is checked against."""
        kvs = [string("general.architecture", "llama"), u32("llama.block_count", 22),
               u32("llama.embedding_length", 2048),
               u32("llama.attention.head_count", 32)]
        self.assertEqual(gm.shape(self.write(kvs))["n_head_kv"], 32)

    def test_file_size_is_reported(self):
        p = self.write(llama_like())
        self.assertEqual(gm.shape(p)["file_size"], p.stat().st_size)


class TestHostileHeaders(GGUFCase):
    """Every existing fixture builds one happy header. These build headers that are valid
    GGUF but wrong in ways the layer count depends on -- n_layers is the denominator of
    c_per_layer_ms, and c_i is what Rule A divides layers by."""

    def test_a_zero_block_count_is_refused(self):
        """`if not n_layers:` -> `if n_layers is None:` survived. Zero layers gives a
        ZeroDivisionError far from the cause, or an infinite 1/c_i that hands one node the
        entire model."""
        p = self.write(llama_like(block_count=0), name="zero.gguf")
        with self.assertRaises(gm.GGUFError) as cm:
            gm.shape(p)
        self.assertIn("block_count", str(cm.exception))

    def test_a_negative_block_count_is_refused(self):
        kvs = [string("general.architecture", "llama"),
               _kv("llama.block_count", 5, struct.pack("<i", -3)),
               u32("llama.embedding_length", 2048),
               u32("llama.attention.head_count", 32)]
        with self.assertRaises(gm.GGUFError):
            gm.shape(self.write(kvs, name="neg.gguf"))

    def test_the_block_count_is_architecture_specific_never_borrowed(self):
        """A qwen2 file carrying only `llama.block_count` must raise, not silently adopt
        another architecture's layer count."""
        kvs = [string("general.architecture", "qwen2"),
               u32("llama.block_count", 22),
               u32("qwen2.embedding_length", 2048),
               u32("qwen2.attention.head_count", 32)]
        with self.assertRaises(gm.GGUFError) as cm:
            gm.shape(self.write(kvs, name="borrow.gguf"))
        self.assertIn("qwen2.block_count", str(cm.exception))


class TestUnknownValueTypeRefusal(GGUFCase):
    """`raise GGUFError("unknown GGUF value type: %d")` -> `return None` survived the whole
    suite, because `test_unknown_value_type_raises` calls shape(), which trips the
    missing-block_count check FIRST -- it has never once executed the line it is named
    after. Returning None leaves the file pointer mid-value, so every subsequent key,
    including block_count, is read from the wrong offset and yields a plausible wrong
    integer: parses, exits 0, mis-scales every cell for that model."""

    def test_read_metadata_raises_and_names_the_tag(self):
        kvs = [string("general.architecture", "llama"),
               _kv("weird.key", 99, b"\x00\x00\x00\x00")]
        with self.assertRaises(gm.GGUFError) as cm:
            gm.read_metadata(self.write(kvs, name="tag99.gguf"))
        msg = str(cm.exception)
        self.assertIn("unknown GGUF value type", msg)
        self.assertIn("99", msg)

    def test_it_refuses_even_when_the_layer_count_came_first(self):
        """Proves refusal rather than a partially-good dict: a valid block_count placed
        BEFORE the unknown tag must not rescue the parse."""
        kvs = [string("general.architecture", "llama"),
               u32("llama.block_count", 22),
               _kv("weird.key", 99, b"\x00\x00\x00\x00"),
               u32("llama.embedding_length", 2048)]
        with self.assertRaises(gm.GGUFError) as cm:
            gm.read_metadata(self.write(kvs, name="tag99b.gguf"))
        self.assertIn("unknown GGUF value type", str(cm.exception))

    def test_an_unknown_array_element_type_is_refused_too(self):
        payload = struct.pack("<I", 99) + struct.pack("<Q", 2) + b"\x00" * 8
        kvs = [string("general.architecture", "llama"),
               _kv("weird.arr", ARRAY, payload),
               u32("llama.block_count", 22)]
        with self.assertRaises(gm.GGUFError):
            gm.read_metadata(self.write(kvs, name="arr99.gguf"))


class TestRejectsBadInput(GGUFCase):
    def test_non_gguf_magic_raises(self):
        p = self.write(llama_like(), magic=b"XXXX")
        with self.assertRaises(gm.GGUFError) as cm:
            gm.shape(p)
        self.assertIn("not a GGUF", str(cm.exception))

    def test_truncated_file_raises(self):
        p = self.write(llama_like())
        data = p.read_bytes()
        p.write_bytes(data[:len(data) // 2])
        with self.assertRaises(gm.GGUFError):
            gm.shape(p)

    def test_empty_file_raises(self):
        p = self.tmp / "empty.gguf"
        p.write_bytes(b"")
        with self.assertRaises(gm.GGUFError):
            gm.shape(p)

    def test_unknown_value_type_raises(self):
        kvs = [string("general.architecture", "llama"),
               _kv("weird.key", 99, b"\x00\x00\x00\x00")]
        with self.assertRaises(gm.GGUFError):
            gm.shape(self.write(kvs))


class TestCLI(GGUFCase):
    """10_isolated_bench.sh consumes this through `--field n_layers` in a command
    substitution and dies if it fails, so the bare-value contract and the exit codes
    are load-bearing."""

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(GGUF_META), *map(str, args)],
                              capture_output=True, text=True)

    def test_field_prints_a_bare_value(self):
        p = self.write(llama_like(block_count=22))
        proc = self.run_cli(p, "--field", "n_layers")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.strip(), "22")

    def test_missing_block_count_exits_nonzero(self):
        p = self.write(llama_like(block_count=None))
        proc = self.run_cli(p, "--field", "n_layers")
        self.assertEqual(proc.returncode, 1)
        self.assertEqual(proc.stdout.strip(), "")

    def test_unknown_field_exits_nonzero(self):
        p = self.write(llama_like())
        self.assertEqual(self.run_cli(p, "--field", "nonsense").returncode, 1)

    def test_no_arguments_exits_two(self):
        self.assertEqual(self.run_cli().returncode, 2)

    def test_default_output_is_json(self):
        import json
        p = self.write(llama_like(block_count=22))
        proc = self.run_cli(p)
        self.assertEqual(json.loads(proc.stdout)["n_layers"], 22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
