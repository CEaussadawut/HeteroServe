"""Tests for the GGUF header reader, the llama.cpp timing parser and config validation.

The GGUF test writes a real header byte by byte rather than mocking the reader, because the
thing that can actually break is the binary layout, and a mock would agree with whatever the
reader does.
"""
import json
import struct
import tempfile
import unittest
from pathlib import Path

from control import config, gguf
from control.harness import parse_timings


# ------------------------------------------------------------ GGUF synthesis
def _string(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, tag: int, payload: bytes) -> bytes:
    return _string(key) + struct.pack("<I", tag) + payload


def write_gguf(path, arch="llama", n_layers=32, n_embd=4096, n_head=32, n_head_kv=8,
               padding=4096):
    entries = [
        _kv("general.architecture", gguf.STRING, _string(arch)),
        _kv("%s.block_count" % arch, gguf.UINT32, struct.pack("<I", n_layers)),
        _kv("%s.embedding_length" % arch, gguf.UINT32, struct.pack("<I", n_embd)),
        _kv("%s.attention.head_count" % arch, gguf.UINT32, struct.pack("<I", n_head)),
        _kv("%s.attention.head_count_kv" % arch, gguf.UINT32, struct.pack("<I", n_head_kv)),
        _kv("%s.context_length" % arch, gguf.UINT32, struct.pack("<I", 8192)),
        # An array, to exercise the array path and the large-array skip
        _kv("tokenizer.ggml.tokens", gguf.ARRAY,
            struct.pack("<I", gguf.STRING) + struct.pack("<Q", 2) + _string("a") + _string("b")),
    ]
    body = b"".join(entries)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 291) + struct.pack("<Q", len(entries))
    Path(path).write_bytes(header + body + b"\x00" * padding)
    return path


def write_split_gguf(path, shard_no, n_shards, arch="qwen2", n_layers=48, padding=4096):
    """write_gguf plus the split.no/split.count keys llama.cpp emits for a sharded model."""
    entries = [
        _kv("general.architecture", gguf.STRING, _string(arch)),
        _kv("%s.block_count" % arch, gguf.UINT32, struct.pack("<I", n_layers)),
        _kv("%s.embedding_length" % arch, gguf.UINT32, struct.pack("<I", 5120)),
        _kv("%s.attention.head_count" % arch, gguf.UINT32, struct.pack("<I", 40)),
        _kv("%s.attention.head_count_kv" % arch, gguf.UINT32, struct.pack("<I", 8)),
        _kv("split.no", gguf.UINT16, struct.pack("<H", shard_no)),
        _kv("split.count", gguf.UINT16, struct.pack("<H", n_shards)),
    ]
    body = b"".join(entries)
    header = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 291) + struct.pack("<Q", len(entries))
    Path(path).write_bytes(header + body + b"\x00" * padding)
    return path


def write_gguf_with_tensors(path, tensors, n_layers=2, extra=(), alignment=32):
    """A GGUF with a real tensor-info table and a zero-filled data section.

    tensors: [(name, dims, n_bytes)]. Each tensor is placed at the next multiple of the
    alignment, as llama.cpp writes them, so a size read back from the offsets is n_bytes
    rounded up to the alignment.
    """
    entries = [
        _kv("general.architecture", gguf.STRING, _string("llama")),
        _kv("llama.block_count", gguf.UINT32, struct.pack("<I", n_layers)),
        _kv("llama.embedding_length", gguf.UINT32, struct.pack("<I", 64)),
        _kv("llama.attention.head_count", gguf.UINT32, struct.pack("<I", 4)),
    ] + list(extra)
    infos, offset = b"", 0
    for name, dims, n_bytes in tensors:
        infos += (_string(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
                  + struct.pack("<I", 12) + struct.pack("<Q", offset))
        offset += -(-n_bytes // alignment) * alignment
    head = (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", len(entries)) + b"".join(entries) + infos)
    Path(path).write_bytes(head + b"\x00" * (-len(head) % alignment) + b"\x00" * offset)
    return path


# Two layers of different size (layer 1's ffn_down at a larger quant, as Q4_K_M does), the
# embedding, and the head.
TENSORS = [
    ("token_embd.weight", (64, 1000), 32000),
    ("blk.0.attn_q.weight", (64, 64), 3200),
    ("blk.0.ffn_down.weight", (64, 128), 6400),
    ("blk.1.attn_q.weight", (64, 64), 3200),
    ("blk.1.ffn_down.weight", (64, 128), 9600),
    ("output_norm.weight", (64,), 256),
    ("output.weight", (64, 1000), 48000),
]


class TestGGUF(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_gguf(Path(self.tmp.name) / "test-7b.gguf")

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_layer_count_and_geometry(self):
        shape = gguf.shape(self.path)
        self.assertEqual(shape["n_layers"], 32)
        self.assertEqual(shape["n_embd"], 4096)
        self.assertEqual(shape["n_head_kv"], 8)
        self.assertEqual(shape["head_dim"], 128)          # 4096 / 32

    def test_kv_cache_size_follows_the_gqa_geometry(self):
        shape = gguf.shape(self.path)
        # 2 (K and V) * 8 kv heads * 128 head dim * 2 bytes (f16) = 4096 B per layer per token
        self.assertEqual(shape["kv_bytes_per_layer_per_token"], 4096)

    def test_activation_size_is_the_hidden_state(self):
        self.assertEqual(gguf.shape(self.path)["activation_bytes"], 4096 * 4)

    def test_rejects_a_file_that_is_not_gguf(self):
        bad = Path(self.tmp.name) / "not.gguf"
        bad.write_bytes(b"XXXX" + b"\x00" * 64)
        with self.assertRaises(gguf.GGUFError):
            gguf.shape(bad)

    def test_profile_built_from_the_header(self):
        from control.profiles import model_profile
        profile = model_profile(self.path, context_len=1024)
        self.assertEqual(profile.n_layers, 32)
        self.assertGreater(profile.per_layer_footprint_bytes(), profile.bytes_per_layer)


class TestTensorTable(unittest.TestCase):
    """Per-layer bytes from the tensor table, with the head and embedding reported apart."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_layers_head_and_embedding_come_from_the_table(self):
        s = gguf.shape(write_gguf_with_tensors(self.dir / "m.gguf", TENSORS))
        self.assertEqual(s["bytes_per_layer_source"], "tensor table")
        self.assertEqual(s["layer_bytes"], [9600, 12800])
        self.assertEqual(s["bytes_per_layer"], 11200)
        self.assertEqual((s["head_bytes"], s["embd_bytes"]), (48256, 32000))
        self.assertFalse(s["head_tied_to_embedding"])
        self.assertEqual(s["params_per_layer"], 64 * 64 + 64 * 128)
        self.assertEqual(s["n_params"], 2 * (64 * 64 + 64 * 128) + 2 * 64000 + 64)
        # The largest single tensor inside a block (layer 1's ffn_down), not a layer's sum or the head.
        self.assertEqual(s["max_block_tensor_bytes"], 9600)

    def test_a_tied_head_is_the_embedding(self):
        tied = [t for t in TENSORS if t[0] != "output.weight"]
        s = gguf.shape(write_gguf_with_tensors(self.dir / "m.gguf", tied))
        self.assertTrue(s["head_tied_to_embedding"])
        self.assertEqual(s["head_bytes"], 256 + 32000)

    def test_header_only_file_falls_back_and_says_so(self):
        s = gguf.shape(write_gguf(self.dir / "h.gguf"))
        self.assertTrue(s["bytes_per_layer_source"].startswith("file size / n_layers"))
        self.assertEqual(s["head_bytes"], 0.0)
        self.assertEqual(s["bytes_per_layer"], s["file_size"] / 32)

    def test_a_missing_layer_is_not_averaged_over(self):
        # block_count says 3 but the table has blocks 0 and 1: do not divide 2 layers' bytes by 3.
        s = gguf.shape(write_gguf_with_tensors(self.dir / "m.gguf", TENSORS, n_layers=3))
        self.assertTrue(s["bytes_per_layer_source"].startswith("file size / n_layers"))

    def test_split_model_sums_every_shard_table(self):
        def split(no):
            return [_kv("split.no", gguf.UINT16, struct.pack("<H", no)),
                    _kv("split.count", gguf.UINT16, struct.pack("<H", 2))]
        first = write_gguf_with_tensors(self.dir / "m-00001-of-00002.gguf", TENSORS[:3], extra=split(0))
        write_gguf_with_tensors(self.dir / "m-00002-of-00002.gguf", TENSORS[3:], extra=split(1))
        s = gguf.shape(first)
        self.assertEqual(s["bytes_per_layer_source"], "tensor table")
        self.assertEqual((s["layer_bytes"], s["head_bytes"]), ([9600, 12800], 48256))

    def test_a_mixture_of_experts_reads_only_its_active_experts(self):
        # 8 experts, 2 used: a token reads the attention, the router and a quarter of each expert tensor.
        per_layer = lambda i: [("blk.%d.attn_q.weight" % i, (64, 64), 3200),
                               ("blk.%d.ffn_gate_inp.weight" % i, (64, 8), 512),
                               ("blk.%d.ffn_up_exps.weight" % i, (64, 128, 8), 8000),
                               ("blk.%d.ffn_down_exps.weight" % i, (128, 64, 8), 8000)]
        moe = [("token_embd.weight", (64, 1000), 32000)] + per_layer(0) + per_layer(1) + \
              [("output_norm.weight", (64,), 256), ("output.weight", (64, 1000), 48000)]
        extra = [_kv("llama.expert_count", gguf.UINT32, struct.pack("<I", 8)),
                 _kv("llama.expert_used_count", gguf.UINT32, struct.pack("<I", 2))]
        s = gguf.shape(write_gguf_with_tensors(self.dir / "moe.gguf", moe, extra=extra))
        self.assertEqual((s["expert_count"], s["expert_used_count"]), (8, 2))
        self.assertEqual(s["layer_bytes"], [19712, 19712])                     # memory holds every expert
        self.assertEqual(s["layer_active_bytes"], [19712 - 16000 * 0.75] * 2)   # 7712 read per token
        self.assertEqual(s["active_params_per_layer"], 4096 + 512 + 2 * 65536 * 0.25)
        self.assertEqual(s["max_block_tensor_bytes"], 8000)                   # a whole expert tensor travels
        dense = gguf.shape(write_gguf_with_tensors(self.dir / "m.gguf", TENSORS))
        self.assertNotIn("layer_active_bytes", dense)                        # a dense model is unchanged

    def test_profile_carries_the_head_and_embedding(self):
        from control.profiles import model_profile
        p = model_profile(write_gguf_with_tensors(self.dir / "m.gguf", TENSORS), context_len=16)
        self.assertEqual((p.bytes_per_layer, p.head_bytes, p.embd_bytes), (11200, 48256, 32000))
        self.assertEqual(p.max_block_tensor_bytes, 9600)
        self.assertFalse(p.host_in_pipeline)

# --------------------------------------------------------- timing extraction
LLAMA_LOG = """
llama_perf_context_print:        load time =    1234.56 ms
llama_perf_context_print: prompt eval time =    5432.10 ms /   128 tokens (   42.44 ms per token,    23.56 tokens per second)
llama_perf_context_print:        eval time =   22000.00 ms /   127 runs   (  173.23 ms per token,     5.77 tokens per second)
llama_perf_context_print:       total time =   28900.00 ms /   255 tokens
"""


class TestTimingParser(unittest.TestCase):
    def test_prefill_and_decode_are_separated(self):
        t = parse_timings(LLAMA_LOG)
        self.assertAlmostEqual(t["prefill_ms"], 5432.10)
        self.assertEqual(t["prefill_tokens"], 128)
        self.assertAlmostEqual(t["decode_ms"], 22000.0)
        self.assertEqual(t["decode_tokens"], 127)

    def test_tpot_is_decode_time_per_token(self):
        t = parse_timings(LLAMA_LOG)
        self.assertAlmostEqual(t["tpot_ms"], 22000.0 / 127, places=2)

    def test_empty_log_yields_nothing_rather_than_zeros(self):
        # A failed run must not look like a fast one.
        self.assertEqual(parse_timings("no timings here"), {})

    def test_phase_windows_lay_the_reported_durations_end_to_end(self):
        from control.harness import Harness
        record = {
            "run_id": "r1", "started_utc": 1000.0, "ended_utc": 1030.0,
            "timings": parse_timings(LLAMA_LOG),
        }
        windows = Harness.phase_windows(object.__new__(Harness), record)
        phases = [w["phase"] for w in windows]
        self.assertEqual(phases, ["load", "prefill", "decode"])
        self.assertAlmostEqual(windows[0]["t_start"], 1000.0)
        self.assertTrue(all(w["t_end"] > w["t_start"] for w in windows))
        # The gap between the reported durations and the wall clock is stated, not hidden.
        self.assertIn("unaccounted_ms", windows[0])

    def test_each_node_window_carries_the_phase_decomposition(self):
        """The recorded window must let a later energy join avoid the Pi4 C4 mistake.

        `t_start`/`t_end` are the whole llama-cli process. Pairing that span with
        `decode_tokens` charges model load AND the entire prefill to decode alone -- the
        same defect that overstated the per-node bench metric by tens of percent. The split
        has to be in the record, because it cannot be recovered afterwards without
        re-running the cluster.
        """
        from control.harness import Harness
        with tempfile.TemporaryDirectory() as tmp:
            h = object.__new__(Harness)
            h.data_dir = Path(tmp)
            run_dir = Path(tmp) / "runs" / "r1"
            run_dir.mkdir(parents=True)
            record = {
                "run_id": "r1", "started_utc": 1000.0, "ended_utc": 1030.0,
                "strategy": "even", "model": "m", "tier": "1g",
                "participants": ["pi4", "pi5"], "layers": {"pi4": 8, "pi5": 20},
                "timings": parse_timings(LLAMA_LOG),
            }
            Harness._append_window(h, "r1", record)
            lines = [json.loads(x) for x in
                     (run_dir / "windows.jsonl").read_text().splitlines() if x.strip()]

        self.assertEqual([w["node"] for w in lines], ["pi4", "pi5"])
        for w in lines:
            # the full-process window keeps its original meaning
            self.assertEqual((w["t_start"], w["t_end"], w["phase"]), (1000.0, 1030.0, "full"))
            self.assertEqual([p["phase"] for p in w["phases"]], ["load", "prefill", "decode"])
            self.assertTrue(w["phase_split"])
            decode = [p for p in w["phases"] if p["phase"] == "decode"][0]
            # decode's own span, not the process span -- that is the whole point
            self.assertLess(decode["t_end"] - decode["t_start"], 1030.0 - 1000.0)
            self.assertEqual(decode["tokens"], w["decode_tokens"])

    def test_a_run_with_no_timings_records_no_guessed_split(self):
        """A failed run must not acquire a plausible-looking phase split."""
        from control.harness import Harness
        with tempfile.TemporaryDirectory() as tmp:
            h = object.__new__(Harness)
            h.data_dir = Path(tmp)
            (Path(tmp) / "runs" / "r2").mkdir(parents=True)
            record = {
                "run_id": "r2", "started_utc": 1000.0, "ended_utc": 1030.0,
                "strategy": "even", "model": "m", "tier": "1g",
                "participants": ["pi4"], "layers": {"pi4": 8}, "timings": {},
            }
            Harness._append_window(h, "r2", record)
            w = json.loads((Path(tmp) / "runs" / "r2" / "windows.jsonl").read_text().strip())
        self.assertEqual(w["phases"], [])
        self.assertEqual(w["phase_split"], "")


# --------------------------------------------------------- config validation
BASE_CONFIG = {
    "cluster": {"name": "t", "subnet": "192.168.50.0/24", "control_node": "t580"},
    "nodes": [
        {"id": "t580", "role": "control", "host": "192.168.50.10", "serves_rpc": False},
        {"id": "pi4", "role": "worker-straggler", "host": "192.168.50.14", "serves_rpc": True,
         "usable_mem_mb": 3200, "ina_channel": 2, "agent_port": 9410, "ssh_user": "pi"},
    ],
    "daq": {"channels": {"2": {"node": "pi4"}}},
}


def load_config(tmpdir, mutate=None):
    raw = json.loads(json.dumps(BASE_CONFIG))
    if mutate:
        mutate(raw)
    path = Path(tmpdir) / "cluster.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return config.load(path)


class TestClusterConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_config_loads(self):
        cluster = load_config(self.tmp.name)
        self.assertEqual(cluster.control.id, "t580")
        self.assertEqual([n.id for n in cluster.workers()], ["pi4"])

    def test_measured_memory_overheads_are_read_and_a_missing_role_stays_unmeasured(self):
        def mutate(raw):
            pi4 = next(n for n in raw["nodes"] if n["id"] == "pi4")
            pi4["memory_overhead_mb"] = {"worker": 159}
            pi4["worker_repays_received_tensor"] = True
        pi4 = next(n for n in load_config(self.tmp.name, mutate).workers() if n.id == "pi4")
        self.assertEqual((pi4.host_overhead_mb, pi4.worker_overhead_mb, pi4.worker_repays_received),
                         (None, 159, True))
        plain = next(n for n in load_config(self.tmp.name).workers() if n.id == "pi4")
        self.assertEqual((plain.host_overhead_mb, plain.worker_overhead_mb, plain.worker_repays_received),
                         (None, None, False))

    def test_a_memory_reserve_is_read_and_is_zero_when_absent(self):
        def mutate(raw):
            next(n for n in raw["nodes"] if n["id"] == "pi4")["memory_reserve_mb"] = 450
        pi4 = next(n for n in load_config(self.tmp.name, mutate).workers() if n.id == "pi4")
        self.assertEqual(pi4.memory_reserve_mb, 450.0)
        plain = next(n for n in load_config(self.tmp.name).workers() if n.id == "pi4")
        self.assertEqual(plain.memory_reserve_mb, 0.0)

    def test_a_weights_cap_is_read_and_none_when_absent(self):
        def mutate(raw):
            next(n for n in raw["nodes"] if n["id"] == "pi4")["max_weights_mb"] = 2845
        pi4 = next(n for n in load_config(self.tmp.name, mutate).workers() if n.id == "pi4")
        self.assertEqual(pi4.max_weights_mb, 2845.0)
        plain = next(n for n in load_config(self.tmp.name).workers() if n.id == "pi4")
        self.assertIsNone(plain.max_weights_mb)

    def test_the_real_jetson_carries_its_measured_weights_cap(self):
        # cluster.json itself: the cap registered rule GPU_CAP set (2026-09-29), with its evidence beside it.
        real = json.loads((Path(__file__).resolve().parents[1] / "config" / "cluster.json").read_text(encoding="utf-8"))
        jetson = next(n for n in real["nodes"] if n["id"] == "jetson")
        self.assertEqual(jetson["max_weights_mb"], 2845)
        self.assertIn("GPU_CAP", jetson["max_weights_source"])
        self.assertTrue(all("max_weights_mb" not in n for n in real["nodes"] if n["id"] != "jetson"))

    def test_a_mapped_host_flag_is_read_and_false_when_absent(self):
        def mutate(raw):
            next(n for n in raw["nodes"] if n["id"] == "pi4")["host_weights_mapped"] = True
        pi4 = next(n for n in load_config(self.tmp.name, mutate).workers() if n.id == "pi4")
        self.assertTrue(pi4.host_weights_mapped)
        plain = next(n for n in load_config(self.tmp.name).workers() if n.id == "pi4")
        self.assertFalse(plain.host_weights_mapped)

    def test_duplicate_daq_channels_are_rejected(self):
        def mutate(raw):
            raw["nodes"].append({"id": "pi5", "role": "worker", "host": "192.168.50.12",
                                 "serves_rpc": True, "usable_mem_mb": 7000, "ina_channel": 2})
        with self.assertRaises(config.ConfigError):
            load_config(self.tmp.name, mutate)

    def test_daq_map_disagreeing_with_the_node_is_rejected(self):
        # This is the mistake that attributes one node's power to another and produces
        # numbers that look entirely reasonable.
        def mutate(raw):
            raw["daq"]["channels"] = {"3": {"node": "pi4"}}
        with self.assertRaises(config.ConfigError):
            load_config(self.tmp.name, mutate)

    def test_serving_node_without_a_memory_budget_is_rejected(self):
        def mutate(raw):
            raw["nodes"][1].pop("usable_mem_mb")
        with self.assertRaises(config.ConfigError):
            load_config(self.tmp.name, mutate)

    def test_rpc_endpoint_and_agent_url(self):
        cluster = load_config(self.tmp.name)
        node = cluster.node("pi4")
        self.assertEqual(node.rpc_endpoint, "192.168.50.14:50052")
        self.assertEqual(node.agent_url, "http://192.168.50.14:9410")


class TestSplitGGUF(unittest.TestCase):
    """`bytes_per_layer` is `file_size / n_layers`, and `file_size` is ONE shard.

    A split model therefore under-counts by the shard count: a 2-shard 14B halves
    bytes_per_layer, doubles the computed memory_cap, and can flip a node from infeasible to
    feasible on `config/experiments/memory_wall_14b.json` -- whose stated headline result is
    that the single-node entries ARE infeasible ("that refusal IS the result"). Inverting a
    published result is the worst outcome available here, so the shards are summed when all
    are present and the read refuses when any is missing.
    """

    def test_shards_are_summed_not_counted_once(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a = write_split_gguf(d / "m-00001-of-00002.gguf", 1, 2, padding=4_000_000)
            b = write_split_gguf(d / "m-00002-of-00002.gguf", 2, 2, padding=4_400_000)
            total = a.stat().st_size + b.stat().st_size
            shape = gguf.shape(a)
            self.assertAlmostEqual(shape["bytes_per_layer"], total / 48.0, places=1)
            # and emphatically not the single-shard figure
            self.assertNotAlmostEqual(shape["bytes_per_layer"], a.stat().st_size / 48.0, places=1)

    def test_a_missing_shard_refuses_rather_than_under_counting(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            a = write_split_gguf(d / "m-00001-of-00002.gguf", 1, 2, padding=4_000_000)
            with self.assertRaises(gguf.GGUFError) as cm:
                gguf.shape(a)
            self.assertIn("missing shard", str(cm.exception))

    def test_a_single_file_model_is_unaffected(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_gguf(Path(d) / "plain.gguf", n_layers=32, padding=3200)
            shape = gguf.shape(path)
            self.assertAlmostEqual(shape["bytes_per_layer"],
                                   Path(path).stat().st_size / 32.0, places=1)



if __name__ == "__main__":
    unittest.main()
