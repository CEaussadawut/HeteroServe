"""The Jetson's attention law by its own op (2026-10-28, lc4n): predict_pool.attn_ms() with law "fa_table"."""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import predict_pool as P  # noqa: E402

SHAPES = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))


class AttentionFa(unittest.TestCase):
    def setUp(self):
        self.fa = P.attention_table(P.ATTENTION_FA)
        self.old = P.attention_table()

    def test_the_default_table_is_unchanged(self):
        self.assertEqual(self.old["boards"]["jetson"]["law"], "kv_bytes")
        self.assertEqual(P.ATTENTION, os.path.join(ROOT, "testbed", "dataset", "attention_cost_20261011.json"))

    def test_tinyllama_is_its_fa_bench_slope(self):
        q = SHAPES["tinyllama-1.1b-q4km"]
        e = self.fa["boards"]["jetson"]["table"]["64x32x4"]
        self.assertAlmostEqual(P.attn_ms(self.fa, "jetson", q), e["ms_per_layer_per_token_long"])
        self.assertAlmostEqual(P.attn_ms(self.fa, "jetson", q) * 22 * 1000, 4.0, delta=0.1)   # lc4n measured 4.02-4.05

    def test_a_hybrid_carries_a_quarter(self):
        q = SHAPES["qwen3.6-35b-a3b-q6_k"]
        e = self.fa["boards"]["jetson"]["table"]["256x16x2"]
        self.assertAlmostEqual(P.attn_ms(self.fa, "jetson", q), e["ms_per_layer_per_token_long"] * 0.25)

    def test_an_unknown_shape_has_no_law(self):
        q = dict(SHAPES["qwen3-14b-q4_k_m"], head_dim=96)
        self.assertIsNone(P.attn_ms(self.fa, "jetson", q))

    def test_the_cpus_keep_their_law(self):
        q = SHAPES["qwen3-14b-q4_k_m"]
        for n in ("pi4", "laptop"):
            self.assertEqual(P.attn_ms(self.fa, n, q), P.attn_ms(P.attention_table(os.path.join(ROOT, "testbed", "dataset",
                                                                                                 "attention_cost_20261011b.json")), n, q))


if __name__ == "__main__":
    unittest.main()
