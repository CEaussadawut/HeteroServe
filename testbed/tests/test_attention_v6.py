"""Tests for the laptop's depth law per attention shape (2026-11-04, dl2): predict_pool's "fa_curve" law and ATTENTION_V6, and
the phase scheduler's hand-off memory check (hm1): phase_plan_20261028.handoff_fits and choose(receiver_avail=...).
stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                      # noqa: E402
import phase_plan_20261028 as PP                              # noqa: E402

MIB = 1024 * 1024
Q06 = {"head_dim": 128, "n_head": 16, "n_head_kv": 8, "architecture": "qwen3"}
Q9 = {"head_dim": 256, "n_head": 16, "n_head_kv": 4, "architecture": "qwen35"}


class FaCurve(unittest.TestCase):
    def setUp(self):
        self.tab = P.attention_table(P.ATTENTION_V6)
        self.lap = self.tab["boards"]["laptop"]

    def test_the_laptop_is_priced_per_shape_and_the_other_boards_as_v5(self):
        v5 = P.attention_table(P.ATTENTION_V5)
        self.assertEqual(self.lap["law"], "fa_curve")
        for n in ("pi5", "pi4", "jetson"):
            self.assertEqual(self.tab["boards"][n], v5["boards"][n])

    def test_the_law_is_the_curve_between_its_cells_over_the_depth(self):
        c = self.lap["shapes"]["128x16x8"]
        # depth 2,047: a token attends over 2,048 cells, a measured cell
        want = (c["2048"] - c["256"]) / 1000.0 / 2047
        self.assertAlmostEqual(P.attn_ms(self.tab, "laptop", Q06, 2047), want, places=9)
        # depth 1,800: 2,048 cells too (256 x ceil(1,801 / 256))
        self.assertAlmostEqual(P.attn_ms(self.tab, "laptop", Q06, 1800), (c["2048"] - c["256"]) / 1000.0 / 1800, places=9)

    def test_a_hybrid_carries_its_share_of_attention_layers(self):
        c = self.lap["shapes"]["256x16x4"]
        self.assertAlmostEqual(P.attn_ms(self.tab, "laptop", Q9, 2047), 0.25 * (c["2048"] - c["256"]) / 1000.0 / 2047, places=9)

    def test_no_depth_or_an_unknown_shape_gives_none_and_depth_0_costs_nothing(self):
        self.assertIsNone(P.attn_ms(self.tab, "laptop", Q06))
        self.assertIsNone(P.attn_ms(self.tab, "laptop", dict(Q06, n_head=7), 1024))
        self.assertEqual(P.attn_ms(self.tab, "laptop", Q06, 0), 0.0)

    def test_between_cells_the_curve_is_linear(self):
        c = {"256": 100.0, "512": 300.0, "1024": 500.0}
        self.assertAlmostEqual(P.fa_curve_us(c, 384), 200.0)
        self.assertAlmostEqual(P.fa_curve_us(c, 768), 400.0)
        self.assertAlmostEqual(P.fa_curve_us(c, 2048), 900.0)      # past the last cell, the last two cells' line

    def test_planner_inputs_prices_the_laptop_by_the_curve_at_the_depth(self):
        _, attn = P.planner_inputs(dict(Q06), "x", 4096, depth=2047, attn_table=P.ATTENTION_V6, nodes=["laptop"])
        self.assertAlmostEqual(attn["laptop"], P.attn_ms(self.tab, "laptop", Q06, 2047), places=12)

    def test_the_older_tables_price_as_before(self):
        v5 = P.attention_table(P.ATTENTION_V5)
        q = dict(Q06)
        self.assertEqual(P.attn_ms(v5, "laptop", q), P.attn_ms(v5, "laptop", q, 1024))


class FaCurvePis(unittest.TestCase):
    """2026-11-04 evening (dl3): ATTENTION_V7, the Pis priced per attention shape by their own op (fa_bench_cpu on each board)."""

    def setUp(self):
        self.v6, self.v7 = P.attention_table(P.ATTENTION_V6), P.attention_table(P.ATTENTION_V7)

    def test_the_pis_are_priced_per_shape_and_the_laptop_and_jetson_as_v6(self):
        self.assertEqual({b: e["law"] for b, e in self.v7["boards"].items()},
                         {"laptop": "fa_curve", "pi5": "fa_curve", "pi4": "fa_curve", "jetson": "fa_table"})
        self.assertEqual(self.v7["boards"]["laptop"], self.v6["boards"]["laptop"])
        self.assertEqual(self.v7["boards"]["jetson"], self.v6["boards"]["jetson"])
        for b in ("pi5", "pi4"):
            self.assertEqual(len(self.v7["boards"][b]["shapes"]), 9)                 # every shape the cluster serves
            self.assertEqual(self.v7["boards"][b]["hybrid_attention_fraction"], self.v6["boards"][b]["hybrid_attention_fraction"])

    def test_v7_reproduces_dl3s_registered_slope_on_the_pi5(self):
        # dl3 registered FA's slope for Qwen3-0.6B on the Pi 5 at 0-4,096 as 62.499 ms per 1,000 tokens (measured 62.14)
        import register_dl2_20261104 as R2
        pts = [(0, 0.0)] + [(d, 28 * P.attn_ms(self.v7, "pi5", Q06, depth=d) * d) for d in (1024, 2048, 4096)]
        self.assertAlmostEqual(R2.slope(pts), 62.499, delta=0.01)

    def test_v7_matches_the_published_build(self):
        import dl3_fa_pi_20261104 as dp, tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "v7.json")
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                dp.v7(out=out)
            built = json.load(open(out))
        self.assertEqual(built["boards"], self.v7["boards"])


class HandoffMemory(unittest.TestCase):
    def setUp(self):
        self.c = PP.load()                                      # Qwen3-0.6B's calibration (its state model)

    def test_fits_is_the_state_plus_the_headroom_within_the_receiver(self):
        s = self.c["state"]
        need = s["fixed_bytes"] + s["bytes_per_token"] * 1025 + PP.HEADROOM_BYTES
        self.assertTrue(PP.handoff_fits(self.c, 1024, need)[0])
        self.assertFalse(PP.handoff_fits(self.c, 1024, need - 1)[0])

    def test_choose_leaves_h_out_when_it_does_not_fit_and_says_why(self):
        free = PP.choose(8192, 64, "time", costs=self.c)
        tight = PP.choose(8192, 64, "time", costs=self.c, receiver_avail=200 * MIB)
        self.assertNotEqual(tight["pick"], "H")
        self.assertFalse(tight["plans"]["H"]["memory"]["fits"])
        self.assertNotIn("memory", free["plans"]["H"])             # no receiver figure: priced as before
        roomy = PP.choose(8192, 64, "time", costs=self.c, receiver_avail=64 * 1024 * MIB)
        self.assertEqual(roomy["pick"], free["pick"])

    def test_the_headroom_agrees_with_the_hand_offs_measured(self):
        # the 4B's server at 2,560 cells left 401 MiB: 77.3 MB of state ran clean, 152.8 MB thrashed
        avail = 401 * MIB
        self.assertGreaterEqual(avail - 77276236 - PP.HEADROOM_BYTES, 0)
        self.assertLess(avail - 152781900 - PP.HEADROOM_BYTES, 0)


if __name__ == "__main__":
    unittest.main()
