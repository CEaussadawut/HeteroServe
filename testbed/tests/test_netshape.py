"""Tests for the network study's scorer (testbed/scripts/score_netshape_20261008.py): its pure parts, stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import score_netshape_20261008 as S                           # noqa: E402
import register_netshape_20261008 as G                        # noqa: E402
import pd_network_crossover as X                              # noqa: E402


class TestNetshape(unittest.TestCase):
    def test_measured_crossover_interpolates_in_inverse_bandwidth(self):
        # gap = host - last: positive (last faster) at high bandwidth, negative below the crossover
        x = S.measured_crossover([(938.0, 6.3), (200.0, 2.5), (100.0, -2.2), (50.0, -11.8), (20.0, -40.6)])
        self.assertTrue(100.0 < x < 200.0)
        # exactly linear in 1/b: g = 2.0 - 260 / b crosses at b = 130
        pts = [(b, 2.0 - 260.0 / b) for b in (938.0, 200.0, 100.0, 50.0)]
        self.assertAlmostEqual(S.measured_crossover(pts), 130.0, places=6)
        self.assertIsNone(S.measured_crossover([(938.0, 6.3), (200.0, 2.5), (100.0, 0.4)]))       # no sign change
        self.assertEqual(S.measured_crossover([(200.0, 0.0), (100.0, -1.0)]), 200.0)              # exactly zero

    def test_slope_through_origin(self):
        self.assertAlmostEqual(S.slope_through_origin([2.0, 10.0], [4.0, 20.0]), 2.0)
        self.assertIsNone(S.slope_through_origin([], []))
        self.assertIsNone(S.slope_through_origin([0.0], [3.0]))

    def test_registration_predicts_the_flip(self):
        r = {"jetson": {"mbps": 938.1, "rtt_ms": 0.612}}
        import json
        shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
        R = G.pp.rows(os.path.join(ROOT, "testbed", "dataset", "characterisation.csv"), "char")
        shape, quant, _ = G.pp.MODELS[G.KEY]
        C, _ = G.pp.costs(R, shapes, quant)
        x = G.crossover(r, shapes[shape], C)
        self.assertTrue(100.0 < x < 170.0, x)            # the registered prediction: about 130 Mbit/s
        self.assertEqual(G.nominal_link({"mbps": 938.1, "rtt_ms": 0.612}, 50, 0.0), {"mbps": 50.0, "rtt_ms": 0.612})
        self.assertEqual(G.nominal_link({"mbps": 938.1, "rtt_ms": 0.612}, 0, 5.0), {"mbps": 938.1, "rtt_ms": 10.612})

    def test_handoff_crossover_formulas(self):
        # two users, 32 tokens back (2026-09-29): state 56,889,668 bytes, 0.51 s at gigabit, hand-off 55.06 s
        self.assertAlmostEqual(X.crossover_two(56889668, 0.51, 55.06, 65.13), 56889668 * 8 / (0.51 + 2 * 10.07) / 1e6)
        self.assertIsNone(X.crossover_two(56889668, 0.51, 153.71, 126.81))          # loses even at gigabit
        self.assertIsNone(X.crossover_single(56889664, 13.0, 13.5, 0.03, 0.07))      # the fast board is not faster
        self.assertAlmostEqual(X.crossover_single(8e6, 11.0, 1.0, 0.0, 0.0), 6.4)     # 8 MB in 10 s = 6.4 Mbit/s

    def test_handoff_crossover_from_the_measured_runs(self):
        p = os.path.join(ROOT, "testbed", "results", "pd_twouser_20260929_215703.log")
        if not os.path.exists(p):
            self.skipTest("the 2026-09-29 two-user log is not in this checkout")
        hr = X.handoff_results(p)
        self.assertEqual(len(hr), 3)
        self.assertTrue(all(h["state_bytes"] == 56889668 for h in hr))


if __name__ == "__main__":
    unittest.main()
