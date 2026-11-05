"""The predictor rule VISIT under test (2026-10-09): predict_pool's visit_ms, its registration and its scorer; stdlib
unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402
import register_visit_20261009 as RV                          # noqa: E402
import score_visit_20261009 as SV                             # noqa: E402

MEM = json.load(open(os.path.join(ROOT, "testbed", "dataset", "memory_at_rest.json")))


class TestVisit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = RV.build()
        shapes = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
        cls.q = shapes[P.MODELS["tinyllama"][0]]

    def test_the_default_reproduces_every_earlier_registration(self):
        r = P.plan_fixed("tinyllama", [("pi4", 1), ("jetson", 21)], "last", 4096, MEM, rtt_trips=1)
        a = P.predict(r["assign"], r["q"], r["C"], r["links"], rtt_trips=1)
        self.assertEqual(a, P.predict(r["assign"], r["q"], r["C"], r["links"], rtt_trips=1, visit_ms=0.0))

    def test_a_visit_adds_its_cost_to_decode_only_once_per_worker(self):
        r = P.plan_fixed("tinyllama", [("pi4", 1), ("jetson", 21)], "last", 4096, MEM, rtt_trips=1)
        d0, p0 = P.predict(r["assign"], r["q"], r["C"], r["links"], rtt_trips=1)
        d1, p1 = P.predict(r["assign"], r["q"], r["C"], r["links"], rtt_trips=1, visit_ms=P.VISIT_MS)
        self.assertAlmostEqual(d1 - d0, P.VISIT_MS)                 # one worker, one visit
        self.assertEqual(p1, p0)                                    # prefill untouched
        self.assertAlmostEqual(P.VISIT_MS, 3.173 / 2, places=3)     # the ring-lap study's per-lap residual, split evenly

    def test_the_layouts_keep_each_boards_bytes_and_differ_only_in_visits(self):
        L = self.reg["layouts"]
        self.assertEqual({k: (v["jetson_visits"], v["loopback_visits"]) for k, v in L.items()},
                         {"h1": (1, 0), "h2": (2, 0), "h4": (4, 0), "a1": (1, 1), "p4": (1, 4), "j3": (3, 1)})
        for lab in ("h1", "h2", "h4"):
            self.assertEqual((L[lab]["ngl"], L[lab]["host_blocks"]), (19, 4))
        for lab in ("a1", "p4", "j3"):
            self.assertEqual((L[lab]["ngl"], L[lab]["host_blocks"]), (None, 0))
        for lab, lay in L.items():                                  # the Jetson always holds blocks 4-21 and the output
            exp = SV.expected_buffers(lay, self.q)
            jet = sum(v for ep, v in exp.items() if ep.startswith("192.168.1.114"))
            want = (sum(self.q["layer_bytes"][4:]) + self.q["head_bytes"]) / 2 ** 20
            self.assertAlmostEqual(jet, want, places=3, msg=lab)
        self.assertEqual(L["h1"]["args"].count("-ngl 19"), 1)
        self.assertEqual(L["a1"]["args"].count("-ngl 99"), 1)

    def test_the_predicted_slopes_differ_by_exactly_the_visit_cost(self):
        s = self.reg["predicted_slopes_ms_per_visit"]
        for h in ("VISIT-J", "VISIT-P"):
            self.assertAlmostEqual(s[h]["visit"] - s[h]["rtt1"], P.VISIT_MS, delta=0.01)

    def _cells(self, vj, vp, const=0.0, drop=()):
        cells = {}
        for r in (1, 2, 3):
            for lab, lay in self.reg["layouts"].items():
                if (r, lab) in drop:
                    continue
                d = lay["predicted"]["rtt1"]["decode_ms"] + const + vj * lay["jetson_visits"] + vp * lay["loopback_visits"]
                cells[(r, "s0", lab)] = {"t0": 0.0, "t1": 1.0, "tg": d, "pp": 26.0, "exit": 0,
                                         "bufs": {ep: v for ep, v in SV.expected_buffers(lay, self.q).items()}}
        return cells

    def test_the_scorer_recovers_the_costs_and_applies_the_registered_rules(self):
        s = SV.score(self.reg, self.q, self._cells(1.6, 1.5, const=1.0))
        self.assertEqual(s["hypotheses"]["VISIT-J"]["verdict"], "HOLDS")
        self.assertEqual(s["hypotheses"]["VISIT-P"]["verdict"], "HOLDS")
        self.assertEqual(s["hypotheses"]["VISIT-MAE"]["verdict"], "HOLDS")
        self.assertEqual(s["adoption"], "ADOPTED")
        fit = s["exploratory"]["residual_over_rtt1_fit"]
        self.assertAlmostEqual(fit["per_jetson_visit_ms"], 1.6, places=6)
        self.assertAlmostEqual(fit["per_loopback_visit_ms"], 1.5, places=6)
        self.assertAlmostEqual(fit["const_ms"], 1.0, places=6)
        s = SV.score(self.reg, self.q, self._cells(0.2, 0.1))       # visits nearly free: the rule fails and is not adopted
        self.assertEqual(s["hypotheses"]["VISIT-J"]["verdict"], "FAILS")
        self.assertEqual(s["adoption"], "NOT ADOPTED")
        s = SV.score(self.reg, self.q, self._cells(1.6, 1.5, drop={(2, "h2")}))
        self.assertEqual(s["hypotheses"]["VISIT-J"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["missing"], ["2.s0.h2"])

    def test_a_cell_whose_layout_did_not_load_as_registered_is_left_out(self):
        cells = self._cells(1.6, 1.5)
        ep = next(iter(cells[(1, "s0", "h2")]["bufs"]))
        cells[(1, "s0", "h2")]["bufs"][ep] += 25.0                    # a block on the wrong device
        s = SV.score(self.reg, self.q, cells)
        self.assertIn("1.s0.h2", s["left_out"])
        self.assertEqual(s["hypotheses"]["VISIT-J"]["verdict"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
