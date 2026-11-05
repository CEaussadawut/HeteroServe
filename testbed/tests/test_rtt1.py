"""The registered rule RTT1 (2026-10-09): predict_pool's rtt_trips, stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402
import energy_model as EM                                     # noqa: E402

MEM = json.load(open(os.path.join(ROOT, "testbed", "dataset", "memory_at_rest.json")))


class TestRtt1(unittest.TestCase):
    def setUp(self):
        self.r = P.plan_fixed("tinyllama", [("pi4", 1), ("jetson", 21)], "last", 4096, MEM)
        self.rtt = self.r["links"]["jetson"]["rtt_ms"]

    def test_the_default_reproduces_every_earlier_registration(self):
        r = self.r
        self.assertEqual(P.RTT_TRIPS_DEFAULT, 2)
        self.assertEqual(r["rtt_trips"], 2)
        self.assertEqual(P.predict(r["assign"], r["q"], r["C"], r["links"]),
                         P.predict(r["assign"], r["q"], r["C"], r["links"], rtt_trips=2))

    def test_one_trip_takes_one_rtt_per_worker_from_decode_and_one_per_batch_from_prefill(self):
        r = self.r
        for head_at in (None, 0):
            d2, p2 = P.predict(r["assign"], r["q"], r["C"], r["links"], head_at=head_at, rtt_trips=2)
            d1, p1 = P.predict(r["assign"], r["q"], r["C"], r["links"], head_at=head_at, rtt_trips=1)
            self.assertAlmostEqual(d2 - d1, self.rtt, places=9)
            self.assertAlmostEqual(p2 - p1, self.rtt / P.PP, places=9)

    def test_two_workers_save_two_rtts(self):
        r = self.r
        L = dict(r["links"], pi5={"mbps": 938.0, "rtt_ms": 0.5, "pair": "test"})
        C = dict(r["C"], pi5=r["C"]["pi4"])
        a = [("pi4", 1), ("pi5", 10), ("jetson", 11)]
        d2, _ = P.predict(a, r["q"], C, L, rtt_trips=2)
        d1, _ = P.predict(a, r["q"], C, L, rtt_trips=1)
        self.assertAlmostEqual(d2 - d1, self.rtt + 0.5, places=9)

    def test_plan_fixed_and_the_energy_model_carry_the_rule(self):
        r1 = P.plan_fixed("tinyllama", [("pi4", 1), ("jetson", 21)], "last", 4096, MEM, rtt_trips=1)
        self.assertEqual(r1["rtt_trips"], 1)
        self.assertAlmostEqual(self.r["decode_ms"] - r1["decode_ms"], self.rtt, places=9)
        # per_node asserts that its terms sum to predict() with the same rtt_trips
        busy, xfer = EM.per_node(r1["assign"], r1["q"], r1["C"], r1["links"], rtt_trips=1)
        busy2, xfer2 = EM.per_node(r1["assign"], r1["q"], r1["C"], r1["links"])
        self.assertEqual(busy, busy2)
        self.assertAlmostEqual(xfer2["tg"] - xfer["tg"], self.rtt, places=9)
        power = {n: {"tg": 6.0, "pp": 7.0, "idle": 2.5} for n in ("pi4", "jetson")}
        e1 = EM.energy(r1["assign"], r1["q"], r1["C"], r1["links"], power, "tg", rtt_trips=1)
        e2 = EM.energy(r1["assign"], r1["q"], r1["C"], r1["links"], power, "tg")
        self.assertAlmostEqual(e2["T_pred_ms"] - e1["T_pred_ms"], self.rtt, places=9)


if __name__ == "__main__":
    unittest.main()
