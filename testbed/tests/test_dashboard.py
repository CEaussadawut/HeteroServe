"""Tests for testbed/scripts/dashboard.py's data extraction (read-only), stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, shutil, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import dashboard as D                                         # noqa: E402

REAL = os.path.join(ROOT, "testbed", "results", "energy_pool_20261008_011447")


class TestDashboard(unittest.TestCase):
    def test_power_from_a_real_strip_log(self):
        if not os.path.exists(os.path.join(REAL, "strip.csv")):
            self.skipTest("the 10-08 strip log is not in this checkout")
        p = D.power(REAL)
        self.assertTrue(0 < len(p["t"]) <= D.MAX_POINTS)
        self.assertEqual(sorted(p["outlets"]), [0, 1, 2, 3])
        self.assertLessEqual(p["t"][-1] - p["t"][0], D.WINDOW_S + 1)
        self.assertTrue(all(len(v) == len(p["t"]) for v in p["outlets"].values()))

    def test_requests_from_a_run_log(self):
        d = tempfile.mkdtemp()
        try:
            ev = [[0.20, [1], 1, 10.0, 15.0]] + [[0.20 + 0.1 * k, [k + 1], k + 1, 10.0 + 95.0 * k, 15.0] for k in range(1, 20)]
            rows = [{"t": 1.0, "kind": "serve start", "data": {"round": 1, "placement": "engine"}},
                    {"t": 2.0, "kind": "req", "data": {"placement": "engine", "round": 1, "i": 0, "sent_wall": 1000.0,
                                                       "done_s": 2.2, "events": ev}}]
            with open(os.path.join(d, "run.jsonl"), "w") as f:
                f.write("\n".join(json.dumps(r) for r in rows) + "\n{\"t\": 3, \"kind\"")      # a line still being written
            r = D.requests(d)
            self.assertEqual(len(r["requests"]), 1)
            self.assertEqual(r["requests"][0]["tpot_p50"], 95.0)
            self.assertEqual(r["per_placement"], {"engine": {"requests": 1, "median_tpot_ms": 95.0}})
            self.assertEqual(r["last"]["kind"], "req")
        finally:
            shutil.rmtree(d)


if __name__ == "__main__":
    unittest.main()
