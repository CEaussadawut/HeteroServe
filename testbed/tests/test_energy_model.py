"""Tests for testbed/scripts/energy_model.py, stdlib unittest only. From the repo root:

    python3 -m unittest discover -s testbed/tests -q

The model is checked against what was measured: the strip's pooled runs of 2026-10-05/06 (validate()), and the per-board
split of predict_pool.predict's own terms (per_node() must sum to it). Skipped where the measured files are not present.
"""
import contextlib, io, json, os, statistics, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import energy_model as EM                                     # noqa: E402
import predict_pool as P                                      # noqa: E402

HAVE = all(os.path.exists(os.path.join(ROOT, p)) for p in EM.POOL_SCORES) and \
    all(os.path.exists(os.path.join(EM.DS, f)) for f in EM.POWER_RUNS.values())


@unittest.skipUnless(HAVE, "the measured energy files are not here")
class TestEnergyModel(unittest.TestCase):
    def test_power_table_reads_every_board(self):
        t = EM.power_table()
        self.assertEqual(set(t), {"pi5", "pi4", "jetson", "laptop"})
        for b, v in t.items():
            self.assertLess(v["idle"], v["tg"])               # a board draws more computing than idling
            self.assertLess(v["idle"], v["pp"])
        self.assertAlmostEqual(t["jetson"]["tg"], 11.17, places=1)   # energy.md: the Jetson GPU decoding

    def test_per_node_sums_to_predict(self):
        shapes, pairs, C = EM._inputs("Q4_K_M")
        for shape, assign, head_at in (("tinyllama-1.1b-q4km", [("pi4", 1), ("jetson", 21)], None),
                                       ("tinyllama-1.1b-q4km", [("pi4", 1), ("jetson", 21)], 0),
                                       ("qwen3-14b-q4_k_m", [("pi4", 27), ("jetson", 13)], 0)):
            q = shapes[shape]
            L, _ = P.link_view(pairs, assign[0][0], [n for n, _ in assign])
            busy, xfer = EM.per_node(assign, q, C, L, head_at)
            dec, pre = P.predict(assign, q, C, L, head_at=head_at)
            self.assertAlmostEqual(sum(b["tg"] for b in busy.values()) + xfer["tg"], dec, places=6)
            self.assertAlmostEqual(sum(b["pp"] for b in busy.values()) + xfer["pp"], pre, places=6)

    def test_the_model_reproduces_the_measured_pooled_runs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rows = EM.validate()
        errs = [100 * (r["model_gross"] / r["measured_gross"] - 1) for r in rows]
        self.assertEqual(len(rows), 16)
        self.assertLess(statistics.fmean(abs(e) for e in errs), 5.0)   # 3.4 % on 2026-10-08
        # all but the run that ended in the Jetson's hang (session 1, run 3: 12.8 W against 10.2) within 6 %
        self.assertEqual(sum(1 for e in errs if abs(e) > 6.0), 2)


if __name__ == "__main__":
    unittest.main()
