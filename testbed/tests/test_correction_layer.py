"""The correction layer (upgrade 2, 2026-10-10): the error table from the record, and the fit, leave-one-family-out and
conformal pieces on synthetic rows (the real table is fitted only by score_correction_20261010.py, after registration).
stdlib unittest (and NumPy, as the fit uses it).

    python3 -m unittest discover -s testbed/tests -q
"""
import json, math, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import correction_layer as CL                                 # noqa: E402


class TestTable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.skipped = CL.table()
        cls.by_id = {r["id"]: r for r in cls.rows}

    def test_the_sources_and_the_exclusions(self):
        import collections
        self.assertEqual(dict(collections.Counter(r["source"] for r in self.rows)),
                         {"pool": 70, "laps": 24, "visit": 18, "served": 6})
        why = {s["id"]: s["why"] for s in self.skipped}
        self.assertEqual(why["pool_results_20260928.json:qwen3moe.rule"], "exit 134")
        self.assertEqual(sum(1 for k in why if k.startswith("pool_results_20261009tcpdiag.json:")), 4)
        self.assertEqual(len(self.skipped), 7)

    def test_every_row_is_finite_and_its_target_is_its_own_ratio(self):
        for r in self.rows:
            for f in CL.FEATURES + ("y",):
                self.assertTrue(math.isfinite(r[f]), (r["id"], f))
            self.assertAlmostEqual(r["y"], r["measured_ms"] / r["predicted_ms"] - 1.0)
            self.assertIn(r["temp_source"], ("poll", "state", "no pi"))

    def test_a_served_14b_row(self):
        r = self.by_id["full_eval_q14b_20261009_20261009_204531:engine"]
        self.assertEqual((r["measured_ms"], r["predicted_ms"], r["visits"], r["pis"]), (1976.48, 2025.7, 1.0, "pi4"))       # summary.json; benchmarks.md rounds it
        q = CL.shapes()["qwen3-14b-q4_k_m"]
        w = sum(q["layer_bytes"]) + q["head_bytes"]
        self.assertAlmostEqual(r["kv_frac"], 4096 * (128 + 31 / 2.0) * 40 / w)    # 128-token prompts, 32 tokens
        self.assertEqual((r["hybrid"], r["moe"]), (0.0, 0.0))
        self.assertEqual(r["pi_temp"], max(0.0, r["pi_temp_c"] - 60.0))

    def test_the_hybrid_and_moe_rows_and_their_kv_layers(self):
        sh = CL.shapes()
        self.assertEqual(CL.kv_layers(sh["qwen3.5-9b-q4_k_m"]), 8)                 # the shapes' notes: 8 full-attention
        self.assertEqual(CL.kv_layers(sh["qwen3.8-27b-q3_k_m"]), 16)               # and 16
        self.assertEqual(CL.kv_layers(sh["qwen3-14b-q4_k_m"]), 40)
        for r in self.rows:
            self.assertEqual(r["hybrid"], 1.0 if r["family"] in ("qwen3.5-9b", "qwen3.8-27b") else 0.0, r["id"])
            self.assertEqual(r["moe"], 1.0 if r["family"] == "qwen3-30b-a3b" else 0.0, r["id"])

    def test_visits_count_the_rpc_stages(self):
        self.assertEqual(self.by_id["ring_laps_20261009_201631:1.s0.k4"]["visits"], 8.0)        # 4 laps: P and J each
        self.assertEqual(self.by_id["ring_laps_20261009_235345:1.s0.h4"]["visits"], 4.0)        # the host's own blocks: no
        self.assertEqual(self.by_id["ring_laps_20261009_235345:1.s0.p4"]["visits"], 5.0)
        self.assertEqual(self.by_id["full_eval_tiny_pilot_20261009_20261009_190242:engine"]["visits"], 0.0)   # Jetson alone
        self.assertEqual(self.by_id["full_eval_tiny_pilot_20261009_20261009_190242:engine"]["temp_source"], "no pi")
        self.assertEqual(self.by_id["pool_results_20260930default.json:d.q3_14b.default"]["visits"], 3.0)  # all RPC

    def test_the_table_is_deterministic(self):
        rows, _ = CL.table()
        self.assertEqual(CL.table_sha256(rows), CL.table_sha256(self.rows))


class TestFit(unittest.TestCase):
    def _rows(self, coef, fams=("a", "b", "c", "d"), n=12, noise=0.0):
        import random
        rnd, out = random.Random(7), []
        for f in fams:
            for i in range(n):
                r = {"id": "%s%d" % (f, i), "family": f, "predicted_ms": 100.0, "hybrid": float(f == "b"),
                     "moe": float(f == "c"), "kv_frac": rnd.uniform(0, 0.01), "pi_temp": rnd.uniform(0, 20),
                     "visits": float(rnd.randint(0, 4))}
                y = coef[0] + sum(c * r[k] for c, k in zip(coef[1:], CL.FEATURES)) + rnd.gauss(0, noise)
                r["y"], r["measured_ms"] = y, 100.0 * (1 + y)
                out.append(r)
        return out

    def test_ols_recovers_known_coefficients(self):
        coef = (0.02, 0.05, -0.03, 2.0, 0.001, 0.004)
        m = CL.fit(self._rows(coef))
        self.assertAlmostEqual(m["intercept"], 0.02, places=9)
        for c, f in zip(coef[1:], CL.FEATURES):
            self.assertAlmostEqual(m["coef"][f], c, places=9)
        self.assertEqual(m["dropped"], [])

    def test_a_feature_constant_in_the_fit_is_left_out(self):
        rows = [r for r in self._rows((0.02, 0.05, -0.03, 2.0, 0.001, 0.004)) if r["family"] != "c"]   # no MoE row left
        m = CL.fit(rows)
        self.assertEqual(m["dropped"], ["moe"])
        self.assertEqual(m["coef"]["moe"], 0.0)

    def test_lofo_corrects_each_family_with_the_others_fit(self):
        rows = self._rows((0.03, 0.0, 0.0, 0.0, 0.0, 0.01))          # a constant 3 % plus 1 % per visit, every family
        out, fits = CL.lofo(rows)
        self.assertEqual(sorted(fits), ["a", "b", "c", "d"])
        for r in out:
            self.assertAlmostEqual(r["e_corr"], 0.0, places=9)      # the shared structure transfers across families
        self.assertLess(CL.mae([r["e_corr"] for r in out]), CL.mae([r["y"] for r in out]))

    def test_conformal_quantile_and_cv_scores(self):
        self.assertAlmostEqual(CL.conformal_q([0.1 * i for i in range(1, 10)], 0.10), 0.9)   # n = 9: rank ceil(10 x 0.9) = 9
        self.assertEqual(CL.conformal_q([0.1 * i for i in range(1, 9)], 0.10), math.inf)    # n = 8: rank 9 > 8
        s = [i / 100.0 for i in range(1, 100)]                        # 99 scores: the 90th smallest
        self.assertAlmostEqual(CL.conformal_q(s, 0.10), 0.90)
        rows = self._rows((0.03, 0.0, 0.0, 0.0, 0.0, 0.01), noise=0.01)
        cv = CL.cv_scores(rows)
        self.assertEqual(len(cv), len(rows))
        self.assertTrue(all(x >= 0 for x in cv))
        self.assertEqual(CL.fold_of({"id": "x"}), CL.fold_of({"id": "x", "y": 9.9}))      # the fold depends on the id only


class TestScorer(unittest.TestCase):
    def _reg(self, rows):
        return {"table": {"rows": rows, "sha256": CL.table_sha256(rows)}}

    def test_the_verdict_follows_the_registered_rule(self):
        import score_correction_20261010 as SC
        fit = TestFit()
        shared = fit._rows((0.03, 0.0, 0.0, 0.0, 0.0, 0.01), fams=("a", "b", "c", "qwen2.5-14b"), noise=0.001)
        _, s = SC.score(self._reg(shared))
        self.assertEqual(s["hypotheses"]["CORR-LOFO"]["verdict"], "HOLDS")      # shared structure: the error collapses
        self.assertEqual(s["adoption"], "ADOPT (default-off option)")
        self.assertIn("without_qwen2.5-14b", s)
        self.assertEqual(s["hypotheses"]["CORR-COVER"]["verdict"], "PENDING")
        noise = fit._rows((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), noise=0.03)           # nothing to learn
        _, s = SC.score(self._reg(noise))
        self.assertEqual(s["hypotheses"]["CORR-LOFO"]["verdict"], "FAILS")
        self.assertEqual(s["adoption"], "NOT ADOPTED")

    def test_a_tampered_table_is_refused(self):
        import score_correction_20261010 as SC
        rows = TestFit()._rows((0.03, 0.0, 0.0, 0.0, 0.0, 0.01))
        reg = self._reg(rows)
        reg["table"]["rows"][0]["y"] += 0.5
        with self.assertRaises(SystemExit):
            SC.score(reg)


if __name__ == "__main__":
    unittest.main()
