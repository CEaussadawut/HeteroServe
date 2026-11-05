"""Scheduler upgrade 1, step 1 (2026-10-10): the pass-time registration's models, run_pool.sh's bench.tool, and the
scorer's verdicts on synthetic logs; stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import register_spec_pass_20261010 as RS                      # noqa: E402
import score_spec_pass_20261010 as SS                         # noqa: E402


def line(pl, ms_per_pass, tg=8):
    return json.dumps({"n_kv_max": 512, "pp": 8, "tg": tg, "pl": pl, "t_pp": 7.0, "t_tg": ms_per_pass * tg / 1000.0,
                       "speed_tg": pl * 1000.0 / ms_per_pass})


class TestSpecPass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = RS.build()
        cls.pk = cls.reg["configs"][RS.CELL]["predicted"]["per_k"]

    def test_both_models_agree_at_one_token_and_bound_each_other_above_it(self):
        dec = self.reg["configs"][RS.CELL]["predicted"]["decode_ms"]
        self.assertAlmostEqual(self.pk["1"]["add_ms"], dec, delta=0.1)
        self.assertAlmostEqual(self.pk["1"]["roof_ms"], dec, delta=0.5)        # the link's extra bytes only
        for k in ("2", "3", "4", "5"):
            self.assertLess(self.pk[k]["roof_ms"], self.pk[k]["add_ms"])

    def test_the_cell_runs_the_engines_served_plan_with_logits_for_every_token(self):
        c = self.reg["configs"][RS.CELL]
        self.assertEqual([(p["node"], p["blocks"]) for p in c["pipeline"]], [("pi4", 27), ("jetson", 13)])
        self.assertEqual(c["bench"]["tool"], "llama-batched-bench")
        self.assertIn(" --rpc ", c["bench"]["args"])
        self.assertNotIn(" -rpc ", c["bench"]["args"])
        self.assertIn("-lm none", c["bench"]["args"])
        self.assertIn("-npl 1,2,3,4,5,1,2,3,4,5,1,2,3,4,5", c["bench"]["protocol"])
        head = [s for s in c["predicted"]["stages"] if s["head"]]
        self.assertEqual(head[0]["node"], "pi4")

    def test_the_tokens_a_check_yields(self):
        self.assertAlmostEqual(RS.tokens_per_check(1, 0.7), 1.0)
        self.assertAlmostEqual(RS.tokens_per_check(2, 0.7), 1.7)
        self.assertAlmostEqual(RS.tokens_per_check(3, 0.7), 2.19)

    def test_a_roofline_like_log_holds_all_three(self):
        T = {k: self.pk[str(k)]["roof_ms"] * 1.01 for k in RS.KS}
        lines = [line(k, T[k] * f) for f in (0.99, 1.0, 1.01) for k in RS.KS] + ["THROTTLE pi4 max_temp_c=70 verdict=clean",
                                                                                 "EXIT_BENCH=0"]
        s = SS.score(self.reg, SS.parse(lines))
        self.assertEqual({h: v["verdict"] for h, v in s["hypotheses"].items()},
                         {"SP-K1": "HOLDS", "SP-BAND": "HOLDS", "SP-PAYS": "HOLDS"})
        self.assertEqual(s["exit_bench"], 0)
        self.assertEqual(s["thermal"]["pi4"]["verdict"], "clean")
        # the roofline's compute passes the Pi 4's memory floor from k = 3, so the additive form fits it with phi ~ 0.6
        self.assertAlmostEqual(s["exploratory"]["phi"], 0.6, delta=0.05)

    def test_an_additive_log_fits_phi_one(self):
        lines = [line(k, self.pk[str(k)]["add_ms"]) for _ in range(3) for k in RS.KS]
        s = SS.score(self.reg, SS.parse(lines))
        self.assertAlmostEqual(s["exploratory"]["phi"], 1.0, delta=0.01)
        self.assertEqual(s["hypotheses"]["SP-BAND"]["verdict"], "HOLDS")

    def test_a_check_dearer_than_its_yield_fails_pays(self):
        T1 = self.pk["1"]["add_ms"]
        T = {1: T1, 2: 1.75 * T1, 3: 2.3 * T1, 4: 2.6 * T1, 5: 2.8 * T1}     # every ratio above E(k - 1; 0.7)
        lines = [line(k, T[k]) for _ in range(3) for k in RS.KS]
        s = SS.score(self.reg, SS.parse(lines))
        self.assertEqual(s["hypotheses"]["SP-PAYS"]["verdict"], "FAILS")

    def test_a_missing_repetition_makes_the_verdicts_incomplete(self):
        lines = [line(k, self.pk[str(k)]["add_ms"]) for _ in range(3) for k in RS.KS if not (k == 4 and _ == 2)]
        s = SS.score(self.reg, SS.parse(lines))
        self.assertEqual(s["hypotheses"]["SP-BAND"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["hypotheses"]["SP-PAYS"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["hypotheses"]["SP-K1"]["verdict"], "HOLDS")
        self.assertEqual(s["per_k"]["4"]["samples"], 2)

    def test_the_buffer_line_is_read(self):
        d = SS.parse(["load_tensors: RPC0[192.168.1.114:50052] model buffer size =  2489.60 MiB"])
        self.assertEqual(d["buffers"], {"192.168.1.114:50052": 2489.60})

    def test_run_pool_defaults_to_llama_bench_and_knows_the_15_character_name(self):
        src = open(os.path.join(ROOT, "testbed", "scripts", "run_pool.sh")).read()
        self.assertIn('tool = c["bench"].get("tool", "llama-bench")', src)
        self.assertIn("llama-batched-b ", src)                        # pgrep -x sees 15 characters
        self.assertIn("pn=${TOOL:0:15}", src)
        self.assertIn("~/llama.cpp/build/bin/$TOOL $ARGS $PROTO", src)


if __name__ == "__main__":
    unittest.main()
