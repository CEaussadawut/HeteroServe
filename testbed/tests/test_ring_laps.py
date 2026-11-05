"""Tests for the ring-lap study: its registration's layouts and score_ring_laps_20261009.py, on synthetic logs, stdlib
unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import copy, json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402
import register_ring_laps_20261009 as R                       # noqa: E402
import score_ring_laps_20261009 as S                          # noqa: E402

SHAPES = json.load(open(os.path.join(ROOT, "testbed", "dataset", "model_shapes.json")))
Q = SHAPES[P.MODELS["tinyllama"][0]]
REG = R.build()


def synth(c_true=1.0, per_lap=1.5, rtt0=1.1, added=5.0, base=None):
    """cells and pings as parse() returns them: decode = base + per_lap (k - 1), + c_true k added at d5."""
    base = base if base is not None else REG["layouts"]["k1"]["predicted"]["s0_rtt1"]["decode_ms"] * 1.02
    cells, pings, t = {}, {}, 1000.0
    for r in range(1, REG["rounds"] + 1):
        pings[(r, "s0")], pings[(r, "d5")] = rtt0, rtt0 + added
        for st in ("s0", "d5"):
            for lab, v in REG["layouts"].items():
                k = v["jetson_visits"]
                dec = base + per_lap * (k - 1) + (c_true * k * added if st == "d5" else 0.0) + 0.05 * r
                cells[(r, st, lab)] = {"t0": t, "t1": t + 30, "tg": dec, "pp": 26.0 + 0.1 * k, "exit": 0,
                                       "bufs": S.expected_buffers(v, Q)}
                t += 40
    return cells, pings


class TestRegistration(unittest.TestCase):
    def test_every_layout_maps_each_block_to_its_stage_and_the_head_last(self):
        for lab, stages in R.LAYOUTS.items():
            lm = R.layer_map(stages, Q["n_layers"])
            self.assertEqual(lm, [i for i, (_, k) in enumerate(stages) for _ in range(k)] + [len(stages) - 1], lab)

    def test_only_the_visits_differ(self):
        self.assertEqual([v["jetson_visits"] for v in REG["layouts"].values()], [1, 2, 3, 4])
        p = REG["predicted_per_lap_s0_ms"]
        for t in (1, 2):
            d = [v["predicted"]["s0_rtt%d" % t]["decode_ms"] for v in REG["layouts"].values()]
            steps = [b - a for a, b in zip(d, d[1:])]
            self.assertLess(max(steps) - min(steps), 0.02)          # uniform: the same bytes on each board
            self.assertAlmostEqual(statistics_mean(steps), p["s0_rtt%d" % t], delta=0.01)

    def test_the_d5_predictions_add_the_round_trips_per_visit(self):
        for v in REG["layouts"].values():
            k = v["jetson_visits"]
            for t in (1, 2):
                self.assertAlmostEqual(v["predicted"]["d5_rtt%d" % t]["decode_ms"] - v["predicted"]["s0_rtt%d" % t]["decode_ms"],
                                       t * k * R.D5_ADDED_RTT_MS, delta=0.02)


def statistics_mean(xs):
    return sum(xs) / len(xs)


class TestScore(unittest.TestCase):
    def test_one_trip_per_visit_holds_rtt1(self):
        s = S.score(REG, Q, *synth(c_true=1.0))
        self.assertEqual(s["hypotheses"]["RL-RTT"]["verdict"], "RTT1 HOLDS")
        self.assertAlmostEqual(s["hypotheses"]["RL-RTT"]["c"], 1.0, places=3)
        self.assertEqual(s["hypotheses"]["RL-SEQ"]["verdict"], "HOLDS")
        self.assertAlmostEqual(s["hypotheses"]["RL-SEQ"]["slope_ms_per_lap"], 1.5, places=3)
        self.assertEqual(s["hypotheses"]["RL-ACC"]["verdict"], "HOLDS")
        self.assertAlmostEqual(s["exploratory"]["per_visit_overhead_beyond_rtt1_ms"], 1.5 - REG["predicted_per_lap_s0_ms"]["s0_rtt1"],
                               places=2)

    def test_two_trips_and_in_between(self):
        self.assertEqual(S.score(REG, Q, *synth(c_true=2.0))["hypotheses"]["RL-RTT"]["verdict"], "2-TRIP MODEL STANDS")
        self.assertEqual(S.score(REG, Q, *synth(c_true=1.5))["hypotheses"]["RL-RTT"]["verdict"], "NEITHER")

    def test_laps_that_cost_nothing_or_less_fail_rl_seq(self):
        self.assertEqual(S.score(REG, Q, *synth(per_lap=-0.5))["hypotheses"]["RL-SEQ"]["verdict"], "FAILS")

    def test_a_wrong_layout_or_a_failed_bench_is_left_out(self):
        cells, pings = synth()
        cells[(2, "s0", "k3")]["bufs"] = dict(cells[(2, "s0", "k3")]["bufs"])
        ep = next(iter(cells[(2, "s0", "k3")]["bufs"]))
        cells[(2, "s0", "k3")]["bufs"][ep] += 2.0
        cells[(1, "d5", "k2")]["exit"] = 1
        s = S.score(REG, Q, cells, pings)
        self.assertEqual(set(s["left_out"]), {"2.s0.k3", "1.d5.k2"})
        self.assertEqual(s["hypotheses"]["RL-RTT"]["verdict"], "INCOMPLETE")
        self.assertEqual(s["hypotheses"]["RL-SEQ"]["verdict"], "INCOMPLETE")

    def test_parse_reads_the_runner_log(self):
        v = REG["layouts"]["k2"]
        exp = S.expected_buffers(v, Q)
        log = ["PING round=1 setting=s0 rtt min/avg/max/mdev = 0.9/1.10/1.4/0.1 ms",
               "##### CELL round=1 setting=s0 layout=k2 start=1000.5"]
        log += ["load_tensors: RPC%d[%s] model buffer size =   %.2f MiB" % (i, ep, mib) for i, (ep, mib) in enumerate(exp.items())]
        log += [json.dumps({"build_commit": "b23701f7", "n_prompt": 64, "n_gen": 0, "samples_ns": [1.6e9, 1.7e9, 1.65e9]}),
                json.dumps({"build_commit": "b23701f7", "n_prompt": 0, "n_gen": 32, "samples_ns": [3.4e9, 3.5e9, 3.45e9]}),
                "EXIT_BENCH=0", "##### CELL_END round=1 setting=s0 layout=k2 end=1030.25 ssh_exit=0"]
        cells, pings = S.parse(log)
        c = cells[(1, "s0", "k2")]
        self.assertEqual((c["t0"], c["t1"], c["exit"]), (1000.5, 1030.25, 0))
        self.assertAlmostEqual(c["tg"], 3450.0 / 32, places=6)
        self.assertAlmostEqual(c["pp"], 1650.0 / 64, places=6)
        self.assertEqual(pings, {(1, "s0"): 1.10})
        self.assertEqual(set(c["bufs"]), set(exp))
        self.assertTrue(all(abs(c["bufs"][ep] - exp[ep]) < 0.01 for ep in exp))


if __name__ == "__main__":
    unittest.main()
