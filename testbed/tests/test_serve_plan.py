"""serve_plan_20261101.py, interactive serving's plan (the scheduler review's item 6, 2026-11-01): live memory with a fallback
for a board that does not answer, the 512 MiB grid, and keeping the plan served last. Contacts no board (reads are faked).

    python3 -m unittest discover -s testbed/tests -q
"""
import copy, json, os, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import serve_plan_20261101 as SP                               # noqa: E402
import predict_pool as P                                       # noqa: E402

SNAP = os.path.join(ROOT, "testbed", "dataset", "memory_snapshot_20261028_033438.json")
N4 = ["laptop", "pi5", "pi4", "jetson"]


class Memory(unittest.TestCase):
    def test_live_with_a_fallback_for_a_board_that_is_off(self):
        snap = json.load(open(SNAP))
        reads = {"laptop": 26201, "pi5": 7600, "pi4": None, "jetson": 3500}
        mem, src = SP.live_memory(N4, snap, read=lambda n, k: reads[n])
        self.assertEqual(mem["laptop"]["mem_available_mib"], 26201)
        self.assertEqual(src["pi4"], "snapshot (no answer)")
        self.assertEqual(mem["pi4"]["mem_available_mib"], snap["pi4"]["mem_available_mib"])
        with self.assertRaises(ValueError):
            SP.live_memory(["pi4"], {"_source": "x"}, read=lambda n, k: None)

    def test_the_grid_rounds_down(self):
        g = SP.on_grid({"laptop": {"mem_available_mib": 26201}, "pi5": {"mem_available_mib": 7680}, "_source": "x"})
        self.assertEqual(g["laptop"]["mem_available_mib"], 26112)                  # 51 x 512
        self.assertEqual(g["laptop"]["mem_available_raw_mib"], 26201)
        self.assertEqual(g["pi5"]["mem_available_mib"], 7680)
        self.assertEqual(SP.on_grid(g)["laptop"]["mem_available_mib"], 26112)      # idempotent: it keeps the raw figure
        self.assertEqual(g["_source"], "x")

    def test_the_grid_cuts_the_32b_flips(self):
        # the review: 13 plan changes over 25.6-28.2 GB of the laptop's free memory, swept every 128 MiB
        os.environ["HETEROSERVE_RESERVE_MB"] = ""
        snap = json.load(open(SNAP))
        def plans(grid):
            seen, prev, flips = [], None, 0
            for lap in range(25600, 28200, 128):
                m = copy.deepcopy(snap)
                m["laptop"]["mem_available_mib"] = lap
                if grid:
                    m = SP.on_grid(m)
                r = P.plan_host_choice("qwen3_32b_q6k", N4, 2048, m, host="laptop", load_mode="none", **P.PLANNER_V3)
                sig = (tuple(r["assign"]), r["head_at"])
                flips += prev is not None and sig != prev
                prev = sig
            return flips
        self.assertGreaterEqual(plans(False), 10)
        self.assertLessEqual(plans(True), 6)                                       # one change at most per grid line


class Keep(unittest.TestCase):
    def setUp(self):
        os.environ["HETEROSERVE_RESERVE_MB"] = ""
        self.mem = SP.on_grid(json.load(open(SNAP)))
        self.key, self.ctx = "qwen3_32b_q6k", 2048

    def last_of(self, assign, head, decode=0.0):
        return {"key": self.key, "ctx": self.ctx, "nodes": N4, "depth": 0, "assign": [list(x) for x in assign],
                "head": head, "decode_ms": decode, "served": "2026-11-01T00:00:00+0700"}

    def test_no_last_plan_plans(self):
        r, why = SP.choose(self.key, N4, self.ctx, self.mem, 0, None, **P.PLANNER_V3)
        self.assertIn("planned", why)

    def test_keeps_the_plan_of_one_grid_line_down(self):
        # the plan served when the laptop had one grid step (512 MiB) less: it still fits now; where the new pick is within 2 %
        # it stays, else the new pick is served. On this snapshot one 512 MiB step is worth 1.9-9.1 % to the 32B Q6_K at ctx
        # 4,096 (2026-11-01; a 128 MiB step 0-1.6 %, the review), so the grid does most of the steadying and the 2 % the rest
        self.ctx, kept = 4096, 0
        for lap in range(25600, 28160 + 1, 512):
            now = copy.deepcopy(self.mem); now["laptop"]["mem_available_mib"] = lap
            before = copy.deepcopy(self.mem); before["laptop"]["mem_available_mib"] = lap - 512
            b, _ = SP.choose(self.key, N4, self.ctx, before, 0, None, **P.PLANNER_V3)
            n, _ = SP.choose(self.key, N4, self.ctx, now, 0, None, **P.PLANNER_V3)
            if (b["assign"], b["head_at"]) == (n["assign"], n["head_at"]):
                continue
            r, why = SP.choose(self.key, N4, self.ctx, now, 0, self.last_of(b["assign"], "host" if b["head_at"] == 0 else "last"),
                               **P.PLANNER_V3)
            old = P.plan_fixed(self.key, b["assign"], "host" if b["head_at"] == 0 else "last", self.ctx, now, load_mode="none",
                               **P.PLANNER_V3)
            if old["decode_ms"] <= n["decode_ms"] * 1.02:
                self.assertTrue(why.startswith("kept"), (lap, why))
                self.assertEqual([list(x) for x in r["assign"]], [list(x) for x in b["assign"]])
                kept += 1
            else:
                self.assertTrue(why.startswith("planned: the new pick"), (lap, why))
        self.assertGreater(kept, 0)

    def test_replans_when_the_last_plan_no_longer_fits(self):
        tight = copy.deepcopy(self.mem)
        tight["laptop"]["mem_available_mib"] -= 4096                                # the laptop lost 4 GiB since
        new, _ = SP.choose(self.key, N4, self.ctx, self.mem, 0, None, **P.PLANNER_V3)
        last = self.last_of(new["assign"], "host" if new["head_at"] == 0 else "last")
        r, why = SP.choose(self.key, N4, self.ctx, tight, 0, last, **P.PLANNER_V3)
        self.assertTrue("no longer fits" in why or "same plan" in why, why)

    def test_recall_matches_model_context_and_boards(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            SP.remember(self.last_of([("laptop", 64)], "last"), path=p)
            self.assertIsNotNone(SP.recall(self.key, self.ctx, list(reversed(N4)), path=p))
            self.assertIsNone(SP.recall(self.key, 4096, N4, path=p))
            self.assertIsNone(SP.recall(self.key, self.ctx, ["laptop", "pi5"], path=p))
            self.assertIsNone(SP.recall(self.key, self.ctx, N4, path=os.path.join(d, "none.json")))

    def test_plan_depth(self):
        self.assertEqual(SP.plan_depth(4096, {}), 0)                                # v4d: depth priced only when asked
        self.assertEqual(SP.plan_depth(4096, {"PLAN_DEPTH": "0"}), 0)
        self.assertEqual(SP.plan_depth(8192, {"PLAN_DEPTH": "1000"}), 1000)


class Wiring(unittest.TestCase):
    def test_serve_pool_plans_v4_for_a_person_and_v3_for_a_harness(self):
        import subprocess
        defs = os.path.join(ROOT, "testbed", "scripts", "serve_defaults.sh")
        b = lambda c: subprocess.run(["bash", "-c", ". '%s'; %s" % (defs, c)], capture_output=True, text=True).stdout.strip()
        self.assertEqual(b("planner_mode __unset__ 1"), "v4")
        self.assertEqual(b("planner_mode __unset__ 0"), "v3")
        self.assertEqual(b("planner_mode shipped 1"), "shipped")
        self.assertEqual(b("planner_mode v4 0"), "v4")
        self.assertEqual(b("planner_mode yes 1"), "invalid")
        s = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")).read()
        self.assertLess(s.index('SERVE_PLANNER_MODE=$(planner_mode'), s.index("exec > >("))
        self.assertIn('elif mode == "v4":', s)
        self.assertLess(s.index('if [ "$bad" = 0 ]') if 'if [ "$bad" = 0 ]' in s else s.index('[ "$bad" = 0 ] ||'),
                        s.index("SP.remember("))                                    # remembered only once the split checks


if __name__ == "__main__":
    unittest.main()
