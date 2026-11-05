"""The token embedding charged to a host that reads its weights into memory (2026-10-09), through predict_pool, the live
gate of serve_pool.sh and full_eval's registration; stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import argparse, contextlib, io, json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402
import full_eval as F                                         # noqa: E402

MEM = json.load(open(os.path.join(ROOT, "testbed", "dataset", "memory_at_rest.json")))
P32 = [("pi4", 32), ("jetson", 8)]


def need(r):
    return P.memory(r["assign"], r["q"], MEM, r["devices"], r["model"], head_at=r["head_at"])


class TestLoadedHost(unittest.TestCase):
    def test_the_default_prices_the_mapped_launch_as_every_registration_did(self):
        r = P.plan_fixed("qwen3_14b", P32, "host", 4096, MEM)
        self.assertFalse(r["model"].host_loads_weights)
        self.assertEqual(need(r)["pi4"]["need_mib"], 7254)           # what the 2026-10-09 gate printed

    def test_load_mode_none_charges_the_embedding_on_the_host_only(self):
        a, b = (P.plan_fixed("qwen3_14b", P32, "host", 4096, MEM, load_mode=m) for m in (None, "none"))
        self.assertTrue(b["model"].host_loads_weights)
        ma, mb = need(a), need(b)
        self.assertEqual(mb["pi4"]["need_mib"] - ma["pi4"]["need_mib"], 417)   # 437,575,680 bytes
        self.assertEqual(mb["jetson"], ma["jetson"])
        self.assertLess(mb["pi4"]["margin_mib"], 0)                  # p32host does not fit the Pi 4 at rest
        self.assertEqual((a["decode_ms"], a["prefill_ms"]), (b["decode_ms"], b["prefill_ms"]))   # memory only

    def test_the_engine_keeps_its_pick(self):
        a = P.plan_host_choice("qwen3_14b", ["pi4", "jetson"], 4096, MEM)
        b = P.plan_host_choice("qwen3_14b", ["pi4", "jetson"], 4096, MEM, load_mode="none")
        self.assertEqual((b["assign"], b["head_at"]), ([("pi4", 27), ("jetson", 13)], 0))
        self.assertEqual((a["assign"], a["head_at"]), (b["assign"], b["head_at"]))
        self.assertEqual(need(b)["pi4"]["need_mib"], 6658)

    def test_serve_pool_gates_on_the_loaded_charge_and_logs_memory_at_ready(self):
        with open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")) as f:
            s = f.read()
        self.assertEqual(s.count('load_mode="none"'), 2)             # the fixed placement and the engine's own
        self.assertEqual(s.count(" -lm none "), 2)                   # what it serves: the echo and the launch
        self.assertIn('echo "mem     $n: MemAvailable $a MiB at READY', s)

    def _register(self, placements, below=False):
        a = argparse.Namespace(name="unittest_never_written", model="qwen3_14b", nodes="pi4 jetson", placements=placements,
                               rounds=2, reqs=2, n=128, gen=32, ctx=4096, rtt_trips=1, warm_n=64, warm_gen=8, note="",
                               below_headroom=below, dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            return F.register(a)

    def test_registration_refuses_what_does_not_fit_and_flags_what_is_tight(self):
        with self.assertRaises(SystemExit) as e:
            self._register("engine,p32host=pi4:32+jetson:8/host", below=True)
        self.assertIn("p32host: pi4 needs 7671", str(e.exception))
        with self.assertRaises(SystemExit) as e:
            self._register("engine,p32last=pi4:32+jetson:8/last")
        self.assertIn("p32last: jetson keeps 149 MiB", str(e.exception))
        reg = self._register("engine,p32last=pi4:32+jetson:8/last", below=True)
        self.assertEqual((reg["load_mode"], reg["below_headroom"]), ("none", True))
        self.assertEqual(reg["configs"]["engine"]["memory"]["pi4"]["need_mib"], 6658)
        self.assertEqual(reg["configs"]["p32last"]["memory"]["pi4"]["need_mib"], 7062)
        self.assertFalse(os.path.exists(os.path.join(ROOT, "testbed", "dataset", "full_eval_unittest_never_written.json")))
        self.assertNotIn("below_headroom", self._register("engine"))


if __name__ == "__main__":
    unittest.main()
