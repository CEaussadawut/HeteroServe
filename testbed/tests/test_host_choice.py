"""Tests for the host choice (predict_pool.plan_host_choice, holders, bench_args on one GPU board), stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q

Synthetic inventories, so the tests do not depend on what a board holds today; the costs, links and memory are the
dataset's own files.
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402

MEM = json.load(open(os.path.join(ROOT, "testbed", "dataset", "memory_at_rest.json")))


def inv(**holders):
    return {"holders": {k: list(v) for k, v in holders.items()}}


class TestHostChoice(unittest.TestCase):
    def test_tinyllama_goes_to_the_jetson_alone_when_it_holds_the_file(self):
        r = P.plan_host_choice("tinyllama", ["pi4", "jetson"], 4096, MEM, inventory=inv(tinyllama=["pi4", "jetson"]))
        self.assertEqual((r["host"], r["assign"]), ("jetson", [("jetson", 22)]))
        self.assertLess(r["decode_ms"], 90.0)                     # 87.0 predicted; 87.3 measured 2026-10-08
        self.assertEqual({c[0] for c in r["considered"]}, {"pi4", "jetson"})

    def test_only_the_holders_may_host(self):
        r = P.plan_host_choice("tinyllama", ["pi4", "jetson"], 4096, MEM, inventory=inv(tinyllama=["pi4"]))
        self.assertEqual(r["host"], "pi4")
        self.assertEqual(r["assign"], [("pi4", 1), ("jetson", 21)])

    def test_a_named_host_wins_over_the_inventory(self):
        r = P.plan_host_choice("tinyllama", ["pi4", "jetson"], 4096, MEM, host="pi4", inventory=inv(tinyllama=["jetson"]))
        self.assertEqual(r["host"], "pi4")

    def test_without_an_inventory_entry_the_old_default_host_is_used(self):
        with self.assertRaises(ValueError):                       # the old default, the Pi 5, is not among the boards
            P.plan_host_choice("qwen3_14b", ["pi4", "jetson"], 4096, MEM, inventory=inv(qwen3_14b=[]))
        r = P.plan_host_choice("qwen3_14b", ["pi4", "jetson"], 4096, MEM, inventory=inv(qwen3_14b=["pi4"]))
        self.assertEqual((r["host"], r["assign"], r["head_at"]), ("pi4", [("pi4", 27), ("jetson", 13)], 0))

    def test_a_gpu_board_hosts_only_itself(self):
        # qwen3_14b does not fit the Jetson alone; hosted there it would need RPC workers, which bench_args cannot write
        with self.assertRaises(ValueError) as e:
            P.plan_host_choice("qwen3_14b", ["pi4", "jetson"], 4096, MEM, inventory=inv(qwen3_14b=["jetson"]))
        self.assertIn("jetson", str(e.exception))

    def test_bench_args_put_a_lone_gpu_board_on_its_gpu(self):
        self.assertTrue(P.bench_args([("jetson", 22)], "tinyllama")["args"].endswith("-ngl 99"))
        self.assertTrue(P.bench_args([("pi4", 22)], "tinyllama")["args"].endswith("-ngl 0"))     # unchanged for a CPU board


if __name__ == "__main__":
    unittest.main()
