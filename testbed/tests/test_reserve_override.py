"""Tests for a registration's own memory reserve (predict_pool.reserve_overrides, HETEROSERVE_RESERVE_MB; 2026-10-21: the
user's 7 GiB laptop reserve for the Qwen3-32B runs only), stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "ThinkPad"))
import predict_pool as P                                      # noqa: E402


class ReserveOverride(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("HETEROSERVE_RESERVE_MB", None)

    def tearDown(self):
        os.environ.pop("HETEROSERVE_RESERVE_MB", None)
        if self.saved is not None:
            os.environ["HETEROSERVE_RESERVE_MB"] = self.saved

    def test_unset_keeps_cluster_json(self):
        b = P.board_memory()
        self.assertEqual(b["laptop"][3], 8192.0)
        self.assertEqual(b["t580"][3], 8192.0)
        self.assertEqual(b["jetson"][3], 450.0)

    def test_empty_keeps_cluster_json(self):
        os.environ["HETEROSERVE_RESERVE_MB"] = ""
        self.assertEqual(P.board_memory()["laptop"][3], 8192.0)

    def test_planner_id_sets_both_names_and_nothing_else(self):
        os.environ["HETEROSERVE_RESERVE_MB"] = "laptop=7168"
        b = P.board_memory()
        self.assertEqual(b["laptop"][3], 7168.0)
        self.assertEqual(b["t580"][3], 7168.0)
        self.assertEqual(b["jetson"][3], 450.0)
        self.assertEqual(b["pi5"][3], 0.0)

    def test_cluster_id_works_too(self):
        os.environ["HETEROSERVE_RESERVE_MB"] = "t580=7168, pi4=100"
        b = P.board_memory()
        self.assertEqual(b["laptop"][3], 7168.0)
        self.assertEqual(b["pi4"][3], 100.0)

    def test_unknown_node_is_refused(self):
        os.environ["HETEROSERVE_RESERVE_MB"] = "lapotp=7168"
        with self.assertRaises(ValueError):
            P.board_memory()

    def test_malformed_is_refused(self):
        for s in ("laptop", "laptop=", "=7168"):
            os.environ["HETEROSERVE_RESERVE_MB"] = s
            with self.assertRaises(ValueError):
                P.board_memory()

    def test_plan_feels_the_override(self):
        """The laptop's margin in P.memory moves by exactly the reserve's difference."""
        mem = {"laptop": {"mem_available_mib": 26432}}
        q = {"layer_bytes": [100 * 2 ** 20] * 4, "head_bytes": 0}

        class D:                                               # the attributes P.memory reads from a device profile
            memory_reserve_mb = None
            max_weights_mb = None

        def margin():
            d = D()
            d.memory_reserve_mb = P.board_memory()["laptop"][3]
            orig = P.memory_need_mb
            P.memory_need_mb = lambda *a, **k: 1000.0
            try:
                return P.memory([("laptop", 4)], q, mem, {"laptop": d}, None)["laptop"]["margin_mib"]
            finally:
                P.memory_need_mb = orig
        base = margin()
        os.environ["HETEROSERVE_RESERVE_MB"] = "laptop=7168"
        self.assertEqual(margin() - base, 1024)


if __name__ == "__main__":
    unittest.main()
