"""control/halda.py: prima.cpp's scheduler, re-implemented, checked against the one decision it made here."""
import unittest

from control import halda


class TestHaldaReproducesPrimaCpp(unittest.TestCase):
    """prima.cpp + Halda on the Pi 5 (rank 0) and Pi 4, Qwen3-14B Q4_K_M, n_ctx 4096, 2026-09-30. Every input below is
    copied from its own log (testbed/results/qr3_primacmpB_qwen3_14b_20260930_061646_prima_rank0.log): its profile
    table and model figures (lines 110-245), its compute buffer (247-250) and its decision (252-283): k = 1, windows
    [24, 16], both devices in M4, objective 689.9 ms."""

    LAYER_FLOPS = {"f32": 273202, "f16": 5242880, "q4k": 566231040, "q6k": 94371840}
    PI5 = {"f32": 38.9, "f16": 76.2, "q4k": 89.8, "q6k": 46.6}
    PI4 = {"f32": 10.0, "f16": 9.0, "q4k": 23.5, "q6k": 17.5}

    def _inputs(self):
        buf = halda.compute_buffer_bytes(n_embd=5120, n_ff=17408, n_head=40, n_ctx=4096)
        costs = {"pi5": halda.layer_ms_from_profile(self.LAYER_FLOPS, self.PI5, 0.02),
                 "pi4": halda.layer_ms_from_profile(self.LAYER_FLOPS, self.PI4, 0.01)}
        caps = halda.memory_caps([5.85, 5.72], layer_bytes=198001664,
                                 kv_layer_bytes=halda.kv_bytes_per_layer(8, 128, 4096), buffer_bytes=[buf, buf],
                                 head_bytes=638151680, embd_row_bytes=437575680 // 151936)
        return buf, costs, dict(zip(["pi5", "pi4"], caps))

    def test_the_compute_buffer_is_prima_cpps_368_mib(self):
        self.assertEqual(halda.compute_buffer_bytes(5120, 17408, 40, 4096), 385878016)   # 368.00 MiB, as logged

    def test_the_caps_bind_the_head_at_24(self):
        _, _, caps = self._inputs()
        self.assertEqual(caps, {"pi5": 24, "pi4": 26})

    def test_it_reproduces_haldas_windows(self):
        _, costs, caps = self._inputs()
        p = halda.plan(["pi5", "pi4"], costs, caps, 40)
        self.assertTrue(p["feasible"])
        self.assertEqual(p["layers"], {"pi5": 24, "pi4": 16})
        self.assertEqual((p["k"], p["removed"]), (1, []))
        # The logged objective adds kappa: the head's layer compute (common.cpp:1152-1164, no KV copy) and reading
        # one embedding row from disk at 0.07 GB/s (line 1167). The rest of the gap is the log's rounding of the
        # profiled GFLOPS to one decimal.
        kappa = halda.layer_ms_from_profile(self.LAYER_FLOPS, self.PI5, 0.0) + 2880 / (0.07e9) * 1000
        self.assertAlmostEqual(p["objective_ms"] + kappa, 689.9, delta=1.0)


class TestHaldaRules(unittest.TestCase):

    def test_a_device_left_with_one_layer_is_removed_and_the_rest_re_solved(self):
        p = halda.plan(["a", "b", "c"], {"a": 1.0, "b": 1.1, "c": 50.0}, {"a": 30, "b": 15, "c": 40}, 40)
        self.assertEqual(p["layers"], {"a": 30, "b": 10})
        self.assertEqual(p["removed"], ["c"])

    def test_the_head_is_never_removed(self):
        p = halda.plan(["a", "b"], {"a": 50.0, "b": 1.0}, {"a": 40, "b": 39}, 40)
        self.assertEqual(p["layers"], {"a": 1, "b": 39})

    def test_shares_that_cannot_fit_are_reported_as_disk_offload(self):
        p = halda.plan(["a", "b"], {"a": 1.0, "b": 2.0}, {"a": 20, "b": 19}, 40)
        self.assertFalse(p["feasible"])
        self.assertIn("disk-offload", p["reason"])

    def test_ties_go_to_the_earlier_device_and_the_head_gets_its_priority(self):
        self.assertEqual(halda.plan(["a", "b"], {"a": 2.0, "b": 2.0}, {"a": 30, "b": 30}, 40)["layers"],
                         {"a": 30, "b": 10})
        # master_priority 1.01 makes the head the cheaper of two equal devices even when it comes second in a sort.
        self.assertEqual(halda.plan(["b", "a"], {"a": 2.0, "b": 2.0}, {"a": 30, "b": 30}, 40)["layers"],
                         {"b": 30, "a": 10})


if __name__ == "__main__":
    unittest.main()
