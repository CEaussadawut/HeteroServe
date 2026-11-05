"""Tests for the placement rule. No hardware needed.

These are about the rule's *decisions*, not its arithmetic: that it respects memory, that
it prefers fast nodes, that it drops a straggler when the straggler does not pay, and that
it keeps the straggler when the model does not fit without it.
"""
import unittest

from control import placement
from control.placement import (DeviceProfile, ModelProfile, compare, evaluate, memory_cap,
                               pareto, plan_even, plan_optimal, plan_single)

MB = 1024 * 1024


def devices(**overrides):
    """A cluster shaped like the real one: one fast node, one straggler, one GPU node."""
    base = {
        "pi5": DeviceProfile("pi5", ms_per_layer_decode=10.0, ms_per_layer_prefill=3.0,
                             usable_mem_mb=7000, power_active_w=7.0, power_idle_w=3.0),
        "pi4": DeviceProfile("pi4", ms_per_layer_decode=30.0, ms_per_layer_prefill=9.0,
                             usable_mem_mb=3200, power_active_w=6.0, power_idle_w=3.0,
                             is_straggler=True),
        "jetson": DeviceProfile("jetson", ms_per_layer_decode=15.0, ms_per_layer_prefill=4.0,
                                usable_mem_mb=2800, power_active_w=9.0, power_idle_w=4.0),
    }
    base.update(overrides)
    return base


def model(n_layers=32, per_layer_mb=120, name="test-7b"):
    return ModelProfile(name=name, n_layers=n_layers,
                        bytes_per_layer=per_layer_mb * MB,
                        kv_bytes_per_layer_per_token=2048,
                        activation_bytes=8192,
                        context_len=2048)


class TestMemoryConstraint(unittest.TestCase):
    def test_cap_counts_kv_cache(self):
        m = model(per_layer_mb=100)
        dev = DeviceProfile("x", 10, 3, usable_mem_mb=1350, power_active_w=5, power_idle_w=2)
        # 1350 - 350 overhead = 1000 MB; each layer is 100 MB of weights plus
        # 2048 B * 2048 tokens = 4 MB of KV, so 9 layers fit, not 10.
        self.assertEqual(memory_cap(dev, m), 9)

    def test_plan_never_exceeds_cap(self):
        devs = devices()
        m = model()
        p = plan_optimal(devs, m)
        self.assertTrue(p.feasible)
        for nid, n in p.layers.items():
            self.assertLessEqual(n, memory_cap(devs[nid], m),
                                 "%s was assigned more layers than it can hold" % nid)

    def test_all_layers_assigned(self):
        m = model()
        p = plan_optimal(devices(), m)
        self.assertEqual(sum(p.layers.values()), m.n_layers)

    def test_model_too_large_for_the_whole_cluster_is_infeasible(self):
        # 200 layers x 120 MB is far beyond the pooled memory; the rule must say so
        # rather than return a plan that would OOM a node.
        p = plan_optimal(devices(), model(n_layers=200))
        self.assertFalse(p.feasible)
        self.assertIn("can hold", p.reason)


class TestRulePrefersFasterNodes(unittest.TestCase):
    def test_fast_node_gets_more_layers_than_straggler(self):
        p = plan_optimal(devices(), model())
        self.assertGreater(p.layers.get("pi5", 0), p.layers.get("pi4", 0))

    def test_rule_beats_even_split(self):
        devs, m = devices(), model()
        rule = plan_optimal(devs, m)
        even = plan_even(devs, m)
        self.assertTrue(rule.feasible)
        if even.feasible:
            self.assertLessEqual(rule.predicted["ms_per_token"], even.predicted["ms_per_token"])

    def test_single_fast_node_wins_when_the_model_fits_on_it(self):
        # A small model that fits on the pi5 alone should not be distributed: the hops
        # cost time and the extra nodes cost idle energy for nothing.
        p = plan_optimal(devices(), model(n_layers=8, per_layer_mb=100))
        self.assertEqual(p.participants, ["pi5"])


class TestStragglerQuestion(unittest.TestCase):
    def test_straggler_is_dropped_when_it_does_not_pay(self):
        # 16 layers at 100 MB fit on pi5 + jetson without the slow pi4.
        p = plan_optimal(devices(), model(n_layers=16, per_layer_mb=100))
        self.assertNotIn("pi4", p.participants)

    def test_straggler_is_kept_when_the_model_needs_its_memory(self):
        # 48 layers x 190 MB (+KV) is roughly a 14B Q4: pi5 (34 layers) plus jetson
        # (12) cannot hold it between them, so the straggler's memory is the only way
        # the model runs at all. This is the case where QR8's answer flips.
        p = plan_optimal(devices(), model(n_layers=48, per_layer_mb=190))
        self.assertTrue(p.feasible)
        self.assertIn("pi4", p.participants)

    def test_compare_quantifies_the_straggler_cost(self):
        result = compare(devices(), model(n_layers=16, per_layer_mb=100))
        self.assertIn("straggler_cost_pct", result)
        # Forcing the slow node in must not look free.
        self.assertGreater(result["straggler_cost_pct"], 0)

    def test_forcing_the_straggler_actually_includes_it(self):
        p = plan_optimal(devices(), model(n_layers=16, per_layer_mb=100),
                         force_include=["pi4"])
        self.assertIn("pi4", p.participants)


class TestEnergyObjective(unittest.TestCase):
    def test_energy_weight_changes_the_plan(self):
        # The jetson is fast but power-hungry; weighting energy should shift work away
        # from it at some lambda.
        devs, m = devices(), model(n_layers=24, per_layer_mb=100)
        latency_plan = plan_optimal(devs, m, energy_weight=0.0)
        energy_plan = plan_optimal(devs, m, energy_weight=1.0)
        self.assertTrue(latency_plan.feasible and energy_plan.feasible)
        self.assertLessEqual(latency_plan.predicted["ms_per_token"],
                             energy_plan.predicted["ms_per_token"] + 1e-6)

    def test_idle_power_of_participants_is_charged(self):
        # A node holding one layer still burns idle power for the whole token period.
        devs = devices()
        m = model(n_layers=4, per_layer_mb=10)
        one = evaluate(["pi5"], {"pi5": 4}, devs, m)
        two = evaluate(["pi5", "pi4"], {"pi5": 3, "pi4": 1}, devs, m)
        self.assertGreater(two["joules_per_token"], one["joules_per_token"])

    def test_pareto_front_is_ordered_and_deduplicated(self):
        front = pareto(devices(), model(n_layers=24, per_layer_mb=100))
        self.assertTrue(front)
        layers = [tuple(sorted(p["layers"].items())) for p in front]
        self.assertEqual(len(layers), len(set(layers)))


class TestRealisation(unittest.TestCase):
    def test_tensor_split_matches_layer_proportions(self):
        p = plan_optimal(devices(), model())
        split = p.tensor_split()
        self.assertEqual(len(split), len(p.participants))
        self.assertAlmostEqual(sum(split), 1.0, places=5)

    def test_single_node_plan_is_infeasible_when_too_big(self):
        p = plan_single(devices(), model(n_layers=32, per_layer_mb=120), "pi4")
        self.assertFalse(p.feasible)
        self.assertIn("pi4", p.reason)

    def test_comm_cost_grows_as_the_link_slows(self):
        fast = devices()
        slow = devices()
        for d in slow.values():
            d.link_mbps = 10.0
        m = model()
        layers = {"pi5": 12, "pi4": 10, "jetson": 10}
        order = ["pi5", "pi4", "jetson"]
        self.assertGreater(evaluate(order, layers, slow, m)["comm_ms"],
                           evaluate(order, layers, fast, m)["comm_ms"])


class TestInfeasibilityRefusalExplainsItself(unittest.TestCase):
    """`config/experiments/memory_wall_14b.json` states the single-node entries must be
    reported infeasible -- "that refusal IS the result". A refusal is therefore a published
    finding, and it has to survive being read.

    The message used to quote the RAW sum of `usable_mem_mb` as what "the cluster offers",
    while `memory_cap` subtracts `runtime_overhead_mb` per node. The two figures disagree by
    the overhead times the node count, which produced refusals whose own numbers said the
    model should have fitted -- "needs 11.6 GB, cluster offers 12.0 GB usable" -- reading as
    a bug in the tool rather than a property of the cluster.
    """

    def _cluster(self, usable_mb=4096, overhead_mb=700, n_layers=48, per_layer_mb=248.4):
        devices = {n: placement.DeviceProfile(
            node_id=n, ms_per_layer_decode=10.0, ms_per_layer_prefill=3.0,
            usable_mem_mb=usable_mb, power_active_w=6.0, power_idle_w=3.0,
            link_mbps=1000.0, rtt_ms=0.3) for n in ("pi5", "pi4", "jetson")}
        model = placement.ModelProfile(
            name="qwen14b-q6k", n_layers=n_layers,
            bytes_per_layer=per_layer_mb * 1024 * 1024,
            kv_bytes_per_layer_per_token=0.0, activation_bytes=0.0,
            context_len=4096, runtime_overhead_mb=overhead_mb)
        return devices, model

    def test_it_still_refuses(self):
        devices, model = self._cluster()
        plan = placement.plan_optimal(devices, model)
        self.assertFalse(plan.feasible)

    def test_the_capacity_it_quotes_is_less_than_the_need(self):
        # The whole defect: a refusal must not quote a capacity larger than the requirement.
        devices, model = self._cluster()
        reason = placement.plan_optimal(devices, model).reason
        self.assertIn("can hold 39 layers", reason)
        self.assertIn("48 layers", reason)

    def test_it_names_the_overhead_that_decided(self):
        devices, model = self._cluster()
        reason = placement.plan_optimal(devices, model).reason
        self.assertIn("700 MB of runtime overhead", reason)
        self.assertIn("raw memory sum", reason)

    def test_it_shows_per_node_capacity(self):
        # Without this a reader cannot tell an evenly-short cluster from one node being tiny.
        devices, model = self._cluster()
        reason = placement.plan_optimal(devices, model).reason
        for node in ("pi5", "pi4", "jetson"):
            self.assertIn("%s=13" % node, reason)

    def test_a_cluster_that_genuinely_fits_is_not_refused(self):
        devices, model = self._cluster(usable_mb=8192)
        self.assertTrue(placement.plan_optimal(devices, model).feasible)



if __name__ == "__main__":
    unittest.main()


class TestLatencyOnlyWithoutPower(unittest.TestCase):
    """Power is None ("not measured") until the INA226 rig exists; the rule must still plan.

    Before this, the honest input crashed and the dishonest one passed: evaluate() multiplied
    None by seconds, while build_devices(strict=False) substituted 0.0 W, so joules_per_token
    came out as a clean-looking measured 0. At energy_weight = 0 the objective never reads
    power, so latency planning needs none; energy figures must say so rather than invent one.
    """

    def _unpowered(self, *nids):
        from dataclasses import replace
        d = devices()
        for nid in (nids or d):
            d[nid] = replace(d[nid], power_active_w=None, power_idle_w=None)
        return d

    def test_latency_plan_works_and_energy_reports_not_measured(self):
        p = plan_optimal(self._unpowered(), model())
        self.assertTrue(p.feasible)
        self.assertGreater(p.predicted["ms_per_token"], 0)
        self.assertIsNone(p.predicted["joules_per_token"])
        self.assertIsNone(p.predicted["mean_power_w"])
        self.assertIn("not measured", p.predicted["energy"])

    def test_describe_prints_a_latency_only_plan(self):
        # heteroctl plan/run print this before anything runs; a None energy used to crash it.
        text = plan_optimal(self._unpowered(), model()).describe()
        self.assertIn("ms/token", text)
        self.assertIn("not measured", text)

    def test_one_unpowered_participant_blanks_energy_rather_than_summing_the_rest(self):
        order = ["pi5", "pi4"]
        e = evaluate(order, {"pi5": 20, "pi4": 12}, self._unpowered("pi4"), model())
        self.assertIsNone(e["joules_per_token"])
        self.assertIn("pi4", e["energy"])

    def test_power_is_never_read_at_zero_energy_weight(self):
        with_power, without = plan_optimal(devices(), model()), plan_optimal(self._unpowered(), model())
        self.assertEqual((with_power.order, with_power.layers), (without.order, without.layers))
        self.assertEqual(with_power.predicted["ms_per_token"], without.predicted["ms_per_token"])

    def test_energy_weighted_planning_refuses_without_power(self):
        with self.assertRaises(ValueError) as cm:
            plan_optimal(self._unpowered("pi4"), model(), energy_weight=0.01)
        self.assertIn("pi4", str(cm.exception))

    def test_pareto_refuses_without_power(self):
        with self.assertRaises(ValueError):
            pareto(self._unpowered(), model())

    def test_compare_keeps_the_latency_answer_and_drops_the_energy_one(self):
        c = compare(self._unpowered(), model())
        self.assertIn("tpot_improvement_pct", c)
        self.assertNotIn("energy_improvement_pct", c)

    def test_build_devices_leaves_unmeasured_power_as_none_not_zero(self):
        import json, tempfile
        from pathlib import Path
        from control import profiles

        class Node:
            def __init__(self, nid, role="worker"):
                self.id, self.role, self.usable_mem_mb = nid, role, 7000
                self.power_idle_w, self.description = None, ""

        class Cluster:
            nodes = [Node("pi5"), Node("pi4", "worker-straggler")]
            def workers(self):
                return self.nodes

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = ["node_id,phase,model,n_threads,ms_per_token,flags"]
            for nid, tg, pp in (("pi5", 55.0, 12.0), ("pi4", 180.0, 80.0)):
                rows += ["%s,tg,m,4,%s," % (nid, tg), "%s,pp,m,4,%s," % (nid, pp)]
            (tmp / "results.csv").write_text("\n".join(rows) + "\n")
            (tmp / "link.json").write_text(json.dumps(
                {"nodes": {n: {"mbps": 940.0, "rtt_ms": 0.4} for n in ("pi5", "pi4")}}))
            devs = profiles.build_devices(Cluster(), results_csv=tmp / "results.csv",
                                          link_json=tmp / "link.json",
                                          n_layers_by_model={"m": 22}, threads=4, strict=True)
        for nid in ("pi5", "pi4"):
            self.assertIsNone(devs[nid].power_active_w)
            self.assertIsNone(devs[nid].power_idle_w)
            self.assertIn("latency-only", devs[nid].notes)
        self.assertAlmostEqual(devs["pi4"].ms_per_layer_decode, 180.0 / 22, places=3)


class TestHeadAndEmbeddingAccounting(unittest.TestCase):
    """The output head and the token embedding are not per-layer: one node each holds them.

    llama.cpp at the pin sends the output layer to the last device of any offload (-ngl N
    offloads it first, then N-1 repeating layers), and keeps the token embedding on the CPU of
    the process that loads the model. The regression case is the measured 14B boundary.
    """

    def _m(self, **kw):
        base = dict(name="t", n_layers=10, bytes_per_layer=100 * MB, kv_bytes_per_layer_per_token=0,
                    activation_bytes=8192, context_len=1, head_bytes=250 * MB, embd_bytes=150 * MB)
        base.update(kw)
        return ModelProfile(**base)

    def _dev(self, nid, c, usable=1350):
        return DeviceProfile(nid, ms_per_layer_decode=c, ms_per_layer_prefill=c / 3, usable_mem_mb=usable)

    def test_the_holder_pays_and_nobody_else(self):
        m, d = self._m(), self._dev("x", 10)          # 1350 - 350 overhead = 1000 MB = 10 layers bare
        self.assertEqual(memory_cap(d, m), 10)
        self.assertEqual(memory_cap(d, m, holds_head=True), 7)
        self.assertEqual(memory_cap(d, m, holds_embd=True), 8)
        self.assertEqual(memory_cap(d, m, holds_head=True, holds_embd=True), 6)

    def test_last_node_holds_the_head_first_the_embedding_only_when_it_hosts(self):
        devs = {"a": self._dev("a", 10), "b": self._dev("b", 30)}
        for hosting, first_cap in ((False, 10), (True, 8)):
            m = self._m(n_layers=15, host_in_pipeline=hosting)
            p = plan_optimal(devs, m, head_placement="last")       # llama.cpp's layout (the default is "best")
            self.assertTrue(p.feasible)
            first, last = p.participants[0], p.participants[-1]
            self.assertLessEqual(p.layers[last], memory_cap(devs[last], m, holds_head=True))
            self.assertLessEqual(p.layers[first], memory_cap(devs[first], m, holds_embd=hosting))

    def test_the_head_is_priced_where_it_lands(self):
        # Memory is ample here, so no cap binds and only the prices decide; both nodes are kept in.
        # a is 3x faster than b. The 250 MB head is 2.5 layers of decode work on whichever node is
        # last: 25 ms on a, 75 on b -- so a goes last.
        devs = {"a": self._dev("a", 10, usable=100000), "b": self._dev("b", 30, usable=100000)}
        for hosting in (False, True):
            p = plan_optimal(devs, self._m(n_layers=15, host_in_pipeline=hosting), force_include=["a", "b"],
                             head_placement="last")
            self.assertEqual((p.order, p.layers), (["b", "a"], {"a": 14, "b": 1}))
            self.assertAlmostEqual(p.predicted["compute_ms"], 14 * 10 + 1 * 30 + 2.5 * 10, places=6)
        # Prefill computes logits for the last prompt token only: the head is not charged.
        p = plan_optimal(devs, self._m(n_layers=15), phase="prefill", force_include=["a", "b"])
        self.assertAlmostEqual(p.predicted["compute_ms"], 14 * 10 / 3 + 1 * 30 / 3, places=2)   # reported to 3 dp

    def test_a_worker_is_priced_at_its_worker_cost(self):
        # host_in_pipeline: the first node runs llama.cpp and every other one is an RPC worker. a is
        # fast as a host (10) and slow as a worker (40, like the Pi 5 without its repacked kernels).
        a = DeviceProfile("a", 10, 10 / 3, 100000, ms_per_layer_decode_worker=40.0)
        devs = {"a": a, "b": self._dev("b", 30, usable=100000)}
        p = plan_optimal(devs, self._m(n_layers=15, host_in_pipeline=True), force_include=["a", "b"],
                         head_placement="last")
        self.assertEqual((p.order, p.layers), (["a", "b"], {"a": 14, "b": 1}))   # a hosts
        self.assertAlmostEqual(p.predicted["compute_ms"], 14 * 10 + 1 * 30 + 2.5 * 30, places=6)
        # The harness launch: every participant is a worker, so a costs 40 wherever it sits, b is
        # now the cheaper node, and it takes the layers and the head.
        p = plan_optimal(devs, self._m(n_layers=15), force_include=["a", "b"])
        self.assertEqual((p.order, p.layers), (["a", "b"], {"a": 1, "b": 14}))
        self.assertAlmostEqual(p.predicted["compute_ms"], 1 * 40 + 14 * 30 + 2.5 * 30, places=6)

    def test_only_a_node_holding_the_model_can_host(self):
        devs = {"a": self._dev("a", 10), "b": self._dev("b", 30)}
        m = self._m(n_layers=15, host_in_pipeline=True)
        self.assertEqual(plan_optimal(devs, m, host_candidates=["b"]).order[0], "b")
        self.assertEqual(plan_optimal(devs, m, host_candidates=["a"]).order[0], "a")

    def test_single_node_pays_the_head_and_its_own_embedding(self):
        m = self._m(n_layers=7, host_in_pipeline=True)
        self.assertFalse(plan_single({"a": self._dev("a", 10)}, m, "a").feasible)   # holds 6
        self.assertTrue(plan_single({"a": self._dev("a", 10)}, self._m(n_layers=7), "a").feasible)

    def test_even_split_is_checked_against_the_charged_caps(self):
        devs = {"a": self._dev("a", 10), "b": self._dev("b", 30)}
        p = plan_even(devs, self._m(n_layers=16))          # 8 + 8: b is last and holds the head (7)
        self.assertFalse(p.feasible)
        self.assertIn("b (8 > 7 layers)", p.reason)

    def test_refusal_names_the_head(self):
        devs = {"a": self._dev("a", 10), "b": self._dev("b", 30)}
        p = plan_optimal(devs, self._m(n_layers=19, host_in_pipeline=True))
        self.assertFalse(p.feasible)
        self.assertIn("output head", p.reason)
        self.assertIn("token embedding", p.reason)

    def test_measured_14b_boundary_and_optimum(self):
        # Qwen2.5-14B Q4_K_M from its GGUF tensor table; MemAvailable at rest (2026-09-27); context
        # 80 = the sweep's pp64 + tg16; per-layer costs from the TinyLlama rows (only c5 < c4 matters).
        # Measured at b23701f: -ngl 6 (Pi 5 43 blocks + embedding, Pi 4 5 + head) ran and was the
        # fastest split; -ngl 4 (Pi 5 45 blocks) was OOM-killed.
        new = ModelProfile("qwen2.5-14b-q4km", 48, bytes_per_layer=164698112.0,
                           kv_bytes_per_layer_per_token=4096, activation_bytes=20480, context_len=80,
                           head_bytes=638689280, embd_bytes=437944320, host_in_pipeline=True)
        old = ModelProfile("qwen2.5-14b-q4km", 48, bytes_per_layer=8988110496 / 48,
                           kv_bytes_per_layer_per_token=4096, activation_bytes=20480, context_len=80)
        devs = {"pi5": DeviceProfile("pi5", 14.1, 3.5, usable_mem_mb=7619),
                "pi4": DeviceProfile("pi4", 44.9, 22.6, usable_mem_mb=7413)}
        self.assertEqual(memory_cap(devs["pi5"], new, holds_embd=True), 43)
        # The sweep's host was the Pi 5, the only board holding the 14B file.
        p = plan_optimal(devs, new, host_candidates=["pi5"])
        self.assertEqual((p.order, p.layers), (["pi5", "pi4"], {"pi5": 43, "pi4": 5}))
        # The older accounting (file size / n_layers, nothing charged apart) refused the split that
        # measured fastest, and planned three layers more onto the straggler.
        self.assertEqual(memory_cap(devs["pi5"], old), 40)
        self.assertEqual(plan_optimal(devs, old).layers, {"pi5": 40, "pi4": 8})


class TestMemoryBudgetPolicy(unittest.TestCase):
    """The planner's memory budget is MemAvailable measured at rest (cluster.json), capped by what
    the node reports free right now -- never the agents' deliberate slack (user decision 2026-09-27)."""

    def _devs(self, inventory, reserve=0.0, mapped=False, cap=None):
        import json, tempfile
        from pathlib import Path
        from control import profiles

        class Node:
            def __init__(self, nid, role="worker"):
                self.id, self.role, self.usable_mem_mb = nid, role, 7413
                self.power_idle_w, self.description = None, ""
                self.memory_reserve_mb = reserve
                self.host_weights_mapped = mapped
                self.max_weights_mb = cap

        class Cluster:
            nodes = [Node("pi4", "worker-straggler")]
            def workers(self):
                return self.nodes

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "results.csv").write_text("node_id,phase,model,n_threads,ms_per_token,flags\n"
                                             "pi4,tg,m,4,175.0,\npi4,pp,m,4,80.0,\n")
            (tmp / "link.json").write_text(json.dumps({"nodes": {"pi4": {"mbps": 938.0, "rtt_ms": 0.23}}}))
            (tmp / "inv.json").write_text(json.dumps({"nodes": {"pi4": inventory}}))
            return profiles.build_devices(Cluster(), results_csv=tmp / "results.csv", link_json=tmp / "link.json",
                                          inventory_json=tmp / "inv.json", n_layers_by_model={"m": 22},
                                          threads=4, strict=True)

    def test_the_agents_slack_is_not_planned_against(self):
        d = self._devs({"memory": {"usable_budget_mb": 2600, "available_mb": 7400}})
        self.assertEqual(d["pi4"].usable_mem_mb, 7400)

    def test_never_more_than_is_free_right_now(self):
        d = self._devs({"memory": {"usable_budget_mb": 2600, "available_mb": 5100}})
        self.assertEqual(d["pi4"].usable_mem_mb, 5100)

    def test_without_a_live_figure_the_measured_at_rest_value_stands(self):
        d = self._devs({"memory": {"usable_budget_mb": 2600}})
        self.assertEqual(d["pi4"].usable_mem_mb, 7413)

    def test_a_mapped_host_reaches_the_profile(self):
        self.assertTrue(self._devs({"memory": {"available_mb": 7400}}, mapped=True)["pi4"].host_weights_mapped)
        self.assertFalse(self._devs({"memory": {"available_mb": 7400}})["pi4"].host_weights_mapped)

    def test_a_reserve_reaches_the_profile_apart_from_the_budget(self):
        d = self._devs({"memory": {"available_mb": 7400}}, reserve=450)
        self.assertEqual((d["pi4"].usable_mem_mb, d["pi4"].memory_reserve_mb), (7400, 450))

    def test_a_weights_cap_reaches_the_profile_and_survives_a_save(self):
        import tempfile
        from pathlib import Path
        from control import profiles
        d = self._devs({"memory": {"available_mb": 7400}}, cap=2845)
        self.assertEqual((d["pi4"].usable_mem_mb, d["pi4"].max_weights_mb), (7400, 2845))
        self.assertIsNone(self._devs({"memory": {"available_mb": 7400}})["pi4"].max_weights_mb)
        with tempfile.TemporaryDirectory() as tmp:
            profiles.save_devices(d, Path(tmp) / "devices.json")
            self.assertEqual(profiles.load_devices(Path(tmp) / "devices.json")["pi4"].max_weights_mb, 2845)


class TestBoardAwareMemoryAccounting(unittest.TestCase):
    """memory_cap by role, where a node's overheads are measured (validated out of sample 2026-09-28,
    testbed/benchmarks.md "Memory accounting, validated out of sample"). A host pays its overhead and
    the largest tensor it uploads, not the mmap'd embedding; a worker pays its overhead, and the
    Jetson's Vulkan worker also pays the largest tensor it receives a second time. A role that is not
    measured keeps the stand-in, so older plans and registrations reproduce."""

    def _m(self, **kw):
        base = dict(name="t", n_layers=20, bytes_per_layer=100 * MB, kv_bytes_per_layer_per_token=0,
                    activation_bytes=8192, context_len=1, head_bytes=250 * MB, embd_bytes=150 * MB,
                    max_block_tensor_bytes=30 * MB)
        base.update(kw)
        return ModelProfile(**base)

    def _dev(self, nid="x", usable=1350, **kw):
        return DeviceProfile(nid, ms_per_layer_decode=10, ms_per_layer_prefill=3, usable_mem_mb=usable, **kw)

    def test_an_unmeasured_role_keeps_the_stand_in(self):
        m, d = self._m(), self._dev()                        # no measured overheads at all
        for role in (None, "host", "worker"):
            self.assertEqual(memory_cap(d, m, holds_head=True, holds_embd=True, role=role, uploads=True), 6)
        self.assertEqual(memory_cap(d, m, role="any"), memory_cap(d, m))
        only_worker = self._dev(worker_overhead_mb=50)       # measured as a worker, not as a host
        self.assertEqual(memory_cap(only_worker, m, holds_embd=True, role="host"), 8)   # 1350-350-150

    def test_a_host_pays_its_overhead_and_largest_upload_not_the_embedding(self):
        m, d = self._m(), self._dev(host_overhead_mb=100)
        # With workers: the head travels to the last node, so it is the largest upload.
        self.assertEqual(memory_cap(d, m, holds_embd=True, role="host", uploads=True), 10)    # 1350-100-250
        # Alone: it holds the head and uploads nothing.
        self.assertEqual(memory_cap(d, m, holds_head=True, holds_embd=True, role="host"), 10)  # 1350-100-250
        # A host whose head stays local but which still uploads pays the largest block tensor.
        self.assertEqual(memory_cap(d, m, holds_head=True, role="host", uploads=True), 9)     # -30 more
        self.assertEqual(memory_cap(d, self._m(embd_bytes=900 * MB), holds_embd=True, role="host",
                                    uploads=True), 10)                                      # never charged

    def test_a_mapped_host_is_not_charged_its_upload_copy(self):
        # A host serving its blocks from the page cache (host_weights_mapped: the Pi 4) holds the copy only while it
        # loads, so its steady charge is blocks + overhead (registered rule ACCOUNT, 2026-09-28).
        m = self._m()
        plain, mapped = self._dev(host_overhead_mb=100), self._dev(host_overhead_mb=100, host_weights_mapped=True)
        self.assertEqual(memory_cap(plain, m, holds_embd=True, role="host", uploads=True), 10)   # 1350-100-250
        self.assertEqual(memory_cap(mapped, m, holds_embd=True, role="host", uploads=True), 12)  # 1350-100
        self.assertEqual(placement.memory_need_mb(mapped, m, 1000 * MB, 10, holds_embd=True, role="host", uploads=True),
                         1100)                                                              # blocks + overhead
        self.assertEqual(placement.memory_need_mb(plain, m, 1000 * MB, 10, holds_embd=True, role="host", uploads=True),
                         1350)
        # Alone it uploads nothing, so the flag changes nothing; a head it holds is still charged.
        self.assertEqual(memory_cap(mapped, m, holds_head=True, holds_embd=True, role="host"), 10)
        # A worker is not a host: the flag does not touch its charge.
        self.assertEqual(memory_cap(self._dev(worker_overhead_mb=50, host_weights_mapped=True), m, role="worker"), 13)

    def test_a_mapped_hosts_copy_must_fit_beside_its_overhead(self):
        # The copy is anonymous: it cannot evict page cache to fit, so overhead + copy must fit the budget alone.
        m = self._m()                                        # head 250 MiB: the largest upload
        tight = self._dev(usable=340, host_overhead_mb=100, host_weights_mapped=True)
        self.assertEqual(memory_cap(tight, m, holds_embd=True, role="host", uploads=True), 0)    # 100+250 > 340
        room = self._dev(usable=360, host_overhead_mb=100, host_weights_mapped=True)
        self.assertEqual(memory_cap(room, m, holds_embd=True, role="host", uploads=True), 2)     # (360-100) // 100
        kept = self._dev(usable=360, host_overhead_mb=100, host_weights_mapped=True, memory_reserve_mb=20)
        self.assertEqual(memory_cap(kept, m, holds_embd=True, role="host", uploads=True), 0)     # the reserve counts

    def test_a_cpu_worker_pays_its_overhead_and_the_head_it_holds(self):
        m, d = self._m(), self._dev(worker_overhead_mb=50)
        self.assertEqual(memory_cap(d, m, role="worker"), 13)                    # 1350-50
        self.assertEqual(memory_cap(d, m, holds_head=True, role="worker"), 10)   # -250

    def test_a_repaying_worker_pays_the_largest_received_tensor_again(self):
        m, d = self._m(), self._dev(worker_overhead_mb=30, worker_repays_received=True)
        self.assertEqual(memory_cap(d, m, holds_head=True, role="worker"), 8)    # 1350-30-250-250
        self.assertEqual(memory_cap(d, m, role="worker"), 12)                    # 1350-30-30 (a block tensor)

    def test_any_is_an_upper_bound_on_every_role(self):
        m = self._m()
        for d in (self._dev(), self._dev(host_overhead_mb=100), self._dev(worker_overhead_mb=30),
                  self._dev(host_overhead_mb=500, worker_overhead_mb=30, worker_repays_received=True)):
            bound = memory_cap(d, m, role="any")
            for role in ("host", "worker"):
                for head in (False, True):
                    for up in (False, True):
                        self.assertLessEqual(memory_cap(d, m, holds_head=head, holds_embd=True, role=role,
                                                        uploads=up), bound)

    def test_the_subset_pre_check_does_not_prune_what_the_roles_admit(self):
        # Under the flat 350 MB these two hold 7 + 7 layers, short of 16; measured as workers at 30 MB,
        # 10 + 10 -- a plan exists, and the pre-check must not throw it away.
        m = self._m(n_layers=16, head_bytes=0.0, max_block_tensor_bytes=0.0)
        legacy = {n: self._dev(n, usable=1050) for n in ("a", "b")}
        self.assertFalse(plan_optimal(legacy, m).feasible)
        measured = {n: self._dev(n, usable=1050, worker_overhead_mb=30) for n in ("a", "b")}
        p = plan_optimal(measured, m)
        self.assertTrue(p.feasible)
        self.assertEqual(sum(p.layers.values()), 16)

    def test_the_refusal_quotes_the_measured_overheads(self):
        m = self._m(n_layers=40)
        reason = plan_optimal({"a": self._dev("a", worker_overhead_mb=30)}, m).reason
        self.assertIn("a 30 MB", reason)
        self.assertNotIn("350 MB of runtime overhead", reason)

    # The measured cells, with the cluster's figures (cluster.json memory_overhead_mb) and the tensor
    # tables (testbed/dataset/model_shapes.json).
    Q14 = dict(name="qwen2.5-14b-q4km", n_layers=48, bytes_per_layer=164698112.0, kv_bytes_per_layer_per_token=4096,
               activation_bytes=20480, context_len=80, head_bytes=638689280, embd_bytes=437944320,
               max_block_tensor_bytes=58060800, host_in_pipeline=True)

    def _cluster(self):
        return {"pi5": DeviceProfile("pi5", 14.1, 3.5, usable_mem_mb=7619, host_overhead_mb=220, worker_overhead_mb=191),
                "pi4": DeviceProfile("pi4", 44.9, 22.6, usable_mem_mb=7413, host_overhead_mb=137, worker_overhead_mb=159),
                "jetson": DeviceProfile("jetson", 20.0, 5.0, usable_mem_mb=3545, worker_overhead_mb=44,
                                        worker_repays_received=True)}

    def test_the_pi4_hosts_the_46_blocks_it_was_measured_to_hold(self):
        # THRASH2, 2026-09-28: the Pi 4 hosted Qwen2.5-14B at normal speed with 46 blocks and 17.6x slower at 47,
        # with the Pi 5 as its worker holding the 609 MiB head. The old charge (with the upload copy) capped it at 42.
        m, d = ModelProfile(**self.Q14), self._cluster()
        self.assertEqual(memory_cap(d["pi4"], m, holds_embd=True, role="host", uploads=True), 42)
        mapped = DeviceProfile("pi4", 44.9, 22.6, usable_mem_mb=7413, host_overhead_mb=137, worker_overhead_mb=159,
                               host_weights_mapped=True)
        self.assertEqual(memory_cap(mapped, m, holds_embd=True, role="host", uploads=True), 46)
        # The Pi 5 repacks: it keeps paying its upload copy, and its boundary does not move.
        self.assertEqual(memory_cap(d["pi5"], m, holds_embd=True, role="host", uploads=True),
                         memory_cap(DeviceProfile("pi5", 14.1, 3.5, usable_mem_mb=7619, host_overhead_mb=220,
                                                  worker_overhead_mb=191), m, holds_embd=True, role="host", uploads=True))

    def test_the_14b_boundary_stays_where_it_was_measured(self):
        # 43 blocks + the embedding ran on the Pi 5 host, uploading the 609 MiB head; 45 were OOM-killed.
        m, d = ModelProfile(**self.Q14), self._cluster()
        self.assertEqual(memory_cap(d["pi5"], m, holds_embd=True, role="host", uploads=True), 43)

    def test_the_7b_fits_the_pi5_alone_as_it_did(self):
        # Mistral-7B Q8_0 alone on the Pi 5 ran with 233 MiB free (2026-09-27 addendum); the stand-in
        # refused it by 70 MiB.
        m = ModelProfile("mistral-7b-q8_0", 32, bytes_per_layer=231768064.0, kv_bytes_per_layer_per_token=4096,
                         activation_bytes=16384, context_len=80, head_bytes=139280384, embd_bytes=139280384,
                         max_block_tensor_bytes=62390272, host_in_pipeline=True)
        pi5 = self._cluster()["pi5"]
        self.assertTrue(plan_single({"pi5": pi5}, m, "pi5").feasible)
        self.assertFalse(plan_single({"pi5": DeviceProfile("pi5", 14.1, 3.5, usable_mem_mb=7619)}, m, "pi5").feasible)

    def test_the_jetson_holding_the_14b_head_is_capped_where_it_would_fit(self):
        # Measured 2026-09-28: 8 blocks + the head cost the Jetson 2,541 MiB -- blocks + head + the head
        # again + ~10. The stand-in would allow 16 blocks + head (about 3,780 MiB by that measure, over
        # its 3,545); the measured accounting allows 14.
        m, d = ModelProfile(**self.Q14), self._cluster()
        self.assertEqual(memory_cap(d["jetson"], m, holds_head=True, role="worker"), 14)
        self.assertEqual(memory_cap(DeviceProfile("jetson", 20.0, 5.0, usable_mem_mb=3545), m,
                                    holds_head=True, role="worker"), 16)

    def test_a_reserve_stays_free_in_every_role_and_is_not_usage(self):
        m = self._m()
        plain, kept = self._dev(worker_overhead_mb=50), self._dev(worker_overhead_mb=50, memory_reserve_mb=250)
        self.assertEqual(memory_cap(plain, m, role="worker"), 13)                  # 1350-50
        self.assertEqual(memory_cap(kept, m, role="worker"), 10)                   # 1350-250-50
        self.assertEqual(memory_cap(kept, m, role="any"), 10)                      # the upper bound keeps it too
        self.assertEqual(memory_cap(self._dev(memory_reserve_mb=250), m), 7)       # and so does the stand-in: 1350-250-350
        # The need a cell is gated and scored against is usage, which a reserve does not change.
        self.assertEqual(placement.memory_need_mb(kept, m, 1000 * MB, 10, role="worker"),
                         placement.memory_need_mb(plain, m, 1000 * MB, 10, role="worker"))

    # The Qwen3-30B-A3B MoE's tensor table (testbed/dataset/model_shapes.json).
    MOE = dict(name="qwen3-30b-a3b-q3_k_m", n_layers=48, bytes_per_layer=298269013.3333333,
               kv_bytes_per_layer_per_token=2048, activation_bytes=8192, context_len=80, head_bytes=255260672,
               embd_bytes=133703680, max_block_tensor_bytes=138412032, active_bytes_per_layer=28424533.333333332,
               host_in_pipeline=True)

    def test_the_jetson_keeps_its_reserve_on_the_moe(self):
        # 2026-09-28: 11 MoE blocks were charged 3,290 MiB of the Jetson's 3,545 and dropped 3,273, but its
        # Vulkan allocator refused 108 MiB with ~290 MiB left, and the cell failed. With the 450 MiB
        # reserve (cluster.json memory_reserve_mb) the cap is 10.
        m, jetson = ModelProfile(**self.MOE), self._cluster()["jetson"]
        self.assertEqual(memory_cap(jetson, m, role="worker"), 11)
        kept = DeviceProfile("jetson", 20.0, 5.0, usable_mem_mb=3545, worker_overhead_mb=44,
                             worker_repays_received=True, memory_reserve_mb=450)
        self.assertEqual(memory_cap(kept, m, role="worker"), 10)

    def test_the_refusal_names_a_reserve(self):
        m = self._m(n_layers=40)
        reason = plan_optimal({"a": self._dev("a", worker_overhead_mb=30, memory_reserve_mb=450)}, m).reason
        self.assertIn("reserve it must keep free (a 450 MB)", reason)

    def test_a_weights_cap_binds_whatever_memory_allows(self):
        # GPU_CAP (2026-09-29): the Jetson's Vulkan0 worker ran every share up to a 2,829 MiB model buffer and died above
        # it with memory to spare, so a node's blocks + a head it holds are capped apart from its memory budget.
        m = self._m()                                                               # 100 MB blocks, a 250 MB head
        roomy = self._dev(usable=3000, worker_overhead_mb=50)
        capped = self._dev(usable=3000, worker_overhead_mb=50, max_weights_mb=650)
        self.assertEqual(memory_cap(roomy, m, role="worker"), 29)                  # (3000-50)/100
        self.assertEqual(memory_cap(capped, m, role="worker"), 6)                  # 650/100
        self.assertEqual(memory_cap(capped, m, holds_head=True, role="worker"), 4)   # (650-250)/100
        self.assertEqual(memory_cap(capped, m, role="any"), 6)                     # the upper bound keeps it too
        self.assertEqual(memory_cap(self._dev(usable=3000, max_weights_mb=650), m), 6)   # and so does the stand-in
        self.assertEqual(memory_cap(self._dev(usable=3000, max_weights_mb=200), m, holds_head=True), 0)  # head > cap
        # Memory still binds where it is the smaller: 450 - 50 = 400 MB of blocks.
        self.assertEqual(memory_cap(self._dev(usable=450, worker_overhead_mb=50, max_weights_mb=650), m, role="worker"), 4)
        # The need a cell is gated and scored against is usage, which the cap does not change.
        self.assertEqual(placement.memory_need_mb(capped, m, 600 * MB, 6, role="worker"),
                         placement.memory_need_mb(roomy, m, 600 * MB, 6, role="worker"))

    # The Qwen3.8-27B hybrid's tensor table (testbed/dataset/model_shapes.json).
    Q27 = dict(name="qwen3.8-27b-q3_k_m", n_layers=65, bytes_per_layer=181596727.13846153, kv_bytes_per_layer_per_token=4096,
               activation_bytes=20480, context_len=80, head_bytes=874106880, embd_bytes=715161600,
               max_block_tensor_bytes=61276160, host_in_pipeline=True)

    def test_the_jetson_holds_the_16_blocks_that_ran_not_the_17_that_crashed(self):
        # The Jetson as cluster.json has it (usable 3545, worker 44, repays, reserve 450, max_weights_mb 2845). Memory
        # alone allows 17 middle blocks of the 27B: the share that died of SIGSEGV 4 times out of 4. The cap allows 16,
        # which ran 3 times (one with 423 MiB MemAvailable). The MoE keeps the 10 blocks that ran, and not the 11 that
        # failed at upload.
        jetson = DeviceProfile("jetson", 20.0, 5.0, usable_mem_mb=3545, worker_overhead_mb=44,
                               worker_repays_received=True, memory_reserve_mb=450, max_weights_mb=2845)
        uncapped = DeviceProfile("jetson", 20.0, 5.0, usable_mem_mb=3545, worker_overhead_mb=44,
                                 worker_repays_received=True, memory_reserve_mb=450)
        self.assertEqual(memory_cap(uncapped, ModelProfile(**self.Q27), role="worker"), 17)
        self.assertEqual(memory_cap(jetson, ModelProfile(**self.Q27), role="worker"), 16)
        self.assertEqual(memory_cap(jetson, ModelProfile(**self.MOE), role="worker"), 10)
        self.assertEqual(memory_cap(DeviceProfile("jetson", 20.0, 5.0, usable_mem_mb=3545, worker_overhead_mb=44,
                                                  worker_repays_received=True, max_weights_mb=2845),
                                    ModelProfile(**self.MOE), role="worker"), 10)     # the cap alone excludes the 11


class TestMixtureOfExpertsPricing(unittest.TestCase):
    """A MoE decodes from its active experts: the caller's per-layer decode cost is per active bytes, and the head
    must be priced in the same unit, or it is under-priced by active/total. Memory still counts every expert."""

    def test_the_head_is_priced_against_the_active_bytes(self):
        dense = ModelProfile("d", 10, bytes_per_layer=400 * MB, kv_bytes_per_layer_per_token=0,
                             activation_bytes=8192, head_bytes=100 * MB)
        moe = ModelProfile("m", 10, bytes_per_layer=400 * MB, kv_bytes_per_layer_per_token=0,
                           activation_bytes=8192, head_bytes=100 * MB, active_bytes_per_layer=50 * MB)
        d = {"a": DeviceProfile("a", 10, 3, usable_mem_mb=100000)}
        self.assertAlmostEqual(evaluate(["a"], {"a": 10}, d, dense)["compute_ms"], 10 * 10 + 0.25 * 10)
        self.assertAlmostEqual(evaluate(["a"], {"a": 10}, d, moe)["compute_ms"], 10 * 10 + 2.0 * 10)
        # ...and memory still holds all 400 MB per layer, whatever a token reads.
        self.assertEqual(memory_cap(DeviceProfile("a", 10, 3, usable_mem_mb=350 + 2000), moe), 5)


class TestRequestAwarePlanning(unittest.TestCase):
    """A request workload {"prefill": P, "decode": G} plans the whole request (P x prefill + G x decode), not one
    phase. prima.cpp's Halda optimises one per-token figure; our rows price the phases separately, so the rule
    can pick the plan that serves the request fastest. Also: the head where a layout puts it, and headroom."""

    def _m(self, **kw):
        base = dict(name="t", n_layers=10, bytes_per_layer=100 * MB, kv_bytes_per_layer_per_token=0,
                    activation_bytes=8192, context_len=1, head_bytes=0.0)
        base.update(kw)
        return ModelProfile(**base)

    def _devs(self, **kw):
        # a decodes faster, b prefills ten times faster: the request's mix decides which serves it.
        return {"a": DeviceProfile("a", ms_per_layer_decode=10, ms_per_layer_prefill=10, usable_mem_mb=100000, **kw),
                "b": DeviceProfile("b", ms_per_layer_decode=12, ms_per_layer_prefill=1, usable_mem_mb=100000, **kw)}

    def test_the_request_mix_changes_the_plan(self):
        m, d = self._m(), self._devs()
        self.assertEqual(plan_optimal(d, m, "decode").order, ["a"])
        self.assertEqual(plan_optimal(d, m, "prefill").order, ["b"])
        long_prompt = plan_optimal(d, m, {"prefill": 128, "decode": 16})
        self.assertEqual(long_prompt.order, ["b"])
        self.assertEqual(long_prompt.predicted["request_ms"], 128 * 10 + 16 * 120)    # 3,200 ms
        self.assertEqual(long_prompt.predicted["ttft_ms"], 128 * 10)
        self.assertEqual(long_prompt.predicted["ms_per_token"], 120)                   # TPOT, b's decode
        self.assertEqual(plan_optimal(d, m, {"prefill": 1, "decode": 1000}).order, ["a"])

    def test_a_single_phase_workload_is_that_phase(self):
        m, d = self._m(), self._devs()
        self.assertEqual(plan_optimal(d, m, {"decode": 1}).layers, plan_optimal(d, m, "decode").layers)

    def test_the_head_is_charged_where_the_layout_puts_it(self):
        m = self._m(head_bytes=100 * MB)            # one layer's worth of bytes: one layer of decode work
        d = self._devs()
        layers = {"a": 5, "b": 5}
        at_last = placement.evaluate(["a", "b"], layers, d, m, "decode")
        at_host = placement.evaluate(["a", "b"], layers, d, m, "decode", head_node="a")
        self.assertAlmostEqual(at_last["compute_ms"], 5 * 10 + 6 * 12)
        self.assertAlmostEqual(at_host["compute_ms"], 6 * 10 + 5 * 12)
        caps_last = placement._caps(["a", "b"], d, m)
        caps_host = placement._caps(["a", "b"], d, m, head_node="a")
        self.assertEqual((caps_host["a"], caps_host["b"]), (caps_last["a"] - 1, caps_last["b"] + 1))

    def test_headroom_lowers_the_caps_and_leaves_the_input_alone(self):
        m, d = self._m(), {"a": DeviceProfile("a", 10, 1, usable_mem_mb=1350, memory_reserve_mb=50)}
        kept = placement.with_headroom(d, 500)
        self.assertEqual(kept["a"].memory_reserve_mb, 550)
        self.assertEqual(d["a"].memory_reserve_mb, 50)
        self.assertEqual(memory_cap(kept["a"], m), memory_cap(d["a"], m) - 5)
        self.assertEqual(placement.with_headroom(d, {"a": 100})["a"].memory_reserve_mb, 150)

    def test_request_energy_sums_the_phases(self):
        m = self._m()
        d = {"a": DeviceProfile("a", ms_per_layer_decode=10, ms_per_layer_prefill=1, usable_mem_mb=100000,
                                power_active_w=10.0, power_idle_w=2.0)}
        r = placement.evaluate(["a"], {"a": 10}, d, m, {"prefill": 100, "decode": 10})
        dec, pre = r["per_phase"]["decode"]["joules_per_token"], r["per_phase"]["prefill"]["joules_per_token"]
        self.assertAlmostEqual(r["joules_per_request"], 100 * pre + 10 * dec, places=2)


class TestHeadPlacement(unittest.TestCase):
    """head_placement: the output head on the last board (llama.cpp's -ngl default), on the host (prima.cpp's Halda;
    -ot "^output\\.weight=CPU" at the pin, anchored), or the cheaper of the two ("best", the default since 2026-10-08).
    On a fast host with a slow last board, a big head belongs on the host; on a slow host it belongs on the faster last
    board."""

    def _m(self):
        return ModelProfile(name="h", n_layers=10, bytes_per_layer=100 * MB, kv_bytes_per_layer_per_token=0,
                            activation_bytes=8192, context_len=1, head_bytes=300 * MB, host_in_pipeline=True)

    def _devs(self, host_ms, worker_ms):
        # Under the 350 MB stand-in the host holds at most 8 layers with the head, so a second board is always needed.
        return {"h": DeviceProfile("h", ms_per_layer_decode=host_ms, ms_per_layer_prefill=1, usable_mem_mb=1450),
                "w": DeviceProfile("w", ms_per_layer_decode=worker_ms, ms_per_layer_prefill=1, usable_mem_mb=100000)}

    def test_a_fast_host_keeps_the_head(self):
        m, d = self._m(), self._devs(host_ms=5, worker_ms=20)
        last = plan_optimal(d, m, "decode", host_candidates=["h"], head_placement="last")
        best = plan_optimal(d, m, "decode", host_candidates=["h"], head_placement="best")
        self.assertIsNone(last.head_node)
        self.assertEqual(best.head_node, "h")
        self.assertLess(best.predicted["ms_per_token"], last.predicted["ms_per_token"])
        # Head last: the host holds 9 (the worker must hold one), 9 x 5 + 1 x 20 + the head's 3 layers x 20 = 125 ms.
        # Head on the host: its 300 MB leaves room for 8, 8 x 5 + 2 x 20 + 3 x 5 = 95 ms.
        self.assertEqual((last.layers["h"], best.layers["h"]), (9, 8))
        self.assertEqual((last.predicted["compute_ms"], best.predicted["compute_ms"]), (125, 95))

    def test_a_slow_host_leaves_the_head_on_the_last_board(self):
        m, d = self._m(), self._devs(host_ms=20, worker_ms=5)
        self.assertIsNone(plan_optimal(d, m, "decode", host_candidates=["h"], head_placement="best").head_node)

    def test_host_forces_it_and_a_bad_value_is_refused(self):
        m, d = self._m(), self._devs(host_ms=20, worker_ms=5)
        self.assertEqual(plan_optimal(d, m, "decode", host_candidates=["h"], head_placement="host").head_node, "h")
        with self.assertRaises(ValueError):
            plan_optimal(d, m, "decode", head_placement="first")

    def test_without_a_host_in_the_pipeline_it_stays_last(self):
        m = self._m()
        m.host_in_pipeline = False
        self.assertIsNone(plan_optimal(self._devs(5, 20), m, "decode", head_placement="best").head_node)


class TestDefaultHead(unittest.TestCase):
    """head_placement="best" is the default since 2026-10-08 (user's decision), after the head-to-head on the boards
    (2026-10-07): with the head on the host, Qwen3-14B on Pi 4 + Jetson decoded 5.4 % faster; TinyLlama kept it last."""

    _m, _devs = TestHeadPlacement._m, TestHeadPlacement._devs

    def _same(self, a, b):
        self.assertEqual((a.order, a.layers, a.head_node, a.feasible), (b.order, b.layers, b.head_node, b.feasible))
        self.assertEqual(a.predicted, b.predicted)

    def test_the_default_is_best(self):
        for host_ms, worker_ms in ((5, 20), (20, 5), (10, 10)):
            m, d = self._m(), self._devs(host_ms, worker_ms)
            for phase in ("decode", "prefill", {"prefill": 128, "decode": 128}):
                self._same(plan_optimal(d, m, phase, host_candidates=["h"]),
                           plan_optimal(d, m, phase, host_candidates=["h"], head_placement="best"))

    def test_a_fast_host_now_gets_the_head_by_default_and_a_slow_one_does_not(self):
        self.assertEqual(plan_optimal(self._devs(5, 20), self._m(), "decode", host_candidates=["h"]).head_node, "h")
        self.assertIsNone(plan_optimal(self._devs(20, 5), self._m(), "decode", host_candidates=["h"]).head_node)

    def test_never_slower_or_less_feasible_than_last(self):
        import random
        rng = random.Random(20261008)
        moved = 0
        for _ in range(400):
            n = rng.choice((2, 3))
            d = {"n%d" % i: DeviceProfile("n%d" % i, ms_per_layer_decode=rng.uniform(2, 40),
                                          ms_per_layer_prefill=rng.uniform(0.5, 10),
                                          usable_mem_mb=rng.choice((900, 1450, 2500, 100000))) for i in range(n)}
            m = ModelProfile(name="r", n_layers=rng.randint(6, 30), bytes_per_layer=rng.choice((50, 100, 150)) * MB,
                             kv_bytes_per_layer_per_token=0, activation_bytes=8192, context_len=1,
                             head_bytes=rng.choice((50, 300, 600)) * MB, embd_bytes=100 * MB, host_in_pipeline=True)
            new, old = plan_optimal(d, m, "decode"), plan_optimal(d, m, "decode", head_placement="last")
            if old.feasible:
                self.assertTrue(new.feasible)
                self.assertLessEqual(new.predicted["ms_per_token"], old.predicted["ms_per_token"] + 1e-9)
            moved += bool(new.head_node)
        self.assertGreater(moved, 0)                                  # the case exists in this sample

    def test_without_a_host_in_the_pipeline_nothing_changes(self):
        m = self._m()
        m.host_in_pipeline = False
        d = self._devs(5, 20)
        self._same(plan_optimal(d, m, "decode"), plan_optimal(d, m, "decode", head_placement="last"))

    def test_llama_args_refuses_a_head_on_the_host(self):
        p = plan_optimal(self._devs(5, 20), self._m(), "decode", host_candidates=["h"])
        self.assertEqual(p.head_node, "h")
        with self.assertRaises(ValueError):
            p.llama_args(None, "/m.gguf")


class TestEnergyPolicy(unittest.TestCase):
    """Per-phase power (power_active_prefill_w) and what a board the plan leaves out does (unused, 2026-10-08): with
    measured power, minimum energy can mean dropping a board -- a real saving only if that board is switched off."""

    def _m(self):
        return ModelProfile(name="e", n_layers=10, bytes_per_layer=10 * MB, kv_bytes_per_layer_per_token=0,
                            activation_bytes=8192, context_len=1)

    def _devs(self):
        # a: fast and hungry (10 ms/layer, 10 W decoding, 12 W prefilling, 3 W idle); b: slow and frugal (20, 4, 2 W)
        return {"a": DeviceProfile("a", 10, 2, 100000, power_active_w=10.0, power_idle_w=3.0, power_active_prefill_w=12.0),
                "b": DeviceProfile("b", 20, 4, 100000, power_active_w=4.0, power_idle_w=2.0)}

    def test_each_phase_is_priced_at_its_own_power(self):
        d = self._devs()
        self.assertAlmostEqual(evaluate(["a"], {"a": 10}, d, self._m(), "prefill")["joules_per_token"], 0.24)  # 20 ms x 12 W
        self.assertAlmostEqual(evaluate(["a"], {"a": 10}, d, self._m(), "decode")["joules_per_token"], 1.0)    # 100 ms x 10 W
        d["a"].power_active_prefill_w = None                                                                   # as before
        self.assertAlmostEqual(evaluate(["a"], {"a": 10}, d, self._m(), "prefill")["joules_per_token"], 0.2)

    def test_a_board_left_powered_is_charged_its_idle(self):
        d = self._devs()
        self.assertAlmostEqual(evaluate(["a"], {"a": 10}, d, self._m(), "decode", idle_on=["b"])["joules_per_token"], 1.2)

    def test_dropping_a_board_saves_energy_only_if_it_is_off(self):
        d, m = self._devs(), self._m()
        fast = plan_optimal(d, m, "decode")
        self.assertEqual((fast.order, fast.predicted["joules_per_token"]), (["a"], 1.0))
        off = plan_optimal(d, m, "decode", energy_weight=1000, unused="off")          # a switched off: 200 ms x 4 W
        self.assertEqual((off.order, off.predicted["ms_per_token"], off.predicted["joules_per_token"]), (["b"], 200.0, 0.8))
        idle = plan_optimal(d, m, "decode", energy_weight=1000, unused="idle")        # b alone would be 0.8 + 3 W x 0.2 s
        self.assertEqual((idle.order, idle.predicted["joules_per_token"]), (["a"], 1.2))
        with self.assertRaises(ValueError):
            plan_optimal(d, m, "decode", unused="sleep")

    def test_defaults_plan_as_before(self):
        d, m = self._devs(), self._m()
        for k in ("power_active_w", "power_idle_w", "power_active_prefill_w"):
            for dev in d.values():
                setattr(dev, k, None)
        p = plan_optimal(d, m, "decode")                       # latency only: no power needed, none read
        self.assertEqual((p.order, p.layers), (["a"], {"a": 10}))
        self.assertIsNone(p.predicted["joules_per_token"])


class TestExactFill(unittest.TestCase):
    """plan_optimal(fill="exact") checks each node's actual contiguous run of blocks (2026-10-07): with blocks of unequal
    size the average block over-counts some nodes (a plan that does not fit) and under-counts others (a plan missed)."""

    def _m(self, layer_bytes, **kw):
        base = dict(name="t", n_layers=len(layer_bytes), bytes_per_layer=sum(layer_bytes) * MB / len(layer_bytes),
                    kv_bytes_per_layer_per_token=0, activation_bytes=8192, context_len=1, runtime_overhead_mb=0,
                    layer_bytes=tuple(b * MB for b in layer_bytes))
        base.update(kw)
        return ModelProfile(**base)

    def _dev(self, nid, c, usable):
        return DeviceProfile(nid, ms_per_layer_decode=c, ms_per_layer_prefill=c / 3, usable_mem_mb=usable)

    def test_exact_needs_the_layer_sizes(self):
        m = ModelProfile("t", 4, 100 * MB, 0, 8192)
        with self.assertRaises(ValueError):
            plan_optimal({"a": self._dev("a", 10, 1000)}, m, fill="exact")
        with self.assertRaises(ValueError):
            plan_optimal({"a": self._dev("a", 10, 1000)}, m, fill="nearest")

    def test_the_average_over_counts_and_the_exact_fill_does_not(self):
        # Blocks of 100, 100, 300, 300 MB (average 200). The fast host a has 250 MB: the average allows it 1 block,
        # and the slow b (650 MB) 3 -- but b's real three are 100 + 300 + 300 = 700 MB, which does not fit.
        devs = {"a": self._dev("a", 10, 250), "b": self._dev("b", 30, 650)}
        m = self._m([100, 100, 300, 300])
        avg = plan_optimal(devs, m, force_include=["a", "b"], fill="average")
        self.assertEqual((avg.order, avg.layers), (["a", "b"], {"a": 1, "b": 3}))
        ex = plan_optimal(devs, m, force_include=["a", "b"], fill="exact", headroom_mb=0)
        self.assertEqual((ex.order, ex.layers), (["a", "b"], {"a": 2, "b": 2}))   # 200 MB on a, 600 MB on b: both fit
        self.assertLess(ex.predicted["ms_per_token"], avg.predicted["ms_per_token"])

    def test_the_exact_fill_finds_what_the_average_misses(self):
        # Blocks of 100, 100, 100, 500 MB (average 200): the host's 300 MB is 1 block by the average, 3 in fact.
        devs = {"a": self._dev("a", 10, 300), "b": self._dev("b", 30, 520)}
        m = self._m([100, 100, 100, 500])
        self.assertFalse(plan_optimal(devs, m, fill="average").feasible)   # 1 + 2 average blocks < 4
        ex = plan_optimal(devs, m, fill="exact", headroom_mb=0)
        self.assertEqual((ex.order, ex.layers), (["a", "b"], {"a": 3, "b": 1}))

    def test_equal_blocks_give_the_average_plan(self):
        devs = {"a": self._dev("a", 10, 450), "b": self._dev("b", 30, 900)}
        m = self._m([100] * 10, head_bytes=50 * MB)
        avg, ex = plan_optimal(devs, m, fill="average"), plan_optimal(devs, m, fill="exact", headroom_mb=0)
        self.assertEqual((ex.feasible, ex.predicted["ms_per_token"]), (avg.feasible, avg.predicted["ms_per_token"]))

    def test_the_weights_cap_is_checked_on_real_blocks(self):
        # b may hold 450 MB of weights (max_weights_mb), plenty of memory otherwise: its real blocks decide.
        devs = {"a": self._dev("a", 10, 250),
                "b": DeviceProfile("b", 30, 10, usable_mem_mb=5000, max_weights_mb=450)}
        m = self._m([100, 100, 200, 250])
        ex = plan_optimal(devs, m, fill="exact", headroom_mb=0)
        self.assertEqual(ex.layers, {"a": 2, "b": 2})                # b's 200 + 250 = 450 MB, exactly its cap
        m2 = self._m([100, 100, 200, 260])
        self.assertFalse(plan_optimal(devs, m2, fill="exact", headroom_mb=0).feasible)


class TestDefaultFill(unittest.TestCase):
    """The default since 2026-10-07 (user's decision): the exact fill when the model carries its block sizes, with
    placement.EXACT_FILL_HEADROOM_MB kept free on every board; the average fill, as before, when it does not."""

    def _devs(self):
        return {"a": DeviceProfile("a", 10, 10 / 3, usable_mem_mb=450), "b": DeviceProfile("b", 30, 10, usable_mem_mb=2000)}

    def _m(self, sizes=True):
        return ModelProfile("t", 6, 100 * MB, 0, 8192, context_len=1, runtime_overhead_mb=0,
                            layer_bytes=tuple([100 * MB] * 6) if sizes else None)

    def test_the_headroom_is_measured_evidence_sized(self):
        self.assertEqual(placement.EXACT_FILL_HEADROOM_MB, 256.0)   # ~2x the worst measured under-estimate, 133 MiB

    def test_block_sizes_get_the_exact_fill_with_the_headroom(self):
        devs, m = self._devs(), self._m()
        p = plan_optimal(devs, m, force_include=["a", "b"])
        # a's 450 MB less the 256 MiB headroom holds one 100 MB block; without the headroom it would hold four
        self.assertEqual(p.layers, {"a": 1, "b": 5})
        ref = plan_optimal(placement.with_headroom(devs, placement.EXACT_FILL_HEADROOM_MB), m,
                           force_include=["a", "b"], fill="exact", headroom_mb=0)
        self.assertEqual((p.order, p.layers, p.predicted["ms_per_token"]),
                         (ref.order, ref.layers, ref.predicted["ms_per_token"]))
        self.assertEqual(plan_optimal(devs, m, force_include=["a", "b"], headroom_mb=0).layers, {"a": 4, "b": 2})

    def test_a_model_without_block_sizes_plans_as_before(self):
        devs, m = self._devs(), self._m(sizes=False)
        p, avg = plan_optimal(devs, m, force_include=["a", "b"]), plan_optimal(devs, m, force_include=["a", "b"], fill="average")
        self.assertEqual((p.layers, p.predicted["ms_per_token"]), (avg.layers, avg.predicted["ms_per_token"]))
        self.assertEqual(p.layers, {"a": 4, "b": 2})                  # no headroom on the average fill

    def test_the_headroom_can_be_set(self):
        devs, m = self._devs(), self._m()
        self.assertEqual(plan_optimal(devs, m, force_include=["a", "b"], headroom_mb=150).layers, {"a": 3, "b": 3})
        with self.assertRaises(ValueError):
            plan_optimal(devs, m, fill="nearest")


class TestLoadedHost(unittest.TestCase):
    """ModelProfile.host_loads_weights (2026-10-09): a host that reads its weights into memory (llama.cpp -lm none, as
    serve_pool.sh serves) pays its token embedding; mapped, it does not, and every earlier plan reproduces. Measured on
    the Pi 4 hosting Qwen3-14B at a 4,096 context (testbed/results/full_eval_q14b_20261009_20261009_204531/)."""

    def _m(self, **kw):
        base = dict(name="t", n_layers=20, bytes_per_layer=100 * MB, kv_bytes_per_layer_per_token=0,
                    activation_bytes=8192, context_len=1, head_bytes=250 * MB, embd_bytes=150 * MB,
                    max_block_tensor_bytes=30 * MB)
        base.update(kw)
        return ModelProfile(**base)

    def _dev(self, nid="x", usable=1350, **kw):
        return DeviceProfile(nid, ms_per_layer_decode=10, ms_per_layer_prefill=3, usable_mem_mb=usable, **kw)

    def test_a_loaded_host_pays_its_embedding(self):
        mapped, loaded = self._m(), self._m(host_loads_weights=True)
        for d in (self._dev(host_overhead_mb=100), self._dev(host_overhead_mb=100, host_weights_mapped=True)):
            self.assertEqual(memory_cap(d, mapped, holds_head=True, holds_embd=True, role="host"), 10)   # 1350-100-250
            self.assertEqual(memory_cap(d, loaded, holds_head=True, holds_embd=True, role="host"), 8)    # -150 more
            self.assertEqual(placement.memory_need_mb(d, loaded, 800 * MB, 8, holds_head=True, holds_embd=True,
                                                      role="host"), 1300)
            self.assertEqual(placement.memory_need_mb(d, mapped, 800 * MB, 8, holds_head=True, holds_embd=True,
                                                      role="host"), 1150)

    def test_only_the_node_holding_the_embedding_pays_it(self):
        loaded = self._m(host_loads_weights=True)
        self.assertEqual(memory_cap(self._dev(worker_overhead_mb=50), loaded, role="worker"), 13)          # as before
        self.assertEqual(memory_cap(self._dev(host_overhead_mb=100), loaded, holds_head=True, role="host"), 10)

    def test_the_stand_in_already_charged_it(self):
        # An unmeasured role charges the embedding to whoever holds it, mapped or not: nothing changes there.
        for m in (self._m(), self._m(host_loads_weights=True)):
            self.assertEqual(memory_cap(self._dev(), m, holds_head=True, holds_embd=True, role="host", uploads=True), 6)

    def test_a_loaded_hosts_upload_copy_is_held_twice_while_it_loads(self):
        # Mapped: overhead + the copy (100 + 250 <= 360). Loaded: the loader's staging buffer and RPC's message, 100 + 500.
        d = self._dev(usable=360, host_overhead_mb=100, host_weights_mapped=True)
        self.assertEqual(memory_cap(d, self._m(), holds_embd=True, role="host", uploads=True), 2)            # (360-100)//100
        self.assertEqual(memory_cap(d, self._m(host_loads_weights=True), holds_embd=True, role="host", uploads=True), 0)
        roomy = self._dev(usable=700, host_overhead_mb=100, host_weights_mapped=True)
        self.assertEqual(memory_cap(roomy, self._m(host_loads_weights=True), holds_embd=True, role="host",
                                    uploads=True), 4)                                                        # 700-100-150
        # ...and still not with its blocks: the copies are gone before the KV cache and compute buffers exist.
        self.assertEqual(placement.memory_need_mb(roomy, self._m(host_loads_weights=True), 400 * MB, 4, holds_embd=True,
                                                  role="host", uploads=True), 650)                           # 400+100+150

    def test_the_exact_fill_charges_it_too(self):
        # A fast host (mapped as the Pi 4 is) and a slow worker: the host fills to its cap, which the embedding lowers.
        devs = {"h": self._dev("h", usable=1606, host_overhead_mb=100, host_weights_mapped=True),
                "w": DeviceProfile("w", ms_per_layer_decode=50, ms_per_layer_prefill=9, usable_mem_mb=3000,
                                   worker_overhead_mb=50)}
        kw = dict(n_layers=20, head_bytes=0.0, embd_bytes=150 * MB, host_in_pipeline=True, layer_bytes=(100 * MB,) * 20)
        p0 = plan_optimal(devs, self._m(**kw), host_candidates=["h"])
        p1 = plan_optimal(devs, self._m(host_loads_weights=True, **kw), host_candidates=["h"])
        self.assertEqual((p0.order[0], p0.layers["h"]), ("h", 12))       # 1606-256 headroom-100 = 1250: 12 blocks
        self.assertEqual((p1.order[0], p1.layers["h"]), ("h", 11))       # -150: 1100, 11 blocks

    # Qwen3-14B Q4_K_M (testbed/dataset/model_shapes.json, from its tensor table) and the Pi 4 as cluster.json measures it.
    Q3_14 = dict(name="qwen3-14b-q4_k_m", n_layers=40, bytes_per_layer=198001664.0, kv_bytes_per_layer_per_token=4096,
                 activation_bytes=20480, context_len=4096, head_bytes=638151680, embd_bytes=437575680,
                 max_block_tensor_bytes=73113600, host_in_pipeline=True)
    BLOCKS_27, BLOCKS_32 = 5309549568, 6287392768          # blocks 0-26 and 0-31, bytes
    PI4 = dict(ms_per_layer_decode=44.9, ms_per_layer_prefill=22.6, usable_mem_mb=7399, host_overhead_mb=137,
               worker_overhead_mb=159, host_weights_mapped=True)

    def _need(self, loaded, blocks, n, head):
        m = ModelProfile(host_loads_weights=loaded, **self.Q3_14)
        return placement.memory_need_mb(DeviceProfile("pi4", **self.PI4), m, blocks, n, holds_head=head, holds_embd=True,
                                        role="host", uploads=True)

    def test_the_mapped_charge_is_what_the_full_evaluations_gate_printed(self):
        # serve_pool.sh's "gate pi4: ... the plan needs N MiB" in every round of 2026-10-09.
        self.assertEqual(round(self._need(False, self.BLOCKS_27, 27, True)), 6241)      # the engine's pick
        self.assertEqual(round(self._need(False, self.BLOCKS_32, 32, False)), 6645)     # p32last
        self.assertEqual(round(self._need(False, self.BLOCKS_32, 32, True)), 7254)      # p32host: admitted, 145 MiB spare

    def test_the_loaded_charge_meets_the_buffers_llama_server_logged(self):
        # CPU model + KV + compute + output buffers on the Pi 4, from the serve logs: the charge is 9.6 MiB under them in
        # both served placements, where the mapped charge was 427 MiB under.
        logged = {"engine": 6089.46 + 432.00 + 146.01 + 0.58, "p32last": 6413.43 + 512.00 + 146.01 + 0.58}
        got = {"engine": self._need(True, self.BLOCKS_27, 27, True), "p32last": self._need(True, self.BLOCKS_32, 32, False)}
        for k in logged:
            self.assertLess(abs(got[k] - logged[k]), 10.0, k)
        self.assertAlmostEqual(got["engine"], 6658.5, delta=0.1)
        self.assertAlmostEqual(got["p32last"], 7062.4, delta=0.1)

    def test_p32host_is_refused_where_the_kernel_killed_it(self):
        # OOM-killed while loading in all five rounds, with 7,399-7,416 MiB available at rest: it needs 7,671.
        self.assertGreater(self._need(True, self.BLOCKS_32, 32, True), 7399 + 250)
        m, d = ModelProfile(host_loads_weights=True, **self.Q3_14), DeviceProfile("pi4", **self.PI4)
        self.assertEqual(memory_cap(d, ModelProfile(**self.Q3_14), holds_head=True, holds_embd=True, role="host",
                                    uploads=True), 32)                     # the old cap admitted p32host's 32 blocks
        self.assertEqual(memory_cap(d, m, holds_head=True, holds_embd=True, role="host", uploads=True), 30)
