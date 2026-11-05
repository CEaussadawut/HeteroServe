"""Tests for sweep run ordering and the per-run thermal record in control/experiment.py.

Neither `run_sweep` nor `Harness.run` had ANY coverage, which is how the ordering defect
survived a full correctness review of the module: nothing executed the loop, so nothing
could observe that it ran every repeat of one strategy before starting the next.

The defect was not a crash. For `placement_vs_even_7b` -- the sweep that decides the
project's ">=10 % better" MVP criterion -- all 5 `rule` repeats ran from a cold cluster and
all 5 `even` repeats started roughly 450 s of sustained load later. Thermal drift was
therefore perfectly confounded with the variable under test, entering the rule-vs-even TPOT
delta with the sign that favours the hypothesis. The sweep recorded no thermal state at any
point, and the throttle taxonomy in analysis.py lives entirely on the per-node llama-bench
path, so the output of a contaminated sweep was indistinguishable from a clean one.

Both halves of the fix are tested here: the interleaved order, and the temperature record
that lets a reader audit the interleaving instead of trusting it.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from control import experiment  # noqa: E402
from control.placement import Placement  # noqa: E402


class FakeHarness:
    """Records the order runs were requested in; runs nothing."""

    def __init__(self):
        self.calls = []
        self.data_dir = Path("/nonexistent")

    def rpc_up(self, node_ids=None):
        return {}

    def idle_baseline(self, seconds, note=""):
        return {"seconds": seconds}

    def run(self, placement, model, prompt="", n_predict=0, tier="gbe",
            run_id=None, seed=0, **kw):
        self.calls.append((placement.strategy, tier, seed))
        self.contexts = getattr(self, "contexts", []) + [kw.get("context")]
        return {"strategy": placement.strategy, "tier": tier, "ok": True,
                "timings": {}, "seed": seed}


class SweepOrderHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(__file__).resolve().parent / "_tmp_sweep"
        self.tmp.mkdir(exist_ok=True)

        # netem is real hardware control: shaping, ssh and a link survey. What is under
        # test is the order runs are issued in, so the whole module is stubbed.
        self._netem = experiment.netem
        self._sleep = experiment.time.sleep
        self._build = experiment.build_placement

        class FakeNetem:
            @staticmethod
            def apply_tier(cluster, tier):
                return {"applied": tier}

            @staticmethod
            def survey(cluster, seconds=10):
                return {"nodes": {}}

            @staticmethod
            def clear(cluster):
                return {}

        experiment.netem = FakeNetem
        experiment.time.sleep = lambda s: None

    def tearDown(self):
        experiment.netem = self._netem
        experiment.time.sleep = self._sleep
        experiment.build_placement = self._build
        for p in sorted(self.tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        self.tmp.rmdir()

    def _spec(self, strategies, repeats=5, tiers=("gbe",)):
        return {"name": "t", "model": "m.gguf", "tiers": list(tiers),
                "strategies": list(strategies), "repeats": repeats, "n_predict": 8,
                "prompt": "p", "energy_weight": 0.0, "idle_baseline_s": 0,
                "cooldown_s": 0, "seed": 1234}

    def _placements(self, feasible=(), infeasible=(), unplannable=()):
        def build(strategy, devices, model, weight):
            if strategy in unplannable:
                raise experiment.ExperimentError("cannot plan %s" % strategy)
            return Placement(strategy=strategy, order=["pi5"], layers={"pi5": 4},
                             model="m", feasible=strategy not in infeasible,
                             reason="does not fit" if strategy in infeasible else "")
        experiment.build_placement = build

    def test_repeats_are_interleaved_across_strategies_not_blocked_by_strategy(self):
        """The MVP criterion's sweep must not run one arm cold and the next hot.

        This is the whole finding. With repeats innermost the order was rule x5, even x5,
        single x5; interleaved it is one repeat of each strategy in turn, so any drift is
        spread across all three arms rather than loaded onto the last two.
        """
        self._placements()
        harness = FakeHarness()
        experiment.run_sweep(object(), harness, self._spec(["rule", "even", "single:pi5"]),
                             devices={}, model=object(), out_dir=self.tmp,
                             skip_preflight=True)

        strategies = [c[0] for c in harness.calls]
        self.assertEqual(strategies, ["rule", "even", "single:pi5"] * 5)
        # and NOT the old blocked order, stated explicitly so the mutation is unambiguous
        self.assertNotEqual(strategies, ["rule"] * 5 + ["even"] * 5 + ["single:pi5"] * 5)

    def test_every_run_is_launched_at_the_context_the_plan_charged(self):
        # Without it llama.cpp sized the KV cache for the model's training context and a
        # plan made for 2048 tokens OOM-killed a worker (2026-09-29).
        from types import SimpleNamespace
        self._placements()
        harness = FakeHarness()
        experiment.run_sweep(object(), harness, self._spec(["rule", "even"], repeats=2),
                             devices={}, model=SimpleNamespace(context_len=4096), out_dir=self.tmp,
                             skip_preflight=True)
        self.assertEqual(harness.contexts, [4096] * 4)

    def test_paired_repeats_share_a_seed_and_now_also_share_a_moment(self):
        """seed = f(rep) alone, so rule-r0 and even-r0 are a matched pair.

        That pairing was worthless when the two halves ran 450 s apart. Interleaved, each
        seed's runs are adjacent, which is what makes a per-repeat paired delta legitimate.
        """
        self._placements()
        harness = FakeHarness()
        experiment.run_sweep(object(), harness, self._spec(["rule", "even"], repeats=3),
                             devices={}, model=object(), out_dir=self.tmp,
                             skip_preflight=True)

        seeds = [c[2] for c in harness.calls]
        self.assertEqual(seeds, [1234, 1234, 1235, 1235, 1236, 1236])

    def test_an_infeasible_strategy_is_recorded_once_not_once_per_repeat(self):
        """Hoisting placement out of the repeat loop must not multiply the refusal.

        `memory_wall_14b` states the single-node entries must be reported infeasible --
        "that refusal IS the result" -- so emitting it five times would inflate the very
        artefact the experiment exists to produce.
        """
        self._placements(infeasible=("single:pi4",))
        harness = FakeHarness()
        result = experiment.run_sweep(
            object(), harness, self._spec(["rule", "single:pi4"]),
            devices={}, model=object(), out_dir=self.tmp, skip_preflight=True)

        refusals = [r for r in result["runs"] if "infeasible" in r]
        self.assertEqual(len(refusals), 1)
        self.assertEqual(refusals[0]["strategy"], "single:pi4")
        # the feasible arm still ran every repeat
        self.assertEqual([c[0] for c in harness.calls], ["rule"] * 5)

    def test_a_strategy_that_cannot_be_planned_is_recorded_once_and_does_not_abort(self):
        self._placements(unplannable=("bogus",))
        harness = FakeHarness()
        result = experiment.run_sweep(
            object(), harness, self._spec(["bogus", "rule"], repeats=2),
            devices={}, model=object(), out_dir=self.tmp, skip_preflight=True)

        errors = [r for r in result["runs"] if "error" in r]
        self.assertEqual(len(errors), 1)
        self.assertEqual([c[0] for c in harness.calls], ["rule", "rule"])

    def test_each_tier_gets_its_own_interleaved_block(self):
        """Shaping is expensive to change, so tier stays the outermost loop."""
        self._placements()
        harness = FakeHarness()
        experiment.run_sweep(
            object(), harness,
            self._spec(["rule", "even"], repeats=2, tiers=("shaped-10", "gbe")),
            devices={}, model=object(), out_dir=self.tmp, skip_preflight=True)

        self.assertEqual([(c[0], c[1]) for c in harness.calls], [
            ("rule", "shaped-10"), ("even", "shaped-10"),
            ("rule", "shaped-10"), ("even", "shaped-10"),
            ("rule", "gbe"), ("even", "gbe"),
            ("rule", "gbe"), ("even", "gbe"),
        ])


class ConditionsHarness(unittest.TestCase):
    """The temperature record that makes the ordering fix auditable."""

    def test_the_hottest_participant_is_what_counts(self):
        snap = {"pi4": {"soc_temp_c": 71.2}, "pi5": {"soc_temp_c": 58.0}}
        self.assertEqual(experiment._hottest(snap), 71.2)

    def test_an_unreadable_temperature_is_none_rather_than_zero(self):
        """Refuse rather than invent: 0 C would read as a very cold, very good run."""
        self.assertIsNone(experiment._hottest({"pi4": {"error": "unreachable"}}))
        self.assertIsNone(experiment._hottest({}))
        self.assertIsNone(experiment._hottest(None))
        # a node that answered without a temperature must not drag the max to 0 either
        self.assertEqual(
            experiment._hottest({"pi4": {"error": "x"}, "pi5": {"soc_temp_c": 44.0}}), 44.0)

    def test_summarise_carries_the_temperatures_into_the_comparable_rows(self):
        """Computed and then discarded is the failure mode this project keeps hitting.

        A snapshot that never reaches `summarise` cannot be used to check whether the
        interleaved order actually levelled the field, which is the only reason it is
        taken.
        """
        rows = experiment.summarise({"runs": [{
            "strategy": "rule", "tier": "gbe", "repeat": 0, "ok": True,
            "participants": ["pi5"], "layers": {"pi5": 4}, "timings": {"tpot_ms": 40.0},
            "conditions_start": {"pi5": {"soc_temp_c": 48.0}},
            "conditions_end": {"pi5": {"soc_temp_c": 63.5}},
        }]})
        self.assertEqual(rows[0]["temp_start_max_c"], 48.0)
        self.assertEqual(rows[0]["temp_end_max_c"], 63.5)

    def test_a_run_taken_without_a_reachable_agent_still_summarises(self):
        rows = experiment.summarise({"runs": [{
            "strategy": "rule", "tier": "gbe", "repeat": 0, "ok": True,
            "participants": ["pi5"], "layers": {"pi5": 4}, "timings": {},
            "conditions_start": {}, "conditions_end": {},
        }]})
        self.assertIsNone(rows[0]["temp_start_max_c"])
        self.assertEqual(rows[0]["status"], "ok")


class FakeNode:
    def __init__(self, nid):
        self.id = nid
        self.agent_port = 9410
        self.rpc_endpoint = "192.168.50.12:50052"


class FakeCluster:
    def node(self, nid):
        return FakeNode(nid)


class HarnessRecordsConditions(unittest.TestCase):
    """`Harness.run` must actually populate the snapshot summarise reads.

    Tested separately from `_hottest`/`summarise` because the two halves failing
    independently is exactly how a field ends up computed and never written.
    """

    def setUp(self):
        from control import harness as harness_mod
        self.mod = harness_mod
        self.tmp = Path(__file__).resolve().parent / "_tmp_harness"
        self.tmp.mkdir(exist_ok=True)

        self._client = harness_mod.nodeclient
        self._subprocess = harness_mod.subprocess
        self.temps = iter([{"pi5": {"soc_temp_c": 45.0}}, {"pi5": {"soc_temp_c": 66.0}}])

        outer = self

        class FakeClient:
            @staticmethod
            def fanout(nodes, fn, workers=8):
                # what `conditions()` consumes: {node_id: {"ok": bool, "data": ...}}
                snap = next(outer.temps)
                return {nid: {"ok": True, "data": v} for nid, v in snap.items()}

            metrics = staticmethod(lambda node: {})
            mark_all = staticmethod(lambda nodes, event, note="": {})

        class FakeProc:
            stdout, stderr, returncode = "", "", 0

        class FakeSubprocess:
            run = staticmethod(lambda *a, **kw: FakeProc())
            TimeoutExpired = outer._subprocess.TimeoutExpired
            # A stand-in for the module has to carry what the module carries. harness.py
            # passes stdin=subprocess.DEVNULL so llama-cli cannot drop into interactive
            # conversation mode and burn the whole 1800 s timeout at a prompt.
            DEVNULL = outer._subprocess.DEVNULL

        harness_mod.nodeclient = FakeClient
        harness_mod.subprocess = FakeSubprocess

    def tearDown(self):
        self.mod.nodeclient = self._client
        self.mod.subprocess = self._subprocess
        for f in sorted(self.tmp.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        self.tmp.rmdir()

    def _run_once(self):
        h = object.__new__(self.mod.Harness)
        h.data_dir = self.tmp
        h.bin = Path("/nonexistent/bin")
        h.cluster = FakeCluster()
        h.dry_run = False
        placement = Placement(strategy="rule", order=["pi5"], layers={"pi5": 4}, model="m")
        return self.mod.Harness.run(h, placement, "m.gguf", run_id="r1", n_predict=4)

    def test_a_run_records_the_board_temperature_at_both_ends(self):
        record = self._run_once()
        self.assertEqual(record["conditions_start"], {"pi5": {"soc_temp_c": 45.0}})
        self.assertEqual(record["conditions_end"], {"pi5": {"soc_temp_c": 66.0}})

    def test_the_launch_carries_the_planned_context(self):
        h = object.__new__(self.mod.Harness)
        h.data_dir, h.bin, h.cluster, h.dry_run = self.tmp, Path("/nonexistent/bin"), FakeCluster(), False
        placement = Placement(strategy="rule", order=["pi5"], layers={"pi5": 4}, model="m")
        record = self.mod.Harness.run(h, placement, "m.gguf", run_id="r2", n_predict=4, context=2048)
        cmd = record["command"]
        self.assertEqual(cmd[cmd.index("-c") + 1], "2048")
        self.assertEqual(record["context"], 2048)

    def test_the_snapshot_survives_into_the_comparable_row(self):
        """End to end: run -> record -> summarise, the path a real sweep takes."""
        row = experiment.summarise({"runs": [self._run_once()]})[0]
        self.assertEqual((row["temp_start_max_c"], row["temp_end_max_c"]), (45.0, 66.0))


class ConditionsNeverRaises(unittest.TestCase):
    def test_an_unreachable_agent_records_why_instead_of_failing_the_run(self):
        """A run is the measurement. Losing it because a thermometer was unreachable
        would discard the measurement to protect the annotation."""
        from control import harness as harness_mod

        class FakeClient:
            @staticmethod
            def fanout(nodes, fn, workers=8):
                return {"pi4": {"ok": False, "error": "timed out"}}
            metrics = staticmethod(lambda node: {})

        saved = harness_mod.nodeclient
        harness_mod.nodeclient = FakeClient
        try:
            snap = harness_mod.conditions([FakeNode("pi4")])
        finally:
            harness_mod.nodeclient = saved
        self.assertEqual(snap, {"pi4": {"error": "timed out"}})
        self.assertIsNone(experiment._hottest(snap))

    def test_no_participating_agents_is_an_empty_snapshot_not_a_crash(self):
        from control import harness as harness_mod
        self.assertEqual(harness_mod.conditions([]), {})


# Literal /metrics bodies, transcribed from the two agents' source. Pi 4 answers a FLAT
# dict (Pi4/node/node_agent.py:68-84); the Pi 5 WRAPS its sample (Pi5/node/agent.py:309).
# Kept as literals on purpose: if either agent's shape changes, these stop matching reality
# and the fix is to re-transcribe them, not to loosen the tests.
PI4_METRICS = {
    "ts": 1.0, "node": "pi4", "soc_temp_c": 61.0, "throttled_word": "0x0",
    "arm_mhz": 1800, "governor": "performance", "mem_available_mb": 3000,
    "swap_total_mb": 0, "load": [1.0, 1.0, 1.0], "rpc_active": True, "rpc_restarts": 0,
}
PI5_METRICS = {
    "sample": {
        "ts_utc": "2026-09-14T09:00:00.000Z", "mono_ns": 1, "node_id": "pi5",
        "soc_temp_c": 78.4, "cpu_mhz": 2400, "governor": "performance",
        "throttled_hex": "0x0", "throttled_now": 0, "mem_avail_mb": 6000,
        "rpc_rss_mb": 2100,
    },
    "run": {"run": "sweep_r0"},
}


class ThePi5IsNotSilentlyDropped(unittest.TestCase):
    """The Pi 5 answers 200 OK with a real temperature and used to vanish.

    `harness.conditions` reads only TOP-LEVEL /metrics keys, and the Pi 5's agent wraps its
    body in {"sample": ..., "run": ...}. So on a sweep where the Pi 5 sat at 78.4 C and the
    Pi 4 at 61.0 C, `temp_start_max_c` -- documented as the hottest participating board --
    read 61.0 C. The Pi 5 carries most layers and runs hottest, so the error was large,
    one-signed, and it defeated the interleaving these tests also cover.

    The fix does NOT unwrap the envelope: the canonical /metrics contract belongs to the
    node sessions. It makes the omission visible instead of silent.
    """

    def _snapshot(self, bodies):
        from control import harness as harness_mod

        class FakeClient:
            @staticmethod
            def fanout(nodes, fn, workers=8):
                return {nid: ({"ok": True, "data": b} if b is not None
                               else {"ok": False, "error": "connection refused"})
                        for nid, b in bodies.items()}
            metrics = staticmethod(lambda node: {})

        saved = harness_mod.nodeclient
        harness_mod.nodeclient = FakeClient
        try:
            return harness_mod.conditions([FakeNode(n) for n in bodies])
        finally:
            harness_mod.nodeclient = saved

    def test_an_unreadable_body_is_named_rather_than_dropped(self):
        snap = self._snapshot({"pi4": PI4_METRICS, "pi5": PI5_METRICS})
        self.assertEqual(snap["pi4"]["soc_temp_c"], 61.0)
        # The envelope is named outright, which is the evidence the Pi 5 session needs.
        self.assertEqual(snap["pi5"], {"unrecognised_payload": ["run", "sample"]})

    def test_the_maximum_ships_with_its_coverage(self):
        snap = self._snapshot({"pi4": PI4_METRICS, "pi5": PI5_METRICS, "jetson": None})
        hottest, contributed, missing = experiment._temps(snap)
        self.assertEqual(hottest, 61.0)
        self.assertEqual(contributed, ["pi4"])
        self.assertEqual(missing, ["jetson", "pi5"])

    def test_a_subset_maximum_is_visibly_a_subset_in_the_comparable_row(self):
        """This is the guard that matters: the row itself must disclose the coverage."""
        snap = self._snapshot({"pi4": PI4_METRICS, "pi5": PI5_METRICS, "jetson": None})
        row = experiment.summarise({"runs": [{
            "strategy": "rule", "tier": "gbe", "repeat": 0, "ok": True,
            "participants": ["pi4", "pi5", "jetson"], "layers": {}, "timings": {},
            "conditions_start": snap, "conditions_end": snap,
        }]})[0]
        self.assertEqual(row["temp_start_max_c"], 61.0)
        self.assertEqual(row["temp_nodes"], "pi4")
        self.assertEqual(row["temp_missing"], "jetson,pi5")

    def test_a_flat_agent_alone_reports_full_coverage(self):
        """The honest case must not acquire a spurious warning."""
        snap = self._snapshot({"pi4": PI4_METRICS})
        self.assertEqual(experiment._temps(snap), (61.0, ["pi4"], []))

    def test_a_reachable_agent_answering_nothing_is_still_counted_missing(self):
        snap = self._snapshot({"pi4": {}})
        self.assertEqual(snap["pi4"], {})
        self.assertEqual(experiment._temps(snap), (None, [], ["pi4"]))


if __name__ == "__main__":
    unittest.main()
