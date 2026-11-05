"""Tests for node/node_agent.py and node/device_info.py.

These run off-target, on whatever machine runs the suite. The health tests therefore CONSTRUCT the
node state they assert on (chrony, the thermal sensor, the llama.cpp build, the build manifest, the
node's identity) instead of assuming it of the host: the control laptop runs chrony, and a
provisioned Pi 4 has a build. What they must never do is pass or fail with the machine. That is deliberate, and it is the point of the most important test here:
**the agent must fail closed.** The control node asks `/health` whether this node may take
part in a measurement run. A node that cannot see its own instrumentation and answers
"ready" anyway would let the harness record a run whose energy can never be attributed --
and nothing downstream would flag it, because the row would look perfectly normal.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import unittest
import unittest.mock
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "node"))

import device_info  # noqa: E402
import node_agent  # noqa: E402


class TestThrottleDecoding(unittest.TestCase):
    """The Raspberry Pi throttle word is the single most important validity signal on this
    node: any non-zero value means the clock moved underneath the benchmark. Storing it
    decoded keeps the dataset self-explanatory once nobody remembers the bit layout."""

    def test_zero_is_all_false(self):
        decoded = device_info.decode_throttled("0x0")
        self.assertTrue(decoded)
        self.assertFalse(any(decoded.values()))

    def test_sticky_under_voltage_since_boot(self):
        # 0x50000 = bits 16 and 18: under-voltage and throttling happened earlier, but
        # neither is happening now. The board is fine this instant and the run is still
        # not certifiable, which is exactly why assert_clean_state demands a literal 0x0.
        decoded = device_info.decode_throttled("0x50000")
        self.assertTrue(decoded["under_voltage_since_boot"])
        self.assertTrue(decoded["throttled_since_boot"])
        self.assertFalse(decoded["under_voltage_now"])
        self.assertFalse(decoded["throttled_now"])

    def test_live_under_voltage(self):
        decoded = device_info.decode_throttled("0x50005")
        self.assertTrue(decoded["under_voltage_now"])
        self.assertTrue(decoded["throttled_now"])
        self.assertTrue(decoded["under_voltage_since_boot"])

    def test_arm_freq_capped_bits(self):
        self.assertTrue(device_info.decode_throttled("0x2")["arm_freq_capped_now"])
        self.assertTrue(device_info.decode_throttled("0x20000")["arm_freq_capped_since_boot"])

    def test_soft_temp_limit_bits(self):
        self.assertTrue(device_info.decode_throttled("0x8")["soft_temp_limit_now"])
        self.assertTrue(device_info.decode_throttled("0x80000")["soft_temp_limit_since_boot"])

    def test_unreadable_word_becomes_none_not_a_false_all_clear(self):
        # If vcgencmd is missing the honest answer is "unknown". Returning an all-false
        # dict would claim the board was verified clean when it was never asked.
        self.assertIsNone(device_info.decode_throttled("unavailable"))
        self.assertIsNone(device_info.decode_throttled(""))
        self.assertIsNone(device_info.decode_throttled(None))


class TestDeviceInfoOffTarget(unittest.TestCase):
    """device_info.py runs inside the agent, which is a long-lived service. An unhandled
    exception from a missing sysfs path would take the control node's only handle on this
    node down with it."""

    def test_collect_returns_the_declared_schema(self):
        data = device_info.collect()
        self.assertEqual(data["schema"], "heteroserve.device_info/1")
        for key in ("node", "cpu", "memory", "accelerator", "network", "thermal",
                    "power", "software", "storage", "collected_at_utc"):
            self.assertIn(key, data)

    def test_collect_is_json_serialisable(self):
        # The agent serves it and device_info --save writes it; a value that json cannot
        # encode would fail at the worst moment rather than here.
        json.dumps(device_info.collect())

    def test_identity_comes_from_node_env(self):
        node = device_info.collect()["node"]
        self.assertEqual(node["id"], "pi4")
        self.assertEqual(node["role"], "worker-straggler")

    def test_accelerator_is_explicitly_none_not_omitted(self):
        # "no usable accelerator" is itself an input to the placement rule -- it is the
        # contrast against the Jetson's GPU. Leaving the key out would make the control
        # node guess.
        acc = device_info.collect()["accelerator"]
        self.assertEqual(acc["type"], "none")
        self.assertIs(acc["usable_for_inference"], False)

    def test_vector_isa_reports_every_flag_as_a_boolean(self):
        # The A72's *absent* flags are the evidence for this node being the straggler, so
        # every flag must appear with an explicit true/false -- never be missing.
        isa = device_info.collect()["cpu"]["vector_isa"]
        for flag in device_info.VECTOR_FLAGS:
            self.assertIn(flag, isa)
            self.assertIsInstance(isa[flag], bool)

    def test_dotprod_and_i8mm_are_tracked(self):
        isa = device_info.collect()["cpu"]["vector_isa"]
        self.assertIn("asimddp", isa)
        self.assertIn("i8mm", isa)

    def test_power_channel_matches_node_env(self):
        power = device_info.collect()["power"]
        env = device_info.ENV
        self.assertEqual(power["ina226_channel"], int(env["INA_CHANNEL"]))
        self.assertEqual(power["i2c_address"], env["INA_I2C_ADDR"])
        self.assertEqual(power["shunt_ohms"], float(env["INA_SHUNT_OHMS"]))

    def test_measurement_boundary_names_the_fan(self):
        # The fan is inside the shunt on purpose and the report has to say so; a reader
        # comparing against an SoC-only datasheet figure needs to know why ours is higher.
        boundary = device_info.collect()["power"]["inside_measurement_boundary"]
        self.assertTrue(any("fan" in item.lower() for item in boundary))

    def test_memory_budget_is_reported_alongside_the_physical_total(self):
        mem = device_info.collect()["memory"]
        self.assertEqual(mem["usable_budget_mb"], int(device_info.ENV["NODE_USABLE_MB"]))
        self.assertEqual(mem["rpc_advertised_mb"], int(device_info.ENV["RPC_MEM_MB"]))

    def test_missing_sysfs_paths_do_not_raise(self):
        self.assertEqual(device_info._read("/definitely/not/here", "fallback"), "fallback")
        self.assertEqual(device_info._run(["definitely-not-a-command"]), "")


class AgentTestCase(unittest.TestCase):
    """Starts the real handler on an ephemeral loopback port and speaks HTTP to it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.events = Path(cls.tmp.name) / "logs" / "events.jsonl"
        cls._saved_events = node_agent.EVENTS
        node_agent.EVENTS = cls.events

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), node_agent.Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)
        node_agent.EVENTS = cls._saved_events
        cls.tmp.cleanup()

    def get(self, path):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def post(self, path, payload=None, raw=None):
        url = "http://127.0.0.1:%d%s" % (self.port, path)
        body = raw if raw is not None else json.dumps(payload or {}).encode()
        req = urllib.request.Request(url, data=body, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())


class TestHealthFailsClosed(AgentTestCase):
    def test_health_refuses_readiness_without_instrumentation(self):
        # The whole point: a node that cannot see its own instrumentation must answer 503.
        # The uninstrumented state is CONSTRUCTED -- no chrony, no thermal sensor, no
        # llama.cpp build, no build manifest -- rather than assumed of whatever machine runs
        # the suite. A provisioned Pi 4 has all four, and would answer 200 here.
        real_sh = node_agent.sh

        def no_chrony(cmd, timeout=20):
            if list(cmd[:1]) == ["chronyc"]:
                return 127, "", "chronyc: command not found"
            return real_sh(cmd, timeout)

        no_sensor = {"soc_temp_c": None, "throttled_word": "unavailable", "throttled_decoded": None}
        with tempfile.TemporaryDirectory() as empty, \
                unittest.mock.patch.object(node_agent, "sh", no_chrony), \
                unittest.mock.patch.object(node_agent.device_info, "thermal", return_value=no_sensor), \
                unittest.mock.patch.dict(node_agent.ENV, {"LLAMA_BIN": empty}), \
                unittest.mock.patch.object(node_agent, "DATA_DIR", Path(empty)):
            status, body = self.get("/health")
        self.assertEqual(status, 503)
        self.assertFalse(body["ready"])
        for missing in ("chrony unavailable", "soc_temp_c unreadable", "rpc server not built",
                        "llama-bench not built", "build manifest absent"):
            self.assertTrue(any(missing in b for b in body["blocking"]),
                            "%r must be named in blocking: %r" % (missing, body["blocking"]))

    # The clock tests stub `chronyc tracking`. The control laptop that runs this suite is the
    # cluster's NTP source, so the real command there reports a synced clock, and a test that
    # read it would pass or fail with the host's NTP state instead of with the code.
    CHRONY_SYNCED = (
        "Reference ID    : CB9F4621 (ntp1.bknix.co.th)\n"
        "Stratum         : 2\n"
        "System time     : 0.000197677 seconds slow of NTP time\n"
        "Leap status     : Normal"
    )
    CHRONY_UNSYNCED = (
        "Reference ID    : 00000000 ()\n"
        "Stratum         : 0\n"
        "System time     : 0.000000000 seconds fast of NTP time\n"
        "Leap status     : Not synchronised"
    )
    CLOCK_BLOCKERS = ("clock not synced", "chrony unavailable", "clock offset")

    def _health_with_chrony(self, rc, out):
        """/health with `chronyc tracking` answering (rc, out); every other command is real."""
        real_sh = node_agent.sh

        def fake_sh(cmd, timeout=20):
            if list(cmd[:2]) == ["chronyc", "tracking"]:
                return rc, out, ""
            return real_sh(cmd, timeout)

        with unittest.mock.patch.object(node_agent, "sh", fake_sh):
            _status, body = self.get("/health")
        return body["blocking"], body.get("warnings", [])

    def test_an_unsynced_clock_is_blocking_not_a_warning(self):
        # Energy is attributed to prefill or decode purely by timestamp. Without a shared
        # clock the attribution is wrong while every number still looks plausible.
        blocking, warnings = self._health_with_chrony(0, self.CHRONY_UNSYNCED)
        self.assertTrue(any("clock not synced" in b for b in blocking),
                        "an unsynced clock must block a run; blocking was %r" % blocking)
        self.assertFalse(any("clock" in w.lower() for w in warnings),
                         "the clock belongs in `blocking`, not `warnings`: %r" % warnings)

    def test_missing_chrony_is_blocking(self):
        blocking, _ = self._health_with_chrony(127, "")
        self.assertTrue(any("chrony unavailable" in b for b in blocking),
                        "no chrony must block a run; blocking was %r" % blocking)

    def test_a_synced_clock_does_not_block(self):
        # The converse. Without it, a gate that always blocked on the clock would pass the
        # tests above.
        blocking, _ = self._health_with_chrony(0, self.CHRONY_SYNCED)
        clock = [b for b in blocking if b.startswith(self.CLOCK_BLOCKERS)]
        self.assertEqual(clock, [], "a synced clock must not block; got %r" % clock)

    def test_a_synced_clock_more_than_5_ms_off_is_blocking(self):
        out = self.CHRONY_SYNCED.replace("0.000197677", "0.050000000")
        blocking, _ = self._health_with_chrony(0, out)
        self.assertTrue(any("exceeds 5 ms" in b for b in blocking),
                        "a 50 ms offset must block; blocking was %r" % blocking)

    def test_a_missing_rpc_server_binary_is_blocking(self):
        # CONSTRUCTED: LLAMA_BIN is an empty directory. The live config points at the real
        # build dir, which is empty off-target but holds the binary on the Pi 4 itself.
        with tempfile.TemporaryDirectory() as empty, \
                unittest.mock.patch.dict(node_agent.ENV, {"LLAMA_BIN": empty}):
            _status, body = self.get("/health")
        self.assertTrue(
            any("rpc server not built" in b for b in body["blocking"]),
            "a missing rpc server must block; blocking was %r" % body["blocking"],
        )

    def test_either_upstream_rpc_binary_name_satisfies_the_gate(self):
        # Upstream renamed the target rpc-server -> ggml-rpc-server in c299a92
        # (2026-06-27) and the pinned SHA is chosen on the Jetson, so this node
        # cannot know which name its build produces. Both must count as built --
        # hard-coding either one makes a perfectly good node read as not ready.
        for name in ("ggml-rpc-server", "rpc-server", "llama-rpc-server"):
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / name).write_text("#!/bin/sh\n")
                with unittest.mock.patch.dict(node_agent.ENV, {"LLAMA_BIN": d}):
                    found = node_agent.rpc_binary()
                self.assertIsNotNone(found, "%s was not recognised" % name)
                self.assertEqual(found.name, name)

    def test_an_unrelated_binary_does_not_satisfy_the_gate(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "llama-server").write_text("#!/bin/sh\n")
            with unittest.mock.patch.dict(node_agent.ENV, {"LLAMA_BIN": d}):
                self.assertIsNone(node_agent.rpc_binary())

    def test_an_unpinned_llama_revision_is_reported(self):
        # It is a warning rather than blocking on purpose: the very first bring-up build
        # has to happen before a SHA exists. It must still be visible.
        #
        # This CONSTRUCTS the unpinned state instead of reading the live config/node.env.
        # It used to pass only because this node's pin happened to be empty, so setting
        # the cluster pin on 2026-09-20 turned it red without a single line of agent code
        # changing. Same defect the ThinkPad session fixed in
        # tests/test_nodes.py::test_an_empty_cluster_pin_blocks -- see
        # ThinkPad/.handoff/2026-09-19-1153-NOTE-pin-rollout-and-a-test-that-reads-live-config.md
        with unittest.mock.patch.dict(node_agent.ENV, {"LLAMA_PIN_SHA": ""}):
            _status, body = self.get("/health")
        self.assertTrue(
            any("pin" in w.lower() for w in body["warnings"]),
            "an unpinned revision must at least warn; warnings were %r" % body["warnings"],
        )

    def test_a_pinned_llama_revision_does_not_warn(self):
        # The converse, which had no coverage at all while the repo pin was empty: once a
        # revision IS pinned the warning must go away, or it becomes noise that operators
        # learn to ignore.
        with unittest.mock.patch.dict(
            node_agent.ENV,
            {"LLAMA_PIN_SHA": "b23701f77d47dad9de834d59ebfcbe25c9e8b46f"},
        ):
            _status, body = self.get("/health")
        self.assertFalse(
            any("not pinned" in w.lower() for w in body["warnings"]),
            "a pinned revision must not warn; warnings were %r" % body["warnings"],
        )

    def test_health_carries_identity_and_a_timestamp(self):
        # Identity is echoed from the node's env. Test values, not the live config/node.env,
        # so renaming a node or its role cannot turn this red.
        with unittest.mock.patch.dict(node_agent.ENV, {"NODE_ID": "node-under-test",
                                                       "NODE_ROLE": "role-under-test"}):
            _status, body = self.get("/health")
        self.assertEqual(body["node"], "node-under-test")
        self.assertEqual(body["role"], "role-under-test")
        self.assertIn("checked_at_utc", body)

    def test_health_embeds_the_metrics_sample(self):
        with unittest.mock.patch.dict(node_agent.ENV, {"NODE_ID": "node-under-test"}):
            _status, body = self.get("/health")
        self.assertIn("metrics", body)
        self.assertEqual(body["metrics"]["node"], "node-under-test")


class TestReadEndpoints(AgentTestCase):
    def test_info_serves_the_full_inventory(self):
        status, body = self.get("/info")
        self.assertEqual(status, 200)
        self.assertEqual(body["schema"], "heteroserve.device_info/1")
        self.assertEqual(body["node"]["id"], "pi4")

    def test_metrics_is_cheap_and_well_formed(self):
        status, body = self.get("/metrics")
        self.assertEqual(status, 200)
        for key in ("ts", "node", "soc_temp_c", "throttled_word", "arm_mhz", "governor",
                    "mem_available_mb", "swap_total_mb", "load", "rpc_active"):
            self.assertIn(key, body)
        self.assertIsInstance(body["rpc_active"], bool)

    def test_unknown_get_route_404s_and_lists_the_real_routes(self):
        status, body = self.get("/nope")
        self.assertEqual(status, 404)
        self.assertIn("/health", body["routes"])

    def test_unknown_post_route_404s(self):
        status, body = self.post("/nope")
        self.assertEqual(status, 404)
        self.assertIn("/mark", body["routes"])


class TestMarkers(AgentTestCase):
    """The control node POSTs the same marker to every node so a distributed run has one
    consistent event boundary across four power channels."""

    def test_mark_appends_and_echoes_the_record(self):
        # The node id is echoed from the node's env: a test value, not the live config.
        with unittest.mock.patch.dict(node_agent.ENV, {"NODE_ID": "node-under-test"}):
            status, body = self.post("/mark", {"event": "cluster_run_begin", "note": "run-42"})
        self.assertEqual(status, 200)
        self.assertEqual(body["event"], "cluster_run_begin")
        self.assertEqual(body["note"], "run-42")
        self.assertEqual(body["node"], "node-under-test")
        self.assertIsInstance(body["ts"], float)

    def test_marker_is_persisted_as_one_json_object_per_line(self):
        self.post("/mark", {"event": "unit_test_marker", "note": "n"})
        lines = [l for l in self.events.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.assertTrue(lines)
        records = [json.loads(l) for l in lines]
        self.assertTrue(any(r["event"] == "unit_test_marker" for r in records))

    def test_marker_timestamp_is_a_unix_epoch_second(self):
        # lib.sh writes `date -u +%s.%N` and this writes time.time(). The collector joins
        # power samples against both, so they must be the same epoch and the same units.
        _status, body = self.post("/mark", {"event": "epoch_check"})
        self.assertGreater(body["ts"], 1_700_000_000)
        self.assertLess(body["ts"], 4_000_000_000)

    def test_mark_defaults_are_safe_when_the_body_is_empty(self):
        status, body = self.post("/mark", {})
        self.assertEqual(status, 200)
        self.assertEqual(body["event"], "mark")
        self.assertEqual(body["note"], "")

    def test_a_non_json_body_is_rejected_not_swallowed(self):
        status, body = self.post("/mark", raw=b"this is not json")
        self.assertEqual(status, 400)
        self.assertIn("error", body)

    def test_events_endpoint_tails_the_log(self):
        self.post("/mark", {"event": "tail_me"})
        status, body = self.get("/events?n=5")
        self.assertEqual(status, 200)
        self.assertIsInstance(body, list)
        self.assertLessEqual(len(body), 5)
        self.assertTrue(any(r["event"] == "tail_me" for r in body))

    def test_events_default_tail_works_without_a_query(self):
        status, body = self.get("/events")
        self.assertEqual(status, 200)
        self.assertIsInstance(body, list)


if __name__ == "__main__":
    unittest.main()


class TestFailClosedSignals(unittest.TestCase):
    """The agent's job is to say "do not measure me" when it cannot verify its own state.

    Every check here guards a path that previously read "unknown" as "fine" -- which is the
    worst answer a readiness gate can give, because the run proceeds and the dataset records
    a number nobody can defend.
    """

    def test_an_unreadable_thermal_sensor_blocks_rather_than_reading_as_cold(self):
        # device_info.thermal() yields soc_temp_c=None when /sys/class/thermal is absent.
        # `(x or 0) >= 70` read that as 0 degrees C and reported the node ready.
        with unittest.mock.patch.object(node_agent.device_info, "thermal", return_value={
                "soc_temp_c": None, "throttled_word": "0x0", "throttled_decoded": {}}):
            out = node_agent.health()
        self.assertFalse(out["ready"])
        self.assertTrue(any("soc_temp_c unreadable" in b for b in out["blocking"]),
                        "a dead sensor must block: %r" % (out["blocking"],))

    def test_a_hot_board_still_blocks(self):
        with unittest.mock.patch.object(node_agent.device_info, "thermal", return_value={
                "soc_temp_c": 81.0, "throttled_word": "0x0", "throttled_decoded": {}}):
            out = node_agent.health()
        self.assertFalse(out["ready"])
        self.assertTrue(any("soc_temp_c=81.0" in b for b in out["blocking"]))

    def test_rpc_listening_is_false_when_nothing_is_bound(self):
        """`systemctl start` returns 0 as soon as a Type=simple process is forked, and
        upstream rpc-server exits 0 on a bad flag -- so the port is the only honest
        liveness signal. It must report False quickly rather than hang the agent."""
        import socket as _socket
        probe = _socket.socket()
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]
        probe.close()                      # nothing is listening there now
        self.assertFalse(node_agent.rpc_listening(port=free_port, attempts=2, delay=0.01))

    def test_rpc_listening_is_true_when_something_accepts(self):
        import socket as _socket
        srv = _socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        try:
            port = srv.getsockname()[1]
            self.assertTrue(node_agent.rpc_listening(port=port, attempts=2, delay=0.01))
        finally:
            srv.close()

    def test_unit_restart_count_degrades_to_none_rather_than_raising(self):
        """`is-active` cannot tell "serving" from "crash-looping"; NRestarts can. Off a Pi
        there is no systemd, so it must return None, not blow up the metrics endpoint."""
        self.assertIn(node_agent._unit_restarts("definitely-not-a-real-unit"),
                      (None, 0), "must be None or a count, never an exception")


class TestMeasurementPreconditions(unittest.TestCase):
    """The checks `scripts/99_healthcheck.sh` had that `/health` did not.

    This matters because `/health` is the gate that actually runs before a cluster run --
    the control node polls it; nobody SSHes in to run `make health` first. A check that
    exists only in the shell gate is, in practice, not enforced.

    Each test drives `_measurement_preconditions()` directly and asserts on the REASON,
    not just on a count: the control node prints `blocking` verbatim to whoever aborted the
    run, so "not ready" with no cause is how an operator starts guessing.
    """

    def names(self, **env):
        with unittest.mock.patch.dict(node_agent.ENV, env):
            return [name for name, _detail in node_agent._measurement_preconditions()]

    def test_undeclared_shaping_blocks(self):
        """A forgotten netem rule silently degrades this run and every later one."""
        with unittest.mock.patch.object(node_agent.device_info, "network", return_value={
                "eth0": {"qdisc": "qdisc netem 10: root refcnt 2 limit 1000 delay 1.0ms"}}):
            with unittest.mock.patch.object(node_agent, "_net_tier_declared_shaped",
                                            return_value=(False, None)):
                found = self.names(ETH_IFACE="eth0")
        self.assertTrue(any("shaping does not match" in n for n in found), found)

    def test_shaping_that_was_declared_does_NOT_block(self):
        """The interconnect sweep shapes the link on purpose. A gate that refused every
        shaped point would be bypassed for that entire sweep, which is worse than useless."""
        with unittest.mock.patch.object(node_agent.device_info, "network", return_value={
                "eth0": {"qdisc": "qdisc htb 1: root refcnt 2 ... qdisc netem 10:"}}):
            with unittest.mock.patch.object(node_agent, "_net_tier_declared_shaped",
                                            return_value=(True, {"tier": "shaped"})):
                found = self.names(ETH_IFACE="eth0")
        self.assertFalse(any("shaping" in n for n in found), found)

    def test_a_declared_tier_that_vanished_also_blocks(self):
        """Declared shaped, link is not: the row would be labelled with a tier the link is
        not running, which is a wrong label rather than a missing one."""
        with unittest.mock.patch.object(node_agent.device_info, "network", return_value={
                "eth0": {"qdisc": "qdisc pfifo_fast 0: root refcnt 2"}}):
            with unittest.mock.patch.object(node_agent, "_net_tier_declared_shaped",
                                            return_value=(True, {"tier": "shaped"})):
                found = self.names(ETH_IFACE="eth0")
        self.assertTrue(any("shaping does not match" in n for n in found), found)

    def test_a_missing_llama_bench_blocks_before_probing_it(self):
        with tempfile.TemporaryDirectory() as d:
            found = self.names(LLAMA_BIN=d)
        self.assertTrue(any("llama-bench not built" in n for n in found), found)
        self.assertFalse(any("--progress" in n for n in found),
                         "must not claim a flag verdict for a binary that is absent")

    def test_a_llama_bench_without_progress_blocks(self):
        """Without --progress the energy window silently falls back to whole-process, and
        the CSV still looks self-consistent -- nothing downstream would reveal it."""
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "llama-bench"
            fake.write_text("#!/bin/sh\necho 'usage: llama-bench [-m MODEL] [-r REPS]'\n")
            fake.chmod(0o755)
            found = self.names(LLAMA_BIN=d)
        self.assertTrue(any("no --progress" in n for n in found), found)

    def test_a_llama_bench_with_progress_passes_that_check(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "llama-bench"
            fake.write_text("#!/bin/sh\necho '  --progress   print progress'\n")
            fake.chmod(0o755)
            found = self.names(LLAMA_BIN=d)
        self.assertFalse(any("--progress" in n for n in found), found)

    def test_the_reason_is_carried_not_just_the_verdict(self):
        with tempfile.TemporaryDirectory() as d:
            with unittest.mock.patch.dict(node_agent.ENV, {"LLAMA_BIN": d}):
                items = list(node_agent._measurement_preconditions())
        self.assertTrue(items)
        for name, detail in items:
            with self.subTest(name=name):
                self.assertTrue(detail and detail.strip(),
                                "every blocking reason must say what to do about it")

    def test_a_pin_that_disagrees_with_the_built_commit_blocks(self):
        """A pin that does not describe the binary on disk is worse than no pin: the row
        records a revision that never ran, and nothing else in the pipeline would notice."""
        with tempfile.TemporaryDirectory() as d:
            inv = Path(d) / "inventory"
            inv.mkdir()
            (inv / "build_manifest.json").write_text(
                json.dumps({"llama_cpp": {"commit": "aaaaaaaaaaaa"}}), encoding="utf-8")
            with unittest.mock.patch.object(node_agent, "DATA_DIR", Path(d)):
                found = self.names(LLAMA_PIN_SHA="bbbbbbbbbbbb")
        self.assertTrue(any("does not match the built commit" in n for n in found), found)

    def test_a_pin_that_matches_the_built_commit_passes(self):
        with tempfile.TemporaryDirectory() as d:
            inv = Path(d) / "inventory"
            inv.mkdir()
            (inv / "build_manifest.json").write_text(
                json.dumps({"llama_cpp": {"commit": "aaaaaaaaaaaa"}}), encoding="utf-8")
            with unittest.mock.patch.object(node_agent, "DATA_DIR", Path(d)):
                found = self.names(LLAMA_PIN_SHA="aaaaaaaaaaaa")
        self.assertFalse(any("built commit" in n for n in found), found)
        self.assertFalse(any("build manifest absent" in n for n in found), found)

    def test_an_unpinned_node_does_not_trip_the_mismatch_check(self):
        """Empty is the documented bring-up state; it warns elsewhere, it must not block here."""
        with tempfile.TemporaryDirectory() as d:
            inv = Path(d) / "inventory"
            inv.mkdir()
            (inv / "build_manifest.json").write_text(
                json.dumps({"llama_cpp": {"commit": "aaaaaaaaaaaa"}}), encoding="utf-8")
            with unittest.mock.patch.object(node_agent, "DATA_DIR", Path(d)):
                found = self.names(LLAMA_PIN_SHA="")
        self.assertFalse(any("built commit" in n for n in found), found)
