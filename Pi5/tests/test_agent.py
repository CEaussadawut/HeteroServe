"""Tests for node/agent.py -- the readiness contract the cluster preflight gate rests on.

ThinkPad/control/nodes.py polls GET /health and reads two keys from the body: `ready` and
`blocking`. This agent used to answer /health with a liveness body containing neither, so
nodes.py recorded ready=None, iterated an empty blocking list, and the cluster gate
declared READY no matter how throttled, hot, swapped or unbuilt this node was. The
readiness logic existed the whole time -- it was served at /ready, a path the control node
never calls. A gate that cannot fail is worse than no gate, because it is reported as a
pass.

These tests drive the real ThreadingHTTPServer over a real socket rather than calling the
handler directly, because the defect was in routing, not in the readiness logic.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "node"))

import envfile  # noqa: E402  -- resolved through the sys.path insert above;
# agent.py imports the same module, so get_int coercion here matches the code under test.


def _load():
    spec = importlib.util.spec_from_file_location("hs_agent", ROOT / "node" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


agent = _load()

ALL_PASS = {
    "governor_pinned": True, "no_swap_in_use": True, "not_throttled": True,
    "no_under_voltage": True, "cool_enough_to_start": True,
    "advertised_memory_available": True, "wired_link_up": True,
}
SAMPLE = {"node_id": "pi5", "soc_temp_c": 48.0, "swap_used_mb": 0}
LINK = {"present": True, "operstate": "up", "speed_mbit": "1000"}


class _DeepBacklogServer(ThreadingHTTPServer):
    """The same server the agent runs, with a deeper listen backlog for the test only.

    `HTTPServer.request_queue_size` is 5. TestConcurrentAccess opens 24 connections at once
    to prove the marker append is atomic, which overflows that backlog and makes the kernel
    reset the excess -- a ~4% `ConnectionResetError` that took 24 consecutive runs to
    reproduce and reads exactly like a lost marker.

    This is a property of the TEST's load, not of the agent: in production the control node
    polls a handful of nodes at a low rate, and the harness posts two markers per cell, so a
    backlog of 5 is ample. Raising it here keeps the concurrency the test is actually about
    and removes the artefact that was masquerading as one.
    """
    request_queue_size = 128
    daemon_threads = True


class TestRealReadinessGates(unittest.TestCase):
    """AgentCase stubs readiness() wholesale, so nothing in this file exercised the gate
    logic itself. These drive the real readiness() with only telemetry.sample stubbed, which
    is where the fail-open bugs lived."""

    def _gates(self, **overrides):
        base = {
            "governor": "performance", "swap_used_mb": 0,
            "throttled_now": 0, "under_voltage_now": 0, "throttled_hex": "0x0",
            "soc_temp_c": 44.0, "mem_avail_mb": 7000, "rpc_rss_mb": 0,
        }
        port_open = base.pop("_port_open", True)
        base.update(overrides)
        port_open = base.pop("_port_open", port_open)
        orig_sample, orig_port = agent.telemetry.sample, agent._port_open
        agent.telemetry.sample = lambda node_id: dict(base)
        agent._port_open = lambda host, port, timeout=0.4: port_open
        try:
            return agent.readiness()[0]
        finally:
            agent.telemetry.sample, agent._port_open = orig_sample, orig_port

    def test_an_unreadable_throttle_word_blocks(self):
        """telemetry.read_throttled returns -1 when vcgencmd is missing, off the unit's
        PATH, or times out, which renders as "". `in (0, "")` read that as not throttled, so
        a node whose firmware word could not be read passed the gate and went on to produce
        measurements nobody could validate. Pi 4's parser states the rule: fail closed."""
        g = self._gates(throttled_now="", under_voltage_now="", throttled_hex="")
        self.assertFalse(g["throttle_word_readable"])
        # Pin each gate separately, not just the overall verdict. These are currently
        # redundant -- throttle_word_readable blocks on its own -- but the original bug was
        # `in (0, "")` treating an unreadable word as clean, and without this assertion that
        # could be reintroduced and stay invisible until someone removed the other gate.
        self.assertFalse(g["not_throttled"], "an unreadable word read as 'not throttled'")
        self.assertFalse(g["no_under_voltage"])

    def test_a_readable_clean_word_passes(self):
        g = self._gates(throttled_now=0, under_voltage_now=0)
        self.assertTrue(g["throttle_word_readable"])
        self.assertTrue(g["not_throttled"])
        self.assertTrue(g["no_under_voltage"])

    def test_an_actual_throttle_blocks(self):
        self.assertFalse(self._gates(throttled_now=1)["not_throttled"])

    def test_a_dead_thermal_sensor_does_not_pass_the_start_gate(self):
        """common.sh falls back to `echo 0`, and -0.0 is the same value after a float
        round-trip. Both used to satisfy `soc_temp_c <= start_max`, so the gate whose whole
        job is refusing to start warm passed on a board nobody had taken the temperature
        of."""
        self.assertFalse(self._gates(soc_temp_c=0.0)["cool_enough_to_start"])
        self.assertFalse(self._gates(soc_temp_c=-0.0)["cool_enough_to_start"])
        self.assertFalse(self._gates(soc_temp_c="")["cool_enough_to_start"])

    def test_a_real_cool_reading_passes_and_a_hot_one_blocks(self):
        self.assertTrue(self._gates(soc_temp_c=44.0)["cool_enough_to_start"])
        self.assertFalse(self._gates(soc_temp_c=70.0)["cool_enough_to_start"])

    # Each entry: gate name, a sample override that must make it FAIL, and one that must
    # make it PASS. Table-driven so every gate has a witness in both directions -- the
    # previous tests covered three of eight, and mutations hardwiring the others to True,
    # or inverting them outright, survived the whole suite. A readiness gate that cannot
    # fail is worse than no gate, because the cluster is told the node is fit to measure.
    GATE_CASES = [
        ("governor_pinned",             {"governor": "ondemand"},        {"governor": "performance"}),
        ("no_swap_in_use",              {"swap_used_mb": 128},           {"swap_used_mb": 0}),
        ("not_throttled",               {"throttled_now": 1},            {"throttled_now": 0}),
        ("no_under_voltage",            {"under_voltage_now": 1},        {"under_voltage_now": 0}),
        ("throttle_word_readable",      {"throttled_now": ""},           {"throttled_now": 0}),
        ("cool_enough_to_start",        {"soc_temp_c": 70.0},            {"soc_temp_c": 44.0}),
        ("advertised_memory_available", {"mem_avail_mb": 10, "rpc_rss_mb": 0},
                                        {"mem_avail_mb": 7000, "rpc_rss_mb": 0}),
        # The gate that answers "is there anything to measure?". It was absent from this
        # table and from ALL_PASS, and _port_open was never stubbed, so hardwiring it True
        # survived the whole suite: the node reports READY with nothing listening, the
        # control node commits a placement, and whatever latency is recorded is attributed
        # to a Pi 5 partition that was never served.
        ("rpc_server_listening",        {"_port_open": False},           {"_port_open": True}),
    ]

    def test_every_gate_fails_on_its_own_bad_input(self):
        for name, bad, _good in self.GATE_CASES:
            with self.subTest(gate=name):
                self.assertFalse(self._gates(**bad)[name],
                                 "%s stayed True on input that should block it" % name)

    def test_every_gate_passes_on_its_own_good_input(self):
        """The mirror. Without it, a gate hardwired to False would look like a passing test
        suite with a node that can never run."""
        for name, _bad, good in self.GATE_CASES:
            with self.subTest(gate=name):
                self.assertTrue(self._gates(**good)[name],
                                "%s stayed False on input that should satisfy it" % name)

    def test_a_bad_gate_actually_blocks_the_node(self):
        """Gate values are only useful if they reach the verdict. Assert the whole path:
        bad input -> gate False -> ready False -> HTTP 503 -> a blocking reason."""
        for name, bad, _good in self.GATE_CASES:
            with self.subTest(gate=name):
                orig = agent.readiness
                gates = dict(ALL_PASS)
                gates[name] = False
                agent.readiness = lambda: (gates, dict(SAMPLE), dict(LINK))
                try:
                    h = agent.health()
                finally:
                    agent.readiness = orig
                self.assertFalse(h["ready"], "%s False did not make the node unready" % name)
                self.assertTrue(h["blocking"], "%s False produced no blocking reason" % name)

    def test_the_wired_link_gate_is_not_inverted(self):
        """`wired_link_up` reads from a different source than the rest (the iface dict), so
        it is not in the table above. Inverting it survived the whole suite -- a node with a
        dead NIC would have reported itself ready."""
        orig = agent._iface
        try:
            agent._iface = lambda name: {"present": True, "operstate": "up"}
            self.assertTrue(self._gates()["wired_link_up"])
            agent._iface = lambda name: {"present": True, "operstate": "down"}
            self.assertFalse(self._gates()["wired_link_up"])
            agent._iface = lambda name: {"present": False}
            self.assertFalse(self._gates()["wired_link_up"])
        finally:
            agent._iface = orig

    def test_the_gate_table_covers_every_gate_readiness_produces(self):
        """Stop the table drifting from production. `rpc_server_listening` and
        `throttle_word_readable` were both set by readiness() and absent from the tests'
        model of it, so neither had a witness."""
        produced = set(self._gates())
        tabled = {name for name, _bad, _good in self.GATE_CASES}
        missing = produced - tabled - {"wired_link_up"}   # covered by its own test
        self.assertEqual(missing, set(),
                         "gates with no entry in GATE_CASES: %s" % sorted(missing))

    def test_port_open_reports_a_real_socket(self):
        """_port_open had no test of any kind. If it swallowed OSError as True, a refused
        connection would read as 'rpc-server is up'."""
        import socket
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        try:
            self.assertTrue(agent._port_open("127.0.0.1", port))
        finally:
            srv.close()
        # same port, nothing listening now
        self.assertFalse(agent._port_open("127.0.0.1", port))

    def test_port_open_does_not_raise_on_a_bad_host(self):
        self.assertFalse(agent._port_open("no-such-host.invalid", 50052))

    def test_every_gate_has_an_explanation(self):
        """A gate whose name is missing from _BLOCKING_TEXT is reported to the control node
        as a bare identifier -- the rpc-server gate was renamed and its sentence was left
        keyed to the old name, so its refusal read as "rpc_server_listening"."""
        missing = set(self._gates()) - set(agent._BLOCKING_TEXT)
        self.assertEqual(missing, set(), "gates with no sentence: %s" % sorted(missing))


class TestReadinessReadsTheConfiguredSources(unittest.TestCase):
    """GATE_CASES proves each gate's COMPARISON is right. It cannot prove the gate is
    reading the right THING.

    `_gates` stubs telemetry.sample and _port_open with lambdas that discard their
    arguments, and never stubs _iface at all. So a mutation that changes WHICH SOURCE a
    gate reads changes no comparison, and the entire suite stays green. Four such were
    confirmed to survive a mutation run on 2026-09-19:

        telemetry.sample(ENV["NODE_ID"])  ->  telemetry.sample("pi4")
        _iface(ENV.get("ETH_IFACE"))      ->  _iface(ENV.get("WLAN_IFACE", "wlan0"))
        mem_avail_mb >= RPC_MEM_MB        ->  mem_avail_mb >= RPC_MEM_MB // 2
        _port_open(RPC_HOST, RPC_PORT)    ->  a different endpoint

    Each produces a node that reports itself fit to measure while sampling another node's
    telemetry, reading the wrong NIC, or honouring half the memory it advertises -- and in
    a characterisation dataset that is the defect class this project exists to avoid: not a
    crash, but a plausible number attributed to the wrong device. The gate logic is not the
    contract; the wiring from node.env to the gate input is.
    """

    SAMPLE = {
        "governor": "performance", "swap_used_mb": 0,
        "throttled_now": 0, "under_voltage_now": 0, "throttled_hex": "0x0",
        "soc_temp_c": 44.0, "mem_avail_mb": 7000, "rpc_rss_mb": 0,
    }

    def _seen(self, **sample_overrides):
        """Drive the real readiness() and record what each collaborator was ASKED for."""
        sample = dict(self.SAMPLE, **sample_overrides)
        seen = {}
        orig = (agent.telemetry.sample, agent._port_open, agent._iface)

        def sample_fn(node_id):
            seen["node_id"] = node_id
            return dict(sample)

        def port_fn(host, port, timeout=0.4):
            seen["endpoint"] = (host, port)
            return True

        def iface_fn(name):
            seen["iface"] = name
            return {"operstate": "up", "speed_mbit": "1000", "carrier": "1"}

        agent.telemetry.sample, agent._port_open, agent._iface = sample_fn, port_fn, iface_fn
        try:
            gates, _s, _link = agent.readiness()
        finally:
            agent.telemetry.sample, agent._port_open, agent._iface = orig
        return seen, gates

    def test_telemetry_is_sampled_for_this_node_not_a_hardcoded_one(self):
        """telemetry.sample stamps its argument into the returned dict as node_id, and the
        control node groups rows by it. A hardcoded id means this node's numbers are filed
        under another device -- silently, and forever, because nothing downstream can tell
        a mislabelled row from a real one."""
        seen, _ = self._seen()
        self.assertEqual(seen["node_id"], agent.ENV["NODE_ID"],
                         "readiness() sampled telemetry for %r, not this node (%r)"
                         % (seen.get("node_id"), agent.ENV["NODE_ID"]))

    def test_the_wired_link_gate_reads_the_interface_named_in_node_env(self):
        """wired_link_up is what asserts this node is on the measured tier. Read the wrong
        interface and a node sitting on Wi-Fi reports a healthy wired link, so every comm
        term derived from that tier is attributed to a cable that was never used."""
        seen, _ = self._seen()
        self.assertEqual(seen["iface"], agent.ENV["ETH_IFACE"],
                         "the wired-link gate read %r, not ETH_IFACE (%r)"
                         % (seen.get("iface"), agent.ENV["ETH_IFACE"]))

    def test_the_rpc_probe_targets_the_configured_endpoint(self):
        """rpc_server_listening answers "is there anything to measure?". Probing a
        different host or port makes it answer about some other socket."""
        seen, _ = self._seen()
        self.assertEqual(
            seen["endpoint"],
            (agent.ENV.get("RPC_HOST", "0.0.0.0"), agent.ENV.get("RPC_PORT", "50052")),
            "the rpc probe targeted %r, not the configured endpoint" % (seen.get("endpoint"),),
        )

    def test_the_memory_gate_requires_the_WHOLE_advertised_budget(self):
        """A boundary, not a magnitude. GATE_CASES uses 10 MB (fails) and 7000 MB (passes),
        which straddle every fraction of RPC_MEM_MB as well as the real figure -- so
        `>= RPC_MEM_MB // 2`, or any other discount, satisfies both cases. Pin the edge:
        one megabyte short of the advertised budget must block, exactly the budget must
        pass. RPC_MEM_MB is the number the placement engine is told it may assign; honour
        half of it and the node accepts a partition it cannot hold, then swaps mid-run --
        which is a corrupted measurement, not a slow one."""
        budget = envfile.get_int(agent.ENV, "RPC_MEM_MB", 0)
        self.assertGreater(budget, 0, "RPC_MEM_MB must be set for this test to mean anything")

        _, short = self._seen(mem_avail_mb=budget - 1, rpc_rss_mb=0)
        self.assertFalse(short["advertised_memory_available"],
                         "the node passed the memory gate %d MB short of the %d MB it "
                         "advertises" % (1, budget))

        _, exact = self._seen(mem_avail_mb=budget, rpc_rss_mb=0)
        self.assertTrue(exact["advertised_memory_available"],
                        "exactly the advertised budget must satisfy the gate")

    def test_an_already_loaded_server_still_satisfies_the_memory_gate(self):
        """The `or rpc_rss_mb > 0` leg: once the model is resident the memory is in use,
        not missing. Without this, the boundary test above could be satisfied by deleting
        that leg, which would make every mid-run readiness check fail."""
        _, g = self._seen(mem_avail_mb=1, rpc_rss_mb=4096)
        self.assertTrue(g["advertised_memory_available"])


class TestInfoIsDerivedFromNodeEnv(unittest.TestCase):
    """/info is this node's card in the cluster inventory: the control node reads it to
    decide what to place here and how to attribute what comes back. The existing tests
    check that two fields are present and correct; the rest were asserted only to EXIST,
    so mutations hardcoding `arch`, doubling `ram_mb`, reading `cores` from the wrong key,
    shifting `ina_channel` by one, or pointing `rpc_endpoint` at the control host all
    survived. A field that is present but wrong is worse than a missing one -- the missing
    one is noticed.

    `ina_channel` is the sharpest of these: it selects the INA226 shunt, i.e. WHICH POWER
    RAIL the joules come from. Off by one and every energy figure for this node is measured
    from another device, with nothing in the number to show for it.

    Table-driven so a new /info field is a deliberate decision: add it here with its source
    or state why it has none.
    """

    # info() key -> how to derive it from ENV. Nested blocks use a dotted path.
    DERIVED = [
        ("node_id",              lambda E: E["NODE_ID"]),
        ("role",                 lambda E: E.get("NODE_ROLE", "")),
        ("desc",                 lambda E: E.get("NODE_DESC", "")),
        ("soc",                  lambda E: E.get("NODE_SOC", "")),
        ("cpu",                  lambda E: E.get("NODE_CPU", "")),
        ("cores",                lambda E: envfile.get_int(E, "NODE_CORES", 4)),
        ("arch",                 lambda E: E.get("NODE_ARCH", "")),
        ("ram_mb",               lambda E: envfile.get_int(E, "NODE_RAM_MB", 0)),
        ("usable_mb",            lambda E: envfile.get_int(E, "NODE_USABLE_MB", 0)),
        ("rpc_mem_mb",           lambda E: envfile.get_int(E, "RPC_MEM_MB", 0)),
        ("rpc_endpoint",         lambda E: "%s:%s" % (E.get("ETH_ADDR", ""), E.get("RPC_PORT", ""))),
        ("llama_cpp_sha",        lambda E: E.get("LLAMA_PIN_SHA", "")),
        ("arch_flags",           lambda E: E.get("LLAMA_ARCH_FLAGS", "")),
        ("threads",              lambda E: envfile.get_int(E, "LLAMA_THREADS", 4)),
        ("ina_channel",          lambda E: envfile.get_int(E, "INA_CHANNEL", 0)),
        ("software.llama_pin_sha", lambda E: E.get("LLAMA_PIN_SHA", "")),
        ("software.arch_flags",    lambda E: E.get("LLAMA_ARCH_FLAGS", "")),
        ("software.threads",       lambda E: envfile.get_int(E, "LLAMA_THREADS", 4)),
        ("memory.usable_budget_mb", lambda E: envfile.get_int(E, "RPC_MEM_MB", 0)),
        ("memory.node_usable_mb",   lambda E: envfile.get_int(E, "NODE_USABLE_MB", 0)),
        ("memory.total_mb",         lambda E: envfile.get_int(E, "NODE_RAM_MB", 0)),
    ]

    @staticmethod
    def _dig(body, path):
        cur = body
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        return cur

    def test_every_info_field_equals_its_node_env_source(self):
        body = agent.info()
        for path, derive in self.DERIVED:
            with self.subTest(field=path):
                self.assertEqual(self._dig(body, path), derive(agent.ENV),
                                 "/info %s does not match its node.env source" % path)

    def test_the_agent_start_time_is_the_process_start_not_the_request_time(self):
        """`agent_started_utc` is the one field DERIVED legitimately excludes -- it is a
        clock reading, not a config value -- which left it with no coverage at all. Reading
        the current time here makes every /info poll report an agent that just started, so
        a unit that has been restarting in a loop looks indistinguishable from a stable
        one, and the restart is invisible in the cluster inventory."""
        from datetime import datetime as _dt
        orig = agent.STARTED
        agent.STARTED = time.time() - 3600.0
        self.addCleanup(lambda: setattr(agent, "STARTED", orig))
        stamp = agent.info()["agent_started_utc"]
        age = time.time() - _dt.fromisoformat(stamp).timestamp()
        self.assertGreater(age, 3000.0,
                           "agent_started_utc reports the current time, not the process "
                           "start (got %r, age %.0fs)" % (stamp, age))

    def test_the_table_covers_every_field_info_actually_emits(self):
        """Without this, adding a field to info() and forgetting to add it here would leave
        it unchecked -- which is exactly how the fields above went unasserted in the first
        place. `agent_started_utc` is the one legitimate exclusion: it is a clock reading,
        not a config value."""
        body = agent.info()
        checked = {p.split(".")[0] for p, _ in self.DERIVED}
        nested = {p.split(".")[0] for p, _ in self.DERIVED if "." in p}
        for key, value in body.items():
            if key == "agent_started_utc":
                continue
            with self.subTest(field=key):
                self.assertIn(key, checked, "/info emits %r with no declared source" % key)
            if isinstance(value, dict):
                self.assertIn(key, nested, "/info block %r has no per-field checks" % key)
                for sub in value:
                    with self.subTest(field="%s.%s" % (key, sub)):
                        self.assertIn("%s.%s" % (key, sub), [p for p, _ in self.DERIVED],
                                      "/info %s.%s has no declared source" % (key, sub))

    def test_the_advertised_budget_stays_below_what_the_node_has(self):
        """Not a drift guard -- an invariant. Advertising at or above usable_mb defeats the
        headroom the whole memory design rests on."""
        body = agent.info()
        self.assertLess(body["rpc_mem_mb"], body["usable_mb"])
        self.assertLessEqual(body["usable_mb"], body["ram_mb"])


class AgentCase(unittest.TestCase):
    gates = ALL_PASS

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

        self._orig_readiness = agent.readiness
        self._orig_sample = agent.telemetry.sample
        self._orig_events = agent.EVENTS
        agent.EVENTS = Path(self._tmp.name) / "logs" / "events.jsonl"
        agent.telemetry.sample = lambda node_id: dict(SAMPLE)
        gates = self.gates
        agent.readiness = lambda: (dict(gates), dict(SAMPLE), dict(LINK))

        def restore():
            agent.readiness = self._orig_readiness
            agent.telemetry.sample = self._orig_sample
            agent.EVENTS = self._orig_events
        self.addCleanup(restore)

        self.srv = _DeepBacklogServer(("127.0.0.1", 0), agent.Handler)
        self.port = self.srv.server_address[1]
        # poll_interval, not the 0.5 s default: `shutdown()` blocks until the serve loop
        # notices, so the default charges every HTTP test ~0.5 s of pure waiting. With ~47
        # such tests that was 23 of this file's 24 seconds -- and mutation testing pays it
        # once per mutation, which is where it actually hurts.
        t = threading.Thread(target=lambda: self.srv.serve_forever(poll_interval=0.01),
                             daemon=True)
        t.start()
        self.addCleanup(self.srv.server_close)
        self.addCleanup(self.srv.shutdown)

    def get(self, path):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d%s" % (self.port, path)) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def post(self, path, payload):
        req = urllib.request.Request(
            "http://127.0.0.1:%d%s" % (self.port, path),
            data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


class TestHealthyNode(AgentCase):
    gates = ALL_PASS

    def test_health_answers_the_readiness_contract(self):
        """The two keys ThinkPad/control/nodes.py:139-145 actually reads."""
        status, body = self.get("/health")
        self.assertEqual(status, 200)
        self.assertIn("ready", body)
        self.assertIn("blocking", body)
        self.assertIs(body["ready"], True)
        self.assertEqual(body["blocking"], [])

    # Fields that legitimately differ between two sequential requests. Comparing whole
    # bodies made this test FLAKY: `uptime_s` is round(time.time() - STARTED, 1), so two
    # calls either side of a 0.1 s boundary gave 15.3 vs 15.4 and the assertion failed
    # roughly one run in six. A flaky test in a verification suite is corrosive -- it
    # teaches the reader to re-run until green, which is exactly how a real regression gets
    # waved through.
    TIME_VARYING = ("uptime_s", "utc")

    def _verdict(self, body):
        return {k: v for k, v in body.items() if k not in self.TIME_VARYING}

    def test_ready_and_health_agree(self):
        """06_verify_node.sh polls /ready; the control node polls /health. Serving two
        different verdicts on those paths is how the gate went missing."""
        ready_status, ready = self.get("/ready")
        health_status, health = self.get("/health")
        self.assertEqual(ready_status, health_status)
        self.assertEqual(self._verdict(ready), self._verdict(health))

    def test_both_paths_carry_the_time_varying_fields(self):
        """The exclusion above must not become a hole: assert the fields are present and
        of the right type on both paths, just not equal across two calls."""
        for path in ("/ready", "/health"):
            with self.subTest(path=path):
                _status, body = self.get(path)
                self.assertIsInstance(body["uptime_s"], float)
                self.assertGreaterEqual(body["uptime_s"], 0.0)
                self.assertTrue(body["utc"].endswith("+00:00"))

    def test_root_also_answers(self):
        self.assertEqual(self.get("/")[0], 200)

    def test_metrics_is_an_alias_for_stat(self):
        """nodes.py:60 fetches /metrics. This node only had /stat, so it 404'd."""
        s_status, s_body = self.get("/stat")
        m_status, m_body = self.get("/metrics")
        self.assertEqual(s_status, 200)
        self.assertEqual(m_status, 200)
        self.assertEqual(s_body.keys(), m_body.keys())


class TestBlockedNode(AgentCase):
    gates = dict(ALL_PASS, no_swap_in_use=False, not_throttled=False)

    def test_blocked_node_is_503(self):
        status, body = self.get("/health")
        self.assertEqual(status, 503)
        self.assertIs(body["ready"], False)

    def test_blocking_reasons_reach_the_control_node(self):
        """The regression that matters: nodes.py iterates h.get("blocking", []) and adds
        each reason to the cluster report. An empty list here means a throttled, swapping
        node is reported as fit to measure."""
        _, body = self.get("/health")
        self.assertTrue(body["blocking"], "a blocked node contributed no blocking reasons")
        joined = " ".join(body["blocking"]).lower()
        self.assertIn("swap", joined)
        self.assertIn("throttl", joined)

    def test_reasons_are_sentences_not_gate_keys(self):
        _, body = self.get("/health")
        self.assertTrue(all(" " in r for r in body["blocking"]),
                        "blocking reasons should read as sentences: %r" % body["blocking"])

    def test_failed_gate_keys_are_still_available(self):
        """docs/SETUP.md troubleshooting refers to gates by name, so keep them."""
        _, body = self.get("/health")
        self.assertIn("no_swap_in_use", body["failed"])
        self.assertIn("gates", body)


class TestControlNodeContract(AgentCase):
    """These mirror the exact expressions ThinkPad/control/nodes.py and profiles.py use.
    Each was a silent degradation: the control node reads with `.get(...) or {}`, so a
    missing block produced None and a check that could never fail, not an error."""

    def test_info_carries_the_pin_where_the_revision_gate_looks(self):
        """nodes.py:188 -- (info(node).get("software") or {}).get("llama_pin_sha").
        Under a flat `llama_cpp_sha` this returned None and the cluster-wide revision gate
        degraded from blocking to a warning, defeating the point of pinning one SHA."""
        _, body = self.get("/info")
        self.assertEqual((body.get("software") or {}).get("llama_pin_sha"),
                         agent.ENV.get("LLAMA_PIN_SHA", ""))

    def test_info_carries_the_memory_budget_profiles_reads(self):
        """profiles.py:154-156 -- inv.get("memory", {}).get("usable_budget_mb").
        Absent, build_devices falls back to cluster.json's configured figure."""
        _, body = self.get("/info")
        budget = (body.get("memory") or {}).get("usable_budget_mb")
        self.assertTrue(budget, "no memory.usable_budget_mb for profiles.build_devices")
        # RPC_MEM_MB, not NODE_USABLE_MB -- advertising less than we have is deliberate.
        self.assertEqual(budget, int(agent.ENV["RPC_MEM_MB"]))
        self.assertLess(budget, int(agent.ENV["NODE_USABLE_MB"]))

    def test_health_carries_metrics_where_preflight_looks(self):
        """nodes.py:161 -- (h.get("metrics") or {}).get("soc_temp_c"). Under "sample"
        alone the cluster preflight report printed temp=None for this node."""
        _, body = self.get("/health")
        self.assertEqual((body.get("metrics") or {}).get("soc_temp_c"),
                         SAMPLE["soc_temp_c"])


class TestMalformedRequests(AgentCase):
    """`5`, `"x"`, `[1,2]` and `null` are all valid JSON. .get() on any of them raises
    AttributeError, which the body-parse clause did not catch, so the handler died and the
    client got no response at all."""

    def test_non_object_bodies_are_400(self):
        for raw in (b"5", b'"x"', b"[1,2]", b"null"):
            req = urllib.request.Request(
                "http://127.0.0.1:%d/mark" % self.port, data=raw, method="POST")
            try:
                with urllib.request.urlopen(req) as r:
                    self.fail("accepted a non-object body %r (status %s)" % (raw, r.status))
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 400, "body %r" % raw)
                self.assertIn("object", json.loads(e.read())["error"])

    def test_an_object_body_still_works_afterwards(self):
        self.assertEqual(self.post("/mark", {"event": "run_begin"})[0], 200)


class TestEventLog(AgentCase):
    def test_mark_appends_and_is_readable(self):
        """harness.mark_all POSTs run_begin/run_end. Without them analysis.py finds no
        window for this node and its active power comes out None."""
        status, row = self.post("/mark", {"event": "run_begin", "note": "cell-1"})
        self.assertEqual(status, 200)
        self.assertEqual(row["event"], "run_begin")
        self.assertEqual(row["note"], "cell-1")
        self.assertEqual(row["node_id"], "pi5")

        status, events = self.get("/events")
        self.assertEqual(status, 200)
        self.assertEqual([e["event"] for e in events], ["run_begin"])

    def test_events_carry_a_timestamp_and_monotonic_clock(self):
        """The energy join needs a wall-clock instant; the monotonic value survives an
        NTP step mid-run, which the wall clock does not."""
        _, row = self.post("/mark", {"event": "run_end"})
        # `endswith("+00:00") or "T" in utc` made the timezone half unfalsifiable: every
        # ISO stamp contains a "T", so a naive local time satisfied the `or` and the
        # assertion could never fail. Both halves are pinned separately now, and the
        # stronger versions of each live in TestMarkerFieldsMeanWhatTheySay.
        self.assertIn("T", row["utc"])
        self.assertTrue(row["utc"].endswith("+00:00"),
                        "the marker's utc stamp carries no UTC offset: %r" % row["utc"])
        self.assertIsInstance(row["monotonic_s"], float)

    def test_markers_survive_the_control_nodes_reader(self):
        """The bug this replaces: markers were keyed `node_id`/`utc`, while
        ThinkPad/control/analysis.py's windows_from_events reads `node`/`ts` and does
        `if not (node and ts): continue`. Every Pi 5 marker was silently discarded, so the
        energy join found no window for this node, its active power came out None, and
        profiles.build_devices refused a Pi 5 profile blaming "unmeasured inputs".

        This mirrors that reader's own logic rather than asserting on key names, so it
        keeps biting if either side is renamed again."""
        self.post("/mark", {"event": "run_begin", "note": "cell-1"})
        self.post("/mark", {"event": "run_end", "note": "cell-1"})
        _, events = self.get("/events")

        opens, active = {}, {}
        for rec in events:
            node, event, ts = rec.get("node"), rec.get("event", ""), rec.get("ts")
            if not (node and ts):
                continue
            if event.endswith("_begin"):
                opens[(node, event[:-6])] = ts
            elif event.endswith("_end"):
                start = opens.pop((node, event[:-4]), None)
                if start is None:
                    continue
                active.setdefault(node, []).append((start, ts))

        self.assertIn("pi5", active, "no window survived windows_from_events' reader")
        (start, end), = active["pi5"]
        self.assertIsInstance(start, float)
        self.assertGreaterEqual(end, start)

    def test_marker_timestamp_is_an_epoch_float(self):
        """`ts` is compared against power-sample timestamps from the DAQ, which are epoch
        floats. An ISO-8601 string is truthy, so it would pass the reader's guard above and
        then fail -- or worse, sort -- against real numbers.

        Asserted by VALUE against an independent clock reading, not just by magnitude: a
        `ts` scaled by 1000 is still > 1_600_000_000 and still a float, so the magnitude
        check alone let that mutation through. A 1000x-scaled marker lands ~54,000 years
        from the power trace, so every window silently finds no samples."""
        import time as _time
        before = _time.time()
        _, row = self.post("/mark", {"event": "run_begin"})
        after = _time.time()
        self.assertIsInstance(row["ts"], float)
        self.assertGreaterEqual(row["ts"], before - 1.0)
        self.assertLessEqual(row["ts"], after + 1.0)

    def test_the_marker_clock_agrees_with_the_iso_field(self):
        """The record carries both an epoch `ts` (for the join) and an ISO `utc` (for a
        human). They are produced by different calls and nothing checked they agree."""
        from datetime import datetime
        _, row = self.post("/mark", {"event": "run_begin"})
        iso = datetime.fromisoformat(row["utc"].replace("Z", "+00:00")).timestamp()
        self.assertAlmostEqual(row["ts"], iso, delta=2.0,
                               msg="the epoch and ISO stamps on one marker disagree")

    def test_markers_carry_the_conditions_they_were_taken_under(self):
        """Pi 4's records carry temp_c and throttled; a marker without them cannot later be
        checked for validity."""
        _, row = self.post("/mark", {"event": "run_begin"})
        self.assertIn("temp_c", row)
        self.assertIn("throttled", row)
        self.assertEqual(row["temp_c"], SAMPLE["soc_temp_c"])

    def test_marks_carry_the_current_run_tag(self):
        self.post("/run-tag", {"tag": "pi5_qwen_p256_g128_t4_r1"})
        _, row = self.post("/mark", {"event": "run_begin"})
        self.assertEqual(row["run_tag"], "pi5_qwen_p256_g128_t4_r1")

    def test_order_is_preserved(self):
        for i in range(5):
            self.post("/mark", {"event": "e%d" % i})
        _, events = self.get("/events")
        self.assertEqual([e["event"] for e in events], ["e0", "e1", "e2", "e3", "e4"])

    def test_tail_is_bounded(self):
        for i in range(10):
            self.post("/mark", {"event": "e%d" % i})
        _, events = self.get("/events?n=3")
        self.assertEqual([e["event"] for e in events], ["e7", "e8", "e9"])

    def test_missing_log_is_empty_not_an_error(self):
        status, events = self.get("/events")
        self.assertEqual(status, 200)
        self.assertEqual(events, [])

    def test_bad_n_falls_back(self):
        self.post("/mark", {"event": "x"})
        self.assertEqual(self.get("/events?n=banana")[0], 200)


class TestDamagedEventLog(AgentCase):
    """`tail_events`' `except json.JSONDecodeError: continue` was never reached. The log is
    appended to by a threaded server during a multi-hour sweep, so a half-written final line
    after a kill is the ordinary case, not an exotic one -- and losing the whole tail would
    lose the run_begin a window needs."""

    def test_a_half_written_line_does_not_lose_the_good_ones(self):
        self.post("/mark", {"event": "run_begin"})
        self.post("/mark", {"event": "run_end"})
        with agent.EVENTS.open("a", encoding="utf-8") as fh:
            fh.write('{"event": "run_beg')          # killed mid-write
        _, events = self.get("/events")
        self.assertEqual([e["event"] for e in events], ["run_begin", "run_end"])

    def test_a_wholly_corrupt_log_returns_empty_not_an_error(self):
        agent.EVENTS.parent.mkdir(parents=True, exist_ok=True)
        agent.EVENTS.write_text("not json at all\n{oops\n", encoding="utf-8")
        status, events = self.get("/events")
        self.assertEqual(status, 200)
        self.assertEqual(events, [])

    def test_a_marker_is_still_written_when_telemetry_raises(self):
        """`append_event` wraps telemetry.sample in `except Exception: pass` on purpose --
        a failing sensor read must not cost the phase marker the energy join depends on.
        Nothing proved that."""
        orig = agent.telemetry.sample
        agent.telemetry.sample = lambda node_id: (_ for _ in ()).throw(OSError("vcgencmd gone"))
        try:
            status, row = self.post("/mark", {"event": "run_begin"})
        finally:
            agent.telemetry.sample = orig
        self.assertEqual(status, 200)
        self.assertEqual(row["event"], "run_begin")
        self.assertIsInstance(row["ts"], float)
        self.assertEqual(row["temp_c"], "")      # absent, not invented
        self.assertEqual(row["throttled"], "")


class TestInterfaceReadout(unittest.TestCase):
    """`_iface`'s absent-interface branch and its per-field OSError fallback had no test.
    It feeds the wired_link_up gate and the /health body."""

    def test_an_absent_interface_is_reported_absent(self):
        got = agent._iface("definitely-not-a-nic-0")
        self.assertFalse(got["present"])
        self.assertIsNone(got.get("operstate"))

    def test_a_real_interface_reads_back(self):
        import os
        names = [n for n in os.listdir("/sys/class/net")] if os.path.isdir("/sys/class/net") else []
        if not names:
            self.skipTest("no /sys/class/net on this host")
        got = agent._iface(names[0])
        self.assertTrue(got["present"])
        self.assertIn("operstate", got)


class TestRunTagStillWorks(AgentCase):
    def test_run_tag_records(self):
        status, snap = self.post("/run-tag", {"tag": "cell-42"})
        self.assertEqual(status, 200)
        self.assertEqual(snap["run_tag"], "cell-42")

    def test_bad_json_is_400(self):
        req = urllib.request.Request(
            "http://127.0.0.1:%d/run-tag" % self.port, data=b"{not json",
            method="POST")
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)


class TestRunTagCoercion(AgentCase):
    """`tag` is written into STATE with `str(tag)` and then stamped onto every phase marker
    and every telemetry row, so whatever arrives here labels a whole cell of the dataset.
    No test posted anything but a string."""

    def test_a_non_string_tag_is_coerced_not_crashed(self):
        for value, expected in ((42, "42"), (3.5, "3.5"), (True, "True"), (None, "None")):
            with self.subTest(tag=value):
                status, snap = self.post("/run-tag", {"tag": value})
                self.assertEqual(status, 200)
                self.assertEqual(snap["run_tag"], expected)
                self.assertIsInstance(snap["run_tag"], str)

    def test_a_list_or_dict_tag_is_still_a_string(self):
        """Not a crash and not a nested object: the run tag has to survive a CSV column."""
        status, snap = self.post("/run-tag", {"tag": ["a", "b"]})
        self.assertEqual(status, 200)
        self.assertIsInstance(snap["run_tag"], str)

    def test_an_absent_tag_clears_it(self):
        """10_isolated_bench.sh posts {"tag": ""} after every cell to un-label the stream."""
        self.post("/run-tag", {"tag": "cell-1"})
        _, snap = self.post("/run-tag", {})
        self.assertEqual(snap["run_tag"], "")

    def test_the_tag_reaches_the_markers(self):
        self.post("/run-tag", {"tag": 7})
        _, row = self.post("/mark", {"event": "run_begin"})
        self.assertEqual(row["run_tag"], "7")


class TestConcurrentAccess(AgentCase):
    """The agent runs under ThreadingHTTPServer and is polled by the control node while the
    sweep posts markers, so LOCK-protected paths are genuinely concurrent in production and
    were exercised by nothing. A lost marker is a lost energy window."""

    def test_markers_posted_concurrently_are_all_recorded(self):
        import threading
        n = 24
        errors = []

        def post(i):
            try:
                self.post("/mark", {"event": "e%02d" % i})
            except Exception as exc:                      # noqa: BLE001 - reported below
                errors.append(exc)

        threads = [threading.Thread(target=post, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [], "concurrent POSTs raised: %r" % errors[:3])

        _, events = self.get("/events?n=200")
        got = sorted(e["event"] for e in events)
        self.assertEqual(got, sorted("e%02d" % i for i in range(n)),
                         "%d markers posted, %d survived -- the append is not atomic"
                         % (n, len(got)))

    def test_the_log_has_no_torn_lines_after_concurrent_writes(self):
        """Each record must be one complete JSON object per line; an unlocked append can
        interleave two writes mid-line."""
        import json as _json
        import threading
        threads = [threading.Thread(target=self.post,
                                    args=("/mark", {"event": "e%02d" % i, "note": "x" * 200}))
                   for i in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for line in agent.EVENTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                _json.loads(line)          # raises if a line was torn

    def test_reading_while_writing_never_fails(self):
        import threading
        stop = threading.Event()

        def writer():
            i = 0
            while not stop.is_set() and i < 40:
                self.post("/mark", {"event": "w%02d" % i})
                i += 1

        t = threading.Thread(target=writer)
        t.start()
        try:
            for _ in range(20):
                status, _events = self.get("/events")
                self.assertEqual(status, 200)
        finally:
            stop.set()
            t.join()


class TestServerEntryPoint(unittest.TestCase):
    """`main()` had no test: the port it binds comes from AGENT_PORT via
    envfile.get_int, and nothing checked that the configured port is the one used."""

    def _run_main_with_port(self, value):
        """Drive the real main() with a stub server, so the bind address is observable
        without actually listening. Asserting `get_int(ENV, "AGENT_PORT")` instead would
        test envfile, not main -- and would still pass if main hardcoded the port."""
        captured = {}

        class FakeServer:
            def __init__(self, addr, handler):
                captured["addr"] = addr

            def serve_forever(self):
                raise KeyboardInterrupt          # the shutdown path main() handles

            def server_close(self):
                captured["closed"] = True

        orig_srv, orig_port = agent.ThreadingHTTPServer, agent.ENV.get("AGENT_PORT")
        agent.ThreadingHTTPServer = FakeServer
        agent.ENV["AGENT_PORT"] = value
        try:
            rc = agent.main()
        finally:
            agent.ThreadingHTTPServer = orig_srv
            if orig_port is None:
                agent.ENV.pop("AGENT_PORT", None)
            else:
                agent.ENV["AGENT_PORT"] = orig_port
        return rc, captured

    def test_main_binds_the_configured_port(self):
        """Varied deliberately away from node.env's 9410: a hardcoded literal would satisfy
        an assertion made at the configured value alone."""
        for value in ("9410", "9999", "18080"):
            with self.subTest(port=value):
                _rc, captured = self._run_main_with_port(value)
                self.assertEqual(captured["addr"], ("0.0.0.0", int(value)))

    def test_main_closes_the_server_on_shutdown(self):
        """The `finally: srv.server_close()` leaves the port free for a systemd restart."""
        rc, captured = self._run_main_with_port("9410")
        self.assertTrue(captured.get("closed"), "main() did not close the server")
        self.assertEqual(rc, 0)

    def test_the_documented_port_matches_the_config(self):
        """docs/SETUP.md publishes the agent's port to whoever drives the cluster."""
        setup = (ROOT / "docs" / "SETUP.md").read_text(encoding="utf-8")
        self.assertIn(str(agent.ENV["AGENT_PORT"]), setup,
                      "node.env says AGENT_PORT=%s but docs/SETUP.md does not mention it"
                      % agent.ENV["AGENT_PORT"])


class TestReadOnlyByDesign(AgentCase):
    """The agent deliberately does not start or stop services: a node reconfigurable
    mid-sweep over HTTP is a node whose measurements cannot be trusted. The control node
    drives POST /rpc/start and /rpc/stop, so this node's 404 is a standing cross-node
    disagreement -- recorded here so it stays a decision rather than drifting into an
    accident."""

    def test_rpc_start_is_refused(self):
        status, body = self.post("/rpc/start", {})
        self.assertEqual(status, 404)
        self.assertIn("does not start or stop", body["note"])

    def test_rpc_stop_is_refused(self):
        self.assertEqual(self.post("/rpc/stop", {})[0], 404)

    def test_unknown_get_is_404_and_lists_routes(self):
        status, body = self.get("/nonsense")
        self.assertEqual(status, 404)
        self.assertIn("/health", body["routes"])


class TestGateBoundariesNotJustObviousCases(unittest.TestCase):
    # Borrow the harness, not the test methods -- subclassing would re-run the whole parent
    # suite under a second name.
    _gates = TestRealReadinessGates._gates

    """GATE_CASES gives every gate a witness in both directions, but each witness is far
    from the threshold -- swap 128 vs 0, temperature 70 vs 44. A gate can be loosened by a
    long way and still fail at 128 and pass at 0. These pin the edges, which is where a
    loosened gate actually changes behaviour."""

    # DELIBERATELY NOT TESTED, so a later mutation run does not re-litigate it:
    #   gates["governor_pinned"] = s["governor"] == ENV["CPU_GOVERNOR"]
    #     ->  str(s["governor"]).startswith(ENV["CPU_GOVERNOR"])
    # survives, and should. The two differ only for a governor name having "performance"
    # as a STRICT prefix, and the Linux cpufreq set is fixed -- performance, powersave,
    # userspace, ondemand, conservative, schedutil -- with no such name. Writing a test
    # around "performance_x" would assert behaviour on an input the kernel cannot produce,
    # which is the same error as trusting a mutation that never applied. Recorded as an
    # equivalent mutation rather than a gap.

    def test_any_swap_at_all_blocks(self):
        """The code's own blocking sentence is "a run that swaps is a corrupted
        measurement, not a slow one" -- so the threshold is zero, not "not much". 64 MB is
        enough to put model weights on the SD card during decode, and the per-layer cost
        that comes back is then a measurement of the card."""
        self.assertFalse(self._gates(swap_used_mb=1)["no_swap_in_use"],
                         "a node with swap in use was admitted")
        self.assertFalse(self._gates(swap_used_mb=64)["no_swap_in_use"])
        self.assertTrue(self._gates(swap_used_mb=0)["no_swap_in_use"])

    def test_the_thermal_ceiling_admits_exactly_the_configured_value(self):
        """node.env sets THERMAL_START_MAX_C=55 and the rule is `0.0 < t <= start_max`, so
        55.0 is admitted and anything above is not. This gate is what makes every cell of
        the sweep begin from the same thermal state -- raising it by even a few degrees is
        the difference between a repeatable c_i and one that drifts with ambient."""
        ceiling = float(agent.ENV.get("THERMAL_START_MAX_C", 55))
        self.assertTrue(self._gates(soc_temp_c=ceiling)["cool_enough_to_start"],
                        "a board at exactly the configured ceiling was rejected")
        self.assertFalse(self._gates(soc_temp_c=ceiling + 0.1)["cool_enough_to_start"],
                         "a board above the configured ceiling was admitted")


class TestTheHealthBodyIsNotHollow(AgentCase):
    """Every payload block of /health could be emptied without a test failing: `gates`,
    `sample`, `wired_link` and `warnings` were asserted to EXIST, never to carry anything.
    `gates` is what docs/SETUP.md tells an operator to read to find out WHY a node is
    blocked, so an empty one turns a diagnosable refusal into a mystery."""

    def test_the_verdict_identifies_this_node(self):
        _, body = self.get("/health")
        self.assertEqual(body["node_id"], agent.ENV["NODE_ID"],
                         "the readiness verdict identified itself as another node")

    def test_the_payload_blocks_carry_their_contents(self):
        _, body = self.get("/health")
        for block in ("gates", "sample", "metrics", "wired_link"):
            with self.subTest(block=block):
                self.assertTrue(body.get(block),
                                "/health %s is present but empty" % block)
        self.assertEqual(set(body["gates"]), set(ALL_PASS),
                         "/health gates did not carry the real gate set")

    def test_uptime_keeps_sub_second_precision(self):
        """Pin STARTED so the expected value is exact. Asserting only a type, a sign, or an
        upper bound cannot see a coarser rounding: in a test the real uptime is under a
        second, so round(x, -2) yields 0.0 and still satisfies all three."""
        orig = agent.STARTED
        agent.STARTED = time.time() - 123.45
        self.addCleanup(lambda: setattr(agent, "STARTED", orig))
        _, body = self.get("/health")
        self.assertIsInstance(body["uptime_s"], float)
        self.assertAlmostEqual(body["uptime_s"], 123.4, delta=0.5,
                               msg="uptime_s lost sub-second precision (got %r)"
                                   % (body["uptime_s"],))


class TestTheWarningChannelIsLive(unittest.TestCase):
    """`warnings` is the non-blocking channel, and the unpinned-revision warning is the one
    condition this node raises through it. Both the channel and the warning's polarity were
    untested: inverting the test makes it fire on the CORRECT configuration and go silent
    on the dangerous one."""

    def _health_with(self, pin):
        orig_env = dict(agent.ENV)
        orig_sample, orig_port = agent.telemetry.sample, agent._port_open
        agent.telemetry.sample = lambda node_id: {
            "governor": "performance", "swap_used_mb": 0, "throttled_now": 0,
            "under_voltage_now": 0, "throttled_hex": "0x0", "soc_temp_c": 44.0,
            "mem_avail_mb": 7000, "rpc_rss_mb": 0}
        agent._port_open = lambda host, port, timeout=0.4: True
        if pin is None:
            agent.ENV.pop("LLAMA_PIN_SHA", None)
        else:
            agent.ENV["LLAMA_PIN_SHA"] = pin
        try:
            return agent.health()
        finally:
            agent.ENV.clear(); agent.ENV.update(orig_env)
            agent.telemetry.sample, agent._port_open = orig_sample, orig_port

    def test_an_unpinned_revision_warns(self):
        body = self._health_with(None)
        self.assertTrue([w for w in body["warnings"] if "not pinned" in w],
                        "an unpinned llama.cpp revision raised no warning")

    def test_a_pinned_revision_does_not_warn(self):
        body = self._health_with("23106f94ea2bc3da929afb7330655fd5515d08dc")
        self.assertFalse([w for w in body["warnings"] if "not pinned" in w],
                         "the pin warning fired on a correctly pinned node")


class TestMarkerFieldsMeanWhatTheySay(AgentCase):
    """Phase markers define the window the energy trace is integrated over, so every field
    here is an input to joules-per-token. Each was asserted to exist or to be the right
    TYPE, never to be the right QUANTITY -- and each of these mutations keeps the type."""

    # The module-level SAMPLE carries neither `throttled_hex` nor `throttled_now`, so both
    # read back as "" and a swap between them is invisible -- the same blind spot that let
    # the original assertion pass. Stub a sample in which the two differ.
    def setUp(self):
        super().setUp()
        rich = dict(SAMPLE, throttled_hex="0x50005", throttled_now=1)
        agent.telemetry.sample = lambda node_id: dict(rich)

    def test_the_throttle_field_is_the_full_firmware_word(self):
        """`throttled_hex` is the whole vcgencmd word (e.g. "0x50005"), which is what lets
        a run be audited afterwards for under-voltage AND frequency capping AND throttling.
        `throttled_now` is one decoded bit from it -- same field, same row, and every
        sticky-bit history gone."""
        _, row = self.post("/mark", {"event": "run_begin"})
        self.assertEqual(row["throttled"], "0x50005",
                         "the marker carried a decoded bit (%r) instead of the firmware "
                         "word" % (row["throttled"],))

    def test_the_monotonic_field_is_the_monotonic_clock(self):
        """Its whole reason for existing, stated in the docstring, is that it survives an
        NTP step mid-run. `time.time()` is also a float and also increases, so a type check
        cannot tell them apart -- but the epoch clock is ~1.8e9 and monotonic is uptime."""
        _, row = self.post("/mark", {"event": "run_end"})
        self.assertLess(row["monotonic_s"], time.time() - 86400,
                        "monotonic_s looks like a wall-clock epoch, not a monotonic clock")

    def test_the_human_stamp_carries_its_offset(self):
        """A naive local stamp on this cluster (+07) reads seven hours off every Pi 4
        marker and off the DAQ's own UTC trace, with nothing in the string to show it."""
        _, row = self.post("/mark", {"event": "run_end"})
        self.assertTrue(row["utc"].endswith("+00:00"),
                        "the marker's utc stamp is naive: %r" % row["utc"])

    def test_the_join_timestamp_keeps_sub_second_resolution(self):
        """Quantising to whole seconds moves each window edge by up to 0.5 s. On a decode
        cell of a few seconds that is a double-digit percentage of the integration window,
        applied to the energy figure and to nothing else."""
        _, row = self.post("/mark", {"event": "run_begin"})
        self.assertNotEqual(row["ts"], round(row["ts"], 0),
                            "marker timestamps were rounded to whole seconds")


class TestTheReadEndpointsKeepTheirContract(AgentCase):
    def test_stat_uses_the_key_names_the_control_node_reads(self):
        """/stat and its /metrics alias are read by name. Renaming the blocks keeps the
        response a valid JSON object of the same shape, so only the names themselves pin
        it -- the same defect class as the `node_id`/`node` marker bug."""
        for path in ("/stat", "/metrics"):
            with self.subTest(path=path):
                status, body = self.get(path)
                self.assertEqual(status, 200)
                self.assertIn("sample", body)
                self.assertIn("run", body)

    def test_stat_reports_the_current_run_tag(self):
        """The `run` block is how a sample is attributed to a cell. Serving an empty dict
        makes every sample unattributable while the endpoint still answers 200."""
        self.post("/run-tag", {"tag": "cell-7"})
        _, body = self.get("/stat")
        self.assertEqual(body["run"].get("run_tag"), "cell-7",
                         "/stat did not carry the run tag it was given")

    def test_events_returns_a_useful_default_window(self):
        """The default is 50. A quietly smaller default truncates the marker history the
        energy join walks, and a truncated history looks exactly like a run that emitted
        fewer markers."""
        for i in range(12):
            self.post("/mark", {"event": "run_begin", "note": "cell-%d" % i})
        _, events = self.get("/events")
        self.assertEqual(len(events), 12,
                         "/events with no n= returned a truncated window")


class TestPortProbeAndInterfaceReadRealSources(unittest.TestCase):
    def test_a_wildcard_bind_address_is_probed_on_the_loopback(self):
        """RPC_HOST is 0.0.0.0 in node.env -- a bind address, not a destination. Connecting
        to it verbatim is not a test of whether the local service is up, and on some stacks
        it is not connectable at all, so the gate would report the rpc-server down while it
        was serving normally."""
        seen = {}
        orig = agent.socket.create_connection

        def fake(addr, timeout=None):
            seen["host"] = addr[0]
            raise OSError("refused")
        agent.socket.create_connection = fake
        try:
            agent._port_open("0.0.0.0", 50052)
        finally:
            agent.socket.create_connection = orig
        self.assertEqual(seen["host"], "127.0.0.1",
                         "a 0.0.0.0 bind address was probed verbatim")

    def _fake_sysfs(self, files):
        """Point _iface at a synthetic /sys/class/net. Using a REAL host NIC cannot
        discriminate here: if the host happens to have an up gigabit link, sysfs returns
        exactly "1000"/"1" and a hardcoded pair is indistinguishable from a real read."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        nic = root / "eth-test"
        nic.mkdir(parents=True)
        for name, value in files.items():
            (nic / name).write_text(value)
        orig = agent.Path
        agent.Path = lambda p: root if str(p) == "/sys/class/net" else orig(p)
        self.addCleanup(lambda: setattr(agent, "Path", orig))
        return agent._iface("eth-test")

    def test_interface_fields_come_from_sysfs_not_constants(self):
        """`speed_mbit` and `carrier` decide whether this node looks like a healthy wired
        link -- they feed the wired_link gate and the /health body. Hardcoding them makes
        a down NIC report a carrier."""
        got = self._fake_sysfs({"operstate": "down", "speed": "2500", "carrier": "0"})
        self.assertTrue(got["present"])
        self.assertEqual(got["operstate"], "down")
        self.assertEqual(got["speed_mbit"], "2500",
                         "speed_mbit did not come from sysfs")
        self.assertEqual(got["carrier"], "0",
                         "a down interface reported a carrier")

    def test_an_unreadable_speed_falls_back_rather_than_inventing_one(self):
        """The "-" and "0" fallbacks exist because a USB 2.5GbE adapter frequently cannot
        report a speed at all -- reading `speed` on a down port raises EINVAL. A number
        there would be a link rate nobody measured."""
        got = self._fake_sysfs({"operstate": "down", "carrier": "0"})   # no `speed` file
        self.assertEqual(got["speed_mbit"], "-",
                         "an unreadable link speed was reported as a number")

    def test_an_absent_interface_is_reported_absent(self):
        got = agent._iface("definitely-not-a-nic-0")
        self.assertFalse(got["present"],
                         "an absent interface reported itself present")
        self.assertNotEqual(got.get("carrier"), "1",
                            "an absent interface reported a carrier")


if __name__ == "__main__":
    unittest.main(verbosity=2)
