"""Tests for testbed/scripts/shelly_strip.py, stdlib unittest only. From the repo root:

    python3 -m unittest discover -s testbed/tests -q

Nothing here contacts the strip: logs are synthetic CSVs, and the HTTP tests talk to a fake strip on 127.0.0.1.
SHELLY_STRIP_UNDER_TEST=<path> runs the suite against another copy of the tool (the mutation check uses it, so the repo's
copy is never edited).
"""
import ast, contextlib, csv, decimal, hashlib, http.server, io, json, math, os, re, stat, tempfile, threading, types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.abspath(os.environ.get("SHELLY_STRIP_UNDER_TEST")
                       or os.path.join(HERE, "..", "scripts", "shelly_strip.py"))


def _load():
    with open(TOOL, encoding="utf-8") as f:
        src = f.read()
    mod = types.ModuleType("shelly_strip_under_test")
    mod.__file__ = TOOL
    exec(compile(src, TOOL, "exec"), mod.__dict__)   # no bytecode cache written next to the tool
    return mod, src


m, SRC = _load()
D = decimal.Decimal
T0 = 1_791_000_000.0
MAC = "E8F60A62B7C0"
GOOD_META = {"device": {"id": "shellypstripg4-e8f60a62b7c0", "mac": MAC, "ver": "2.0.1", "fw_id": "fw-a"},
             "expected_mac": MAC, "outlet_map": {"0": "pi5", "1": "pi4", "2": "jetson", "3": "router"},
             "end_wall": T0 + 10000, "firmware_changed": False, "problems_at_start": [], "requested_hz": 2.0,
             "host": "test"}


def fake_status(uptime=1000, outlets=None):
    """A Shelly.GetStatus answer shaped like the strip's (Decimals, as the tool parses them)."""
    st = {"sys": {"uptime": uptime, "unixtime": None}}
    for i in range(4):
        o = {"output": True, "apower": "0.0", "voltage": "230.0", "current": "0.000", "pf": "0.00", "total": "0.000",
             "errors": None}
        o.update((outlets or {}).get(i, {}))
        sw = {"id": i, "source": "init", "output": o["output"], "apower": D(str(o["apower"])),
              "voltage": D(str(o["voltage"])), "current": D(str(o["current"])), "pf": D(str(o["pf"])), "freq": D("50.0"),
              "aenergy": {"total": D(str(o["total"])), "by_minute": [D("0.000"), D("0.000"), D("0.000")], "minute_ts": 0}}
        if o["errors"]:
            sw["errors"] = o["errors"]
        st["switch:%d" % i] = sw
    return st


def build(hz, seconds, outlet_fn=None, uptime_fn=None, fail=lambda r: False, mono_fn=None, start=0.0):
    """Samples every 1/hz s for `seconds`, r = seconds since T0. outlet_fn(r) -> {outlet: {...}}; fail(r) -> failed row."""
    out = []
    for k in range(int(round(seconds * hz)) + 1):
        r = start + k / hz
        if fail(r):
            out.append((r, None))
        else:
            up = uptime_fn(r) if uptime_fn else 1000 + int(math.floor(r))
            out.append((r, fake_status(uptime=up, outlets=outlet_fn(r) if outlet_fn else None)))
    return out, mono_fn


def write_log(dirpath, samples, mono_fn=None, meta=GOOD_META, name="log.csv"):
    path = os.path.join(dirpath, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=m.csv_header())
        w.writeheader()
        for r, st in samples:
            mono = mono_fn(r) if mono_fn else r + 100.0
            if st is None:
                w.writerow(m.make_row(T0 + r, mono, 0.0, False, "TimeoutError: timed out"))
            else:
                w.writerow(m.make_row(T0 + r, mono, 0.0, True, fields=m.status_fields(st)))
    if meta is not None:
        with open(path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f)
    return path


class SummariseCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def summ(self, built, window, idle=None, tokens=None, meta=GOOD_META):
        samples, mono_fn = built
        path = write_log(self.tmp.name, samples, mono_fn, meta)
        return m.summarise(m.load_log(path), meta, (T0 + window[0], T0 + window[1]),
                           (T0 + idle[0], T0 + idle[1]) if idle else None, tokens)

    @staticmethod
    def codes(s, outlet):
        """The flags that apply to this outlet: its own, and those raised for every outlet at once."""
        return {f["code"] for f in s["flags"] if f["outlet"] in (outlet, "all")}


def counter_of(power_integral_j):
    """aenergy_total text (Wh) for a known energy in J, at high precision (no quantisation)."""
    return "%.9f" % (5.0 + power_integral_j / 3600.0)


class TestEnergy(SummariseCase):
    def test_trapezoid_integrates_a_ramp_exactly_with_interpolated_edges(self):
        P = lambda r: 10.0 + 0.5 * r
        s = self.summ(build(2, 120, lambda r: {3: {"apower": "%.6f" % P(r), "output": True}}), (10.3, 100.7))
        a, b = 10.3, 100.7
        # unix timestamps near 1.79e9 carry ~1e-7 s of float error, hence a 1e-4 J tolerance on ~3.4 kJ
        self.assertAlmostEqual(s["outlets"]["3"]["energy_trap_j"], 10.0 * (b - a) + 0.25 * (b * b - a * a), delta=1e-4)

    def test_trapezoid_smears_a_step_over_one_sample_interval(self):
        P = lambda r: 0.0 if r <= 50.0 else 100.0
        s = self.summ(build(2, 100, lambda r: {3: {"apower": "%.1f" % P(r)}}), (40.0, 70.0))
        self.assertAlmostEqual(s["outlets"]["3"]["energy_trap_j"], 100.0 * (70.0 - 50.5) + 0.5 * 100.0 * 0.5, delta=1e-4)

    def test_counter_delta_is_wh_times_3600_interpolated_at_the_edges(self):
        # 36 W constant: the counter rises 0.01 Wh a second; samples at 2 Hz, window edges between samples
        fn = lambda r: {3: {"apower": "36.0", "total": "%.4f" % (5.0 + 0.01 * r)}}
        s = self.summ(build(2, 100, fn), (10.25, 90.75))
        o = s["outlets"]["3"]
        self.assertAlmostEqual(o["energy_counter_j"], 0.01 * 80.5 * 3600.0, places=6)
        self.assertAlmostEqual(o["energy_trap_j"], 36.0 * 80.5, places=6)
        self.assertAlmostEqual(o["disagreement_pct"], 0.0, places=6)
        self.assertNotIn("METHODS_DISAGREE", self.codes(s, 3))

    def test_disagreement_sign_and_flag(self):
        # apower says 10 W, the counter accumulates as if 10.5 W: trapezoid is 4.76 % below the counter
        fn = lambda r: {3: {"apower": "10.0", "total": counter_of(10.5 * r)}}
        s = self.summ(build(2, 100, fn), (10.0, 90.0))
        self.assertAlmostEqual(s["outlets"]["3"]["disagreement_pct"], 100.0 * (10.0 - 10.5) / 10.5, places=6)
        self.assertNotIn("METHODS_DISAGREE", self.codes(s, 3))
        fn = lambda r: {3: {"apower": "10.0", "total": counter_of(12.0 * r)}}
        s = self.summ(build(2, 100, fn), (10.0, 90.0))
        self.assertIn("METHODS_DISAGREE", self.codes(s, 3))

    def test_idle_subtraction_and_j_per_token(self):
        P = lambda r: 2.0 if r < 100 else 12.0
        E = lambda r: 2.0 * r if r < 100 else 200.0 + 12.0 * (r - 100.0)
        fn = lambda r: {3: {"apower": "%.1f" % P(r), "total": counter_of(E(r))}}
        s = self.summ(build(2, 300, fn), (120.0, 280.0), idle=(10.0, 90.0), tokens=400)
        o = s["outlets"]["3"]
        for meth in ("trap", "counter"):
            self.assertAlmostEqual(o["idle_mean_w_%s" % meth], 2.0, places=6)
            self.assertAlmostEqual(o["energy_%s_j" % meth], 1920.0, places=4)
            self.assertAlmostEqual(o["net_energy_%s_j" % meth], 1600.0, places=4)
            self.assertAlmostEqual(o["j_per_token_gross_%s" % meth], 4.8, places=6)
            self.assertAlmostEqual(o["j_per_token_net_%s" % meth], 4.0, places=6)
            self.assertAlmostEqual(o["wh_per_1000_tokens_net_%s" % meth], 1600.0 / 400 * 1000.0 / 3600.0, places=6)
        self.assertNotIn("IDLE_OVERLAP", {f["code"] for f in s["flags"]})

    def test_idle_overlap_and_negative_net_are_flagged(self):
        P = lambda r: 12.0 if r < 100 else 2.0
        s = self.summ(build(2, 300, lambda r: {3: {"apower": "%.1f" % P(r)}}), (120.0, 280.0), idle=(10.0, 130.0))
        allc = {f["code"] for f in s["flags"]}
        self.assertIn("IDLE_OVERLAP", allc)
        self.assertIn("NET_NEGATIVE", self.codes(s, 3))


class TestFlags(SummariseCase):
    def test_restart_when_uptime_goes_back(self):
        up = lambda r: 1000 + int(r) if r < 60 else 5 + int(r - 60)
        s = self.summ(build(2, 100, uptime_fn=up), (40.0, 80.0))
        self.assertIn("RESTART", {f["code"] for f in s["flags"]})
        s = self.summ(build(2, 100, uptime_fn=up), (10.0, 50.0))
        self.assertNotIn("RESTART", {f["code"] for f in s["flags"]})

    def test_restart_hidden_in_an_outage_is_caught(self):
        # unreachable 20.5..60; afterwards uptime is LARGER than before (35 > 30) but advanced 5 s in 40.5 s
        up = lambda r: 10 + int(r) if r <= 20 else 35 + int(r - 60.5)
        s = self.summ(build(2, 100, uptime_fn=up, fail=lambda r: 20 < r < 60.5), (10.0, 90.0))
        self.assertIn("RESTART", {f["code"] for f in s["flags"]})

    def test_counter_reset_flagged_and_counter_energy_withheld(self):
        tot = lambda r: "%.4f" % (5.0 + 0.01 * r) if r < 60 else "%.4f" % (0.01 * (r - 60))
        s = self.summ(build(2, 100, lambda r: {3: {"apower": "36.0", "total": tot(r)}}), (40.0, 80.0))
        self.assertIn("COUNTER_RESET", self.codes(s, 3))
        self.assertIsNone(s["outlets"]["3"]["energy_counter_j"])
        self.assertIsNotNone(s["outlets"]["3"]["energy_trap_j"])

    def test_gap_and_request_errors(self):
        s = self.summ(build(2, 100, lambda r: {3: {"apower": "5.0"}}, fail=lambda r: 50.0 <= r <= 56.0), (40.0, 80.0))
        c = self.codes(s, 3)
        self.assertIn("GAP", c)
        self.assertIn("REQUEST_ERRORS", c)
        msg = [f["message"] for f in s["flags"] if f["outlet"] in (3, "all") and f["code"] == "REQUEST_ERRORS"][0]
        self.assertIn("13 of 81", msg)
        self.assertAlmostEqual(s["outlets"]["3"]["max_gap_s"], 7.0, places=6)
        # a short outage: errors flagged, no gap flag
        s = self.summ(build(2, 100, fail=lambda r: 50.0 <= r <= 51.0), (40.0, 80.0))
        self.assertIn("REQUEST_ERRORS", self.codes(s, 3))
        self.assertNotIn("GAP", self.codes(s, 3))

    def test_fresh_reading_rate(self):
        # 4 Hz polling; outlet 1's voltage changes every second, outlet 2's every 2 s; apower never changes
        fn = lambda r: {1: {"voltage": "%.1f" % (230.0 + 0.1 * (int(r) % 7))},
                        2: {"voltage": "%.1f" % (230.0 + 0.1 * (int(r // 2) % 7))}}
        s = self.summ(build(4, 60, fn), (10.1, 50.1))
        o1, o2 = s["outlets"]["1"], s["outlets"]["2"]
        self.assertAlmostEqual(o1["fresh_voltage_hz"], 1.0, delta=0.03)
        self.assertAlmostEqual(o1["fresh_any_hz"], 1.0, delta=0.03)
        self.assertEqual(o1["fresh_apower_hz"], 0.0)
        self.assertAlmostEqual(o1["fresh_any_median_interval_s"], 1.0, places=6)
        self.assertAlmostEqual(o2["fresh_voltage_hz"], 0.5, delta=0.03)
        self.assertAlmostEqual(o1["request_rate_hz"], 4.0, delta=0.05)
        self.assertNotIn("FRESH_BELOW_1HZ", self.codes(s, 1))
        self.assertIn("FRESH_BELOW_1HZ", self.codes(s, 2))
        self.assertTrue(any("outlet 1" in n and "apower never changed" in n for n in s["notes"]))

    def test_output_false_flag_only_where_off(self):
        s = self.summ(build(2, 60, lambda r: {0: {"output": r < 30}}), (10.0, 50.0))
        self.assertIn("OUTPUT_FALSE", self.codes(s, 0))
        self.assertNotIn("OUTPUT_FALSE", self.codes(s, 3))

    def test_outlet_errors_flag(self):
        s = self.summ(build(2, 60, lambda r: {2: {"errors": ["overpower"] if 20 < r < 25 else None}}), (10.0, 50.0))
        self.assertIn("OUTLET_ERRORS", self.codes(s, 2))

    def test_window_not_covered_withholds_energy(self):
        s = self.summ(build(2, 60, lambda r: {3: {"apower": "5.0", "total": counter_of(5.0 * r)}}), (-10.0, 30.0))
        self.assertIn("NOT_COVERED", self.codes(s, 3))
        self.assertIsNone(s["outlets"]["3"]["energy_trap_j"])
        self.assertIsNone(s["outlets"]["3"]["energy_counter_j"])

    def test_counter_that_does_not_move_is_flagged(self):
        s = self.summ(build(2, 60), (10.0, 50.0))
        self.assertEqual(s["outlets"]["0"]["energy_counter_j"], 0.0)
        self.assertIn("COUNTER_COARSE", self.codes(s, 0))

    def test_coarse_counter_step_is_reported(self):
        # 1 W with a 0.001 Wh counter: one step (3.6 J) every 3.6 s; a 20 s window holds ~20 J
        tot = lambda r: "%.3f" % (5.0 + math.floor(r / 3.6) * 0.001)
        s = self.summ(build(2, 60, lambda r: {3: {"apower": "1.0", "total": tot(r)}}), (10.0, 30.0))
        o = s["outlets"]["3"]
        self.assertIn("COUNTER_COARSE", self.codes(s, 3))
        self.assertAlmostEqual(o["counter_step_wh"], 0.001, places=9)
        self.assertAlmostEqual(o["counter_display_resolution_wh"], 0.001, places=9)
        self.assertAlmostEqual(o["counter_update_interval_s"], 3.5, delta=0.6)
        # 36 W over 100 s with per-second 0.01 Wh steps: 2 steps are 2 % of the energy, not flagged
        tot = lambda r: "%.3f" % (5.0 + math.floor(r) * 0.01)
        s = self.summ(build(2, 140, lambda r: {3: {"apower": "36.0", "total": tot(r)}}), (20.0, 120.0))
        self.assertNotIn("COUNTER_COARSE", self.codes(s, 3))
        self.assertAlmostEqual(s["outlets"]["3"]["counter_quantisation_pct"], 2.0, places=6)

    def test_too_few_samples(self):
        s = self.summ(build(2, 60), (10.0, 13.0))
        self.assertIn("TOO_FEW_SAMPLES", self.codes(s, 3))

    def test_clock_step_withholds_energy(self):
        mono = lambda r: r + 100.0 if r < 30 else r + 98.0      # the wall clock jumps 2 s against the monotonic one
        s = self.summ(build(2, 60, lambda r: {3: {"apower": "5.0"}}, mono_fn=mono), (10.0, 50.0))
        self.assertIn("CLOCK_STEP", self.codes(s, 3))
        self.assertIsNone(s["outlets"]["3"]["energy_trap_j"])

    def test_meta_flags(self):
        built = build(2, 60)
        s = self.summ(built, (10.0, 50.0), meta=None)
        self.assertIn("NO_META", {f["code"] for f in s["flags"]})
        meta = dict(GOOD_META, end_wall=None, firmware_changed=True, device_at_end={"fw_id": "fw-b"},
                    expected_mac="AABBCCDDEEFF")
        s = self.summ(built, (10.0, 50.0), meta=meta)
        allc = {f["code"] for f in s["flags"]}
        self.assertTrue({"LOG_NOT_CLOSED", "FW_CHANGED", "DEVICE_MISMATCH"} <= allc)
        s = self.summ(built, (10.0, 50.0))
        self.assertFalse({"NO_META", "LOG_NOT_CLOSED", "FW_CHANGED", "DEVICE_MISMATCH"} & {f["code"] for f in s["flags"]})

    def test_a_clean_loaded_window_raises_no_flag(self):
        # 1 Hz fresh readings of a 36 W load (as the strip refreshes), 4 Hz polling, counter 0.01 Wh a second, 100 s
        fn = lambda r: {3: {"apower": "%.1f" % (36.0 + 0.1 * (int(r) % 2)), "voltage": "%.1f" % (230 + 0.1 * (int(r) % 3)),
                            "total": "%.3f" % (5.0 + 0.01 * math.floor(r))}}
        s = self.summ(build(4, 140, fn), (20.0, 120.0))
        self.assertEqual(self.codes(s, 3), set(), [f for f in s["flags"] if f["outlet"] == 3])


class TestRows(unittest.TestCase):
    def test_status_fields_keep_the_strips_decimals(self):
        st = fake_status(outlets={3: {"total": "9.450", "apower": "17.0"}})
        f = m.status_fields(st)
        self.assertEqual(f["o3_aenergy_total"], "9.450")
        self.assertEqual(f["o3_apower"], "17.0")
        self.assertEqual(f["o3_output"], "1")

    def test_missing_outlet_fails_the_row(self):
        st = fake_status()
        del st["switch:2"]
        with self.assertRaises(ValueError):
            m.status_fields(st)

    def test_truncated_log_is_refused_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as d:
            path = write_log(d, build(2, 5)[0])
            with open(path, "a", encoding="utf-8") as f:
                f.write("%.6f,12.5,0." % (T0 + 6))            # the logger killed mid-line
            with self.assertRaises(SystemExit) as cm:
                m.load_log(path)
            self.assertIn("malformed", str(cm.exception))

    def test_failed_row_has_empty_measurement_fields(self):
        row = m.make_row(T0, 5.0, 2.0, False, "TimeoutError: timed out")
        self.assertEqual(row["ok"], "0")
        self.assertIn("timed out", row["error"])
        for k in m.csv_header():
            if k not in ("wall_time", "mono_time", "rtt_s", "ok", "error"):
                self.assertEqual(row[k], "", k)

    def test_poll_never_carries_a_value_forward_and_never_bursts(self):
        clock = {"t": 0.0}

        class Client:
            n = 0

            def get(self, path):
                Client.n += 1
                clock["t"] += 1.3 if Client.n == 5 else 0.05        # request 5 is slow
                if Client.n in (3, 4):
                    raise OSError("timed out")
                return fake_status(uptime=1000 + Client.n, outlets={3: {"apower": "%d.0" % Client.n}})

        rows = []
        def sleep(d):
            clock["t"] += max(d, 1e-6)

        n, n_ok = m.poll(Client(), rows.append, 5.0, 3.0, lambda: False, clock=lambda: clock["t"],
                         wall=lambda: T0 + clock["t"], sleep=sleep)
        self.assertEqual(n, len(rows))
        self.assertEqual(n - n_ok, 2)
        failed = [r for r in rows if r["ok"] == "0"]
        self.assertEqual(len(failed), 2)
        for r in failed:
            self.assertEqual(r["o3_apower"], "")
            self.assertEqual(r["dev_uptime"], "")
        self.assertEqual(rows[1]["o3_apower"], "2.0")
        starts = [float(r["wall_time"]) for r in rows]
        self.assertTrue(all(b - a >= 0.2 - 1e-5 for a, b in zip(starts, starts[1:])), starts)   # no burst after the slow one
        self.assertAlmostEqual(starts[5] - starts[4], 1.4, delta=1e-5)     # the slots the slow request covered are skipped
        self.assertLessEqual(len(rows), 16)


# ------------------------------------------------------------------------------------------------- read-only guarantees

COMPONENTS = ("Shelly|Switch|Sys|Wifi|WiFi|Cloud|Mqtt|MQTT|BLE|Ble|Ws|WS|Matter|Script|KVS|Schedule|Webhook|Input|Eth|"
              "Ui|UI|HTTP|Light|Cover|PM1|EM|EM1|EMData|EM1Data|Temperature|Humidity|Voltmeter|BTHome|BTHomeDevice|"
              "BTHomeSensor|Knx|KNX|Sensor|DevicePower|PowerStrip|PowerstripUi|Plugs|PLUGS_UI|Service|Zigbee|Modbus|OTA")
FORBIDDEN = ("Toggle", "Reboot", "FactoryReset", "ResetCounters", "SetConfig", "SetAuth", "CheckForUpdate",
             "ResetWiFiConfig", "PutUserCA", "PutTLS", "SetTime", ".Set", ".Update", ".Create", ".Delete", ".Start",
             ".Stop", ".Eval", ".Call", "turn=", "on=true", "on=false", "/relay/", "/light/", "/roller/", "/settings",
             "/reboot", "/ota", "/script/", '"POST"', "'POST'", '"PUT"', "'PUT'", '"DELETE"', "'DELETE'", '"PATCH"',
             "'PATCH'", "urlopen", "putrequest", "socket.socket(", "create_connection")


class TestReadOnly(unittest.TestCase):
    def test_source_names_no_state_changing_rpc(self):
        for comp, meth in re.findall(r"\b(%s)\.([A-Za-z_][A-Za-z0-9_]*)" % COMPONENTS, SRC):
            self.assertTrue(meth.startswith("Get"), "%s.%s in shelly_strip.py is not a Get* method" % (comp, meth))
        for p in re.findall(r"/rpc/([A-Za-z0-9_.]+)", SRC):
            self.assertRegex(p, r"^[A-Z][A-Za-z0-9]*\.Get[A-Za-z0-9]+$")
        for bad in FORBIDDEN:
            self.assertNotIn(bad, SRC, "shelly_strip.py contains %r" % bad)

    def test_allow_list_is_get_only(self):
        self.assertIn("/shelly", m.READ_ONLY_PATHS)
        for p in m.READ_ONLY_PATHS:
            self.assertTrue(p == "/shelly" or re.fullmatch(r"/rpc/[A-Z][A-Za-z0-9]*\.Get[A-Za-z0-9]+", p), p)

    def test_one_network_entry_point_and_it_is_a_get(self):
        tree = ast.parse(SRC)
        reqs = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "request"]
        self.assertEqual(len(reqs), 1)
        self.assertIsInstance(reqs[0].args[0], ast.Constant)
        self.assertEqual(reqs[0].args[0].value, "GET")
        self.assertEqual(SRC.count("HTTPConnection("), 1)
        for call in re.findall(r"\bcli\.get\(([^)]*)\)", SRC):
            self.assertIn(call.strip('"\''), m.READ_ONLY_PATHS, call)

    def test_client_refuses_anything_else_before_connecting(self):
        cli = m.ShellyClient("192.0.2.1", timeout=0.01)       # TEST-NET: never reached, the refusal comes first
        for p in ("/rpc/Switch." + "Set", "/rpc/Shelly.GetStatus?id=0", "/rpc/shelly.getstatus", "/relay/0", "/",
                  "/rpc/Sys." + "SetConfig", "/shelly/"):
            with self.assertRaises(m.ShellyError, msg=p):
                cli.get(p)
        with self.assertRaises(m.ShellyError):
            m.ShellyClient("http://10.0.0.1/rpc")


# --------------------------------------------------------------------------------------------- a fake strip, 127.0.0.1

REALM, NONCE = "shellypstripg4-e8f60a62b7c0", "6c2c5b3e"
SHELLY = {"id": REALM, "mac": MAC, "model": "S4PL-00416EU", "gen": 4, "fw_id": "fw-a", "ver": "2.0.1", "app": "PowerStrip",
          "auth_en": False, "auth_domain": None}
CONFIG = {"sys": {"sntp": {"server": "time.example"}}, "cloud": {"enable": False}, "matter": {"enable": True},
          "ble": {"rpc": {"enable": False}},
          "wifi": {"sta": {"ssid": "net1", "enable": True, "pass": "SECRET-PASS"}, "sta1": {"ssid": None, "enable": False},
                   "ap": {"enable": False, "pass": "SECRET-AP"}}}
for _i, _n in enumerate(("pi5", "pi4", "jetson", "router")):
    CONFIG["switch:%d" % _i] = {"id": _i, "name": _n, "initial_state": "on", "auto_off": False}


def status_json(n):
    st = json.loads(json.dumps(fake_status(uptime=500 + n, outlets={3: {"apower": "40.0", "total": "1.250"}}),
                               default=float))
    st["sys"]["available_updates"] = {"beta": {"version": "2.1.0-beta2"}}
    st["wifi"] = {"ssid": "net1", "sta_ip": "127.0.0.1", "rssi": -40, "status": "got ip"}
    return st


def make_handler(password=None, fail_status=()):
    class H(http.server.BaseHTTPRequestHandler):
        seen, n_status = [], 0

        def _any(self):
            H.seen.append((self.command, self.path))
            if password is not None and not self._auth_ok():
                self.send_response(401)
                self.send_header("WWW-Authenticate", 'Digest qop="auth", realm="%s", nonce="%s", algorithm=SHA-256'
                                 % (REALM, NONCE))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body, code = None, 200
            if self.command != "GET":
                code = 405
            elif self.path == "/shelly":
                body = SHELLY
            elif self.path == "/rpc/Shelly.GetConfig":
                body = CONFIG
            elif self.path == "/rpc/Shelly.GetStatus":
                H.n_status += 1
                body = status_json(H.n_status)
                if H.n_status in fail_status:
                    code, body = 500, {"code": -1, "message": "boom"}
            else:
                code, body = 404, {"code": 404, "message": "No handler"}
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_PUT = do_DELETE = _any

        def _auth_ok(self):
            h = self.headers.get("Authorization", "")
            if not h.startswith("Digest "):
                return False
            f = {k: (q or b) for k, q, b in re.findall(r'(\w+)=(?:"([^"]*)"|([^\s,]+))', h[7:])}
            s = lambda x: hashlib.sha256(x.encode()).hexdigest()
            ha1, ha2 = s("admin:%s:%s" % (REALM, password)), s("GET:%s" % self.path)
            want = s("%s:%s:%s:%s:%s:%s" % (ha1, NONCE, f.get("nc"), f.get("cnonce"), "auth", ha2))
            return (f.get("username") == "admin" and f.get("uri") == self.path and f.get("nonce") == NONCE
                    and f.get("algorithm") == "SHA-256" and f.get("response") == want)

        def log_message(self, *a):
            pass
    return H


class FakeStripCase(unittest.TestCase):
    password, fail_status = None, ()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.H = make_handler(self.password, self.fail_status)
        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self.H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.host = "127.0.0.1:%d" % self.srv.server_address[1]
        self.cfg = os.path.join(self.tmp.name, "strip.json")
        with open(self.cfg, "w") as f:
            json.dump({"host": self.host, "mac": MAC, "device_id": REALM,
                       "outlets": {"0": "pi5", "1": "pi4", "2": "jetson", "3": "router"}}, f)

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = m.main(list(argv))
            except SystemExit as e:                      # a refusal: exit code and message, as the shell would see
                rc = e.code if isinstance(e.code, int) else 1
                err.write(str(e.code))
        return rc, out.getvalue(), err.getvalue()


class TestAgainstFakeStrip(FakeStripCase):
    fail_status = (5,)          # info, info --json and status take GetStatus calls 1-3; the log's 2nd request is call 5

    def test_every_request_is_an_allow_listed_get_and_failed_rows_stay_empty(self):
        rc, out, _ = self.run_tool("info", "--config", self.cfg)
        self.assertEqual(rc, 0)
        self.assertIn("2.0.1", out)
        self.assertNotIn("SECRET", out)
        rc, out, _ = self.run_tool("info", "--config", self.cfg, "--json")
        self.assertEqual(rc, 0)
        self.assertNotIn("SECRET", out)
        info = json.loads(out)
        self.assertEqual(info["available_updates"], {"beta": "2.1.0-beta2"})
        self.assertEqual([o["name"] for o in info["outlets"]], ["pi5", "pi4", "jetson", "router"])
        self.assertEqual(info["control_paths"]["ble_rpc"], False)
        rc, out, _ = self.run_tool("status", "--config", self.cfg)
        self.assertEqual(rc, 0)
        self.assertIn("router", out)
        csvp = os.path.join(self.tmp.name, "log.csv")
        rc, _, err = self.run_tool("log", "--config", self.cfg, "--out", csvp, "--hz", "5", "--duration", "1.0")
        self.assertEqual(rc, 0, err)
        for method, path in self.H.seen:
            self.assertEqual(method, "GET")
            self.assertIn(path, m.READ_ONLY_PATHS)
        with open(csvp, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len(rows), 3)
        self.assertEqual(rows[0]["ok"], "1")
        self.assertEqual(rows[0]["o3_apower"], "40.0")
        self.assertEqual(rows[1]["ok"], "0")
        self.assertIn("500", rows[1]["error"])
        self.assertEqual(rows[1]["o3_apower"], "")
        self.assertEqual(rows[1]["o3_aenergy_total"], "")
        with open(csvp + ".meta.json") as f:
            meta = json.load(f)
        self.assertEqual(meta["stop_reason"], "duration")
        self.assertEqual(meta["rows"], len(rows))
        self.assertEqual(meta["error_rows"], 1)
        self.assertEqual(meta["device"]["mac"], MAC)
        self.assertFalse(meta["firmware_changed"])
        self.assertEqual(meta["outlet_names_on_device"]["3"], "router")
        rc, _, err = self.run_tool("log", "--config", self.cfg, "--out", csvp, "--duration", "1")
        self.assertNotEqual(rc, 0)        # never overwrites a log
        self.assertIn("never overwritten", err)

    def test_refuses_another_device_and_unsafe_rates(self):
        with open(self.cfg, "w") as f:
            json.dump({"host": self.host, "mac": "AABBCCDDEEFF", "outlets": {"3": "router"}}, f)
        rc, _, err = self.run_tool("status", "--config", self.cfg)
        self.assertEqual(rc, 3)
        self.assertIn("DEVICE_MISMATCH", err)
        rc, _, err = self.run_tool("log", "--config", self.cfg, "--out", os.path.join(self.tmp.name, "x.csv"), "--hz", "6")
        self.assertNotEqual(rc, 0)
        self.assertIn("--hz", err)
        self.assertFalse(os.path.exists(os.path.join(self.tmp.name, "x.csv")))

    def test_unreachable_is_nonzero(self):
        rc, _, err = self.run_tool("info", "--host", "127.0.0.1:9", "--config", self.cfg, "--timeout", "0.5")
        self.assertEqual(rc, 2)
        self.assertIn("UNREACHABLE", err)


class TestDigestAuth(FakeStripCase):
    password = "correct horse"

    def test_rfc7616_vectors(self):
        args = ("Mufasa", "http-auth@example.org", "Circle of Life", "GET", "/dir/index.html",
                "7ypf/xlj9XXwfDPEoM4URrv/xwf94BcCAzFZH4GiTo0v", "00000001",
                "f2/wE4q74E6zIJEtWaHKaf5wv/H5QzzpXusqGemxURZJ", "auth")
        self.assertEqual(m.digest_response("MD5", *args), "8ca523f5e9506fed4657c9700eebdbec")
        self.assertEqual(m.digest_response("SHA-256", *args),
                         "753927fa0e85d155564e2e272a28d1802ca10daf4496794697cf8db5856cb6c1")

    def test_answers_the_strips_challenge_from_an_auth_file(self):
        pw = os.path.join(self.tmp.name, "shelly.auth")
        with open(pw, "w") as f:
            f.write(self.password + "\n")
        os.chmod(pw, 0o600)
        rc, out, err = self.run_tool("status", "--config", self.cfg, "--auth-file", pw)
        self.assertEqual(rc, 0, err)
        self.assertNotIn(self.password, out + err)
        rc, _, err = self.run_tool("status", "--config", self.cfg)          # no password: a clear refusal
        self.assertEqual(rc, 2)
        self.assertIn("401", err)
        with open(pw, "w") as f:
            f.write("wrong\n")
        rc, _, err = self.run_tool("status", "--config", self.cfg, "--auth-file", pw)
        self.assertEqual(rc, 2)
        self.assertNotIn("wrong", err.replace("wrong password", ""))

    def test_auth_file_rules(self):
        pw = os.path.join(self.tmp.name, "a.auth")
        with open(pw, "w") as f:
            f.write("pw1\nignored\n")
        os.chmod(pw, 0o644)
        with self.assertRaises(SystemExit):
            m.read_password(pw)                                     # readable by others
        os.chmod(pw, 0o600)
        self.assertEqual(m.read_password(pw), "pw1")
        old = m.ROOT
        try:
            m.ROOT = self.tmp.name                                  # pretend the temp dir is the repo
            with self.assertRaises(SystemExit):
                m.read_password(pw)
        finally:
            m.ROOT = old


if __name__ == "__main__":
    unittest.main()
