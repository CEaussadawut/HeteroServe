"""Tests for the throttle poll (testbed/scripts/throttle_poll.sh v2, throttle_summary.py) and its use in compare_pool.py,
stdlib unittest only. From the repo root:

    python3 -m unittest discover -s testbed/tests -q

Nothing here contacts a board: the poller runs locally against a fake vcgencmd, and the v1 log of 2026-10-07 is read
from testbed/results.
"""
import json, os, re, shutil, stat, subprocess, sys, tempfile, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SCRIPTS = os.path.join(ROOT, "testbed", "scripts")
RESULTS = os.path.join(ROOT, "testbed", "results")
sys.path.insert(0, SCRIPTS)
import throttle_summary as ts                                    # noqa: E402
import compare_pool                                              # noqa: E402

T = 1791380000


def line(t, bits, temp="60.0'C", gov="1800000", freq=None):
    return "%d throttled=0x%x temp=%s gov_khz=%s%s" % (t, bits, temp, gov, " frequency(48)=%s" % freq if freq else "")


def v2(rows):
    return ["# throttle_poll v2 pid=1 start=%d" % T] + rows


class TestSummary(unittest.TestCase):
    def test_clean_cell(self):
        f = ts.summarize(v2([line(T + i, 0x80000) for i in range(10)]))
        self.assertEqual((f["poll"], f["samples"], f["span_s"], f["active_s"], f["sticky_new"]), ("v2", 10, 9, 0, "0x0"))
        self.assertEqual(f["sticky_first"], "0x80000")      # sticky from before the cell: not this cell's event
        self.assertEqual(ts.verdict(dict(f, alive="1")), "clean")

    def test_a_now_bit_throttles_and_its_depth_is_reported(self):
        rows = [line(T + i, 0x80000) for i in range(5)] + [line(T + 5, 0x80008, "81.3'C", freq="1726000000")]
        f = ts.summarize(v2(rows))
        self.assertEqual((f["active_s"], f["now_bits"], f["active_min_clock_mhz"], f["max_temp_c"]), (1, "0x8", "1726", "81.3"))
        self.assertEqual(ts.verdict(f), "THROTTLED")

    def test_a_sticky_bit_coming_on_between_polls_throttles(self):
        f = ts.summarize(v2([line(T, 0x0), line(T + 1, 0x10000), line(T + 2, 0x10000)]))
        self.assertEqual((f["active_s"], f["sticky_new"]), (0, "0x10000"))      # an under-voltage between two seconds
        self.assertEqual(ts.verdict(f), "THROTTLED")

    def test_a_sticky_bit_set_before_the_cell_does_not(self):
        self.assertEqual(ts.verdict(ts.summarize(v2([line(T + i, 0x50000) for i in range(3)]))), "clean")

    def test_what_makes_a_poll_not_cover_the_cell(self):
        ok = [line(T + i, 0) for i in range(3)]
        cases = {"no samples": [], "a failed read": ok + ["%d throttled=NA temp=60.0'C gov_khz=1800000" % (T + 3)],
                 "a gap over 3 s": ok + [line(T + 7, 0)], "an unreadable line": ok + ["garbage"],
                 "a failed clock read": ok + [line(T + 3, 0x8, freq="NA")]}
        for why, rows in cases.items():
            with self.subTest(why):
                self.assertEqual(ts.verdict(ts.summarize(v2(rows))), "UNKNOWN")
        self.assertEqual(ts.verdict(dict(ts.summarize(v2(ok)), alive="0")), "UNKNOWN")      # the poller had died
        self.assertEqual(ts.verdict(ts.summarize(v2(ok + [line(T + 5, 0)]))), "clean")        # a 3 s gap is still covered

    def test_the_clock_is_reported_never_judged(self):
        f = ts.summarize(v2([line(T + i, 0, gov="600000") for i in range(5)]))   # the governor idling: no bit
        self.assertEqual((f["below_1750mhz_s"], f["min_clock_mhz"], f["clock"]), (5, "600", "governor"))
        self.assertEqual(ts.verdict(f), "clean")

    def test_a_window(self):
        f = ts.summarize(v2([line(T + i, 0x8 if i == 0 else 0) for i in range(10)]), T + 1, T + 4)
        self.assertEqual((f["samples"], f["active_s"]), (4, 0))

    def test_verdict_reads_a_logged_line_and_refuses_what_it_cannot_read(self):
        f = dict(re.findall(r"(\w+)=(\S+)", "samples=3 span_s=2 max_gap_s=1 na_s=0 bad_lines=0 active_s=0 sticky_new=0x0 alive=1"))
        self.assertEqual(ts.verdict(f), "clean")
        self.assertEqual(ts.verdict({"verdict": "UNKNOWN"}), "UNKNOWN")             # "could not copy its poll log"
        self.assertEqual(ts.verdict(dict(f, active_s="x")), "UNKNOWN")


class TestTheV1LogOf20261007(unittest.TestCase):
    """The summary reproduces, from the raw v1 log, what the evening's write-up reported per cell."""
    LOG = os.path.join(RESULTS, "throttle_poll_pi4_20261007exact.log")
    CELLS = {  # STATE before .. STATE after on the Pi 4's clock: (active_s, below_1750mhz_s, max_temp_c, active_min_clock_mhz)
        "default": ((17, 43, 4), (17, 51, 18), (2, 32, "82.7", "1775")),
        "ref_average": ((17, 52, 0), (18, 0, 18), (5, 46, "82.7", "1677")),
        "default_rep": ((18, 2, 16), (18, 11, 44), (9, 91, "83.2", "1726"))}

    @unittest.skipUnless(os.path.exists(LOG), "the 2026-10-07 poll log is not here")
    def test_per_cell(self):
        import datetime
        tz = datetime.timezone(datetime.timedelta(hours=7))
        ep = lambda hms: int(datetime.datetime(2026, 10, 7, *hms, tzinfo=tz).timestamp())
        for cell, (a, b, want) in self.CELLS.items():
            with self.subTest(cell):
                f = ts.summarize(open(self.LOG), ep(a), ep(b))
                self.assertEqual((f["active_s"], f["below_1750mhz_s"], f["max_temp_c"], f["active_min_clock_mhz"]), want)
                self.assertEqual((f["poll"], f["clock"], f["min_clock_mhz"], ts.verdict(f)), ("v1", "measured", "600", "THROTTLED"))


class TestComparePool(unittest.TestCase):
    def cell_log(self, extra):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, "pool_x_20261007_000000.log")
        with open(p, "w") as f:
            f.write("########## 2026-10-07T21:00:00+07:00 cell x: host pi4, workers [jetson]\n" + "".join(e + "\n" for e in extra))
        return compare_pool.parse(p)

    def test_parse_and_the_cell_takes_the_worst_pi(self):
        clean = "samples=9 span_s=8 max_gap_s=1 na_s=0 bad_lines=0 max_temp_c=70.1 active_s=0 sticky_new=0x0 alive=1 verdict=clean"
        hot = clean.replace("active_s=0", "active_s=3").replace("verdict=clean", "verdict=THROTTLED")
        c = self.cell_log(["THROTTLE pi4 " + clean, "THROTTLE pi5 " + hot])
        self.assertEqual(c["throttle"]["pi4"]["max_temp_c"], "70.1")
        self.assertEqual(compare_pool.thermal(c, ["pi5", "pi4", "jetson"])["verdict"], "THROTTLED")
        self.assertEqual(compare_pool.thermal(c, ["pi4", "jetson"])["verdict"], "clean")
        self.assertNotIn("jetson", compare_pool.thermal(c, ["pi4", "jetson"])["per_node"])     # no vcgencmd there

    def test_the_logged_verdict_is_not_trusted(self):
        c = self.cell_log(["THROTTLE pi4 samples=9 span_s=8 max_gap_s=1 na_s=2 bad_lines=0 active_s=0 sticky_new=0x0 alive=1 verdict=clean"])
        self.assertEqual(compare_pool.thermal(c, ["pi4"])["verdict"], "UNKNOWN")

    def test_old_logs_are_not_polled_and_a_half_polled_cell_is_unknown(self):
        self.assertEqual(compare_pool.thermal(self.cell_log([]), ["pi4", "jetson"])["verdict"], "not polled")
        c = self.cell_log(["THROTTLE pi4 samples=9 span_s=8 max_gap_s=1 na_s=0 bad_lines=0 active_s=0 sticky_new=0x0 alive=1"])
        self.assertEqual(compare_pool.thermal(c, ["pi5", "pi4"])["verdict"], "UNKNOWN")
        self.assertEqual(compare_pool.thermal(self.cell_log([]), ["jetson"])["verdict"], "n/a")

    def test_the_exact_campaign_scores_as_published_plus_thermal(self):
        logs = [os.path.join(RESULTS, f) for f in ("pool_x.q3_14b.default_20261007_174257.log",
                "pool_x.q3_14b.ref_average_20261007_175122.log", "pool_x.q3_14b.default_rep_20261007_180022.log")]
        pub = os.path.join(ROOT, "testbed", "dataset", "pool_results_20261007exact.json")
        if not all(map(os.path.exists, logs + [pub])):
            self.skipTest("the exact campaign's files are not here")
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        out = os.path.join(d, "x.json")
        subprocess.run([sys.executable, os.path.join(SCRIPTS, "compare_pool.py"), "--pred",
                        os.path.join(ROOT, "testbed", "dataset", "pool_predictions_20261007exact.json"), "--out", out] + logs,
                       check=True, capture_output=True, cwd=ROOT)
        new, old = json.load(open(out)), json.load(open(pub))
        for cid, r in new["cells"].items():
            self.assertEqual(r.pop("thermal")["verdict"], "not polled")
            pi = r.pop("page_in")                          # 2026-10-08: the repeat's cold first prefill repetition
            self.assertEqual(pi["flag"], cid == "x.q3_14b.default_rep", (cid, pi))
            self.assertEqual(r, old["cells"][cid])


class TestThePoller(unittest.TestCase):
    ZONE = "/sys/class/thermal/thermal_zone0/temp"
    FREQ = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"

    def test_scripts_parse(self):
        for s in ("run_pool.sh", "throttle_poll.sh", "throttle_poll_ab_pi4.sh"):
            with self.subTest(s):
                subprocess.run(["bash", "-n", os.path.join(SCRIPTS, s)], check=True)

    def test_no_redirection_inside_a_command_substitution(self):
        # bash 5.2 forks twice for $(cmd 2>/dev/null) and once for $(cmd): the poll must stay at one process a second
        src = [l for l in open(os.path.join(SCRIPTS, "throttle_poll.sh")) if not l.lstrip().startswith("#")]
        self.assertFalse([l for l in src if re.search(r"\$\([^)]*[<>]", l)])

    @unittest.skipUnless(os.path.exists(ZONE) and os.path.exists(FREQ), "no thermal zone or cpufreq here")
    def test_it_polls_each_whole_second_and_measures_the_clock_only_when_a_bit_is_set(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        fake = os.path.join(d, "vcgencmd")
        with open(fake, "w") as f:      # soft limit active from the third call on; the clock answers 1775 MHz
            f.write('#!/bin/bash\nn=$(cat %s/n 2>/dev/null || echo 0); echo $((n + 1)) > %s/n\n'
                    'case "$*" in get_throttled) [ $n -ge 2 ] && echo throttled=0x80008 || echo throttled=0x80000 ;;\n'
                    '"measure_clock arm") echo "frequency(48)=1775000000" ;; esac\n' % (d, d))
        os.chmod(fake, stat.S_IRWXU)
        log = os.path.join(d, "poll.log")
        env = dict(os.environ, PATH=d + os.pathsep + os.environ["PATH"])
        subprocess.run(["timeout", "3.5", "bash", os.path.join(SCRIPTS, "throttle_poll.sh"), log], env=env)
        rows = open(log).read().splitlines()
        self.assertTrue(rows[0].startswith("# throttle_poll v2 pid="))
        ts_ = [int(r.split()[0]) for r in rows[1:]]
        self.assertGreaterEqual(len(ts_), 3)
        self.assertEqual(ts_, list(range(ts_[0], ts_[0] + len(ts_))))            # one line per whole second, none skipped
        self.assertNotIn("frequency", rows[1])                                    # no bit: no measure_clock
        self.assertIn("frequency(48)=1775000000", rows[-1])
        f = ts.summarize(rows)
        self.assertEqual((f["poll"], f["na_s"], f["bad_lines"], f["clock"], f["active_min_clock_mhz"]), ("v2", 0, 0, "governor", "1775"))
        self.assertEqual(ts.verdict(f), "THROTTLED")


if __name__ == "__main__":
    unittest.main()
