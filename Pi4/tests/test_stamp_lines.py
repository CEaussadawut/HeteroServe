"""Tests for node/stamp_lines.py -- the wall-clock stamper on llama-bench's stderr.

This process sits INSIDE the measured window and on the downstream end of

    llama-bench ... 2>&1 >"$json" | python3 node/stamp_lines.py > "$tag.stderr.tsv"

so if it dies, llama-bench takes SIGPIPE on its next progress write and the benchmark is
destroyed mid-run. Its only real obligation is therefore: **never die**, whatever arrives
on stdin and whatever the locale is. These tests run it as a real subprocess, because the
failure being guarded against is an encoding property of the actual streams and cannot be
reproduced by calling main() in-process.

    python -m unittest discover -s tests -t .
"""
from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAMPER = ROOT / "node" / "stamp_lines.py"

REAL_MARKERS = (
    b"llama-bench: benchmark 1/1: warmup prompt run\n"
    b"llama-bench: benchmark 1/1: prompt run 1/5\n"
    b"llama-bench: benchmark 1/1: prompt run 2/5\n"
)


def run_stamper(stdin_bytes, env_extra=None):
    env = dict(os.environ)
    env.pop("PYTHONIOENCODING", None)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run([sys.executable, str(STAMPER)], input=stdin_bytes,
                          capture_output=True, env=env, timeout=30)
    return proc


# Forcing a genuinely non-UTF-8 stdout is fiddlier than it looks, and getting it wrong
# makes the regression test below pass for the wrong reason -- which it did, once.
#
# `LC_ALL=C` does NOT work: PEP 540 auto-enables Python's UTF-8 Mode when the locale is C
# or POSIX, so sys.stdout.encoding comes back as 'utf-8' and the failure never occurs.
# PYTHONIOENCODING is the reliable lever, and it is a faithful proxy: what matters is only
# that the stream's encoding cannot represent U+FFFD, which is equally true of a real
# latin-1/ASCII locale on the Pi.
NON_UTF8_ENV = {"PYTHONIOENCODING": "ascii", "LC_ALL": "C", "LANG": "C"}


class TestStamper(unittest.TestCase):
    def test_it_stamps_each_marker_line_with_a_tab_separated_timestamp(self):
        proc = run_stamper(REAL_MARKERS)
        self.assertEqual(proc.returncode, 0, proc.stderr.decode("utf-8", "replace"))
        lines = proc.stdout.decode("utf-8").splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            ts, tab, text = line.partition("\t")
            self.assertTrue(tab, "no tab separator in %r" % line)
            float(ts)                                  # must parse as an epoch time
            self.assertTrue(text.startswith("llama-bench: benchmark"))

    def test_timestamps_are_monotonic_across_lines(self):
        proc = run_stamper(REAL_MARKERS)
        stamps = [float(l.split("\t", 1)[0]) for l in proc.stdout.decode().splitlines()]
        self.assertEqual(stamps, sorted(stamps))

    def test_a_non_utf8_locale_plus_a_stray_byte_does_not_kill_it(self):
        """The regression this file exists for.

        Guarding only stdin meant a stray byte decoded to U+FFFD and then failed to ENCODE
        on a stdout whose encoding came from the locale. Under systemd or a bare ssh that
        locale is often ANSI_X3.4-1968, so the stamper exited 1 having written nothing at
        all, and llama-bench died on the resulting SIGPIPE. One unlucky log byte destroyed
        the measurement.
        """
        dirty = b"llama-bench: benchmark 1/1: prompt run 1/5 \xff\xfe trailing\n"
        proc = run_stamper(dirty, NON_UTF8_ENV)
        self.assertEqual(proc.returncode, 0,
                         "stamper died: " + proc.stderr.decode("utf-8", "replace"))
        self.assertTrue(proc.stdout, "stamper produced no output at all")
        # guard the guard: if this env stops producing a non-UTF-8 stdout, the test above
        # becomes vacuous and would keep passing after a regression.
        probe = subprocess.run(
            [sys.executable, "-c", "import sys;print(sys.stdout.encoding)"],
            capture_output=True, text=True, env=dict(os.environ, **NON_UTF8_ENV))
        self.assertNotIn("utf-8", probe.stdout.strip().lower(),
                         "precondition lost: stdout is UTF-8, so this test proves nothing")
        text = proc.stdout.decode("utf-8", "replace")
        # the marker itself must survive intact -- bench_windows.py regexes it
        self.assertIn("llama-bench: benchmark 1/1: prompt run 1/5", text)

    def test_undecodable_input_still_produces_one_line_per_input_line(self):
        dirty = b"\xff\xfe\xff\n" + REAL_MARKERS
        proc = run_stamper(dirty, NON_UTF8_ENV)
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(len(proc.stdout.decode("utf-8", "replace").splitlines()), 4)

    def test_empty_input_exits_cleanly(self):
        proc = run_stamper(b"")
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout, b"")


if __name__ == "__main__":
    unittest.main()
