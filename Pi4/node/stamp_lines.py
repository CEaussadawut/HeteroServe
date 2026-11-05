#!/usr/bin/env python3
"""Stamp every line arriving on stdin with CLOCK_REALTIME and pass it on.

Used by scripts/06_bench_local.sh to timestamp llama-bench's `--progress` lines so the
energy integral can be cut at repetition boundaries instead of at process boundaries.
Without it the window also contains the GGUF page-in off the microSD and a full-length
warmup prefill, neither of which contributes a token to the denominator.

Deliberately tiny and stdlib-only: this process sits INSIDE the measured window, so
everything it does is charged to the run. It blocks in readline() for all but a few
microseconds per line, and llama-bench emits only reps+1 progress lines per invocation.

The stamp is CLOCK_REALTIME (time.time()), the same base as `date -u +%s.%N` in lib.sh
and as the control node's power stream, which is what makes the two joinable. It is
therefore only meaningful while chrony is synced -- scripts/99_healthcheck.sh gates on it.

    ... --progress -o json 2>&1 >out.json | python3 node/stamp_lines.py > out.stderr.tsv
"""
from __future__ import annotations

import sys
import time


def main() -> int:
    src, dst = sys.stdin, sys.stdout
    # BOTH ends, and both with an explicit encoding. Guarding only stdin was worse than
    # useless: a stray byte decoded to U+FFFD, and then writing U+FFFD to a stdout whose
    # encoding came from the locale raised UnicodeEncodeError. Under systemd or a bare
    # ssh the locale is often ANSI_X3.4-1968, so the stamper died, wrote NOTHING, and
    # llama-bench took SIGPIPE on its next progress write -- destroying the measurement
    # this process exists to record, over one unlucky byte in a log line.
    for stream in (src, dst):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):     # pragma: no cover - <3.7, or a pipe that
            pass                                 # cannot be reconfigured; best effort
    while True:
        # readline(), not `for line in src`: readline returns as soon as a newline is in
        # the pipe. llama.cpp writes stderr unbuffered (C stdio default), so one fprintf
        # is one write(2) is one stamped line. Iteration semantics are not worth
        # depending on here -- the whole value of this file is that it cannot surprise us.
        line = src.readline()
        if not line:
            break
        dst.write("%.6f\t%s\n" % (time.time(), line.rstrip("\n")))
        dst.flush()   # the trace must be complete even if the sweep is killed mid-run
    return 0


if __name__ == "__main__":
    sys.exit(main())
