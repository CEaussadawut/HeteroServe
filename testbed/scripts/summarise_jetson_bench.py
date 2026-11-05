#!/usr/bin/env python3
"""Summarise a jetson_bench_tinyllama.sh result directory (copied from the board).

Throughput: per test, the mean, the min-max and the rep range (max-min)/mean over the n
repetitions of ONE llama-bench invocation, from the `-oe jsonl` lines. That is a within-run
repetition range -- the same repetitions llama-bench's own `±` (a within-run standard deviation,
not shown) is taken over -- and NOT the across-run spread the house rule asks for: the same
config minutes apart drifts ~2% (testbed/benchmarks.md, "A caution about error bars"). Each
row also carries the power mode its case started in. Test labels come from the record's
n_prompt / n_gen. What a case ran on comes from the record's `devices` and `n_gpu_layers`, not
from `backends` alone: that lists every GPU backend that registered at run time, whatever -dev
said, so it says 'Vulkan' for the CPU-only case too. Its 'CPU' means that none registered -- none
compiled in, or one that failed to initialise (no or broken Vulkan ICD: at the pin
ggml_backend_vk_reg() then returns nullptr and the backend is never registered) -- so a case that
asked for GPU layers ran on the CPU: GPU NOT USED. A GPU label also needs the record's `gpu_info`
(the GPUs llama-bench enumerated at run time) to be non-empty and the case's log not to say
'ggml_vulkan: No devices found.' (what ggml-vulkan prints at the pin when it enumerates none --
its 'Found N Vulkan devices' line is then never reached) or 'Found 0 ... devices'; otherwise it is
UNKNOWN, since llama-bench falls back to the CPU when no GPU is enumerated, whatever -ngl says. An
UNKNOWN row makes its case DEVICE NOT VERIFIED. Both reach the status line and the power line too:
that case's power figures are, or may be, a CPU run's.

Case names: whatever run_case was given -- everything up to the ': ' in '##### case NAME: args'.
The EXIT_, power_window_ and '[before/after NAME]' lines are matched on their fixed suffixes, so
'ngl=12' or 'gpu half' parse. Such a line for a name with no '##### case' header of its own is
reported, and the case whose block it sits in is FAILED: losing that header files the lost
case's rows under the case before it.

Status: a case is FAILED if its EXIT_<case> line is missing or non-zero, if it lacks a pp or a
tg result row, or if it has two rows with the same test label (the bench makes one of each). A
failed case's power window is not summarised. With no result row it covers a load-and-crash;
with rows (a crash at teardown, a run killed in the tg reps) what it holds cannot be vouched for
either, and the report says which. A case is NOT CLEAN (power mode changed mid-case) if its
'[after NAME]' power state (pmode, online cores, clock limits, alarm count) differs from its
'[before NAME]' one -- this needs no guard log -- or if the guard log, where it gives the case
guard figures at all (see Guard), shows a TRIP between '[before NAME]' and the window's end, or
more than one power mode (or a different one from '[before NAME]') inside the window. A case that
starts in a different power state from the run's '[clocks locked]' line is flagged too. A case
whose power_window_end is not after its power_window_start, or whose own repetitions (its records'
samples_ns; not rows another case may have left in its block) add up to more than that whole
window, is CLOCK WENT BACK DURING THE CASE: llama-bench times each repetition on the wall clock
(get_time_ns() is std::chrono::high_resolution_clock at the pin, which libstdc++ makes
system_clock), and the window brackets its whole process, so only a step back makes the
repetitions outlast it -- one inside a repetition wraps its unsigned `t_ns = get_time_ns() -
t_start` (llama-bench.cpp:2473 at the pin) to ~1.8e19 ns. Its repetition timings and its power
window are unreliable, and the window is not summarised. A smaller step back, or any step forward,
leaves the records consistent (the window, and a repetition it falls in, are mis-timed alike): only
the guard session covering the case can show it. A case is NOT MAXN if its '[before NAME]' (the
run's '[clocks locked]' when that is missing), its '[after NAME]' or any guard line inside its
window shows a power mode other than pmode:0000 or cores other than 0-3: MAXN in the board's
/etc/nvpmodel.conf, the mode the protocol runs in. That check is absolute, because the relative
ones all agree on a run done wholly in 5 W -- and the guard never restores MAXN after a trip,
nor does the bench set it. A guard TRIP before the window that no later 'guard start' at
pmode:0000 cleared is named with it, in an unbroken log only (see Guard); a TRIP in a session the
report cannot date, or before any 'guard start', is named only as a TRIP whose place relative to
the run cannot be established. A case is CLOCKS NOT LOCKED if the same state lines show
cpu0 or GPU min != max: jetson_clocks pins min == max, and the bench logs '[clocks locked]'
without checking that jetson_clocks worked (no exit check, no `set -e`), so the tag is no
evidence. Every flag (FAILED, NOT CLEAN, CLOCK WENT BACK, the pin checks, NOT MAXN, CLOCKS NOT
LOCKED, GPU NOT USED, DEVICE NOT VERIFIED, STARTED WARM, MAXN or clock lock not verified, a power
file's sample rate, guard figures withheld, INERT, MIXED, coverage, ...) is repeated on each of the
case's rows in the throughput table, and every one after FAILED and NOT CLEAN heads the case's
power line and its guard line. The power line states NOT CLEAN in its own terms, on each figure
the change may touch (see Power), and the guard line puts it in front of its own: no row and no
power figure -- the guard's input min and peak included -- can be copied out without them.

Pins: run.log's 'model :' SHA-256 must be the TinyLlama 'SHA-256 (full)' row of testbed/models.md,
its 'commit :' and every record's build_commit (the revision the binary itself reports, shown on
each row) llama.pin_sha in ThinkPad/config/cluster.json, and 'patched files' 0. Both files are
found from this script's place in the repo (MODELS_MD / CLUSTER_JSON override that). For MODE=cuda
a commit llama.exceptions declares is allowed as well, and a patched tree is expected -- the CUDA
10.2 port is a patch on the pin -- but which patch it holds cannot be seen from run.log, so that is
BUILD NOT VERIFIED, not passed. A mismatch (MODEL NOT VERIFIED, BUILD NOT AT PIN) or a pin file
that cannot be read (NOT CHECKED) flags every case, and no case is then 'OK'.

Power: the module's own INA3221 (tegrastats POM_5V_IN, channel 0: the Nano MODULE's input rail --
carrier-board loads such as USB, the M.2 card and the fan are outside it) -- mean and peak over each
case's window, with the sample rate actually achieved there (the bench asks tegrastats for more than
it delivers). Its lines carry no time, so the count is checked instead: tegrastats writes at most
one per requested interval (RUN_INTERVAL_MS, default 200, the bench's `--interval 200`), so a file
holding more -- a second writer, or a daemon whose `--stop` failed and kept appending (the bench
checks neither) -- gets its run figures n/a. That cap is loose here: this board delivers ~1.95 Hz
of the 5 Hz asked, so a daemon left running passes it only ~1.6 windows after its case ended. The
rate achieved is checked against the case's idle capture: a TIMED capture (`timeout 15`,
IDLE_CAPTURE_S) minutes before the case, so its count / IDLE_CAPTURE_S is what this board's
tegrastats really delivers -- about 1.9 Hz whatever interval it is asked for (the real run: idle
28 / 15 s = 1.87 Hz, run files at 200 ms 206 / 106.44 s = 1.94 Hz and 909 / 465.72 s = 1.95 Hz),
so a run file should reach about that rate. If the case's capture is missing or unusable (its idle
floor n/a) another case's is taken, and the report says whose; with none, the rate is not checked
(no idle reference). A run file is never the reference: it may be the truncated one. Under
RATE_LOW (0.85) x the reference is flagged LOW SAMPLE RATE (a daemon that died mid-run: the file
may cover only part of the window), over RATE_HIGH (1.2) x HIGH SAMPLE RATE (samples from outside
the window -- a `--stop` that failed, a second writer -- or an idle capture cut short?), figures
kept either way. On this board, whose run files delivered 1.04 and 1.05 x the idle rate (one run's
two cases: all the 1.2 margin rests on), the margins leave small errors unseen: a file that stopped
past ~82% of its window (a loss under ~18%) is not flagged, nor one holding up to ~15% extra
samples. A `tegrastats --stop` that failed on the run's last case (cpu in MODE=vulkan, the only
case in MODE=cuda) thus goes unseen if the dir was collected within ~15% of that case's window
after the run (~70 s for the real cpu case), is HIGH after that, and n/a only past ~1.6 windows.
An idle file holding more than its IDLE_CAPTURE_S (15 s) capture can is n/a too, and no
reference. The window is the whole llama-bench process: start-up, model load, a full-length pp
warm-up (llama-bench.cpp:2382 at the pin; the tg warm-up is one token), then every pp and tg
repetition. So a run mean is neither a pp nor a tg figure, and two cases' run means are not
comparable when their pp/tg time shares differ; a legend says so, and each case's 'mix' line gives
those shares from its records' samples_ns (the warm-ups estimated as one pp repetition plus one
token). Against it, the locked-clock idle floor: the mean of
the idle file taken in the cooldown just before the case, with clocks locked, so it is NOT the
board's real idle. 'Locked' is checked on the state line just before the capture ('[before NAME]',
or '[clocks locked]' without it): cpu0 and GPU min == max, else the figure is labelled 'idle floor
(clocks NOT locked: ...)'. cpuidle off and GPU railgating off, the rest of what jetson_clocks does,
are not logged; they are assumed from the lock. Its first IDLE_DROP_S (default 3 s, the duration
Jetson/docs/10_measurement_protocol.md drops) are discarded. The idle file carries no timestamps, so
that time becomes a sample count at IDLE_INTERVAL_MS (default 500, the bench's
`tegrastats --interval 500`); if tegrastats runs slower than asked, more than 3 s is dropped. Both
are environment overrides. A change of power state is marked where the guard places it. A guard TRIP
between '[before NAME]' and the window opening, or one other power mode than '[before NAME]''s
throughout a window the guard covered, falls in the idle capture and marks the idle floor; a TRIP or
two modes inside the window mark the run mean as mixing power modes. A
'[before NAME]'/'[after NAME]' change the guard cannot place -- a clock or alarm change with the
mode unchanged, or a guard that left holes, logged nothing there or had its figures withheld --
marks both: the run mean may mix power modes, and the idle floor may hold the change, as the capture
starts right after '[before NAME]'.

Start temperature: from each case's '[before NAME]' line; the protocol wants the CPU below
40 C when a case starts (the chain's gate, `-lt 40` on the same truncated reading). The chain
gates only the first case; later ones get the bench's fixed 30 s pause and can start warm.

Guard: jetson_power_guard.sh's log -- the second argument, or <result_dir>/power-guard.log when
that is omitted (the report says which). Per case window: input-voltage minimum, guard peak,
power mode and TRIP lines, with how much of the window the guard actually logged ('covered X of
Y s', window edges included). The guard reads V, I and P with separate `cat`s, so one read can
fail alone: a line's V counts for the input min, the coverage and the guard's own low-voltage
rule, its P for the guard peak, and where the two sets of lines differ in a window each figure
gets its own n and coverage. Any stretch longer than GUARD_HOLE_S (3 s; the guard samples about
once a second) without a reading is printed as a hole, the figures are marked as covering only
the logged part, and the case is flagged; with no dated V reading in the window, its trip count
is n/a, not 0. A hole is called 'guard NOT logging' only where the log shows it: past the log's
latest dated sample it may be where the copy ends, before its earliest where the guard started,
and where the guard wrote a TRIP, or a line with an unreadable value ('V=' or 'P=': a failed
INA3221 read), it was running -- blind to its input where V is empty, as that resets its low
count (a new alarm could still trip it). Also every TRIP in the whole log with its time, including
one a brownout cut off before the restarted guard's 'guard start' landed on the same line.

A session is the lines from one 'guard start' to the next, in log order; lines before the first
'guard start' (a copy whose head was lost) belong to none and are never used. The per-second
lines carry only a time of day, so each is dated from its session's stamp ('guard start', found
anywhere in its line, as a brownout can cut off the line before it) by continuity: a line 0 to
GUARD_STALL_S (60 s) after the one before is the same day, a wrap past midnight within
GUARD_STALL_S the next. Anything else -- any drop, whatever its size, a jump of more than
GUARD_STALL_S, or a TRIP whose own stamp is not 0 to GUARD_STALL_S after the line before it -- is
a clock step, and the session is CLOCK-STEPPED (one WARNING). A case whose window lies in such a
session (the last one stamped at or before the window's start, or one stamped inside the window:
the guard restarted mid-case) gets 'guard session clock-stepped (line N: HH:MM:SS -> HH:MM:SS) --
guard figures withheld' on its status line, every row, its power line and its guard line -- no
input min, guard peak, n, coverage or trip count, and no INERT, MIXED or sub-threshold verdict --
and its rows add that a repetition spanning the step was mis-timed. The part of the session
before the step is withheld too: a step shows the clock was wrong on one side of it, and which
side cannot be told, so nothing is reconstructed. A 'guard start' whose stamp cannot be parsed
opens a session that is never dated either, and a case whose window may lie in it -- the last time
dated before that restart is not after the window's end, and no later 'guard start' is stamped at
or before the window's start -- is withheld the same way. The log is ORDER-BROKEN if a 'guard
start' is stamped no later than the one before it, or earlier than a time already dated before it
(the clock came back across a restart: no RTC battery, no NTP); then no case gets guard figures:
'guard figures withheld -- log order broken by a clock-back restart (lines N, M)', one WARNING. In
an unbroken log the sessions follow each other in time, so a window's dated lines are its own
sessions'. Sample lines carry the time of day only; any clock step in the guard session covering a
case withholds that case's guard figures; a whole-day step of at most GUARD_STALL_S, or a clock
that was already wrong when the guard started, cannot be detected -- the procedure syncs the clock
before the guard starts.

A window in clean sessions is summarised: with lines dated the same second as the one before in one
session (a second writer) it is guard MIXED, figures n/a. The guard acts once per start (tripped=1,
never reset), so each start is tracked on its own: in a window covered by a start whose TRIP line
came before its samples there the guard is INERT, 'trips in window' is n/a, and the case is flagged.
Independently, any run of GUARD_LOW_N (3, the guard's own rule) or more consecutive guard lines
below that start's threshold ('min N mV') that overlaps a window with no TRIP from that start while
it lasted is flagged; a line whose V is unreadable ends the run, as it resets the guard's own count.

Deliberately NOT computed: joules per token. The power window spans model load, warm-up and
every repetition, so dividing it by throughput would produce a figure that looks measured
and is not. Energy per token wants the project's phase-aligned capture.

    python3 testbed/scripts/summarise_jetson_bench.py <result_dir> [power-guard.log]
"""
import datetime as dt
import json
import math
import os
import pathlib
import re
import statistics as st
import sys

ISO = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?[+-]\d\d:\d\d"
START_MAX_C = 40        # a case must START with the CPU below this (chain gate: -lt 40)
GUARD_HOLE_S = 3        # longer than this with no guard sample = the guard was not logging
GUARD_LOW_N = 3         # jetson_power_guard.sh trips after this many samples in a row below its min
IDLE_CAPTURE_S = 15     # the bench's idle capture: `sudo -n timeout 15 tegrastats --interval 500`
RATE_LOW, RATE_HIGH = 0.85, 1.2     # a run file's rate against its idle reference: LOW under, HIGH over (the
                                    # one real run's files sit at 1.04 / 1.05 x: hence the docstring's ~18% / ~15%)
CLOCKBACK = "CLOCK WENT BACK DURING THE CASE: repetition timings and the power window are unreliable"
# The protocol's power mode. The board's /etc/nvpmodel.conf: ID=0 NAME=MAXN, CORE_0..3 online;
# ID=1 NAME=5W, cores 2-3 off. /var/lib/nvpmodel/status (in state lines and guard lines) says pmode:0000.
MAXN = {"pmode": "pmode:0000", "online": "0-3"}
STATE_KEYS = ("pmode", "online", "cpu clocks", "gpu clocks", "alarms")
BASE_KEYS = ("pmode", "online", "cpu clocks", "gpu clocks")   # alarms only ever count up
HERE = pathlib.Path(__file__).resolve().parent                # testbed/scripts: the pins are found from here


def env_num(name, default, zero_ok):
    try:
        v = float(os.environ.get(name, default))
    except ValueError:
        v = -1.0
    if v > 0 or (zero_ok and v == 0):
        return v
    sys.exit(f"{name} must be a {'non-negative' if zero_ok else 'positive'} number")


IDLE_INTERVAL_MS = env_num("IDLE_INTERVAL_MS", "500", zero_ok=False)
RUN_INTERVAL_MS = env_num("RUN_INTERVAL_MS", "200", zero_ok=False)    # `tegrastats --start --interval 200`
IDLE_DROP_S = env_num("IDLE_DROP_S", "3", zero_ok=True)
IDLE_DROP_N = math.ceil(IDLE_DROP_S * 1000 / IDLE_INTERVAL_MS - 1e-9)
# A guard line at most this much later than the one before it (or that much past midnight) follows it;
# anything else is a clock discontinuity. The guard logs every 1-2 s, so a longer stall is a hole anyway.
GUARD_STALL_S = env_num("GUARD_STALL_S", "60", zero_ok=False)
MODELS_MD = pathlib.Path(os.environ.get("MODELS_MD") or HERE.parent / "models.md")
CLUSTER_JSON = pathlib.Path(os.environ.get("CLUSTER_JSON") or HERE.parents[1] / "ThinkPad/config/cluster.json")


def pom(path):
    """POM_5V_IN samples (mW), or None if the file does not exist -- a run that aborted in a
    case's cooldown has its header but no idle/power file, and the rest of the report (trips,
    state log) is what diagnoses that abort, so it must not die here."""
    if not path.exists():
        return None
    vals = []
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"POM_5V_IN (\d+)/", line)
        if m:
            vals.append(int(m.group(1)))
    return vals


def tod(t):
    return t.hour * 3600 + t.minute * 60 + t.second


def hms(x):
    return f"{x // 3600:02d}:{x // 60 % 60:02d}:{x % 60:02d}"


def tod_step(last, now):
    """What a guard line's time of day `now` means after `last` (s since midnight), the dated line
    before it in the same session: 'same' day (0 to GUARD_STALL_S s later), 'next' day (a wrap past
    midnight within GUARD_STALL_S: the guard logs every 1-2 s, so a real midnight is 23:59:5x ->
    00:00:0x) or 'step'. Anything else -- any drop, whatever its size, or a jump of more than
    GUARD_STALL_S -- is a clock discontinuity: a time of day gives neither the sign nor the size of
    a step (the boards' first NTP syncs stepped them 10.9-13.3 h forward, which shows as a drop of
    10.7-13.1 h), so no date is guessed for what follows."""
    if 0 <= now - last <= GUARD_STALL_S:
        return "same"
    return "next" if now < last and now + 86400 - last <= GUARD_STALL_S else "step"


def safe(s, n=100):
    """Printable form of a log fragment: a brownout can leave NULs or binary in a file's tail."""
    s = re.sub(r"\x00+", lambda m: f"<{len(m.group(0))} NUL>", s)
    s = "".join(c if c.isprintable() else f"\\x{ord(c):02x}" for c in s)
    return s if len(s) <= n else s[:n] + "..."


def iso_ts(s):
    try:
        return dt.datetime.fromisoformat(s)
    except ValueError:
        return None


def read_guard(path):
    """Date the guard log, session by session. The per-second lines carry only HH:MM:SS (`date +%T`)
    and the log is appended forever, so matching on clock time alone merges earlier days' readings
    into a later day's window. A session is the lines from one 'guard start' to the next, in log
    order. Its stamp sets the date and UTC offset, and its lines are dated from it by time-of-day
    continuity (tod_step): up to GUARD_STALL_S later is the same day, a wrap past midnight within
    GUARD_STALL_S the next one. Anything else -- or a TRIP whose own stamp is not 0 to GUARD_STALL_S
    after the line before it -- is a clock step: a WARNING naming both lines, and the session is
    clock-stepped ('bad' holds the flag a case it covers gets). None of its lines is then used, those
    dated before the step included: a step shows the clock was wrong on one side of it, and which side
    cannot be told. Nothing is reconstructed; a TRIP is dated by its own stamp, never used to date the
    lines after a step. The stamp is looked for anywhere in the line: power lost mid-write leaves the
    last line cut off (or a NUL tail) and the restarted guard's 'guard start' is appended onto it. So
    every stamped event on a line is read, in order: a TRIP cut off that way is the running session's,
    and only then does the restart open the next session. A 'guard start' with no parseable stamp
    opens a session that is never dated ('from', where it may begin, is the latest time dated before
    it), and lines before the first 'guard start' (a copy whose head was lost) belong to no session.

    The log's order is checked at each stamped 'guard start': one stamped no later than the stamped
    start before it, or earlier than a time already dated (the clock came back across a restart: no
    RTC battery, no NTP), breaks it -- a WARNING for the first, and 'broken' holds the flag that
    withholds every case's guard figures. In an unbroken log the sessions follow each other in time,
    so the dated lines of two sessions never overlap. The flag names both lines -- or one, '(line N)',
    when what the restart undercuts is on the restart's line itself (a TRIP or 'guard start' glued
    before it).

    The guard reads V, I and P with separate `cat`s, so one read can fail and not another: an empty V
    or P (the INA3221 read failed) is kept as None, and the line is still one of the guard's
    iterations -- with V empty its `[ "$v" -lt ... ]` fails and resets its low count: running, blind to
    its input. (Anything else in those fields is not the guard's writing: the line is filed as
    'other'.) Each dated line and each TRIP is filed under its session: the guard trips at most once
    per start, so a window covered by a session whose TRIP line came before its samples there could not
    have got a TRIP line whatever the input did. The session keeps its own threshold ('min N mV') for
    the sub-threshold runs worked out at the end."""
    g = {"iters": [], "trips": [], "sessions": [], "lowruns": [], "warnings": [], "broken": "", "unused": {},
         "slines": [], "vblank": [], "pblank": [], "other": [], "tz": None}
    day = tz = last = prev = hi = None  # the session's running date, zone and time of day, and (epoch, line,
    lost = "before any 'guard start' line"  # time) of its last dated line / of the latest time dated so far

    def session(t, n, text):          # t None: its stamp cannot be read
        mm, pm = re.search(r"\bmin (\d+) mV", text), re.search(r"(pmode:\S+)", text)
        g["sessions"].append({"t": t, "n": n, "from": t if t is not None else hi[0] if hi else -math.inf,
                              "min": int(mm.group(1)) if mm else None, "pmode": pm.group(1) if pm else None,
                              "text": text, "trips": [], "i0": len(g["iters"]), "step": False,
                              "bad": "" if t is not None else f"guard session undated ('guard start' on line {n} "
                                     "has no parseable stamp) -- guard figures withheld"})

    def stepped(n, a, b, why):        # a clock step on line n (time of day a -> b): no line of the session is used
        s = g["sessions"][-1]
        s.update(step=True, bad=f"guard session clock-stepped (line {n}: {a} -> {b}) -- guard figures withheld")
        lost = f"in the clock-stepped session from line {s['n']}"
        g["unused"][lost] = len(g["iters"]) - s["i0"]
        del g["iters"][s["i0"]:]      # those dated before the step go too: which side was wrong cannot be told
        g["warnings"].append(f"line {n}: {why} -- so the guard session from line {s['n']} is clock-stepped: none of"
                             " its lines is used for any case, those before the step included")
        return lost

    for n, line in enumerate(open(path, encoding="utf-8", errors="replace"), 1):
        line = line.rstrip("\r\n")
        ms = [(m, t) for m in re.finditer(rf"({ISO}) (guard start|TRIP)", line)
              if (t := iso_ts(m.group(1))) is not None]
        # Every stamped event on the line, in order, each with its own text up to the next: a brownout can cut
        # off the TRIP the guard wrote as the input sagged, and the restarted guard's 'guard start' then lands on
        # that same line -- the TRIP is the running session's, and only then does the restart open the next.
        for i, (m, t) in enumerate(ms):
            kind, e, a = m.group(2), t.timestamp(), ms[i - 1][0].start() if i else 0
            text = safe(line[m.start():ms[i + 1][0].start() if i + 1 < len(ms) else None], 200)
            if m.start():
                g["warnings"].append(
                    f"line {n}: '{kind}' starts mid-line, after {safe(line[a:m.start()], 40)!r} -- the "
                    "line before it was cut off (power lost mid-write?); dated from the stamp on this line"
                    + (f", and that '{ms[i - 1][0].group(2)}' is read too, from its own stamp" if i else ""))
            g["tz"] = g["tz"] or t.tzinfo
            if kind == "guard start":
                s0 = next((s for s in reversed(g["sessions"]) if s["t"] is not None), None)
                if s0 and not g["broken"] and (e <= s0["t"] or e < hi[0]):
                    p = hi[1] if e < hi[0] else s0["n"]
                    on = f"line {p}" if p != n else "this line, before it"    # (an event glued before the restart)
                    g["broken"] = ("guard figures withheld -- log order broken by a clock-back restart "
                                   + (f"(lines {p}, {n})" if p != n else f"(line {n})"))
                    g["warnings"].append(f"line {n}: 'guard start' stamped {t.isoformat()} is " + (
                        f"earlier than {hi[2]}, already dated on {on}" if e < hi[0] else
                        f"no later than the 'guard start' on {on}") + ": the clock came back across a restart, so "
                        "the log's order is broken and no case gets guard figures")
                session(e, n, text)
                day, tz, last, prev = t.date(), t.tzinfo, tod(t), (e, n, t.isoformat())
            else:                     # a TRIP, dated by its own stamp: in a clean session it follows the line before it
                k = len(g["sessions"]) - 1 if g["sessions"] else None
                g["trips"].append((e, text, n, k))
                if k is not None:
                    g["sessions"][k]["trips"].append((e, n))
                if day is not None and not 0 <= e - prev[0] <= GUARD_STALL_S:
                    day, lost = None, stepped(n, hms(last), hms(tod(t)), f"TRIP stamped {t.isoformat()}, yet line "
                                              f"{prev[1]} before it in the same session reads {prev[2]}")
                elif day is not None:
                    day, tz, last, prev = t.date(), t.tzinfo, tod(t), (e, n, t.isoformat())
            if day is not None:
                hi = prev if not hi or prev[0] >= hi[0] else hi
        if ms:
            continue
        # `date +%T`, then the readings: a number each, or nothing at all when the INA3221 read failed
        m = re.match(r"([01]\d|2[0-3]):([0-5]\d):([0-5]\d) V=(\d*) I=\S* P=(\d*)(?!\S)", line)
        if m:
            g["slines"].append(n)     # every guard iteration, dated or not
            h, mi, s = (int(x) for x in m.groups()[:3])
            now = h * 3600 + mi * 60 + s
            v, p = (int(x) if x else None for x in m.group(4, 5))
            for x, key in ((v, "vblank"), (p, "pblank")):
                if x is None:
                    g[key].append(n)
            if day is not None and tod_step(last, now) == "step":
                how = (f"{now - last} s later, more than GUARD_STALL_S ({GUARD_STALL_S:g} s): a clock step, or a "
                       "stall that long (the two look alike)" if now > last else
                       f"back {last - now} s, or {86400 - last + now} s later across midnight, which is more than "
                       f"GUARD_STALL_S ({GUARD_STALL_S:g} s): a clock step (or a stall across midnight)")
                day, lost = None, stepped(n, hms(last), hms(now), f"the time of day went from {hms(last)} (line "
                                          f"{prev[1]}) to {hms(now)} within one guard session -- {how}")
            if day is None:           # no date to hang it on, or its session's clock stepped
                g["unused"][lost] = g["unused"].get(lost, 0) + 1
                continue
            if now < last:            # tod_step said 'next': past midnight
                day += dt.timedelta(days=1)
            last = now
            at = dt.datetime.combine(day, dt.time(h, mi, s), tz)
            prev = (at.timestamp(), n, at.isoformat())
            hi = prev if not hi or prev[0] >= hi[0] else hi
            pm = re.search(r"(pmode:\S+)", line)
            g["iters"].append((prev[0], v, p, pm.group(1) if pm else "pmode:?", len(g["sessions"]) - 1, n,
                               len(g["slines"])))
            continue
        if "guard start" in line:     # a restart whose date cannot be read: its session is never dated
            session(None, n, safe(line, 200))
            g["warnings"].append(f"line {n}: 'guard start' with no parseable ISO stamp ({safe(line, 60)!r}): the "
                                 "guard restarted at an unknown date, so none of its session's lines is used, and a "
                                 "case that session may cover gets no guard figures")
            day, lost = None, f"in the session from line {n}, whose 'guard start' has no parseable stamp"
            continue
        if "TRIP" in line:            # never lose a trip because its line is not in the usual shape
            k = len(g["sessions"]) - 1 if g["sessions"] else None
            g["trips"].append((None, safe(line, 200), n, k))
            if k is not None:
                g["sessions"][k]["trips"].append((None, n))
            g["warnings"].append(f"line {n}: TRIP with no parseable ISO stamp ({safe(line, 60)!r}): "
                                 "it cannot be placed in any case window")
        elif line.strip():
            g["other"].append((n, line))
    # The TRIPs that can be held against a window: dated, in a session whose clock did not step.
    g["placed"] = [t for t in g["trips"] if t[0] is not None and t[3] is not None and not g["sessions"][t[3]]["bad"]]
    # The guard's own trip rule, replayed over the clean sessions' dated iterations: GUARD_LOW_N or more
    # in a row (one session, as a restart resets its counter) whose V is below that session's
    # threshold. An empty V resets the count, as the guard's own test does; an empty P does not.
    # By log order: 'notrip' when its session wrote no TRIP line from the run's first line to the
    # guard iteration after it (it writes one after the 3rd low sample), 'spent' when that session
    # had tripped before the run began. Matched to windows later, by dates.
    run = []
    for x in g["iters"] + [None]:     # (epoch, V, P, mode, session, line, iteration)
        thr = g["sessions"][x[4]]["min"] if x else None
        low = thr is not None and x[1] is not None and x[1] < thr
        if low and (not run or (run[-1][4] == x[4] and run[-1][6] + 1 == x[6])):
            run.append(x)
            continue
        if len(run) >= GUARD_LOW_N:
            z, tl = run[-1][6], [ln for _, ln in g["sessions"][run[0][4]]["trips"]]   # z: next iteration slines[z]
            lz = g["slines"][z] if z < len(g["slines"]) else math.inf
            g["lowruns"].append({"a": run[0][0], "b": run[-1][0], "n": len(run), "la": run[0][5],
                                 "notrip": not any(run[0][5] < ln < lz for ln in tl),
                                 "spent": bool(tl) and tl[0] < run[0][5], "vmin": min(y[1] for y in run),
                                 "thr": g["sessions"][run[0][4]]["min"]})
        run = [x] if low else []
    return g


def parse_state(line):
    """A bench `state` line: '[TAG] ISO pmode:NNNN online=.. cpu0 min/max=.. gpu min/max=..
    swap_used_kB=.. alarms=N CPU-therm=NNC GPU-therm=NNC'. Missing fields come back as None.
    The tag ends at the '] ' before the stamp, so a case name holding ']' still parses."""
    m = re.match(rf"\[(.+?)\] ({ISO})", line)
    if not m:
        return None
    s = {"tag": m.group(1), "at": m.group(2)}
    pm = re.search(r"(?<!\S)(pmode:\S+)", line)
    s["pmode"] = pm.group(1) if pm else None
    for key, pat in (("online", r"online=(\S+)"), ("cpu clocks", r"cpu0 min/max=(\S+)"),
                     ("gpu clocks", r"gpu min/max=(\S+)"), ("alarms", r"alarms=(\d+)")):
        mm = re.search(pat, line)
        s[key] = mm.group(1) if mm else None
    s["therm"] = dict(re.findall(r"(\S+)-therm=(-?\d+)C", line))
    return s


def state_diff(a, b, keys):
    return [f"{k} {a[k]}->{b[k]}" for k in keys if a.get(k) != b.get(k)]


def off_maxn(s):
    """A state line's mode, if it is not MAXN; '' if it is, or if the line does not say."""
    bad = any(s[k] is not None and s[k] != v for k, v in MAXN.items())
    return f"{s['pmode'] or 'pmode ?'} online={s['online']}" if bad else ""


def clock_lock(s):
    """jetson_clocks' lock as a state line shows it -- 'locked' (cpu0 and GPU min == max), 'NOT
    locked' (either differs) or 'not logged' (a limit missing or unreadable) -- and the limits read.
    The bench's '[clocks locked]' tag is written whether jetson_clocks worked or not."""
    lim = f"cpu0 min/max={s['cpu clocks'] or '?'} gpu min/max={s['gpu clocks'] or '?'}"
    mm = [re.fullmatch(r"(\d+)/(\d+)", s[k] or "") for k in ("cpu clocks", "gpu clocks")]
    if any(m and m.group(1) != m.group(2) for m in mm):
        return "NOT locked", lim
    return ("locked" if all(mm) else "not logged"), lim


def uncleared_trip(g, ref):
    """The last guard TRIP before `ref` that no later 'guard start' at MAXN (up to `ref`) cleared, or
    None: its time -- or, for a TRIP in a clock-stepped or undated session, or before any 'guard
    start', whose stamp cannot be trusted, why its place relative to the run cannot be established.
    Walked only in an unbroken log, where log order is time order. The guard never restores MAXN
    itself, so such a TRIP leaves the board in 5 W unless someone ran nvpmodel -m 0 by hand -- which
    only the pmode on later lines can tell."""
    if g["broken"]:
        return None
    last = "it comes before any 'guard start' line" if any(t[3] is None for t in g["trips"]) else None
    for s in g["sessions"]:
        if s["from"] > ref:
            break
        if s["t"] is not None and s["pmode"] == MAXN["pmode"]:
            last = None
        own = [t for t, _ in s["trips"] if t is not None and t < ref]
        if s["bad"] and s["trips"]:   # its stamps may lie on either side of the run
            last = "its session's clock stepped" if s["step"] else "its session's 'guard start' has no parseable stamp"
        elif own:                     # sessions run in time order: a later TRIP replaces an earlier one
            last = max(own)
    return last


def test_label(r):
    p, n, d = r.get("n_prompt", 0), r.get("n_gen", 0), r.get("n_depth", 0)
    lab = "+".join(x for x in (f"pp{p}" if p else "", f"tg{n}" if n else "") if x) or "?"
    return lab + (f"@d{d}" if d else "")


def kind(r):
    p, n = r.get("n_prompt", 0), r.get("n_gen", 0)
    return "pp" if p and not n else "tg" if n and not p else "pg"


def ran_on(r, found=()):
    # `backends` is every GPU backend that registered at run time, whatever -dev/-ngl said: never
    # use it alone. `devices` is the real choice ('none' = CPU only, 'auto' = every device the build
    # has). `gpu_info` lists the GPUs enumerated at run time (it is filled even for -dev none): if it
    # is empty, no GPU was there to take the layers and llama-bench ran on the CPU whatever -ngl said.
    if "devices" not in r:
        return "UNKNOWN (record has no 'devices' field)"
    dev, ngl, be = str(r["devices"]), r.get("n_gpu_layers"), r.get("backends")
    if dev == "none":
        return "CPU only"
    if be == "CPU":                   # no GPU backend registered: none compiled in, or one failed to initialise
        if str(r.get("gpu_info") or "").strip():
            return f"UNKNOWN (backends='CPU' yet gpu_info={json.dumps(r['gpu_info'])})"
        if ngl is None:
            return f"UNKNOWN (dev={dev}, record has no 'n_gpu_layers')"
        return "CPU only" if ngl == 0 else (f"CPU (-ngl {ngl} asked for the GPU, but no GPU backend "
                                            "registered: backends='CPU')")
    if "gpu_info" not in r:
        return f"UNKNOWN (dev={dev}, record has no 'gpu_info': GPU enumeration not recorded)"
    if not str(r["gpu_info"]).strip():
        return "UNKNOWN: no GPU enumerated (gpu_info empty)"
    if found and max(k for k, _ in found) == 0:
        return "UNKNOWN: case log says " + ", ".join(sorted({f"'{said}'" for _, said in found}))
    if ngl is None:
        return f"UNKNOWN (dev={dev}, record has no 'n_gpu_layers')"
    if ngl == 0:                      # weights on the CPU, but a GPU device can still take big ops
        return f"ngl=0 but dev={dev}: GPU may take ops (NOT CPU-only)"
    if dev != "auto":
        return f"GPU ({dev})"
    return f"GPU ({be})" if be else "GPU (backend unknown: record has no 'backends')"


def coverage(w, xs):
    """(holes, covered s, largest gap) of guard samples `xs` over window `w`. The window's own
    edges count as coverage points, so a guard started late or stopped early shows up as a hole,
    not as a short 'largest gap' between the samples that happen to be inside. Sorted, in case two
    writers' lines interleave."""
    pts = [w["start"]] + sorted(max(x[0], w["start"]) for x in xs) + [w["end"]]
    gaps = list(zip(pts, pts[1:]))
    holes = [(a, b) for a, b in gaps if b - a > GUARD_HOLE_S]
    return holes, w["end"] - w["start"] - sum(b - a for a, b in holes), max(b - a for a, b in gaps)


def guard_window(g, w, before_at):
    """Guard lines for one closed window, and how much of the window they actually cover. None in an
    order-broken log, nor when a session the window lies in -- the last one that may have begun at or
    before its start, and any that began inside it (the guard restarted mid-case) -- is clock-stepped
    or undated: 'withheld' then holds the case's flag ('stepped': a clock step is why), and every
    figure and verdict below stays empty, as no part of such a session is used. Otherwise the window's
    dated lines are those sessions': in an unbroken log sessions follow each other in time, and a
    stepped or undated one has no dated line. 'inw' / 'pw': its lines with a V (input min, coverage:
    the guard watching the input its trip rule reads) and with a P (guard peak), each with its holes,
    seconds covered and largest gap; 'same' when they are the same lines. 'trips' / 'pre': the TRIPs
    dated in the window / after '[before NAME]' before it, from clean sessions (read_guard's 'placed'),
    and 'pmodes' the power modes its lines show. 'mixed': lines dated the same second as the one
    before in one session (one guard sleeps 1 s after each: a second writer). 'inert': (trip time,
    from, to, TRIP line) for each session in the window whose TRIP line, not dated in this window, came
    before its samples here, so that it could write no TRIP there. 'low': sub-threshold runs
    overlapping the window with no TRIP from their own session while they lasted (read_guard, by log
    order: inside a clean session log order and dated order agree)."""
    lo = math.floor(w["start"])       # guard stamps are whole seconds
    ss = g["sessions"]
    near = [k for k, s in enumerate(ss) if s["from"] <= lo][-1:] + [k for k, s in enumerate(ss)
                                                                   if lo < s["from"] <= w["end"]]
    sk = next((k for k in near if ss[k]["bad"]), None)
    bad = g["broken"] or (ss[sk]["bad"] if sk is not None else "")
    it = [] if bad else [x for x in g["iters"] if lo <= x[0] <= w["end"]]
    inw, pw = [x for x in it if x[1] is not None], [x for x in it if x[2] is not None]
    ok = [] if bad else g["placed"]
    tw = [t for t in ok if lo <= t[0] <= w["end"]]
    tp = [t for t in ok if before_at is not None and before_at <= t[0] < lo]
    dup = sum(bool(i) and x[0] == it[i - 1][0] and x[4] == it[i - 1][4] for i, x in enumerate(it))
    sess = sorted({x[4] for x in inw})

    def first_trip(k):                # (time, line) of the session's first TRIP line, or None
        return next(iter(ss[k]["trips"]), None)

    inert = []
    for k in sess:
        t0 = first_trip(k)
        if t0 is None or any(t[2] == t0[1] for t in tw):   # no TRIP, or one dated here: counted, not inert
            continue
        ks = [x for x in inw if x[4] == k]
        after = [x[0] for x in ks if x[5] > t0[1]]         # its samples here logged after its TRIP line
        if after:
            whole = len(after) == len(ks)
            inert.append((t0[0], w["start"] if whole and k == sess[0] else min(after),
                          w["end"] if whole and k == sess[-1] else max(after), t0[1]))
    (holes, covered, largest), (pholes, pcovered, plargest) = coverage(w, inw), coverage(w, pw)
    return {"withheld": bad, "stepped": not g["broken"] and sk is not None and ss[sk]["step"], "it": it,
            "inw": inw, "pw": pw, "same": [x[5] for x in inw] == [x[5] for x in pw], "dur": w["end"] - w["start"],
            "holes": holes, "covered": covered, "largest": largest, "pholes": pholes, "pcovered": pcovered,
            "plargest": plargest, "trips": tw, "pre": tp, "pmodes": sorted({x[3] for x in it}), "inert": inert,
            "low": [] if bad else [r for r in g["lowruns"] if r["notrip"] and r["b"] >= lo and r["a"] <= w["end"]],
            "nothr": [k for k in sess if ss[k]["min"] is None],
            "mixed": (f"{dup} sample(s) dated the same second as the one before (one guard sleeps 1 s between"
                      " samples: two guards writing one log?)") if dup else ""}


def window_mix(recs, dur, nowarm):
    """Where a case's power window went, from its records: the measured repetitions (the sum of
    each row's samples_ns), the warm-ups llama-bench runs before each test unless --no-warmup
    (at the pin a full-length pp run and a single token: taken as one pp repetition and one tg
    repetition / n_gen, so an estimate), and the rest (start-up, model load, contexts, exit). A
    run mean weights the phases by these shares, so it is comparable only where they match."""
    parts, used, warm = [], 0.0, []
    for r in recs:
        ns = r.get("samples_ns")
        if not isinstance(ns, list) or not ns or not all(
                isinstance(x, (int, float)) and not isinstance(x, bool) and x >= 0 for x in ns):
            return f"n/a ({test_label(r)} has no usable samples_ns): the window's pp/tg shares are unknown"
        rep = st.mean(ns) / 1e9
        used += sum(ns) / 1e9
        parts.append(f"{test_label(r)} reps {100 * sum(ns) / 1e9 / dur:.0f}% ({len(ns)} x {rep:.2f} s)")
        if kind(r) == "pp":
            warm.append((rep, f"one {test_label(r)} run"))
        elif kind(r) == "tg":
            warm.append((rep / r["n_gen"], "one token"))
        else:                         # a pp+tg test warms up with both: one repetition is the upper bound
            warm.append((rep, f"up to one {test_label(r)} run"))
    wu = 0.0 if nowarm else sum(x for x, _ in warm)
    parts.append("no warm-up (--no-warmup)" if nowarm else
                 f"warm-ups ~{100 * wu / dur:.0f}% (est.: {' + '.join(x for _, x in warm) or 'none'})")
    return (f"of the {dur:.1f} s window: " + ", ".join(parts)
            + f", the rest ~{100 * (dur - used - wu) / dur:.0f}% (start-up, model load, contexts, exit)")


def pin_check(log, recs):
    """The run against its pins, as case verdicts ([] = verified), and what the commit matched.
    run.log's 'model :' SHA-256 is checked against the TinyLlama 'SHA-256 (full)' row of
    testbed/models.md (MODELS_MD); its 'commit :' -- and every record's build_commit, the revision
    the binary itself reports -- against llama.pin_sha in ThinkPad/config/cluster.json
    (CLUSTER_JSON), with 'patched files: 0'. For mode cuda a value llama.exceptions declares is
    allowed too, and a patched tree is expected (the CUDA 10.2 port is a patch on the pin:
    jetson_build_llama_cuda.sh), but which patch it holds cannot be seen from here, so it is NOT
    verified. A pin file that cannot be read makes its check NOT CHECKED: never passed."""
    def field(key):                   # the value of the bench's 'key : value' header line, or ''
        return next((l.split(":", 1)[1].strip() for l in log if re.match(rf"{key}\s*:", l)), "")

    mode, v = (field("mode").split() or [""])[0], []
    try:
        md = re.search(r"^#+[^\n]*TinyLlama.*?(?=^#+ |\Z)", MODELS_MD.read_text(encoding="utf-8"), re.S | re.M)
        want = re.search(r"SHA-256 \(full\)\s*\|\s*`?([0-9a-f]{64})\b", md.group(0)) if md else None
        got = re.match(r"[0-9a-f]{64}\b", field("model"))
        v += ([f"MODEL NOT CHECKED ({MODELS_MD} has no TinyLlama 'SHA-256 (full)' row)"] if not want else
              ["MODEL NOT VERIFIED (run.log's 'model :' line holds no SHA-256)"] if not got else
              [f"MODEL NOT VERIFIED (sha256 {got.group(0)[:16]}... is not testbed/models.md's TinyLlama "
               f"{want.group(1)[:16]}...)"] if got.group(0) != want.group(1) else [])
    except OSError as e:
        v.append(f"MODEL NOT CHECKED ({MODELS_MD}: {e.strerror or e})")
    try:
        ll = json.loads(CLUSTER_JSON.read_text(encoding="utf-8"))["llama"]
        ok = ({x: "a declared llama.exceptions value" for x in ll.get("exceptions", {}).values()}
              if mode == "cuda" else {})
        ok[ll["pin_sha"]] = "llama.pin_sha"
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        why = e.strerror if isinstance(e, OSError) and e.strerror else "no llama.pin_sha in it"
        return v + [f"BUILD NOT CHECKED ({CLUSTER_JSON}: {why})"], ""
    cm = re.fullmatch(r"([0-9a-f]{40})\s+patched files:\s*(\d+)", field("commit"))
    if not cm:
        v.append("BUILD NOT VERIFIED (run.log's 'commit :' line gives no commit and 'patched files' count)")
    else:
        if cm.group(1) not in ok:
            v.append(f"BUILD NOT AT PIN (commit {cm.group(1)[:8]} is not llama.pin_sha {ll['pin_sha'][:8]}"
                     + (" nor a declared llama.exceptions value" if mode == "cuda" else "") + ")")
        if cm.group(2) != "0":
            v.append(f"BUILD NOT VERIFIED (mode cuda, patched files {cm.group(2)} -- the CUDA 10.2 port is a patched"
                     " tree; which patch, this script cannot see)" if mode == "cuda" else
                     f"BUILD NOT AT PIN (patched files {cm.group(2)}: the tree is not its commit)")
    bad = sorted({str(r.get("build_commit")) for r in recs if not re.fullmatch(r"[0-9a-f]{7,40}", str(
        r.get("build_commit"))) or not any(x.startswith(r["build_commit"]) for x in ok)})
    if bad:
        v.append(f"BUILD NOT AT PIN (llama-bench build_commit {', '.join(bad)}: not an allowed commit)")
    return v, ok.get(cm.group(1), "") if cm else ""


def main(d, guard=None):
    d = pathlib.Path(d)
    if not (d / "run.log").exists():
        sys.exit(f"{d}: no run.log -- not a jetson_bench_tinyllama.sh result directory")
    log = (d / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
    cases, cur, windows, exits, sts, found, bad, args = {}, None, {}, {}, {}, {}, [], {}
    foreign, dups, loose = {}, [], 0  # lines naming another case than their block's; repeated headers;
                                      # result rows before any header

    def owner(name, n):
        if name != cur:
            foreign.setdefault((cur, name), []).append(n)

    for n, line in enumerate(log, 1):
        # A case name is whatever run_case got, so take all of it up to the ': ' the bench writes
        # after it, and key the other lines on their fixed suffixes. A name the regex missed would
        # file that case's rows, silently, under the case before it.
        m = re.match(r"##### case (.+?):(?: |$)", line)
        if m:
            cur = m.group(1)
            if cur in cases:
                dups.append(cur)
            cases.setdefault(cur, [])
            found.setdefault(cur, [])
            args[cur] = line[m.end():].split()    # the llama-bench arguments run_case passed on
            continue
        m = re.match(r"power_window_(start|end) (.+) (\d+(?:\.\d+)?)\s*$", line)
        if m:
            windows.setdefault(m.group(2), {})[m.group(1)] = float(m.group(3))
            owner(m.group(2), n)
            continue
        m = re.match(r"EXIT_(.+)=(\d+)\s*$", line)
        if m:
            exits[m.group(1)] = int(m.group(2))
            owner(m.group(1), n)
            continue
        s = parse_state(line)
        if s:
            sts[s["tag"]] = s
            m = re.match(r"(?:before|after) (.+)$", s["tag"])
            if m:
                owner(m.group(1), n)
            continue
        # The backend's own enumeration, inside this case's llama-bench. With no device, ggml-vulkan
        # at the pin prints 'No devices found.' and returns before its 'Found N Vulkan devices'.
        m = re.search(r"(?i)\bfound (\d+) (\S+) devices?\b|\b(ggml_\w+): no devices found\b", line)
        if m and cur:
            found[cur].append((int(m.group(1)), f"Found {m.group(1)} {m.group(2)} devices") if m.group(1)
                              else (0, f"{m.group(3)}: No devices found."))
            continue
        if line.startswith('{"build_commit"'):
            if cur is None:           # no case to file it under: counted and reported, never shown
                loose += 1
                continue
            try:
                cases[cur].append(json.loads(line))
            except json.JSONDecodeError:  # a line cut off by a crash: say so, keep going
                bad.append(cur)
    states = [l for l in log if l.startswith("[")]

    gauto = guard is None and (d / "power-guard.log").exists()
    if gauto:
        guard = d / "power-guard.log"
    gpath = pathlib.Path(guard) if guard else None
    g = read_guard(gpath) if gpath and gpath.exists() else None
    # Print times in the board's zone, taken from its own ISO stamps -- never the laptop's TZ.
    stamp = next((m.group(0) for l in log if (m := re.search(ISO, l)) and iso_ts(m.group(0))), None)
    if stamp:
        tz, tzsrc = iso_ts(stamp).tzinfo, "run.log ISO stamps"
    elif g and g["tz"]:
        tz, tzsrc = g["tz"], "guard log"
    else:
        tz, tzsrc = dt.timezone.utc, "UNKNOWN -- shown as UTC"

    def ts(e):
        return dt.datetime.fromtimestamp(e, tz).isoformat(timespec="seconds")

    def tripped(t0, ln):              # when a session's first TRIP line was written, as far as it says
        return f"at {ts(t0)}" if t0 is not None else f"on log line {ln} (undated)"

    # The log's dated extremes (lines with a reading), and whether they are its first / last sample lines at all.
    sx = sorted((x for x in g["iters"] if x[1] is not None or x[2] is not None), key=lambda x: (x[0], x[5])) \
        if g else []
    if sx:
        g_first, g_last = sx[0], sx[-1]

    def gap_why(a, b, i=1):
        """What the log shows about [a, b], a stretch of a window with no dated V (i=1) or P (i=2) reading
        -- and no more: '' when the stretch lies inside the log's dated span and the guard wrote no line
        there at all (a window in a session the report cannot date is withheld, never summarised)."""
        why = []
        nb = [x for x in g["iters"] if a <= x[0] <= b and x[i] is None]
        nt = sum(1 for t in g["placed"] if a <= t[0] <= b)
        if not sx:
            why.append("the log holds no dated sample")
        elif a >= g_last[0]:
            why.append(f"the log has no sample dated after {ts(g_last[0])}" + (
                ", its last sample line (the copy ends there, or the guard stopped?)"
                if g_last[5] == g["slines"][-1] else ""))
        elif b <= g_first[0]:
            why.append(f"the log has no sample dated before {ts(g_first[0])}" + (
                ", its first sample line (the guard started later?)" if g_first[5] == g["slines"][0] else ""))
        if nb and i == 1:             # the guard wrote lines here, so it was running: 'NOT logging' is wrong
            why.append(f"{len(nb)} guard line(s) here with " + ("unreadable INA3221 values" if all(
                x[2] is None for x in nb) else "an unreadable V") + ": the guard was running but blind (an empty V "
                "resets its low count: it could not trip on low voltage)")
        elif nb:
            why.append(f"{len(nb)} guard line(s) here with an unreadable P: the guard was running, its peak misses them")
        if nt:
            why.append(f"{nt} TRIP line(s) dated here: the guard was running, its sample lines are missing")
        return "; ".join(why)

    def lockword(s):
        lk, lim = clock_lock(s)
        return ("clocks locked" if lk == "locked" else f"clocks NOT locked ({lim})" if lk == "NOT locked"
                else "clock limits not logged")

    # ---- per-case verdicts first: every flag has to reach the throughput table's rows
    base = sts.get("clocks locked")
    pins, pinned = pin_check(log, [r for rs in cases.values() for r in rs])   # the same for every case
    info = {}
    for name, recs in cases.items():
        c = info[name] = {"why": [], "changed": [], "offbase": [], "proto": [], "notmaxn": [], "pflags": [],
                          "gflags": [], "ga": None, "known": [], "unlocked": [], "mx": "", "pin": pins, "ref": ""}
        if name not in exits:
            c["why"].append("no EXIT line (run aborted?)")
        elif exits[name] != 0:
            c["why"].append(f"llama-bench exit {exits[name]}")
        for k in ("pp", "tg"):        # the bench always runs -p and -n: one row of each
            if not any(kind(r) == k for r in recs):
                c["why"].append(f"no {k} result row")
        labs = [test_label(r) for r in recs]
        for lab in sorted({x for x in labs if labs.count(x) > 1}):
            c["why"].append(f"{labs.count(lab)} {lab} rows where the bench makes one "
                            "(another case's rows filed here?)")
        if name in bad:
            c["why"].append(f"{bad.count(name)} unparseable result line(s)")
        if name in dups:
            c["why"].append(f"{dups.count(name) + 1} '##### case {name}:' headers: runs mixed under one name")
        for (blk, other), ns in foreign.items():
            if blk == name:
                c["why"].append(f"line(s) {','.join(map(str, ns))} of case {other!r} sit in this block"
                                + (" and it has no '##### case' header: its rows may be filed here"
                                   if other not in cases else " (run.log out of order?)"))

        b, a = sts.get(f"before {name}"), sts.get(f"after {name}")
        c["before"], c["after"] = b, a
        cpu = b["therm"].get("CPU") if b else None
        if b is None:
            c["proto"].append("start temperature NOT logged")
        elif cpu is None:
            c["proto"].append("no CPU-therm reading at start")
        elif int(cpu) >= START_MAX_C:
            c["proto"].append(f"STARTED WARM (CPU {cpu}C; protocol wants < {START_MAX_C}C)")
        if b and a:
            diff = state_diff(b, a, STATE_KEYS)
            if diff:
                c["changed"].append(f"'[after {name}]' vs '[before {name}]': " + ", ".join(diff))
        elif b and not c["why"]:
            c["proto"].append(f"no '[after {name}]' line: a mid-case power-mode change NOT checked")
        if b and base:
            diff = state_diff(base, b, BASE_KEYS)
            if diff:
                c["offbase"].append(", ".join(diff))
        # The idle capture follows '[before NAME]' at once, so that line ('[clocks locked]' without
        # it) says whether the idle floor was taken at locked clocks; the tag itself proves nothing.
        s0, lab0 = (b, f"'[before {name}]'") if b else (base, "'[clocks locked]'")
        lk, lim = clock_lock(s0) if s0 else ("not logged", "")
        c["idlename"] = "locked-clock idle floor" if lk == "locked" else "idle floor"
        c["idlelab"] = (c["idlename"] if lk == "locked" else
                        f"idle floor (clocks NOT locked: {lab0} says {lim})" if lk == "NOT locked" else
                        f"idle floor (clock lock NOT verified: {lab0} logs no clock limits)" if s0 else
                        f"idle floor (clock lock NOT verified: no '[before {name}]' or '[clocks locked]')")
        # What the case ran on, from its rows: an UNKNOWN one may have been a CPU run whatever -ngl said,
        # and one that asked for GPU layers with no GPU backend registered was one.
        dv = {ran_on(r, found.get(name, ())) for r in recs}
        c["dev"] = sorted(o for o in dv if o.startswith("UNKNOWN"))
        c["dv"] = f"DEVICE NOT VERIFIED ({' / '.join(c['dev'])})"
        c["nogpu"] = sorted(o for o in dv if o.startswith("CPU (-ngl"))
        c["ng"] = f"GPU NOT USED ({' / '.join(o[5:-1] for o in c['nogpu'])}; it ran on the CPU)"

        # llama-bench times its repetitions on the wall clock, and the window brackets its whole process: an end at
        # or before the start, or the case's repetitions (samples_ns) adding up to more than the window -- a step back
        # inside one wraps its unsigned t_ns to ~1.8e19 ns -- is the clock going back during the case. Not from rows
        # another case may have left in this block (see 'why'): they ran in another window.
        w = windows.get(name, {})
        c["span"] = w["end"] - w["start"] if "start" in w and "end" in w else None
        own = len(set(labs)) == len(labs) and name not in dups and not any(blk == name for blk, _ in foreign)
        c["reps"] = sum(x for r in recs if own and isinstance(r.get("samples_ns"), list) for x in r["samples_ns"]
                        if isinstance(x, (int, float)) and not isinstance(x, bool) and 0 < x < 2 ** 64) / 1e9
        c["clockback"] = c["span"] is not None and (c["span"] <= 0 or c["reps"] > c["span"])
        c["w"], c["closed"] = w, c["span"] is not None and not c["clockback"]
        bt = iso_ts(b["at"]) if b else None
        if g is not None and c["closed"]:
            ga = c["ga"] = guard_window(g, w, bt.timestamp() if bt else None)
            mx = c["mx"] = " (guard log MIXED here: may be another run's)" if ga["mixed"] else ""
            # TRIPs and modes of a session whose clock stepped are not among these (guard_window): if
            # they were this run's, '[after NAME]' would show the 5 W.
            for t, _, _, _ in ga["trips"]:
                c["changed"].append(f"guard TRIP in the window at {ts(t)}{mx}")
            for t, _, _, _ in ga["pre"]:
                c["changed"].append(f"guard TRIP at {ts(t)}, after '[before {name}]' and before the "
                                    f"window opened ({c['idlename']} and whole run affected){mx}")
            known = c["known"] = [p for p in ga["pmodes"] if p != "pmode:?"]
            if len(known) > 1:
                c["changed"].append(f"guard saw {','.join(known)} inside the window{mx}")
            elif known and b and b["pmode"] and known[0] != b["pmode"]:
                c["changed"].append(f"guard saw {known[0]} inside the window, '[before {name}]' says "
                                    f"{b['pmode']}{mx}")
            if ga["withheld"]:        # an order-broken log, or a session that stepped or cannot be dated
                c["gflags"].append(ga["withheld"])
            elif ga["mixed"]:
                c["gflags"].append("guard log MIXED in the window: guard figures n/a")
            elif not ga["inw"] and not ga["pw"]:    # a guard that logged nothing here is no evidence, like an INERT one
                c["gflags"].append("NO dated guard sample in the window: input min, guard peak and trip count n/a")
            elif ga["same"]:
                if ga["holes"]:
                    c["gflags"].append(f"guard log covers only {ga['covered']:.0f} of {ga['dur']:.0f} s of the window")
            else:                     # V and P reads failed on different lines: each figure has its own coverage
                c["gflags"] += (["NO dated V reading in the window: input min and trip count n/a"] if not ga["inw"]
                                else [f"guard V readings cover only {ga['covered']:.0f} of {ga['dur']:.0f} s of the "
                                      "window"] if ga["holes"] else [])
                c["gflags"] += (["NO dated P reading in the window: guard peak n/a"] if not ga["pw"]
                                else [f"guard P readings cover only {ga['pcovered']:.0f} of {ga['dur']:.0f} s of the "
                                      "window"] if ga["pholes"] else [])
            for t0, _, _, ln in ga["inert"]:
                c["gflags"].append(f"guard INERT in the window (tripped {tripped(t0, ln)}; acts once per start)")
            for r in ga["low"]:
                c["gflags"].append(f"input under the guard's {r['thr']} mV threshold for {r['n']} samples"
                                   f" in a row from {ts(r['a'])}, and no TRIP")

        # NOT MAXN: absolute, since every relative check above agrees on a run done wholly in 5 W.
        sl = ([(f"'[before {name}]'", b)] if b else [("'[clocks locked]' (no '[before]')", base)] if base
              else []) + ([(f"'[after {name}]'", a)] if a else [])
        modes = []
        for lab, s in sl:
            if off_maxn(s):
                c["notmaxn"].append(f"{lab} says {off_maxn(s)}")
                modes += [off_maxn(s)] if off_maxn(s) not in modes else []
        offg = [p for p in c["known"] if p != MAXN["pmode"]]
        if offg:
            c["notmaxn"].append(f"guard saw {','.join(offg)} inside the window{c.get('mx', '')}")
            modes += [f"guard saw {p}" for p in offg if not any(m.startswith(p + " ") for m in modes)]
        ref = math.floor(w["start"]) if "start" in w else bt.timestamp() if bt else None
        tr = uncleared_trip(g, ref) if g is not None and ref is not None else None
        tu = f"a guard TRIP whose place relative to the run cannot be established ({tr})" if isinstance(tr, str) else ""
        if c["notmaxn"] and tr is not None:
            c["notmaxn"].append(tu or f"guard TRIP at {ts(tr)}, before the window, never cleared "
                                      f"(no later 'guard start' at {MAXN['pmode']}; the guard never restores MAXN)")
            modes.append(tu or f"guard TRIP at {ts(tr)} never cleared")
        c["nm"] = f"NOT MAXN ({'; '.join(modes)})"
        if not any(s["pmode"] for _, s in sl) and not c["known"]:
            c["proto"].append("power mode NOT logged: MAXN not verified" + (
                "" if tr is None else f" ({tu})" if tu else
                f" (and the guard TRIP at {ts(tr)} was never cleared: probably 5 W)"))

        # CLOCKS NOT LOCKED: absolute as well, on the same lines -- the relative checks all agree on
        # a run where jetson_clocks failed, and the bench logs '[clocks locked]' either way.
        bylim, nolog = {}, []
        for lab, s in sl:
            lk, lim = clock_lock(s)
            if lk == "NOT locked":
                bylim.setdefault(lim, []).append(lab)
            elif lk == "not logged":
                nolog.append(lab)
        c["unlocked"] = [f"{', '.join(labs)} {'say' if len(labs) > 1 else 'says'} {lim}"
                         for lim, labs in bylim.items()]
        c["cl"] = f"CLOCKS NOT LOCKED ({'; '.join(c['unlocked'])})"
        if nolog or not sl:
            c["proto"].append("clock lock NOT verified (" + (
                f"{', '.join(nolog)} {'log' if len(nolog) > 1 else 'logs'} no clock limits" if nolog
                else "no state line") + ")")

    # The power file against its window. tegrastats lines carry no time, so the count is all there is:
    # it writes at most one per requested interval, so more (a sample or two of slack for its start
    # and stop) means samples from outside the window -- the bench never checks its `tegrastats --stop`
    # -- and the figures are n/a; at this board's ~1.95 Hz a daemon left running gets there only ~1.6
    # windows on, so a shorter tail is the rate check's (HIGH from ~15% extra, unseen below). The rate
    # achieved is held against the idle capture: timed, so its count / IDLE_CAPTURE_S is what the
    # board's tegrastats delivers -- where the report can use the capture at all (an idle floor n/a for
    # too few samples, or for more than it can hold, is no reference either). Never against another run
    # file, which may be the truncated one: with no usable idle capture in the run, the rate is not
    # checked.
    icap = int(IDLE_CAPTURE_S * 1000 / IDLE_INTERVAL_MS) + 1
    for name, c in info.items():
        c["idle"] = pom(d / f"idle_before_{name}.txt")
    iref = {n: len(c["idle"]) / IDLE_CAPTURE_S for n, c in info.items()
            if c["idle"] is not None and IDLE_DROP_N < len(c["idle"]) <= icap}
    for name, c in info.items():
        c["run"] = pom(d / f"power_{name}.txt")
        if c["run"] and c["closed"] and not c["why"]:
            dur = c["w"]["end"] - c["w"]["start"]
            c["cap"] = int(dur * 1000 / RUN_INTERVAL_MS) + 2
            if len(c["run"]) > c["cap"]:
                continue
            r, src = len(c["run"]) / dur, name if name in iref else next((k for k in info if k in iref), None)
            if src is None:
                c["ref"] = "sample rate not checked (no idle reference)"
                c["pflags"].append(c["ref"])
                continue
            c["ref"] = f"idle reference {iref[src]:.2f} Hz" + ("" if src == name else (
                f" from case {src}'s idle capture -- this case's is {'missing' if c['idle'] is None else 'unusable'}"))
            if r < RATE_LOW * iref[src]:
                c["pflags"].append(f"LOW SAMPLE RATE ({r:.2f} Hz in the power file, under {RATE_LOW:g} x the "
                                   f"{c['ref']}): it may cover only part of the window")
            elif r > RATE_HIGH * iref[src]:
                c["pflags"].append(f"HIGH SAMPLE RATE ({r:.2f} Hz in the power file, over {RATE_HIGH:g} x the "
                                   f"{c['ref']}): a second writer or samples from outside the window -- or an idle "
                                   "capture cut short?")

    def verdicts(c):                  # every flag after FAILED and NOT CLEAN (those head the status line)
        return [CLOCKBACK] * c["clockback"] + c["pin"] + ([c["nm"]] if c["notmaxn"] else []) \
            + ([c["cl"]] if c["unlocked"] else []) + ([c["ng"]] if c["nogpu"] else []) + ([c["dv"]] if c["dev"] else []) \
            + [f"started OFF the run's '[clocks locked]' power state ({o})" for o in c["offbase"]] \
            + c["pflags"] + c["gflags"] + c["proto"]

    def tail(c, row=False):           # a row adds what a clock step in the window's guard session does to it
        t = (["case FAILED"] if c["why"] else []) \
            + (["POWER MODE CHANGED MID-CASE -- not a clean run"] if c["changed"] else []) + verdicts(c)
        if row and c["ga"] and c["ga"]["stepped"]:
            t = [x + " (the board clock stepped in this guard session: a repetition spanning the step was mis-timed)"
                 if x == c["ga"]["withheld"] else x for x in t]
        return ("  <- " + "; ".join(t)) if t else ""

    print(f"results : {d}")
    for l in log[:6]:
        if l.startswith(("mode", "commit", "model")):
            print(f"  {l[:110]}")
    print("  pins   : " + ("; ".join(pins) + "  <- on every case" if pins else
                           "verified -- model = testbed/models.md's TinyLlama 'SHA-256 (full)'; commit and every"
                           f" record's build_commit = {pinned} (ThinkPad/config/cluster.json); patched files 0"))
    be = sorted({r.get("backends", "?") for c in cases.values() for r in c})
    print(f"  backends registered at run time (llama-bench 'backends'; NOT what a case ran on -- see 'ran on'):"
          f" {', '.join(be) or '-'}" + ("  <- 'CPU' = no GPU backend registered: none compiled in, or one that"
                                        " failed to initialise (no or broken Vulkan ICD)" if "CPU" in be else ""))
    print(f"  board time zone: {dt.datetime.now(tz).strftime('UTC%z')} (from {tzsrc})")
    if gpath is None:
        print("  guard log: none (not given, and no power-guard.log in the result dir)")
    else:
        print(f"  guard log: {gpath}" + ("  (not given on the command line: the result dir's own copy, used)"
                                         if gauto else "") + ("" if g is not None else "  -- NOT FOUND"))
    if base:
        print(f"  power mode at '[clocks locked]': {base['pmode'] or 'pmode not logged'} online={base['online']}"
              + (f"  <- NOT MAXN (MAXN = {MAXN['pmode']} online={MAXN['online']}, /etc/nvpmodel.conf ID=0)"
                 if off_maxn(base) else " (MAXN)" if base["pmode"] else ""))
        lk, lim = clock_lock(base)
        print(f"  clocks at '[clocks locked]': {lim}" + (
            " (locked: min == max)" if lk == "locked" else
            "  <- NOT locked (min != max: jetson_clocks did not take; the bench writes this tag whatever"
            " jetson_clocks returned)" if lk == "NOT locked" else
            "  <- lock NOT verified (clock limits not logged)"))
    else:
        print("  power mode: NOT logged (no '[clocks locked]' line)")
    for (blk, other), ns in foreign.items():
        if other not in cases:
            print(f"  WARNING run.log line(s) {','.join(map(str, ns))} belong to case {other!r}, which has no"
                  " '##### case' header"
                  + (f" (its EXIT line: llama-bench exit {exits[other]})" if other in exits else "")
                  + (f": filed in case {blk!r}'s block, which is marked FAILED" if blk
                     else ", before any case header"))
    if loose:
        print(f"  WARNING {loose} result row(s) before any '##### case' header: filed under no case, NOT shown")
    print()

    tw = max([8] + [len(test_label(r)) for c in cases.values() for r in c])
    cw = max([7] + [len(n) for n in cases])  # any name run_case took, so the column fits it
    print(f"{'case':{cw}} {'test':{tw}} {'mean t/s':>9}  {'min-max':>17}  {'rep range':>9}  n  ran on")
    for name, recs in cases.items():
        tl = tail(info[name], row=True)
        s0 = info[name]["before"] or base
        mode = (f" {'start' if info[name]['before'] else 'clocks-locked'} {s0['pmode'] or 'pmode ?'}"
                f" online={s0['online']}" if s0 else " start mode NOT logged")
        for r in recs:
            s = r.get("samples_ts") or []
            if not s:
                print(f"{name:{cw}} {test_label(r):{tw}} {'(no samples)':>9}{tl}")
                continue
            m = st.mean(s)
            # Under 3 repetitions a range (0.00% at n=1) reads as precision: blank it and say so.
            spread = f"{100 * (max(s) - min(s)) / m:8.2f}%" if len(s) >= 3 else "      n/a"
            note = "  n<3: NOT reportable" if len(s) < 3 else ""
            gi = json.dumps(r["gpu_info"]) if "gpu_info" in r else "(absent)"
            print(f"{name:{cw}} {test_label(r):{tw}} {m:9.3f}  [{min(s):7.3f}-{max(s):7.3f}]"
                  f"  {spread}  {len(s)}  {ran_on(r, found.get(name, ()))}"
                  f"  [dev={r.get('devices', '?')} ngl={r.get('n_gpu_layers')} t={r.get('n_threads')}"
                  f" rev {r.get('build_commit', '?')}{mode} gpu_info={gi}]{note}{tl}")
        for k in ("pp", "tg"):
            if not any(kind(r) == k for r in recs):
                print(f"{name:{cw}} {k + '?':{tw}} {'(no row)':>9}{tl}")
    print("  rep range = (max-min)/mean over the n repetitions of ONE llama-bench invocation: a within-run")
    print("    repetition range, NOT the across-run spread the house rule asks for (the same config minutes")
    print("    apart drifts ~2%: testbed/benchmarks.md, \"A caution about error bars\")")
    print("  gpu_info = GPUs llama-bench enumerated at run time (listed even for -dev none); a GPU label needs it")
    print()

    allk = all(info[n]["idlename"] == "locked-clock idle floor" for n in cases)
    print(f"{'locked-clock idle floor' if allk else 'idle floor (locked-clock only where a case says so)'}:"
          f" first {IDLE_DROP_S:g} s dropped = {IDLE_DROP_N} samples at the requested {IDLE_INTERVAL_MS:g} ms"
          " (IDLE_DROP_S / IDLE_INTERVAL_MS)")
    print("  run mean / peak = the whole llama-bench invocation (start-up, model load, a full-length pp warm-up,")
    print("    every pp and tg repetition): NOT a pp or a tg figure, and NOT comparable across cases whose pp/tg")
    print("    time shares differ -- each case's 'mix' line gives its shares, from the records' samples_ns")
    print(f"  sample counts: tegrastats writes at most one per requested interval (run {RUN_INTERVAL_MS:g} ms,"
          f" idle {IDLE_INTERVAL_MS:g} ms: RUN_INTERVAL_MS / IDLE_INTERVAL_MS), so a file holding more is n/a;")
    print(f"    a run file's rate against its case's idle capture (timed {IDLE_CAPTURE_S} s: what this board's tegrastats"
          f" delivers): under {RATE_LOW:g} x is LOW, over {RATE_HIGH:g} x HIGH")
    print("    -- small errors pass: a file missing under ~18% of its samples, or holding up to ~15% extra (e.g. a"
          " `tegrastats --stop` that")
    print("    failed on the run's last case, its dir collected soon after the run; n/a only past ~1.6 windows) --"
          f" the {RATE_HIGH:g} margin and these")
    print("    limits rest on one run's two cases (its run files at 1.04 / 1.05 x the idle rate)")
    for name in cases:
        c = info[name]
        why, b, a, w, closed, ga = c["why"], c["before"], c["after"], c["w"], c["closed"], c["ga"]
        fl = verdicts(c)
        flags = ("  <- " + "; ".join(fl)) if fl else ""
        if why:
            head = f"FAILED -- {'; '.join(why)}"
        elif c["changed"]:
            head = f"NOT CLEAN -- POWER MODE CHANGED MID-CASE, not a clean run (llama-bench exit {exits[name]})"
        elif c["clockback"]:
            head = f"{CLOCKBACK.replace(':', ' --', 1)} (llama-bench exit {exits[name]})"
        elif c["notmaxn"]:
            head = f"NOT MAXN -- not run in the protocol's power mode (llama-bench exit {exits[name]})"
        elif c["unlocked"]:
            head = f"CLOCKS NOT LOCKED -- not run at the protocol's locked clocks (llama-bench exit {exits[name]})"
        elif c["nogpu"]:
            head = f"GPU NOT USED -- asked for GPU layers, ran on the CPU (llama-bench exit {exits[name]})"
        elif c["dev"]:
            head = f"DEVICE NOT VERIFIED -- what it ran on is not established (llama-bench exit {exits[name]})"
        elif c["pin"]:                # a pin not matched, or not checked: never 'OK'
            head = (f"{c['pin'][0].split(' (')[0]} -- the pinned model on the pinned tree is not established"
                    f" (llama-bench exit {exits[name]})")
        else:
            head = f"OK (llama-bench exit {exits[name]})"
        print(f"case {name}: {head}{flags}")
        for ch in c["changed"]:
            print(f"  changed  {ch}")
        for x in c["notmaxn"]:
            print(f"  NOT MAXN {x}")
        if b:
            th = b["therm"]
            temps = " ".join(f"{k} {v}C" for k, v in th.items()) or "no temperatures"
            flag = ("  <- no CPU-therm reading" if th.get("CPU") is None else
                    f"  <- STARTED WARM: CPU not below {START_MAX_C} C (protocol)"
                    if int(th["CPU"]) >= START_MAX_C else "")
            print(f"  start    {temps} {b['pmode'] or 'pmode not logged'} online={b['online']} {lockword(b)}"
                  f" at {b['at']}{flag}")
        else:
            print(f"  start    not logged (no '[before {name}]' line)")
        if a:
            temps = " ".join(f"{k} {v}C" for k, v in a["therm"].items()) or "no temperatures"
            same = ("power state unchanged from start" if b and not state_diff(b, a, STATE_KEYS)
                    else "power state CHANGED from start (see 'changed')" if b else "no start line to compare")
            print(f"  end      {temps} {a['pmode'] or 'pmode not logged'} online={a['online']} {lockword(a)}"
                  f" at {a['at']}; {same}")
        else:
            print(f"  end      not logged (no '[after {name}]' line)")

        if closed:
            dur = w["end"] - w["start"]
            print(f"  window   {ts(w['start'])} -> {ts(w['end'])} ({dur:.1f} s)")
        elif c["clockback"] and c["span"] <= 0:
            print(f"  window   opened {ts(w['start'])}, 'closed' {ts(w['end'])} ({w['end'] - w['start']:.1f} s): "
                  "power_window_end is not after power_window_start -- the board clock went back during the case")
        elif c["clockback"]:
            print(f"  window   {ts(w['start'])} -> {ts(w['end'])} ({c['span']:.1f} s), yet its records' samples_ns put "
                  f"{c['reps']:.1f} s of repetitions in it: more than the whole window -- the board clock went back "
                  "during the case")
        elif "start" in w:
            print(f"  window   opened {ts(w['start'])}, never closed properly (no/bad power_window_end)")
        else:
            print("  window   none (no power_window_start line: aborted before it opened)")

        idle = c["idle"]
        run = c["run"]
        il = c["idlelab"]           # 'locked-clock' only when the line before the capture shows the lock
        if idle is None:
            ip = f"{il}: file missing (idle_before_{name}.txt)"
        elif len(idle) > icap:      # more than the capture can hold: samples from elsewhere
            ip = (f"{il}: n/a -- {len(idle)} samples, more than a {IDLE_CAPTURE_S} s capture at the requested "
                  f"{IDLE_INTERVAL_MS:g} ms holds (at most {icap}): the file holds samples from elsewhere")
        elif len(idle) > IDLE_DROP_N:
            ip = f"{il} {st.mean(idle[IDLE_DROP_N:]):5.0f} mW (n={len(idle) - IDLE_DROP_N} of {len(idle)})"
        else:
            ip = f"{il}: n/a (only {len(idle)} samples, {IDLE_DROP_N} dropped)"
        # Where did a change fall? A TRIP after '[before]' but before the window is in the idle
        # capture (it follows '[before]' at once and runs up to the window), and so is a mode the
        # guard saw throughout the window other than '[before]''s; only a TRIP or two modes inside
        # the window make the run mean a mix. The guard's pmode can place a mode change, not a clock
        # or alarm change with the mode unchanged, nor anything when it logged nothing in the window,
        # left holes or had its figures withheld: such a change may lie in the window or in the idle
        # capture, so it marks both.
        known = c["known"]
        placed = bool(known) and not ga["holes"] and not ga["mixed"]     # (withheld: nothing known)
        seen = len(known) == 1 and b and b["pmode"] and known[0] != b["pmode"]
        if ga and ga["pre"]:
            ip += f" NOT CLEAN (guard TRIP at {ts(ga['pre'][0][0])} during the idle capture{c['mx']})"
        elif placed and seen and not ga["trips"]:
            ip += f" NOT CLEAN (power mode changed to {known[0]} after '[before {name}]', before the window)"
        diff = state_diff(b, a, STATE_KEYS) if b and a else []
        if ga and (ga["trips"] or len(known) > 1):
            mix = "NOT CLEAN (mixes power modes) -- "
        elif (diff or seen) and not (placed and (seen or b["pmode"] != a["pmode"])):
            mix = "NOT CLEAN (power state changed during the case, not placed: may mix power modes) -- "
            ip += " NOT CLEAN (power state changed during the case, not placed: may lie in this capture)"
        else:
            mix = ""
        if run is None:
            rp = f"run: file missing (power_{name}.txt)"
        elif why and not cases[name] and name not in bad:
            rp = (f"run: NOT A RUN -- the case FAILED before any result row (load, warm-up or the first "
                  f"test): a load/crash, not a benchmark; not summarised ({len(run)} samples)")
        elif why:
            rp = (f"run: not summarised -- case FAILED ({'; '.join(why)}): what this window contains "
                  f"cannot be vouched for, although result rows survive ({len(run)} samples)")
        elif not closed:
            rp = ("run: not summarised -- " + ("the window cannot hold its own repetitions" if c["clockback"]
                                               and c["span"] > 0 else "no closed window") + f" ({len(run)} samples)")
        elif not run:
            rp = "run: no samples"
        elif len(run) > c["cap"]:
            rp = (f"run: n/a -- {len(run)} samples in the {dur:.1f} s window, more than tegrastats writes at the "
                  f"requested {RUN_INTERVAL_MS:g} ms (at most {c['cap']}): the file holds samples from outside the "
                  "window (a tegrastats that did not stop, or a second writer)")
        else:
            # n / window length; the file holds the odd sample from just outside the window too.
            rp = mix + (f"run mean {st.mean(run):5.0f} mW, peak {max(run):5.0f} mW "
                        f"(n={len(run)}, achieved {len(run) / dur:.2f} Hz; {c['ref']})")
        # Power figures get copied out on their own, so every verdict after FAILED / NOT CLEAN (which
        # this line states in its own terms) heads it: STARTED WARM, MAXN not verified, ... as well.
        pfx = "; ".join(fl) + " -- " if fl else ""
        print(f"  power    {pfx}{ip} | {rp}")
        if not why and closed:
            print(f"  mix      {window_mix(cases[name], dur, '--no-warmup' in args.get(name, []))}")

        # The guard's own figures are power figures too: the case's verdicts head them as they do the
        # power line, and NOT CLEAN with them -- this line has no figure of its own to mark.
        gfl = ["NOT CLEAN (power state changed during the case)"] * bool(c["changed"]) + fl
        gpfx = "; ".join(gfl) + " -- " if gfl else ""
        if gpath is None:
            print("  guard    no guard log supplied")
        elif g is None:
            print(f"  guard    log NOT FOUND: {gpath}")
        elif not sx and not g["trips"]:
            print(f"  guard    {gpfx}log supplied but " + (
                f"none of its {sum(g['unused'].values())} sample line(s) can be used (see its WARNINGs below)"
                if g["unused"] and g["warnings"] else "no dated line in it (no 'guard start' line?)"))
        elif not closed:
            print("  guard    " + ("no reliable window to match against (the board clock went back during the case)"
                                   if c["clockback"] and c["span"] > 0 else "no closed window to match against"))
        else:
            inw, pw, trips, holes = ga["inw"], ga["pw"], ga["trips"], ga["holes"]
            cov, pcov = (f"{ga[k]:.0f} of {ga['dur']:.0f} s" for k in ("covered", "pcovered"))
            undated = sum(1 for t in g["trips"] if t[0] is None)
            label = "  guard    (diagnostic only) " if why else "  guard    "
            mixed, gw = ga["mixed"], ""
            if ga["withheld"]:        # an order-broken log, or a session that stepped or cannot be dated
                msg = "no input min, guard peak, n, coverage or trip count"
            elif ga["it"] and mixed:  # merged figures would mix two writers' readings
                msg = (f"MIXED -- n={len(inw)} samples dated in this window, but {mixed}: input min, guard "
                       "peak and coverage n/a (which samples, if any, are this run's cannot be told), "
                       f"{','.join(ga['pmodes'])}")
            elif inw and ga["same"]:
                msg = (f"input min {min(x[1] for x in inw)} mV, guard peak {max(x[2] for x in inw)} mW "
                       f"(n={len(inw)}, covered {cov}, largest gap {ga['largest']:.0f} s), "
                       f"{','.join(ga['pmodes'])}")
            elif inw or pw:           # V and P reads failed on different lines: each figure on its own lines
                msg = ((f"input min {min(x[1] for x in inw)} mV (n={len(inw)}, covered {cov}, largest gap "
                        f"{ga['largest']:.0f} s)" if inw else "input min n/a (no V reading dated in this window)")
                       + (f", guard peak {max(x[2] for x in pw)} mW (n={len(pw)}, covered {pcov}, largest gap "
                          f"{ga['plargest']:.0f} s)" if pw else ", guard peak n/a (no P reading dated in this window)")
                       + f", {','.join(ga['pmodes'])}")
            else:                     # say only what the log shows: its copy may simply not reach here
                gw = gap_why(w["start"], w["end"])
                msg = (f"NO dated guard samples in this window (log supplied, covered {cov}; "
                       f"{gw or 'see its dated span below'})")
            # A guard start that has tripped keeps logging but can never trip again: its zero is no
            # evidence -- and neither is the zero of a guard that logged nothing here, or read no V.
            inert = ga["inert"]
            if not ga["withheld"]:    # (a withheld window's trip count is not given, said above)
                if mixed:
                    msg += f"; trips in window: n/a ({len(trips)} TRIP line(s) dated here)"
                elif not inw:
                    msg += "; trips in window: n/a (" + ("no V reading here: blind to its input" if pw else
                                                        "no guard samples here") + (
                        f"; {len(trips)} TRIP line(s) dated here" if trips else "") + ")"
                elif len(inert) == 1 and not trips and inert[0][1] == w["start"] and inert[0][2] == w["end"]:
                    msg += (f"; trips in window: n/a (guard INERT: this 'guard start' already tripped "
                            f"{tripped(inert[0][0], inert[0][3])} and acts once per start)")
                else:
                    msg += f"; trips in window: {len(trips)}" + "".join(
                        f" (guard INERT {ts(a0)} -> {ts(b0)}: its 'guard start' already tripped {tripped(t0, ln)})"
                        for t0, a0, b0, ln in inert)
                if ga["nothr"]:
                    msg += " (a 'guard start' here gives no 'min N mV': sub-threshold runs NOT checked)"
                if not inw and not pw:        # (lines dated here, blank or doubled, show it was running)
                    msg += "" if gw or ga["it"] else " (guard NOT logging at any point in this window)"
                elif not mixed and ga["same"]:
                    msg += f" (PARTIAL: guard samples cover only {cov})" if holes else ""
                elif not mixed:
                    msg += (f" (PARTIAL: V readings cover only {cov})" if inw and holes else "") + (
                        f" (PARTIAL: P readings cover only {pcov})" if pw and ga["pholes"] else "")
            if undated:
                msg += f" (+{undated} undated TRIP line(s) that cannot be placed)"
            if g["warnings"]:
                msg += f" [guard log has {len(g['warnings'])} WARNING(s): see below]"
            lines = [f"             {t[1]}" for t in trips + ga["pre"]]
            for r in ga["low"]:
                lines.append(f"           <- input below the guard's {r['thr']} mV threshold for {r['n']} samples"
                             f" in a row, {ts(r['a'])} -> {ts(r['b'])} (min {r['vmin']} mV), and no TRIP: "
                             + ("its 'guard start' had already tripped (INERT)" if r["spent"] else
                                "the guard was armed, so its TRIP line is missing from the log"))
            # The holes: one list where the V and P lines are the same, else V's, then those only P has.
            parts = [(1, holes, cov, "guard sample", "min V / peak P / trip count cover")] if ga["same"] else (
                [(1, holes, cov, "V reading", "min V / trip count cover")] * bool(inw)
                + [(2, [h for h in ga["pholes"] if not inw or h not in holes], pcov, "P reading", "peak P covers")]
                * bool(pw))
            for i, hs, cv, what, figs in parts if (inw or pw) and not mixed else ():
                for h0, h1 in hs[:5]:
                    hw = gap_why(h0, h1, i)
                    lines.append(f"           <- {'no ' + what if hw else 'guard NOT logging'} {ts(h0)} -> "
                                 f"{ts(h1)} ({h1 - h0:.0f} s{'; ' + hw if hw else ''}): {figs} only the logged {cv}")
                if len(hs) > 5:
                    lines.append(f"           <- ... and {len(hs) - 5} more hole(s) longer than {GUARD_HOLE_S} s")
            print(label + gpfx + msg + "".join("\n" + x for x in lines))

    if gpath is not None:
        print(f"\npower guard log: {gpath}" + ("" if g is not None else "  -- NOT FOUND, nothing matched"))
    if g is not None:
        if sx:
            print(f"  dated span {ts(g_first[0])} -> {ts(g_last[0])}, {len(sx)} samples")
        print("  sample lines carry the time of day only; any clock step in the guard session covering a case withholds"
              " that case's guard figures; a whole-day step of at most GUARD_STALL_S, or a clock that was already"
              " wrong when the guard started, cannot be detected -- the procedure syncs the clock before the guard"
              " starts.")
        print(f"  'guard start' lines: {sum(s['t'] is not None for s in g['sessions'])}")
        for s in g["sessions"]:
            if s["t"] is not None or s["trips"]:
                print(f"    {s['text']}" + (f"  <- {s['bad'].split(' -- ')[0]}: none of its lines is used"
                                            if s["bad"] and s["t"] is not None else ""))
            if s["trips"]:            # one guard process writes one TRIP at most
                t0, ln = s["trips"][0]
                print(f"      tripped {tripped(t0, ln)}{f' (log line {ln})' if t0 is not None else ''}: it acts once"
                      " per start, so it could not trip again until the next 'guard start'"
                      + (f" -- yet {len(s['trips'])} TRIP lines follow this start (two guards writing one log?)"
                         if len(s["trips"]) > 1 else ""))
        for why_u, k in g["unused"].items():
            print(f"  {k} sample line(s) {why_u}: not used for any case")
        if g["vblank"] and g["vblank"] == g["pblank"]:
            print(f"  {len(g['vblank'])} guard line(s) with unreadable INA3221 values ('V= I= P='), first on log "
                  f"line {g['vblank'][0]}: the guard was running but blind -- an empty V resets its low count")
        else:                         # the V and the P read fail on their own: V blinds the guard, P only its peak
            if g["vblank"]:
                print(f"  {len(g['vblank'])} guard line(s) with an unreadable V ('V='), first on log line "
                      f"{g['vblank'][0]}: the guard was running but blind -- an empty V resets its low count")
            if g["pblank"]:
                print(f"  {len(g['pblank'])} guard line(s) with an unreadable P ('P='), first on log line "
                      f"{g['pblank'][0]}: missing from the guard peak")
        for wmsg in g["warnings"]:
            print(f"  WARNING {wmsg}")
        if g["other"]:
            print(f"  {len(g['other'])} other line(s), first: " + " | ".join(safe(l) for _, l in g["other"][:3]))
        print(f"  trips in whole log: {len(g['trips'])}"
              + "".join(f"\n    {'' if t is not None else '(undated) '}{l}" for t, l, _, _ in g["trips"]))
        for r in (r for r in g["lowruns"] if r["notrip"]):     # the guard's own rule, wherever the windows are
            print(f"  input below the guard's {r['thr']} mV threshold for {r['n']} samples in a row from log line "
                  f"{r['la']}, dated {ts(r['a'])} -> {ts(r['b'])} (min {r['vmin']} mV), and no TRIP: "
                  + ("its 'guard start' had already tripped" if r["spent"] else
                     "the guard was armed, so its TRIP line is missing from the log"))
    print("\nstate log:")
    for s in states:
        print(f"  {s}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
