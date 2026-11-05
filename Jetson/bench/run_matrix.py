#!/usr/bin/env python3
"""
Drive llama-bench across the sweep in bench/matrix.json with power logging
wrapped around every case, and write one tidy CSV that has throughput and
energy on the same row.

What it does per case
  1. pre-reads the model so a cold SD-card read is not charged to one case
  2. writes a phase label to the marker file (the logger stamps it on samples)
  3. runs llama-bench with -o csv for that case, stdout and stderr kept apart
  4. writes an idle label and waits `cooldown_s` so the next case does not
     inherit this one's heat
  5. after the sweep, joins each case's power window to its bench rows, and
     trims the window to the trailing compute_s seconds llama-bench reports via
     avg_ns so model load and CUDA init are not billed to the tokens

Every row carries `energy_basis` saying which window the per-token columns were
built from, plus raw `joules`, `load_joules` and `load_s` alongside.

Why a driver instead of a shell loop: the join between "this throughput number"
and "these watts" has to be exact, and that means one process owning both the
marker timeline and the subprocess boundaries.

Usage
  ./run_matrix.py                          # whole matrix, config from node.env
  ./run_matrix.py --only gpu_decode_sweep
  ./run_matrix.py --dry-run
  ./run_matrix.py --tag wired-gbe          # label the condition in the output
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
JETSON_DIR = os.path.dirname(HERE)


DEFAULT_ENV_FILE = os.path.join(JETSON_DIR, "config", "node.env")


def load_env(envfile=None):
    """Read a node.env by sourcing it in a shell -- it is bash, and it uses
    ${VAR} expansion, so parsing it by hand would be wrong.

    The path is a parameter because config/node.env is LIVE MACHINE CONFIG:
    it is gitignored, it is the only record of this node's paths and its
    llama.cpp pin, and overwriting it is silent and unrecoverable. Tests and
    fixtures pass --env-file pointing at a tmpdir instead of writing the real
    one. Nothing in this kit should ever write config/node.env.
    """
    envfile = envfile or DEFAULT_ENV_FILE
    if not os.path.exists(envfile):
        sys.exit("missing %s -- cp config/node.env.example config/node.env" % envfile)
    # Pass the path as $1 rather than splicing it into the snippet, so a path
    # with spaces cannot break the sourcing.
    out = subprocess.check_output(
        ["bash", "-c", 'set -a; . "$1"; env', "bash", envfile],
        universal_newlines=True, encoding="utf-8", errors="replace")
    env = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            env[k] = v
    return env


def as_list(v):
    return v if isinstance(v, list) else [v]


def find_bin(bindir, *names):
    for n in names:
        p = os.path.join(bindir, n)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def model_path(env, short_name):
    manifest = os.path.join(JETSON_DIR, "config", "models.tsv")
    with open(manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] == short_name:
                return os.path.join(env["MODEL_DIR"], parts[1])
    return None


def expand(case, defaults):
    """Cartesian product of the list-valued axes of one case.

    Refuses a case that sets both n_prompt and n_gen. That combination is ONE
    llama-bench invocation running TWO tests inside ONE power window, so its row
    would pair one phase's throughput with both phases' energy -- audit finding
    #3. The refusal is here, rather than at join time, so it fires during
    --dry-run instead of two hours into a sweep. For a deliberately fused
    request use the "pg" key, which emits -pg: a SINGLE test instance, one row.
    """
    merged = dict(defaults)
    merged.update({k: v for k, v in case.items() if not k.startswith("_")})
    models = as_list(merged.get("model"))
    ngls = as_list(merged.get("n_gpu_layers", 0))
    threads = as_list(merged.get("threads", 4))
    prompts = as_list(merged.get("n_prompt", [0]))
    gens = as_list(merged.get("n_gen", [0]))

    pg = merged.get("pg")
    if pg is not None:
        if len(as_list(pg)) != 2 or not all(as_list(pg)):
            sys.exit("case %r: 'pg' must be [n_prompt, n_gen], both non-zero"
                     % merged["id"])
        if any(prompts) or any(gens):
            sys.exit("case %r: 'pg' replaces n_prompt/n_gen -- do not set both"
                     % merged["id"])
        prompts, gens = [0], [0]        # -pg carries the counts; -p 0 -n 0 silences

    mixed = sorted({(p, g) for p in prompts for g in gens if p and g})
    if mixed:
        sys.exit(
            "case %r sets both n_prompt and n_gen %s.\n"
            "  That is one llama-bench invocation running two tests in one power\n"
            "  window, so its row would pair one phase's throughput with both\n"
            "  phases' energy (audit finding #3). Split it into a prefill case\n"
            "  (n_gen: [0]) and a decode case (n_prompt: [0]), giving both the\n"
            "  same 'group' so their runs interleave; or, for a deliberately\n"
            "  fused request, use \"pg\": [n_prompt, n_gen]."
            % (merged["id"], mixed))

    runs = []
    for m in models:
        for ngl in ngls:
            for th in threads:
                for p in prompts:
                    for g in gens:
                        if p == 0 and g == 0 and pg is None:
                            continue  # llama-bench needs at least one of the two
                        runs.append({
                            "case_id": merged["id"],
                            "group": merged.get("group", ""),
                            "note": merged.get("note", ""),
                            "model": m,
                            "n_gpu_layers": ngl,
                            "threads": th,
                            "n_prompt": p,
                            "n_gen": g,
                            "pg_n_prompt": as_list(pg)[0] if pg is not None else 0,
                            "pg_n_gen": as_list(pg)[1] if pg is not None else 0,
                            "repetitions": merged.get("repetitions", 3),
                            "flash_attn": merged.get("flash_attn", 0),
                            "cooldown_s": merged.get("cooldown_s", 20),
                            "cooldown_target_c": merged.get("cooldown_target_c", 3),
                            "max_cooldown_s": merged.get("max_cooldown_s", 300),
                        })
    return runs


def interleave_groups(runs):
    """Reorder so the halves of a split case alternate: prefill, decode,
    prefill, decode ... rather than every prefill followed by every decode.

    A fanless Nano heats monotonically across a 1.5-3 h sweep, so position in the
    run order is confounded with temperature. Cases that share a "group" are the
    two phases of one configuration and get compared directly to each other, so
    they are the pairs that must not sit hours apart. Runs keep their relative
    order within each case; a group's block occupies the positions its runs
    already held, so ungrouped cases do not move at all.
    """
    by_group = {}
    for i, r in enumerate(runs):
        by_group.setdefault(r.get("group") or "", []).append(i)

    out = list(runs)
    for group, idxs in by_group.items():
        if not group or len(idxs) < 2:
            continue
        per_case = []
        seen = {}
        for i in idxs:
            cid = runs[i]["case_id"]
            if cid not in seen:
                seen[cid] = len(per_case)
                per_case.append([])
            per_case[seen[cid]].append(runs[i])
        if len(per_case) < 2:
            continue
        woven = []
        for k in range(max(len(c) for c in per_case)):
            for c in per_case:
                if k < len(c):
                    woven.append(c[k])
        for slot, run in zip(idxs, woven):
            out[slot] = run
    return out


def phase_of(run):
    """Label the window by what it actually stresses. A prompt-only run is
    prefill, a generation-only run is decode, a -pg run is combined. "mixed"
    means a -p N -n M invocation, which expand() refuses -- it survives only so
    that a row read back from an older, pre-fix results CSV is recognisable as
    one, rather than being silently pooled with the split rows."""
    if run.get("pg_n_prompt") and run.get("pg_n_gen"):
        return "combined"
    if run["n_prompt"] and not run["n_gen"]:
        return "prefill"
    if run["n_gen"] and not run["n_prompt"]:
        return "decode"
    return "mixed"


def phase_of_counts(n_prompt, n_gen):
    """phase_of() for a bench CSV row's OWN counts.

    At row level, both counts non-zero means a single fused instance: a mixed
    -p N -n M invocation does not produce such a row, it produces two rows
    carrying one count each. So "both" here is -pg, i.e. combined -- never
    "mixed", which is a property of an invocation rather than of a row.
    """
    if (n_prompt or 0) and (n_gen or 0):
        return "combined"
    if n_prompt:
        return "prefill"
    if n_gen:
        return "decode"
    return "none"


def run_label(run):
    if phase_of(run) == "combined":
        return "%s|%s|ngl%s|t%s|pg%s,%s" % (
            run["case_id"], run["model"], run["n_gpu_layers"],
            run["threads"], run["pg_n_prompt"], run["pg_n_gen"])
    return "%s|%s|ngl%s|t%s|p%s|g%s" % (
        run["case_id"], run["model"], run["n_gpu_layers"],
        run["threads"], run["n_prompt"], run["n_gen"])


def build_cmd(bench_bin, run, mpath):
    cmd = [bench_bin, "-m", mpath, "-o", "csv",
           "-r", str(run["repetitions"]),
           "-t", str(run["threads"]),
           "-ngl", str(run["n_gpu_layers"])]
    # llama-bench rejects -p 0 / -n 0 on some refs; only pass the axis in play.
    if run["n_prompt"]:
        cmd += ["-p", str(run["n_prompt"])]
    else:
        cmd += ["-p", "0"]
    if run["n_gen"]:
        cmd += ["-n", str(run["n_gen"])]
    else:
        cmd += ["-n", "0"]
    # -pg builds ONE fused test instance emitting ONE CSV row. The -p 0 -n 0
    # above is what stops llama-bench also running its default standalone pp and
    # tg tests, which would put three rows in this window.
    if run.get("pg_n_prompt") and run.get("pg_n_gen"):
        cmd += ["-pg", "%s,%s" % (run["pg_n_prompt"], run["pg_n_gen"])]
    if run["flash_attn"]:
        cmd += ["-fa", "1"]
    return cmd


# A llama-bench CSV header always carries these; a CUDA banner or a log line
# never does. Used to find where the real CSV starts in a captured stream.
BENCH_HEADER_HINTS = ("avg_ts", "build_commit")


def bench_csv_text(stdout):
    """The llama-bench CSV out of a captured stdout, preamble stripped.

    Returns "" when no header line is found, so a polluted or empty capture
    shows up as an empty file instead of as one whose first line is a log
    message that csv.DictReader would happily accept as the header.
    """
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if any(h in line for h in BENCH_HEADER_HINTS):
            return "\n".join(lines[i:]).rstrip("\n") + "\n"
    return ""


def row_tps(row):
    """avg_ts off ONE bench row. Column names have shifted between refs.

    Deliberately per-row: the function this replaced looped over every row
    without breaking and kept the LAST one, which for a two-test invocation is
    always the decode row -- llama-bench orders its instances pp before tg. That
    was half of audit finding #3.
    """
    for k in ("avg_ts", "t/s", "tokens_per_second"):
        if row.get(k):
            try:
                return float(row[k])
            except ValueError:
                pass
    return None


def bench_int(row, key):
    """An integer column off a bench row, or None if absent/unparseable. None
    and 0 mean different things here: absent vs "this test does no such work"."""
    v = row.get(key)
    if v in ("", None):
        return None
    try:
        return int(float(v))
    except ValueError:
        return None


def test_label(n_prompt, n_gen):
    """llama-bench's own naming, for a ref that does not emit a "test" column."""
    if n_prompt and n_gen:
        return "pp%d+tg%d" % (n_prompt, n_gen)
    if n_prompt:
        return "pp%d" % n_prompt
    if n_gen:
        return "tg%d" % n_gen
    return "-"


def row_compute_seconds(row, reps):
    """Seconds ONE bench row spent inside llama-bench's timed loops.

    avg_ns is the mean nanoseconds of ONE repetition of that test. The window
    also holds an unmeasured warm-up before the repetition loop: a full
    n_prompt-token pass for a pp test, one generated token for a tg test, and
    for a fused -pg instance both. Twin of bench_compute_s() in
    telemetry/energy_window.py -- keep the two in step.
    """
    try:
        ns = float(row.get("avg_ns") or "")
    except (TypeError, ValueError):
        return None
    if not ns:
        return None
    n_p = bench_int(row, "n_prompt") or 0
    n_g = bench_int(row, "n_gen") or 0
    if n_p and n_g:
        # A fused row's avg_ns covers prompt+generate together, and the warm-up
        # is a prompt pass plus ONE token -- so it is somewhere between 0 and ns,
        # and avg_ns alone cannot say where. ns is the upper bound, which makes
        # compute_s long and load_s correspondingly short. It only affects those
        # two reported figures: a fused row's per-request energy is measured over
        # request_window_s(), which needs no warm-up estimate.
        warm_ns = ns
    elif n_p:
        warm_ns = ns
    elif n_g:
        warm_ns = ns / n_g
    else:
        warm_ns = 0.0
    return round((ns * reps + warm_ns) / 1e9, 3)


def compute_seconds(rows, reps):
    """Timed-loop seconds over EVERY row in one window, i.e. the length of the
    trailing slice of the marker window that processed tokens. Model load, CUDA
    context creation and the weight upload all precede the first timed loop.

    For a single-row window -- which after the finding-#3 split is every window
    -- this equals row_compute_seconds() of that row and the trailing slice is
    exactly that test. For a multi-row window the sum is still the right total,
    but the slice is NOT aligned per test: llama-bench builds and frees a
    context between instances, so unmeasured multi-watt work sits inside the
    trimmed slice. That is why a multi-row window is refused a per-token number
    rather than being split by time.
    """
    total = 0.0
    seen = False
    for r in rows:
        cs = row_compute_seconds(r, reps)
        if cs is None:
            continue
        total += cs
        seen = True
    return round(total, 3) if seen else None


# Why a per-token number is or is not on a row. Blank numeric cells already mean
# at least five different things in this CSV -- refused, run failed, no CSV, no
# avg_ns, no power log -- and the ones that matter for defensibility are
# indistinguishable from each other. Every row carries one of these.
PT_OK = "ok"
PT_MULTI_PHASE = "refused_multi_phase"
PT_COMBINED = "refused_combined_request"
PT_RUN_FAILED = "run_failed"
PT_NO_BENCH_CSV = "no_bench_csv"
PT_NO_POWER = "no_power_window"
PT_NO_TOKENS = "no_token_count"
PT_PHASE_MISMATCH = "refused_phase_mismatch"
PT_MODEL_MISSING = "model_missing"
PT_COUNTS_MISMATCH = "refused_counts_mismatch"
PT_COUNTS_UNVERIFIED = "refused_counts_unverified"
PT_CONFIG_MISMATCH = "refused_config_mismatch"


def phase_rows(bench_rows):
    """The rows in a window that did real work. build_cmd() always passes both
    -p and -n, using 0 for the idle axis, so a ref that emits a row for the
    zero-valued test would make EVERY window look multi-row and blank every
    per-token column in the study. Counting only rows with non-zero work is what
    keeps the guard a tripwire instead of a sweep-wide outage."""
    out = []
    for r in bench_rows:
        if (bench_int(r, "n_prompt") or 0) or (bench_int(r, "n_gen") or 0):
            out.append(r)
    return out


def requested_counts(run):
    """What this invocation asked llama-bench to run, as (n_prompt, n_gen)."""
    if run.get("pg_n_prompt") and run.get("pg_n_gen"):
        return run["pg_n_prompt"], run["pg_n_gen"]
    return run.get("n_prompt") or 0, run.get("n_gen") or 0


def config_mismatch(run, row):
    """True when the bench row says it ran a different configuration than the one
    the driver asked for. Only fields the ref actually reports are compared, so a
    ref that omits them is not penalised here -- that is what counts_source and
    PT_COUNTS_UNVERIFIED are for."""
    ngl = bench_int(row, "n_gpu_layers")
    if ngl is not None and ngl != run.get("n_gpu_layers"):
        return True
    name = (row.get("model_filename") or "").strip()
    if name and run.get("model_file") and os.path.basename(name) != run["model_file"]:
        return True
    return False


def per_token_status(rec):
    """Whether this row may carry a per-token energy figure, and if not, why."""
    if rec.get("status") == "failed":
        return PT_RUN_FAILED
    if rec.get("status") == "skipped":
        return PT_MODEL_MISSING
    if not rec.get("bench_rows"):
        return PT_NO_BENCH_CSV
    # Checked BEFORE the combined case on purpose. A fused row that shares its
    # window with other tests is not a clean per-request measurement either --
    # the window's joules cover work this row does not describe -- so it must be
    # refused outright rather than re-denominated.
    #
    # The second clause matters as much as the first: phase_rows() identifies a
    # working row by its reported n_prompt/n_gen, so a ref that omits, blanks or
    # renames those columns yields working_rows == 0 for the whole window, and
    # "> 1" would never fire. A multi-row window is only safe when EXACTLY ONE
    # row did identifiable work; zero means the driver cannot tell which test the
    # tokens describe, which is the same problem wearing a different hat.
    if (rec.get("working_rows") or 0) > 1 or (
            (rec.get("bench_rows") or 0) > 1 and (rec.get("working_rows") or 0) != 1):
        return PT_MULTI_PHASE

    # The driver knows what it asked llama-bench to run. If the row that came
    # back is a different phase, the ref did not run that instance -- and then
    # neither denominator is trustworthy, because the window's joules cover work
    # the row's counts do not describe. Seen for real: a ref that honours -pg but
    # reports the fused row's prompt count in a column this driver does not read,
    # leaving n_prompt=0. That row scores as decode and, without this check,
    # publishes J/token over a decode-only divisor against prefill+decode energy
    # -- audit finding #3 coming back through the ref instead of the matrix.
    if rec.get("phase") and rec.get("row_phase") and rec["phase"] != rec["row_phase"]:
        return PT_PHASE_MISMATCH

    # The phase check compares LABELS. It cannot see a ref that answers -p 512
    # with a 128-token prompt: same phase, right label, and a divisor four times
    # too small against the energy of the 512 that actually ran -- the same
    # magnitude of error as the original finding #3, and it passed every guard.
    if rec.get("counts_mismatch"):
        return PT_COUNTS_MISMATCH

    # The phase and count checks cover WHAT ran. This covers WHERE: a ref that
    # ignores -ngl -- a CPU-only build, or a clamp to the model's layer count --
    # would otherwise publish CPU numbers under ngl99, and the whole GPU-offload
    # result is a comparison between those two settings.
    if rec.get("config_mismatch"):
        return PT_CONFIG_MISMATCH

    # And if the ref reported no counts at all, row_phase was derived from the
    # request itself, so the phase check above compared the request with itself
    # and could never fail. An unverifiable divisor is not a divisor.
    if rec.get("counts_source") == "matrix":
        return PT_COUNTS_UNVERIFIED

    if rec.get("row_phase") == "combined":
        # A fused request is one instance whose joules cannot be attributed to
        # a phase. It gets joules_per_request instead -- not a defect, a
        # different denominator.
        return PT_COMBINED
    if not rec.get("tokens"):
        return PT_NO_TOKENS
    return PT_OK


def bench_records(run, bench_rows, base):
    """One output record per llama-bench row -- or exactly ONE record when the
    invocation produced none, so a parse failure still appears in the joined CSV
    instead of vanishing from it.

    Sibling rows share `marker` (unique per invocation) and carry bench_rows,
    bench_row_index and window_shared, so they read as ONE window seen twice
    rather than as two independent measurements.
    """
    reps = run["repetitions"]
    working = phase_rows(bench_rows)
    req_p, req_g = requested_counts(run)

    if not bench_rows:
        rec = dict(base)
        rec.update({"bench_rows": 0, "working_rows": 0, "bench_row_index": 0,
                    "window_shared": 0, "test": "-", "row_phase": rec.get("phase"),
                    "avg_ts": None, "tokens": None})
        rec["per_token_status"] = per_token_status(rec)
        return [rec]

    out = []
    for i, r in enumerate(bench_rows):
        n_p, n_g = bench_int(r, "n_prompt"), bench_int(r, "n_gen")
        eff_p = n_p if n_p is not None else run["n_prompt"]
        eff_g = n_g if n_g is not None else run["n_gen"]
        rec = dict(base)
        rec.update({
            "bench_rows": len(bench_rows),
            "working_rows": len(working),
            "bench_row_index": i,
            "window_shared": int(len(bench_rows) > 1),
            "test": r.get("test") or test_label(eff_p, eff_g),
            "row_phase": phase_of_counts(eff_p, eff_g),
            "bench_n_prompt": eff_p,
            "bench_n_gen": eff_g,
            "counts_source": "bench" if (n_p is not None or n_g is not None) else "matrix",
            "req_n_prompt": req_p,
            "req_n_gen": req_g,
            "counts_mismatch": int(
                (n_p is not None or n_g is not None)
                and len(bench_rows) == 1
                and (eff_p, eff_g) != (req_p, req_g)),
            "bench_n_gpu_layers": bench_int(r, "n_gpu_layers"),
            "bench_n_threads": bench_int(r, "n_threads"),
            "bench_model_filename": r.get("model_filename") or "",
            "config_mismatch": int(config_mismatch(run, r)),
            "avg_ts": row_tps(r),
            "avg_ns": r.get("avg_ns") or "",
            "stddev_ts": r.get("stddev_ts") or "",
            "stddev_ns": r.get("stddev_ns") or "",
            # Carried verbatim off the bench row so bench/emit_results_csv.py can
            # build the cluster's shared schema without re-opening the per-case
            # CSVs -- and so the pinned llama.cpp revision travels with every
            # number, spelled the way llama-bench spells it.
            "build_commit": r.get("build_commit") or "",
            "bench_model_type": r.get("model_type") or "",
            "bench_model_size": r.get("model_size") or "",
            "bench_n_params": r.get("model_n_params") or "",
            "bench_n_batch": r.get("n_batch") or "",
            "row_compute_s": row_compute_seconds(r, reps),
        })
        rec["tokens"] = tokens_of(rec)
        rec["requests"] = requests_of(rec)
        rec["request_window_s"] = request_window_s(rec)
        rec["per_token_status"] = per_token_status(rec)
        out.append(rec)
    return out


def _tegrastats_already_running():
    """True when some other process is already reading the INA3221."""
    try:
        out = subprocess.check_output(["pgrep", "-x", "tegrastats"],
                                      stderr=subprocess.DEVNULL,
                                      universal_newlines=True,
                                      encoding="utf-8", errors="replace")
        return bool(out.strip())
    except (subprocess.CalledProcessError, OSError):
        return False


def _power_progress(power_csv):
    """(sample count, achieved Hz) for the power CSV written so far."""
    try:
        rows = _rows(power_csv)
    except IOError:
        return 0, 0.0
    ts = [t for t in (_num(r, "ts_unix") for r in rows) if t is not None]
    if len(ts) < 2:
        return len(ts), 0.0
    span = ts[-1] - ts[0]
    return len(ts), (len(ts) - 1) / span if span > 0 else 0.0


def _stop_logger(logger):
    """Take the logger down. An abort that leaves a stray tegrastats behind makes
    the NEXT run fail the already-running check for a reason it did not cause."""
    if logger is None or logger.poll() is not None:
        return
    logger.terminate()
    try:
        logger.wait(timeout=10)
    except Exception:
        logger.kill()


def tail_sample(power_csv, nbytes=16384):
    """The last parsed row of the power CSV, reading only its tail.

    The cooldown gate polls this every couple of seconds for up to three hours.
    Re-parsing a 100k-row file each time would put the driver's own CPU load
    inside the cooldown it is trying to measure, which is self-defeating -- and
    it would get worse as TELEMETRY_HZ rises.
    """
    try:
        with open(power_csv, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - nbytes))
            tail = fh.read().decode("utf-8", "replace")
        with open(power_csv, "r", encoding="utf-8") as fh:
            header = fh.readline().strip()
    except (IOError, OSError):
        return None
    lines = [l for l in tail.splitlines() if l.strip()]
    if not lines or not header:
        return None
    # the first line of the tail is probably a fragment; drop it unless we read
    # the whole file
    if size > nbytes and len(lines) > 1:
        lines = lines[1:]
    rows = list(csv.DictReader(lines, fieldnames=header.split(",")))
    return rows[-1] if rows else None


def cooldown(run, power_csv, baseline_c, have_power):
    """Wait out the fixed floor, then wait for the board to come back down.

    `cooldown_s` used to be the whole story: an open-loop 20 s sleep. A fanless
    Nano dissipating ~10 W has a heatsink time constant of MINUTES, so 20 s
    recovers almost nothing and the board ratchets upward across a 2-3 h sweep --
    which makes position in the run order a hidden variable, confounded with
    every axis the sweep varies. So `cooldown_s` is now the FLOOR, and after it
    the driver waits until the board is within `cooldown_target_c` of the idle
    baseline, capped by `max_cooldown_s`.

    The gate reads `temp_thermal_c` from the power CSV -- the same column every
    row is scored on -- so the gate and the audit column cannot disagree.

    Returns (waited_s, reached) where reached is None when no gate ran.
    """
    floor = run.get("cooldown_s") or 0
    t0 = time.time()
    time.sleep(floor)
    target = run.get("cooldown_target_c")
    cap = run.get("max_cooldown_s") or 300
    if not have_power or baseline_c is None or not target:
        # --no-power, or a logger that wrote no temperature: stay open loop and
        # report NO gate, rather than letting a row claim one that did not run.
        return round(time.time() - t0, 1), None
    while time.time() - t0 < cap:
        row = tail_sample(power_csv)
        t = _num(row, "temp_thermal_c") if row else None
        if t is None:
            return round(time.time() - t0, 1), None
        if t - baseline_c <= target:
            return round(time.time() - t0, 1), 1
        time.sleep(2)
    return round(time.time() - t0, 1), 0


def precache(path):
    """Read the GGUF once before the window opens, so the first case to touch a
    model does not pay a cold SD-card read that later cases of the same model do
    not. On a 4 GB board the page cache can still evict it -- this makes the
    load cost more uniform across the axis being compared, it does not remove
    it. That is what compute_joules is for."""
    try:
        with open(path, "rb") as fh:
            while fh.read(8 << 20):
                pass
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", default=os.path.join(HERE, "matrix.json"))
    ap.add_argument("--only", action="append", default=[],
                    help="run only these case ids (repeatable)")
    ap.add_argument("--tag", default="", help="condition label, e.g. wired-gbe / wifi / 5W")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--no-power", action="store_true",
                    help="skip tegrastats logging (off-Jetson dry runs)")
    ap.add_argument("--baseline-s", type=float, default=30.0,
                    help="seconds of idle to record before the first case (default 30). "
                         "This interval is ALSO the idle floor the first case is "
                         "corrected against, so shortening it degrades that row -- it is "
                         "here for off-hardware tests, not for real sweeps.")
    ap.add_argument("--no-drift-control", action="store_true",
                    help="skip the repeat of the first case as the last one. That "
                         "pair is the only direct measurement of how far the board "
                         "drifted over the sweep; without it, run order stays a "
                         "hidden variable.")
    ap.add_argument("--no-precache", action="store_true",
                    help="do not pre-read each model before its window; leaves "
                         "cold-vs-warm SD reads confounded with the swept axis")
    ap.add_argument("--env-file", default=None,
                    help="node.env to source (default config/node.env). Point this "
                         "at a tmpdir for tests -- never write the real one, it is "
                         "gitignored machine config and the clobber is silent.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    env = load_env(args.env_file)
    node_id = env.get("NODE_ID", "unknown")
    bindir = env["LLAMA_BIN"]
    out_dir = args.out_dir or env.get("RESULTS_DIR") or os.path.join(JETSON_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)

    bench_bin = find_bin(bindir, "llama-bench", "bench")
    if not bench_bin and not args.dry_run:
        sys.exit("llama-bench not found in %s -- run scripts/20_build_llama_cpp.sh" % bindir)

    with open(args.matrix, encoding="utf-8") as fh:
        matrix = json.load(fh)
    defaults = matrix.get("defaults", {})

    runs = []
    for case in matrix["cases"]:
        if not case.get("enabled", True):
            continue
        if args.only and case["id"] not in args.only:
            continue
        runs.extend(expand(case, defaults))
    if not runs:
        sys.exit("no cases selected")
    runs = interleave_groups(runs)

    # Run 1 again, unchanged, as the final run. Same case, same configuration,
    # hours apart -- so the difference between those two rows is thermal drift
    # and nothing else, measured rather than argued about. This is the audit's
    # own "at minimum" suggestion; its other option, shuffling with a seed, was
    # deliberately not taken: randomising fights interleave_groups(), which
    # depends on a chosen order, and a sweep you cannot re-run in sequence is
    # harder to defend than one whose drift is published.
    # Always set the column, even when the control is skipped. A column that
    # exists only when a flag is on is a column a consumer cannot rely on -- and
    # `--no-drift-control` made it vanish entirely rather than read 0, which is
    # the same sometimes-absent defect this kit has been chasing elsewhere.
    runs = [dict(r, drift_control=0) for r in runs]
    if not args.no_drift_control and len(runs) > 1:
        runs.append(dict(runs[0], drift_control=1))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = "%s_%s%s" % (node_id, stamp, ("_" + args.tag) if args.tag else "")
    power_csv = os.path.join(out_dir, base + "_power.csv")
    joined_csv = os.path.join(out_dir, base + "_bench_energy.csv")
    marker_file = os.path.join(out_dir, ".marker_" + node_id)
    log_path = os.path.join(out_dir, base + "_run.log")

    print("node      : %s" % node_id)
    print("tag       : %s" % (args.tag or "-"))
    print("runs      : %d" % len(runs))
    phases = {}
    for r in runs:
        phases[phase_of(r)] = phases.get(phase_of(r), 0) + 1
    print("phases    : %s" % ", ".join("%s %d" % kv for kv in sorted(phases.items())))
    print("power csv : %s" % power_csv)
    print("output    : %s" % joined_csv)
    print()

    if args.dry_run:
        for i, r in enumerate(runs, 1):
            mp = model_path(env, r["model"])
            print("  %3d %-9s %-64s %s"
                  % (i, phase_of(r), run_label(r),
                     "OK" if mp and os.path.exists(mp) else "MODEL MISSING"))
        return

    def set_marker(text):
        with open(marker_file, "w", encoding="utf-8") as fh:
            fh.write(text)

    set_marker("idle")

    temp_baseline_c = None
    logger = None
    if not args.no_power and shutil.which("tegrastats"):
        # A second tegrastats makes the first one's output stop, so OUR logger
        # dies at startup and every energy column comes back empty -- after the
        # sweep, not before it. Checked here, where it costs nothing.
        if _tegrastats_already_running():
            sys.exit("a tegrastats process is already running.\n"
                     "  Two instances cannot both read the INA3221: the second one "
                     "silences the first,\n  so this sweep would produce a complete-"
                     "looking CSV with no energy in it.\n"
                     "  Stop it first:  sudo systemctl stop heteroserve-telemetry   "
                     "(or pkill tegrastats)")
        logger = subprocess.Popen(
            [sys.executable, os.path.join(JETSON_DIR, "telemetry", "tegrastats_logger.py"),
             "--out", power_csv, "--hz", str(env.get("TELEMETRY_HZ", "5")),
             "--node-id", node_id, "--marker-file", marker_file])
        # Let the logger establish an idle baseline before the first case. Since
        # the idle floor became per-case (each case is corrected against the
        # cooldown that preceded it), this interval is not just a warm-up: it IS
        # the floor for case 1, so its length is a measurement decision.
        if args.baseline_s < IDLE_SETTLE_S + 2.0:
            print("!! --baseline-s %.1f is shorter than the %.1f s settle window plus "
                  "margin;\n   the first case's idle floor will be read off an "
                  "unsettled interval." % (args.baseline_s, IDLE_SETTLE_S))
        print("collecting %.0f s idle baseline..." % args.baseline_s)
        time.sleep(args.baseline_s)

        # The baseline is also the proof that logging works. Checking it here
        # turns a silent two-hour sweep with empty energy columns into a ten-line
        # failure before any measurement is taken.
        rc = logger.poll()
        n, hz = _power_progress(power_csv)
        if rc is not None or n < 2:
            _stop_logger(logger)
            sys.exit(
                "the power logger produced %d sample(s) in %.0f s%s.\n"
                "  Likely causes: another tegrastats instance (this sweep checked, "
                "but one can start\n  in between), tegrastats not permitted for this "
                "user, or a logger crash --\n  see the traceback rows in %s.\n"
                "  Aborting rather than running the sweep with no energy data."
                % (n, args.baseline_s,
                   "" if rc is None else " and exited rc=%s" % rc,
                   os.path.basename(power_csv)))
        # The ACHIEVED rate, not the requested one: it is what bounds every
        # integral downstream, and the two differ on a loaded board.
        print("power logging: %d samples, %.1f Hz achieved (requested %s Hz)"
              % (n, hz, env.get("TELEMETRY_HZ", "5")))
        # The board's cold temperature, measured once from the baseline window.
        # Every later case is scored against THIS, because the quantity that
        # matters is how far the board has drifted from cold -- not how much it
        # rose inside one window, which goes to ~0 exactly when the board has
        # reached a hot steady state.
        temps0 = [t for t in (_num(r, "temp_thermal_c") for r in _rows(power_csv))
                  if t is not None]
        if temps0:
            temp_baseline_c = round(sum(temps0) / len(temps0), 2)
            print("thermal baseline: %.1f C (cases cool back to within %s C of it)"
                  % (temp_baseline_c, defaults.get("cooldown_target_c", 3)))
    else:
        print("!! no power logging (tegrastats missing or --no-power)")
        print("   every row will carry per_token_status=no_power_window")

    records = []
    logfh = open(log_path, "w", encoding="utf-8")
    try:
        for i, run in enumerate(runs, 1):
            mpath = model_path(env, run["model"])
            if not mpath or not os.path.exists(mpath):
                # Still emit a record. The joined CSV is the archived artefact;
                # stdout is not. A run that silently vanishes from it is
                # indistinguishable from a run that was never in the matrix.
                print("[%d/%d] SKIP %s (model missing)" % (i, len(runs), run_label(run)))
                now = time.time()
                skipped = {**run, "marker": "(skipped):%s:%03d" % (run["case_id"], i),
                           "phase": phase_of(run), "row_phase": phase_of(run),
                           "run_index": i,
                           "status": "skipped", "rc": None,
                           # A zero-length window: the run never happened, but the
                           # record still needs a position in time, because the
                           # join indexes these unconditionally.
                           "t_start_unix": now, "t_end_unix": now,
                           "bench_rows": 0, "working_rows": 0, "window_shared": 0,
                           "node_id": node_id, "node_role": env.get("NODE_ROLE", ""),
                      "cluster_node_id": env.get("CLUSTER_NODE_ID", ""),
                           "model_path": mpath or "", "tag": args.tag}
                logfh.write("\n=== %s\n!! SKIPPED: model not found at %s\n"
                            % (run_label(run), mpath or "(no manifest entry)"))
                logfh.flush()
                skipped["per_token_status"] = PT_MODEL_MISSING
                records.append(skipped)
                continue

            case_csv = os.path.join(out_dir, "%s_%03d.csv" % (base, i))
            cmd = build_cmd(bench_bin, run, mpath)
            phase = phase_of(run)
            label = "%s:%s:%03d" % (phase, run["case_id"], i)

            print("[%d/%d] %s" % (i, len(runs), run_label(run)))
            logfh.write("\n=== %s\n$ %s\n" % (run_label(run), " ".join(cmd)))
            logfh.flush()

            # Warm the page cache before the window opens, so a cold SD read is
            # not charged to whichever case happens to touch the model first.
            #
            # Under its OWN marker, not the idle one. Reading 0.4-1.9 GB off an SD
            # card draws several watts for tens of seconds, and it used to happen
            # while the marker still said "idle" -- so the idle floor, which is
            # subtracted from every active-energy figure, was measured partly over
            # a multi-watt file read. "idle" has to mean idle.
            if not args.no_precache:
                set_marker("precache:%s:%03d" % (run["case_id"], i))
                cached = precache(mpath)
                set_marker("idle")
            else:
                cached = False

            t_start = time.time()
            set_marker(label)
            try:
                # stdout and stderr must stay SEPARATE. llama-bench writes its
                # CSV to stdout and its log to stderr, and with the CUDA backend
                # the ggml banner lands on stderr before main() quiets the log --
                # merged, that banner becomes the first line of the "CSV" and
                # csv.DictReader reads it as the header, so avg_ts is None on
                # every row and the sweep produces joules with no throughput.
                # encoding pinned: universal_newlines alone decodes with the
                # LOCALE encoding, which on a C-locale 3.6 is ASCII -- and a
                # model name or a CUDA banner with one non-ASCII byte would then
                # kill the sweep mid-run. errors="replace" because a mangled log
                # line must never be fatal.
                out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     universal_newlines=True, encoding="utf-8",
                                     errors="replace", check=False)
            finally:
                set_marker("idle")
            t_end = time.time()

            logfh.write(out.stderr or "")
            logfh.flush()

            if out.returncode != 0:
                # A single OOM on a 4 GB board must not abort a two-hour sweep;
                # record the failure as a row and keep going.
                print("    FAILED rc=%d (see %s)" % (out.returncode, log_path))
                failed = {**run, "marker": label, "phase": phase, "run_index": i,
                          # row_phase too: the shared-schema projection keys off
                          # it alone, and a record carrying only `phase` vanished
                          # from the cross-node CSV with no row and no flag.
                          "row_phase": phase,
                          "status": "failed", "rc": out.returncode,
                          "t_start_unix": t_start, "t_end_unix": t_end,
                          "bench_rows": 0, "working_rows": 0,
                          "node_id": node_id, "node_role": env.get("NODE_ROLE", ""),
                      "cluster_node_id": env.get("CLUSTER_NODE_ID", ""),
                          "precached": int(cached), "tag": args.tag}
                failed["per_token_status"] = per_token_status(failed)
                records.append(failed)
                cd_s, cd_ok = cooldown(run, power_csv, temp_baseline_c, logger is not None)
                failed["cooldown_actual_s"] = cd_s
                failed["cooldown_reached"] = "" if cd_ok is None else cd_ok
                continue

            csv_text = bench_csv_text(out.stdout or "")
            with open(case_csv, "w", encoding="utf-8") as fh:
                fh.write(csv_text)
            bench_rows = list(csv.DictReader(csv_text.splitlines())) if csv_text else []
            if not bench_rows:
                print("    !! no llama-bench CSV header in stdout (see %s)" % log_path)
                logfh.write("\n!! unparseable stdout for %s:\n%s\n"
                            % (run_label(run), out.stdout or ""))

            # One record per bench row. Everything describing the WINDOW --
            # marker, timestamps, compute_s, precached, tag -- is computed once
            # and copied onto each row; everything describing the TEST -- test,
            # avg_ts, avg_ns, tokens -- comes off that row alone.
            # NB: not `base` -- that name is the output-file prefix in this
            # scope, and shadowing it silently renames every later case CSV.
            run["model_file"] = os.path.basename(mpath)
            window = {**run, "marker": label, "phase": phase, "run_index": i,
                      "status": "ok", "rc": 0,
                      "t_start_unix": t_start, "t_end_unix": t_end,
                      "bench_csv": os.path.basename(case_csv),
                      "compute_s": compute_seconds(bench_rows, run["repetitions"]),
                      "node_id": node_id, "node_role": env.get("NODE_ROLE", ""),
                      "cluster_node_id": env.get("CLUSTER_NODE_ID", ""),
                      "precached": int(cached), "tag": args.tag}
            emitted = bench_records(run, bench_rows, window)
            records.extend(emitted)
            for rec in emitted:
                print("    %-10s %8.2f t/s   %s tok   %.1f s"
                      % (rec.get("test") or "-", rec.get("avg_ts") or 0.0,
                         rec.get("tokens") or "-", t_end - t_start))
            if any(r.get("per_token_status") == PT_MULTI_PHASE for r in emitted):
                print("    !! %d working bench rows in ONE power window -- per-token "
                      "energy refused for this case. Split it in matrix.json."
                      % emitted[0].get("working_rows"))
            cd_s, cd_ok = cooldown(run, power_csv, temp_baseline_c, logger is not None)
            for rec in emitted:
                rec["cooldown_actual_s"] = cd_s
                rec["cooldown_reached"] = "" if cd_ok is None else cd_ok
    finally:
        set_marker("idle")
        if logger and logger.poll() is not None:
            print("\n!! the power logger exited during the sweep (rc=%s)." % logger.poll())
            print("   Rows after that point carry no energy; per_token_status marks")
            print("   them no_power_window, but a half-instrumented sweep otherwise")
            print("   reads like a whole one. Check %s." % os.path.basename(log_path))
        if logger:
            time.sleep(3)
            logger.terminate()
            try:
                logger.wait(timeout=10)
            except Exception:
                logger.kill()
        logfh.close()

    # ---- join throughput to energy ----------------------------------------
    idle_mw = 0.0
    if os.path.exists(power_csv):
        idle_pooled = idle_floor(power_csv)
        spans = idle_intervals(power_csv)
        bounds = marker_bounds(power_csv)
        baseline_mw = spans[0][2] if spans else None
        if not spans:
            # 0.0 here is the absence of a measurement, not a measurement of zero,
            # and printing it as "idle floor: 0 mW" invited it to be quoted.
            print("\n!! no idle samples in the power CSV -- no idle floor could be")
            print("   measured, so active_joules will equal joules on every row.")
        else:
            print("\nidle floor: %.0f mW pooled over %d interval(s); pre-sweep "
                  "baseline %s mW"
                  % (idle_pooled, len(spans),
                     "%.0f" % baseline_mw if baseline_mw is not None else "-"))
            print("            each row is corrected against the idle interval that "
                  "preceded IT; see idle_mw_source")
        for rec in records:
            # A record without a window is not joinable. Guarded rather than
            # indexed: a bare rec["t_start_unix"] here once killed an entire
            # completed sweep with KeyError because a skipped run carried no
            # timestamps, and the joined CSV -- the archived deliverable -- was
            # never written at all.
            if rec.get("t_start_unix") is None or rec.get("t_end_unix") is None:
                continue
            e = energy_between(power_csv, rec["t_start_unix"], rec["t_end_unix"])
            rec.update(e)

            # The marker window spans the whole llama-bench process, so model
            # load off the SD card, CUDA context creation and the host->device
            # weight upload are all inside it -- seconds of multi-watt work that
            # processed no tokens, and that cost -ngl 99 more than -ngl 0. All of
            # it strictly PRECEDES the first timed loop, so the trailing
            # compute_s seconds of the window are the part that did the tokens.
            # Raw joules stays on the row; compute_joules is the figure the
            # per-token columns are built from.
            cs = rec.get("compute_s")
            cs_meas = None
            dur = e.get("duration_s")
            # Anchor on the marker's last sample when we have one.
            anchor = bounds.get(rec.get("marker"), (None, None))[1] or rec["t_end_unix"]
            rec["compute_anchor_unix"] = anchor
            if cs and dur and 0 < cs < dur and e.get("joules") is not None:
                ce = energy_between(power_csv, anchor - cs, anchor)
                if ce.get("joules") is not None:
                    # energy_between returns the counter difference between the
                    # FIRST and LAST sample inside the slice, which spans less
                    # than the nominal compute_s by up to one sampling interval.
                    # Report the measured span, not the nominal one, and subtract
                    # the idle floor over the same span -- mixing the two
                    # over-subtracted the floor and understated J/token by ~7-10%,
                    # by a different amount for prefill than for decode, which
                    # distorts the very ratio the study exists to measure.
                    rec["compute_joules"] = ce["joules"]
                    rec["compute_mean_mw"] = ce.get("mean_mw")
                    rec["compute_s"] = ce["duration_s"]
                    rec["compute_s_nominal"] = cs
                    rec["load_s"] = round(dur - ce["duration_s"], 3)
                    rec["load_joules"] = round(e["joules"] - ce["joules"], 4)
                    cs_meas = ce["duration_s"]

            # The BETWEEN-run quantity temp_rise_c structurally cannot express:
            # how far from cold this case started. temp_rise_c goes to ~0 exactly
            # when the board has reached a hot steady state, i.e. the worst case
            # passes the docs' old "> 8 C" rule.
            rec["temp_baseline_c"] = temp_baseline_c if temp_baseline_c is not None else ""
            if temp_baseline_c is not None and rec.get("temp_start_c") is not None:
                rec["temp_drift_c"] = round(rec["temp_start_c"] - temp_baseline_c, 2)

            tokens = rec.get("tokens")
            if "compute_joules" in rec:
                energy_j, window_s = rec["compute_joules"], cs_meas
                rec["energy_basis"] = "compute"
            else:
                energy_j, window_s = e.get("joules"), dur
                # No avg_ns to trim with, or the window was shorter than the
                # compute it supposedly contains -- say so on the row rather
                # than quietly publishing a load-inclusive J/token.
                rec["energy_basis"] = "whole-process (includes model load)"
            # The floor this case actually ran against, not a sweep-wide mean.
            local_mw, src = idle_floor_at(spans, rec.get("t_start_unix") or 0.0)
            if local_mw is None:
                local_mw, src = (baseline_mw, "baseline-fallback:" + src) \
                    if baseline_mw is not None else (idle_pooled, "pooled-fallback:" + src)
            rec["idle_mw"] = round(local_mw, 1)
            rec["idle_mw_source"] = src
            rec["idle_mw_pooled"] = round(idle_pooled, 1)
            rec["idle_mw_baseline"] = round(baseline_mw, 1) if baseline_mw is not None else ""
            # No max(..., 0.0). A negative active energy means the floor is wrong
            # for this row -- too high, or measured in a different thermal state --
            # and clamping it to zero hid exactly that, silently, on the rows where
            # it mattered most.
            rec["active_joules"] = (round(energy_j - local_mw / 1000.0 * (window_s or 0.0), 4)
                                    if energy_j else None)

            # A row that could not see a power window is not "ok but blank" --
            # say which of the reasons it is.
            if not energy_j and rec.get("per_token_status") == PT_OK:
                rec["per_token_status"] = PT_NO_POWER

            if rec.get("per_token_status") == PT_COMBINED:
                # A fused -pg instance is ONE test whose joules cannot be split
                # between prompt and generation, so it gets a per-REQUEST
                # denominator instead. This is the only case in the matrix that
                # deliberately spans both phases, and it never reports J/token.
                #
                # Its energy is measured over the TIMED repetitions alone, not
                # over compute_s: llama-bench warms a -pg instance with a prompt
                # plus a single token, which is neither a whole request nor
                # separable from avg_ns. The timed loops run last, so the
                # trailing request_window_s seconds are exactly `requests` whole
                # requests -- numerator and denominator agree with no estimate.
                reqs, rws = rec.get("requests"), rec.get("request_window_s")
                # rws <= dur is not enough. An inflated avg_ns makes the request
                # window long enough to swallow model-load energy while still
                # fitting inside the process lifetime -- measured at 22.6
                # J/request against a true 7.9. Require the compute trim to have
                # SUCCEEDED first (so we know where compute began) and the
                # request window to fit inside it. When the driver has already
                # concluded the timing is impossible -- compute_s > duration, so
                # no compute_joules -- it must not then publish a per-request
                # number on the strength of the same avg_ns.
                if reqs and rws and cs and "compute_joules" in rec and 0 < rws <= cs:
                    re_ = energy_between(power_csv, anchor - rws, anchor)
                    rj = re_.get("joules")
                    if rj is not None:
                        rec["request_joules"] = rj
                        rec["request_window_s_measured"] = re_.get("duration_s")
                        rec["joules_per_request"] = round(rj / reqs, 4)
                        rec["wh_per_krequest"] = round(rj / reqs * 1e3 / 3600.0, 4)
                        # same span as the energy, not the nominal one
                        act = rj - local_mw / 1000.0 * (re_.get("duration_s") or rws)
                        # Put active_joules on the SAME basis as energy_basis
                        # says. Left as the compute-slice figure it contradicted
                        # the per-request columns beside it:
                        # active_joules_per_request x requests must equal it.
                        rec["active_joules"] = round(act, 4)
                        rec["active_joules_per_request"] = round(act / reqs, 4)
                        rec["energy_basis"] = "timed requests"
            elif rec.get("per_token_status") == PT_OK and energy_j and tokens:
                rec["joules_per_token"] = round(energy_j / tokens, 6)
                rec["wh_per_mtok"] = round(energy_j / tokens * 1e6 / 3600.0, 2)
                if rec.get("active_joules") is not None:
                    rec["active_joules_per_token"] = round(rec["active_joules"] / tokens, 6)

    # Outside the power-CSV guard on purpose: with --no-power, or with tegrastats
    # missing, the loop above never runs at all, and every row would otherwise
    # keep per_token_status=ok while carrying no energy whatsoever. "ok" has to
    # mean the number is really there.
    for rec in records:
        # `not joules` deliberately catches 0.0 as well as absent: a window whose
        # energy counter did not advance yields no per-token figure either (the
        # join guards on a truthy energy), so a row left saying "ok" would be
        # claiming a number it does not have.
        if rec.get("per_token_status") == PT_OK and not rec.get("joules"):
            rec["per_token_status"] = PT_NO_POWER

    # Lead with the columns a reader needs to trust a row, in a fixed order, so
    # the header does not reshuffle because one case failed or emitted an extra
    # bench row. Everything else follows in first-seen order.
    LEAD = ["case_id", "group", "run_index", "drift_control",
            "phase", "row_phase", "test",
            "model", "model_path", "n_gpu_layers", "threads", "repetitions",
            "status", "per_token_status", "bench_rows", "working_rows",
            "counts_mismatch", "config_mismatch", "counts_source",
            "window_shared", "bench_row_index", "marker",
            "bench_n_prompt", "bench_n_gen", "tokens", "requests",
            "avg_ts", "energy_basis", "joules", "compute_joules", "active_joules",
            "joules_per_token", "wh_per_mtok", "active_joules_per_token",
            "joules_per_request", "wh_per_krequest", "active_joules_per_request",
            "temp_baseline_c", "temp_start_c", "temp_drift_c", "temp_end_c",
            "temp_max_c", "temp_rise_c", "cooldown_actual_s", "cooldown_reached",
            "cpu_freq_start_mhz", "cpu_freq_min_mhz",
            "gpu_freq_start_mhz", "gpu_freq_min_mhz",
            "idle_mw", "idle_mw_source", "idle_mw_pooled", "idle_mw_baseline",
            "power_samples", "sample_gap_max_s", "pow_in_plateau_frac",
            "joules_vs_mean_pct"]
    present = set()
    for rec in records:
        present.update(rec)
    keys = [k for k in LEAD if k in present]
    for rec in records:
        for k in rec:
            if k not in keys:
                keys.append(k)
    with open(joined_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    # The cluster's shared 40-column projection, beside the node's own richer CSV.
    # Emitted automatically because a step nobody remembers to run is a step that
    # does not happen -- and this node was the last one the cross-node table could
    # not read. Failure here must never cost the sweep its own results, which are
    # already safely written above.
    # TWO names on purpose. ThinkPad/control/analysis.py:135 opens
    # `run_dir / "results.csv"` -- the exact name, no globbing -- so a
    # <base>-prefixed file alone would never be found by the energy join. The
    # prefixed copy keeps a multi-sweep results/ directory readable.
    shared_csv = os.path.join(out_dir, base + "_results.csv")
    shared_canonical = os.path.join(out_dir, "results.csv")
    try:
        sys.path.insert(0, HERE)
        import emit_results_csv as shared
        rows, omitted = [], 0
        for rec in records:
            row = shared.row_to_shared(rec, base)
            if row is None:
                omitted += 1
            else:
                rows.append(row)
        for path in (shared_csv, shared_canonical):
            with open(path, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=shared.COLUMNS)
                w.writeheader()
                for r in rows:
                    w.writerow(r)
        print("wrote %s and results.csv  (%d shared-schema rows%s)"
              % (os.path.basename(shared_csv), len(rows),
                 ", %d fused row(s) omitted: the shared schema has no phase "
                 "meaning both" % omitted if omitted else ""))
    except Exception as exc:
        # Loud on stdout AND in the archived log: nobody reads a three-hour
        # sweep's console scrollback, and a missing cross-node CSV is otherwise
        # indistinguishable from a sweep that was never projected.
        msg = ("!! could not write the shared-schema CSV: %s\n"
               "   the node's own %s is unaffected; re-run "
               "bench/emit_results_csv.py against it"
               % (exc, os.path.basename(joined_csv)))
        print(msg)
        try:
            with open(log_path, "a", encoding="utf-8") as lf:
                lf.write("\n" + msg + "\n")
        except IOError:
            pass

    print("\nwrote %s" % joined_csv)
    print("wrote %s" % log_path)
    ok = sum(1 for r in records if r.get("status") == "ok")
    ran = {r.get("marker") for r in records if r.get("status") != "skipped"}
    n_skip = sum(1 for r in records if r.get("status") == "skipped")
    print("%d/%d rows ok (%d invocations%s)"
          % (ok, len(records), len(ran),
             ", %d run(s) skipped" % n_skip if n_skip else ""))
    by_status = {}
    for r in records:
        st = r.get("per_token_status") or "?"
        by_status[st] = by_status.get(st, 0) + 1
    for st, n in sorted(by_status.items()):
        print("  per_token_status %-26s %d" % (st, n))

    # The drift-control pair, printed because a number nobody looks at is a
    # number nobody acts on.
    ctrl = next((r for r in records
                 if r.get("drift_control") == 1 and r.get("avg_ts")), None)
    if ctrl:
        first = next((r for r in records
                      if r.get("drift_control") == 0
                      and r.get("case_id") == ctrl.get("case_id")
                      and r.get("row_phase") == ctrl.get("row_phase")
                      and r.get("avg_ts")), None)
        if first:
            a, b = float(first["avg_ts"]), float(ctrl["avg_ts"])
            d0, d1 = first.get("temp_drift_c"), ctrl.get("temp_drift_c")
            hotter = ("%+.1f C" % (d1 - d0)) if d0 not in (None, "") and d1 not in (None, "") else "?"
            print("\ndrift control: %s ran %.2f t/s first and %.2f t/s last (%+.1f%%), "
                  "board %s hotter" % (ctrl.get("case_id"), a, b, (b - a) / a * 100.0, hotter))
            print("               that delta is thermal drift, not a property of any "
                  "axis the sweep varies")
    if by_status.get(PT_MULTI_PHASE):
        print("  ^ those rows share a power window with another phase, so no "
              "per-token energy was written for them. Split the case in "
              "%s." % os.path.basename(args.matrix))


_ROWS_CACHE = {}


def _rows(power_csv):
    """Parsed power CSV, memoised on (path, size, mtime).

    Up to three energy_between() calls per record, plus the idle scan, each
    re-parsed the whole file: O(records x rows). A 32-run sweep at 10 Hz over
    three hours is ~100k rows, so the join took minutes and grew quadratically
    with the sampling rate -- which is a problem when raising that rate is the
    point of finding #29. Keyed on size and mtime so a rewritten file is not
    served from cache.
    """
    try:
        st = os.stat(power_csv)
        key = (power_csv, st.st_size, st.st_mtime)
    except OSError:
        key = (power_csv, None, None)
    hit = _ROWS_CACHE.get(key)
    if hit is None:
        with open(power_csv, newline="", encoding="utf-8") as fh:
            hit = list(csv.DictReader(fh))
        _ROWS_CACHE.clear()          # one file at a time; do not grow unbounded
        _ROWS_CACHE[key] = hit
    return hit


def marker_bounds(power_csv):
    """{marker: (first_ts, last_ts)} over the power CSV, computed once.

    The trim must anchor on the last sample that still carries this run's
    marker, not on the wall-clock time the process exited: teardown, the CSV
    write and interpreter shutdown all sit between those two, and anchoring on
    the later one slides the whole compute slice forward, dragging in load
    energy. It also made the driver disagree with telemetry/energy_window.py --
    which anchors on the marker -- by 7.4% on decode rows, while the protocol
    names that tool as the way to re-derive a published figure by hand.
    """
    out = {}
    for r in _rows(power_csv):
        m = (r.get("marker") or "").strip()
        if not m or m == "idle":
            continue
        try:
            t = float(r.get("ts_unix") or 0)
        except ValueError:
            continue
        lo, hi = out.get(m, (t, t))
        out[m] = (min(lo, t), max(hi, t))
    return out


# Seconds of each idle interval to discard before averaging. Clocks and fans ramp
# down after a run, so the first samples of a "cooldown" are still carrying the
# previous case's heat and duty.
IDLE_SETTLE_S = 3.0


def idle_intervals(power_csv):
    """Contiguous idle runs as (t_start, t_end, mean_mw), in file order.

    Only samples marked exactly "idle" count. The pre-read of a model now carries
    its own precache:<case>:<NNN> marker, so a multi-watt SD read can no longer
    land in here -- which it used to, because it happened before the run's marker
    was set and therefore under the idle one.
    """
    out, cur = [], []
    for r in _rows(power_csv):
        m = (r.get("marker") or "").strip()
        t, p = _num(r, "ts_unix"), _num(r, "pow_in_mw")
        if m == "idle" and t is not None and p is not None:
            cur.append((t, p))
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)

    spans = []
    for run_ in out:
        t0 = run_[0][0]
        kept = [p for t, p in run_ if t - t0 >= IDLE_SETTLE_S] or [p for _, p in run_]
        spans.append((run_[0][0], run_[-1][0], sum(kept) / len(kept)))
    return spans


def idle_floor(power_csv):
    """Mean module power over EVERY idle sample in the file.

    Kept, but no longer the subtrahend. It pools the cold pre-sweep baseline with
    every hot cooldown, and by sample count the cooldowns dominate -- so on a
    board whose leakage rises with junction temperature it sits several hundred
    mW above the baseline, and subtracting one such scalar from all 30-odd rows
    biases along the case-order axis, which is the axis the sweep varies. It is
    written to every row as `idle_mw_pooled` so an older figure stays
    reproducible and the choice is auditable from the archived CSV alone.
    """
    vals = []
    for r in _rows(power_csv):
        if (r.get("marker") or "").strip() == "idle":
            p = _num(r, "pow_in_mw")
            if p is not None:
                vals.append(p)
    return sum(vals) / len(vals) if vals else 0.0


def idle_floor_at(spans, t_window_start):
    """The idle interval that ended most recently BEFORE this case opened.

    This is the floor the case actually ran against: same board temperature, same
    fan duty, same minute. For the first case it is the 30 s pre-sweep baseline,
    so there is one definition rather than two. Returns (mw, source) where source
    names the interval, or (None, reason) when there is none to use.
    """
    prior = [s for s in spans if s[1] <= t_window_start]
    if not prior:
        return None, "no-idle-before-window"
    t0, t1, mw = prior[-1]
    return mw, "idle[%.3f..%.3f]" % (t0, t1)


def _num(row, k):
    try:
        return float(row.get(k) or "")
    except (TypeError, ValueError):
        return None


def energy_at(rows, t, col="energy_in_j"):
    """The running energy integral at exactly time `t`, linearly interpolated
    between the samples that bracket it.

    Without this, a slice's energy is the counter difference between the first
    and last sample INSIDE it, which spans up to one sampling interval less than
    the slice actually asked for -- and the missing energy is then divided by the
    full token count. At the default 5 Hz that understated prefill J/token by
    about 10% and decode by about 6%: not a wash, a systematic bias that differs
    between the two phases and so distorts the prefill-vs-decode ratio the study
    exists to measure. The counter is a trapezoidal integral of a smoothly
    varying rail, so linear interpolation between neighbours is the right
    reconstruction. Returns None if `t` lies outside the sampled range.
    """
    prev = None
    for r in rows:
        ts = _num(r, "ts_unix")
        e = _num(r, col)
        if ts is None or e is None:
            continue
        if ts == t:
            return e
        if ts > t:
            if prev is None:
                return None                     # before the first sample
            pts, pe = prev
            if ts == pts:
                return e
            return pe + (e - pe) * (t - pts) / (ts - pts)
        prev = (ts, e)
    return None                                 # after the last sample


def energy_between(power_csv, t0, t1):
    rows = _rows(power_csv)
    sel = []
    for r in rows:
        try:
            t = float(r.get("ts_unix") or 0)
        except ValueError:
            continue
        if t0 <= t <= t1:
            sel.append(r)
    if len(sel) < 2:
        return {}

    num = _num

    # Interpolate the counter to the requested bounds so no energy is dropped at
    # the edges; fall back to the enclosed samples when a bound lies outside the
    # sampled range (the whole-window case, where t0/t1 come from wall clock).
    e0 = energy_at(rows, t0)
    e1 = energy_at(rows, t1)
    span = (t1 - t0) if (e0 is not None and e1 is not None) else None
    if e0 is None:
        e0 = num(sel[0], "energy_in_j")
    if e1 is None:
        e1 = num(sel[-1], "energy_in_j")
    ps = [num(r, "pow_in_mw") for r in sel]
    ps = [p for p in ps if p is not None]
    temps = [num(r, "temp_thermal_c") for r in sel]
    temps = [t for t in temps if t is not None]
    swaps = [num(r, "swap_used_mb") for r in sel]
    swaps = [s for s in swaps if s is not None]
    # What bounds this integral, carried on the row instead of left underivable.
    # A trapezoid across a long hole is a straight line drawn through whatever
    # the rail actually did; without the gap on the row, a reader cannot tell a
    # densely sampled window from one crossed in two steps.
    ts = [t for t in (num(r, "ts_unix") for r in sel) if t is not None]
    gap_max = max((b - a for a, b in zip(ts, ts[1:])), default=0.0)
    # How often the rail repeated its previous reading. A high fraction means the
    # requested rate is not real -- tegrastats is re-printing a value the INA has
    # not refreshed -- so the effective bandwidth is lower than the sample count
    # suggests. It is NOT an aliasing check; nothing here can detect aliasing
    # against the token cadence.
    plateau = 0
    for a, b in zip(ps, ps[1:]):
        if a == b:
            plateau += 1
    out = {
        # the span the energy actually covers: the requested bounds when both
        # were interpolable, else the sampled extent
        "duration_s": round(span if span is not None
                            else float(sel[-1]["ts_unix"]) - float(sel[0]["ts_unix"]), 3),
        "power_samples": len(sel),
        "sample_gap_max_s": round(gap_max, 4),
        "pow_in_plateau_frac": round(plateau / (len(ps) - 1), 4) if len(ps) > 1 else None,
        "mean_mw": round(sum(ps) / len(ps), 1) if ps else None,
        "peak_mw": max(ps) if ps else None,
    }
    if e0 is not None and e1 is not None:
        out["joules"] = round(e1 - e0, 4)
        # The logger's running integral vs mean power x duration. These are two
        # arithmetics over the SAME samples, so agreement proves the bookkeeping
        # and nothing about whether the samples represent the rail. Disagreement
        # means one of the two is wrong; agreement means neither is checked.
        if out.get("mean_mw") and out["duration_s"]:
            naive = out["mean_mw"] / 1000.0 * out["duration_s"]
            if naive:
                out["joules_vs_mean_pct"] = round(
                    (out["joules"] - naive) / naive * 100.0, 2)
    freqs_c = [num(r, "cpu_freq_mhz") for r in sel]
    freqs_c = [f for f in freqs_c if f is not None]
    freqs_g = [num(r, "gpu_freq_mhz") for r in sel]
    freqs_g = [f for f in freqs_g if f is not None]
    if freqs_c:
        # The clock at the START of the window and the lowest it reached. A
        # post-hoc throttling audit needs both, and neither was on the row: a
        # case that ran at a clamped clock was indistinguishable from one that
        # did not.
        out["cpu_freq_start_mhz"] = freqs_c[0]
        out["cpu_freq_min_mhz"] = min(freqs_c)
    if freqs_g:
        out["gpu_freq_start_mhz"] = freqs_g[0]
        out["gpu_freq_min_mhz"] = min(freqs_g)
    if temps:
        # temp_start_c/temp_end_c bring the driver into step with
        # telemetry/energy_window.py, which has had them all along.
        out["temp_start_c"] = temps[0]
        out["temp_end_c"] = temps[-1]
        out["temp_max_c"] = max(temps)
        # The temperature at the END of the window, which is what the cluster's
        # shared schema means by temp_end_c -- not the max over the window. The
        # two differ whenever the board cooled during a run, and quoting one
        # under the other's name is the kind of substitution nobody can catch
        # downstream.
        out["temp_end_c"] = temps[-1]
        out["temp_rise_c"] = round(temps[-1] - temps[0], 2)
    if swaps:
        out["swap_peak_mb"] = max(swaps)
    return out


def tokens_of(rec):
    """Tokens actually processed by THIS bench row, counting repetitions and the
    warm-up run llama-bench performs before each measured set.

    Reads the row's OWN bench_n_prompt / bench_n_gen, falling back to the
    matrix's request. That matters: on a two-test invocation the matrix says
    p=512 g=128 for both rows, while the rows themselves say (512, 0) and
    (0, 128). Dividing each row's energy by the matrix's combined count was the
    second half of audit finding #3.

    n_prompt / n_gen are PER repetition, and the warm-up before each test's
    repetition loop -- a full prompt pass for pp, one generated token for tg --
    happens inside the same marker window and burns real joules. Both terms
    belong in the divisor; without the warm-up term every prefill J/token comes
    out (reps+1)/reps too high, which is +33% at the matrix default of -r 3 and
    +50% for the -r 2 cases, so the bias is not even constant across the sweep.
    Must stay in step with bench_tokens() in telemetry/energy_window.py.
    """
    n_p = rec.get("bench_n_prompt")
    n_g = rec.get("bench_n_gen")
    if n_p is None and n_g is None:
        n_p, n_g = rec.get("n_prompt"), rec.get("n_gen")
    n_p, n_g = n_p or 0, n_g or 0
    reps = rec.get("repetitions") or 1
    # One formula covers all three shapes, because llama-bench's warm-up is the
    # same either way: `if (t.n_prompt > 0)` runs a full prompt pass and
    # `if (t.n_gen > 0)` runs test_gen(ctx, 1) -- a SINGLE token, even for a
    # fused -pg instance. So a fused row warms up with a prompt plus one token,
    # exactly as a pp row and a tg row do between them. That is also why the
    # phase split changes no measured work: prefill tokens + decode tokens
    # equals the old combined count, at any -r.
    return (n_p * (reps + 1) + n_g * reps + (1 if n_g else 0)) or None


def requests_of(rec):
    """Whole fused requests served inside the window: the TIMED repetitions only.

    The warm-up is deliberately excluded, and so is its energy -- see
    request_window_s(). llama-bench warms a -pg instance with a full prompt plus
    a single generated token, which is not a whole request, so counting it as one
    would divide by more requests than were actually served and understate the
    headline number. Restricting both numerator and denominator to the timed
    loops needs no assumption about the warm-up at all.
    """
    if rec.get("row_phase") != "combined":
        return None
    return rec.get("repetitions") or 1


def request_window_s(rec):
    """Seconds of the TIMED repetitions of a fused row: avg_ns x reps.

    The warm-up runs before the repetition loop, so the trailing avg_ns x reps
    seconds of the window are exactly the whole requests -- no warm-up, no model
    load, nothing to estimate.
    """
    if rec.get("row_phase") != "combined":
        return None
    try:
        ns = float(rec.get("avg_ns") or "")
    except (TypeError, ValueError):
        return None
    if not ns:
        return None
    return round(ns * (rec.get("repetitions") or 1) / 1e9, 3)


if __name__ == "__main__":
    main()
