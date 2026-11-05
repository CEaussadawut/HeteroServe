"""Synthetic run directories for the Pi 4 node tests.

Builds exactly what `scripts/06_bench_local.sh` writes into
`$DATA_DIR/bench/<RUN_ID>/` -- a set of llama-bench `-o json` files plus a
`windows.jsonl` -- so the parser can be tested off-target. Nothing here talks to a
Raspberry Pi, a model file or a power meter.

The point of testing against a *fabricated* sweep is that it is the only way to assert
the energy-join keys agree. On real hardware a join miss looks like a blank column and
gets noticed months later; here it fails in a second.
"""
from __future__ import annotations

import json
from pathlib import Path

# The models the Pi 4 actually holds, copied from config/models.txt. Kept as a literal
# rather than parsed from the config so that a test failure says "the parser broke",
# not "the config moved"; tests/test_bench_parse.py separately asserts the two agree.
MODELS = [
    ("tinyllama-1.1b-q4km", "llama 1B Q4_K - Medium", "Q4_K_M"),
    ("tinyllama-1.1b-q8", "llama 1B Q8_0", "Q8_0"),
    ("qwen2.5-1.5b-q4km", "qwen2 1.5B Q4_K - Medium", "Q4_K_M"),
    ("llama3.2-3b-q4km", "llama 3B Q4_K - Medium", "Q4_K_M"),
]

BUILD_COMMIT = "1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b"


def bench_entry(model_type, *, n_prompt, n_gen, n_threads, avg_ts, reps=5):
    """One llama-bench result object, in the shape `-o json` emits."""
    # Real llama-bench avg_ns is the time for the WHOLE test, not for one token:
    # avg_ts = n / (avg_ns / 1e9). The parser never read avg_ns, so the old
    # `int(1e9 / avg_ts)` was invisible -- but samples_ns is now used as a duration when
    # the energy window is cut at repetition boundaries, so it has to be right.
    n = n_prompt or n_gen
    avg_ns = int(n * 1e9 / avg_ts) if avg_ts > 0 else 0
    return {
        "build_commit": BUILD_COMMIT[:8],
        "build_number": 4321,
        "cpu_info": "ARMv8 Processor rev 3 (v8l)",
        "gpu_info": "",
        "backends": "CPU",
        "model_filename": "does-not-matter.gguf",
        "model_type": model_type,
        "model_size": 668788224,
        "model_n_params": 1100048384,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_threads": n_threads,
        "type_k": "f16",
        "type_v": "f16",
        "n_gpu_layers": 0,
        "split_mode": "layer",
        "main_gpu": 0,
        "no_kv_offload": False,
        "flash_attn": False,
        "tensor_split": "0.00",
        "use_mmap": True,
        "embeddings": False,
        "n_prompt": n_prompt,
        "n_gen": n_gen,
        "test_time": "2026-09-12T09:00:00Z",
        "avg_ns": avg_ns,
        "stddev_ns": int(avg_ns * 0.02),
        "avg_ts": avg_ts,
        "stddev_ts": round(avg_ts * 0.02, 4),
        "samples_ns": [avg_ns] * reps,
        "samples_ts": [avg_ts] * reps,
    }


def write_run(
    run_dir: Path,
    models=None,
    *,
    threads=(1, 2, 4),
    pp=(128,),
    tg=(128,),
    throttled_on=(),
    skip_windows_for=(),
    t0=1_789_000_000.0,
):
    """Write a whole synthetic sweep.

    `throttled_on`  -- iterable of tags whose window records a non-zero throttle word.
    `skip_windows_for` -- iterable of tags deliberately given NO window, to exercise the
                       "energy join impossible" path.
    A tag is the basename the sweep uses: "<model>_<phase><n>_t<threads>".
    """
    models = models or MODELS
    run_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    clock = t0

    for name, model_type, _quant in models:
        for th in threads:
            for phase, values in (("pp", pp), ("tg", tg)):
                for n in values:
                    tag = "%s_%s%d_t%d" % (name, phase, n, th)
                    # Prefill is fast, decode is slow, and both scale poorly on an A72 --
                    # roughly what this node is expected to show.
                    rate = (9.0 if phase == "pp" else 2.4) * (0.55 + 0.15 * th)
                    entry = bench_entry(
                        model_type,
                        n_prompt=n if phase == "pp" else 0,
                        n_gen=0 if phase == "pp" else n,
                        n_threads=th,
                        avg_ts=round(rate, 3),
                    )
                    (run_dir / (tag + ".json")).write_text(
                        json.dumps([entry], indent=2), encoding="utf-8"
                    )

                    if tag in skip_windows_for:
                        clock += 60.0
                        continue
                    start = clock

                    # Schema-2 window: what node/bench_windows.py emits. t_start/t_end
                    # stay the RAW process window; the repetition span is additive.
                    # The offsets are deliberately non-integer so that
                    # test_window_timestamps_are_carried_through_unrounded keeps testing
                    # sub-second precision.
                    rep_ns = entry["samples_ns"]
                    busy = sum(rep_ns) / 1e9
                    load_s, warm_s, gap_s, tear_s = 3.211, 4.926, 0.05, 2.4
                    meas_start = start + load_s + warm_s        # 8.137 s, non-integer
                    rep_windows, cursor = [], meas_start
                    for i, ns in enumerate(rep_ns, 1):
                        rep_windows.append({"i": i, "t_start": round(cursor, 6),
                                            "t_end": round(cursor + ns / 1e9, 6), "ns": ns})
                        cursor += ns / 1e9 + gap_s
                    meas_end = rep_windows[-1]["t_end"]
                    clock = meas_end + tear_s   # process ends AFTER the last rep (gate G5)

                    windows.append(
                        {
                            "run": run_dir.name,
                            "node": "pi4",
                            "model": name,
                            "phase": phase,
                            "n": n,
                            "threads": th,
                            "t_start": round(start, 6),
                            "t_end": round(clock, 6),
                            "json": tag + ".json",
                            "temp_end_c": 52.4,
                            "throttled_end": "0x50005" if tag in throttled_on else "0x0",
                            "schema": 2,
                            "window_kind": "reps-span",
                            "window_fallback_reason": "",
                            "t_meas_start": round(meas_start, 6),
                            "t_meas_end": round(meas_end, 6),
                            "t_load_start": round(start, 6),
                            "t_load_end": round(start + load_s, 6),
                            "t_warmup_start": round(start + load_s, 6),
                            "t_warmup_end": round(meas_start, 6),
                            "t_teardown_start": round(meas_end, 6),
                            "t_teardown_end": round(clock, 6),
                            "reps": rep_windows,
                            "rep_span_s": round(meas_end - meas_start, 6),
                            "rep_busy_s": round(busy, 6),
                            "rep_gap_s": round(meas_end - meas_start - busy, 6),
                            "energy_duty": round(busy / (meas_end - meas_start), 4),
                            "min_gap_s": gap_s,
                            "max_gap_s": gap_s,
                            "idle_start": round(start - 15.0, 6),
                            "idle_end": round(start - 3.0, 6),
                            "rc": 0,
                            "stderr_tsv": tag + ".stderr.tsv",
                            "temp_start_c": 47.1,
                            "throttled_start": "0x0",
                        }
                    )
                    clock += 20.0  # the sweep's inter-run settle

    (run_dir / "windows.jsonl").write_text(
        "".join(json.dumps(w) + "\n" for w in windows), encoding="utf-8"
    )
    (run_dir / "build_manifest.json").write_text(
        json.dumps({"node": "pi4", "llama_cpp": {"commit": BUILD_COMMIT}}, indent=2),
        encoding="utf-8",
    )
    return run_dir
