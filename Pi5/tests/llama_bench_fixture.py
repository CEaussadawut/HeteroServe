"""Build run directories shaped exactly like the ones 10_isolated_bench.sh produces.

The point of this module is fidelity, not convenience. An earlier fixture -- the one
recorded in the first handoff's "How to verify" -- invented the field names `n_layer`
and `backend_name`. Neither exists in llama-bench. Tests written against it would have
passed while the real sweep produced blank columns, which is precisely the failure that
made `c_per_layer_ms` silently empty on every row of the first dataset.

So the field list below is transcribed from upstream `tools/llama-bench/llama-bench.cpp`:

  * `test::get_fields()` is a fixed list. It carries `model_size`, `model_n_params`,
    `model_type` and `n_gpu_layers` -- and **no total layer count of any kind**.
    `n_gpu_layers` is the offload count, which is 0 on a CPU-only worker, so it is not
    a stand-in. The layer count must come from the GGUF header instead; that is why
    `node/gguf_meta.py` exists and why `cells.csv` carries an `n_layers` column.
  * The backend field is spelled `backends`, plural.
  * `json_printer::print_test` appends `samples_ns` and `samples_ts` to every entry,
    beyond `get_fields()`. Those hold the per-repetition distribution.
  * The document is a top-level JSON *array*, one object per (n_prompt, n_gen) leg.
  * `avg_ts`/`stddev_ts` print unquoted as floats; `n_prompt`, `n_gen`, `model_size`,
    `model_n_params` print unquoted as ints; `backends` and `model_type` are strings.

If upstream changes these, this file is where the disagreement should surface first.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

# The header 10_isolated_bench.sh writes, character for character (line 42).
CELLS_HEADER = [
    "cell_id", "started_utc", "ended_utc", "model", "n_layers",
    "n_prompt", "n_gen", "threads", "rep", "exit_code",
    "temp_start_c", "temp_end_c", "throttled_start", "throttled_end",
    "idle_start", "idle_end", "raw_json",
]


def entry(*, n_prompt, n_gen, avg_ts, samples_ts=None, model_size=4_700_000_000,
          model_n_params=7_615_616_512, model_type="Q4_K - Medium",
          backends="CPU", n_threads=4, **extra):
    """One llama-bench result object: every field upstream emits, nothing invented.

    CAUTION for tests that care about the CENTRE of the samples: with `samples_ts` left at
    its default the sample list is `[avg_ts]`, so mean == median == avg_ts and the three
    are indistinguishable. A mutation recomputing tokens_per_s from the samples, or
    swapping `fmean` for `median`, survives on any fixture built this way. Pass an
    explicitly asymmetric `samples_ts` (e.g. [2.0, 2.1, 5.0], mean 3.033 vs median 2.1)
    whenever the test is about which statistic is published.
    """
    samples = list(samples_ts) if samples_ts else [avg_ts]

    def _stdev(values):
        """Sample stddev (n-1), 0 for a single value -- upstream's `::stdev` helper."""
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        return (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5

    stddev_ts = _stdev(samples)
    # avg_ns/samples_ns are the duration of the WHOLE test, not of one token. Upstream
    # `test::get_ts()` is `1e9 * n_tokens / t` with `n_tokens = n_prompt + n_gen`, so the
    # inverse carries that factor: t = n_tokens * 1e9 / ts. Writing `1e9 / ts` here is
    # invisible while only avg_ts/samples_ts are read -- and wrong by a factor of n
    # (64, 128, 256, 512) the moment anything derives a DURATION from these fields, which
    # the Pi4-schema columns avg_ms, stddev_ms and rep_busy_s do. Pi 4 hit this exact trap
    # and fixed it (Pi4/tests/fixtures.py:32-37); the note is carried here so it stays named.
    n_tokens = n_prompt + n_gen
    avg_ns = int(n_tokens * 1e9 / avg_ts) if avg_ts else 0
    samples_ns = [int(n_tokens * 1e9 / s) if s else 0 for s in samples]
    # stddev_ns was hardcoded to 0 here, which made the whole `stddev_ms` column
    # unreachable: 0 divided by 1e6 is 0 whatever the conversion does, so any scaling of it
    # -- or blanking it outright -- passed the suite. Upstream computes it as
    # `::stdev(samples_ns)`, the same helper it uses for stddev_ts, so derive it the same
    # way. Exactly the sibling of the avg_ns-per-token bug named above: the fixture modelled
    # the FIELD but not its VALUE.
    stddev_ns = int(_stdev(samples_ns))
    e = {
        "build_commit": "deadbeef", "build_number": 1234,
        "cpu_info": "ARM Cortex-A76", "gpu_info": "", "backends": backends,
        "model_filename": "models/test.gguf", "model_type": model_type,
        "model_size": model_size, "model_n_params": model_n_params,
        "n_batch": 2048, "n_ubatch": 512, "n_threads": n_threads,
        "cpu_mask": "0x0", "cpu_strict": False, "poll": 50,
        "type_k": "f16", "type_v": "f16",
        # The offload count, and the reason it cannot serve as a layer count: this node
        # is CPU-only, so upstream reports 0 here no matter how many layers the model has.
        "n_gpu_layers": 0,
        "n_cpu_moe": 0, "split_mode": "layer", "main_gpu": 0,
        "no_kv_offload": False, "flash_attn": 0, "devices": "none",
        "tensor_split": "0.00", "tensor_buft_overrides": "none",
        "load_mode": "mmap", "lazy_mode": "none", "embeddings": False,
        "no_op_offload": 0, "no_host": False, "fit_target": 0, "fit_min_ctx": 0,
        "n_prompt": n_prompt, "n_gen": n_gen, "n_depth": 0,
        "test_time": "2026-09-13T00:00:00Z",
        "avg_ns": avg_ns, "stddev_ns": stddev_ns,
        "avg_ts": avg_ts, "stddev_ts": round(stddev_ts, 6),
        "samples_ns": samples_ns,
        "samples_ts": samples,
    }
    e.update(extra)
    return e


DEFAULT_NODE_ENV = {"NODE_ID": "pi5", "NODE_ROLE": "worker-strong"}


def write_run(root, cells, manifest=None, node_env=DEFAULT_NODE_ENV):
    """Materialise a run directory. `cells` is a list of dicts; see `cell()`.

    node.env.snapshot is copied unconditionally by 10_isolated_bench.sh, so it is written
    here by default; pass node_env=None to model a directory that lacks it.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        (root / "build_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8")
    if node_env is not None:
        (root / "node.env.snapshot").write_text(
            "# snapshot of config/node.env\n"
            + "".join("%s=%s\n" % (k, v) for k, v in node_env.items()),
            encoding="utf-8")

    with (root / "cells.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CELLS_HEADER)
        w.writeheader()
        for c in cells:
            row = {k: c[k] for k in CELLS_HEADER if k in c}
            w.writerow(row)
            entries = c.get("entries")
            if entries is None:
                continue
            (root / c["raw_json"]).write_text(
                json.dumps(entries, indent=2), encoding="utf-8")
    return root


def cell(cell_id="c001", *, model="qwen2.5-7b-q4km", n_layers=32, n_prompt=256,
         n_gen=128, threads=4, rep=1, exit_code=0, temp_start_c="58.1",
         temp_end_c="81.2", raw_json=None, entries=None,
         throttled_start="0x0", throttled_end="0x0",
         idle_start="1789257300.0", idle_end="1789257560.0",
         prefill_ts=11.5, decode_ts=2.35, **extra):
    """One cells.csv row plus the llama-bench document it points at.

    The default numbers are the ones the Pi 5 handoff carried forward as its worked
    example: 2.35 tok/s decode over 32 layers on a 4.7 GB model, ending at 81.2 C.
    """
    raw_json = raw_json or ("%s.json" % cell_id)
    if entries is None:
        entries = [
            entry(n_prompt=n_prompt, n_gen=0, avg_ts=prefill_ts, n_threads=threads),
            entry(n_prompt=0, n_gen=n_gen, avg_ts=decode_ts, n_threads=threads),
        ]
    row = {
        "cell_id": cell_id, "started_utc": "2026-09-13T00:00:00Z",
        "ended_utc": "2026-09-13T00:04:00Z", "model": model, "n_layers": n_layers,
        "n_prompt": n_prompt, "n_gen": n_gen, "threads": threads, "rep": rep,
        "exit_code": exit_code, "temp_start_c": temp_start_c,
        "temp_end_c": temp_end_c,
        # The firmware word at each cell boundary, and the cool-down that preceded this
        # cell. Defaults model a clean board with a recorded idle window; pass "" to model
        # a run directory written before 10_isolated_bench.sh carried these columns.
        "throttled_start": throttled_start, "throttled_end": throttled_end,
        "idle_start": idle_start, "idle_end": idle_end,
        "raw_json": raw_json, "entries": entries,
    }
    row.update(extra)
    return row
