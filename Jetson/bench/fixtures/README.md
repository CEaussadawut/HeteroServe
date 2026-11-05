# Off-hardware fixture for the bench driver

Exercises `bench/run_matrix.py` end to end with no Jetson: a fake `tegrastats`
whose reported power follows a phase file, and a fake `llama-bench` that behaves
like the real one in the three ways that matter — it writes its CUDA banner to
**stderr** and its CSV to **stdout**, it emits **two CSV rows** when both `-p`
and `-n` are non-zero, and it emits **one** row for `-pg`.

That two-row case is audit finding #3: one invocation, one power window, two
phases. `BEFORE_mixed_bench_energy.csv` is the driver's output for it *before*
the fix — keep it as the regression baseline.

## The three matrices

| File | What it proves |
|---|---|
| `matrix_mixed.json` | a mixed case is **refused by name** at `--dry-run`, before any hardware time is spent |
| `matrix_split.json` | the post-fix shape: a grouped prefill/decode pair comes out **interleaved**, and a `-pg` case reports `joules_per_request` with no `joules_per_token` |
| `bench/matrix.json` | the real sweep — `--dry-run` it to see 32 runs and zero mixed |

## Run it

`./bench/fixtures/run_regression.sh` does all of this for you and asserts the
results. The manual recipe below is for poking at one piece.

```sh
FIX=$(mktemp -d)
mkdir -p "$FIX/bin" "$FIX/models" "$FIX/results"
cp bench/fixtures/fake_tegrastats.py  "$FIX/bin/tegrastats"
cp bench/fixtures/fake_llama_bench.py "$FIX/bin/llama-bench"
chmod +x "$FIX/bin/tegrastats" "$FIX/bin/llama-bench"
head -c 20000000 /dev/urandom > "$FIX/models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"

# NEVER write config/node.env -- it is live, gitignored machine config (this
# node's paths and its llama.cpp pin) and the clobber is silent. Pass --env-file.
cat > "$FIX/node.env" <<ENV
NODE_ID=fixture
LLAMA_BIN="$FIX/bin"
MODEL_DIR="$FIX/models"
RESULTS_DIR="$FIX/results"
TELEMETRY_HZ=5
ENV

export PATH="$FIX/bin:$PATH" HS_FAKE_PHASE="$FIX/phase"
echo idle > "$HS_FAKE_PHASE"

# 1. a mixed case must be refused, exit 1, naming the case
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix bench/fixtures/matrix_mixed.json --dry-run

# 2. the post-fix shape must run clean and satisfy every invariant
python3 bench/run_matrix.py --env-file "$FIX/node.env" --matrix bench/fixtures/matrix_split.json --tag fixture
python3 bench/check_finding3_invariants.py "$FIX"/results/*_bench_energy.csv

```

Budget ~45 s per driver run: it sleeps 30 s collecting an idle baseline, and
`cooldown_s` is 1 in the fixture matrices. Models are random bytes — nothing
reads their contents, only their size, so `precache()` has something real to pull
through the page cache.

## Environment knobs on the fake

| Variable | Effect |
|---|---|
| `HS_FAKE_PHASE` | **required** — path the fake writes its current phase to, and the fake tegrastats reads power from |
| `HS_FAKE_LOAD_S` | **seconds** spent in the `load` phase before the timed loops (default 3.0). It should show up as `load_s` on a split row, and `load_joules` should be roughly that many seconds at the fake's load power of 5200 mW — so ~15.6 J at the default. On a `combined` row `load_s` comes back short, because a fused row's warm-up cannot be separated from its `avg_ns` and `compute_s` is deliberately the upper bound |
| `HS_FAKE_EXTRA_ROW` | emit a second, opposite-phase row for a single-phase request — simulates a ref that returns a row for the zero-valued axis. This is the **only** way to exercise the driver's `refused_multi_phase` guard, since `expand()` now blocks the matrix route into that state |
| `HS_FAKE_HIDE_PROMPT` | report a fused `-pg` row with `n_prompt=0`, as a ref would if it put the prompt count in a column the driver does not read. The row then looks like decode while the window's energy covers prompt **and** generation — exercises `refused_phase_mismatch` |
| `HS_FAKE_NO_COUNTS` | omit the `n_prompt`/`n_gen` columns entirely, so no row is identifiable as having done work — exercises the zero-working-rows half of the multi-phase guard |

## Gotcha

`tegrastats` output contains `%` (`EMC_FREQ 21%@1600`). In a Python `%`-format
string those must be written `%%` or the fake dies at startup — and the logger
will not tell you, it will happily log the traceback as CSV rows. A sample count
far below `hz * duration` is the symptom.
