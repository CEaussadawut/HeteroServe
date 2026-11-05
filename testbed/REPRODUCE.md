# Reproducing HeteroServe's results

Topic-form scope 8 ("Reproducibility and Automation"). Three levels, from the cheapest:

1. **Re-derive every published figure from the raw data**: the laptop alone, no boards, a few minutes.
2. **Re-run an experiment on the boards**: the testbed as `README.md` describes it.
3. **Use the data elsewhere**: the CSV export, and a manifest that pins every file.

Nothing here has been published outside this machine. Making the code and data public is the user's decision.

## 1. Re-derive every figure (no boards)

```bash
python3 testbed/scripts/reproduce.py verify
```

The command re-runs every scorer on scratch copies and compares each published file leaf by leaf. It writes nothing into
`testbed/dataset/` or `testbed/results/`, and checks afterwards that no file there changed. Its latest full run, on
2026-10-28 (09:54–10:49, from the project folder), gave **277 checks, 0 failing**
(`testbed/results/verify_full_20261028b.log`); the release folder, rebuilt with every study, gave on its own on 2026-10-28 (13:04–13:58)
**282 checks, 0 failing** (`testbed/results/verify_release_20261028_130213.log`). On 2026-10-10 at 04:30 it gave
**114 checks, 0 failing**: 106 that morning before the scheduler upgrades, and 83 on 2026-10-08, before the run-directory
family and the 10-08/10-09 studies:

| family | what is re-derived | how |
|---|---|---|
| dataset | `characterisation.csv` (126 rows) and `results.csv`, byte for byte | `build_dataset.py` on the 23 run dirs of `dataset/README.md`'s `Rebuild:` line |
| pool | the 14 `pool_results*.json` campaigns, cell by cell | `compare_pool.py` on each cell's own log, against its registration |
| energy | the 16 `energy_results_*.json` runs and the 5 `energy_pool_*/score.json` sessions | their scorers, on a copy of each run dir |
| twouser | both two-user results | `score_twouser.py` on its log |
| 0930 | the default-split, prima.cpp and QR3 results | their scorers in `.handoff/assets/`, with their output pointed at scratch |
| rundirs | the 15 scores written into their run directories: the ring laps, VISIT, the TCP diagnostic, the network study, the full evaluation's 4 summaries and 2 scores, the energy head-to-head, the correction layer (upgrade 2), and upgrade 1's pass time, acceptance and served test | each scorer on a copy of its run dir, with the files it writes removed first (`full_eval.py summary`, then `score_full_eval.py`) |
| regs | 57 registrations (`*predictions*.json`, `full_eval_*.json`, `rule_*.json`): read-only, and every quoted sha256 resolves | sha256 |

**Three things it found on its first run (2026-10-08), all fixed:**
- **The dataset README's rebuild command was incomplete.** It listed the 10 run dirs of 09-28 (62 rows). The 13 dirs added
  on 09-29 were never added. The full, ordered command was recovered from the published files and now rebuilds both csvs
  byte for byte.
- **`pool_results_20261008loadmode.json` predated a documented scorer fix.** Under `-lm none` the Pi 4's buffer read 0. It
  was re-scored from the same two logs, and that one field changed, to 6,089.46 MiB.
- **The prima.cpp scorer needed NumPy < 2.** It read the GGUF vocabulary with prima.cpp's vendored `gguf-py`. It now uses
  the pinned llama.cpp's `gguf-py`, and its result is unchanged.

**What the run-directory family found on its first run (2026-10-10):**
- Every score matched, apart from three figures of the energy head-to-head (`energy_pool_20261008_011447/h2h_score.json`).
- **Why they differ:** that scorer prices its model with today's `link.json`, and the Pi 4 <-> Jetson pair was re-measured
  after it scored (10-08 18:05, the new router). Its model figures moved by up to 0.5 points, and its ACC worst case went
  from −6.7 to −7.2 %.
- **What is still compared:** every verdict, which is unchanged, and every measured figure.
- **What is not:** `model_gross`, `model_err_pct` and `worst_pct`, and the check says so on its line.
- **Not compared anywhere:** `run` (the copy's path) and `code_changed_since_registration` (it depends on when a run is
  scored: `ring_laps.sh` changed on 10-09, after the ring laps ran).

A `PUBLISHED-ONLY` note is not a failure. It marks a field that a wrapper scorer added, for example the 09-30 default
split's device list, which `score_default.py` writes and which itself re-derives exactly.

`--only dataset,pool` limits the families. `--keep DIR` keeps the scratch copies.

## 2. Re-run an experiment on the boards

### The pins

| what | pinned to | where it is checked |
|---|---|---|
| llama.cpp | `b23701f7` on every board; the Jetson's CUDA build adds `testbed/patches/jetpack46-cuda102-b23701f.patch` | the four pin locations; the ThinkPad, Pi 5 and Pi 4 suites fail on undeclared divergence (`Pi5/docs/PIN_CONFLICT.md`) |
| models | upstream revision and sha256 per file | `testbed/models.md`; every runner hashes the file on its host before a figure counts |
| the cluster | `ThinkPad/config/cluster.json` | `cd ThinkPad && python3 -m unittest discover -s tests -q` |
| each campaign | a registration written before it ran: plans, predictions, hypotheses and the sha256 of its inputs and code, read-only | `reproduce.py verify` (regs) |

### The protocol, as the house rules have it

- **At least 3 repetitions;** quote the spread across runs, never llama-bench's within-run `±` at `-r 1`.
- **Tee every run's raw output to a file.**
- **Poll every Pi's `get_throttled` at 1 Hz in every cell** (`throttle_poll.sh` v2, inside `run_pool.sh` and `full_eval.py`).
  The verdict comes from `throttle_summary.py`.
- **The Jetson's power guard is started before any Jetson load and stopped after.** Its procedure is in
  `.handoff/nodes/Jetson/`: clock synchronized, the script's sha256 matching the repo's copy.

### The runners

| experiment | command |
|---|---|
| a pooled cell | `CELL=<id> PRED=<registration> bash testbed/scripts/run_pool.sh`, then `compare_pool.py --pred … --out … <logs>` |
| serving, and the full evaluation (TPOT percentiles with a two-level bootstrap, energy per placement) | `full_eval.py register …`, `run …`, `summary …`, then `score_full_eval.py <run dir>` (the pilot: `.handoff/assets/full-eval-pilot-2026-10-08/run_pilot.sh`; Qwen3-14B: `.handoff/assets/full-eval-14b-2026-10-09/run_14b.sh`). Since 2026-10-09 `register` refuses a placement that does not fit at rest, or that fits inside the engine's 256 MiB headroom unless `--below-headroom` is given and recorded |
| energy per token, one board | `energy_phase_run.py` with its registration, then `score_energy_phase_run.py` |
| energy, pooled | `energy_pool_run.py`, then `score_energy_pool_run.py` |
| the network study | `register_netshape_20261008.py`, `netshape_campaign.sh`, then `score_netshape_20261008.py` |
| the TCP diagnostic | `register_tcpdiag_20261009.py`, `tcpdiag_campaign_20261009.sh`, then `score_tcpdiag_20261009.py <run dir>` |
| the ring laps, and the VISIT test | `register_ring_laps_20261009.py` or `register_visit_20261009.py`; `bash testbed/scripts/ring_laps.sh <registration>`; then `score_ring_laps_20261009.py` or `score_visit_20261009.py <run dir>` |
| a link between two boards | `PAIR="pi4 jetson" bash testbed/scripts/measure_link.sh`, then `link_to_json.py <logs>`; since 2026-10-09 the receiver times each transfer (`RECV` lines), and the sender's figure is kept beside it |
| speculative decoding (scheduler upgrade 1) | the pass time of a k-token check: `register_spec_pass_20261010.py`, `spec_pass_campaign_20261010.sh` (`run_pool.sh` with `bench.tool` `llama-batched-bench`), `score_spec_pass_20261010.py`; the draft's acceptance on the laptop: `register_spec_accept_20261010.py`, `spec_accept_20261010.py run` / `score`; served on the boards: `register_spec_serve_20261010.py --accept <step 2 dir>`, `spec_serve_20261010.py run` / `score` (`serve_pool.sh` with `SERVE_DRAFT`, `DRAFT_HOST_MIB`, `SERVE_RPC_EXTRA`) |
| prefill/decode hand-off | `pd_handoff.py sweep`, `pd_twouser.py run` |

### What cannot be re-run as it was

- **The Pi 5 was returned on 2026-10-05.** Its cells re-score from their logs, but cannot be run again until a replacement
  is characterised.
- **prima.cpp's head-to-head of 09-30 needs the Pi 5.** prima.cpp does not build on the Jetson.
- **Energy needs the Shelly strip** and its password file, which is kept outside the repo (`energy.md`). It also needs its
  DHCP address, which moves until it is reserved.
- **The Jetson's SD card is the suspect in three hangs.** Its guard stays down outside runs.

## 3. Use the data elsewhere

```bash
python3 testbed/scripts/reproduce.py export     # testbed/dataset/export/pool_cells.csv, all_figures.csv
python3 testbed/scripts/reproduce.py manifest   # testbed/dataset/MANIFEST.json
```

- **`pool_cells.csv`:** one row per scored pool cell. Each row carries:
  - the campaign, its registration and the registration's sha256;
  - the model, quantisation, pipeline, head, host, llama.cpp commit, repetitions and protocol;
  - measured and predicted decode and prefill, with the error;
  - the exit status;
  - the thermal verdict and the poll's peak Pi temperature (from 2026-10-07);
  - every board's temperature before and after the cell, and the Pis' `throttled` after it;
  - the page-in flag, the memory drops, the start time and the raw log.
- **`all_figures.csv`:** every figure of every results file, one per row, as file, JSON path and value.
- **`MANIFEST.json`:** the sha256 and size of every file in `testbed/dataset/`, of every raw input a result names, and of
  every file in the run directories `verify` re-scores (736 files on 2026-10-10 at 04:30; 704 that morning; 523 on
  2026-10-08).

## Environment notes (2026-10-08)

- **python-docx** (the progress-report tools) is installed only in Python 3.12's user site, while the system Python is now
  3.14. Run those tools with `PYTHONPATH=/home/aun/.local/lib/python3.12/site-packages`.
- **NumPy is 2.3 on the laptop.** Use the pinned llama.cpp's `gguf-py`, not prima.cpp's.
