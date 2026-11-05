# Pi5 — HeteroServe strong CPU worker

Everything needed to take a stock Raspberry Pi 5 (8 GB) to a measurement-ready
HeteroServe worker node, and to run its slice of the Phase 2 device
characterisation sweep.

The Pi 5 is the **strong worker** in the testbed: Cortex-A76 at ARMv8.2-A with
dotprod and FP16, ~7 GB usable, the best memory bandwidth in the cluster (~17 GB/s
peak) and therefore the node best suited to bandwidth-bound decode. It is also the
only serving node that can hold a 7B Q4_K_M model on its own, which makes it the
*distribution baseline* every split placement is scored against. Its counterpart is
`../Pi4`, the deliberate straggler.

## Layout

| Path | What it is |
|---|---|
| `config/node.env` | the single source of truth; shell-sourceable and parsed by `node/envfile.py` |
| `config/models.txt` | GGUF models this node runs **in isolation** (sharded models arrive over RPC, never resident) |
| `config/cluster.hosts` | testbed inventory as seen from this node |
| `scripts/00`–`06` | bring-up, in order |
| `scripts/10_isolated_bench.sh` | the Phase 2 sweep |
| `scripts/11_link_survey.sh` | the interconnect survey -> `link_survey.json` for the placement engine |
| `node/` | the Python services: `envfile.py`, `agent.py`, `telemetry.py`, `gguf_meta.py` |
| `bench/` | sweep definition, the parser that produces `dataset.csv` **and** the shared-schema `results.csv`, and `link_report.py` |
| `tests/` | stdlib `unittest` suite for the dataset path; runs anywhere, no Pi needed |
| `systemd/` | the four units |
| `docs/` | `SETUP.md` (runbook), `NODE_CARD.md` (the numbers to quote) |
| `results/`, `telemetry/` | run output; nothing here is checked in |

## Bring-up, in order

```bash
sudo ./scripts/00_prepare_os.sh        # governor, fan, swap off, UTC, NTP -> control node
sudo ./scripts/01_install_deps.sh      # toolchain + /opt/heteroserve/venv
     ./scripts/02_build_llama_cpp.sh   # pinned llama.cpp, RPC on, explicit A76 flags
# LLAMA_PIN_SHA is already set (b23701f7, moved forward 2026-09-20) -- 02 checks that
# revision out; no paste-back step
     ./scripts/03_fetch_models.sh      # GGUF ladder up to 7B Q4_K_M
sudo ./scripts/05_network_setup.sh addr
sudo ./scripts/04_install_services.sh  # rpc-server + agent + telemetry + boot pins
     ./scripts/06_verify_node.sh       # acceptance gate — must exit 0
```

Then the sweep:

```bash
./scripts/10_isolated_bench.sh --reps 3
./bench/parse_bench.py /opt/heteroserve/results/pi5_isolated_<RUN_ID>
```

That writes two files into the run directory. `dataset.csv` is this node's own view — one
row per cell, and the only place `c_per_layer_ms` exists. `results.csv` is the shared
cross-node schema (Pi 4's 38 columns, one row per phase); copy it to the control node and
concatenate it with the other nodes' `results.csv` to build the characterisation table.
`tests/test_pi4_schema.py` reads Pi 4's `COLUMNS`, `QUANT_HINTS` and flag vocabulary live
and fails if the two drift apart.

Both files fail closed. `parse_bench.py` exits non-zero if any cell in `cells.csv`
produced no row (a missing, unparseable or empty llama-bench JSON), if any row lacks
`c_per_layer_ms`, and it warns on cells implying more than the SoC's ~17 GB/s peak memory
bandwidth — those did not read the weights from DRAM and the cell is void.

Then the interconnect survey, which produces the other measured input the placement
engine needs. Start a server on the peer first (`./scripts/11_link_survey.sh serve`):

```bash
./scripts/11_link_survey.sh              # this node -> orchestrator, over the active tier
./scripts/11_link_survey.sh --all        # every peer, for the pipeline hops
./scripts/11_link_survey.sh --tier 2.5gbe
```

The file it writes carries **this node's entry only**. Merge it into the control node's
`inventory/link_survey.json` rather than passing it straight to
`profiles.build_devices(link_json=...)` — that raises `ProfileError` for every other
worker, whose entries are absent rather than unmeasured.

It refuses to run while `tc netem` is attached: a survey taken through the Phase 4
shaping measures the shaping, not the link, and the result is indistinguishable from a
real measurement once it is in the JSON. Clear it with `05_network_setup.sh clear` first.

## Tests

Everything the dataset passes through is covered by a stdlib `unittest` suite. It needs
no Raspberry Pi, no GGUF and no network, so run it before deploying anything to the node:

```bash
python3 -m unittest discover -s tests -t .
```

It pins the two things that have already gone wrong silently once: the layer count behind
`c_per_layer_ms` (llama-bench emits none, so it comes from the GGUF header) and the
per-repetition spread behind the stddev columns. Fixtures are built from llama-bench's
real field list in `tests/llama_bench_fixture.py` — if upstream renames a field, that file
is where the disagreement should surface.

## The three things this node exists to keep honest

1. **One pinned llama.cpp revision.** `LLAMA_PIN_SHA` must be set, must match
   `build_manifest.json`, and must be the same SHA as the Pi 4 and the Jetson.
   `06_verify_node.sh` fails if it is empty or mismatched.
2. **No swap, ever.** A run that swaps is a corrupted measurement, not a slow one.
   Swap is off, `RPC_MEM_MB` (6144) is held below usable (7168), and the
   `heteroserve-rpc` unit sets `MemorySwapMax=0`.
3. **Flat thermal history.** Governor pinned to `performance`, active cooler pinned
   to full, a ≤55 °C start gate before *every* cell, and a cool-down after each.
   Any cell that ends ≥80 °C is flagged `throttled=1` in `dataset.csv` rather than
   averaged in silently (proposal risk R5). A cell whose temperature could not be read
   gets `throttled=""`, not `0` — `common.sh` falls back to `echo 0` on an unreadable
   thermal zone, so a dead sensor would otherwise publish a dataset asserting that no cell
   was throttled. The gate itself now fails closed too: `soc_temp_c()` returns `nan` rather
   than `0` on an unreadable thermal zone, and `assert_cool_enough` dies on it — a sweep that
   cannot evaluate its own start gate produces rows whose thermal validity nobody can
   establish afterwards, which is worse than not running.

## Known open items

- **Addressing — resolved and propagated.** The user settled it on 2026-09-13: Pi 5 =
  `.12`, Pi 4 = `.13`, each following its own `node.env`. Verified the same day across
  `Pi5/config/cluster.hosts`, `Pi4/config/node.env`, `ThinkPad/config/cluster.json` and
  `Jetson/config/cluster.hosts.example`.
- **INA226 channel map.** The Pi 4 claims channel 2 / `0x41`. This node is assigned
  channel 1 / `0x40`; the Jetson and Orange Pi are pencilled in at `0x44`/`0x45` in
  `config/node.env`. Confirm against the Pico 2 firmware when the DAQ is wired.
- **Interconnect — now measured, not assumed.** `11_link_survey.sh` writes the
  `link_survey.json` that `ThinkPad/control/profiles.py` reads. Until it existed nothing
  produced that file, so `profiles.py` fell through to `link_mbps=1000.0` / `rtt_ms=0.3`
  and every placement rested partly on a datasheet figure. Run it on each node before
  computing a placement; `build_devices(strict=True)` cannot tell a default from a
  measurement, so this is the node's responsibility.
- **The agent does not serve `POST /rpc/start` / `/rpc/stop`, and the control node
  drives them.** `ThinkPad/control/nodes.py:67-72` posts to both; this node answers 404.
  That is the recorded design decision — a node reconfigurable mid-sweep over HTTP is a
  node whose measurements cannot be trusted — not an oversight, and `tests/test_agent.py`
  pins it so. But the disagreement is real and belongs to the cluster, not to this node:
  either the control node stops driving those routes, or the decision is revisited for
  every worker at once. `GET /health`, `/metrics`, `/events` and `POST /mark` are served
  now, so the readiness gate and the energy join both work.
- **Pi 5 power above 3.2767 A is censored, not measured.** `config/node.env` declares
  `INA_MAX_A=5.0`, but the DAQ firmware hard-codes one 3.2768 A full scale for all four
  channels, so this node's current register clips instead of erroring. It biases the
  Pi4-vs-Pi5 energy gap in the Pi 5's favour. Fix belongs in
  `ThinkPad/firmware/pico2/`; until then treat a 3.2767 A sample as a ceiling hit.
- **Emitting the shared schema is necessary but NOT sufficient for this node to reach the
  placement rule.** `results.csv` now matches Pi 4's 38 columns exactly, but three things
  outside this folder still drop every Pi 5 row, each silently:
  `ThinkPad/control/profiles.py:90-92` deletes any row with a non-empty `flags` cell using
  a bare truthiness test and never prints the count it computes at `:113`;
  `ThinkPad/calibration/` holds only `ch2.json.example`, so `energy.py:96`'s strict
  `ch[0-9].json` glob finds nothing and `analysis.py:161-164` stamps `uncalibrated-channel`
  on every joined row of every node; and the hand-written `--layers` JSON (`cli.py:402`)
  covers Pi 4's four models, while four of this node's eight —
  `qwen2.5-0.5b-q4km`, `llama3.2-1b-q4km`, `qwen2.5-1.5b-q8`, `qwen2.5-7b-q4km` — appear in
  no Pi 4 file, and `profiles.py:96-98` skips a model-key miss with no counter and no message.
- **No per-phase energy window exists on this node.** `10_isolated_bench.sh` runs prefill and
  decode in one llama-bench process, so `t_start`/`t_end` are emitted empty with
  `energy_window=""` saying so in band; the raw process span is recorded in `t_proc_*` as
  epoch seconds, where it cannot feed the energy integral. Filling `t_start` would either
  double-count the cell's joules or, if left as the ISO-8601 string `cells.csv` holds, crash
  `heteroctl join` at `analysis.py:139`. The source-side fix is one invocation per phase,
  Pi 4 style.
- **`throttled_end` is now a real measurement.** `10_isolated_bench.sh` records the firmware
  word at both cell boundaries (`throttled_start`/`throttled_end` in `cells.csv`), and
  `parse_bench.py` applies the same delta rule Pi 4 does: the word latches bits 16-19 for
  the rest of the boot, so what invalidates a row is what changed *during* it, not that the
  board throttled at some point since boot. `common.sh`'s `throttled_word()` returns
  `unavailable` — never `0x0` — when `vcgencmd` cannot be read, so an unchecked cell is
  excluded rather than published as clean.
- **The cool-down before each cell is now a timed idle window.** `idle_start`/`idle_end` are
  epoch seconds on the DAQ's time base, so a marginal (above-idle) power figure can be
  computed per row rather than against a session-wide mean that pools a cold board with a
  hot one.
- **2.5GbE adapter.** `ETH25_IFACE=eth1` is assumed. Check the name the USB adapter
  actually enumerates as, or `05_network_setup.sh` will skip the tier.
- **`RPC_MEM_MB` IS ignored — settled 2026-09-20, measured not inferred.** `-m/--mem`
  was removed from `rpc-server` upstream on 2025-10-17 (`41386cf`), and the cluster pin
  is now `b23701f7` (2026-09-19), which is after that. Confirmed on the bench:

      ggml-rpc-server --help   # -> only -t -d -H -p -c -h

  So this node advertises its **actual free memory** (~7.6 GiB), not the deliberate
  6144 MB headroom. The pin is no longer chosen on the Jetson — that constraint
  inverted when the pin moved; see `docs/PIN_CONFLICT.md`.
