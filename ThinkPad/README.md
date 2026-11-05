# Control node — ThinkPad T580

The cluster's control plane and its measurement instrumentation: cluster manager, placement
engine, benchmark harness and energy collector, plus the Pico 2 DAQ firmware.

Per the proposal, this software is **the instrument, not the product**. It exists so the
experiments can produce a device-characterisation dataset, a placement rule derived from
measured data, and reproducible results.

## What runs here

| Component | Where | What it does |
|---|---|---|
| Cluster manager | `control/nodes.py` | device inventory, node roles, the readiness gate |
| Placement engine | `control/placement.py` | the layer-split rule, its baselines, the Pareto sweep |
| Benchmark harness | `control/harness.py`, `control/experiment.py` | runs placements, records windows, sweeps experiments |
| Energy collector | `control/energy.py`, `firmware/pico2/` | four INA226 channels on one time base |
| Analysis | `control/analysis.py`, `control/report.py` | the energy join, tables, the Pareto front |
| CLI | `control/cli.py` | `heteroctl` |

This node is also the cluster's **NTP server**. Every joules-per-token figure depends on the
workers agreeing with its clock.

## Layout

```
ThinkPad/
├── config/
│   ├── cluster.json          nodes, roles, addresses, DAQ channels, link tiers, paths
│   ├── models.txt            the GGUFs the cluster serves (7B/14B live only here)
│   └── experiments/          declarative sweep specs - the reproducible form
├── control/                  the Python package (installed as `heteroctl`)
│   ├── config.py             loads and cross-validates cluster.json
│   ├── nodes.py              agent client, fan-out, preflight gate
│   ├── gguf.py               reads model geometry from the GGUF header
│   ├── profiles.py           measured data -> the rule's inputs (refuses to invent any)
│   ├── placement.py          the rule, the baselines, the cost model
│   ├── harness.py            one run: markers, execution, timings, windows
│   ├── experiment.py         sweeps, and how well the rule predicted reality
│   ├── energy.py             DAQ link, clock alignment, calibration, recorder
│   ├── analysis.py           E = integral P dt, the join, and its refusals
│   ├── netem.py              link tiers across the cluster, and link surveys
│   ├── report.py             markdown tables, Pareto CSV, optional plots
│   └── cli.py                heteroctl
├── firmware/pico2/           MicroPython DAQ: main.py + ina226.py
├── scripts/                  bootstrap, build, models, deploy, flash, calibrate, preflight
├── systemd/                  collector unit
├── calibration/              ch*.json - measured, versioned WITH the data
├── tests/                    unit tests, no hardware required
└── docs/                     01 setup · 02 operations · 03 energy · 04 placement ·
                              05 protocol · 06 troubleshooting · CHECKLIST
```

## Bring-up

```bash
make bootstrap   # apt, venv, heteroctl, dialout, chrony as the cluster's NTP server
make build       # llama.cpp, AVX2 + FMA + RPC        -> pin the printed SHA
make models      # 7B/14B GGUFs + checksums + shapes
make deploy      # push Pi4/, Pi5/, Jetson/ ... to their nodes
make flash-daq   # DAQ firmware onto the Pico 2
make test        # unit tests, no hardware
make preflight   # cluster + DAQ + calibration gate
```

Details in `docs/01_setup.md`; the printable version is `docs/CHECKLIST.md`.

## A measurement session

```bash
make preflight
heteroctl daq record &                 # or the systemd unit
heteroctl rpc up
heteroctl idle-baseline --seconds 120
heteroctl sweep config/experiments/placement_vs_even_7b.json
heteroctl tier clear && heteroctl rpc down
```

Then join energy and build the tables:

```bash
heteroctl join ./data/pi4/bench/<RUN_ID> --power /opt/heteroserve/data/power/power_*.csv
heteroctl power-summary --power ... --events ... --out power_summary.json
heteroctl report characterisation results_energy.csv --out characterisation.md
heteroctl report sweep /opt/heteroserve/data/sweeps/<ID>/sweep.json --out sweep.md
```

## The placement rule in one screen

```
minimise  T + lambda*E      subject to  mem_i(x) <= M_i

T = sum_i n_i*c_i + C                      C = per-hop activation transfer + RTT
E = sum_i [ n_i*c_i*P_act_i + (T - n_i*c_i)*P_idle_i ]
```

Every input is measured: `c_i` from the Phase 2 isolated sweeps, `P_act/P_idle` from the
INA226 channels, bandwidth and RTT from iperf3 and ping, memory from each node's agent.
`profiles.build_devices()` **refuses** to produce a profile from anything unmeasured, because
a placement computed from invented numbers looks exactly like one computed from real ones.

Substituting `T` into `E` makes the objective linear in the layer counts, so for a fixed
participating set the optimum needs no search; only the set and its order are enumerated.
Full derivation, limits and the realisation to `--tensor-split`: `docs/04_placement_engine.md`.

```bash
heteroctl compare models/qwen2.5-7b-q4km.gguf    # rule vs even vs single vs straggler
heteroctl pareto  models/qwen2.5-7b-q4km.gguf --out front.csv
```

## Three things that silently ruin a dataset

1. **Two llama.cpp revisions.** Preflight compares each node's build against
   `llama.pin_sha` and blocks on a mismatch.
2. **Clock drift.** Energy is attributed to prefill or decode purely by timestamp. Offsets
   above 5 ms are blocking; the DAQ's own offset is measured per session and stored beside
   the trace.
3. **A forgotten `tc` rule.** netem shaping outlives the run that set it and degrades the
   *next* one. Always `heteroctl tier clear`; preflight looks for leftovers.

## What the analysis refuses to do

- report energy for a window with fewer than 8 samples, or with a DAQ gap
- report marginal energy without a marked idle baseline
- claim ±5 % accuracy for an uncalibrated channel
- average rows that carry validity flags into a device profile

Each refusal leaves the throughput number intact and drops only the part that is not
supported. A missing number is recoverable; a plausible wrong one is not.

## Contract with the worker nodes

| Direction | What |
|---|---|
| control → node | NTP, RPC client traffic, agent calls (`/health`, `/info`, `/rpc/*`, `/mark`), tier changes over ssh |
| node → control | `device_info.json`, `results.csv` with wall-clock windows, `events.jsonl` markers |
| DAQ → control | four INA226 channels, timestamped on the Pico, mapped to host UTC |

Node ids are the join key everywhere — `cluster.json`, each node's `node.env`, the DAQ
channel map and the power CSV must all agree. `control/config.py` cross-checks the parts it
can see and refuses to load when they disagree.
