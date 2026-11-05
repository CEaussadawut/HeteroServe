# Pi 4 node — HeteroServe testbed

Everything needed to bring the **Raspberry Pi 4 Model B (8 GB, Cortex-A72)** into the
HeteroServe measurement cluster and keep the data it produces trustworthy.

## This node's job

Per the proposal, the Pi 4 is the **deliberate straggler**: the weak worker kept in the
cluster on purpose so the study can answer whether a markedly slower node should be given
reduced load or dropped entirely (research question QR8).

It has three duties:

1. contribute its rows to the **device-characterisation dataset** — the project's first
   deliverable — by being benchmarked in isolation, prefill and decode, at each
   quantisation, with energy measured on the same time base;
2. serve layer shards over **llama.cpp `rpc-server`** so the cluster runs models no single
   node can hold;
3. be the **slow node** in the placement experiments.

It is not meant to be fast. Do not tune it beyond the fixed, documented state in
`scripts/01_system_tune.sh` — a Pi 4 that has been quietly optimised stops being the
straggler the study is about.

Its defining hardware fact: the Cortex-A72 has NEON but **no dotprod**, the
instruction the Pi 5's Cortex-A76 uses for exactly these quantised kernels.
(Neither chip has i8mm — that is ARMv8.6, Cortex-A710 and later — so dotprod alone
carries the Pi4/Pi5 ISA contrast.) That gap, not clock speed, is why this node is
slow at this workload — and `node/device_info.py` records
it as evidence rather than leaving it an assertion.

## Layout

```
Pi4/
├── config/
│   ├── node.env             single source of truth: identity, addresses, build flags,
│   │                        rpc memory, INA226 channel. Sourced by bash AND python.
│   └── models.txt           small GGUF models for isolated characterisation
├── scripts/
│   ├── lib.sh               shared helpers: logging, phase markers, state assertions
│   ├── 00_bootstrap.sh      one-time OS bootstrap (apt, chrony, dirs, hostname)
│   ├── 01_system_tune.sh    the fixed measurement state (governor, swap off, link)
│   ├── 02_build_llama_cpp.sh  pinned build for Cortex-A72 + build_manifest.json
│   ├── 03_fetch_models.sh   download + checksum the GGUF models
│   ├── 04_rpc_server.sh     start/stop/status the cluster worker
│   ├── 05_net_tier.sh       gbe | wifi | netem shaping | clear | verify
│   ├── 06_bench_local.sh    Phase 2 characterisation sweep
│   ├── 07_rpc_smoke.sh      prove the control node can run graphs here
│   └── 99_healthcheck.sh    readiness gate — run before EVERY measurement
├── node/
│   ├── envfile.py           dependency-free node.env parser
│   ├── device_info.py       device inventory (scope item 1) as JSON
│   ├── stamp_lines.py       stamps llama-bench's --progress stderr with CLOCK_REALTIME
│   ├── bench_windows.py     per-repetition energy windows, behind 8 validation gates
│   ├── bench_parse.py       llama-bench JSON → this node's results.csv rows
│   ├── bench_validate.py    independent R=1,3,5 cross-check of the energy window
│   └── node_agent.py        HTTP agent: /info /health /metrics /events /rpc/* /mark
├── systemd/                 units for rpc-server, the agent, and the boot-time tune
├── docs/                    01 OS setup · 02 power · 03 build · 04 protocol · 05 troubleshooting
│   └── CHECKLIST.md         printable bring-up and per-run checklist
└── Makefile                 `make help`
```

## Bring-up

Copy this folder to the Pi and run, in order:

```bash
make bootstrap   # sudo: apt, toolchain, chrony → control node, /opt/heteroserve
make tune        # sudo: governor=performance, swap off, GbE forced, services quiet
make build       # llama.cpp for Cortex-A72 → prints the SHA to pin
make models      # small GGUF models
make install     # sudo: systemd units + sudoers
make health      # must print READY
make info        # save the device inventory
make bench-quick # smoke-test the whole chain before the long sweep
```

Full detail in `docs/01_os_setup.md`; the printable version is `docs/CHECKLIST.md`.

**After the first build, copy the printed commit SHA into `LLAMA_PIN_SHA` in
`config/node.env`** — on this node and every other one. The healthcheck fails until you do.
A characterisation dataset gathered across two llama.cpp revisions cannot be compared with
itself.

## Daily use

```bash
make health                          # the gate
make bench                           # full Phase 2 sweep (hours)
make rpc                             # DEBUG ONLY -- see below; not a measurement path
sudo scripts/05_net_tier.sh shape 100 1.0  # one point of the interconnect sweep
make collect                         # the rsync line to pull data to the control node
```

From the control node, over the agent:

```bash
curl  http://pi4:9410/health          # 200 ready / 503 not ready, with reasons
curl  http://pi4:9410/info            # full inventory
curl -XPOST http://pi4:9410/rpc/start
curl -XPOST http://pi4:9410/mark -d '{"event":"cluster_run_begin","note":"run-42"}'
```

## Contract with the control node

| Direction | What |
|---|---|
| Pi 4 → control | `inventory/device_info.json` (device characteristics), `bench/<RUN_ID>/results.csv` (throughput rows with wall-clock windows), `logs/events.jsonl` (phase markers) |
| control → Pi 4 | NTP (the shared time base), RPC client traffic, agent calls to start/stop the worker and inject markers |
| DAQ → control | INA226 channel 2 power stream for this node's 5 V rail |

`results.csv` deliberately leaves `energy_j`, `joules_per_token` and `mean_power_w` empty:
the control node's collector fills them by integrating channel 2 over each row's
`[t_start, t_end]` window. One collector owns all four channels on one clock, so energy is
never estimated locally and never on a second time base.

**A collector MUST read `energy_window` before pooling rows.** It says which window
`t_start`/`t_end` are:

- `reps-span` — the repetition span. This is the normal case and the one the metric is
  defined over.
- `process` — a validation gate refused the derivation, so `t_start`/`t_end` fall back to
  the **whole `llama-bench` process**, including model load and a full-length warmup
  prefill. Integrating one of these yields a joules-per-token tens of percent too high.
  **Do not average `process` rows together with `reps-span` rows.** The row also carries a
  `energy-span:whole-process:<gate>` entry in `flags` naming which gate refused.

`t_proc_start`/`t_proc_end` always carry the raw process window for diagnostics, and
`energy_duty` states what fraction of the reported window actually produced tokens.

## Two things that will silently ruin the data

1. **Clock drift.** Energy is attributed to prefill or decode purely by timestamp. Over a
   few milliseconds of offset the attribution is wrong while every number still looks
   plausible. The healthcheck treats chrony sync as blocking.
2. **A forgotten `tc` rule.** netem shaping survives until removed and quietly degrades the
   *next* experiment. Always `scripts/05_net_tier.sh clear`. Both readiness gates check for
   it — `make health` and the agent's `/health`, which is the one the control node actually
   polls before a cluster run. Neither refuses shaping outright: the interconnect sweep
   shapes on purpose, so they compare the live qdisc against the tier last declared through
   `scripts/05_net_tier.sh` and fail only on a mismatch. Shaping applied by hand, without
   going through that script, reads as undeclared and blocks.

## Expected numbers

Sanity bands, not targets — far outside means something is misconfigured:

| | |
|---|---|
| Idle power (fan + PHY included) | ~2.7–3.5 W |
| Power under decode | ~5.5–6.5 W |
| Usable RAM | **7.20 GiB (7373 MiB) of 8 GB** — *measured 2026-09-20, swap off* |
| layer budget (`NODE_USABLE_MB`) | 2600 MB — **sized for a 4 GB board; now stale** (see note) |
| Largest model resident alone | ~7B at Q4_K_M on the real 8 GB board |
| Build time, `-j2` | 25–40 min |

> **The board is 8 GB, not 4 GB.** Revision `d03115`, measured MemAvailable 7373 MiB.
> The budget chain above was sized when it was believed to be 4 GB, so this node is
> currently advertised at roughly a third of its real capacity — which matters now
> that it is one half of the 14B pooling experiment.
>
> **Deliberately not changed here**, because three values move together:
> `NODE_USABLE_MB` / `RPC_MEM_MB` in `config/node.env`, `usable_mem_mb` in
> `ThinkPad/config/cluster.json`, and `MemoryMax=3G` / `MemoryHigh=2800M` in
> `systemd/heteroserve-rpc.service`. `placement.py` subtracts runtime overhead from
> the cluster.json figure, so raising that alone would let the engine assign layers
> the service would then OOM on. Settle all three together.
>
> See `.handoff/nodes/Pi4/2026-09-20-1530-NOTE-pin-vs-measured-revision.md` (repo root).
