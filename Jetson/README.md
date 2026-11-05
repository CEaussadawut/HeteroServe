# HeteroServe — Jetson Nano node

Everything needed to turn a Jetson Nano developer kit into a working node of the
HeteroServe testbed, and into a measurement point that produces energy-per-token
data rather than just tokens per second.

The testbed is the **instrument** for the study
*"Characterizing Prefill–Decode Placement and Energy per Token for Distributed
LLM Inference on a Heterogeneous Low-Cost Edge Cluster"*. Nothing in this folder
is a product; every script exists to make a measurement reproducible.

## This node's role

| | |
|---|---|
| Device | Jetson Nano developer kit (B01), 4 GB |
| SoC | Tegra X1 — 4× Cortex-A57 @ 1.43 GHz + 128-core Maxwell (GM20B, sm_53) |
| Memory | 4 GB LPDDR4 **shared** between CPU and GPU (~25.6 GB/s) |
| Role in the cluster | CUDA worker behind `ggml` RPC; the only node with a GPU |
| Power envelope | 5 W (nvpmodel 1) or 10 W (nvpmodel 0, needs barrel jack + J48) |
| Instrumented by | carrier-board INA3221 via `tegrastats` (whole-devkit 5 V input + SoC rails) |

Three properties of this device drive most of the experimental design:

1. **FP16 yes, INT8 no.** Maxwell has real half-precision throughput but no
   useful `dp4a` integer path. So the quantisation that is cheapest on the Pi 5
   is not automatically cheapest here, and the Q4_K dequantisation cost is a term
   worth measuring rather than assuming.
2. **CPU and GPU share one memory pool and one bandwidth budget.** Offloading a
   layer to the GPU does not free memory bandwidth, it just moves who is using
   it. This is why decode (bandwidth-bound) is expected to gain far less from
   `-ngl` than prefill (compute-bound) does — and measuring that gap on this
   device is one of the study's results.
3. **4 GB total, minus the OS.** The node cannot hold much of a model on its
   own, which is exactly why it is a *partition holder* in a split run rather
   than a standalone server.

## Quick start

On a freshly flashed Nano:

```bash
git clone <your repo> ~/HeteroServe && cd ~/HeteroServe/Jetson
cp config/node.env.example config/node.env
$EDITOR config/node.env          # set NODE_ID, NVP_MODEL, paths

make prep                        # headless, swap, clocks, packages   (sudo, reboots recommended)
make toolchain                   # cmake >= 3.22, gcc-8, python deps
make build                       # llama.cpp with GGML_RPC (+CUDA if it compiles)
make models                      # GGUFs from config/models.tsv
make facts                       # this node's characterisation record -> results/
```

Then, per experiment:

```bash
make bench                       # single-node throughput + energy sweep (~32 runs, 2-3.5 h)
make rpc                         # serve this node to the orchestrator
make smoke                       # cluster reachability + two-node inference
make collect                     # tarball for the analysis machine
```

Run long sweeps inside `tmux`. An ssh drop otherwise kills a two-hour run.

## Layout

```
Jetson/
├── Makefile                     ordered entry points
├── config/
│   ├── node.env.example         every knob; copy to node.env
│   ├── cluster.hosts.example    testbed inventory
│   └── models.tsv               GGUF manifest + expected sha256 (same on all four nodes)
├── scripts/
│   ├── 00_system_prep.sh        headless, swap, nvpmodel, jetson_clocks, fan, chrony
│   ├── 10_install_toolchain.sh  cmake/gcc-8/python
│   ├── 20_build_llama_cpp.sh    RPC + CUDA build, writes build_info.json
│   ├── 30_fetch_models.sh       resumable download + sha256
│   ├── 40_run_rpc_server.sh     the worker
│   ├── 45_capture_window.sh     phase-marked power capture for an orchestrator-driven run
│   ├── 50_bench_device.sh       pre-flight checks, then the sweep
│   ├── 60_collect_node_facts.sh device characterisation record (JSON)
│   ├── 70_cluster_smoke.sh      ping / iperf3 / RPC port / two-node run
│   └── 90_collect_results.sh    tarball with full provenance
├── telemetry/
│   ├── tegrastats_logger.py     tegrastats -> CSV with running energy integral
│   └── energy_window.py         per-window joules, joined to llama-bench
├── bench/
│   ├── matrix.json              the sweep definition (never mixes prefill and decode)
│   ├── run_matrix.py            driver: one row per phase, throughput and watts together
│   ├── emit_results_csv.py      projection into the cluster's shared 40-column schema
│   └── check_finding3_invariants.py   asserts no row mixes two phases
├── systemd/                     optional always-on worker + logger units
└── docs/
    ├── 00_flash_jetpack.md      imaging and first boot
    ├── 10_measurement_protocol.md   how to take a number you can defend
    ├── 20_troubleshooting.md     the failures this hardware actually produces
    └── 30_experiment_log.md      lab-notebook template
```

## The one thing that will bite you

Stock **JetPack 4.6.x ships CUDA 10.2**, and current llama.cpp targets CUDA ≥ 11.
The CUDA build of `master` will very likely not compile on this board.
`20_build_llama_cpp.sh` detects this and tells you your three options: pin an
older llama.cpp ref, build CPU-only, or move off the stock image. Whichever you
choose, **record it in `config/node.env` (`LLAMA_CPP_REF`)** — `build_info.json`
then carries it into every result file, and the sweep stays reproducible.

Building CPU-only is not a failure. The Nano as an ARM CPU worker is a
legitimate experimental condition, and the CPU-vs-GPU comparison on the *same*
device is one of the measurements the study wants.

## Measurement honesty

- `POM_5V_IN` is the **whole dev kit's 5 V input**, not module-only power. On the
  B01 the INA3221 lives on the carrier board, so the figure includes carrier, USB
  and HDMI draw; only wall-adapter conversion loss is outside it. (Module-only is
  a TX2/Xavier property — see `docs/10_measurement_protocol.md` §2.) Keep the
  peripheral set identical across runs so that offset stays constant.
- **"All four nodes ran the same bytes" is only proven for models whose 4th
  column in `config/models.tsv` holds a real sha256.** Where it is still `-`,
  `SHA256SUMS` records what was downloaded — which proves the file has not
  changed since, not that it was ever the right file. The fetcher says so rather
  than letting the check look authoritative. Fill the column in once from a
  download you trust.
- **The shared `results.csv` carries this node's timing, not its energy.** Its
  energy columns are blank by design — the control node's integrator cannot read
  a tegrastats power stream, assumes a 28 Hz sample rate, and uses a denominator
  that omits the warm-up this node's window contains. The real energy stays in
  `<base>_bench_energy.csv`. See `docs/10_measurement_protocol.md` §4.
- **There is no external power meter, so there is no across-device comparison.**
  The Pis have no on-board telemetry and nothing here reads an inline meter.
  Energy results are within-device only; cross-node joules-per-token is future
  work, not a number this kit can produce.
- **No power window ever contains more than one phase.** The matrix ships
  prefill-only and decode-only cases, interleaved so the two halves of a pair are
  measured at nearly the same board temperature. A window that produced two
  `llama-bench` tests gets `joules` but no `joules_per_token`, because that energy
  cannot be attributed to either phase. The one deliberately fused case reports
  `joules_per_request` instead.
- Every energy figure is reported **twice**: total joules, and joules net of the
  measured idle floor. On a 10 W board the ~2 W idle floor dominates a short
  decode window, and quoting only one of the two numbers is misleading.
- `jetson_clocks` is mandatory before any timed run. Without it the governor
  wanders and repetitions are not comparable.
- Wired GbE and Wi-Fi are separate, labelled conditions (`--tag wired-gbe` /
  `--tag wifi`), never mixed into one table.

See `docs/10_measurement_protocol.md` for the full procedure.
