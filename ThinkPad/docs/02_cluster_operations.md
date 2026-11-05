# 02 — Operating the cluster

Day-to-day commands. Everything runs from `ThinkPad/` with the venv active (or via
`make`, which uses `.venv/bin/python` directly).

## The command surface

```bash
heteroctl status                    # what the cluster is, and live temps/clocks/rpc state
heteroctl preflight [--save]        # the readiness gate; exit 1 when blocked
heteroctl inventory --save          # device_info from every agent -> cluster_inventory.json
heteroctl rpc up | down             # rpc-server everywhere, via each node's agent
heteroctl tier gbe|wifi|shaped-100|clear|verify
heteroctl survey --save             # iperf3 + ping to every node
heteroctl daq sync|status|record
heteroctl calibrate --channel 2 --point 0.812:0.806 --point 1.204:1.197
heteroctl idle-baseline --seconds 120
heteroctl profiles --results ... --power ... --links ... --layers ...
heteroctl plan|compare|pareto MODEL.gguf
heteroctl run MODEL.gguf --strategy rule --tier gbe
heteroctl sweep config/experiments/placement_vs_even_7b.json
heteroctl join RUN_DIR --power power_*.csv
heteroctl power-summary --power ... --events ...
heteroctl report characterisation results_energy.csv --out table.md
heteroctl report sweep sweep.json --out sweep.md
```

`python -m control.cli …` is the same entry point if the console script is not on PATH.

## A normal measurement session

```bash
make preflight                       # cluster + DAQ + calibration, one gate
heteroctl daq record &               # or leave the systemd unit running
heteroctl rpc up
heteroctl idle-baseline --seconds 120
heteroctl sweep config/experiments/placement_vs_even_7b.json
heteroctl tier clear                 # always, even if the sweep failed
heteroctl rpc down
```

The sweep does preflight, idle baseline, tier changes, link re-surveys and cooldowns
itself — running it is meant to be one command, because a sweep assembled by hand is not
reproducible and reproducibility is a deliverable.

## What the manager guarantees

`heteroctl preflight` blocks a run when any node reports throttling, a wrong governor,
swap enabled, an over-temperature board, a clock offset above 5 ms, an unpinned or
mismatched llama.cpp revision, or leftover netem shaping. The harness refuses to start on a
block, and `--skip-preflight` exists only for bring-up — rows produced that way must be
labelled unverified.

The pin check is worth restating: **one llama.cpp commit across the whole cluster**. The
control node's `llama.pin_sha` is compared against each node's reported build. Two
revisions in one dataset is two half-datasets.

## Link tiers

```bash
heteroctl tier shaped-100      # htb rate + netem delay on every node
heteroctl survey --save        # measure what the link ACTUALLY is
heteroctl tier verify          # what qdisc is really in force
heteroctl tier clear
```

Two things this enforces, because neither announces itself in the data:

1. netem shapes **egress only**, so a tier is applied to every participating node, not just
   one end.
2. Shaping **outlives** the experiment that set it. Always clear; preflight looks for
   leftovers.

Re-survey after every tier change and rebuild the device profiles from the survey. The rule
must be fed the link that exists, not the tier that was requested — a shaped 100 Mbit link
that negotiated half-duplex is a different link, and only a measurement can tell.

## Data layout

```
/opt/heteroserve/data/
├── inventory/      cluster_inventory.json, device_profiles.json, link_survey.json,
│                   power_summary.json, control_build_manifest.json, model_sha256.txt
├── power/          power_<session>.csv + power_<session>.sync.json
├── runs/<RUN_ID>/  record.json, llama.log, windows.jsonl
├── sweeps/<ID>/    spec.json, preflight.json, sweep.json
└── logs/           collector.log
```

Per-node benchmark output stays on the node under its own `data/bench/` until pulled:

```bash
rsync -av pi4:/opt/heteroserve/data/ ./data/pi4/
```

## Roles and the control node as a worker

`serves_rpc` in `cluster.json` decides whether a node offers layers. It is `false` for the
T580 by default — the control node coordinates and measures. Set it `true` to make the
T580 a compute node as well, which is how the prefill-on-the-strong-node disaggregation
experiment is expressed. Say so in the write-up when you do: the T580 is not on the DAQ, so
a placement that uses it produces latency numbers with an incomplete energy picture.

## Enabling the stretch-scope RISC-V node

`orangepi-rv2` ships `"enabled": false`. Flip it to `true` once the main cluster produces
stable measurements, deploy its folder, and re-run `heteroctl inventory --save`. Everything
else — planning, sweeps, the energy join — picks it up with no further changes.
