# 05 — Experiment protocol

What to run, in what order, and what each run is for. Phases follow the proposal's plan.

## Phase 2 — device characterisation (the first deliverable)

Every node benchmarked **alone**, before the cluster is assembled. Nothing downstream works
without this: the placement rule's `cᵢ`, `P_actᵢ` and `P_idleᵢ` all come from here.

On each node:

```bash
ssh pi4 'cd /opt/heteroserve/Pi4 && make health && make bench'
```

On the control node, while those run:

```bash
heteroctl daq record          # continuous, for the whole session
heteroctl idle-baseline --seconds 120
```

Then pull and join:

```bash
rsync -av pi4:/opt/heteroserve/data/bench/ ./data/pi4/bench/
heteroctl join ./data/pi4/bench/<RUN_ID> --power /opt/heteroserve/data/power/power_*.csv
heteroctl power-summary --power ... --events ./data/pi4/logs/events.jsonl --out power_summary.json
heteroctl report characterisation ./data/pi4/bench/<RUN_ID>/results_energy.csv --out characterisation.md
```

Do this for **one** node end to end before running the other three. Discovering a clock
skew or a mis-wired channel after four nodes have each done an eight-hour sweep is the
expensive way to learn it.

Coverage per node: every model in its list, Q4_K_M and Q8_0, prefill (128 and 512) and
decode (128), at 1, 2 and 4 threads, 5 repetitions.

## Phase 2b — links

```bash
iperf3 -s -D                       # on the control node
heteroctl tier gbe   && heteroctl survey --save
heteroctl tier wifi  && heteroctl survey --save
for t in shaped-10 shaped-25 shaped-50 shaped-100 shaped-300; do
  heteroctl tier $t && heteroctl survey --save
done
heteroctl tier clear
```

Record the Wi-Fi conditions — AP, band, channel, distance — and keep them fixed for the
whole project. A Wi-Fi tier that moves between runs is not a tier.

## Phase 3 — placement (the core contribution)

Build the profiles, then let the rule decide:

```bash
heteroctl profiles \
  --results all_nodes_results_energy.csv \
  --power   power_summary.json \
  --links   link_survey.json \
  --inventory cluster_inventory.json \
  --layers  '{"tinyllama-1.1b-q4km":22,"llama3.2-3b-q4km":28}'

heteroctl compare /opt/heteroserve/models/qwen2.5-7b-q4km.gguf --out placement.md
heteroctl sweep config/experiments/placement_vs_even_7b.json
heteroctl sweep config/experiments/straggler_study_7b.json
```

The experiment specs in `config/experiments/` are the reproducible form. Edit them rather
than passing ad-hoc flags — the spec file is what gets attached to the report and what
someone else re-runs.

| Spec | Question it answers |
|---|---|
| `placement_vs_even_7b` | does the rule beat an even split? (MVP ≥ 10 % TPOT) |
| `straggler_study_7b` | should the Pi 4 be under-loaded or dropped? (QR8) |
| `link_tier_sweep_7b` | below what link speed does distribution stop paying? |
| `memory_wall_14b` | the stretch target: a model only the pooled cluster can hold |
| `energy_pareto_7b` | the latency/energy trade-off, one λ per sweep |

## Phase 3b — prefill/decode disaggregation (exploratory)

The proposal is explicit that this capability does not exist upstream, so it is exploratory
scope: either emulate the hand-off or extend llama.cpp. Two things to do regardless:

- measure the KV-cache transfer cost directly (the analysis in Appendix B estimates ~15 MB
  and ~133 ms on GbE for a 7B at 256 tokens — confirm it rather than citing it);
- set `serves_rpc: true` on the T580 so the strong AVX2 node can take prefill, and note in
  the write-up that the T580 is not on the DAQ.

## Order of every individual run

1. `make preflight` — blocked means do not record.
2. Collector running, clock synced, calibration present.
3. Idle baseline, marked, cluster up and idle.
4. Set the tier explicitly; survey it.
5. Run, with cooldown between repetitions.
6. Clear shaping.
7. Join energy, check the flags, then look at the numbers.

Looking at the numbers before checking the flags is how a throttled run becomes a finding.

## What to record beyond the data

Every session, in the run log: date, llama.cpp SHA, tier and its measured bandwidth, DAQ
clock offset and spread, calibration file versions, ambient temperature, and anything
unusual. Most of it is captured automatically in the manifests and sync files — the log is
for what the machines cannot see.

## Reporting negative results

Two outcomes are likely, and both are results the proposal already predicts:

- distribution may **reduce** throughput, because a pipeline runs at the pace of its
  slowest stage;
- distribution may **increase** joules per token, because the interconnect and the idle
  cost of extra nodes are not free.

Report them with numbers. The contribution is the characterisation and the rule's response
to it, not a win. A study that only reports the cases where distribution helped would be
answering a question nobody asked.
