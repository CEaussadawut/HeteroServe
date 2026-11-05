# 06 — Troubleshooting the control node

## `heteroctl preflight` blocks and I need to run anyway

Don't, unless it is bring-up. Each blocking condition changes a measured number without
raising an error at the time:

| Block | What it silently does |
|---|---|
| throttling | changes the clock under the benchmark |
| governor not `performance` | makes prefill race the governor's ramp-up |
| swap enabled | turns an overload into microSD thrashing |
| clock offset > 5 ms | attributes energy to the wrong phase |
| revision mismatch | makes two nodes' rows incomparable |
| leftover netem | measures a link you are not describing |

`--skip-preflight` exists, and rows produced with it must be labelled unverified.

## An agent is unreachable

```bash
curl http://pi4:9410/health
ssh pi4 'systemctl status heteroserve-agent'
ssh pi4 'journalctl -u heteroserve-agent -n 50'
```

Usually the unit is not enabled after a re-flash: `ssh pi4 'cd /opt/heteroserve/Pi4 && make install'`.

## `rpc up` succeeds but llama-cli cannot connect

```bash
ssh pi4 'ss -ltnp | grep 50052'
```

- Nothing listening → the unit failed; check `journalctl -u heteroserve-rpc`.
- Listening on `127.0.0.1` → `RPC_HOST` is not `0.0.0.0` in that node's `node.env`.
- Listening but refused → leftover shaping or a switch port issue; `heteroctl tier verify`.

## A worker OOMs during a cluster run

Lower that node's `RPC_MEM_MB`, and lower `usable_mem_mb` in `cluster.json` to match. The
placement engine believes what the node advertises: an over-optimistic figure gets the node
more layers than it can hold. Better to advertise less and be reliable.

## `llama-cli: unknown argument`

Upstream flags drift. Check `llama-cli --help`, update `placement.Placement.llama_args()`
or `harness.Harness.run()`, and note the change in the run log so the dataset says which
invocation produced it.

## The energy join produced no rows

Work through it in this order:

1. Is the collector running, and is the CSV growing?
2. Does the trace cover the run's window? (`late-start` / `early-end` flags say no.)
3. Do the `node` values in `power.csv` match `node_id` in `results.csv`? The join is on
   that string.
4. Was the clock synced when recording started? Check the `power_*.sync.json` beside it.

## Energy numbers look plausible but wrong by a constant factor

Almost always one of two things:

- **Repetitions.** llama-bench repeats each measurement; the window covers `reps × n`
  tokens. `analysis.tokens_in_row` handles it — hand-rolled arithmetic often does not.
- **Channel swap.** Check `daq.channels` against each node's `ina_channel`. `config.py`
  cross-checks the JSON, but it cannot see which sensor you actually wired where. Run one
  node under load with the others idle and confirm the right channel moves.

## DAQ reports `i2c read failed`

Long bus runs at 400 kHz, missing or duplicated pull-ups, or a loose ground. Drop
`I2C_FREQ` to 100 kHz in the firmware to confirm the diagnosis; fix the wiring rather than
leaving it slow.

## Clock offset spread is tens of milliseconds

The host is too busy for a clean round trip. Stop other work and re-sync. If it persists,
the USB link is being shared with something heavy — move the Pico to its own port.

## The sweep aborted at preflight halfway through the night

By design. `sweep.json` is written incrementally, so completed runs survive; fix the block
and run the remaining strategies as a second sweep. Two sweep directories with the same
spec are fine — the analysis concatenates them.

## Predictions disagree with measurements by a large, consistent margin

The cost model has a systematic bias. Check, in order: per-layer times taken at the same
thread count as the cluster run, link bandwidth re-surveyed after the tier change, whether
llama.cpp's realised split matches the planned one (it rounds to whole layers), and whether
model load time is inside the window but outside the prediction.

Fix the bias before reporting placement wins.

## Matplotlib is not installed

Only figures need it. `heteroctl pareto --out front.csv` still writes the numbers and says
the plot was skipped. `pip install matplotlib` if you want the PNG.
