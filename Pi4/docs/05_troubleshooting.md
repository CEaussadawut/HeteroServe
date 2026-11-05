# 05 — Troubleshooting the Pi 4 node

## Throttling and under-voltage

```bash
vcgencmd get_throttled
```

| Bit | Meaning |
|---|---|
| 0 | under-voltage **now** |
| 1 | ARM frequency capped now |
| 2 | throttled now |
| 3 | soft temperature limit now |
| 16–19 | the same four, **since boot** (sticky) |

The sticky bits do not clear without a reboot. That is why `assert_clean_state` demands a
literal `0x0`: a run that starts with `0x50000` may be fine right now, but you cannot tell
from the flags alone whether it stayed fine, and a measurement you cannot certify is not
a measurement.

- `0x50000` (bits 16+18): it was under-volted earlier. Reboot, re-check, and if it returns,
  replace the USB-C cable before suspecting the PSU.
- Temperature bits: improve cooling, raise the inter-run sleep in `06_bench_local.sh`, and
  discard the affected rows.

## `make build` dies with the compiler killed

Out of memory. Swap is off by design. Confirm `-j2` in the script, close every other
session on the board, and rebuild. If it still fails, `-j1` finishes reliably.

Do **not** re-enable swap to get the build through. Turning it back on and forgetting is
how a latency tail ends up measuring the microSD card.

## `rpc server: unknown argument` — or the server exits 0 and nothing is listening

Upstream flag names drift, and upstream's unknown-argument branch calls **`exit(0)`, not
`exit(1)`** — so a bad flag prints usage, exits *successfully*, and systemd records a clean
start with nothing bound to the port. That is why `scripts/04_rpc_server.sh` no longer
hard-codes flags: `rpc_extra_args()` in `scripts/lib.sh` probes `--help` and passes only
what this build actually accepts. `bin_rpc()` likewise resolves the binary by trying
`ggml-rpc-server` first, then `rpc-server` (upstream renamed it).

So if the server will not come up, check what is *listening*, not the exit code:

```bash
scripts/04_rpc_server.sh status      # says both "unit active" and "listening on :PORT"
ss -ltn | grep ":$RPC_PORT "
```

If a genuinely new flag is needed, add it to `rpc_extra_args()` behind a `--help` probe
rather than passing it unconditionally, and note the change in the run log so the dataset
records which invocation produced it.

## The control node cannot reach `rpc-server`

```bash
scripts/04_rpc_server.sh status      # is it listening at all?
ss -ltnp | grep 50052
ping 192.168.50.10
sudo tc qdisc show dev eth0          # a forgotten netem rule can look like a dead link
```

`RPC_HOST=0.0.0.0` in `config/node.env`; bound to `127.0.0.1` it works locally and is
invisible from the cluster.

## The node is OOM-killed during a cluster run

Lower `NODE_USABLE_MB`. (`RPC_MEM_MB` only reaches the server on a build old enough to
still have `--mem`; upstream removed it in 41386cf, and `scripts/lib.sh rpc_extra_args()`
warns when it is gone.) The placement engine
plans against `NODE_USABLE_MB` (through the agent's `usable_budget_mb`), so an
over-optimistic figure gets this node assigned more layers than it can hold. Better to
budget 2400 MB and be reliable than 3000 MB and lose a sweep. Keep
`systemd/heteroserve-rpc.service`'s `MemoryMax` above the budget so the cgroup stays a
backstop rather than the thing that binds first.

## Results scatter far more than on the Pi 5

Partly real — it is the straggler, and variance under thermal pressure is a finding. But
check first, in this order: governor actually `performance` on all four cores, no sticky
throttle bits, the inter-run sleep long enough that `temp_end_c` is not trending upward
across `windows.jsonl`, and nothing else running on the board.

## Clock offset above 5 ms

```bash
chronyc sources -v
chronyc tracking
sudo chronyc makestep        # one-off correction
```

If the control node is unreachable as an NTP source, the healthcheck blocks the node. That
is intended: without a shared clock, joules cannot be attributed to a phase, and a
plausible-looking but misattributed joules-per-token number is worse than no number.

## Wi-Fi tier is slower than expected

The Pi 4's on-board radio is 2.4/5 GHz 802.11ac, single-stream. Expect roughly 40–90 Mbit
of real throughput at 5 GHz. Keep the AP, the channel and the distance fixed across the
whole sweep and write them down — a Wi-Fi tier that moves between runs is not a tier.

## Benchmark JSON exists but `results.csv` is empty

`bench_parse.py` matches filenames of the form `<model>_<phase><n>_t<threads>.json`. If
runs were launched by hand with different names, the window join fails and rows are flagged
`no-window:energy-join-impossible`. Run through `scripts/06_bench_local.sh`, or write the
matching `windows.jsonl` entries yourself.

## Recovering a half-finished sweep

Each run directory is self-contained and named by UTC timestamp. Re-running the sweep
creates a new directory rather than touching the old one, so a partial sweep can be kept
and parsed as-is:

```bash
python3 node/bench_parse.py /opt/heteroserve/data/bench/<RUN_ID>
```
