# 03 — The energy collector

Four INA226 sensors, one Pico 2, one USB cable, and the arithmetic that turns them into
joules per token. This is the measurement the project's primary metric rests on, so this
document is as much about what invalidates a number as how to produce one.

## The chain

```
supply ──[ 0.01 Ω shunt ]── node (DUT)
              │  │
            IN+  IN-
              INA226  ──I²C(0x40/41/44/45)──▶  Pico 2  ──USB CDC──▶  control node
                                                 │                      │
                                       ticks_us timestamp        offset → UTC, CSV
```

Per node: current flows supply → shunt → node. The INA226 reads bus voltage and shunt
voltage, derives current, and the Pico polls all four sensors on a fixed schedule and
streams timestamped lines.

## Channel map

| ch | node | I²C | rail |
|---|---|---|---|
| 1 | pi5 | 0x40 | 5 V USB-C |
| 2 | pi4 | 0x41 | 5 V USB-C |
| 3 | jetson | 0x44 | 5 V barrel jack (J25, J48 fitted) |
| 4 | orangepi-rv2 | 0x45 | 5 V DC |

This table exists in two places — `config/cluster.json` (`daq.channels` and each node's
`ina_channel`) and the firmware's `CHANNELS`. `control/config.py` cross-checks the two
halves of the JSON at load time and refuses to start when they disagree, because a swapped
channel attributes one node's power to another and produces numbers that look completely
reasonable.

## Wiring notes

- **High side, always.** The shunt goes in the supply line before the node, not in the
  ground return. A low-side shunt would shift the node's ground reference.
- One set of 4.7 kΩ pull-ups for the whole I²C bus, not one per sensor.
- Keep the sensor-to-Pico runs short. Four sensors on a long bus at 400 kHz is where
  intermittent `i2c read failed` messages come from.
- The Jetson's barrel jack is on the same kind of channel as the others — that is why the
  method covers every node, including the one that cannot be measured through USB.

## Sizing, and why these values

| Quantity | Value | Reason |
|---|---|---|
| Shunt | 0.01 Ω | At 3 A it drops 30 mV, inside the INA226's ±81.92 mV range, and burns only 90 mW. |
| Current LSB | 100 µA | `CAL = 0.00512 / (LSB × R) = 5120`, full scale 3.2768 A — above what the supplies deliver, so the ADC cannot be driven off-scale. |
| Averaging | 16 | With 1.1 ms conversions on both channels: 16 × 2.2 ms ≈ 35 ms, i.e. ~28 Hz per sensor. |

Pick the sample rate from the **shortest** window you need to integrate, not the longest.
A decode run lasts seconds and is easy; a 128-token prefill on the Pi 5 can be under a
second, and a rate too low to place several samples inside it turns prefill energy into an
extrapolation. 28 Hz gives ~25 samples in a one-second window, and the analysis refuses
anything below 8.

## Firmware

`firmware/pico2/main.py` plus `ina226.py`, copied to a Pico 2 running MicroPython.

```bash
make flash-daq                 # uses mpremote
heteroctl daq sync             # PING/PONG clock alignment
heteroctl daq status --seconds 10
```

The Pico timestamps samples with its own `ticks_us` counter. That is deliberate: USB CDC
delivery jitters by milliseconds under host load, so a host-side timestamp would record
when Linux got round to reading the line, not when the sample was taken.

A missing sensor is reported once at boot and then skipped. Three working channels are
worth having; a silent channel that looked alive would be much worse.

## Clock alignment

`heteroctl daq sync` sends 21 PINGs, brackets each with host timestamps, and takes the
median of `(t_before + t_after)/2 − pico_us` as the offset — the same assumption NTP makes.
The spread across rounds is recorded as the residual uncertainty and belongs in the error
budget.

Expect an offset stability around a millisecond and an RTT under 2 ms. If the spread is
tens of milliseconds, the host is too busy: stop other work before recording.

The offset is written to `power_*.sync.json` beside each trace. A restarted collector
re-syncs, so a dropped USB link costs a *gap* in the trace — visible in the analysis — but
never a wrong timestamp.

## Calibration

Per the proposal, every channel is calibrated against a known resistive load rather than
trusted from the datasheet.

```bash
scripts/05_calibrate.sh 2          # guided, one channel at a time
```

For each point: disconnect the node, put a low-tolerance power resistor in its place,
measure the voltage **across the resistor** with the multimeter, and take the reference
current as `I = V / R` from those two measured values. The script records what the INA226
reported for the same load and fits gain and offset.

Use two or three resistors spanning the node's real range (roughly 0.5–1.5 A for a Pi).
A single point sets the gain only, which is a scale check rather than a calibration — the
tool says so when you give it one.

Calibrations live in `calibration/ch*.json` and are **data**: they are versioned with the
results, not ignored. An uncalibrated channel still records, but every row derived from it
is flagged `uncalibrated-channel`, because it cannot carry the ±5 % accuracy the success
criteria claim.

## Error budget

| Source | Contribution |
|---|---|
| Shunt tolerance (1 %) | ≤ 1 % |
| INA226 gain error | ~0.1 % |
| Multimeter accuracy on the reference | ~0.5 % |
| Clock alignment residual | negligible for windows of seconds |
| Sampling (28 Hz, trapezoid) | small; dominated by the shortest window |

Combined, roughly 0.5–1.1 %, against a ≤ 5 % requirement. Quote the method and the budget
in the report, not just the conclusion.

## Recording and joining

```bash
heteroctl daq record                          # or the systemd unit
heteroctl idle-baseline --seconds 120         # marked, with the cluster up and idle
heteroctl join /opt/heteroserve/data/bench/<RUN_ID> --power .../power_*.csv
heteroctl power-summary --power ... --events .../events.jsonl
```

The join integrates `E = ∫P dt` (trapezoidal) over each row's `[t_start, t_end]` and
divides by the tokens that window actually covered — including llama-bench's repetitions,
which is an easy factor-of-5 error to make.

### When the join refuses

| Flag | Meaning |
|---|---|
| `too-few-samples:N` | fewer than 8 points in the window; the integral would be an extrapolation |
| `gap:Xs` | the DAQ dropped out mid-run |
| `late-start` / `early-end` | the run was not fully covered by the recording |
| `no-window` | the benchmark row has no wall-clock bracket |
| `uncalibrated-channel` | the channel has no calibration file |

A flagged row keeps its throughput and loses its energy. That is the intended trade: a
missing number is recoverable, a plausible wrong one is not.

## Idle baseline

Take it **with the cluster up and idle**, not powered down. An `rpc-server` waiting for
work costs more than a node at rest, and it is the waiting cost that a placement decision
actually trades against.

It must come from an explicit `idle_baseline` marker. Picking quiet-looking stretches of
the trace by eye biases the baseline downward and overstates every marginal-energy figure
derived from it — `node_power_summary` therefore returns `None` and says why, rather than
guessing.

Report total node energy in the main table and idle power per node in an appendix, so a
reader can compute the marginal figure if they want it. State which convention each table
uses; mixing them across nodes ruins the comparison.

## Sanity bands

| Node | Idle | Under load |
|---|---|---|
| Pi 5 | ~3.5–4.5 W | ~7–9 W |
| Pi 4 (fan included) | ~2.7–3.5 W | ~5.5–6.5 W |
| Jetson Nano | ~2–3 W | ~7–10 W |

Far outside these means a wiring or calibration problem, not a discovery. Check shunt
orientation (negative current = IN+/IN− swapped) and the calibration factor first.
