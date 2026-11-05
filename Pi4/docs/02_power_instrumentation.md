# 02 — Power instrumentation for the Pi 4 channel

This node is channel **2** of the four-channel INA226 + Raspberry Pi Pico 2 DAQ. The DAQ
firmware and the collector live with the control node; what follows is everything specific
to wiring, calibrating and interpreting *this* channel.

## Measurement boundary

```
5.1V 3A USB-C PSU ──┬── shunt 0.01 Ω ──┬── Pi 4 USB-C input
                    │                  │
                 IN+ │              IN- │
                    └──── INA226 ───────┘
                            │ I²C (0x41) + GND  →  Pico 2
```

High-side shunt on the 5 V input rail. Everything downstream of the shunt is inside the
boundary: SoC, LPDDR4, microSD, the Gigabit PHY **and the case fan**.

The fan is inside the boundary on purpose. The question the project asks is what a token
costs on this device, and a Pi 4 that is not cooled cannot sustain a run at all — so the
cooling is part of the cost. Report it; do not subtract an estimate for it. State the
boundary explicitly in the report, because a reader comparing against a vendor datasheet
figure (SoC only) needs to know why our numbers sit higher.

## Component sizing

| Quantity | Value | Why |
|---|---|---|
| Shunt | 0.01 Ω | At the Pi 4's ~3 A ceiling the shunt drops 30 mV, comfortably inside the INA226's ±81.92 mV range, while burning only 90 mW. |
| Bus voltage | ~5.1 V | Well within the INA226's 36 V input range. |
| Current LSB | 100 µA | `CAL = 0.00512 / (Current_LSB × R_shunt) = 0.00512 / (1e-4 × 0.01) = 5120`, giving a 3.2768 A full scale — above the PSU's 3 A limit, so the Pi cannot drive the ADC off-scale. |
| I²C address | 0x41 | A0=VS, A1=GND. Each of the four channels needs a distinct address on the shared bus: 0x40/0x41/0x44/0x45. |

## Sampling and averaging

Configure the INA226 for ~10–100 Hz effective sampling: conversion time 1.1 ms on both bus
and shunt with averaging of 16 gives ~28 Hz, which is the sweet spot here.

Decode on the Pi 4 is a steady, roughly constant load, so a high rate buys nothing.

**On this node the sampling rate is not the binding constraint, and the rationale should not
pretend otherwise.** The Pi 4 is the deliberate straggler: a 128-token prefill at roughly
10 tok/s takes about 12 s, and the energy window covers all five repetitions, so the
shortest interval the collector ever integrates is on the order of a minute — around 1700
samples at 28 Hz, against `ThinkPad/control/analysis.py`'s `MIN_SAMPLES = 8`. That guard
cannot fire here, and it is honest to say so rather than to imply the rate was chosen to
avoid it.

Where the constraint *would* bind is a fast node or a much shorter shape, and that is the
reason to keep the rate fixed across the whole dataset: 28 Hz is chosen once, for the
cluster, so that per-node energy figures are comparable — not because this node needs it.
If the shape list ever gains a short prefill, or a faster node is added, re-check the
sample count per window before trusting the integral.

(What 28 Hz genuinely cannot do is resolve sub-second power structure *inside* a prefill.
The integral over the window is sound; a reported peak power would not be.)

## Calibration

Per the proposal, each channel is calibrated against a known resistive load rather than
trusted from the datasheet:

1. Disconnect the Pi. Put a low-tolerance power resistor (e.g. 4.7 Ω 10 W, 1 %) across the
   channel in place of the node.
2. Measure the actual bus voltage with the multimeter at the resistor terminals.
3. Reference current is `I = V / R` from those two measured values — not from the nominal
   5 V and not from the nominal resistance.
4. Log the INA226's reported current at the same moment; the ratio is the channel's
   correction factor.
5. Repeat at two or three points spanning the node's real range (roughly 0.6 A idle to
   1.2 A under load on the Pi 4) so the correction covers the operating band, not one point.
6. Store the factors with the dataset — a correction that lives only in someone's notebook
   makes every downstream number unreproducible.

Combined uncertainty lands around 0.5–1.1 % on this chain (shunt tolerance, INA226 gain
error, multimeter accuracy), comfortably inside the ≤ 5 % target in the success criteria.

## What this node actually draws

Rough expectations, useful as a sanity check rather than a specification:

| State | Expected |
|---|---|
| Idle, fan on, network up | ~2.7–3.5 W |
| All four cores in decode | ~5.5–6.5 W |
| Prefill burst | ~6–7 W peak |

Reading far outside this band means something is wrong with the channel, not something
interesting about the Pi. Check the shunt orientation and the calibration factor first.

## Joules per token

The collector integrates `E = Σ P·Δt` over each run's **repetition span**, then divides by
the token count for that run. Because prefill and decode are run as **separate llama-bench
invocations** (see `scripts/06_bench_local.sh`), each window contains exactly one phase and
no disentangling is required.

**Mind which window.** In `windows.jsonl`, `t_start`/`t_end` are the whole `llama-bench`
**process** — permanently, because that file is the primary artefact and redefining a field
in it would be its own defect. That bracket also contains process start, the GGUF page-in
off the microSD, a full-length warmup prefill and teardown, none of which produce a token;
integrating over it overstates joules-per-token by tens of percent. The repetition span is
`t_meas_start`/`t_meas_end` in the same record.

`node/bench_parse.py` resolves this for consumers: the `t_start`/`t_end` columns **in
`results.csv`** already carry the repetition span, and the `energy_window` column names
which window they are (`reps-span`, or `process` when a validation gate refused the
derivation). So a collector reading `results.csv` needs no special handling — one reading
`windows.jsonl` directly must pick the right pair itself. See
`docs/04_measurement_protocol.md`, "What is in the numerator of joules per token".

Subtract nothing for idle unless you say so in the report. Both figures are defensible —
total energy per token, and marginal energy above idle — but they are different numbers
and mixing them across nodes ruins the comparison. The recommended convention for this
project: report **total** node energy in the main table, and give idle power per node in an
appendix so a reader can compute the marginal figure if they want it.

## Failure modes seen on this channel

- **Under-voltage flag set while current looks normal.** The USB-C cable, not the PSU.
  Thin cables drop enough at 3 A to trip the Pi's 4.63 V detection. Use the cable that
  ships with the official supply.
- **Current reads negative.** IN+/IN− swapped across the shunt.
- **Noise on the I²C bus when all four channels run.** Keep the Pico–INA226 runs short and
  add 4.7 kΩ pull-ups once for the whole bus, not per sensor.
- **Fan cycling shows as a square wave in the trace.** Expected, and it is real energy the
  node consumed. Do not filter it out; note it in the analysis.
