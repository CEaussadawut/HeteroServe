# Flashing and first boot — Jetson Nano

## What to flash

The Nano developer kit boots from a microSD card. Use NVIDIA's official
**JetPack 4.6.x** SD card image for the Jetson Nano developer kit
(L4T 32.7.x, Ubuntu 18.04 aarch64, CUDA 10.2, cuDNN, TensorRT).

JetPack 4.6.x is the **last** release line for the original Nano — JetPack 5/6
are Orin-only. Everything downstream in this folder assumes L4T 32.7.x; if you
are on something else, `scripts/60_collect_node_facts.sh` will record it and the
build script will tell you what changes.

> The Nano's age is a premise of the study, not an obstacle. A 2019 device with
> a CUDA toolkit from 2019 is precisely the "cheap legacy hardware a Thai SME
> already owns" that the work is about.

## Card

- 64 GB or larger, A1/A2-rated. 32 GB fills up fast once you have several GGUFs
  and a llama.cpp build tree.
- Write with Balena Etcher or `dd`. Verify after writing — a silently bad card
  produces build failures that look like compiler bugs.
- **Strongly preferred:** move `$HS_ROOT` (models + build tree) to a USB 3.0 SSD.
  Model load time drops several-fold and you stop wearing out the card during
  long sweeps. Set `HS_ROOT` in `config/node.env` to the mount point.

## Power

| Supply | nvpmodel | Notes |
|---|---|---|
| micro-USB 5 V / 2 A | mode 1 (5 W) only | will brown out under GPU + 4 cores |
| barrel jack 5 V / 4 A + **J48 jumper fitted** | mode 0 (MAXN, 10 W) | required for any 10 W measurement |

The J48 jumper is the two-pin header next to the barrel jack. Without it the
board ignores the barrel jack. A brown-out mid-sweep corrupts the SD card as
often as not, so fit it before the first 10 W run.

Power mode is an **experimental condition**, not a setting: run the sweep at
both 5 W and 10 W and tag the results (`./50_bench_device.sh --tag 5W`). The
5 W-vs-10 W energy-per-token comparison is a result in its own right.

## Cooling

The stock heatsink is passive. Under a sustained prefill sweep the SoC will
reach its throttle point and your last cases will run slower than your first —
which silently corrupts the whole table.

Fit a 5 V PWM fan (the 40 mm fan mounts on the heatsink). `00_system_prep.sh`
pins it to full. If you have no fan, keep `cooldown_s` high in
`bench/matrix.json` and watch `temp_rise_c` in the output — a rise above ~8 °C
across a window means the number is thermally contaminated.

## First boot

1. Attach keyboard/monitor for the initial Ubuntu setup (user, locale, network).
   After that the box is headless; `00_system_prep.sh` disables the desktop.
2. Set a static IP or a DHCP reservation. The cluster inventory
   (`config/cluster.hosts`) refers to nodes by address.
3. Enable ssh (it is on by default on the JetPack image) and copy your key.
4. Connect **wired Ethernet** for the primary experiments. Wi-Fi is a separate
   labelled condition, not the default.

## Sanity check before you continue

```bash
cat /etc/nv_tegra_release          # expect: # R32 (release), REVISION: 7.x
nvcc --version                     # expect: release 10.2
free -h                            # expect: ~3.9 Gi total
tegrastats --interval 1000         # expect POM_5V_IN / _GPU / _CPU fields
```

If `tegrastats` shows no `POM_*` fields, the INA3221 is not being read and this
node cannot contribute on-board power data — fall back to the external inline
meter alone and record that in the node facts.

Next: `make prep`.
