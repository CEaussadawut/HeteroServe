# 01 — OS setup for the Pi 4 node

Target state: a headless Raspberry Pi OS (64-bit) worker named `pi4`, on a static wired
address, reachable by key-only SSH from the control node, with its clock disciplined by
the control node.

## 1. Flash the card

Raspberry Pi Imager → **Raspberry Pi OS Lite (64-bit)**.

64-bit is not optional. The 32-bit userland cannot address the full 8 GB in one process
and loses the AArch64 NEON paths llama.cpp relies on — measurements taken on a 32-bit
image are not comparable with the Pi 5 or the Jetson and would have to be thrown away.

Lite (no desktop) for the same reason: a desktop session holds RAM and wakes the CPU on
its own schedule, which shows up directly in both the latency tail and the power trace.

In Imager's advanced options set:

- hostname `pi4`
- SSH enabled, **public-key only** (paste the control node's `~/.ssh/id_ed25519.pub`)
- username `pi`, locale/timezone `Asia/Bangkok`
- Wi-Fi credentials — configured but *not relied on*; the wired link is the primary tier
  and Wi-Fi is one of the two real link tiers under study

The 128 GB A2 card from the procurement list is right for this node: the models plus the
benchmark output fit with room to spare, and A2 random-read matters when a model is
mapped from disk on first load.

## 2. First boot

```bash
ssh pi@raspberrypi.local          # or find the lease on the switch
sudo raspi-config                 # expand filesystem if the image did not
```

## 3. Static address

The energy collector joins power samples to a node by address, and `tc`/netem rules are
written per interface. A DHCP lease that moves mid-experiment silently breaks both. Fix
the address — either a reservation on the router, or on the node:

```bash
sudo nmcli con mod "Wired connection 1" \
  ipv4.addresses 192.168.50.13/24 \
  ipv4.gateway 192.168.50.1 \
  ipv4.dns 192.168.50.1 \
  ipv4.method manual
sudo nmcli con up "Wired connection 1"
```

Suggested cluster map (keep it consistent with `config/node.env` on every node):

| Node | Role | Address |
|---|---|---|
| T580 | control: manager, placement, harness, energy collector, NTP | 192.168.50.10 |
| pi5 | strong CPU worker | 192.168.50.12 |
| **pi4** | **deliberate straggler worker** | **192.168.50.13** |
| jetson | GPU (FP16) worker | 192.168.50.15 |
| orangepi-rv2 | RISC-V worker (stretch scope) | 192.168.50.16 |

## 4. Firmware and EEPROM

```bash
sudo rpi-eeprom-update -a && sudo reboot
```

Record the bootloader date in the run log. Firmware revisions have changed the Pi 4's
idle power and its USB/PHY behaviour; a dataset gathered across two different firmware
versions needs that noted, not discovered later.

## 5. Cooling

Fit the heatsink-and-fan case from the procurement list before any benchmarking. The
Pi 4 soft-throttles at 80 °C and hard-throttles at 85 °C, and a throttled run measures
the heatsink rather than the SoC. `scripts/99_healthcheck.sh` refuses to certify the node
above 70 °C for that reason.

The fan is powered from the same 5 V rail the INA226 watches, so its consumption appears
in this node's energy figures. That is deliberate — see `docs/02_power_instrumentation.md`.

## 6. Bring-up

```bash
# ON THE PI -- create the destination first, or the rsync below has nowhere to land
ssh pi@192.168.50.13 'sudo mkdir -p /opt/heteroserve && sudo chown pi:pi /opt/heteroserve'

# ON THE CONTROL NODE -- from the directory that contains Pi4/
rsync -a Pi4/ pi@192.168.50.13:/opt/heteroserve/Pi4/

# ON THE PI
ssh pi@192.168.50.13
cd /opt/heteroserve/Pi4
make bootstrap     # apt, toolchain, chrony pointed at the control node, hostname
make tune          # governor, swap off, link speed, service quiet-down
make build         # llama.cpp for Cortex-A72  -> writes build_manifest.json
# Copy the SHA `make build` printed into LLAMA_PIN_SHA in config/node.env now.
# `make health` gates on it, so it cannot print READY until you do.
make models        # small GGUF models for isolated characterisation
make install       # systemd units + sudoers
make test          # off-target suite; catches a half-finished deploy before the Pi does
make health        # must print READY before any measurement
```

## 7. Time base

The control node must run an NTP server; `make bootstrap` points this node's chrony at
`$CONTROL_NTP`. On the T580:

```bash
# /etc/chrony/chrony.conf
allow 192.168.50.0/24
local stratum 10
```

Verify from the Pi:

```bash
chronyc sources -v
chronyc tracking | grep 'System time'     # want < 5 ms, and the healthcheck enforces it
```

Energy is attributed to a phase by matching the INA226 sample timestamps against this
node's phase markers. If the clocks disagree by more than a few milliseconds, joules get
attributed to the wrong phase, and joules-per-token is the project's primary metric —
this step is load-bearing, not housekeeping.
