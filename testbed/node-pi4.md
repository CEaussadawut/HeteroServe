# Node: Raspberry Pi 4 Model B (straggler node)

Baselined 2026-09-19. Reached as `ssh pi4`.

| Field | Value |
|---|---|
| Model | Raspberry Pi 4 Model B Rev 1.5 |
| Board revision | `d03115` — **8 GB** (student-owned; the 4 GB budget line was dropped 2026-09-20 to match) |
| CPU | BCM2711, Cortex-A72 ×4 @ 1.80 GHz |
| Memory | 7.6 GiB usable |
| OS | Debian GNU/Linux 13 (trixie), arm64 userland |
| Kernel | 6.18.39+rpt-rpi-v8 |
| Storage | mmcblk0, 59.4 GB — SanDisk (`manfid 0x03`, `oemid 0x5344`), fw `0x5` |
| Hostname | `stayaun` — **not renamed**, predates this project |
| Address | **since 2026-09-26: eth0 `192.168.100.106`/24**, DHCP-reserved on the T628L (`192.168.100.1`), plus static overlay `192.168.10.12`/24; Wi-Fi off (wired-only). *Before:* wlan0 `192.168.1.106` on the F6107A, reserved 2026-09-25 |
| Wi-Fi | `AUN_5G`, ch 100, 5500 MHz, -40 dBm, 433 Mbit/s PHY, `brcmfmac` |
| eth0 | DOWN — no switch yet |
| Idle temp | 52.1 °C, `get_throttled=0x0` |
| Swap | **none** — zram disabled 2026-09-19 |
| Login | key auth as `aun`; password auth **disabled** |
| sudo | NOPASSWD (`/etc/sudoers.d/010_aun-nopasswd`) |

## Measured 2026-09-19

Storage, `dd` with `O_DIRECT` so the page cache is bypassed without needing root:

| | |
|---|---|
| Sequential read, 512 MB | **44.7 MB/s** |
| Sequential write, 512 MB | 33.4 MB/s |
| Random 4 K read, 4096 blocks | 14.9 MB/s |

Network, raw TCP via `nc` (no SSH crypto in the path):

| | |
|---|---|
| Laptop -> Pi 4, on 2.4 GHz | 5.0 MB/s (40 Mbit/s) |
| Laptop -> Pi 4, on 5 GHz | **10.7 MB/s** (86 Mbit/s) |
| TCP connect RTT (power-save off) | 7.0–12.1 ms |

## Reading these numbers

- **CONFIRMED:** the 44.7 MB/s read is the **SD interface**, not the card —
  `/sys/kernel/debug/mmc0/ios` reports `timing spec 7 (sd uhs DDR50)` at 50 MHz,
  against the Pi 5's `SDR104` at 200 MHz. Swapping cards would barely move it.
- Moved to 5 GHz on 2026-09-19: 5.0 -> 10.7 MB/s. Still below the Pi 5's 18.5 MB/s
  **despite an identical radio picture** (433 Mbit/s PHY, -40 dBm, same channel, same
  driver). The remaining limit is the Pi 4's SDIO bus to its wireless chip.
- At 44.7 MB/s a 4.7 GB model takes **~105 s** to load, against ~50 s on the Pi 5.
  Budget for that in experiment turnaround.

## To do

1. Decide whether to rename `stayaun` -> `hs-pi4`. No mDNS clash, so cosmetic; the
   counter-argument is that the box may have other uses under that name.
2. Wire it to the switch when one exists.

Toolchain: gcc/g++ 14.2.0, make 4.4.1, git 2.47.3, python 3.13.5,
**cmake 3.31.6, ccache 4.11.2** (installed 2026-09-19).
Note `iw` is at `/usr/sbin/iw` — not on a normal user's `PATH` over SSH.

## CPU features — the most important result so far

    Pi 4 (Cortex-A72):  asimd crc32
    Pi 5 (Cortex-A76):  asimd crc32 fphp asimdhp asimdrdm asimddp

The A72 **lacks `asimddp`** (ARM dot product, `SDOT`/`UDOT`). `llama.cpp`'s quantised
ARM kernels lean on dot-product instructions; without them this node falls back to a
slower plain-NEON path.

This means the Pi 4 is not merely "the same architecture, clocked lower". It is
**missing an instruction the workload depends on**, so its decode throughput will lag
the Pi 5 by more than the 1.8 vs 2.4 GHz clock ratio predicts. Neither board has
`i8mm` or `sve`.

This is a genuine characterisation result and belongs in the dataset: it is the
clearest evidence so far that the cluster is heterogeneous in a way that *matters for
placement*, not just in badge specs. It also gives a concrete, testable prediction —
measure tokens/s on each node at the same quantisation and the gap should exceed
1.33x. If it does not, the dot-product path is not the bottleneck and that is worth
knowing too.
