# Node: Jetson Nano (GPU node)

First contact 2026-09-25. Board cabled, powered and given SSH by the user; first reached
from the control node that morning. Control node: aun-ThinkPad-T580.

| Field | Value |
|---|---|
| Model | NVIDIA Jetson Nano Developer Kit — **B01 carrier** (`p3449-0000-b00`), 4 GB module (`p3448-0000`) |
| Serial | `1420121029902` |
| SoC | Tegra X1 — 4× Cortex-A57 (1479 MHz ceiling), 128-core Maxwell GM20B (sm_53, 921.6 MHz ceiling) |
| Memory | 3963 MiB total, **shared by CPU and GPU**; 2789 MiB available at idle with the desktop running |
| OS | Ubuntu 18.04.6 LTS, **L4T R32.7.6** (JetPack **4.6.6**-b24), kernel 4.9.337-tegra |
| CUDA | **10.2** (`nvcc` release 10.2) |
| Vulkan | 1.2.131, driver 32.7.6 — **`shaderFloat16` true** (probed directly, see below) |
| Storage | mmcblk0, 116.2 GB microSD, root on p1, 97 GB free |
| Hostname | `jetson-desktop` *(not renamed; the plan calls it `jetson-nano-01`)* |
| Address | **since 2026-09-26: eth0 `192.168.100.111`/24**, DHCP-reserved on the T628L (`192.168.100.1`); taken at boot 13:37. No wired overlay. Clock synced from the laptop's NTP (stepped +13 h 15 m). *Before:* eth0 `192.168.1.111` on the F6107A, reserved 2026-09-25, 1000 Mb/s full duplex |
| Second interface | wlan0 — Intel M.2 card (`iwlwifi`), `AUN_5G`, 5500 MHz, **−68 dBm**. Reserved at **192.168.1.110** (binding `jetson-wl`, 2026-09-25) and **on it since 12:18 the same day**. A plain reconnect would not have done it: the lease renewed at 12:03, after the binding existed, and the router re-granted `.112`. What worked was `nmcli con down AUN_5G`, moving `dhclient-af39fc9b-…-wlan0.lease` aside (kept as `.lease.pre110-*`), then `nmcli con up` — a fresh DISCOVER got `.110`. eth0 and the ssh session were untouched |
| Wired overlay | **none** — no `192.168.10.x` address, unlike both Pis |
| **Power input** | **micro-USB** (user, 2026-09-25) — *not* the planned J25 barrel jack with J48 fitted |
| **Power mode** | **MAXN (nvpmodel 0) since 2026-09-25 08:23**, at the user's request — 4 cores at 1479 MHz, GPU 921.6 MHz. As received it was 5 W (nvpmodel 1: 2 cores, 921.6 / 614.4 MHz). **Still on micro-USB — idle and light load only** |
| **Idle power** | **1167 mW** in 5 W, **1159 mW** in MAXN — the same, within one sensor step (see below) |
| `jetson_clocks` | **not applied** — apply at measurement time only (it pins clocks and fan at maximum) |
| Swap | **on** — zram, 2 × ~1 GB (JetPack default). Both Pis have swap off |
| Desktop | **off since 2026-09-25 10:2x** — default target `multi-user.target`, autologin session ended; freed ~470 MiB (3.2 GB now available). It was an unattended autologin (`gnome-initial-setup` still open). Revert: `sudo systemctl set-default graphical.target && sudo systemctl start gdm` |
| sudo | **NOPASSWD since 2026-09-25**, installed by the user (`/etc/sudoers.d/010-hs-nopasswd`, 0440 root:root); verified with cached credentials cleared |
| SSH password login | **enabled** (sshd default). Both Pis are hardened; this board is not |
| Toolchain | system gcc 7 unchanged; **added 2026-09-25** for Path 1: gcc/g++ **11.4** (`ppa:ubuntu-toolchain-r/test`), cmake **3.31.12** (`~/opt/cmake`, Kitware binary, SHA-256 checked), Vulkan-Headers **v1.4.363**, `glslc` from shaderc **v2026.4** built on the board (`~/opt/shaderc`), Python 3.8, ninja, ccache, curl — see below |
| `llama.cpp` | **not built** |
| Idle temps | CPU 29 °C, GPU 28 °C (5 W, idle, desktop up) |
| Fan | **fitted and working** — 3963 RPM at PWM 128 (tach test, 08:24). The kernel drives it from temperature (`temp_control=1`), so it idles at PWM 0 and spins up under load. nvpmodel's fan *profile* is unset (`fanNull`, and `-q` warns about it) — harmless, since profiles are `N/A` on this release |

## SSH

Reached as `ssh jetson` (aliases `hs-jetson`, `jetson-nano-01`) — user `jetson`,
key `~/.ssh/id_ed25519`.

The alias shipped pointing at **192.168.1.104**, which refuses port 22. That address
belongs to a different device with a randomised MAC (`36:41:c7:59:37:84`, the
locally-administered bit set — typical of a phone). The Jetson was found by its NVIDIA
OUI `00:04:4b:ec:52:f2`. `HostName` corrected to `.111` (eth0) on 2026-09-25; backup at
`~/.ssh/config.bak.20260925-jetson`.

**ARP flux, measured 2026-09-25:** a broadcast ARP (`arping -b`, from the Pi 5's wlan0) for
*either* address is answered by **both** interfaces — the eth0 MAC in ~5 ms, then the wlan0
MAC 16–99 ms later — because eth0 and wlan0 sit on the same subnet and every interface runs
`arp_ignore=0`. Linux senders usually keep the first (eth0) reply, but nothing guarantees it.
The same test between the Pis on `192.168.10.x` gets **eth0 replies only**, so the Pi 5 ↔ Pi 4
wired results are not affected. Harmless for management, but it makes the path a packet
takes ambiguous. Before this node carries experiment traffic, either take wlan0
down or give eth0 a static address on the `192.168.10.0/24` overlay like the Pis.

Host key fingerprints accepted 2026-09-25 (TOFU — verify on the board before trusting):

    RSA     SHA256:bTUvsflyR4QB5ma50KkXbYJsF4VPLH/oMhsyKkcEm5E
    ECDSA   SHA256:7INLojN9Mk7BwZjD8cLz4J/yDlHvFkJ14ARIr306PJ4
    ED25519 SHA256:irqgpDnrtUE62POFqTNGYxmpuaYNn7l85MsSpxI0E/M

## Power: the mode is now MAXN, the supply is still micro-USB

**Update, 08:23.** Switched to MAXN at the user's request, once passwordless sudo was in
place: `nvpmodel -m 0`, no reboot needed, `/var/lib/nvpmodel/status` now `pmode:0000`, all
four cores online at 1479 MHz and the GPU capped at 921.6 MHz. `dmesg` showed **zero**
over-current or undervoltage alarms before the switch (5 h 39 min uptime), after it, and after
the idle sample. Rows collected before 08:23 are 5 W rows; rows after are MAXN rows.

The plan (`ThinkPad/config/cluster.json`, jetson `notes`) is a 5 V / 4 A barrel supply on
J25 with the J48 jumper fitted, running MAXN (nvpmodel 0, 10 W). The board as received
runs from **micro-USB in 5 W mode**, which is what NVIDIA prescribes for that input — it
is rated 5 V / 2 A, and MAXN can ask for 10 W from the module alone before the carrier,
the Wi-Fi card and USB are counted.

What 5 W mode costs, read from the board's own caps:

| | 5 W (as received) | MAXN (since 08:23) | 5 W as a share |
|---|---|---|---|
| CPU cores online | 2 | 4 | 50% |
| CPU clock cap | 921.6 MHz | 1479 MHz | 62% |
| CPU core·GHz | 1.84 | 5.92 | **31%** |
| GPU clock cap | 614.4 MHz | 921.6 MHz | **67%** |

So a CPU-bound figure taken in 5 W would sit near a third of the MAXN figure, and a
GPU-bound one near two thirds. **Do not mix 5 W and MAXN rows**, and record the mode
with every run — `/var/lib/nvpmodel/status` reads it without sudo.

MAXN on micro-USB is possible but is the configuration the plan warns against: under a
sustained CPU+GPU load the input can sag, the board then throttles or resets, and a reset
mid-write risks the SD card. If MAXN is set before the barrel supply arrives, treat it as
**idle and light-load only**, and check `dmesg` for over-current alarms after any run.
The GLINK **GAC5-5** (5 V / 5 A, 5.5 × 2.5 mm barrel) is a suitable J25 supply.

### Idle power, measured as root

The collector's method — mean `POM_5V_IN` over 15 s of `tegrastats`, first 3 samples dropped
— run with sudo, so the rails are actually present. Board idle, desktop running, Wi-Fi and
Ethernet up. Raw lines: `results/jetson_idle_tegrastats_20260925_{5W,MAXN}.txt`.

| | 5 W (nvpmodel 1) | MAXN (nvpmodel 0) |
|---|---:|---:|
| `POM_5V_IN` — module input (excludes carrier loads) | **1166.9 mW** [1084–1372], n=11 | **1158.6 mW** [1084–1291], n=10 |
| `POM_5V_CPU` | 116.6 mW | 128.3 mW |
| `POM_5V_GPU` | 45.0 mW | 45.4 mW |

`POM_5V_IN` is the **module's** input rail, not the whole developer kit's: the INA3221 is on the module, per NVIDIA's Jetson Linux R32.7.5 guide (power_management_nano): "The original Jetson Nano and Jetson TX1 modules have a three-channel INA3221 power monitor at I2C address 0x40"; channel 0, `VDD_IN`: "Main module power input". Carrier-board loads — USB, the M.2 Wi-Fi card, the fan — are outside it *(corrected 2026-09-26; this row was labelled "whole dev kit", after Jetson audit finding #15, which inferred the opposite from the I²C bus the chip enumerates on)*.

**MAXN costs nothing at idle**: −8 mW, well inside one ~40 mW step of the INA3221. The
clocks idle low under DVFS either way, and the two extra cores add ~12 mW on the CPU rail.
At ~1.16 W the board draws about 0.23 A — far inside micro-USB's 2 A. The risk is sustained
load, not the mode.

This is a **device-table** figure, taken with the board cold. It is not the idle floor
subtracted from a sweep; that floor is measured per case, from the cooldown before it.

## Path 1 toolchain, installed 2026-09-25

Each version is pinned to what was current when the `llama.cpp` commit was written
(`b23701f7`, committed 2026-09-19), not simply the newest:

| Component | Version | Where from | Why this one |
|---|---|---|---|
| gcc / g++ | 11.4.0 | `ppa:ubuntu-toolchain-r/test` (apt) | 18.04 stops at gcc 8; 11 is Ubuntu 22.04's default, the compiler `llama.cpp` CI is surest to build with |
| cmake | 3.31.12 | Kitware aarch64 binary → `~/opt/cmake`, SHA-256 matched Kitware's list | `ggml-vulkan` needs ≥ 3.19; last 3.x series, because CMake 4 rejects old `cmake_minimum_required` lines in dependencies |
| Vulkan-Headers | v1.4.363 (2026-09-18) | Khronos, `~/opt/src/Vulkan-Headers` | newest tag *before* the commit — v1.4.364 was tagged 2026-09-25, and `vulkan.hpp` does break its C++ API between releases |
| glslc | shaderc v2026.4 (2026-09-11) | built on the board → `~/opt/shaderc` | not packaged for 18.04. Only glslang, SPIRV-Headers and SPIRV-Tools fetched, at shaderc's pinned revisions |
| Python | 3.8.0 | apt (bionic universe) | SPIRV-Tools' code generators; 3.6 is too old to trust |
| Vulkan loader | 1.2.141 | NVIDIA `nvidia-l4t-libvulkan` (already present) | what `libvulkan.so.1` resolves to. Headless Vulkan confirmed working with X stopped |

**Power guard.** While the board runs MAXN from micro-USB, `testbed/scripts/jetson_power_guard.sh`
runs as root. It logs the INA3221 input rail every second, and drops the board to 5 W if the
input stays below 4.75 V for 3 s or any over-current alarm appears. Log:
`~/heteroserve/logs/power-guard.log`. A 4-core compile draws **4.3–5.0 W** (~1 A), with the
input never below 5.03 V.

## Vulkan fp16 — Path 1 vs Path 2, decided on the board

`Jetson/docs/50_pooled_gpu_options.md` makes the choice between Path 1 (Vulkan backend at
the cluster pin `b23701f7`) and Path 2 (CUDA patch set) hang on a single fact: whether the
Nano's Vulkan driver exposes `shaderFloat16`. Its own rule: *"If `shaderFloat16 = true`,
Path 1 becomes strictly better than Path 2 — same destination, zero patches, no
maintenance — and should be taken."*

**The literal test in that doc cannot answer it on this board.** `vulkaninfo | grep -i
shaderFloat16` prints nothing, because Ubuntu 18.04's `vulkaninfo` only prints core
`VkPhysicalDeviceFeatures` and never the extension feature structs. The extension
`VK_KHR_shader_float16_int8` *is* advertised — but a driver may advertise it with only
`shaderInt8` set, so the extension alone proves nothing either.

So the bits were read straight from the driver with `vkGetPhysicalDeviceFeatures2`, via
Python `ctypes` against `libvulkan.so.1` (no headers are installed, and nothing was
installed or left on the board):

    device                              : NVIDIA Tegra X1 (nvgpu)
    apiVersion                          : 1.2.131
    driverVersion (raw)                 : 0x0801c180        (= 32.7.6)
    shaderFloat16                       : True
    shaderInt8                          : True
    storageBuffer16BitAccess            : True
    uniformAndStorageBuffer16BitAccess  : True
    storagePushConstant16               : True
    storageInputOutput16                : False

**`shaderFloat16` is true → by the doc's own rule, Path 1 is indicated.** What this does
*not* establish: that `llama.cpp`'s Vulkan backend builds on this toolchain, that it
selects its fp16 shaders here, or how fast it is. None of that has been tried. The
build needs, at minimum, `glslc` (not packaged for 18.04), Vulkan-Hpp headers, cmake ≥ 3.14
and a newer C++ compiler than gcc 7 — most of which needs sudo.

## First run of the project's fact collector — two fields are wrong

`Jetson/scripts/60_collect_node_facts.sh` was run on the board for the first time
(`~/heteroserve/Jetson`, kit copied over 2026-09-25). Raw output kept as
`results/jetson-nano-01_facts_20260925_5W-microUSB_prebuild_RAW.json`. **Two of its
fields are not true, so do not ingest that file as-is:**

| Field | Collector says | Reality | Why |
|---|---|---|---|
| `power.idle_board_mw` | `0` | **not measured** | tegrastats run as non-root **silently omits every `POM_*` rail** — the INA3221 nodes under `/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/` are root-only (`Permission denied`). Zero samples, and `${IDLE_MW:-0}` turned "no data" into a confident zero. Its own basis string does say "0 samples". |
| `power.clocks_locked` | `true` | **false** | echoed from the config's `LOCK_CLOCKS=1`, not read from the board. `jetson_clocks` has never run here. |

The fields it got right are worth noting too: `nvpmodel` `"0"` (config) against
`nvpmodel_active` `"1"` (board) is exactly the mismatch it was designed to expose.
`cpu.cores` reads `2` — correct as the *online* count in 5 W mode, but unlike the clock
fields it has no ceiling beside it, so a reader could take it for the silicon's core count.

Flagged to the Jetson session in `Jetson/.handoff/`, not edited here.

## What is needed next

Needs **the user** (sudo password, or hands on the board):

1. ~~Passwordless sudo~~ — **done** 2026-09-25, installed by the user.
2. **Barrel supply on J25 + J48 jumper** before any sustained MAXN load.
3. ~~**DHCP reservation**~~ — **done** 2026-09-25 for `eth0` (`.111`) and `wlan0` (`.110`), along with the Pis, the ThinkPad and the Shelly strip. `wlan0` moved from `.112` to `.110` at 12:18 (see the Address table).
4. **Decide on SSH password login** — disabling it matches the Pis.

Then, from the control node:

5. ~~Switch to MAXN~~ — **done** 08:23. `jetson_clocks` at measurement time only.
6. ~~Measure idle power with sudo~~ — **done**: 1167 mW (5 W) / 1159 mW (MAXN).
7. Swap off; desktop off (`multi-user.target`) to reclaim ~1 GB of a 4 GB node.
8. Toolchain for Path 1, then build `llama.cpp` `b23701f7` with `-DGGML_VULKAN=ON -DGGML_RPC=ON`.
