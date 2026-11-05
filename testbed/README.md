# HeteroServe testbed — start here

This folder is the running record of the physical cluster. If you have been away,
read this page, then `JOURNAL.md` for how things got this way.

> **2026-10-19 — THE PI 5 IS BACK, on a replacement board** (serial `b8ed9049ada12a03`, eth0 MAC `2c:cf:67:b2:af:52`; the
> returned board was `b266e1877252f91e`, `88:a2:9e:f0:e7:8a`) running the old SD card: hostname `hs-pi5`, the same host
> keys, llama.cpp at the pin, its seven models sha256-verified on 10-19. The cluster is four nodes again.
>
> | node | address | strip outlet |
> |---|---|---|
> | Pi 4 | `192.168.1.110` (+ overlay `.10.12`) | 0 |
> | Jetson | `192.168.1.114` | 1 |
> | Pi 5 | `192.168.1.122` (DHCP; the user reserves it) + overlay `.10.11` | 2 |
> | laptop | `192.168.1.108` (+ overlay `.10.5`) | 3 (with its USB-C hub) |
>
> - **The outlets moved** when the Pi 5's adapter went back on the strip: proven by load, the strip's names set to match
>   (`energy/strip_config_20261019.json`). Runs pass `--outlets pi4=0,jetson=1,pi5=2,laptop=3`. Anything below that says
>   "outlet 1 = laptop" is historical.
> - **The Pi 5's Wi-Fi is off** (it had joined a foreign network and routed its replies there). `ssh pi5` now points at
>   `192.168.1.122` and checks the host key known since 09-26 (`HostKeyAlias 192.168.100.107`).
> - **The laptop's Wi-Fi is on the same subnet as its cable** (since about 10-19). The boards resolve `192.168.1.108` to the
>   wired NIC; `measure_link.sh` measures the Pi 5–laptop pair on the overlay, where only the wired NIC has an address.
> - **Time:** the Pis sync to the laptop's overlay address `192.168.10.5`, so they lose time whenever the laptop's cable is
>   out; after a board reboot, check `timedatectl` before any run.
> - **Its rows:** every Pi 5 figure before 2026-10-06 is the old board's. The new board is characterised and compared under
>   `testbed/dataset/pi5b_predictions_20261019.json` (and, after its bootloader update, `pi5b_fw_predictions_20261020.json`)
>   before its rows join the dataset.

> **2026-10-04 to 10-06 — energy at the wall.** A Shelly Power Strip Gen4 (**`192.168.1.111`, reserved on the home router since 2026-10-09**; `192.168.100.34`, then `.33`, on the T628L: `energy/strip.json`) meters one outlet per node:
> 0 Pi 4, 1 laptop (with the USB-C hub its charger feeds), 2 Jetson, 3 unused (it fed the T628L; the never-switch-off rule
> was retired 2026-10-08) -- **until 2026-10-19; since then 0 Pi 4, 1 Jetson, 2 Pi 5, 3 laptop** (banner above). Every board,
> and one Pi 4 + Jetson pool, has registered energy-per-token figures. The method, every result and the setup checklist
> are in `energy.md`.

> **2026-10-09 — the four addresses are RESERVED on the home router** (Local Network → LAN → IPv4 → DHCP Binding; applied
> 19:06–19:10 in the user's logged-in router page, during the pilot, on the user's "Now"; confirmed by re-reading the
> router's own rule table). Each reservation keeps the device's lease at that moment, so nothing moved:
>
> | binding | MAC | reserved |
> |---|---|---|
> | `laptop` (eth `enp0s31f6`) | `48:2a:e3:2b:44:9d` | `192.168.1.108` |
> | `pi4` (`eth0`) | `d8:3a:dd:b4:30:97` | `192.168.1.110` |
> | `strip` (Wi-Fi `WIFI-A`) | `e8:f6:0a:62:b7:c0` | `192.168.1.111` |
> | `jetson` (`eth0`) | `00:04:4b:ec:52:f2` | `192.168.1.114` |
>
> The Jetson's lease had moved from `.109` to `.114` when the boards were power-cycled on 10-09, so every live script was
> updated to `.114` first; `compare_pool.py` maps `.109`, `.114` and `192.168.100.111` to the Jetson. The boards' Wi-Fi
> is off, so they have no other address on this network to reserve.

> **2026-10-08 — NEW ROUTER: the T628L was replaced by the home router (`192.168.1.1`, MAC `02:00:00:00:00:03`).** Its LAN
> ports and its Wi-Fi (`AUN_5G`) are one bridged network, and it has internet. The cluster now lives on **192.168.1.0/24**:
>
> | device | wired MAC | address (DHCP lease on 10-08; **reserved since 10-09**, banner above) |
> |---|---|---|
> | laptop `enp0s31f6` | `48:2a:e3:2b:44:9d` | `192.168.1.108` (its Wi-Fi was `192.168.1.105`, on the same subnet) |
> | Pi 4 `eth0` | `d8:3a:dd:b4:30:97` | `192.168.1.110`, plus the overlay `192.168.10.12` (unchanged) |
> | Jetson `eth0` | `00:04:4b:ec:52:f2` | `192.168.1.109` on 10-08; **`192.168.1.114` since 10-09** |
> | strip (Wi-Fi) | `e8:f6:0a:62:b7:c0` | `192.168.1.111` since it joined `WIFI-A` (10-08 18:16; found by `find_strip.py`) |
>
> - **The boards' Wi-Fi goes OFF (user's decision).** Both had joined `AUN_5G` on the same subnet as their Ethernet, and each
>   answered for its Wi-Fi address on the wire (ARP flux), so cluster traffic could cross to Wi-Fi unseen. With Wi-Fi off,
>   each board has one interface on 192.168.1.x. The Jetson's gate (`GATE-NET`) refuses a run otherwise.
> - **Time:** the Pi 4 still syncs to the laptop over the overlay (`192.168.10.5`). The Jetson pointed only at the laptop's
>   old `192.168.100.105`, so it syncs to public NTP now that the router has internet.
> - **ssh:** `pi4` and `jetson` point at the new addresses. Each still verifies the host key known since 09-26
>   (`HostKeyAlias`), so a different machine at the new address is refused.
> - **Isolation, a recorded deviation:** the topic form limits RPC to an isolated test network; the user accepted the home
>   network as is (2026-10-08). This is for the advisor.
> - **Scripts:** every live address was updated (`run_pool.sh`, `serve_pool.sh`, `predict_pool.py`, `qr3_tpot.py`,
>   `measure_link.sh`, `netshape.sh`, …), and `cluster.json` carries a dated note. Old logs keep their old addresses, and
>   `compare_pool.py` maps both. Everything below that says `192.168.100.x` is historical.

Control node is the ThinkPad T580 (`aun-ThinkPad-T580`, `192.168.1.108` on its Ethernet, reserved; the overlay
`192.168.10.5` is on the same port).

## How to connect

> **2026-10-05 — the Pi 5 has been returned.** Its data was collected first (`testbed/results/pi5_final_collect_20261005/`,
> `.handoff/nodes/Pi5/2026-10-05-1740-NOTE-pi5-returned.md`). The rows below that mention it are historical. The cluster is
> now the Pi 4, the Jetson and the laptop.

> **2026-09-26 — NETWORK MOVED to a True GIGATEX T628L (`192.168.100.1`, MAC `02:00:00:00:00:02`). DHCP enabled and
> reservations live.** Every device was re-cabled onto it; everything below that still says `192.168.1.x` is historical.
>
> | device | wired MAC | reserved address | verified |
> |---|---|---|---|
> | laptop `enp0s31f6` | `48:2a:e3:2b:44:9d` | `192.168.100.105` | holds it; default route **stays on the hotspot** (`never-default yes`) |
> | Pi 4 `eth0` | `d8:3a:dd:b4:30:97` | `192.168.100.106` | holds it + overlay `.10.12`; `ssh pi4` works |
> | Pi 5 `eth0` | `88:a2:9e:f0:e7:8a` | `192.168.100.107` | holds it + overlay `.10.11`; `ssh pi5` works |
> | Jetson `eth0` | `00:04:4b:ec:52:f2` | `192.168.100.111` | holds it (took it at boot, 13:37); `ssh jetson` works. Power guard restarted 13:38:12 |
>
> - **DHCP server:** pool **`.33`–`.120`** (widened from the default `.96`), lease 24 h, DNS proxy. **The T628L has no WAN**,
>   so no internet and no NTP reach the boards through it.
> - **Time: the laptop is the testbed's NTP server** (chrony, `testbed/scripts/laptop_ntp_server.sh`, run by the user
>   2026-09-26). It serves `192.168.100.0/24` and `192.168.10.0/24` (ufw opened for udp/123 from those two only) and
>   keeps itself in sync from Ubuntu's pools over the hotspot. All three boards point `systemd-timesyncd` at it
>   (`/etc/systemd/timesyncd.conf.d/heteroserve.conf`; the Pis also list the overlay `192.168.10.5` as a fallback).
>   On first sync they stepped **Pi 5 +10 h 54 m, Pi 4 +10 h 54 m, Jetson +13 h 15 m**; all now match the laptop to the
>   second. `local stratum 10` makes the laptop keep serving its own clock if it loses internet: the boards stay mutually
>   consistent, which is what cross-node log correlation needs. **Order after any Jetson reboot: let its clock sync
>   (`timedatectl status` → synchronized: yes) BEFORE starting the power guard**, or the guard log gets a clock step
>   mid-session (the summariser's open must-fix).
> - **Reservations must lie INSIDE the pool.** The router's page accepts an out-of-pool reservation silently and the
>   server then ignores it (proven: with the pool at `.33`–`.96`, the Pis were offered `.33`/`.34`; widened to `.120`,
>   they were offered `.106`/`.107`). Keep any new reservation inside `.33`–`.120`, or widen the pool.
> - **Boards are wired-only** (user decision): the Pis' `hs-wired` profile is now `ipv4.method auto` + `never-default no`,
>   keeping their static overlay address. Switched with an on-board auto-rollback timer armed and then disarmed.
>   Consequence to record: the **Wi-Fi link tier (QR9) is out** while the boards are wired-only.
> - `ssh pi5` / `pi4` / `jetson` now point at `.107` / `.106` / `.111` (`~/.ssh/config`, backed up alongside).
> - Router sign-in needs a CAPTCHA (the user does it). Navigating to the router's root logs the session out, and so does the
>   laptop's address changing.


    ssh pi5      # Raspberry Pi 5  -> hs-pi5,  192.168.100.107 (eth0, DHCP-reserved on the T628L)
    ssh pi4      # Raspberry Pi 4  -> stayaun, 192.168.100.106 (eth0, DHCP-reserved on the T628L)
    ssh pi5-eth  # Raspberry Pi 5  -> 192.168.10.11           (wired overlay, experiments)
    ssh pi4-eth  # Raspberry Pi 4  -> 192.168.10.12           (wired overlay, experiments)
    ssh pi5-via4 # Pi 5 through the Pi 4 -- use when its Wi-Fi lease has moved
    ssh jetson   # Jetson Nano     -> jetson-desktop, 192.168.100.111 (eth0, DHCP-reserved; board not up yet)

**All testbed addresses are now DHCP-reserved on the router (2026-09-25).** Before
that, the Pi 5 moved `.101` -> `.107` mid-session on 2026-09-20 and `ssh pi5` simply
stopped working while the board was perfectly healthy. Bindings on the F6107A
(Local Network -> LAN -> IPv4 -> DHCP Binding), each pinning the address the device
already held:

| binding | MAC | address |
|---|---|---|
| `jetson-eth` | `00:04:4b:ec:52:f2` | 192.168.1.111 |
| `jetson-wl` | `40:1c:83:41:38:9a` | 192.168.1.110 |
| `hs-pi5` | `88:a2:9e:f0:e7:8b` | 192.168.1.107 |
| `stayaun` | `d8:3a:dd:b4:30:98` | 192.168.1.106 |
| `thinkpad` | `48:f1:7f:1e:ad:70` | 192.168.1.105 |
| `shelly` | `e8:f6:0a:62:b7:c0` | 192.168.1.100 |

All six verified by reloading the page from the router. A binding takes effect at the
device's next lease renewal: `jetson-eth` already shows lease `Infinity`. `jetson-wl`
is live too since 12:18: applying it had kicked the Jetson's Wi-Fi lease back onto `.112`, and
a renewal did not correct it, so the lease file was set aside and `wlan0` reconnected. The router's name field caps at 10
characters. Two leases were deliberately *not* bound: the Lenovo tablet (`36:41:c7:…`)
and `V2348` (`fa:5b:0b:…`) use randomised, locally administered MACs (first octet ends
in binary `10`), which rotate, so a binding keyed to them would silently stop matching.

Still worth knowing:

- **`ssh pi5-via4` is the escape hatch.** It jumps through the Pi 4 to the *static*
  wired address, so it is immune to Wi-Fi lease changes. Authentication is still
  laptop -> Pi 5 with the laptop's key; the Pi 4 only forwards the TCP.
- **The mDNS warning that stood here is obsolete (2026-09-25).** It said `hs-pi5.local`
  resolves to the *wired* address `192.168.10.11`, which this laptop could not reach
  because its wired NIC `enp0s31f6` was DOWN. That NIC is now **up on the overlay at
  `192.168.10.5`** — `ssh pi5-eth` and `ssh pi4-eth` both verified from here — so the
  reason is gone. Prefer the `-eth` aliases: static, and immune to Wi-Fi lease moves.

(The DHCP reservations above were that fix.)

Both use key `~/.ssh/id_ed25519`. Aliases `hs-pi5` / `hs-pi4` also work.
Definitions are in `~/.ssh/config` (backups: `~/.ssh/config.bak.20260919-1337`, and
`~/.ssh/config.bak.20260925-jetson` from before the Jetson's `HostName` was corrected
`.104` -> `.111` — `.104` is a different device, see `node-jetson.md`).

## State of the cluster, 2026-09-25

| | Pi 5 | Pi 4 | Jetson Nano | RISC-V |
|---|---|---|---|---|
| Present | yes | yes | **yes — since 2026-09-25** | not yet (stretch) |
| SSH by key | yes | yes | yes | — |
| Hostname | `hs-pi5` | `stayaun` *(not renamed)* | `jetson-desktop` *(not renamed)* | — |
| Management address | 192.168.1.107 (Wi-Fi, DHCP) | 192.168.1.106 (Wi-Fi, DHCP) | 192.168.1.111 (**eth0**, DHCP) | — |
| Wired overlay | 192.168.10.11, 1000 Mb/s | 192.168.10.12, 1000 Mb/s | **none yet** | — |
| Password login | disabled | disabled | **enabled** | — |
| NOPASSWD sudo | yes | yes | yes (since 2026-09-25) | — |
| Swap | off | off | **zram on, ~2 GB** | — |
| Power mode | — | — | **MAXN since 08:23 — micro-USB, so idle/light load only** | — |
| Wi-Fi band | 5 GHz | 5 GHz | 5 GHz (−68 dBm) | — |
| cmake | installed | installed | 3.10.2 — **too old** (needs ≥ 3.14) | — |
| `llama.cpp` built | yes (`b23701f`) | yes (`b23701f`) | **no** | — |

## Measured baseline (2026-09-19; Wi-Fi unless noted — wired tier measured separately below)

| | Pi 5 | Pi 4 | Ratio |
|---|---|---|---|
| microSD seq read | 93.2 MB/s | 44.7 MB/s | 2.1x |
| MMC timing | SDR104 @ 200 MHz | DDR50 @ 50 MHz | — |
| Network, raw TCP | 18.5 MB/s | 10.7 MB/s | 1.7x |
| Wi-Fi PHY / signal | 433 Mbit/s / -40 dBm | 433 Mbit/s / -40 dBm | same |
| CPU dot-product (`asimddp`) | **yes** | **no** | — |
| 4.7 GB model load | ~50 s | ~105 s | 2.1x |
| `llama.cpp` prefill (pp512) | 81.10 t/s | 12.48 t/s | 6.4x |
| `llama.cpp` decode (tg128) | 18.16 t/s | 5.50 t/s | 3.0x |
| Build time | 570 s | 1735 s | 3.0x |

## Nine results worth knowing (full detail in `benchmarks.md`)

> Results 7 and 8 were re-measured at `-r 3` on 2026-09-25 — they had been single-sample
> `-r 1` figures. Every pooled number below is now a mean of three repetitions.

1. **Prefill is compute-bound, decode is memory-bandwidth-bound** — confirmed twice by
   independent means, on this project's own hardware. Decode peaks at **3 threads** on
   both boards and gets *slower* with 4.
2. **ARM dot-product is worth 2.96x on prefill and exactly nothing on decode.**
   Measured on one board with the instruction compiled in and out. Gives a placement
   rule directly: *prefill belongs on dot-product-capable nodes; decode does not care.*
3. **RPC distribution costs 1.9x against the worker alone** for a model that fits on
   one node — the cost is round-trip latency, not bandwidth.
4. **Link speed is really link *latency*, and it only affects decode.** Wired GbE vs
   5 GHz Wi-Fi: prefill +0.6%, decode +11.6%. Prefill batches its round trips; decode
   serialises one per token.
5. **7B Q8_0 fits on a single 8 GB Pi 5** (7.17 GiB model, no swap, no OOM) at
   1.51 tok/s decode (*indicative only, not reportable*: taken before the `-r 3` rule; `models.md`) — below the QR2 floor of 2 tok/s, and it means QR1 does not
   actually require pooling. A bigger model is needed to demonstrate that.
6. **QR1 PROVED (2026-09-20).** Qwen2.5-14B Q4_K_M (8.37 GiB) **fails to load on either
   node alone** — clean allocator refusal, zero kernel OOM records — and **runs pooled
   across both**. The experiment the project exists to run, and it lands on QR1's
   *stretch* target, not its MVP.
7. **The placement rule, measured — and the optimum is the memory boundary.** Throughput
   rises monotonically as layers move OFF the straggler, with no interior optimum: 36/12
   gives 1.20 pp64 / 0.51 tg16, the even 24/24 gives 1.66 / 0.63, and the best, **6/42,
   gives 3.78 / 0.97** — **+127% prefill and +54% decode over an even split**. Neighbouring
   splits' repetition ranges never overlap, and the splits were run in interleaved order, so
   the trend is not warm-up drift. One step
   further (4/44) the kernel OOM-kills it. *The optimal placement is the smallest offload
   that still fits.* Distribution here is a memory escape valve, not work-sharing.
   *Corrected 2026-09-27: these splits are `-ngl` values, and at this pin `-ngl N` offloads
   the output layer first. So "6/42" is really Pi 5 43 blocks / Pi 4 5 blocks + output head,
   and "4/44" is 45 blocks on the Pi 5. The measurements stand; see `benchmarks.md`
   "2026-09-27".*
8. **QR2 is comprehensively missed at that size** — 0.97 tok/s (0.967–0.970 over three
   repetitions) at the best split against a 2 tok/s floor, 2.1x short. QR1 and QR2 are in direct tension with numbers on both
   sides now.
9. **The failure mode changes at the boundary.** The whole model on one node is *refused
   cleanly* by llama.cpp's allocator (exit 1, no kernel OOM record); one step past the
   boundary the kernel *OOM-kills* the process (exit 137). With swap off there is no
   soft-landing region between them, so the placement engine must **predict** the
   boundary from measured per-layer footprint — it cannot safely find it by trial.
10. **The placement engine predicts the pool from single-board rows (2026-09-27).** Fed only
    TinyLlama Q4_K_M rows from each board, the 14B's tensor table and the link, the engine
    predicts all eight measured splits: decode within 3.5 % (mean 1.3 %), prefill within
    1.7 %. Its memory caps reproduce the boundary, and its rule picks **exactly the measured
    fastest split**. The characterisation dataset is in `testbed/dataset/`.
11. **Three boards, three models, pre-registered (2026-09-27 night).** The rule beats an even split by 44–45 % in
    decode on TinyLlama, Mistral-7B and Qwen2.5-14B. On the 14B it drops the Pi 4: **Pi 5 + Jetson reach 1.17 tok/s**,
    22 % faster than the best Pi 5 + Pi 4 split. The engine predicted the rank order of all five 14B configurations
    exactly. See `benchmarks.md`, "2026-09-27 (night)".
12. **The rule beats llama.cpp's own default split, pre-registered (2026-09-30).** On Qwen3-14B the engine's plan decodes
    at 834 ms/token (1.20 tok/s). llama.cpp's default split over every board (`--rpc pi5,pi4,jetson -ngl 99`) decodes at
    1,453 ms/token, and its RPC README setup at 1,993: **−42.6 % and −58.1 %**; prefill −69.3 % and −77.0 %. The default
    hands out blocks by reported memory, so the Pi 4, the slowest board per block, gets 16 of 40. The predictor priced all
    three within 2.8 %. See `benchmarks.md`, "DEFAULT".
13. **QR3, pre-registered (2026-09-30): only the mixture of experts meets the P95 target.** One user, the engine's serving
    plans at a 4096 context, 1,270 timed tokens a model. The P95 time per output token:
    - Qwen3-30B-A3B MoE **331.5 ms: MVP (≤ 500) met**;
    - Qwen3.5-9B 541, Mistral-Nemo 672, Qwen3-14B 855: not met;
    - none meets the stretch (≤ 250).
    A dense model's P95 is within 0.7 % of its mean, so its QR3 is its QR2. Serving decodes as predicted (within 3.2 %) once the
    Jetson's clocks are locked. Left to their governors, they cost 24 % of mean TPOT, so `serve_pool.sh` now locks them
    (user, 2026-09-30). See `benchmarks.md`, "QR3".
14. **Against prima.cpp, pre-registered (2026-09-30): the engine is 11 % faster per token and 45 % faster to first token.**
    Qwen3-14B on the Pi 5 + Pi 4, the boards prima.cpp can use (it cannot build on the Jetson), after porting Qwen3 into it.
    - The engine + llama.cpp: 1,189 ms/token, TTFT 45.2 s. prima.cpp with its Halda scheduler: 1,339 ms/token, 81.6 s.
    - Halda's split (24/16) is memory-bound, because prima.cpp keeps the output layer and embedding on its head device. Our cost
      model prices its split within 2 %, so the difference is placement, not runtime.
    - With the Jetson added (the engine only): 852 ms/token. See `benchmarks.md`, "PRIMA".

## What is blocking progress

1. ~~**No network switch.**~~ **Resolved without buying one.** The router (AIS F6107A)
   bridges 4 GbE LAN ports, 3 free, which covers the whole 4-node cluster. Both `eth0`
   are up at 1000 Mb/s on a private `192.168.10.0/24` overlay, and both the wired
   throughput tier and the wired-vs-Wi-Fi RPC A/B have been measured. A switch is only
   needed for a 5th node. **As of 2026-09-25** the laptop, both
   Pis and the Jetson are all cabled to the router (user), and the laptop's `enp0s31f6`
   is up on the overlay at `192.168.10.5`. That is four devices; if the router has the
   four LAN ports stated above, none is left, and the stretch RISC-V node needs a switch.
2. **Addresses are pinned on the wired overlay**, still DHCP on Wi-Fi management.
   Deliberate — see `JOURNAL.md`.
3. ~~**Jetson Nano not present.**~~ **Present since 2026-09-25, but not yet usable for
   measurement:** it is in **MAXN** (user's request, 08:23; passwordless sudo installed)
   but still powered from **micro-USB**, so it is safe only at idle and light load until a
   J25 barrel supply with J48 is fitted. `llama.cpp` is not built. Idle draw is 1.16 W in
   either mode. See `node-jetson.md`. The RISC-V node is still not present.
4. **No energy measurement.** The INA226 sensors and Pico 2 DAQ are not wired yet, so
   QR5–QR7 are untouched. **This is now the largest untested part of the proposal** —
   with QR1 proved and the placement direction measured, energy is the main gap left.
   **2026-10-04: the INA226 rig was never bought; the instrument is now a Shelly Power Strip Gen4 (AC wall power per outlet, 1 Hz refresh, so QR7 is not met). Tool, method and the user's setup checklist: `energy.md`.**
   **2026-10-05: first measurements.** The outlet map was confirmed by load, and the idle floors were measured. The **first energy per token**: Pi 5 with TinyLlama-1.1B Q4_K_M, prefill 0.093 and decode 0.357 J/token, net at the wall by the strip's counter. Integrated power reads a steady 5 % lower, and which is right needs QR6. One board and one model; no placement has been measured yet (`energy.md`). **Then the Pi 4 at four quantisations:** prefill 0.415–0.48, decode 0.69–0.98 J/token. Q4_K_M costs 4.6× the Pi 5's per prefill token and 2.0× per decode token.
5. **Resolved 2026-09-29: one revision again.** The Jetson joins pooled runs at the pin as a Vulkan or a CUDA worker (CUDA = the pin
   + `testbed/patches/jetpack46-cuda102-b23701f.patch` v2). `llama.exceptions` is empty; see `benchmarks.md`, 2026-09-29.
   **Its limit, 2026-09-29 evening:** its Vulkan worker segfaults inside NVIDIA's driver when its model buffer passes ~2.9 GB, so the
   engine caps it at 2,845 MiB (`cluster.json` `max_weights_mb`). That includes the qwen35 hybrids, up to 16 middle blocks of the
   27B. The afternoon's "1.25 GB optimistic" reading was a misread of `-ngl`, and is withdrawn. What follows is the history. ~~**The Jetson is scoped out of pooled runs for now**~~ — it keeps CUDA/FP16 on b5050
   while the Pis are on `b23701f7`, a declared two-revision split. Getting it into
   pooled runs *with* its GPU is scoped in `Jetson/docs/50_pooled_gpu_options.md`;
   the first thing to test is whether `-DGGML_RPC=ON` builds on a Nano at all.
   **Update 2026-09-25:** that doc's deciding check has now been run on the board — the
   Nano's Vulkan driver reports **`shaderFloat16 = true`** (read straight from the driver;
   the doc's `vulkaninfo | grep` is silent on 18.04 either way). By the doc's own rule that
   makes **Path 1, the Vulkan backend at `b23701f7`**, the one to take: one revision with
   the Pis, GPU kept, no CUDA patches. Not yet built.

## Two things that contradict the proposal

- The **Pi 4 has 8 GB**, not the 4 GB written in the budget table. Pooled memory is
  larger than planned, which changes the QR1 arithmetic in your favour.
- The **Pi 5 has 8 GB**, which matches the budget. But QR1 targets 7B at Q8_0
  (~7.5 GB), and that no longer fits on this node alone — which is the premise of
  pooling, so it is arguably the point, but be deliberate about it.

## Files here

| File | What it holds |
|---|---|
| `README.md` | this page — current state |
| `JOURNAL.md` | what was done, when, and why |
| `node-pi5.md` | Pi 5 facts and measurements |
| `node-pi4.md` | Pi 4 facts and measurements |
| `node-jetson.md` | Jetson Nano facts — power mode, the Vulkan fp16 probe, what is blocked |
| `benchmarks.md` | every measurement in full, with method |
| `models.md` | model provenance, upstream revision and SHA-256 |
| `REPRODUCE.md` | how to re-derive every figure from the raw data (`scripts/reproduce.py verify`), re-run an experiment, and use the CSV export and manifest |
| `results/` | raw logs, tee'd — the evidence behind `benchmarks.md` |
| `scripts/` | the scripts that produced them |
