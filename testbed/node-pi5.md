# Node: Raspberry Pi 5 (strong CPU worker)

First contact 2026-09-19. Control node: aun-ThinkPad-T580.

| Field | Value |
|---|---|
| Model | Raspberry Pi 5 Model B Rev 1.1 |
| Board revision | `d04171` (8 GB) |
| Serial | `b266e1877252f91e` |
| SoC / CPU | BCM2712, Cortex-A76 ×4 @ 2.40 GHz |
| Memory | 7.9 GiB usable |
| OS | Debian GNU/Linux 13 (trixie), arm64 userland |
| Kernel | 6.18.50+rpt-rpi-2712 |
| Storage | mmcblk0, 58.3 GB microSD — p1 512 M vfat `/boot/firmware`, p2 57.8 G ext4 `/` |
| Hostname | `hs-pi5` (renamed 2026-09-19; `hs-pi5.local` resolves) |
| Address | **since 2026-09-26: eth0 `192.168.100.107`/24**, DHCP-reserved on the T628L (`192.168.100.1`), plus static overlay `192.168.10.11`/24; Wi-Fi off (wired-only). *Before:* wlan0 `192.168.1.107` on the F6107A, reserved 2026-09-25 (it had moved .101 → .107 on 2026-09-20) |
| eth0 | DOWN — no switch yet |
| Idle temp | 46.1 °C, `get_throttled=0x0` |
| Swap | **none** — zram disabled 2026-09-19, verified across a reboot |

## SSH

Reached as `ssh pi5` (alias `hs-pi5`) — user `pi5`, key `~/.ssh/id_ed25519`, `IdentitiesOnly yes`.
SSH was not enabled by the Imager image; `sshd` was started on the board by hand.
Password authentication is **disabled** — `/etc/ssh/sshd_config.d/10-hetero-hardening.conf`
sets `PasswordAuthentication no`, `KbdInteractiveAuthentication no`,
`PermitRootLogin prohibit-password`. Original config saved as
`/etc/ssh/sshd_config.bak.20260919`. Key auth is now the only way in.

`pi5` is in group `sudo` but has **no NOPASSWD rule** — remote `sudo` prompts for a
password. Anything unattended (benchmark harness, energy collection, governor or
thermal changes) will block on that until a sudoers drop-in is added.

Host key fingerprints accepted 2026-09-19 (TOFU, verify against the board before trusting):

    RSA     SHA256:yakmlGgjRQJ+rBBIMSoGYldFITZjo+Us2PbrueL5ZLY
    ECDSA   SHA256:PAc7Ocy75sfajsJgnQBjvhFpmZ+ZrLSuxMhd8vv0oTk
    ED25519 SHA256:GwGu3f6Uv9ET2tIWC+KEohc1wA6D6MpUi1rTVf+Y1Ug

## Settled 2026-09-19

- zram swap disabled. Masked `systemd-zram-setup@zram0.service`,
  `rpi-zram-writeback.timer`, `var-swap.swap`; added
  `/etc/systemd/zram-generator.conf.d/99-heteroserve-no-swap.conf`.
  Note the vendor drop-in already carried `host-memory-limit=0` and zram ran anyway —
  Pi OS orchestrates `dev-zram0.swap` itself as a "rpi-swap managed swap device
  (zram+file)". Masking alone was not provably enough, hence the reboot check.
  Post-reboot: `/proc/swaps` empty, `zram0` 0 B.
- Renamed `raspberrypi` -> `hs-pi5` (`hostnamectl` + `/etc/hosts` 127.0.1.1).
  `raspberrypi.local` is now unclaimed and free for the BlowCount board.
- Password authentication off; key-only.

## Measured 2026-09-19

Storage, `dd` with `O_DIRECT` (bypasses page cache, needs no root):

| | |
|---|---|
| Sequential read, 512 MB | **93.2 MB/s** |
| Sequential write, 512 MB | 46.2 MB/s |
| Random 4 K read, 4096 blocks | 11.6 MB/s |
| Card | Phison (`manfid 0x27`, `oemid 0x5048`), fw `0x1` |
| MMC bus | **SDR104, 200 MHz, 4-bit** |

Network, raw TCP via `nc` (no SSH crypto in the path):

| | |
|---|---|
| Laptop -> Pi 5 | **18.4 MB/s** (147 Mbit/s) |
| Same over SSH aes128-gcm | 13.8 MB/s |
| TCP connect RTT (power-save off) | 6.7–13.0 ms |
| Wi-Fi | `AUN_5G`, ch 100, 5500 MHz, signal 82, `brcmfmac` |

At 93.2 MB/s a 4.7 GB model loads in **~50 s**, against ~105 s on the Pi 4.

## Toolchain (checked here directly)

gcc/g++ 14.2.0, make 4.4.1, git 2.47.3, python 3.13.5, pip 25.1.1 present.
**cmake 3.31.6 and ccache 4.11.2 installed 2026-09-19.**
Note `iw` is at `/usr/sbin/iw`, not on a normal user's `PATH` over SSH.

CPU features: `asimd crc32 fphp asimdhp asimdrdm asimddp`.
`asimddp` (dot product) is present and is the one that matters for quantised
`llama.cpp` kernels. **No `i8mm`, no `sve`** — the A76 in the Pi 5 has neither, so
kernels gated on those will fall back.

## Open items that affect measurement validity

1. **Address is still DHCP — deliberately deferred.** Pinning is worth doing on the
   wired cluster segment, not on household Wi-Fi that the experiments will leave
   anyway. Recorded so drift is detectable:

   | | |
   |---|---|
   | wlan0 MAC | `88:a2:9e:f0:e7:8b` |
   | Lease | 192.168.1.101/24, 86400 s, from 192.168.1.1 |
   | DNS handed out | 8.8.8.8, 8.8.4.4 |
   | Wi-Fi | `AUN_5G` (NetworkManager) |

   Router is a **F6107A** GPON ONT at 192.168.1.1 (MAC `02:00:00:00:00:04`). The user
   holds the admin login; DHCP reservations were added 2026-09-25 (see `README.md`).
   Observed leases all sit at .101+, so the pool probably starts at .100 and
   192.168.1.2–.99 is probably safe static space. That is an inference from four
   observed leases, not a fact read off the router — verify before relying on it.
2. **eth0 down.** Link tier is an independent variable in this study; every result
   until the switch arrives is Wi-Fi Tier 1 only, and Wi-Fi shares the band with
   household traffic. Do not mix these runs with later wired runs.
3. *(settled)* Wi-Fi power-save was already off on this node; the Pi 4's was on.
   Latency is now stable at 6.7–13.0 ms on both.
4. **No NOPASSWD sudo.** Blocks unattended runs that need privilege. Note polkit
   also refuses `nmcli con mod` over SSH (`Insufficient privileges`) — there is no
   active local session for it to authenticate against, so network changes need
   real root, not group membership. `pi5` is already in `netdev` and that is not
   enough.
