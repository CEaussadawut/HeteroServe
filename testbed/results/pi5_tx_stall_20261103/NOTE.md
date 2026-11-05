# The Pi 5 MTP hang: the Pi 5's network driver holds a reply (2026-11-03, 19:30–19:45)

**What hung.** MTP speculation (`--spec-type draft-mtp`, 2 or 3 drafts) with the MTP block on the Pi 5, the plan's last worker.
That happened twice in sp10 (`testbed/results/sp10_20261103_151444/NOTE.txt`) and once more here, deliberately:
- the reproduction: Qwen3.5-9B Q4_K_M on laptop 24 + Pi 5 9, MTP 2;
- the warm-up completed; the first real request hung, and the next three timed out behind it (`requests.log`).

**The live state at 19:43, 8 minutes into the hang** (`live_state_during_hang.txt`):
- **The Pi 5's side** of the RPC connection (:50052):
  - 2,896 bytes (two segments) sent and never acknowledged; rto at its 120 s maximum;
  - during the hang, TcpExtTCPTimeouts rose 40 → 59 and TcpExtTCPSpuriousRtxHostQueues 41 → 62. Each RTO found the original
    segment "still in the host queue" and did not resend it. TcpRetransSegs rose only 1 → 3.
- **The Pi 5's driver** (macb, hardware TSO on, PTP clock registered at boot): the qdisc held nothing (backlog 0) and the
  transmit queue's byte queue limit showed 0 bytes in flight. So the driver had reported the bytes sent while still holding
  the packet. TCP waits for a held packet to be freed before it resends one; it never was.
- **The laptop's side:** nothing unsent, nothing unread, last data 167 s before. It was waiting for exactly that reply.
- **The end:** after tcp_retries2 (15) the Pi 5's kernel abandons the connection. That is sp10's 15.7 minutes and the
  "recv failed (bytes_recv=0, size_to_recv=1)" in the Pi 5's RPC log. The laptop's server then blocks or aborts on its
  next send.

**Not the laptop.** An earlier hypothesis blamed the laptop's ufw dropping packets its strict connection tracking marks INVALID.
It is withdrawn:
- the reply never left the Pi 5;
- ufw is disabled in `/etc/ufw/ufw.conf` (ENABLED=no; `systemctl is-active ufw` reports only that its start unit ran).

**Why the Pi 5 only.**
- The Pi 4 (bcmgenet) runs with hardware TSO **off** and the Jetson has another driver. With the MTP block on the Pi 4
  (exploration A, sp10c P4) or the Jetson (sp10c P3), every MTP serve completed.
- MTP puts the drafting's small round trips on the last worker: each draft is one RPC visit. That is the traffic the Pi 5
  failed on, two or three drafts at a time.
- Non-MTP serving on the Pi 5 has not hung, in hundreds of serves.

**The guard (in our code).** `serve_pool.sh` now refuses MTP when the Pi 5 is the plan's last worker, naming this folder.
- `SERVE_MTP_PI5=1` tries anyway.
- Test: `test_planner_v4.py`, `test_serve_pool_refuses_mtp_with_the_mtp_block_on_the_pi5`.

**The fix to try (a system setting on the Pi 5, for its owner to run).** Turn the Pi 5's hardware TSO off, as the Pi 4 has it.
The kernel then segments in software:

    ssh pi5 'sudo ethtool -K eth0 tso off'          # until the next reboot

The check after that: the reproduction above, with `SERVE_MTP_PI5=1`, three times. Each hung within its first request before.
If it holds, make the setting persistent (e.g. a udev rule running `ethtool -K eth0 tso off` on eth0's add). If it does not,
the next suspects are generic segmentation offload (`gso off`) and the Pi 5's kernel (6.18.50+rpt-rpi-2712).

**21:45 addendum: not only with the MTP block on the Pi 5.** sp10d (`testbed/results/sp10d_20261103_203213`) ran two 9B plans
with the Pi 5 a middle worker and the Pi 4 last. Three requests hung, each ended by the runner's 600 s timeout:
- P5's no-speculation serve, on the scan's first batch of 8 tokens (20:33:08);
- P5's MTP 3, on copy request 1;
- P6's MTP 3, on free request 0.
The Pi 5's counters rose from 62 to 103 (TcpExtTCPSpuriousRtxHostQueues) and from 59 to 97 (TCPTimeouts) over the
reproduction's tail and sp10d. The Pi 4's are zero. So the stall can hit any pooled plan with the Pi 5 under small-batch
traffic: the scans' tiny prompts, MTP's drafts. Long prompts and steady decoding (q9b, jmoe, every earlier session) have not
hit it. serve_pool.sh's guard now refuses MTP with the Pi 5 anywhere in the plan (SERVE_MTP_PI5=1 overrides). The board-side
fix to try is unchanged: `sudo ethtool -K eth0 tso off` on the Pi 5, by its owner.

**22:20, the fix holds.** The owner turned the Pi 5's hardware TSO off at about 22:00 (`sudo ethtool -K eth0 tso off`;
tcp-segmentation-offload off, GSO on). `check_pi5_tso_20261103.py` then ran the serve that had hung on its first request every
time: the 9B on laptop 24 + Pi 5 9 with MTP 2, three rounds (`testbed/results/pi5_tso_check_20261103_220804`).
- **3 of 3 rounds hold.** Every request answered: the warm-up, the scan's 12 tiny prompts and four 128-token requests per
  round.
- **The counters did not move:** SpuriousRtxHostQueues 103 → 103, TCPTimeouts 97 → 97, RetransSegs 41 → 41.
- MTP 2 ran at 260–342 ms a token with this serve's usual acceptance (72 of 109 drafted, free; 80 of 92, copy).
- serve_pool.sh's guard now asks the Pi 5 whether its TSO is off and allows MTP if so.
- **Not persistent:** TSO comes back on at a reboot. The Pi 5 runs NetworkManager (connection `hs-wired`), so the persistent
  form, for the owner to run, is `sudo nmcli connection modify hs-wired ethtool.feature-tso off`.
- sp10d is re-run as sp10e (22:21).
- **22:21, made persistent by the owner:** `sudo nmcli connection modify hs-wired ethtool.feature-tso off`. Checked read-only:
  the profile's ethtool.feature-tso reads off, the runtime TSO is off, the counter is still 103. The modify only edits the
  stored profile, so sp10e's serve, starting then, kept its link.
