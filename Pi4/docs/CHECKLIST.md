# Pi 4 — bring-up and per-run checklist

Print this. Tick it. The failures it prevents are the ones that are invisible at the time
and fatal three weeks later.

## A. One-time bring-up

- [ ] Raspberry Pi OS **Lite 64-bit** flashed (not 32-bit, not desktop)
- [ ] hostname `pi4`, SSH key-only, locale `Asia/Bangkok`
- [ ] heatsink-and-fan case fitted
- [ ] official 5.1 V 3 A USB-C PSU, with **its own cable**
- [ ] static address `192.168.50.13`, wired into the managed switch
- [ ] `sudo rpi-eeprom-update -a`, bootloader date recorded in the run log
- [ ] `make bootstrap` — apt, toolchain, chrony pointed at the control node
- [ ] `chronyc tracking` shows offset < 5 ms
- [ ] `make tune` — governor `performance`, swap **0 kB**, eth0 at 1000 Mb/s
- [ ] `make build` completes; SHA noted
- [ ] **`LLAMA_PIN_SHA` filled into `config/node.env` on every node**
- [ ] `make models`, checksums written
- [ ] `make install` — units enabled, `visudo -c` clean
- [ ] `make test` — the off-target suite passes (needs no Pi; catches a bad deploy early)
- [ ] `make health` prints **READY** (it now also gates on `llama-bench --progress`)
- [ ] `make info` saved
- [ ] `make bench-quick` completes and parses to `results.csv`

## B. Power channel (once, then re-verify monthly)

- [ ] INA226 high-side on the 5 V rail, 0.01 Ω shunt, IN+ from PSU / IN− to Pi
- [ ] I²C address **0x41**, distinct from the other three channels
- [ ] calibration factor measured at 2–3 points (`I = V/R`, both values measured)
- [ ] calibration factors stored **with the dataset**, not in a notebook
- [ ] idle reading lands in 2.7–3.5 W; load reading in 5.5–6.5 W
- [ ] measurement boundary written into the report: SoC + RAM + SD + PHY + **fan**

## C. Before every measurement run

- [ ] `make health` → READY (do not proceed past a FAIL)
- [ ] `vcgencmd get_throttled` → `0x0` (sticky bits need a reboot)
- [ ] SoC below 70 °C
- [ ] no leftover `tc` shaping, unless this run is a shaped point
- [ ] link tier set explicitly and written in the run log
- [ ] nothing else running on the board — no upgrades, no polling loops, no `tail -f`
- [ ] DAQ streaming, control-node collector recording
- [ ] run ID agreed with the control node

## D. After every run

- [ ] `results.csv` has no unexplained `flags`
- [ ] `energy_window` is `reps-span` on every row — a `process` value means the window
      derivation fell back and that row's joules include model load and a full warmup
- [ ] `energy_duty` > 0.99 on every row. Below 0.98 is flagged `low-duty:` — but **only on
      `reps-span` rows**. A `process` row is flagged `energy-span:whole-process:` instead
      and its duty is *expected* to be far below 0.98, so check `energy_window` first
- [ ] no `throttled:` flag in `results.csv`. **Do not** check `throttled_end` == `0x0` in
      `windows.jsonl` — that word is *sticky since boot*, so after one blip it stays set for
      the rest of the sweep. The `throttled:` flag compares the word at the start of each
      run against the word at its end, which is what says whether *that row* was affected
- [ ] `temp_end_c` is not trending upward across the sweep
- [ ] energy join completed on the control node; `joules_per_token` populated
- [ ] `sudo scripts/05_net_tier.sh clear`
- [ ] data pulled: `make collect`
- [ ] run logged: date, link tier, run ID, llama.cpp SHA, anything unusual

## E. Straggler-study runs (QR8)

- [ ] thread-limited conditions (`-t 1`, `-t 2`) recorded alongside full `-t 4`
- [ ] cluster runs recorded **with** and **without** this node, same model, same prompt
- [ ] throughput *and* joules per token compared for both
- [ ] result recorded even when it is negative — "adding this node made things worse" is
      the answer the research question is asking for
