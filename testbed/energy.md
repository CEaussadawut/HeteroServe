# Energy: the instrument, the method, the plan

*Started 2026-10-04. Energy per token is half of the project's claim and has not been measured yet. The INA226 sensors and
the Pico 2 DAQ of the proposal were **never bought**. The instrument for Project 1 is a **Shelly Power Strip Gen4**,
read by `testbed/scripts/shelly_strip.py`.*

**Status, 2026-10-05 14:45.** The tooling is built, tested (35 unit tests, 8 of 8 mutations caught) and smoke-tested
read-only (2026-10-04). Since ~14:12 the strip is **on the router's Wi-Fi "WIFI-B" at `192.168.100.34`** (DHCP, not reserved
yet). **First measurements, 2026-10-05:** the outlet map (0 Pi 4, 1 Pi 5, 2 Jetson, 3 router) is **confirmed by load**, and
the **idle floors** are measured. They show a systematic gap: integrated `apower` reads 3.8–5.6 % below the strip's own
counter on every outlet (see "Measured at home"). **The first energy per token** (15:23) is in: the Pi 5 with
TinyLlama-1.1B Q4_K_M, prefill 0.093 and decode 0.357 J/token, net at the wall by the counter, 5 % lower by the trapezoid
(see "Energy per token on the Pi 5"). That is one board and one model; no placement has been measured yet. The 10-04 smoke-test load on outlet 3 was a
device of the user's, **not the router**.

| Requirement | As found |
|---|---|
| QR5 energy per 1,000 tokens | **First figures 2026-10-05**, one board and one model, not yet per placement: the Pi 5 with TinyLlama-1.1B Q4_K_M, prefill 0.026 and decode 0.099 Wh per 1,000 tokens (net at the wall, by the counter) |
| QR6 error against a reference | **No reference instrument yet**. Not measured |
| QR7 sampling rate ≥ 10 Hz (MVP) | **Not met.** Each outlet's readings refresh **once a second** (1 Hz measured) |

---

## The instrument

| | |
|---|---|
| Device | Shelly Power Strip Gen4, model `S4PL-00416EU`, gen 4, app `PowerStrip` |
| Id / MAC | `shellypstripg4-e8f60a62b7c0` / `E8:F6:0A:62:B7:C0` |
| Address | **`192.168.100.34`** on the T628L's Wi-Fi "WIFI-B", since 2026-10-05 ~14:12 (DHCP, **not reserved yet**; mDNS `shellypstripg4-e8f60a62b7c0.local` also resolves). Until then it was `10.67.218.193` on the phone hotspot "WIFI-C". **Never hard-code it.** The address lives in `testbed/energy/strip.json` only, and `--host` overrides it |
| Readings per outlet | `output`, `apower` (W, 0.1 W shown), `voltage` (V, 0.1), `current` (A, 0.001), `pf` (0.01), `freq` (Hz, 0.1). Also `aenergy.total` (Wh, **3 decimals = 1 mWh = 3.6 J**) and `aenergy.by_minute` (mWh, 3 decimals, the last three *complete* minutes, `[0]` the newest), with `minute_ts` (the start of the current minute). `source` says what last switched the outlet |
| Protections (all four outlets) | `power_limit` 3360 W, `current_limit` 12 A, `voltage_limit` 280 V; `auto_on` / `auto_off` off |
| Mains, as seen | 228.8–232.1 V, 49.9–50.0 Hz |
| Clock | SNTP `time.cloudflare.com`. `sys.unixtime` agreed with the laptop within ±1 s (whole seconds) over the hotspot. The logger timestamps on the laptop anyway |
| Other ways in | cloud off, MQTT off, websocket off, KNX off, BLE RPC off; **Matter on**. Each of these could switch an outlet |

**Two states, both read from the strip** (the "after" state twice by the tool, 20:22:55 and 20:36:12):

| | As found, 2026-10-04 ~20:05 | After 20:10–20:14 |
|---|---|---|
| Firmware | `2.0.1-beta3` (`20260917-100656/2.0.1-beta3-g31658db`) | **`2.0.1` stable** (`20260923-075623/2.0.1-ge1a198b`) |
| Updates offered | stable 2.0.1, beta 2.1.0-beta2 | beta 2.1.0-beta2 only |
| Outlet names | all null | `pi5`, `pi4`, `jetson`, `router` |
| `initial_state` (power-on default) | `off` on all four | `on` on all four |
| `auth_en` | false | false |
| Wi-Fi | `sta` "WIFI-C", `sta1` empty and disabled | unchanged |
| Outputs | all four off | 0–2 off, 3 on (the user's device) |

- **The changes were applied by the main session with the user's go-ahead** on 2026-10-04: names and `initial_state` at ~20:10,
  needing no restart; the firmware update at 20:12:19, with the strip back by 20:13:42. **This tool changed nothing.** It is
  read-only by construction.
- **After the update's reboot the outputs did NOT follow `initial_state` "on".** Outlets 0–2 stayed off (`source` "init")
  and outlet 3 stayed on. So `initial_state` evidently applies at a mains power-up, not at a software reboot. **Unverified for
  a real power cut:** only pulling the strip's plug would show it. Until that is done, a blackout leaving the boards
  unpowered is still possible.
  - **Verified at a real power-up, 2026-10-05.** At 13:59:03 the strip's uptime was 773 s (it was plugged in at home at
    ~13:46), and **all four outlets were on with `source` "init"**. So `initial_state` "on" holds after mains returns.
- **Outlet 1 read `source` "short_push" (still off) at 20:22:55.** Its physical button was pressed after the reboot. That is
  recorded as found; no tool switches outlets.
- **The "Shelly plug" of earlier records is this strip.** It has the same MAC: the F6107A binding `shelly`
  `e8:f6:0a:62:b7:c0` → `192.168.1.100` (`README.md`), and "a Shelly metering plug" (`JOURNAL.md`, 2026-09-19). The
  2026-09-26 handoffs say it "meters the control node". Under the user's outlet map it meters the boards and the router, and the
  laptop is not metered.

## What it measures, and what it does not

- **AC wall power per outlet.** That includes each power adapter's conversion loss. The planned INA226 chain would have read
  the board's 5 V DC rail after the adapter. So a board's figure here is its adapter plus the board, and it is larger than its
  DC rail figure by the adapter's loss. The loss is not measured. Small switch-mode adapters are generally least efficient at
  light load, so the gap is expected to matter most at the few-watt idle floors (an expectation, not a measurement).
- **Outlet 2's DC cross-check is the Jetson's on-module INA3221.** That is root `tegrastats` `POM_5V_IN`, channel 0. It is
  the **module input only**: carrier-board loads (USB, the M.2 Wi-Fi card, the fan) are outside it (`node-jetson.md`).
  Outlet 2 must therefore read at least `POM_5V_IN`. The difference is the adapter's loss plus the carrier loads, and it cannot
  be given to either alone.
- **No brownout signal.** The strip reads mains voltage, not the board's 5 V rail. The proposal's "bus voltage as a brownout
  detector" (≥ 4.75 V) does not exist with this instrument. Under-voltage stays with `vcgencmd get_throttled` on the Pis
  (house rule: log it with every run) and the Jetson's power guard (INA3221).
- **The router is metered** (outlet 3). It belongs to no board's figure. Whether a placement's energy includes the network
  is a reporting decision.
- **The laptop is not metered.** `cluster.json` already puts the T580 outside the measurement boundary, and the report must
  say so.
- **Low-power behaviour is unknown.** Switched-off outlets read exactly 0.0 W, 0.000 A and pf 0.00. The boards idle at a few
  watts, and how the meter reads at that level is the first thing to check at home.

## Measured: refresh, request rate, counter and agreement (2026-10-04, over the hotspot)

**Before this tool** (main session, read-only, ~20:05): `Shelly.GetStatus` at ~7 requests/s for 20 s. RTT median ~100 ms
(max 754), 2 errors in 138 requests, and the **4-outlet voltage vector** changed every ~0.65 s (median).

**With the tool**, firmware 2.0.1, outlets 0–2 off, outlet 3 carrying the user's device: 17.0–58.3 W, 0.157–0.497 A,
pf 0.44–0.51, swinging from second to second. Two 45 s logs, one at 2 Hz and one at 5 Hz, each summarised over a 39 s window
inside it. The logs are kept outside the repo; these are the conclusions.

| | 2 Hz requested | 5 Hz requested |
|---|---:|---:|
| Log | 20:23:26–20:24:11 | 20:24:11–20:24:57 |
| Requests / failed | 71 / **0** | 195 / **0** |
| Achieved request rate, whole log (window) | 1.57/s (1.61) | 4.26/s (4.41) |
| RTT median / p90 / max | 0.103 / 0.928 / 3.024 s | 0.101 / 0.135 / 1.154 s |
| Requests stalled > 0.5 s | 8 | 7 |
| **Outlet 3 (loaded): fresh `apower` / `current` per s** | **0.92 / 0.89** | **0.96 / 0.96** |
| Outlet 3: median interval between fresh readings | 1.001 s | 1.000 s |
| Outlets 0–2 (no load): fresh `voltage` per s | 0.43 / 0.59 / 0.56 | 0.60 / 0.53 / 0.63 |
| Outlet 3: mean / peak `apower` | 24.60 / 57.2 W | 24.36 / 58.3 W |
| Outlet 3: energy, trapezoid / counter | 960.8 / 996.2 J | 960.4 / 1042.3 J |
| **Outlet 3: disagreement (trapezoid − counter) / counter** | **−3.55 %** | **−7.86 %** (flagged) |
| Counter: observed step / update interval | 0.006 Wh (21.6 J) / 1.00 s | 0.007 Wh (25.2 J) / 1.00 s |
| Counter quantisation (one step at each end / energy) | 4.3 % | 4.8 % |

What that says:

1. **Each outlet refreshes once a second.** Under load, `apower` and `current` changed at a median interval of 1.000 s at
   either polling rate. Polling at 5 Hz showed 0.96 changes a second and 2 Hz showed 0.89–0.92, with the request stalls
   costing the difference. These are lower bounds: a repeated value cannot be seen. Faster polling does not give faster
   readings. **The freshness signal:** on outlets 0–2, with no load, `apower` and `current`
   are a constant 0. There it was **voltage changes**, which undercount: at 0.1 V resolution a fresh voltage often equals the
   one before. On outlet 3, voltage showed 0.54–0.55 changes/s while `apower` showed 0.92–0.96.
2. **The outlets refresh in two pairs, at different instants.** Outlets 0 and 1 change together at one point of the second,
   and outlets 2 and 3 together at another. As median phase in the laptop's second: 0.33 against 0.83 in the 2 Hz log, 0.12
   against 0.92 in the 5 Hz log. Outlet 3's `apower` and counter change with its pair. That fits the earlier "~0.65 s" for the
   4-outlet vector: two 1 Hz refreshes staggered within the second (an observation; the strip's internals are not known).
   The voltages within a pair are close but not identical, and the pairs differ by ~0.5 V.
3. **The counter.** `aenergy.total` shows 0.001 Wh (3.6 J) and advanced once a second, in steps of 0.006–0.007 Wh at
   ~24 W. `by_minute[0]` held still within a minute and moved when `minute_ts` did, so it is the last *complete* minute:
   348.856 mWh for 20:22, then 411.617 mWh for 20:23 (a 24.70 W mean), at 0.001 mWh resolution. A minute-aligned
   `by_minute` check against `total` needs a log longer than a minute; it was not done (the smoke tests were capped at 60 s).
4. **Trapezoid against counter: −3.55 % and −7.86 % on 39 s windows.** One counter step at each end of a 39 s window is
   4.3–4.8 % of its energy, so part of that is the window edges. The edges are not all of it, though. Taking each log between
   two counter-change instants (no counter edge error), the gap was **−0.5 % over 43.5 s** at 2 Hz (1089.0 J against 1094.4 J)
   but **−8.6 % over 44.9 s** at 5 Hz (1086.1 J against 1188.0 J). The trapezoid read low in both logs. **Cause not
   established.** The likely reading is that 1 Hz `apower` snapshots under-sample a load that swings within the second, while
   the counter integrates inside the meter. That is not verified: whether `apower` is a 1 s average or a shorter snapshot is
   not documented here. On fluctuating loads, the trapezoid cannot be trusted at QR6's 5 % yet. **2026-10-05:** the gap is
   also there on steady idle loads, on every outlet, and on the router under-sampling cannot explain it. See
   "Measured at home", B.2.
5. **Request stalls.** 8 of 71 and 7 of 195 requests took 0.5–3.0 s, about the same count per minute at either rate. The
   ~1 s and ~3 s values look like TCP retransmission over the hotspot (not established). No request failed. Re-measure on
   the router's Wi-Fi.

## Measured at home: the outlet map and the idle floors (2026-10-05)

- **Run:** `testbed/results/energy_mapcheck_idle_20261005_142645/` (14:26:45–14:39:19), by
  `testbed/scripts/energy_mapcheck_idle.sh`.
- **Scoring:** `testbed/scripts/score_energy_mapcheck_idle.py`, against the registration
  `testbed/dataset/energy_predictions_20261005.json` (14:26:38, sha256 `7e147d64…`, read-only). Results are in
  `testbed/dataset/energy_results_20261005.json`.
- **Logging:** the strip was on "WIFI-B" (RSSI −9 dBm), logged at 5 Hz from the laptop over Ethernet → T628L → Wi-Fi. 1,644 +
  1,794 requests, **0 failed**.
- **Firmware and MAC** were unchanged through both logs, and no strip restart was flagged.

**Conditions:**
- The boards were booted ~13:46 by the strip's power-up, and were idle (load average ≤ 0.1).
- The Pis' desktop sessions were running (`pcmanfm`, `wf-panel-pi`, 3 users logged in). The Jetson's desktop is off.
- The Jetson ran MAXN on micro-USB with the power guard running.
- Both Pis read `throttled=0x0` before, after each load and after the idle log.

**A. The outlet map, confirmed by load.**
- **Load:** one board at a time ran timeout-bounded busy loops.
- **Rise:** the mean `apower` over [load start + 10 s, end − 2 s], minus the baseline (the first 28 s of the log).
- **Result:** the registered rule passed for all three boards.

| Load (duration by the board's clock) | outlet 0 (pi4) | outlet 1 (pi5) | outlet 2 (jetson) | outlet 3 (router) | rule |
|---|---:|---:|---:|---:|---|
| baseline, W | 2.40 | 2.30 | 2.03 | 8.31 | |
| Pi 5, 4 cores, 60.006 s | +0.08 | **+5.09** | +0.07 | +0.02 | PASS |
| Pi 4, 4 cores, 60.010 s | **+3.79** | +0.07 | +0.10 | +0.00 | PASS |
| Jetson, 2 cores, 45.011 s | +0.13 | −0.02 | **+2.83** | +0.05 | PASS |

- **The busy-loop rises identify the outlets. They are not LLM-workload power**, which was registered as not reportable.
- **After the loads:**
  - the Pi 5 was at 57.6 °C and the Pi 4 at 67.6 °C, both `0x0`;
  - the Jetson guard saw ≥ 5.104 V and ≤ 3.511 W, with no alarm and no trip.

**B. The idle floors.** 300 s (14:33:47–14:38:47), nothing started on any board. The by_minute columns use the five complete
minutes 14:34–14:39.

| Outlet | mean `apower` (min–max), W | E trapezoid, J | E counter, J | gap | counter mean, W | by_minute vs counter | by_minute, W per minute |
|---|---:|---:|---:|---:|---:|---:|---|
| 0 pi4 | 2.43 (1.8–3.1) | 728.4 | 763.2 | −4.56 % | 2.54 | +0.39 % | 2.45 → 2.66 |
| 1 pi5 | 2.31 (2.1–3.9) | 693.4 | 734.4 | −5.58 % | 2.45 | +0.36 % | 2.42–2.53 |
| 2 jetson | 2.10 (1.6–3.2) | 630.2 | 655.2 | −3.82 % | 2.18 | −0.51 % | 2.16–2.18 |
| 3 router | 8.33 (8.1–8.7) | 2497.4 | 2642.4 | −5.49 % | 8.81 | −0.35 % | 8.67–8.81 |

What that says:

1. **The meter reads steady, non-zero values at 2 W.** On every outlet the readings changed 0.93–0.99 times a second
   (median interval 1 s): the 1 Hz refresh of 10-04. `FRESH_BELOW_1HZ` is raised on all four; that is the known 1 Hz limit,
   not a fault. `apower` shows 0.1 W steps.
2. **Integrated `apower` reads 3.8–5.6 % below the strip's own counter, on all four outlets.**
   - **The gap is real.** The counter's quantisation at the window edges is only 0.5–1.1 %.
   - **The counter and `by_minute` agree** within ±0.5 % over the five minutes. They may share one accumulator inside the
     strip, so their agreement does not make either of them right.
   - **Under-sampling cannot explain it on the router.** The counter's mean there, 8.81 W, is above **every one of its 1,495
     `apower` readings** (max 8.7 W). So the 10-04 reading (1 Hz snapshots under-sample a swinging load) does not hold
     there.
   - **What it looks like:** a systematic difference between the strip's power reading and its energy counter, roughly in
     proportion to the power (+0.08 to +0.48 W).
   - **Open:** which one is right needs a reference (QR6). Which to quote stays open, as registered.
3. **The Jetson, AC against DC.**
   - Outlet 2: 2.10 W mean `apower` (2.18 W by the counter).
   - The guard's INA3221 module input: 1.26 W (n 277, V ≥ 5.208).
   - AC − DC = 0.84 W, and DC / AC = 0.60. AC > DC, as expected: the adapter's loss plus the carrier board's loads (USB, fan,
     Wi-Fi card), which are outside the module rail.
4. **The Pi 4's floor drifted up inside the window**, by_minute 2.45 → 2.66 W over five minutes at load average ~0. The other
   three held steady (ranges above). Cause not established. Because the floor can drift, the floor subtracted from a run must
   come from the idle window just before that run, as the protocol does.
5. **Requests on the router's Wi-Fi:**
   - 0 of 1,495 failed in the window (0 of 3,438 in the whole run);
   - RTT median 0.084 s, p95 0.091 s, max 0.288 s;
   - none over 0.5 s, where the hotspot had stalls of 0.5–3 s.

## Energy per token on the Pi 5, the first measurement (2026-10-05)

- **Run:** `testbed/results/energy_phase_pi5_20261005_145817/` (14:58:17–15:22:49), by the new
  `testbed/scripts/energy_phase_run.py`.
- **Scoring:** `testbed/scripts/score_energy_phase_run.py`, against `testbed/dataset/energy_predictions_20261005b.json`
  (registered 14:58:10, sha256 `42112f2e…`, read-only). Results: `testbed/dataset/energy_results_20261005b.json`.
- **What ran:**
  - TinyLlama-1.1B-Chat Q4_K_M (sha256 `9fecc3b3…`, re-verified on the board in preflight), llama.cpp `b23701f7` (build
    200), 4 threads.
  - Three cycles of prefill-only `-p 512 -n 0 -r 20` and decode-only `-p 0 -n 128 -r 17`, both with `--progress`, between
    5-minute idle windows.
  - All six invocations exited 0. The strip answered 7,323 requests, 0 failed.
- **The board:** `throttled=0x0` in all 293 samples, 43.9–68.1 °C, the ARM clock at 2,400 MHz in every window, fan states
  0–3. Its desktop session was running; the Pi 4 and the Jetson sat idle.
- **Windows** come from the laptop's stamps of the progress lines. Each one matched llama-bench's own sum of repetition
  times to within 0.01 s.
- **Every figure here is AC at the wall**, the Pi 5's adapter included, as measured by the strip.

| Invocation | t/s | window, s | W (`apower`) | W (counter) | gap | net J/token, counter | net J/token, trapezoid | gross J/token, counter |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pp1 | 81.08 | 126.3 | 9.33 | 9.83 | −5.13 % | 0.0911 | 0.0866 | 0.1213 |
| pp2 | 81.08 | 126.3 | 9.42 | 9.95 | −5.27 % | 0.0925 | 0.0877 | 0.1227 |
| pp3 | 80.95 | 126.5 | 9.64 | 10.20 | −5.46 % | 0.0957 | 0.0906 | 0.1260 |
| tg1 | 17.66 | 123.3 | 8.25 | 8.70 | −5.17 % | 0.3543 | 0.3367 | 0.4930 |
| tg2 | 17.60 | 123.6 | 8.29 | 8.74 | −5.15 % | 0.3573 | 0.3395 | 0.4963 |
| tg3 | 17.49 | 124.5 | 8.25 | 8.74 | −5.63 % | 0.3596 | 0.3394 | 0.4996 |

The idle floor subtracted (the 300 s idle-before window) was 2.310 W by `apower` and 2.448 W by the counter.

**Per phase:** the mean of the three invocations. Spread = 100 × (max − min) / mean.

| | t/s (spread) | net J/token, counter | net J/token, trapezoid | net Wh per 1,000 tokens, counter (trapezoid) |
|---|---:|---:|---:|---:|
| prefill (pp512) | 81.04 (0.16 %) | **0.0931** (4.97 %) | **0.0883** (4.50 %) | 0.0259 (0.0245) |
| decode (tg128) | 17.58 (0.98 %) | **0.3571** (1.47 %) | **0.3385** (0.85 %) | 0.0992 (0.0940) |

What that says:

1. **Decode costs about 3.8× more energy per token than prefill** on this board and model: net 0.357 against 0.093 J by
   the counter, and 0.339 against 0.088 J by the trapezoid. Prefill draws more power (9.5 W against 8.3 W at the wall) but
   processes 4.6× more tokens a second.
2. **It repeats well.**
   - Throughput spread across invocations: 0.16 % for prefill, 0.98 % for decode.
   - Decode's net J/token spread: 0.85–1.47 %.
   - **Prefill's 4.5–5.0 % is not scatter but a steady rise with warming.** From pp1 to pp3 the board went from 58–66 to
     65–68 °C, and the fan stepped from 1–2 to 2–3. Prefill power rose from 9.33 to 9.64 W at a constant 81 t/s, with the ARM
     clock at 2,400 MHz (no throttling).
   - The first cycle started from a board cooled by the idle window. **A campaign that compares placements should pre-warm
     or interleave**, or this ~5 % drift lands in the comparison.
3. **The method gap holds under load: −5.1 to −5.6 % in all six windows.** It is the same as at idle on this outlet (−5.6 %)
   and on the router (−5.5 %).
   - So on outlet 1 the counter reads a near-constant ~5 % above integrated `apower`, from 2.3 W to 10 W.
   - The method choice (still open, QR6) therefore scales every figure on this outlet together. **Comparisons on one outlet
     are unaffected; absolute values are not.**
   - The gap differs between outlets (3.8–5.6 % at idle), so cross-board comparisons carry up to ~2 % from it.
4. **Registered checks: 17 PASS, 2 FAIL.**
   - Passed: throughput, mean power, the spreads, the gap and throttling.
   - **Failed: both counter-based net J/token means came out above their bands** (prefill 0.0931 > 0.089; decode 0.3571 >
     0.342). The trapezoid-based means are inside (0.0883 and 0.3385).
   - The bands were derived from `apower` power estimates, so the miss is the method gap again. Recorded as found.
5. **The harness.**
   - The result line arrives promptly through ssh, the protocol's open point: each window matched its timed repetitions.
   - **One defect:** the runner stopped the strip log at the idle-after window's end, so that window's energies were withheld
     (`NOT_COVERED`). Its mean `apower` was 2.387 W, 3 % above the idle-before floor, with the fan still at 0–1 and up to
     53 °C.
   - The runner now keeps logging 5 s past the last window (`--tail`). The net figures above use the idle-before window and
     are unaffected.

## Energy per token on the Pi 4, four quantisations (2026-10-05)

- **Runs:** `testbed/results/energy_phase_pi4_{q4km,q4_0,q8_0,q3km}_*` (17:52–19:30), back to back, by
  `energy_phase_run.py`.
- **Registrations:** `testbed/dataset/energy_predictions_20261005_pi4_*.json`, all four registered 17:52:51, read-only.
  Results: `energy_results_20261005_pi4_*.json`.
- **What ran:** TinyLlama-1.1B at 4 threads, the hash re-verified in each preflight. Three cycles each, with 300 s idle
  before and 120 s after. Every invocation exited 0, and no strip request failed.
- **Outlet 0** idled at 2.39–2.47 W (`apower`) across the four idle windows.

| Quant | prefill t/s | prefill W | prefill net J/token, counter (trapezoid) | decode t/s | decode W | decode net J/token, counter (trapezoid) | checks |
|---|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 12.49 | 7.46 | **0.425** (0.406) | 5.66 | 6.32 | **0.721** (0.694) | 18 / 19 |
| Q4_0 | 11.92 | 7.21 | **0.415** (0.400) | 5.81 | 6.36 | **0.696** (0.674) | 18 / 19 |
| Q8_0 | 12.30 | 7.41 | **0.415** (0.401) | 3.30 | 5.60 | **0.984** (0.948) | 15 / 19 |
| Q3_K_M | 10.15 | 7.16 | **0.480** (0.462) | 6.60 | 6.87 | **0.689** (0.666) | 18 / 19 |

The across-invocation spread of net J/token was 1.7–3.4 % everywhere. The trapezoid-vs-counter gap ran from −2.9 to
−4.7 %: smaller than outlet 1's ~5 %, as at idle (−4.6 %).

What that says:

1. **Prefill costs the same per token at Q4_K_M, Q4_0 and Q8_0 (0.415–0.425 J). Q3_K_M costs 16 % more**, because it
   prefills 19 % slower at the same power.
2. **Decode is where the quantisation matters.**
   - Q8_0 costs **0.98 J/token**, 41 % more than Q4_0, even though it draws **less** power (5.60 W against 6.32–6.87 W).
     Its 1.17 GB of weights make decode wait on memory: the cores stall, so power drops, but the tokens come 43 % slower.
   - Q3_K_M decodes cheapest (0.689 J), Q4_0 next (0.696).
3. **Registered checks.**
   - The **Q8_0 run failed three:** its decode power, and both decode J/token, fell *below* the bands. The bands assumed
     the same power for every quantisation, and the memory-bound decode shows that guess was wrong. Recorded as found.
   - **`get_throttled 0x0` failed in all four runs.** The Pi 4 reached **80.8–81.8 °C** in every prefill window, with
     its fan at state 4. The sticky soft-temperature flag `0x80000` was set at ~18:06:3x (Q4_K_M's pp2, ~81 °C), so the
     three later runs show it from their start.
   - Per window, the 5 s sampler read **the ARM clock at 1,800 MHz with no live throttle bit in any sample** of the 24
     windows. A cap shorter than 5 s cannot be ruled out: Q4_K_M's pp3 was 1.7 % slower than pp1–pp2. **The Pi 4's
     cooling is marginal for sustained prefill.**
4. **Against the Pi 5, Q4_K_M, the same protocol and the same day:**
   - the Pi 4 spends **4.6× more energy per prefill token** (0.425 against 0.093 J) and **2.0× more per decode token**
     (0.721 against 0.357);
   - it is 6.5× slower in prefill and 3.1× slower in decode, at 79 % and 77 % of the Pi 5's wall power.
   - So for this model, the faster board is also the frugal one, and more so in prefill.

## Energy per token on the Jetson Nano (2026-10-05)

- **Runs:** `testbed/results/energy_phase_jetson_*` (19:35–21:40), registered 19:35:46 (`energy_predictions_20261005_jetson_*.json`,
  read-only). Results: `energy_results_20261005_jetson_*.json`.
- **Conditions:** MAXN on micro-USB with the power guard required by the runner. The clocks were locked for each whole
  session (`jetson_clocks`, stored and restored: RESTORED after every run). The guard stayed up with **0 trips** in every
  run. The input dipped to 4.832 V at ~9 W DC, 82 mV above the trip.

**The GPU (Vulkan, `-ngl 99`):**

| Quant | prefill t/s | prefill W | prefill net J/token, counter (trapezoid) | decode t/s | decode W | decode net J/token, counter (trapezoid) | idle W (locked) | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 72.53 | 8.89 | **0.083** (0.079), spread 6.3 % | 11.37 | 11.17 | **0.736** (0.707) | 3.14 | 14 / 19 |
| Q4_0 | 79.20 | 8.67 | **0.072** (0.070), spread 5.8 % | 8.20 | 9.08 | **0.752** (0.722) | 3.15 | 15 / 19 |
| Q8_0 | 78.51 | 9.01 | **0.078** (0.075), spread 5.1 % | 6.76 | 10.08 | **1.064** (1.025) | 3.16 | 14 / 19 |
| Q3_K_M | 67.97 | 8.81 | **0.086** (0.083), spread 5.3 % | 7.87 | 9.88 | **0.885** (0.853) | 3.16 | 14 / 19 |

What that says:

1. **The Jetson's GPU prefill is the cheapest in the cluster:** 0.072–0.086 J/token, against the Pi 5's 0.093.
2. **Its GPU decode draws *more* power than its prefill** (9.1–11.2 W against 8.7–9.0 W) and costs 0.74–1.06 J/token: about
   2× the Pi 5's decode (0.357), and level with the Pi 4's (0.69–0.98). Q8_0 is the most expensive, as on the Pi 4.
3. **So, for energy, prefill and decode favour different boards:** prefill the Jetson GPU, decode the Pi 5. This is the
   energy side of the prefill/decode split the study asks about. It is single-board evidence; no pooled cell has measured it.
4. **Registered checks: 14, 15, 14 and 14 of 19.** Throughput, the method gap, the guard and prefill power all passed. The
   misses are as found:
   - prefill's across-invocation spread was 5.1–6.3 %, just past the registered 5 %;
   - decode power and decode J/token came out above bands that were a guess (4–9 W); GPU decode power had never been
     measured at the wall.

**The "CPU" run was misconfigured, and is not a CPU measurement.** It was registered as the Jetson's CPU at Q4_K_M
(`-ngl 0`), but the Jetson's CPU rows are taken with **`-ngl 0 -dev none`** (`jetson_bench_tinyllama.sh`). Without
`-dev none` the Vulkan device stays active, and llama.cpp at this pin sends the large prefill operations to the GPU while the
weights stay in CPU memory.
- **What it measured:** prefill 62.6 t/s at 8.84 W (net 0.091 J/token), and decode **1.54 t/s** at 5.68 W (net **1.71 J/token**).
- **What it is:** a different configuration, "weights in CPU memory with GPU offload of large operations", recorded as
  that. It scored 7 of 19 checks: throughput and both J/token bands failed, as they should.
- **Redone** with `-ngl 0 -dev none` at 22:37 (below).

**The CPU, redone (`-ngl 0 -dev none`, 22:37–23:04).**
- **Registered 22:29:08** (`energy_predictions_20261005_jetson_cpu_q4km_redo.json`, `c4248420…`, read-only;
  `.handoff/assets/energy-2026-10-05/register_jetson_cpu_redo.py`). The first CPU registration's predictions, basis and
  scoring were copied verbatim: they were set before any true CPU energy figure existed. Only the flags, the conditions, the
  timestamp and the input hashes changed. Results: `energy_results_20261005_jetson_cpu_q4km_redo.json`.
- **Conditions:** as above (MAXN, micro-USB, the clocks locked for the session and restored, the guard required). The board had
  been power-cycled at 22:26 after the pooled session's hang (below); the guard was restarted, stayed up and recorded 0 trips.
  That note was meant for the registration's conditions, but the edit failed an assertion and the file was written without it.

| Quant | prefill t/s | prefill W | prefill net J/token, counter (trapezoid) | decode t/s | decode W | decode net J/token, counter (trapezoid) | idle W (locked) | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M, CPU | 8.72 | 8.28 | **0.620** (0.595), spread 3.8 % | 6.41 | 8.81 | **0.932** (0.891) | 3.10 | 16 / 19 |

- **Throughput matches the CPU rows** (8.59 / 6.40 t/s), so `-dev none` kept the GPU out.
- **The three misses are all decode:** its power (8.81 W, above the guessed 4.5–8.5 W) and its net J/token by both methods,
  just above their bands (0.932 against ≤ 0.910 by the counter; 0.891 against ≤ 0.843 by the trapezoid). As found.
- **On its CPU the Jetson spends 7.5× the GPU's energy per prefill token** (0.620 against 0.083) and **1.27× per decode
  token** (0.932 against 0.736). Against the Pi 4's CPU at Q4_K_M, it costs 1.46× per prefill token and 1.29× per decode
  token. Of the engines measured, the Jetson's CPU is the costliest per token in both phases.

## Pooled energy on the Pi 4 + Jetson (2026-10-05 21:57–22:26, and 2026-10-06 02:46–03:38)

**Registered before it ran** (`testbed/dataset/pool_energy_predictions_20261005_215708.json`, 21:57:08, `0fa42d05…`, read-only;
`register_pool_energy.py`). Run by `energy_pool_run.py`, which puts `run_pool.sh` cells inside one strip log, and scored by
`score_energy_pool_run.py` (`testbed/results/energy_pool_20261005_215722/`, `score.json`).
- **The cells:** TinyLlama Q4_K_M, the engine's `rule` (Pi 4 1 + Jetson 21, hosted on the Pi 4) against `even` (11 + 11),
  alternated three times, then `qwen3_14b.rule` (Pi 4 31 + Jetson 9) twice.
- **The protocol:** a 300 s idle baseline, `-p 512 -n 128` (TinyLlama) with `-r` sized for ≥ 120 s per phase, and windows
  cut from llama-bench's `--progress` lines on the laptop's clock.
- **The hypothesis:** the rule's pool net J/token is at least 25 % below the even split's, in each phase, by each method.
- **A one-cell shakedown came first** (scratchpad, not reported). It found that the laptop's OpenSSH 10.2 prints a
  post-quantum key-exchange warning on stderr for the Jetson's old sshd, which `run_pool.sh`'s busy check (`2>&1`) read as
  "busy". The busy check and the Jetson gate now read stdout only, in `run_pool.sh` and `serve_pool.sh`; an unreachable board
  still fails by ssh's exit code. The re-run shakedown passed: exit 0, both phases windowed, throughput within 2 %.

**What happened.**
- Cells 1–3 ran (rule, even, rule).
- **The Jetson hung at the end of cell 3.** Its storage stopped committing writes at ~22:16:08: the guard log's last line is
  22:16:08, and its tail holds 1,110 NUL bytes. The board kept computing for the RPC worker; cell 3 finished, exit 0. It stopped
  answering at ~22:20:50: ssh "Connection reset", ping fine, outlet 2 stuck at ~5.7 W.
- Cells 4–8 were refused at `run_pool.sh`'s gate ("unreachable", ssh exit 255) and are not figures.
- The board was power-cycled through the strip (22:26:27, outlet 2, 10 s `toggle_after`, the user's yes) and recovered (clock
  synced, guard restarted).
- **This is the 14:43 hang's signature,** and that one came at idle on a healthy supply. Both point at the storage, not at load
  or power. The card is an "SD128" 116 GiB (manufacturer 0x9f, OEM "TI", made 06/2019) in SDR104 mode. This boot's kernel log
  shows no card errors. Not resolved.
- **A third hang came at ~10:14 on 2026-10-06,** at idle, 6.5 h after the second session ended. `power-guard.log`'s last line is
  10:14:41, followed by 1,863 NUL bytes. The board was hard power-cycled at ~10:46; `wtmp` has no clean shutdown.
  - The Jetson's local files and builds were then backed up to the laptop (`testbed/backups/jetson_20261006/`).
  - The guard was not restarted, to keep writes off the card. It must be restarted before any load.

| cell | runs | prefill t/s (pred.) | prefill net J/token | decode t/s (pred.) | decode net J/token | Jetson mean W, decode |
|---|---|---|---|---|---|---|
| rule (Pi 4 1 + Jetson 21) | 2 | 57.69 (59.04) | 0.1193, 0.1450 → **0.132** | 10.20 (10.485) | 0.8915, 1.1512 → **1.021** | 10.17, 12.81 |
| even (11 + 11) | 1 | 20.68 (21.34) | **0.316** | 7.50 (7.61) | **0.919** | 6.05 |

(Counter method; the trapezoid is 3–5 % lower throughout. Gross J/token: rule 0.200 / 0.226 prefill, 1.350 / 1.610 decode;
even 0.542 / 1.543.)

**The registered verdicts of this first session, applied as written:**
- **Throughput holds in all four checks:** every phase within 10 % of the engine's prediction (−1.5 % to −3.1 %).
- **The energy hypothesis splits.**
  - Prefill **holds**: the rule's net J/token is 58 % below the even split's, by both methods.
  - Decode **fails**: the rule's is 11 % *above* the even split's.
- 3 of 8 runs exited 0. The five failures are the hang, as above; they were run as a second session (below).

**What it shows, and how far it can be trusted.**
- **The speed-optimal placement is not the energy-optimal one in decode.** The rule puts 21 of 22 layers on the Jetson's GPU. That
  is 1.36× faster in decode, but the GPU decodes at 10–13 W, so a decode token costs about the same as in the even split, or
  more. In prefill the GPU is both faster and far cheaper per token. This matches the single-board finding above: prefill
  favours the Jetson GPU; decode does not.
- **The decode figures are not steady.** The two rule runs differ by 25 % (net 0.89 and 1.15 J/token), because the Jetson drew
  12.8 W in run 3's decode against 10.2 W in run 1's, for the same cell. Run 3 ended in the hang. The registered single-board
  bar is a 5 % spread.
- **The idle floor is not the locked one** (a method gap, reported as found). `run_pool.sh` locks the Jetson's clocks per cell,
  so during the session's idle window the Jetson idled at 2.13 W, under its governors. The single-board protocol locks them for
  the whole session, giving a 3.1 W floor. The registered net therefore charges each Jetson window ~1 W of lock overhead: +0.017
  (rule) and +0.048 (even) J per prefill token, +0.096 and +0.131 per decode token.
  - Recomputed with the locked floor (exploratory, not the registered method), prefill keeps its saving (57–62 %), and in decode
    the rule is at best level with the even split (run 1 alone −0.9 %).
  - So the conclusion stands either way. A future pooled registration should lock the Jetson's clocks for the whole session,
    as the single-board runs do.
- **Cells 4–8 were measured in a second session,** below.

**The second session: cells 4–8 (2026-10-06, 02:46–03:38).**
- **Setup:** the same registration (`0fa42d05…`) and the same runner arguments (idle 300 s before, 120 s after; settle 30 s;
  gap 20 s). Output in `testbed/results/energy_pool_20261006_024600/` (`score.json`). The cells were even, rule, even, then
  the 14B rule twice.
- **The Jetson stayed up throughout:** it had been up since 22:37, about 5 h by the end. Its clocks were unlocked for the
  idle window (2.2 W), as in the first session.
- **All 5 runs exited 0, and 13 of 15 checks passed.** All six throughput checks held (−3.6 % to +4.5 % against the engine).
  The two misses are the decode hypothesis.

| cell (session 2) | runs | prefill t/s (pred.) | prefill net J/token | decode t/s (pred.) | decode net J/token | Jetson mean W, decode |
|---|---|---|---|---|---|---|
| even (11 + 11) | 2 | 20.58 (21.34) | 0.3122, 0.3201 → **0.316** | 7.50 (7.61) | 0.9086, 0.9240 → **0.916** | 5.99, 6.07 |
| rule (Pi 4 1 + Jetson 21) | 1 | 57.80 (59.04) | **0.121** | 10.45 (10.485) | **0.908** | 10.46 |
| qwen3_14b.rule (Pi 4 31 + Jetson 9) | 2 | 1.100 (1.13) | 6.058, 6.072 → **6.065** | 0.510 (0.488) | 11.98, 12.04 → **12.01** | 4.62, 4.13 |

(Counter method; the trapezoid is 3–9 % lower. Gross J/token, counter: even 0.544 / 1.540, rule 0.202 / 1.356, 14B
10.32 / 21.18.)

**The registered verdicts, per session and over both** (`combine_pool_sessions.py` pools the scored runs and recomputes no
energy; `testbed/results/energy_pool_combined_20261006.json`):

| rule's net J/token against even's | session 1 (rule ×2, even ×1) | session 2 (rule ×1, even ×2) | both (3 runs each) |
|---|---|---|---|
| prefill, counter (trapezoid) | −58.2 % (−58.6 %) | −61.7 % (−61.7 %) | **−59.4 % (−59.7 %): holds** |
| decode, counter (trapezoid) | +11.1 % (+11.3 %) | −0.9 % (−0.6 %) | **+7.2 % (+7.5 %): fails** |

- **The hypothesis splits the same way in each session and over both:** prefill holds, by more than twice its 25 % margin,
  and decode fails.
- **Pooling two sessions departs from the registered plan,** which was one session: there are two idle floors instead of one.
  Both floors were the Jetson's governor idle.
- **The rule's decode is not steady, and the even split's is.**
  - The rule's three runs read 0.892, 1.151 and 0.908 J per decode token, a 26 % spread. The even split's read 0.919, 0.909
    and 0.924, a 1.7 % spread.
  - The high rule run is session 1's run 3. That is the run during which the Jetson's storage stopped writing (22:16), and the
    Jetson drew 12.8 W.
  - Without it (exploratory), the rule's decode is 0.900 against 0.917: level (−1.9 %).
  - Either way the rule saves nothing in decode, which matches the single-board finding: the Jetson's GPU decodes fast but not
    frugally.
- **Qwen3-14B on the pool** (rule only). An even 20 + 20 split cannot run: 20 blocks need about 5.3 GB of weights on the
  Jetson (9 blocks hold 2,366 MiB), against its 2,845 MiB cap and 3,545 MiB available.
  - prefill 1.10 tokens/s at 6.06 J per token net (10.3 gross); decode 0.51 tokens/s at 12.0 J per token net (21.2 gross),
    that is 3.3 Wh per 1,000 decoded tokens;
  - the two runs agree within 0.2–0.4 %;
  - the Jetson drew 4.1–4.6 W in decode (9 of 40 blocks) and the Pi 4 5.8–5.9 W.
- **`NET_NEGATIVE` flags** appear only on outlet 1 (the laptop: the controller, not in the pool) and outlet 3 (the router,
  reported apart), whose draw sat at its idle. The pool's outlets 0 and 2 have none.

**Exploratory, post-hoc (not registered): do single-board readings predict the pooled energy?**
(`.handoff/assets/energy-2026-10-05/energy_from_single_board.py`; output in
`testbed/results/energy_pool_from_single_board_20261006.json`.)

The speed side predicts a pooled cell from single-board rows; this asks the same of energy, with nothing fitted.
- **Time:** the engine's own time per token on each board for the cell. This is `predict_pool.py`'s cost model, mirrored per
  board and checked against its total.
- **Power:** each board's mean wall power while computing that phase alone, from its single-board TinyLlama Q4_K_M run.
- **Waiting:** a board waits while the other computes. The Jetson, whose clocks `run_pool.sh` locks per cell, waits at its
  locked idle; the Pi 4 is assumed to wait at its idle.
- **Net per board:** (compute W − floor) × its time, plus (waiting W − floor) × the rest of the token, against the session's
  own floor.

| cell | prefill net J/token, predicted / measured (counter) | decode net J/token, predicted / measured (counter) |
|---|---|---|
| TinyLlama rule, 3 runs | 0.117 / 0.129 (−9.1 %) | 0.846 / 0.984 (−14.0 %) |
| TinyLlama rule, without session 1's run 3 | 0.117 / 0.120 (−2.8 %) | 0.846 / 0.900 (−6.0 %) |
| TinyLlama even, 3 runs | 0.303 / 0.316 (−4.3 %) | 0.871 / 0.917 (−5.0 %) |
| Qwen3-14B rule, 2 runs (TinyLlama's board powers) | 5.66 / 6.06 (−6.7 %) | 12.01 / 12.01 (−0.0 %) |

(The trapezoid gives the same picture: −3.6 % to −7.3 %, and +2.8 % for the 14B decode, with the same run left out.)

- **Within 7 % everywhere, always low,** once the run during which the Jetson's storage failed is left out.
- **The Jetson's share is predicted within ±3 %.** For example, 0.814 against 0.829 J per decode token in the rule cell.
- **The shortfall is the host's own overhead.** The Pi 4, hosting, draws about 0.3–0.4 W more than "its blocks, then idle"
  while it waits: in the rule cell 0.071 J per decode token against a predicted 0.032. Its threads, sampling and the RPC
  traffic are not idle.
- **A board's compute power carries across models.** TinyLlama's power levels, with the engine's 14B times, predicted the
  14B pool's decode energy within 0–3 %.
- **What it suggests:** an energy-aware engine could price energy from single-board power and its existing time model. One
  added term, each board's power while waiting in a pool, would need measuring.
- **The limits:** post-hoc; 3 runs per TinyLlama cell and 2 for the 14B; one pool. The waiting assumptions were chosen with
  the measurements in hand, though not tuned to them.

## Energy per token on the laptop (ThinkPad T580), four quantisations (2026-10-06, 00:59–02:22)

- **Runs:** `testbed/results/energy_phase_laptop_{q4km,q4_0,q8_0,q3km}_20261006_*`, back to back in one chain
  (`.handoff/assets/energy-2026-10-05/run_laptop_chain.sh`; log `testbed/results/energy_laptop_chain_20261006_005639.log`),
  run on the laptop itself by `energy_phase_run.py --local`. Each run started once the CPU package read below 55 °C; it read
  50–53 °C each time, so none waited.
- **Registrations:** `testbed/dataset/energy_predictions_20261006_laptop_*.json`, all four registered 00:56:39, read-only
  (`.handoff/assets/energy-2026-10-05/register_laptop.py`). Results: `energy_results_20261006_laptop_*.json`.
- **A shakedown came first** (23:05 on 10-05, one cycle, under a scratch registration). It ran at 48 / 20 t/s with a 31.1 W
  idle, with Discord, LibreOffice, a browser and VS Code open.
  - The bands were written before it ran and were not changed after it. The apps were closed instead.
  - The four registrations' predictions, commands and basis are identical to that scratch registration's, checked field by
    field. Both are kept in `.handoff/assets/energy-2026-10-05/laptop_shakedown/`.
- **What ran:** TinyLlama-1.1B at 4 threads, the hash re-verified in each preflight. Three cycles each, with 300 s idle before
  and 120 s after. Every invocation exited 0, and the battery read "Not charging" in all 967 sampler lines.
- **Windows were 105–138 s,** shorter than the ~120 s that `-r` was sized for at the characterisation rows' rate, because the
  laptop ran faster than its rows (point 5 below). All are above the method's 60 s floor.
- **Conditions:**
  - AC, power profile `performance`, turbo on (as for the T580's characterisation rows);
  - the internal panel off and the external monitor on, never blanking, so the display is the same in idle and load
    windows;
  - running: GNOME, the Claude desktop app (this session, which made no tool calls during the runs) and LINE (under Wine).
- **Outlet 1 idled at 14.1–15.9 W** (`apower`) across the four idle-before windows. That compares with 19.7–24.5 W at the
  17:49 map check and 31.1 W at the shakedown, both with apps open.

| Quant | prefill t/s | prefill W | prefill net J/token, counter (trapezoid) | decode t/s | decode W | decode net J/token, counter (trapezoid) | idle W | checks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4_K_M | 85.46 | 29.34 | **0.182** (0.179) | 30.66 | 32.12 | **0.602** (0.590) | 14.06 | 18 / 19 |
| Q4_0 | 80.85 | 29.23 | **0.195** (0.181) | 32.54 | 32.53 | **0.580** (0.552) | 14.62 | 17 / 19 |
| Q8_0 | 67.13 | 29.60 | **0.235** (0.204) | 19.45 | 32.32 | **0.985** (0.845) | 15.90 | 16 / 19 |
| Q3_K_M | 54.16 | 28.80 | **0.291** (0.252) | 31.13 | 31.83 | **0.598** (0.536) | 15.17 | 16 / 19 |

(Gross J/token, counter: Q4_K_M 0.366 prefill / 1.114 decode; Q4_0 0.386 / 1.055; Q8_0 0.461 / 1.762; Q3_K_M 0.568 / 1.080.
The idle is about half the loaded draw, so gross is 1.8–2× net.)

What that says:

1. **Prefill is cheapest at Q4_K_M (0.182 J/token).** Against it, Q4_0 costs 7 % more, Q8_0 29 % more and Q3_K_M 60 % more.
   Q3_K_M prefills 37 % slower at the same power, as on the Pi 4, where it was also the costliest prefill.
2. **Decode costs the same at Q4_0, Q3_K_M and Q4_K_M (0.58–0.60 J/token); Q8_0 costs 64 % more (0.985).**
   - Q8_0 decodes 37 % slower.
   - Unlike the Pi 4's Q8_0, it does so at the **same** power (32.3 W), so the slowdown goes straight into energy per token.
3. **Decode draws more power than prefill** (31.8–32.5 W against 28.8–29.6 W), as on the Jetson's GPU.
4. **The first prefill of each run is the fastest and the costliest.**
   - In every run, prefill t/s and net J/token both fall from pp1 to pp3. For Q8_0, 70.4 t/s at 0.222 J becomes 65.4 t/s
     at 0.192 J (trapezoid).
   - The package reached 78–81 °C in every run. A cool laptop turbos harder: faster, but less efficient per token.
5. **Registered checks: 18, 17, 16 and 16 of 19 (67 of 76).** Power, all eight J/token bands, the method gap (−3.3 to
   −6.9 %) and the battery passed. The misses, as found:
   - **Throughput ran above its band in 7 of 8 phases**, by 6–15 % over the characterisation rows. The band (−20 % to +5 %)
     assumed that sustained runs would be at or below the short rows. Only Q4_K_M's prefill was inside (−5.2 %).
   - **Prefill spread (trapezoid) was 14.5 % for Q8_0 and 13.1 % for Q3_K_M**, against a registered 12 %. This is the pp1 →
     pp3 trend in point 4, not noise.
6. **At idle the two methods disagree** (`IDLE_METHODS_DISAGREE` on every window).
   - The counter read −5 % to +12 % against the trapezoid in the idle-before windows.
   - Outlet 1 refreshed `apower` below 1 Hz there (`IDLE_FRESH_BELOW_1HZ`). At a steady ~15 W the strip reports few fresh
     readings, so the trapezoid integrates stale values. The load windows refresh normally.
   - The disagreement passes straight into the net figures: the two methods' net J/token differ by 2–17 % (Q8_0 decode:
     0.985 against 0.845). Both are reported. Which is closer needs the QR6 reference.
7. **Disturbances** (`.handoff/assets/energy-2026-10-05/laptop_disturbances.py`; output in
   `testbed/results/energy_laptop_chain_20261006_005639_disturbances.txt`):
   - **The user's two messages and the session's replies** fell at 01:14:54–01:15:25 (Q4_K_M's pp3) and 01:30:00–01:30:34
     (Q4_0's tg1). Outlet 1 read no higher during them than in the rest of each window (28.0 against 28.5 W; 32.5 against
     33.4 W).
   - **A user-level Google Drive sync** (`aunsync.timer`, every 16 min, ~20 s per pass) ran five times. It was not in the
     system timer list checked before registration.
     - The passes fell in Q4_K_M's pp2, at the end of Q4_0's idle window, inside Q8_0's idle window, in Q8_0's tg3 and in
       Q3_K_M's tg2.
     - No pass shows a power step (Q8_0's idle: 16.00 W during the pass, 15.90 W around it).
   - **The system timers** were momentary: sysstat every 10 min, fwupd-refresh 01:25:31, man-db 01:29:40 (1 s),
     NetworkManager-dispatcher 02:07:56 and 02:10:56.
   - Two readings near 40 W at 02:07:56 and 02:10:00 coincide with those events, but they are the turbo burst that opens
     every invocation.
   - **Recomputing each idle-before mean without the event spans moves net J/token by at most 0.6 %** (Q4_0). The registered
     figures stand as scored.
8. **Against the other boards at Q4_K_M (net, counter):**
   - **Prefill:** the laptop's 0.182 J/token is 2.0× the Pi 5's (0.093) and 2.2× the Jetson GPU's (0.083), at a similar
     speed (85 against 81 and 73 t/s). It is 0.43× the Pi 4's (0.425).
   - **Decode:** the laptop is the fastest decoder (30.7 t/s against the Pi 5's 17.6) and the second cheapest (0.602 J/token).
     It is 1.69× the Pi 5's 0.357, and below the Pi 4's 0.721 and the Jetson GPU's 0.736.
   - **Gross, its 14–16 W idle counts against it:** 0.366 / 1.114 J per prefill / decode token, against the Pi 5's
     0.123 / 0.496.
9. **Outlet 1 holds a hub as well as the laptop.**
   - The laptop's only power input is a USB-C port whose partner also carries the DisplayPort link to the external monitor
     (USB PD, 20 V / 2.75 A offered; sysfs, 00:50).
   - The user confirmed after the runs that the laptop's charger plugs into a USB-C hub, which feeds the laptop over that one
     cable. The USB tree shows the keyboard and the USB microphone on a Genesys Logic hub, most likely this one.
   - So outlet 1's figure is the laptop plus the hub and its USB devices; their draw has not been measured apart. A constant
     draw cancels in the net figures, but not in W, idle W or gross J/token. The monitor itself is not on the strip.

## The method (the run protocol)

1. **Two energies per outlet, cross-checked.** Every window is integrated twice: the counter delta (`aenergy_total` × 3600,
   linearly interpolated at the window edges) and the trapezoid of `apower` (sample time = request start + RTT/2, laptop
   clock). Both are reported, never one quietly. A disagreement over 5 % is flagged (`METHODS_DISAGREE`). Which one is closer
   to the truth needs the QR6 reference.
2. **Idle-subtracted.** Each outlet's idle floor comes from a window of ≥ 5 min with the board booted and idle. Take one
   just before the run, and ideally one after. Net energy = gross − idle mean power × window length, by each method
   (`summarise --idle`).
3. **Phases only from long single-phase runs.** The readings refresh at 1 Hz, so a prefill window of a second or two cannot
   be separated from decode by sampling.
   - Run a **prefill-only** `llama-bench` invocation (`-p N -n 0`) and a **decode-only** one (`-p 0 -n N`).
   - Size each with `-p`/`-n` and `-r` so that its timed repetitions last **≥ 60 s**. Use **`-r` ≥ 3** (never `-r 1`), and
     quote the spread across runs.
   - **Markers on the laptop clock:** run with `--progress`. At the pin it prints `prompt run i/N` / `generation run i/N` to
     stderr as each timed repetition starts, and `warmup …` before the warm-up (checked in the source at `b23701f7`).
   - The harness stamps each line with the laptop's `time.time()` as it arrives. The window runs from the first timed
     repetition's line to the end of the last repetition (its result line or the process exit; whether the result line
     arrives promptly through ssh is to be checked on the first run). Model load and warm-up stay outside the window.
   - **J/token = net energy over the window / tokens processed in the timed repetitions** (N × r). Wh per 1,000 tokens follows.
   - A window spanning mixed phases gives the run's energy, which cannot be attributed to either phase. The summariser says
     so on every output.
4. **Windows long enough for the counter.** One counter step of ~1 s of energy at each end is about 2/T of the window's
   energy (arithmetic). That is ~3 % at 60 s and ~0.7 % at 300 s. `COUNTER_COARSE` fires above 5 %, with the observed step
   printed.
5. **With every run:** the logger's CSV and its `.meta.json` (firmware, MAC, outlet names, start and end); `get_throttled`
   per Pi (house rule); the raw benchmark output, tee'd (house rule); the window markers. The summariser flags a firmware
   change, a strip restart, a counter reset, an outlet that was off, request errors, gaps > 3 s, too few samples, a window
   the log does not cover, a coarse counter, fresh readings below 1 Hz, and a laptop clock step. It prints them and never hides
   them.
6. Poll at **5 Hz** (the cap). It caught the most 1 Hz refreshes for the same cost of being one request at a time.

```bash
python3 testbed/scripts/shelly_strip.py info                       # identity, firmware, names, initial_state, Wi-Fi, SNTP
python3 testbed/scripts/shelly_strip.py status                     # one reading per outlet
python3 testbed/scripts/shelly_strip.py log --out testbed/results/energy_<what>_<stamp>.csv --hz 5 --duration 900
python3 testbed/scripts/shelly_strip.py summarise <csv> --window START END --idle START END --tokens N [--json]
python3 -m unittest discover -s testbed/tests -q                   # 35 tests
```

## The requirements, as written and as found

Quoted from the proposal (EN), not re-worded:

- **QR5**: "Energy per 1,000 tokens: MVP baseline measured for every placement / Stretch find a placement reducing energy by
  at least 25%". **Measurable with this instrument**, per board at the wall, by the long single-phase protocol above.
  **First measured 2026-10-05** for one board and one model ("Energy per token on the Pi 5"). The MVP asks for every
  placement.
- **QR6**: "Power-measurement error against a known resistive-load reference: MVP within 5% / Stretch within 2%". **No
  reference instrument yet.** The proposal's calibration (known resistive loads and a multimeter, A.2.4) was designed for the
  DC INA226 chain. A wall meter needs an AC reference: a known resistive AC load, or a reference wattmeter. On the first
  load seen, the strip's two own methods already disagree by up to 8.6 %. At idle on 2026-10-05 they disagree by 3.8–5.6 % on
  all four outlets, in the same direction (integrated `apower` below the counter).
- **QR7**: "Power sampling rate: MVP at least 10 Hz / Stretch at least 100 Hz". **NOT met by this instrument:** 1 Hz per
  outlet, measured, a tenth of the MVP. The proposal's own reason (A.2.4) is that "≥ 10 Hz suffices for second-scale decode
  windows; the ≥ 100 Hz stretch target is needed to capture prefill windows".

**Take to the advisor, with QR1/QR2/QR3.** The instrument changed, from DC rails at ≥ 10 Hz to AC wall power at 1 Hz. That
changes QR6's reference and leaves QR7 unmet, and phase energy now rests on the long single-phase protocol rather than on
sampling. The requirements stay as written; this is the tension, recorded.

## Wiring plan

| Outlet | Device | Note |
|---|---|---|
| 0 | `pi4`: Raspberry Pi 4 and its USB-C adapter | Swapped with outlet 1 by the user on 2026-10-05 (it was `pi5` on 10-04) |
| 1 | `laptop`: the ThinkPad T580 (control and prefill node) and its charger, since ~17:45 on 2026-10-05 | Until 2026-10-05 it was `pi5`; the Pi 5 was returned that day. Renamed on the strip and in `strip.json`, and proven by load (+5.45 W on outlet 1, ≤ 0.03 W elsewhere). Valid only while the battery reads full. **The charger feeds a USB-C hub** (user, 2026-10-06), whose one cable to the laptop carries its power, the external monitor's DisplayPort link and USB: this outlet's figure is the laptop plus the hub and its USB devices |
| 2 | `jetson`: Jetson Nano and its adapter | The barrel supply with J48 once fitted (it is still on micro-USB with the power guard). DC cross-check: `POM_5V_IN` |
| 3 | `router`: True GIGATEX T628L | **Never switch off remotely**: it cuts the network, and the strip, then on the router's Wi-Fi, becomes unreachable |
| — | the external monitor | **not metered** (the laptop was here until ~17:45 on 2026-10-05; it is now outlet 1) |

Only the board's own adapter goes on its outlet (no monitor, hub or charger sharing it), or the outlet's figure stops being
the board's. **Outlet 1 does not meet this** (reported as found): the laptop's charger feeds a hub. A constant draw cancels in
net J/token, but W, idle W and gross J/token include the hub and its USB devices, which have not been measured apart.

## Setup checklist for the user

Things only the user can do or approve. Nothing here is done by the tool, which is read-only.

- [x] **(a) The strip on the router's Wi-Fi.** **Done 2026-10-05 ~14:12 by the user**, as **Wi-Fi 1 = "WIFI-B"** (the original
  plan was Wi-Fi 2). Read back at 14:13: `sta` "WIFI-B", got ip `192.168.100.34`, RSSI −8/−9 dBm (the router sits on the strip).
  The user also set **Wi-Fi 2 = "WIFI-C"**, but `sta1` read **empty and disabled** at 14:13: that save did not take.
  - **2026-10-05:** the T628L's Wi-Fi is **"WIFI-B"** (BSSIDs `02:00:00:00:00:01`, 2.4 GHz ch 11, and `02:00:00:00:00:06`, 5 GHz ch 36). The
    strip's own page has the form: `http://<strip>/#/settings/wifi` → "Wi-Fi 2 settings" → Network, Password, Enable, Save.
  - **"WIFI-C" cannot stay as Wi-Fi 1 at home.** The hotspot is the laptop's internet (the T628L has none), so it is on
    whenever the laptop is in use, and Wi-Fi 2 is only the fallback when Wi-Fi 1 is unreachable (expected behaviour, not
    tested here). The user wanted the router (2026-10-04) and chose Wi-Fi 1 = "WIFI-B", Wi-Fi 2 = "WIFI-C" (2026-10-05).
  - **If the strip ever falls back to "WIFI-C", the laptop cannot see it** (since 10-05 the laptop's Wi-Fi is on another network,
    "Chonrakit_5G"). The logger then records failed rows, not wrong numbers; whether the strip returns to Wi-Fi 1 by itself
    is untested.
- [ ] **(b) A DHCP reservation on the T628L** for MAC `e8:f6:0a:62:b7:c0`, **inside the pool `.33`–`.120`** (a reservation
  outside the pool is silently ignored on this router, `README.md`), e.g. `192.168.100.112`. Then update `host` in
  `testbed/energy/strip.json` and check with `info`.
- [x] **(c) Power-on default ON.** **Done 2026-10-04 ~20:10**: `initial_state` "on" on all four, applied by the main session
  with the user's go-ahead. **Caveat:** after the 20:12–20:14 update's reboot the outputs did not follow it (0–2 stayed off,
  3 stayed on). So it evidently applies at a mains power-up. **Verified at a real power-up 2026-10-05:** uptime 773 s at
  13:59, all four outlets on, `source` "init".
- [x] **(d) Outlet names** pi5 / pi4 / jetson / router. **Done 2026-10-04 ~20:10**, same as (c). The tool flags any name
  that differs from `strip.json`. **Renamed 2026-10-05 14:03** at the user's request, to match how the boards were plugged
  in at home: **0 `pi4`, 1 `pi5`**, 2 `jetson`, 3 `router` (`Switch.SetConfig`, the name only, read back). `strip.json`
  matches.
- [ ] **(e) Time sync:** point the strip's SNTP at the laptop's chrony, `192.168.100.105`, since the router has no internet.
  chrony already serves `192.168.100.0/24` (ufw udp/123), which is where the strip will be. The logger timestamps on the
  laptop anyway; this keeps the strip's own clock sane.
  - **Set 2026-10-05 14:17:43** at the user's go-ahead (`Sys.SetConfig` `sntp.server`, read back), but **pending a strip
    restart**: the strip answered `restart_required` true. Its last sync is still 13:54:53, over the hotspot, and its clock
    equalled the laptop's at 14:17.
  - **Not restarted**, because a restart might briefly drop all four outlets (the boards and the router). Do it with the
    boards shut down, or let the next power-up apply it.
- [x] **(f) Firmware.** **Done 2026-10-04 20:12–20:14**: stable 2.0.1 (`20260923-075623/2.0.1-ge1a198b`), from 2.0.1-beta3,
  applied by the main session with the user's go-ahead. **Keep it pinned:** the strip offers beta 2.1.0-beta2; do not take
  it mid-campaign. Every log records the firmware, and a change is flagged.
- [ ] **(g) A password (optional).** With auth off, anyone on the network can switch any outlet, the router's included.
  Matter is also on, another way in. If a password is set, put it in `~/.config/heteroserve/shelly.auth` (one line,
  `chmod 600`), never in the repo. The tool reads it with `--auth-file` or `SHELLY_AUTH_FILE`.
- [ ] **(h) Never switch outlet 3 (the router) off remotely.** A standing rule. The tool cannot switch anything.

## First run at home

**Steps 1–2 done 2026-10-05**, apart from reservation (b): see "Measured at home". The map was checked with busy loops,
not `llama-bench`.

1. **The strip on the router:** (a) and (b) above, `strip.json` updated, then `info` and `status`. Expect outlets on,
   `firmware 2.0.1`, the names, and the MAC check passing.
   - **Confirm the outlet map by load before any figure is attributed** (it changed on 2026-10-05): load one board for a minute (a
     `llama-bench` run on the Pi 5, say) and check that only its outlet (1) rises. The names are labels the user typed;
     only a load proves which board is behind which outlet.
2. **Idle floors, per outlet, ≥ 5 min.**
   - Boards booted and idle; `log --hz 5 --duration 360`; summarise a 300 s window.
   - Read the meter at a few watts: does `apower` hold steady non-zero values? At ~3 W the counter needs ~1.2 s per
     0.001 Wh step (arithmetic), so a 300 s window holds ~250 steps.
   - Check the Jetson's outlet against root `tegrastats` `POM_5V_IN` over the same window (AC ≥ DC module input).
   - Re-measure the RTT stalls and the fresh rate on the router's Wi-Fi.
3. **A first prefill-only / decode-only pair on one board** (the Pi 5), by the protocol above. **Done 2026-10-05**, as three
   cycles: see "Energy per token on the Pi 5".
   - ≥ 60 s of timed repetitions each, `-r` ≥ 3, `--progress` markers, an idle window before.
   - `summarise --idle --tokens`.
   - The first things to read: the trapezoid against the counter on a real board's load, and the J/token spread across
     repetitions.
4. **Then QR6:** choose the reference with the advisor (a known resistive AC load, or a reference wattmeter) before any J/token
   is quoted as more than "as measured by the strip".

## Files

| File | What it is |
|---|---|
| `testbed/scripts/shelly_strip.py` | `info` / `status` / `log` / `summarise`; read-only by construction (GET of five allow-listed paths) |
| `testbed/energy/strip.json` | host (`192.168.100.34`, DHCP, not reserved yet), MAC, model, firmware seen, outlet map, `do_not_switch_off` |
| `testbed/scripts/energy_mapcheck_idle.sh` | the 2026-10-05 run: the outlet-map check by load, then the idle floors |
| `testbed/scripts/score_energy_mapcheck_idle.py` | scores that run against its registration (map rule, summarise, by_minute, RTT, Jetson AC vs DC) |
| `testbed/dataset/energy_predictions_20261005.json` / `energy_results_20261005.json` | its registration (read-only) and its scored results |
| `testbed/scripts/energy_phase_run.py` | the energy runner: one strip log, idle windows, alternating prefill-only / decode-only llama-bench invocations with laptop-stamped `--progress` lines, a 5 s board sampler, preflight refusals (model hash, outlet name, a running llama process, `-r` < 3) |
| `testbed/scripts/score_energy_phase_run.py` | scores a runner folder: windows from the stamps, `summarise` per window, idle-subtracted J/token by both methods, the across-invocation spread, the registered checks |
| `testbed/dataset/energy_predictions_20261005b.json` / `energy_results_20261005b.json` | the first J/token run's registration (read-only) and its scored results |
| `testbed/scripts/register_pool_energy.py`, `energy_pool_run.py`, `score_energy_pool_run.py` | the pooled-energy registration, runner (`run_pool.sh` cells inside one strip log) and scorer |
| `testbed/dataset/energy_predictions_20261006_laptop_*.json` / `energy_results_20261006_laptop_*.json` | the laptop's four registrations (read-only) and their scored results |
| `testbed/results/energy_pool_20261005_215722/`, `energy_pool_20261006_024600/`, `energy_pool_combined_20261006.json` | the two pooled sessions (each with its `score.json`) and their pooled verdict |
| `.handoff/assets/energy-2026-10-05/` | the registration generators for the Jetson CPU redo and the laptop, the laptop chain (`run_laptop_chain.sh`), its disturbance check, and the laptop shakedown with its scratch registration |
| `testbed/tests/test_shelly_strip.py` | 35 tests: synthetic logs, a fake strip on 127.0.0.1, digest auth (RFC 7616 vectors), the read-only guard |
