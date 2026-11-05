- 22:44 (arm-run 1, A, request 1): the user's message 'How long everything will take' and Claude's reply (two short reads of the session log); recorded as found.
- Arm-run 1 (A, the engine, laptop 32 + Pi 4 13 + Pi 5 14 + Jetson 5): complete, 5 requests, client TPOT P50 3,547.7 ms (mean
  3,545.9; predicted 3,666.8, -3.3 %), P95 3,584.3, TTFT P50 155.4 s; done 23:07:50, the Jetson's clocks restored.
- Arm-run 2 (B, prima.cpp) FAILED at 23:24:57 (rc 1), not a copy problem (every partial copy verified): Halda decided laptop
  42 / Pi 5 11 / Pi 4 11 / Jetson 0 -- the laptop had 27,821 MiB free at its start (26,452 at the decision run), so its budget
  (x 0.8) took 42 blocks, the Jetson was left one and removed ("No layer is assigned to me, exit."). qr3_tpot.py's prima_start
  treated that exit as a crash ("FATAL: prima.cpp rank 3 on jetson exited") and stopped every rank. The three-node ring had
  every layer inside its copies (Pi 5 0-10, Pi 4 11-21).
- The user, ~22:47: "Stop at safest point and then /handoff:cutoff". .handoff/assets/prima4-2026-10-21/stop_after_arm_20261021.sh
  started watching only at 23:15:48 (after arm-run 2 had begun) and stopped the session when arm-run 2 ended: strip logger
  SIGINT, unit stopped 23:25:04, the Jetson's guard stopped (0 trips), every node idle, the Jetson's clocks at their defaults,
  no serve_pool.sh state. The session was not scored (incomplete: 1 A, 0 B).
