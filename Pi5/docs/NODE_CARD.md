# Pi 5 node card

The numbers to quote for this node, and where each one comes from.

**Updated 2026-09-20.** The first real measurements now exist — see
`testbed/benchmarks.md` for full method and caveats. Rows below marked *measured*
are from the assembled bench; rows marked *estimate* are still unvalidated and must
not be cited as results.

## Identity

| Field | Value | Source |
|---|---|---|
| node_id | `pi5` | `config/node.env` |
| role | strong CPU worker | proposal §Hardware |
| SoC | Broadcom BCM2712 | vendor |
| CPU | 4 × Cortex-A76 @ 2.4 GHz | vendor |
| ISA | ARMv8.2-A + NEON + **dotprod** + FP16 + CRC | `/proc/cpuinfo` Features |
| RAM | 8 GB LPDDR4X | vendor |
| Usable | **7.63 GiB MemAvailable at rest** (7.9 GiB total) | *measured* 2026-09-20, swap off |
| Advertised to placement engine | 6144 MB | `RPC_MEM_MB` |
| Peak memory bandwidth | ~17 GB/s | vendor |
| PSU | Official 27 W USB-C PD | proposal budget |
| Cooling | official active cooler, pinned to full during sweeps | — |

## Why this node matters to the study

- It carries the ISA difference the dataset rests on. The A72 on the Pi 4 is
  ARMv8-A: **no dotprod, no FP16 arithmetic**. The A76 has both. Quantised GGUF
  kernels lean on SDOT/UDOT heavily, so this is the single largest source of the
  Pi4/Pi5 gap, and it is why both nodes build with explicit, different
  `-march` strings rather than `-march=native`.
- **It does *not* have i8mm.** i8mm arrives with ARMv8.6 (Cortex-A710 and later).
  Do not add `+i8mm` to `LLAMA_ARCH_FLAGS`: the build succeeds and the kernels trap
  at runtime. `06_verify_node.sh` warns if the kernel ever reports it.
- It fits a 7B Q4_K_M model (~4.7 GB) alone, so 7B Q4 is the **distribution
  baseline**, not evidence that pooling is required.
- **Correction (measured 2026-09-19): 7B at Q8_0 also fits on this node alone.**
  `mistral-7b-instruct-v0.2.Q8_0` is 7.17 GiB and loaded with ~250 MiB to spare, no
  swap, zero kernel OOM records, under both `-lm none` and `-lm mmap`. The earlier
  prediction that it would be OOM-killed was wrong. So 7B Q8_0 **cannot** serve as
  the pooling proof, and QR1's MVP wording is self-contradictory as written.
  The smallest configuration that genuinely exceeds this node is **14B at Q4_K_M**
  (Qwen2.5-14B-Instruct, 8.37 GiB measured) — which is QR1's *stretch* target.
- **Confirmed both ways, 2026-09-20.** 14B Q4 **fails to load on this node alone**
  (`llama_bench: error: failed to load model`, exit 1, and **zero kernel OOM records** —
  llama.cpp's own allocator refused, the kernel never had to kill anything), and **runs
  pooled** with the Pi 4 over wired RPC. That is QR1 satisfied. As this node's role
  implies, it should carry **as many layers as it can hold**, and the sweep says so with
  no interior optimum anywhere: 36/12 → 1.19 pp64 / 0.51 tg16, even 24/24 → 1.64 / 0.64,
  best **6/42 → 3.73 / 0.97** (+127% prefill, +52% decode over an even split). One step
  further, 4/44, and the **kernel OOM-kills** this node's `llama-bench` (exit 137,
  `anon-rss 7849552kB`) — so 42 of 48 layers of a 14B Q4 is this node's practical ceiling
  in a two-node pool, and the boundary must be predicted, not probed.
- Because a layer split is additive, the cluster cannot beat this node on a model
  this node can hold. Pooling buys capacity, not speed — every reported result must
  respect that.

## Measured, 2026-09-19/20 (TinyLlama-1.1B Q4_K_M unless stated)

Software: `llama.cpp` **as built on this node**, commit
`b23701f77d47dad9de834d59ebfcbe25c9e8b46f`, `-DGGML_RPC=ON`. Model byte-identical to
the Pi 4's copy (SHA-256 prefix `9fecc3b3cd76bba8`). All figures are the mean of three
repetitions; single-repetition runs are never reported.

> **Revision:** these numbers were produced on `b23701f7`, which is now also the
> cluster pin — moved forward on 2026-09-20 so the pin names the revision that was
> actually built and measured. See `docs/PIN_CONFLICT.md` for what that cost the Jetson.

| Quantity | Measured | Note |
|---|---|---|
| Prefill, pp512, 4 threads | **81.10 t/s** | compute-bound |
| Decode, tg128, 4 threads | **18.16 t/s** | bandwidth-bound |
| Decode, tg128, **3 threads** | **18.33 t/s** | **the optimum — 4 threads is 9% slower** |
| Prefill thread scaling, 1→4 | 3.84x | near-linear |
| Decode thread scaling, 1→4 | 1.38x (peak 1.51x at 3) | saturates, then regresses |
| Effect of `dotprod` on prefill | **2.96x** | same board, `-march=armv8-a` build |
| Effect of `dotprod` on decode | **1.00x — none** | memory-bound, ISA cannot help |
| 7B Q8_0 (7.17 GiB), decode | 1.51 t/s | fits; **below the 2 t/s QR2 floor** |
| microSD sequential read | 93.2 MB/s | SDR104 @ 200 MHz |
| Thermal | no throttling, peak 63 °C | `get_throttled=0x0` every run |
| CPU governor | no measurable effect | `ondemand` ramps under sustained load |

**Placement consequence:** set decode to **3 threads on this node**, not 4. A rule that
assumes "use all cores" leaves ~9% of decode throughput on the floor. The optimum is a
per-device constant and belongs in the characterisation dataset.

**Run-to-run variance:** `llama-bench`'s reported `±` understates it. The same config
minutes apart drifted ~2% on tg128 (17.19 → 16.79 → 16.71) as the board warmed 47→63 °C.
Quote the across-run spread, not the within-run one.

## Estimated envelope (to be replaced by measurement)

| Quantity | Estimate | Basis |
|---|---|---|
| Decode, 7B Q4 (4.7 GB) | 2.2–2.5 tok/s realistic; 3.6 tok/s at peak bandwidth | 17 GB/s ÷ 4.7 GB at 60–70% efficiency, proposal A.2 |
| TPOT, 7B Q4 | 400–450 ms | 1/decode |
| Power under load | ~5–8 W | proposal §Energy |
| Thermal | soft-throttle 80 °C, hard 85 °C | firmware |

The Baseline target of ≥2 tok/s (mean TPOT ≤500 ms) is only reachable on placements
that concentrate the model here and shed the slowest stage — which is exactly what
the straggler study (QR8) tests.

## Instrumentation

| Field | Value |
|---|---|
| INA226 channel | 1 |
| I²C address | `0x40` |
| Shunt | 0.01 Ω, high-side on the 5 V USB-C rail |
| Measures | SoC, RAM, NVMe/microSD, Ethernet PHY, active cooler — everything on that rail |
| Node-side telemetry | `heteroserve-telemetry` at 5 Hz, joined to the DAQ on `ts_utc` |

Energy is `E = ∫P dt ≈ Σ P·Δt` over the INA226 stream, reported as joules/token and
Wh per 1,000 tokens over the **serving path**, with total-system energy alongside.
