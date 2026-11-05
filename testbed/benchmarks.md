# Benchmark results

All runs use the same pinned software and the same model bytes on every node, so any
difference is hardware.

| | |
|---|---|
| `llama.cpp` commit | `b23701f77d47dad9de834d59ebfcbe25c9e8b46f` (2026-09-19) |
| Build | `cmake -DCMAKE_BUILD_TYPE=Release -DGGML_RPC=ON -DLLAMA_CURL=ON`, ccache |
| Model | TinyLlama-1.1B-Chat-v1.0, Q4_K_M, 636.18 MiB, 1.10 B params |
| Model SHA256 (first 16) | `9fecc3b3cd76bba8` — **identical on both nodes** |
| Tool | `llama-bench`, `pp512` = prefill, `tg128` = decode |

`i8mm` was tested and **failed on both nodes** during cmake configure, independently
confirming neither A72 nor A76 has the int8 matmul extension.

## Single node, 4 threads

| Node | pp512 (prefill) t/s | tg128 (decode) t/s |
|---|---|---|
| Pi 5 (Cortex-A76 @ 2.4 GHz) | **80.33** | **16.71** |
| Pi 4 (Cortex-A72 @ 1.8 GHz) | **12.48** | **5.50** |
| **Ratio Pi 5 : Pi 4** | **6.44x** | **3.04x** |

Pi 4 figures re-taken from cold on 2026-09-19 after a power cycle, `throttled=0x0` at
both start (53.0 °C) and end (73.0 °C).

## Pi 5 thread scaling — the core premise, measured

| threads | pp512 | vs 1 thread | tg128 | vs 1 thread |
|---:|---:|---:|---:|---:|
| 1 | 20.91 | 1.00x | 12.10 | 1.00x |
| 2 | 42.45 | 2.03x | 17.78 | 1.47x |
| 3 | 61.34 | 2.93x | **18.33** | **1.51x** |
| 4 | 80.33 | 3.84x | 16.71 | 1.38x |

**Prefill scales almost linearly** — 3.84x on 4 cores. Compute-bound, as the
literature says.

**Decode saturates at 3 threads and then regresses.** Going from 3 to 4 threads makes
decode **9% slower**. That is memory-bandwidth saturation: the fourth core adds
contention, not throughput.

Two consequences:

1. This is direct evidence on this project's own hardware for the prefill/decode
   asymmetry the whole study rests on. It did not have to be taken on faith from
   DistServe or Splitwise — it reproduces on a Raspberry Pi.
2. **Decode should be configured with 3 threads on the Pi 5, not 4.** A placement
   rule that assumes "use all cores" is leaving ~9% of decode throughput on the floor.
   Worth checking whether the same holds on every node — the optimum is probably a
   per-device constant that belongs in the characterisation dataset.

## How much did thermal throttling actually cost? ~2%

The first Pi 4 sweep ran hot — it started at 67.6 °C straight after a 29-minute build,
peaked at 77.9 °C and set `throttled=0x80000` (bit 19, *soft temperature limit has
occurred*). That looked like it would invalidate the comparison. Re-measured from cold:

| Pi 4, 4 threads | Throttled run | Cold run | Difference |
|---|---:|---:|---:|
| pp512 | 12.25 | 12.48 | **+1.9%** |
| tg128 | 5.47 | 5.50 | **+0.5%** |

**The throttling cost about 2%, inside run-to-run variance.** Worth checking, not worth
discarding data over. Empirically the flag tripped somewhere between 73 °C (clean) and
77.9 °C (tripped), so the Pi 4 has more headroom than its 60 °C soft-limit spec implies.
Record `get_throttled` alongside every run regardless — it costs nothing and turns a
question into a fact.

## Pi 4 thread scaling (measured while throttled; ~2% low, shape unaffected)

| threads | pp512 | vs 1 thread | tg128 | vs 1 thread |
|---:|---:|---:|---:|---:|
| 1 | 3.23 | 1.00x | 2.83 | 1.00x |
| 2 | 6.44 | 1.99x | 5.28 | 1.87x |
| 3 | 9.49 | 2.94x | **5.69** | **2.01x** |
| 4 | 12.25 | 3.79x | 5.47 | 1.93x |

**The shape reproduces exactly.** Prefill scales near-linearly (3.79x on 4 cores);
decode peaks at **3 threads** and regresses at 4, identical to the Pi 5. Two different
microarchitectures, one throttled and one not, and the same answer: *decode should not
use all cores.* Throttling scales everything down but does not change where the knee
is, so this conclusion survives the thermal problem.

## Pi 4 thread sweep re-taken from cold (2026-09-20) — and what it revealed

The sweep above was measured while the board was throttling, and carried that caveat.
Re-run from cold on `b23701f7`, same model bytes (`9fecc3b3cd76bba8`), `-r 3`, with
temperature and `get_throttled` logged at the start and end of every configuration.

| threads | pp512 | tg128 | temp start → end | throttle at end |
|---:|---:|---:|---|---|
| 1 | 3.23 | 2.78 | 47.7 → 62.3 °C | `0x0` |
| 2 | 6.44 | 5.47 | 61.8 → 70.6 °C | `0x0` |
| 3 | 9.53 | **5.63** | 70.6 → 73.0 °C | `0x0` |
| 4 | **12.10** | 5.29 | 74.5 → 76.4 °C | **`0x80000`** |

**The conclusion holds, and is now clean where it matters.** Decode peaks at **3
threads** and regresses **6.0%** at 4 — measured on a board that was demonstrably not
throttling for configurations 1–3 (`0x0` throughout). Prefill scales 3.75x on 4 cores.
Two microarchitectures, two thermal histories, same answer: *decode should not use all
cores.*

### The sweep heats the board it is measuring

This is the part worth keeping. Starting cold did **not** produce a cleaner 4-thread
number — it produced a *worse* one:

| 4 threads | earlier run | this "cold" run |
|---|---:|---:|
| pp512 | 12.25 | **12.10** |
| tg128 | 5.47 | **5.29** |
| peak temp | 77.9 °C | **83.7 °C**, `0x80008` — soft limit *active* mid-run |

A sequential 1→2→3→4 sweep heats monotonically, so the highest thread count is always
measured on the hottest board. The Pi 4 climbed 47.7 → 83.7 °C across this sweep and
tripped the soft temperature limit *during* the 4-thread configuration. The earlier
run happened to reach 4 threads cooler than this one did.

**So "re-take it from cold" does not remove the caveat at t=4; it moves it.** On this
node, with this cooling, a thermally clean 4-thread figure is not obtainable from a
sequential sweep at all. Either cool between configurations or randomise their order —
and either way, keep logging the throttle word, because it is the only thing that made
this visible.

Cross-check that the effect is thermal and not noise: at 1 and 2 threads, where the
board is still cool in both runs, the two sweeps agree to three significant figures
(3.23/3.23 and 6.44/6.44 on pp512). The divergence appears only where the heat does.

**The standing action item is unchanged and now better evidenced: the Pi 4 needs
better cooling before it can produce citable 4-thread numbers.**

## The `asimddp` prediction — confirmed in direction, cause not yet isolated

Predicted: the Pi 5 / Pi 4 gap would exceed the 2.4/1.8 = **1.33x** clock ratio,
because the A72 lacks the ARM dot-product instruction `llama.cpp`'s quantised kernels
use.

Measured at 4 threads, both runs clean: **prefill 6.44x, decode 3.04x.** Far above
1.33x, so the prediction holds.

**But this does not prove the cause is `asimddp`.** Three differences are confounded:

1. missing `asimddp` on the A72,
2. the A76 is a far wider, deeper core — roughly 1.5–2x the IPC at equal clock even
   without any ISA difference,
3. the Pi 5 has faster memory (LPDDR4X-4267 vs LPDDR4-3200), which matters most for
   the bandwidth-bound decode phase.

Attributing 6.44x to the instruction alone would be wrong. To separate them, a second
`llama.cpp` is being built **on the Pi 5** with `-DGGML_NATIVE=OFF
-DGGML_CPU_ARM_ARCH=armv8-a`, which makes cmake report `HAVE_DOTPROD - Failed`.
Same board, same model, same threads, dot-product kernels compiled out — the delta is
then the instruction's contribution alone, with microarchitecture and memory held
constant.

### Result: dot-product is worth 2.96x on prefill and **nothing** on decode

Pi 5, 4 threads, same model, board at 45.5 °C, `throttled=0x0`:

| build | pp512 (prefill) | tg128 (decode) |
|---|---:|---:|
| normal (`HAVE_DOTPROD` yes) | **81.10** | 18.16 |
| `-march=armv8-a` (`HAVE_DOTPROD - Failed`) | **27.38** | 18.59 |
| **effect of dot-product** | **2.96x faster** | **1.00x — no effect** |

Decode is *marginally faster without* it (18.59 vs 18.16), i.e. unchanged inside noise.

**This is the prefill/decode asymmetry confirmed a second time, by a completely
independent mechanism.** The thread sweep showed it through core scaling; this shows it
through the instruction set. Prefill is compute-bound, so widening the multiply-
accumulate path nearly triples it. Decode is memory-bandwidth-bound, so the same
widening buys exactly nothing — the core is already waiting on memory.

### Decomposing the Pi 5 : Pi 4 gap

| | prefill | decode |
|---|---:|---:|
| Total measured gap | 6.44x | 3.04x |
| ...attributable to dot-product | **2.96x** | **1.00x** |
| ...remaining (clock + microarchitecture + memory bandwidth) | 2.18x | 3.04x |

Cross-check: the Pi 5 *without* dot-product scores 27.38 against the Pi 4's 12.48 —
a 2.19x gap from non-ISA causes alone. And 2.96 x 2.19 = 6.48, against the 6.44x
measured directly. The decomposition is arithmetically consistent.

So for **prefill**, over half the Pi 4's disadvantage is one missing instruction. For
**decode**, none of it is — that gap is clock and memory bandwidth, and no compiler
flag will close it.

### Why this matters for the placement rule

This is a placement rule falling out of measurement, which is exactly what the project
set out to produce:

> **Prefill belongs on dot-product-capable nodes. Decode does not care.**

A node without `asimddp` is penalised ~3x for prefill and ~0% for decode. So in a
heterogeneous cluster the Pi 4 is a far better decode worker than its headline specs
suggest, and a far worse prefill worker. Sorting nodes by a single "speed" score — the
obvious naive approach — would get this backwards.

**Check before generalising:** verify on the Jetson Nano and the RISC-V node, and at a
larger model size. This is one 1.1B model at Q4_K_M; the balance may shift when weights
no longer fit in cache.

## CPU governor — tested, and it does not matter here

| governor | clock | pp512 | tg128 |
|---|---|---|---|
| `ondemand` (default) | 1600 MHz idle | 80.42 | 17.19 |
| `performance` | 2400 MHz pinned | 80.34 | 16.79 |

No meaningful difference. `ondemand` ramps to full clock under sustained load, and
`llama-bench` sustains load long enough that the ramp is invisible. Setting
`performance` is still worth doing to remove a variable, but it should not be claimed
as a speedup.

## A caution about error bars

`llama-bench` reports very tight within-run deviations (±0.00 to ±0.14). But the same
configuration measured minutes apart drifted ~2% on tg128 (17.19 -> 16.79 -> 16.71) as
the board warmed from 47 to 63 °C. **The reported ± understates true run-to-run
variance.** Repeat across separate runs and report that spread, not the within-run one.

No thermal throttling occurred in any **Pi 5** run (`get_throttled=0x0`, peak 63 °C).
The **Pi 4 throttled in every run** — see the warning above. Build times tell the same
story: Pi 5 **570 s**, Pi 4 **1735 s** (3.0x), and that build is what pre-heated it.

**Action: the Pi 4 needs better cooling before it can produce citable numbers.**
Its soft limit trips at 60 °C, and it idles at 52 °C — so it has almost no headroom.

> **Correction, 2026-09-25.** The 60 °C figure does not hold for this board. In the `-r 3`
> pooled re-run the Pi 4 sat at **65.7 °C at the end of a cell with `get_throttled` still
> `0x0`**, so its soft limit is above that — consistent with the empirical bracket measured
> on 2026-09-19 (earlier in this file): clean at 73 °C, tripped by 77.9 °C. 60 °C is the
> Raspberry Pi 3B+'s `temp_soft_limit` default, and `temp_soft_limit` is unset in this
> board's `config.txt`. The board now has a PWM fan
> (`dtoverlay=pwm-gpio-fan,fan_gpio=18`). Note also that bit 19 of `get_throttled` is
> **sticky until reboot**, so "throttled in every run" cannot be told apart from
> "throttled once, before the first run" unless the board was rebooted between runs.


---

# Distributed inference over `llama.cpp` RPC

First working distributed run, 2026-09-19. Pi 5 as the main node, Pi 4 as an
`ggml-rpc-server` worker on `192.168.1.106:50052`, over 5 GHz Wi-Fi.

| Configuration | pp512 (prefill) | tg128 (decode) |
|---|---:|---:|
| Pi 5 alone | **81.10** | **18.16** |
| Pi 4 alone | 12.48 | 5.50 |
| Pi 5 + Pi 4 over RPC | **6.43** | **4.84** |

## Distribution made it slower than either node by itself

- **12.6x slower** than the Pi 5 alone on prefill, 3.8x slower on decode.
- **1.9x slower than the Pi 4 alone** — the worker doing all the work unaided beats
  the same worker driven over RPC.

This is not a defect and not a misconfiguration. It is `llama.cpp`'s RPC backend
behaving exactly as its own documentation describes: a mechanism for **pooling memory
so a too-large model can run at all**, not for accelerating computation. The proposal
already states this; this is the measurement that backs the claim with a number
instead of a citation.

## What was actually observed

`--rpc` placed **every** layer on the remote worker (`ngl -1`, backend reported as
`RPC`). During the run:

| | |
|---|---|
| Worker (`ggml-rpc-server`) | 154% CPU, 687 MB resident — the model transferred and it is computing |
| Main node (`llama-bench`) | 0.2% CPU — idle, waiting |
| Network during compute | ~7 KiB / 20 s, i.e. **nothing** |

So the cost is **not** bandwidth. The link moved almost no data once weights were
loaded. The cost is **per-operation round-trip latency**: every tensor op becomes a
network exchange, and at ~7 ms RTT over Wi-Fi those serialise into the critical path.
The Pi 5 cannot start work until the Pi 4 answers, and it has nothing else to do.

**This reframes the link-tier variable.** The naive expectation is that a faster link
helps because more bytes move. These numbers say the opposite: with a model this size
the link is nearly idle, and what matters is *latency per round trip*, not throughput.
A 2.5 GbE switch may therefore help far less than expected, while anything that cuts
RTT — wired instead of Wi-Fi, fewer hops — may help far more. Worth testing directly
once the switch arrives: measure RTT and throughput as separate variables, because
this run suggests they are not interchangeable.

## The honest caveat

**This model fits comfortably on either node** — 636 MiB against 8 GB of RAM. So there
was no memory to pool and nothing to gain, only overhead to pay. This measurement
establishes the *cost* of distribution; it says nothing yet about the *benefit*.

The benefit case requires a model that does **not** fit on one node. At that point the
comparison is no longer "distributed vs single node" but "distributed vs cannot run at
all", and the 1.9x penalty may be entirely acceptable. **That is the experiment the
project exists to run**, and it needs a model in the 7B-at-Q8_0 range (~7.5 GB), which
exceeds either board's 8 GB once KV cache and overhead are counted.

Until then, quote these numbers as the overhead of RPC, not as evidence that
distribution does not work.

---

# 7B at Q8_0 on a single Pi 5 — it fits. I predicted it would not.

`mistral-7b-instruct-v0.2.Q8_0.gguf`, 7.70 GB / **7.17 GiB**, on a Pi 5 with 7.87 GiB
total and **7.59 GiB available**, **no swap** (so an over-commit would be a clean kill,
not a thrash).

| load mode | pp64 | tg16 | wall time | exit | kernel OOM lines |
|---|---:|---:|---:|---:|---:|
| `-lm none` (real allocation, no mmap) | **6.88** | **1.51** | 136 s | 0 | 0 |
| `-lm mmap` (default) | 6.47 | 1.48 | 224 s | 0 | 0 |

**It loads and runs on one node, with roughly 250 MiB to spare.** I predicted it would
not fit and would be OOM-killed. That was wrong. Both load modes completed cleanly with
zero OOM records, and the no-mmap run — which cannot fall back to paging — is the proof:
the model genuinely resides in RAM.

A related correction: mid-run I saw `llama-bench` at 93% RSS and 23% CPU with the SD
card reading ~1 MiB/s, and read that as thrashing. It was not. `-lm none` (which cannot
page) is only 6% faster than mmap, so paging was never the constraint — the low CPU was
the load phase, and the SD reads were the model being read in once.

## What this does to QR1

> QR1 — *"Model size the cluster can run but no single serving node can: MVP at least
> 7B at Q8_0 (~7.5 GB) or 13B at Q4."*

**The first half of that target is now met by one Pi 5 on its own**, so 7B-at-Q8_0 no
longer demonstrates the thing the requirement exists to demonstrate. It cannot serve as
the pooling proof. Options:

- **13B at Q4_K_M** (~7.9 GB) — just over one node's ceiling. The honest minimum.
- **13B at Q8_0** (~14 GB) — comfortably impossible on one node, needs both.
- **14B at Q4** — already written as the stretch target.

Recommend re-baselining QR1 on 13B Q4_K_M and keeping 14B Q4 as stretch. Note this cuts
both ways: the Pi 5 being *more* capable than assumed is good news for the project and
bad news for this particular requirement's wording.

## And QR2 is not met at this model size

> QR2 — *"Single-stream decode throughput: MVP at least 2 tokens/s."*

Measured **1.51 tok/s** at 7B Q8_0 on the best node in the cluster, single-node, no
distribution overhead. That is **below the MVP floor**, and distribution can only make
it worse. Either QR2's floor is unreachable at Q8_0 and should be stated per
quantisation, or Q8_0 is the wrong operating point and Q4_K_M is the realistic one
(4.07 GiB, and the 1.1B results suggest decode would be several times faster).

Worth settling before the proposal is submitted, because as written the two
requirements are in tension: QR1 pushes toward a model size at which QR2 cannot be met.

---

# Link tiers measured: Wi-Fi vs wired GbE

Pi 5, same node, same method (raw `nc`, 400 MB, no SSH crypto). Router = AIS F6107A,
LAN ports confirmed **1000 Mb/s full duplex**.

| | throughput | TCP-connect RTT |
|---|---:|---:|
| Tier 1 — Wi-Fi 5 GHz | 20.9 MB/s (167 Mbit/s) | 7.6–11.6 ms |
| Tier 2 — wired GbE | **87.0 MB/s (696 Mbit/s)** | 6.6–8.1 ms |
| **gain** | **4.2x** | **~1.2x** |

**Bandwidth improved 4.2x; latency barely moved.** That asymmetry is the whole point of
the pending experiment: the RPC run was bottlenecked on round trips, not bytes, so if
that reading is right, a 4.2x bandwidth gain should produce very little RPC improvement.

*Caveat on the RTT figures:* TCP-connect time includes the handshake plus process
wake-up, and both tiers floor out near 7 ms, which is implausibly high for wired GbE
(true wire RTT should be well under 1 ms). These numbers are dominated by scheduling
overhead, not the network. Treat the RTT column as an upper bound, not a measurement —
a proper test needs ICMP or a tight TCP echo loop.

## Topology note

The router bridges LAN ports and Wi-Fi into **one L2 broadcast domain** (Wi-Fi hosts
appear in the wired ARP table). The cluster therefore runs a separate
**192.168.10.0/24** overlay on that shared L2, with no gateway and `never-default` set,
while management and internet stay on `192.168.1.0/24` over Wi-Fi. Different subnets
mean the kernel selects the interface deterministically by destination, so "wired run"
and "Wi-Fi run" are unambiguous rather than whichever path happened to win.

| node | wired (experiment) | Wi-Fi (management) |
|---|---|---|
| laptop | 192.168.10.5 | 192.168.1.105 |
| Pi 5 | 192.168.10.11 — `ssh pi5-eth` | 192.168.1.101 — `ssh pi5` |
| Pi 4 | 192.168.10.12 — `ssh pi4-eth` | 192.168.1.106 — `ssh pi4` |

---

# Link tier A/B: wired GbE vs 5 GHz Wi-Fi (2026-09-19)

Pi 5 main, Pi 4 RPC worker, TinyLlama-1.1B Q4_K_M, 4 threads, `-r 3`.
Paths **alternated** across two rounds to cancel thermal and time drift.

| Path | pp128 (prefill) | tg32 (decode) |
|---|---:|---:|
| Wired, round 1 | 6.61 ± 0.00 | 5.47 ± 0.00 |
| Wired, round 2 | 6.61 ± 0.00 | 5.46 ± 0.01 |
| Wi-Fi, round 1 | 6.57 ± 0.00 | 4.90 ± 0.02 |
| Wi-Fi, round 2 | 6.57 ± 0.00 | 4.91 ± 0.02 |
| **Wired advantage** | **+0.6%** | **+11.6%** |

Rounds agree to two decimals, so this is reproducible, not noise.

## Link characteristics being compared

| | Wi-Fi 5 GHz | Wired GbE | change |
|---|---:|---:|---:|
| Throughput, pi5 -> pi4 | 27.3 MB/s | 71.4 MB/s | 2.6x |
| RTT | ~7 ms | 0.83 / 2.30 / 5.23 ms | ~3x lower |

## The result: latency affects decode, bandwidth affects nothing

A 2.6x bandwidth increase and 3x latency reduction bought **+0.6% on prefill** and
**+11.6% on decode**. The asymmetry is the finding, and it runs opposite to intuition —
prefill moves far more data per call, so it "should" be the one that cares about the
link.

The mechanism explains it. Decode generates tokens **one at a time**, and each token
needs a round trip to the remote backend, so 32 tokens serialise 32+ round trips;
cutting RTT from ~7 ms to ~2 ms removes real wall-clock from the critical path.
Prefill processes its 128 tokens as **one batch**, so the same round-trip cost is
amortised across the whole batch and is already negligible against compute.

**So the link variable is really a latency variable, and it only acts on decode.**

## Consequence for the 2.5 GbE hardware

The gain came from **latency, not bandwidth** — bandwidth was never saturated
(the link sat near-idle at ~7 KiB/20 s during compute). Going GbE -> 2.5 GbE raises
bandwidth that is not the constraint, and barely changes RTT, which already sits
under 1 ms on a wired switch.

**Predicted gain from the 2.5 GbE switch and USB adapters (~4,000 THB): close to zero.**
That is a testable prediction rather than a certainty, but the money is better spent
elsewhere until a measurement contradicts it. Wi-Fi (Tier 1) and the router's GbE
(Tier 2) already give two points, and the curve between them is nearly flat.

## Correction to an earlier reading

A first single-repetition run reported wired *slower* on prefill (4.75 vs 6.44) and was
written up as "wired gave nothing". That was wrong: `-r 1` produced a `± 0.00` that
looked like precision but was a single sample, and the 4.75 was an outlier. Replicated
and alternated, prefill is unchanged and decode gains 11.6%. **Never report a
`llama-bench` figure taken at `-r 1`.**


---

# QR1 proved: a model that fits no single node runs on the pooled pair (2026-09-20; throughput re-run at `-r 3` 2026-09-25)

**This is the experiment the project exists to run.** Everything before it established
the *cost* of distribution; nothing had yet shown the *benefit*, because every model
tested so far fitted on one node.

Model: **Qwen2.5-14B-Instruct Q4_K_M**, 8.37 GiB, Apache-2.0, pinned to repo revision
`b466e1f8…`, all three split parts SHA-256 verified against upstream before use
(`models.md`). Software `b23701f7` on both nodes. Link: wired GbE overlay,
192.168.10.11 ↔ .12.

> **Re-run 2026-09-25 at `-r 3`.** This sweep was first taken on 2026-09-20 at
> `llama-bench -r 1` — one sample per figure, which the house rules forbid reporting. Every
> throughput figure in this section is now the **mean of three repetitions**, with the
> min–max across those repetitions; the superseded single-sample values stay in the table
> for traceability. What changed between the two runs, and nothing else:
>
> | | 2026-09-20 | 2026-09-25 |
> |---|---|---|
> | repetitions | 1 | **3** (`-oe jsonl` records each one) |
> | order of splits | sequential | **interleaved** `20 6 30 12 36 8 24 16` |
> | thermal logging | Pi 4 only, after each cell | **both boards, before and after** |
> | link path | direct cable, Pi 5 ↔ Pi 4 | **via the router's switch** (one hop, same addresses) |
> | boards before the run | not rebooted (Pi 4 flag already `0x80000`) | **rebooted, both `0x0`** |
>
> Model bytes were re-verified against upstream before the run (`models.md`).

## Step 1 — the negative. It does not load on one node.

    llama_bench: error: failed to load model 'qwen2.5-14b-instruct-q4_k_m-00001-of-00003.gguf'
    EXIT=1
    oom before: 0     oom after: 0

Pi 5 alone, `-ngl 0`, swap off, 7.45 GiB available against an 8.37 GiB model. Note
**zero kernel OOM records on either side of the run** — llama.cpp's own allocator
refused; the kernel never had to kill anything. That is a clean capacity failure, not a
thrash and not a crash. The Pi 4 was not tested separately because it has *less* free
memory (7.20 GiB vs 7.45 GiB), so its failure follows a fortiori.

| node | MemAvailable at rest | holds 8.37 GiB alone? |
|---|---:|---|
| Pi 5 | 7,629 MiB (7.45 GiB) | **no — measured** |
| Pi 4 | 7,373 MiB (7.20 GiB) | no — by arithmetic |
| pooled | 15,002 MiB (14.65 GiB) | yes |

## Step 2 — the positive. Pooled, it runs.

Pi 5 as the main node (CPU), Pi 4 as an `ggml-rpc-server` worker, `-ngl N` placing N of
the model's 48 layers on the worker and the remaining 48−N on the Pi 5.

*Corrected 2026-09-27: that sentence, and the two layer columns below, are off by one. At
this pin `-ngl N` counts the **output layer** as one of the N and offloads it first
(`llama-model.cpp`, `load_tensors`, logs "offloading output layer" and then "offloading N-1
repeating layers"). So at `ngl=N` the Pi 4 held **N−1 blocks plus the output head**, and the
Pi 5 **49−N blocks plus the token embedding**: the optimum `ngl=6` is Pi 5 43 / Pi 4 5 +
head, and the OOM-killed `ngl=4` put 45 blocks on the Pi 5. The measurements are unchanged;
only the labels were wrong. The same holds for every `-ngl` figure in this file, including
the RPC smoke test's `-ngl 11` (10 blocks + head on the Jetson). See "2026-09-27" below.*

Tokens per second. Each figure is the mean of three repetitions, with the min–max across
them in brackets — a *repetition* range from one invocation, not an across-run spread
(see "A caution about error bars").

| layers on the Pi 4 | layers on the Pi 5 | pp64 (prefill) | tg16 (decode) | exit | 2026-09-20 `-r 1` (superseded) |
|---:|---:|---:|---:|---:|---:|
| 36 | 12 | 1.198 [1.190–1.204] | 0.514 [0.514–0.514] | 0 | 1.19 / 0.51 |
| 30 | 18 | 1.404 [1.403–1.405] | 0.566 [0.565–0.569] | 0 | 1.38 / 0.57 |
| 24 | 24 | 1.665 [1.661–1.667] | 0.628 [0.626–0.629] | 0 | 1.64 / 0.64 |
| 20 | 28 | 1.900 [1.899–1.901] | 0.700 [0.698–0.702] | 0 | 1.87 / 0.69 |
| 16 | 32 | 2.207 [2.206–2.208] | 0.753 [0.745–0.757] | 0 | 2.19 / 0.76 |
| 12 | 36 | 2.644 [2.643–2.646] | 0.826 [0.826–0.826] | 0 | 2.59 / 0.82 |
| 8 | 40 | 3.258 [3.255–3.262] | 0.901 [0.900–0.901] | 0 | 3.24 / 0.90 |
| **6** | **42** | **3.775 [3.773–3.779]** | **0.968 [0.967–0.970]** | 0 | 3.73 / 0.97 |
| 4 | 44 | — | — | **137 (SIGKILL)** | *2026-09-20 only; not re-run* |

The single-sample values held up: every one is within 2.1% of its three-repetition mean,
and prefill came in slightly higher throughout.

Every configuration completed, in both runs. **A model that loads on neither node runs on both.**
That is QR1's requirement met, and it is met at the *stretch* target (14B at Q4), not
the MVP one.

## The placement rule falls straight out of it

Throughput is **monotonically decreasing** in layers-on-the-straggler across the whole
sweep, in both phases, with no interior optimum anywhere. Against the even 24/24 split:

| vs 24/24 | prefill | decode |
|---|---:|---:|
| 36/12 (straggler-heavy) | **−28%** | **−18%** |
| 6/42 (straggler-light, the optimum) | **+127%** | **+54%** |

End to end, 36/12 → 6/42 is a **3.15x** swing in prefill and **1.88x** in decode — on
identical hardware, identical model bytes and an identical link. Only the placement
changed.

**How firm "monotonic" is.** At every adjacent step, in both phases, the repetition ranges
of neighbouring splits **do not overlap**. The smallest step is +14.1% in prefill (24 → 20)
and +7.5% in decode (20 → 16 and 8 → 6) — 7x and 3.7x the ~2% run-to-run drift measured
earlier on a single configuration. And because the splits were run in interleaved order,
every adjacent-split comparison is between two separate invocations minutes apart, which
is the across-run comparison the error-bar caution asks for. Warming over the sweep cannot
have produced the trend either: it would have had to line up with an order that jumps
20 → 6 → 30 → 12 → 36 → 8 → 24 → 16.

> **Give the slower node as few layers as memory allows.** An even split is not neutral —
> it is a choice, and on this pair it is the wrong one by a factor of two.

This is the QR8 comparison (rule vs even split) appearing directly in measurement rather
than being argued from specs, and it is a larger effect than the proposal's MVP target
for that requirement (≥10% TPOT improvement) assumes.

**There is no interior optimum anywhere — the curve runs straight into the memory wall.**
`ngl=6` (42 of 48 layers on the Pi 5) is the fastest configuration measured, and `ngl=4`
is killed. So the best split is not a balance point at all:

> **The optimal placement is exactly the memory boundary** — the smallest offload that
> still lets the model fit. Pooling here is a memory escape valve, not a work-sharing
> mechanism, and every layer handed to the straggler beyond that minimum is pure loss.

That is a measured fact with a demonstrated failure on the far side of it, not an
extrapolation.

### The failure mode changes at the boundary, and that is an operational warning

The two failures in this experiment are **not the same kind**:

| configuration | what happened | exit | kernel OOM record |
|---|---|---:|---|
| `ngl=0` — whole model on the Pi 5 | llama.cpp's allocator **refused up front** | 1 | **none** |
| `ngl=4` — 44 layers on the Pi 5 | **kernel OOM-killed the process** | **137** (SIGKILL) | yes |

    llama-bench invoked oom-killer
    Out of memory: Killed process 3160 (llama-bench) total-vm:16762864kB anon-rss:7849552kB

At `ngl=0` the model obviously cannot fit and llama.cpp declines cleanly. At `ngl=4` the
allocation is small enough to *attempt* and too large to *complete*, so the kernel kills
it mid-flight with 7.49 GiB of anonymous memory resident.

**Consequence for the placement engine: the boundary cannot be found safely by trial.**
Walking `ngl` downward to find the optimum means the step past it is a hard kill, not an
error return — and with swap off there is no soft-landing region in between. The engine
must predict the boundary from measured per-layer footprint and node memory, and stop
short of it, rather than discovering it empirically on a live run.

## And QR2 is comprehensively not met at this model size

**0.968 tok/s decode at the best split** (0.967–0.970 over three repetitions), against an
MVP floor of 2 tok/s — a factor of **2.1** short.

*Corrected 2026-09-25: this paragraph previously said 0.64 tok/s and 3.1x — the even
24/24 split's figure, left over from before the low-offload splits were measured. The
README and both proposals already used the best split.* This is consistent with, and worse than, the 1.51 tok/s already measured for
7B Q8_0 on a single Pi 5, exactly as the additive-pipeline bound predicts: pooling buys
capacity, not speed, and the cluster cannot beat its fastest member on a model that
member could hold.

The two requirements are therefore in direct tension, now with numbers on both sides:
**QR1 is only satisfiable at a model size where QR2 cannot be satisfied.** Settle this
with the advisor before submission — either QR2's floor becomes per-quantisation, or
QR1's proof model is accepted as a capacity demonstration with throughput reported as
found.

## Reproducing it

`testbed/scripts/run_14b_pooled.sh`, driven from the ThinkPad control node:

    NGL_LIST="20 6 30 12 36 8 24 16" bash testbed/scripts/run_14b_pooled.sh    # ~62 min

Raw logs: **`testbed/results/pool14b_20260925_025332.log`** — the `-r 3` re-run, source of
every figure above — and `pool14b_full_20260920.log`, the superseded `-r 1` run. Recompute
the table, the ratios and the adjacent-step range check from the raw JSON lines with:

    python3 testbed/scripts/summarise_pool14b.py testbed/results/pool14b_20260925_025332.log

Quote the ranges, not the markdown table's `±`: at `-r 3` it *still* prints `± 0.00`
whenever the standard deviation is under 0.005 tok/s, which on the page is identical to the
one-sample artifact.

### Thermal record of the re-run

Both boards were rebooted shortly before, so `get_throttled` began at `0x0` on both. The
**Pi 5 stayed at `0x0` throughout** (45.5–55.4 °C at the snapshots). **The Pi 4 raised
`0x80000` — soft temperature limit occurred — during the ngl=30 cell,** its second-heaviest
load. The bit is sticky, so the cells after it in run order (12, 36, 8, 24, 16) cannot be
certified throttle-free by that flag. No snapshot exceeded 71.5 °C, below even the lower
end of the trip window measured on 2026-09-19 (clean at 73 °C, tripped by 77.9 °C), so the
peak fell *between* the before and after readings: bracketing a cell cannot see its peak.

What the data show despite that: the repetition ranges stay narrow (widest 1.2%, the
36/12 prefill), and every adjacent step still separates. That 36/12 prefill is the one
within-cell trend in the table — its repetitions decline 1.204 → 1.200 → 1.190 — and it is
the heaviest Pi 4 load, which fits the board warming inside the cell. It does not touch the
ordering: the next step, to 30/18, is +17%. **To certify every cell, poll `get_throttled`
bit 3 (limit active *now*) or `vcgencmd measure_clock arm` on the Pi 4 during the cell.**

Two traps are baked into that script as comments, both found the hard way here:

1. **`pkill -f ggml-rpc-server` kills the launcher.** The orchestrating `bash -c` line
   contains that string, so `-f` matches its own shell and the worker is never started.
   Use `pkill -x`, which matches the process name.
2. **Never probe the worker with a bare TCP connect.** `ss` reports `LISTEN 0 1` — a
   backlog of exactly one — and a `/dev/tcp` connect-then-close **kills the server**; it
   prints its banner and vanishes. The readiness check must read the listening socket
   from the worker's own side (`ss` over ssh) and never open a client connection. This
   is the concrete form of the known "tolerates rapid reconnects poorly" note.

## A side result: the advertised-memory question is settled empirically

    Available devices:
      RPC0: 192.168.10.12:50052 (7819 MiB, 7819 MiB free)

The worker advertises its **real 7819 MiB**, not the `RPC_MEM_MB=2600` configured in
`Pi4/config/node.env`. `--mem` was removed upstream in `41386cf` (2025-10-17) and the
cluster pin is after that, so the deliberate-headroom mechanism is inert on this
revision. Inferred from source during the pin review; here it is confirmed in the
experiment itself.

---

# Jetson Nano standalone, Vulkan at the cluster pin (2026-09-25)

The third node, measured alone on the same model bytes and the same software revision as the
Pis. This is **Path 1**: the Vulkan backend, which builds at the pin with **no patches**.
CUDA (Path 2) needs a 34-file patch and is deferred until the board is on a proper supply.

*Verified 2026-09-26: three adversarial lenses (numbers, facts, honesty), a skeptic triage (23 of
33 issues confirmed), then every changed figure re-derived from the raw files. Record:
`.handoff/assets/vulkan-verify-2026-09-26/`.*

| | |
|---|---|
| Board | Jetson Nano B01 carrier, 4 GB module (3963 MiB, **shared by CPU and GPU**), L4T R32.7.6 / JetPack 4.6.6 |
| `llama.cpp` | `b23701f77d47dad9de834d59ebfcbe25c9e8b46f` — the cluster pin, **0 patched files**, tree clean |
| Build | `-DGGML_VULKAN=ON -DGGML_RPC=ON -DGGML_NATIVE=ON`, gcc/g++ 11.4, glslc (shaderc v2026.4), Vulkan-Headers v1.4.363, loader `libvulkan.so.1.2.141` |
| GPU as llama.cpp sees it | `ggml_vulkan: 0 = NVIDIA Tegra X1 (nvgpu) (NVIDIA) \| uma: 1 \| fp16: 1 \| bf16: 0 \| fp4: 0 \| warp size: 32 \| shared memory: 49152 \| int dot: 0 \| matrix cores: none` (verbatim, `run.log`) |
| Model | TinyLlama-1.1B Q4_K_M, full SHA-256 `9fecc3b3…d2776a0` — **byte-identical to the Pis' copies** |
| Power mode | MAXN (`pmode:0000`) throughout, **0 guard trips**; `jetson_clocks` locked for the run (CPU 1479 MHz, GPU 921.6 MHz, EMC 1600 MHz) and restored after |
| Protocol | swap off, `-t 4`, `-p 512 -n 128`, **`-r 5`**, every repetition kept; 30 s cooldown before each case |
| Raw data | `testbed/results/jetson_tinyllama_vulkan_20260925_115834/` |

`fp16: 1` matters: the proposal calls this node **the FP16 node**, and under Vulkan that depends on
the driver exposing `shaderFloat16`. It does, and llama.cpp's device probe reports `fp16: 1`: its
fp16 path is **enabled** on this device. Which kernels actually ran in FP16, and how much FP16
contributes to these figures, was **not measured** (no `GGML_VK_DISABLE_F16` A/B).

## Results

Mean over 5 repetitions, with the min–max across those repetitions, all computed at full
precision from the jsonl `samples_ns`.

| case | pp512 (prefill) t/s | tg128 (decode) t/s | start temp |
|---|---:|---:|---|
| **GPU — Vulkan, `-ngl 99`** | **73.02** (72.85–73.09) | **11.46** (11.45–11.47) | CPU 39 °C (39.5 °C at tegrastats' 0.5 °C resolution) — inside the <40 °C gate by ≤0.5 °C |
| CPU only — `-ngl 0 -dev none`, 4× A57 | 8.43 (8.30–8.64) | 6.33 (6.28–6.42) | **CPU 46 °C — started warm, see caveat 1** |

Repetition range, (max−min)/mean, **within one `llama-bench` invocation per case**: GPU 0.33% (pp) /
0.12% (tg); CPU 3.95% / 2.26%. This is **not** an across-run spread — no repeat run was taken, and
on the Pi the same configuration minutes apart drifted ~2%.

An earlier draft of this table printed the two pp512 minima as 72.86 and 8.31, by re-rounding values
already rounded to 3 dp or to 6 significant figures; both are corrected above. The summariser
(`summarise_jetson_bench.py`) and the separate ground-truth script
(`.handoff/assets/fixwf-2026-09-25/groundtruth.py`) agree with the raw data at the precision they
print. The summariser itself is still under adversarial review; these figures do not depend on it.

## GPU against the Jetson's own CPU

| Jetson GPU vs its own CPU | prefill | decode |
|---|---:|---:|
| speed-up from the GPU | **8.66x** | **1.81x** |

**Both ratios are provisional.** Their denominator is the CPU-only row, which started warm
(caveat 1) and ran at `-t 4`, while decode peaked at 3 threads on both Pis; the Jetson CPU's own
thread optimum was never measured.

**The GPU speeds up prefill 8.66x but decode only 1.81x.** That asymmetry is real, but it is **not**
a third confirmation of the Pis' bandwidth-bound account of decode. A 1.81x decode gain on the same
shared DRAM means the CPU's decode was not held at a DRAM-bandwidth ceiling. During decode,
tegrastats read EMC utilisation ~27% (GPU) and ~15% (CPU) at 1600 MHz — consistent with streaming
the 667 MB of weights once per token, 7.6 and 4.2 GB/s against LPDDR4's 25.6 GB/s peak — and the
GPU (GR3D) ~98% busy. What limits decode on either unit was not measured: GR3D busy is occupancy,
not proof of a bottleneck, and the CPU's ~99% load includes llama.cpp's thread polling (`poll: 50`).
The prefill ratio is also confounded: the A57 lacks `asimddp`, which was worth 2.96x on Pi 5
prefill, so part of the 8.66x reflects a CPU baseline without dot-product instructions.

## Against the Pis

| | prefill | decode |
|---|---:|---:|
| Jetson GPU / Pi 5 | **0.90–0.91x** | **0.63–0.69x** |
| Jetson GPU / Pi 4 | 5.85x | 2.08x |

The Pi 5 figures have two measurements at 4 threads: 80.33 / 16.71 (the single-node table) and
81.10 / 18.16 (the "Pi 5 alone" row). The ratios are given as a range across both, not picked from
one. **Treat cross-device ratios as indicative, not exact** — these conditions differ:

- **Clock policy.** The Pi runs' governor was not logged (`ondemand` or `performance`). A governor
  A/B was run on the **Pi 5 only** (decode −2.3%, prefill unchanged); the Pi 4's governor and the
  Jetson's `jetson_clocks` lock (CPU, GPU and EMC pinned at maximum, cpuidle and GPU railgating off)
  were never A/B-tested.
- **Repetitions** `-r 3` (Pis) against `-r 5`; **dates** 2026-09-19 against 2026-09-25; **host
  compiler** gcc 14.2 against 11.4; **supply** — the Jetson on micro-USB.

**Placement reading** — under Vulkan at `b23701f`, clocks locked, on TinyLlama Q4_K_M: as a prefill
node the Jetson GPU runs at 0.90–0.91x a Pi 5; as a decode node at 0.63–0.69x. It ranks between the
Pi 5 and the Pi 4 in **both** phases, but its gap to the Pi 5 depends on the phase, so a rule that
sizes work by one speed score would mis-size its share — as it would the Pi 4's (6.44x behind the
Pi 5 on prefill but 3.04x on decode: the `asimddp` result). Whether CUDA, unlocked clocks or a larger
model change this is unmeasured. This is a per-phase characterisation. **Whether the Jetson helps in
a *pooled* run is still unmeasured.** The RPC smoke test (2026-09-27, the last section of this file)
shows that the pooled run *works* and what the network costs, but it has no Pi 5-alone cell, so it
cannot answer this.

## Power — and why there is no energy-per-token figure

Module input rail — `POM_5V_IN`, INA3221 channel 0 — from root `tegrastats`. It **excludes carrier-board loads** such as
USB, the M.2 Wi-Fi card and the fan: the INA3221 is on the module, per NVIDIA's Jetson Linux R32.7.5 guide (power_management_nano): "The original Jetson Nano and Jetson TX1 modules have a three-channel INA3221 power monitor at I2C address 0x40"; channel 0, `VDD_IN`: "Main module power input".

| case | locked-clock idle floor | run mean | run peak |
|---|---:|---:|---:|
| GPU — Vulkan | 2178 mW | **7167 mW** | 9331 mW |
| CPU only | 2283 mW | **6229 mW** | 6791 mW |

- **The two run means are not comparable with each other**, and neither is a device's inference
  draw. Each mixes phases in this protocol's proportions, and the draw differs by phase: splitting
  each trace at the decode test's start time, the GPU drew ≈8.2 W in decode against ≈5.9 W before it
  (load, warm-up and prefill), the CPU ≈6.6 W against ≈6.1 W. By time, the GPU window is 54% decode
  and the CPU window 22%.
- **The idle floor is taken with clocks already locked** (cpuidle off, GPU railgating off, CPU, GPU
  and EMC clocks pinned at maximum; plain `jetson_clocks` leaves the fan on automatic control). It is
  *not* the idle figure in `node-jetson.md` (~1.16 W `POM_5V_IN`, MAXN, no `jetson_clocks`), which
  was taken with the desktop running and Wi-Fi and Ethernet up, by a different sampling method — the
  two are not a controlled `jetson_clocks` on/off pair. Call it the locked-clock idle floor.
- **Sampling rate:** the run trace asked for 200 ms (5 Hz) and **achieved 1.94–1.95 Hz**. Quote the
  achieved rate, never "200 ms".
- **No J/token or tokens-per-joule is computed, deliberately.** The power window spans the whole
  `llama-bench` invocation, which includes model load, warm-up and five repetitions of both tests.
  It cannot be attributed to the generated tokens. A per-token energy figure needs the harness's
  event-aligned INA226 measurement, which is not built yet (QR5–QR7).

## Caveats that must travel with these numbers

1. **The CPU-only row started warm: 46 °C against the protocol's <40 °C**, 6 °C above the gate.
   Only the first case gets the temperature gate; later cases get a fixed 30 s pause. Its repetition
   range is ~12x the GPU row's on prefill and ~20x on decode — but heat does not explain it: the
   clocks stayed at 1479 MHz throughout, the temperature rose steadily (51 → 63 °C) during the case,
   and the last, hottest repetition was the fastest in both tests. The source of the spread is
   unidentified, and the root power guard and tegrastats shared the four saturated cores during this
   case. **Treat the CPU row, and the 8.66x / 1.81x built on it, as provisional**, and re-take it from
   cold — with the monitors pinned or at low priority, or their cost measured — before it is quoted.
   The GPU row passed the gate, by ≤0.5 °C.
2. **Supply margin was thin, and the run broke the node record's load rule.** The board runs on
   **micro-USB**, not the planned J25 barrel supply, and `node-jetson.md` restricts it to idle and
   light load until the barrel supply is fitted. This sustained MAXN run, and the build before it,
   deviated from that, with the power guard as the mitigation. The lowest of the guard's ~1 Hz
   samples in the GPU case was **4808 mV**, taken at the guard's own peak of 9.19 W. tegrastats read
   the same INA3221 channel at ~1.94 Hz and peaked higher, at 9.33 W, at an instant the guard did not
   sample, so the true minimum was at or below 4808 mV. That is 58 mV above the **4750 mV**
   threshold; the guard trips only after 3 consecutive samples below it (or on a new over-current /
   under-voltage alarm). Nothing tripped, and the power state was unchanged start to end. This is why
   sustained CUDA load waits for the barrel supply and the J48 jumper.
3. **Power was read on the module's own INA3221 through tegrastats** (channel 0, the module input,
   excluding carrier loads), not on the project's planned INA226 chain, which is not yet built or
   calibrated. These readings are uncalibrated against the bench multimeter.
4. **The CPU-only case is labelled "Vulkan" by llama-bench's `backends` field**, which lists the
   non-CPU backends registered in the build (RPC only when RPC servers are given), not the device a
   case ran on. What it actually ran on is `devices: none`, `n_gpu_layers: 0`. The summariser shows
   both; anyone reading the raw jsonl should use `devices`.
5. **The summariser is frozen at `44456e39`, with four known limits.** They were confirmed by its last
   review round (2026-09-27) and accepted rather than fixed, because none of them changes this run's
   figures:
   - **It cannot yet be trusted on a CUDA run.** A `MODE=cuda` run made with the default `BIN` (the
     Vulkan build) is reported as `case cuda: OK` with `pins: verified`. A genuine patched CUDA build is
     flagged BUILD NOT VERIFIED instead. Until this is fixed, pass `BIN` and `BUILDINFO` for
     `MODE=cuda`, and check that the records' `backends` names CUDA.
   - A case that FAILED before its power window closed still prints an unmarked idle floor, even if the
     guard tripped during the idle capture. Do not quote a FAILED case's idle floor.
   - Record fields are not type-checked. A malformed record usually stops the report with a traceback
     (rc 1), but a `samples_ts` of booleans prints a coerced 1.000 t/s as OK. An empty `run.log`
     prints an empty table and exits with rc 0.

## Reproducing it

```sh
# The standalone bench has NO temperature gate: wait for CPU < 40 °C first (or launch through
# jetson_chain_build_bench.sh, whose own wait gives up after 10 minutes and runs anyway).
# The power guard must already be running.
ssh jetson 'while [ "$(cat /sys/devices/virtual/thermal/thermal_zone1/temp)" -ge 40000 ]; do sleep 10; done
            MODE=vulkan setsid nohup bash ~/heteroserve/tools/jetson_bench_tinyllama.sh </dev/null >/dev/null 2>&1 &'
# Collect -- guarded: an empty LATEST would copy the Jetson's whole home directory
R=$(ssh jetson 'cat ~/heteroserve/results/LATEST'); [ -n "$R" ] || exit 1
D=testbed/results/$(basename "$R"); scp -r "jetson:$R" testbed/results/
scp jetson:heteroserve/build_info.json jetson:heteroserve/logs/{llama_configure.log,llama_build.log,chain.log,glslc_version.txt,power-guard.log} "$D/"
python3 testbed/scripts/summarise_jetson_bench.py "$D" "$D/power-guard.log"
```

This run's `summary.txt` was regenerated on 2026-09-27 with the frozen summariser (`44456e39`), using
exactly that command. The one it replaced is kept beside it as `summary.pre-44456e39.txt`.

**Check the `EXIT_<case>=` lines in `run.log` yourself.** A confirmed defect makes `bench.done` and
the chain's `DONE` status report success even if every case crashed. For this run both lines read
`EXIT_vulkan=0` and `EXIT_cpu=0`.

# Pi 5 + Jetson pooled over RPC: functional smoke with a matched local baseline (2026-09-27)

The first run in which the Jetson joins a pool. The Pi 5 is the main node (CPU), and the Jetson is a
`ggml-rpc-server` worker exposing its GPU through Vulkan. Both are at the cluster pin. This is a
**functional smoke test, not a characterisation**, and its settings are deliberately small. Around
the two RPC cells, the Jetson runs the same settings **alone** (local1 before, local2 after), so the
cost of the network can be judged against the drift of the same configuration across the run.

| | |
|---|---|
| Nodes | Pi 5 (main, CPU, 4 threads) + Jetson Nano (RPC worker, `-d Vulkan0`), over the wired overlay (Jetson 192.168.10.13 for the run) |
| `llama.cpp` | `b23701f` on both boards: trees clean at the pin; the Pi 5's `llama-bench --version` reads commit `b23701f`; the Jetson's binaries are no newer than its `build_info.json` (commit = pin, backend vulkan); every jsonl row reads `build_commit b23701f` |
| Model | TinyLlama-1.1B Q4_K_M, full SHA-256 `9fecc3b3…d2776a0`, matched on **both** boards against `models.md` |
| Settings, every cell | `-t 4 -p 64 -n 16 -r 3 -oe jsonl`; RPC cells `-ngl 99` (every layer on the Jetson) and `-ngl 11` (half the model's 22 layers on the Jetson); local cells `-dev Vulkan0 -ngl 99` on the Jetson |
| Jetson | MAXN, `jetson_clocks` **locked for the whole run** (GPU 921.6 MHz, CPU 1479 MHz, checked before and after every cell), restored at the end; **0 guard trips**; on micro-USB |
| Script | `testbed/scripts/rpc_smoke_pi5_jetson.sh` at `a5f5b02a`, **frozen without an independent review** (caveat 5) |
| Raw data | `testbed/results/rpc_smoke_pi5_jetson_20260927_174752.log`; summary `…_174752.summary.txt` (from `summarise_rpc_smoke.py`) |

## Results

Mean of 3 repetitions, with their min–max, computed from the jsonl `samples_ts` (3 dp; a one-line
check with no shared code agrees). The repetitions of a cell are **one `llama-bench` invocation**;
the local1 → local2 bracket is the only across-invocation evidence.

| cell | where the layers ran | pp64 t/s | tg16 t/s |
|---|---|---:|---:|
| local1 | Jetson alone, Vulkan0 | 63.349 (63.330–63.383) | 11.530 (11.524–11.533) |
| ngl=99 | all 22 on the Jetson GPU, driven from the Pi 5 over RPC | 62.549 (62.538–62.572) | 11.241 (11.239–11.243) |
| ngl=11 | 11 on the Jetson GPU, the rest on the Pi 5 CPU | 71.610 (69.631–72.631) | 13.639 (13.581–13.686) |
| local2 | Jetson alone, Vulkan0 | 63.306 (63.296–63.319) | 11.525 (11.523–11.526) |

| against the mean of local1 and local2 | pp64 | tg16 |
|---|---:|---:|
| drift across the run: local2 vs local1 | −0.07% | −0.04% |
| **ngl=99: the cost of the network** | **−1.23%** | **−2.49%** |
| ngl=11 | +13.08% | +18.32% |

- **The pooled run works**, at both splits, with every cell `ok`.
- **The network cost is real, and small at this size.** With every layer on the Jetson, driving it
  from the Pi 5 over RPC cost 1.2% of prefill and 2.5% of decode. That is about 18x (pp) and 63x (tg)
  the run's own drift (computed from the unrounded means), and the ranges do not overlap. Decode pays about twice what prefill pays,
  consistent with decode's per-token round trips. Other prompt lengths and models are unmeasured.
- **The 11/11 split is faster than the Jetson alone.** Moving half the layers onto the Pi 5's CPU
  raised prefill 13% and decode 18%. That is consistent with the standalone ranking, where the Jetson
  GPU runs at 0.90–0.91x a Pi 5 on prefill and 0.63–0.69x on decode. It is **not** evidence that the
  Jetson helps the Pi 5: see caveat 2.

## What the run's log proves, cell by cell

- **The RPC traffic went over the wire.** The route runs Pi 5 eth0 192.168.10.11 → 192.168.10.13, and
  the Pi 5 resolves 192.168.10.13 to the Jetson's **eth0 MAC** `00:04:4b:ec:52:f2`, checked before
  every RPC cell. The Jetson's **wlan0 moved 0 B** in every cell. Both ends read **1000 Mb/s full
  duplex** before and after each RPC cell. ARP-flux settings were applied for the run and restored to
  the saved values.
- **The weights are shipped at every load.** The worker has no local cache (`local cache: n/a`). The
  Jetson's eth0 received **665.5 MB for ngl=99** (99.8% of the 667,078,656-byte model file) and
  **336.2 MB for ngl=11** (50.4%). Load time is outside llama-bench's t/s figures.
- **The local cells ran on the GPU**: their rows read `devices Vulkan0`, `gpu_info NVIDIA Tegra X1
  (nvgpu)` and `backends Vulkan`.
- **Thermal and power state:**
  - Pi 5 `throttled=0x0` before and after each RPC cell, with bits 0–3 clear and no sticky bits.
  - Pi 5 at 35.1 → 37.3 °C; Jetson CPU at 27.5 → 31.5 °C.
  - Guard live before every load, trips 0 → 0, MAXN throughout.

## Caveats that must travel with these numbers

1. **Small settings.** These are `-p 64 -n 16` at `-r 3`, not the standalone tables' `pp512` / `tg128`,
   so the figures are not comparable with those tables. The t/s values of the 64-token and 16-token
   tests are not the same quantities as the 512 / 128 ones.
2. **This run cannot say whether the Jetson helps the Pi 5.** It has no Pi 5-alone cell at these
   settings; there was deliberately only a Jetson baseline (user decision, 2026-09-26). The standalone
   ranking above predicts that the Pi 5 alone would outrun the 11/11 split on this model, which fits
   on the Pi 5 by itself. The pooling case that matters is a model that fits on no single node (QR1),
   which this run does not test.
3. **ngl=11's prefill was the one noisy cell.** Its first repetition (69.63) was slower than the other
   two (72.63, 72.57): its range is 4.19% of the mean, against at most 0.08% for every other pp64
   cell. Its decode range is 0.77%. The cause is unidentified; it is reported as found.
4. **Supply margin.** The Jetson is on micro-USB. The guard's lowest input sample was **4888 mV**, and
   its peak draw **8055 mW**, both reached during local1 (the first load after the clock lock), and
   neither was exceeded later. That is 138 mV above the guard's 4750 mV threshold, and nothing tripped.
5. **The script is frozen at `a5f5b02a` without an independent review** (user decision, 2026-09-27).
   Its last repair (8 items: the bash 5.2.21 trap misparse, the matched local baseline and the locked
   clocks among them) was verified only by that repair's own 218-case mocked suite and signal sweeps
   (SIGINT 329/329, SIGTERM 55/55, SIGHUP 55/55, SIGQUIT 109/110). The record is in
   `.handoff/2026-09-26-2203-r13b-rpc-repair.md` (Outcome). Everything claimed above is read from the
   run's own log, not from the script's word.
6. **No energy figure**, for the reasons given in the standalone section.

## Reproducing it

```sh
# From the repo root on the laptop. The Jetson's power guard must be running and the board in MAXN:
# the script refuses otherwise (exit 3), undoes everything it changed, and exits 5 if it cannot.
bash testbed/scripts/rpc_smoke_pi5_jetson.sh
python3 testbed/scripts/summarise_rpc_smoke.py testbed/results/rpc_smoke_pi5_jetson_<stamp>.log
```

---

# 2026-09-27 — the characterisation dataset, and the placement engine checked against the 14B pool

## The dataset

TinyLlama-1.1B at Q4_0 / Q4_K_M / Q8_0 on the Pi 5 and Pi 4 CPUs and the Jetson's CPU and Vulkan GPU:
26 rows, 24 clean and 2 flagged (the 2026-09-25 Jetson CPU rows, which started warm and are superseded by a
cold re-take). Every Pi run read `get_throttled 0x0`, and the Jetson guard tripped 0 times. The table,
the protocol and the findings are in **`testbed/dataset/README.md`**. The finding that shapes the engine is
that **a board's cost per byte (decode) or per parameter (prefill) transfers within a quantisation, but
not always across one**. The Pi 5's prefill spans 2.1× across the three quants because at this pin every
one of them gets a different dotprod-repacked kernel, and the Jetson CPU's decode spans 1.4×. The engine
therefore takes each board's cost from its row at the **target model's** quantisation.

## The link, re-measured

`testbed/scripts/measure_link.sh`: raw TCP, 1 GiB × 3 each way, on the path the pool has used since
2026-09-25 (across the router's switch). **938 Mbit/s Pi 5 → Pi 4, 942 Mbit/s back** (each direction's
three transfers agree within 0.1 %), which is gigabit line rate. **RTT 0.23 ms** each way. The 2026-09-19
figures (571 Mbit/s, 2.30 ms) came from the old direct cable. Why that cable was slower was not
investigated. The new figures are in `testbed/dataset/link.json`, and the log is
`testbed/results/link_20260927_195141.log`.

## The engine, fed only single-node rows, against the measured 14B pool

`testbed/scripts/validate_placement.py`. Inputs:

- the Pi 5 and Pi 4 **TinyLlama Q4_K_M rows**;
- the 14B's **GGUF tensor table**;
- the measured link.

No pooled measurement goes in. The per-board cost is ms per byte read per decoded token (the blocks plus
the output head), and ms per block parameter per prompt token. That cost is scaled block by block through
the 14B's tensor table. The prediction adds the output head on the node that holds it (the Pi 4, see the
correction above) and the per-token transfers, which are the hidden state out and the logits back.

| blocks Pi 5 / Pi 4 (+head) | `ngl` | decode ms/token, measured | predicted | error | prefill ms/token, measured | predicted | error |
|---|---:|---:|---:|---:|---:|---:|---:|
| **43 / 5** | 6 | **1032.9** | 1041.9 | +0.9 % | 264.9 | 264.8 | −0.0 % |
| 41 / 7 | 8 | 1110.0 | 1108.4 | −0.1 % | 306.9 | 303.0 | −1.3 % |
| 37 / 11 | 12 | 1210.5 | 1230.3 | +1.6 % | 378.2 | 379.6 | +0.4 % |
| 33 / 15 | 16 | 1328.5 | 1352.2 | +1.8 % | 453.1 | 456.1 | +0.7 % |
| 29 / 19 | 20 | 1428.0 | 1477.9 | +3.5 % | 526.3 | 532.7 | +1.2 % |
| 25 / 23 | 24 | 1593.3 | 1599.8 | +0.4 % | 600.7 | 609.2 | +1.4 % |
| 19 / 29 | 30 | 1766.2 | 1784.5 | +1.0 % | 712.3 | 724.1 | +1.7 % |
| 13 / 35 | 36 | 1944.2 | 1969.3 | +1.3 % | 834.6 | 838.9 | +0.5 % |

**Decode: every split within 3.5 %, mean |error| 1.3 %; slope 30.84 predicted against 30.33 ms per block
fitted (+1.7 %). Prefill: every split within 1.7 %, mean 0.9 %.** The same prediction taken from the other
quants' rows moves decode by −4 % to +1 %. It moves prefill at the 43 / 5 split by −20 % (Q4_0) to +16 %
(Q8_0), so the same-quantisation rule is not optional.

**Memory.** Two corrections to the engine's accounting, both from the tensor table and the llama.cpp
source rather than from fitting:

1. Bytes per layer come from each block's tensors: 157.1 MiB mean, against 178.6 for file size ÷ 48.
2. The output head (609 MiB) is charged to the node that holds it, and the token embedding (418 MiB) to
   the host.

The runtime overhead stays at the default 350 MB. With each run's own MemAvailable at start:

| check | engine cap | measured | |
|---|---:|---|---|
| Pi 5 as host holds 43 blocks (`ngl=6`) | 43 | ran | OK |
| Pi 5 refuses 45 blocks (`ngl=4`, 2026-09-20) | 43 | OOM-killed, anon-rss 7,666 MiB | OK |
| Pi 4 holds 35 blocks + head (`ngl=36`) | 41 | ran | OK |

The old accounting capped the Pi 5 at **40**, which would have refused the fastest split. The OOM
record fits the new accounting: 45 blocks (7,040 MiB) plus the embedding (418 MiB) plus about 208 MiB of
runtime.

**The rule's pick, latency-only: Pi 5 43 / Pi 4 5 + head, which is exactly the measured fastest split.**
The old accounting picked 40 / 8. Against the near-even `ngl=24`, the fastest split **cuts decode
TPOT by 35.2 % measured and 34.9 % predicted.** That meets QR8's 25 % stretch.

### What changed in the engine (ThinkPad/control; suite 173 → 185 tests, all passing)

- `gguf.py` reads the **tensor-info table**. Each tensor's size is the distance between data offsets,
  so no per-quant layout is needed. It reports per-block bytes, the output head, the embedding and
  parameters, and matches numpy's `GGUFReader` to the byte on the 14B. File size ÷ layers remains only
  as a labelled fallback, used when a file has no readable table.
- `ModelProfile.head_bytes` is charged to the pipeline's **last** node. At this pin, any offload sends
  the output layer to the last device, in both the harness's launch and `run_14b_pooled.sh`'s.
- `ModelProfile.embd_bytes` is charged to the **first** node when `host_in_pipeline` is set, that is,
  when a participant runs llama.cpp itself. Otherwise no participant pays.
- The 12 new tests all fail against the old code. One is the measured 14B case: cap 43, pick 43 / 5,
  old accounting 40 / 8.

## Caveats that must travel with these numbers

1. **Prefill batch.** The sweep ran pp64 and the single-node rows pp512. Cost per prompt token depends on
   the batch, so the prefill agreement is the weaker result.
2. **The embedding's residency is inferred.** The file is mmap'd, and RPC devices report mmap support.
   That the 418 MiB embedding is resident rests on the OOM record: without it the engine would admit the
   `ngl=4` split that was killed. A memory probe during a pooled run would settle it.
3. **One model, one pair.** QR8 asks for the comparison on more models, the three-board pool and a Pi
   5-alone baseline. Those runs have not been made.
4. **Resolved (user decision, 2026-09-27): the live planner now plans against measured MemAvailable.**
   It used to read the node agents' `usable_budget_mb`, which is deliberate slack: the Pi 5 agent reports
   `RPC_MEM_MB` 6,144 and the Pi 4 agent `NODE_USABLE_MB` 2,600 (`cluster.json` then said 7,000 / 2,600).
   With those budgets the rule picked 39 / 9 (predicted 1,168 ms, 12 % slower) and called the near-even
   split infeasible, although it ran. `profiles.build_devices` now takes `cluster.json`'s MemAvailable at
   rest (7,619 / 7,413 / 3,545 MiB, with provenance), and never more than a node reports free right now.
   The engine's 350 MB overhead is the only headroom. The node folders were not edited; see the NOTEs in
   `.handoff/nodes/Pi4/` and `.handoff/nodes/Pi5/`.
5. **The head sits on the straggler in every measured split,** about 180 ms of the fastest split's 1,033.
   Hosting on the Pi 4 would move the head to the Pi 5. But a board serving as an RPC worker gets **no
   repacked kernels** (`ggml-rpc.cpp` allocates the default CPU buffer type), so the Pi 5's rows do not
   describe that role, and that plan cannot be priced until the Pi 5 is characterised as a worker.

## Reproducing it

    bash testbed/scripts/run_quant_campaign.sh          # ~42 min, all three boards
    python3 testbed/scripts/build_dataset.py testbed/results/quant_campaign_<stamp> testbed/results/jetson_tinyllama_vulkan_20260925_115834
    bash testbed/scripts/measure_link.sh && python3 testbed/scripts/link_to_json.py testbed/results/link_<stamp>.log
    python3 testbed/scripts/validate_placement.py

---

# 2026-09-27 (night) — the pre-registered pool campaign: three boards, three models

## Protocol

Every configuration was planned by the engine and its outcome predicted **before any of them ran**. The
predictions come from single-board rows only (the same-quant TinyLlama rows, host and worker roles), the tensor
tables and the measured links:

- `testbed/dataset/pool_predictions.json`, registered 20:55:56, sha256 `f997e90c…`;
- one addendum, registered 21:39:03 (`…_addendum.json`, `d0db5dc2…`), before its cell ran.

`testbed/scripts/run_pool.sh` ran each cell as `llama-bench -t 4 -p 64 -n 16 -r 3 -oe jsonl -v` on the host. Around
every cell it gated, and after every cell it stopped the workers and restored the Jetson's clocks:

- **gates:** every board idle and at the pin; the Jetson in MAXN with its guard live, clocks locked and eth0 its only
  interface on the router LAN; a live MemAvailable check against the pre-registered need;
- **probes:** a 1 Hz memory probe on every board, and `get_throttled` before and after.

Every cell exited 0 and every Pi read `0x0` after every cell. The Jetson's guard tripped 0 times.

- **Two cells were refused at first,** both by the live memory gate: right after the previous 14B cell the Pi 5 had
  7,432 MiB free against a need of 7,475. The gate was then changed to wait (up to 10 min) for MemAvailable to
  recover, and the two were re-run at 21:37–21:50.
- **Every split landed on the plan.** llama.cpp's own per-device model buffers (`-v`) match the planned bytes within
  0.1 MiB on every worker. The Pi 5 host's `CPU_REPACK` is within 3 MiB; the difference is the F32 norms, which stay
  mapped.
- **The wire is proven** by the Jetson's eth0 counters, which bracket every cell it was in.

## Results (measured tonight / pre-registered prediction)

| model | cell | pipeline (blocks; + head on the last) | decode ms/token | prefill ms/token |
|---|---|---|---|---|
| Qwen2.5-14B Q4_K_M | **rule** | **Pi 5 43 + Jetson 5** | **855.9** / 836.9 (−2.2 %) | **188.2** / 170.5 |
| | rule, Pi 4 forced in | Pi 5 43 + Pi 4 1 + Jetson 4 | 893.0 / 861.9 | 209.0 / 189.3 |
| | Pi 4 hosts | Pi 4 6 + Pi 5 42 (worker) | 1027.0 / 924.3 | 398.6 / 394.2 |
| | the 09-25 best | Pi 5 43 + Pi 4 5 | 1102.5 / 1045.5 | 276.5 / 263.8 |
| | even | Pi 5 16 + Jetson 16 + Pi 4 16 | 1540.4 / 1535.2 (−0.3 %) | 496.2 / 479.6 |
| | Pi 5 alone | — | *infeasible, not run* | |
| Mistral-7B Q8_0 | **rule** | **Pi 5 31 + Jetson 1** | **719.0** / 643.6 | **159.8** / 111.5 |
| | Pi 5 alone (addendum) | Pi 5 32 | 724.5 / 620.7 | 167.0 / 111.9 |
| | even | Pi 5 11 + Pi 4 11 + Jetson 10 | 1311.9 / 1245.5 | 312.7 / 268.7 |
| TinyLlama Q4_K_M | **rule** = Pi 5 alone | Pi 5 22 | **60.8** / 55.2 | **11.9** / 12.3 |
| | even | Pi 5 8 + Pi 4 7 + Jetson 7 | 109.0 / 106.3 | 36.9 / 34.3 |

Each figure is the mean of 3 repetitions. The raw logs are `testbed/results/pool_<cell>_<stamp>.log`, and the scored
set is `testbed/dataset/pool_results.json`, from `compare_pool.py`. The same 14B cell run twice tonight, 43/5 at
20:58 and 21:50, read 1112.6 and 1102.5 ms: 0.9 % apart.

## What it answers

**QR8, the rule against an even split, on three models.** The rule's pick cuts decode TPOT by **44.2 %**
(TinyLlama), **45.2 %** (7B) and **44.4 %** (14B), and prefill by 67.8 / 48.9 / 62.1 %. That beats the 25 % stretch
target on every model. The even split is the naive baseline, in the cluster's own board order. For the 14B it uses
the other order, because the cluster's order would leave the 4 GB Jetson 36 MiB (recorded in the registration).

**Pooling all three boards: the rule drops the straggler.** On the 14B, the rule's plan is the Pi 5 plus the Jetson,
with the Pi 4 out. It measured **1.17 tok/s decode and 5.3 tok/s prefill**, the fastest 14B configuration yet, and
**22 % faster in decode than the best Pi 5 + Pi 4 split** run in the same session (1102.5 ms).

**The straggler study, all measured on the 14B:**
- **Forcing the Pi 4 in** with a single block costs +4.3 % in decode.
- **Making it carry its minimum** without the Jetson (5 blocks + the head) costs +29 %.
- **Making it the host** moves the head onto the Pi 5, but the Pi 5 then works as an RPC worker without its repacked
  kernels. Against the Pi 5-hosted 43/5 split, decode is **6.8 % faster** and prefill **44 % slower**. That is the
  per-phase trade-off the worker rows predicted, and the case for splitting prefill from decode (QR4, Project 2).
- The engine **predicted the rank order of all five 14B configurations exactly, in both phases.**

**The Pi 5-alone baseline: does the Jetson help the Pi 5?** It depends on the model:
- **TinyLlama:** no. The Pi 5 alone is the rule's pick. The RPC smoke test's Pi 5 + Jetson split was 17 % slower in
  decode than the Pi 5 alone tonight, at the same pp64 / tg16.
- **The 7B:** only marginally. It fits the Pi 5 alone (the addendum cell). With the Jetson taking one block and the
  head, the rule's pick measured 0.8 % faster in decode and 4.3 % faster in prefill, and left 186 MiB more headroom.
  The engine had predicted the Pi 5 alone 3.6 % faster. It **ranked these two wrongly**: a difference this small is
  below what it can resolve.
- **The 14B:** it cannot run on the Pi 5 alone. The Jetson makes it run, and faster than the Pi 4 does.

## How good the predictions were

- **Decode:** the error ranged from −0.3 % to −14.3 %, mean |error| 6.3 %, and **every error was optimistic**.
- **Prefill:** within 1–10 %, except the Q8_0 7B cells, which were **30–33 % optimistic**. Those use the Pi 5's
  repacked Q8_0 prefill kernel, whose pp512 row does not carry over to pp64.
- **TinyLlama decode:** 9 % optimistic, from the tg16 protocol's fixed per-run overhead that a 55 ms token cannot hide.
- **A session effect:** the same 14B split measured 1033 ms on 09-25 (predicted within 1.2 %) and 1102–1113 ms tonight.
  The boards had been up 6 minutes then and 29 hours now. It is not transparent huge pages: the Pi 5's kernel (16 KiB
  pages) has none. The cause is not established.

## Memory, measured instead of inferred (the 1 Hz probe)

- **The mmap'd token embedding is not resident.** At steady state, about 30 MiB of the file stays mapped. The host's
  steady footprint is its repacked blocks plus KV, compute and about 60 MiB.
- **The boundary is set by a load-time transient.** `ggml-rpc.cpp`'s `serialize_set_tensor` copies each tensor it
  uploads whole into a fresh buffer, so the host peaks at its steady footprint **plus the largest tensor it sends**:
  - the 14B host: **+557 MiB** (its 609 MiB output head);
  - the Pi 4 as the 14B's host: +592 MiB;
  - the 7B host: +207 MiB (a 133 MiB head);
  - a worker: about +200 MiB.

  This is why the 09-20 `ngl=4` run was killed *mid-load*: 45 blocks fit in steady state, but not at the peak.
- **The engine's accounting is right on the 14B for a different reason.** Blocks + embedding + 350 MB is a stand-in:
  on the 14B it equals the measured MemAvailable drop to the MiB (7,475). On the 7B it is about 290 MiB conservative:
  it refused the 7B on the Pi 5 alone by 70 MiB. Under the refined accounting registered in the addendum (host = blocks
  + a local head + ~160 MiB + the largest uploaded tensor) that cell fits. **It ran: minimum free 233 MiB, no OOM
  record.** The refinement is proposed, not adopted. Its overhead constant was estimated from tonight's own probe data,
  and one confirmed cell is not a validation.

## Memory accounting, validated out of sample (2026-09-28)

The refined accounting proposed on 09-27 rested on one confirmed cell, and its constant was fitted on that same night's
data. Re-reading that night's 1 Hz probes board by board (the `MEMPROBE` lines of every `pool_*_20260927_*.log`) showed
that the right accounting depends on the board and its role, not only on the model:

- **The Pi 5 as host repacks** its weights into anonymous `CPU_REPACK` buffers. It pays its blocks, about 150–220 MiB,
  and the largest tensor it uploads. That is the refined accounting's shape.
- **The Pi 4 as host does not repack.** Every tensor stays `CPU_Mapped`, which is file-backed, so its local weights never
  lower MemAvailable. It pays only about 137 MiB plus the largest tensor it uploads. In the 14B `pi4_host` cell it
  dropped 746 MiB while holding 980 MiB of blocks.
- **CPU workers** (Pi 4, Pi 5) pay their received blocks and the head, plus about 30–190 MiB.
- **The Jetson as a worker (`Vulkan0`) pays the largest tensor it receives a second time.** Holding the 14B's 609 MiB
  head, it paid +643 MiB over its blocks and head. Both registered accountings under-predicted that, which is the
  unsafe direction.

A board-aware accounting ("v2") was fitted to those in-sample figures. Then four NEW cells were registered, with every
node's predicted MemAvailable drop under all three accountings, before any of them ran:

- `testbed/dataset/memval_predictions.json`: registered 2026-09-28 11:00:43, sha256 `2d4b93e2…`. It includes the
  scoring and decision rules.
- The builder is `.handoff/assets/memval-2026-09-28/register_memval.py`. The scorer is `score_memval.py`, which writes
  `testbed/dataset/memval_results.json`.

The cells ran through `run_pool.sh` exactly as on 09-27: `-t 4 -p 64 -n 16 -r 3 -oe jsonl -v`, with the 1 Hz probe on
every board.
- **Before the runs**, the Pi 4's copies of the 14B (all three parts) and of TinyLlama Q8_0 were checked with
  `sha256sum -c` against the `models.md` hashes. All four were OK.
- **Every cell** exited 0, and every Pi read `0x0` before and after.
- **The Jetson's** guard tripped 0 times. The runner locked its clocks and restored them afterwards, and its zram swap
  stayed at 0 MiB used.

| cell | node | measured drop (MiB) | standin | refined | v2 |
|---|---|---|---|---|---|
| TinyLlama Q8_0, Pi 4 alone | Pi 4 host | 31 | +1,434 | +1,178 | +106 |
| 7B Q8_0, Pi 5 20 + Pi 4 12 (head) | Pi 5 host | 4,713 | +190 | 0 | +58 |
| | Pi 4 worker | 2,859 | +276 | +126 | +85 |
| 14B Q4_K_M, Pi 4 hosts 20 + Pi 5 28 (head) | Pi 4 host | 686 | +3,223 | +3,224 | +60 |
| | Pi 5 worker | 5,088 | +269 | +119 | +110 |
| 14B Q4_K_M, Pi 5 40 + Jetson 8 (head) | Pi 5 host | 6,982 | +12 | +14 | +22 |
| | Jetson worker | 2,541 | **−269** | **−419** | +19 |

Each error is the prediction minus the measured drop. A negative error means the accounting promised memory the board
did not have. The measured drop is the node's `memavail_mib` in the cell's own `STATE before` line minus the probe's
minimum.

**Decision, by the registered rule: v2.** Its worst error is 110 MiB, and it is never optimistic: its errors run from
+19 to +110, with a mean of +66. The stand-in and the refined accounting both fail the rule. Both are optimistic on the
Jetson (by 269 and 419 MiB), and both overstate a Pi 4 host's MemAvailable drop by more than 3 GB. That last one is
about the drop, not the planning limit: see the correction below.

**What it changes:**
- **The stand-in accounting was unsafe when the Jetson holds the head.** In last night's rule cell (Pi 5 43 +
  Jetson 5) the Jetson had about 1.5 GB spare, so it did not matter. But the stand-in would let the Jetson hold 16
  blocks plus the 14B head, which by this measurement needs about 3,780 MiB of its 3,545. The engine now charges the
  received head again and caps it at 14 (see below).
- **A Pi 4 host's weights cost no MemAvailable, but they still need the RAM.** *Corrected the same day:* this
  section first said that made the engine's Pi 4 host limit "gigabytes too strict". It does not. The weights are
  mmap'd page cache, so they never show up as a MemAvailable drop, and they cannot cause an OOM. But decode reads
  every one of them on every token, so they must stay resident. If they did not fit, the Pi 4 would re-read them from
  the SD card on every token: a speed failure, not an OOM, and untested. So the engine keeps charging a Pi 4 host for
  its blocks. What changes is only its overhead: 137 MiB plus the largest upload, instead of 350 MB plus the embedding.

**In the engine (2026-09-28).** `ThinkPad/control/placement.py`'s `memory_cap` is now role-aware wherever
`cluster.json` measures a board's overheads (`memory_overhead_mb`, `worker_repays_received_tensor`). Each figure is the
largest fitted value for that board and role, so the engine is never less conservative than the validated v2:
- Pi 5: host 220, worker 191;
- Pi 4: host 137, worker 159;
- Jetson: worker 44, and it repays the largest received tensor.

A host pays its largest upload but not the mmap'd embedding. A board or role that is not measured keeps the stand-in,
so older plans and registrations reproduce.
- `gguf.py` now reports each model's `max_block_tensor_bytes`. `model_shapes.json` carries it; the file it replaced is
  kept as `model_shapes.20260927.json` (`e038be4a…`), the hash the earlier registrations quote.
- `testbed/scripts/predict_pool.py` now uses the same function, `placement.memory_need_mb`, for the `need_mib` that
  `run_pool.sh`'s live memory gate enforces.
- **Checks:**
  - suites ThinkPad 202 (191 + 11 new), Pi 5 357, Pi 4 158, all OK;
  - 6 of 6 deliberate breaks of the new terms were caught by the new tests;
  - `validate_placement.py` re-run on a copy reproduces `placement_validation.json` exactly;
  - the 14B boundary stays at 43 blocks.
- **Consequences for the 09-27 campaign's plans, if re-planned** (not re-registered):
  - the 14B rule's pick is unchanged (Pi 5 43 + Jetson 5), at a thinner registered margin: 69 MiB instead of 144,
    against 134 measured;
  - the 7B's pick becomes the Pi 5 alone, which the stand-in had refused. It measured within 0.8 % of the old pick.

**Caveats:**
- v2's constants were fitted on the 09-27 cells. Seven out-of-sample node measurements from four cells confirm it, all
  from one session on one day.
- Its fixed bases are conservative for small models: +106 MiB on TinyLlama, whose compute buffer is 2.2 MiB.
- The drops are the minima of 1 Hz samples, so a sub-second peak could be missed.
- The Jetson ran with zram swap enabled but unused. A Jetson without swap would be OOM-killed rather than swap, at the
  same boundary.

**Throughput, incidental.** These cells were registered for memory, not speed. Each is one `-r 3` invocation, so these
are means, not across-run spreads (pp64 / tg16, t/s):
- TinyLlama Q8_0, the Pi 4 alone: 12.42 / 3.11;
- 7B, Pi 5 20 + Pi 4 12: 3.05 / 0.81;
- 14B, the Pi 4 hosting 20: 1.57 / 0.74;
- 14B, Pi 5 40 + Jetson 8: 5.30 / 1.13. That is slightly slower in decode than last night's 43 + 5 split (1.17).

---

# 2026-09-28 — the second pre-registered pool campaign: Mistral-Nemo 12B, Qwen3-14B, Qwen3-30B-A3B

## Protocol

The same as 09-27. Every configuration was planned and its outcome predicted **before any of them ran**:
`testbed/dataset/pool_predictions_20260928.json`, registered 16:13:30, sha256 `3ddb0bbe…`, and read-only (the 09-27
registration and its addendum are now read-only too). The inputs:
- **the rows:** the TinyLlama Q3_K_M rows measured this afternoon (the MoE's quantisation);
- **the tensor tables:** the four new models' tables in `model_shapes.json`;
- **the links:** `link.json` with the new Pi 4 ↔ Jetson pair;
- **memory at rest:** from 09-27, checked against the boards before registering (all within 30 MiB);
- **the accounting:** the board-aware memory accounting in `cluster.json`.

The engine was `placement.py 8c5735a6…`, `gguf.py 805ddefc…`. **One change to the predictor:** each worker is now priced on
its own measured link with the host. It matters here because the MoE's only copy is on the Pi 4, so the Pi 4 hosts it.
Every 09-27 configuration reproduces identically.

Eight configurations were registered. **Three were predicted infeasible and not run:**
- Qwen3-14B on the Pi 5 alone (−775 MiB);
- the MoE on the Pi 4 alone (−6,629 MiB);
- the MoE split evenly (16 blocks leave the Jetson −1,161 MiB).

The other five ran one after another, 16:14–16:51, through `run_pool.sh` with its usual gates and probes.

## Results (measured / pre-registered)

| model | cell | pipeline (blocks; + head on the last) | decode ms/token | prefill ms/token |
|---|---|---|---|---|
| Mistral-Nemo 12B Q4_K_M | **rule** = Pi 5 alone | Pi 5 40 | **676.1** / 620.7 (−8.2 %) | **154.5** / 138.6 (−10.3 %) |
| | even | Pi 5 14 + Pi 4 13 + Jetson 13 | 1171.7 / 1175.7 (+0.3 %) | 399.5 / 388.9 (−2.7 %) |
| Qwen3-14B Q4_K_M | **rule** | **Pi 5 35 + Jetson 5** | **877.3** / 847.8 (−3.4 %) | **193.9** / 170.9 (−11.9 %) |
| | even | Pi 5 14 + Jetson 13 + Pi 4 13 | 1588.4 / 1520.5 (−4.3 %) | 488.0 / 471.9 (−3.3 %) |
| | Pi 5 alone | — | *infeasible (−775 MiB), not run* | |
| Qwen3-30B-A3B Q3_K_M (MoE) | **rule** | Pi 4 12 (host) + Jetson 11 + Pi 5 25 | **failed: the Jetson's worker died while loading** / 265.8 | — / 119.7 |
| | even; Pi 4 alone | — | *infeasible, not run* | |

Each figure is the mean of 3 repetitions. An error is (predicted − measured) / measured, so a negative one is
optimistic. The scored set is `testbed/dataset/pool_results_20260928.json` (`compare_pool.py --pred … --out …`). The raw
logs are `testbed/results/pool_<cell>_20260928_*.log`.

**Evidence for the four that ran:**
- every one exited 0, and every Pi read `0x0` after every cell;
- the Jetson's guard tripped 0 times, and its swap was never used;
- **every split landed on the plan.** llama.cpp's own buffers match the planned bytes within 0.1 MiB on every worker and
  within 1.6 MiB on the Pi 5 host (the F32 norms stay mapped).

## What it answers

**QR8, the rule against an even split.** The rule cuts decode by **42.3 %** (Nemo) and **44.8 %** (Qwen3-14B), and
prefill by 61.3 % and 60.3 %. With 09-27 that makes **five models, all between −42 % and −45 % in decode**, against the 25 %
stretch target.

**The Pi 5 alone.** Mistral-Nemo fits the Pi 5 alone, as the tensor table said and as the manifest did not. The rule
keeps it there. Qwen3-14B does not fit, and the rule adds the Jetson rather than the Pi 4, as it did for Qwen2.5-14B.

**The predictions.** Decode came within −8.2 % to +0.3 % (mean |error| 4.1 %), and prefill within −2.7 % to −11.9 % (mean
7.1 %). Nearly all errors were optimistic, as on 09-27.

**Memory, out of sample on three more models.** On every board of every cell the engine charged at least the measured
MemAvailable drop, by 17 to 122 MiB. The v2 accounting holds. The Pi 4's MoE host dropped only 325 MiB against 3,850
charged: its weights are mapped from the file, and the engine charges them on purpose because they must stay resident.

## The MoE cell failed, and why

**What happened.** Near the end of the weight upload, the Jetson's `ggml-rpc-server` aborted:
`ggml_vulkan: Memory allocation of size 113246208 failed … vk::Device::allocateMemory: ErrorOutOfDeviceMemory`. The Pi 4
host then lost its RPC server and aborted (exit 134), so no figure was measured.
- **It got almost all the way.** All three planned buffers had been allocated exactly as planned, and the Jetson's eth0
  shows about 3.3 GB received, nearly all of its 3,112 MiB share.
- **No board ran out, in the kernel's terms.** At their lowest, the Pi 5 had 193 MiB available and the Jetson 290 MiB.
- **The accounting was right about the drop.** The Jetson dropped 3,273 MiB against 3,290 charged.

**What failed** is the assumption that a board can be planned down to its last MiB. The Jetson's Vulkan allocator
refuses a 108 MiB allocation while MemAvailable still reads about 290. Across all eleven Jetson pool cells (09-27 and
09-28), every one whose lowest MemAvailable stayed at or above **450 MiB** ran, and the one at 290 failed. The "free"
figure the RPC server prints at start is not the limit: the 09-27 14B even cell held 2,457 MiB when it printed 1,426 MiB
free. The Pi 5, a CPU worker, ran at 193 MiB without trouble.

**Open.** The Jetson's reserve lies somewhere between about 180 MiB (290 minus the 108 it asked for) and 450 MiB. The
MoE's registered result stands as this failure.

## The MoE, re-planned with a reserve (user: "Add reserve, re-run")

**The engine change.** The engine now keeps a per-board reserve free: `cluster.json` `memory_reserve_mb`, 450 MiB on the
Jetson, the lowest MemAvailable at which any Jetson cell has run.
- `memory_cap` subtracts it in every role. `memory_need_mb` does not, so a cell's need is still usage.
- The registration records `reserve_mib` beside each need, and `run_pool.sh` gates on need plus reserve.
- Suites: ThinkPad 209 (5 new), Pi 5 357, Pi 4 158, all OK. A mutation check caught each of the three pieces.

**Registered before it ran.** The re-plan is an addendum:
`testbed/dataset/pool_predictions_20260928_addendum.json`, 17:05:05, `dfc9fb9a…`, read-only. It was written by
`.handoff/assets/moe-reserve-2026-09-28/register_moe_reserve.py`, under a new cell id so the failed cell keeps its own
record. The plan is Pi 4 13 (host) + Jetson 10 + Pi 5 25, with the head on the Pi 5.

**It ran clean**, 17:05–17:18:
- exit 0, every Pi `0x0`, the guard 0 trips;
- the splits on the plan within 0.1 MiB;
- the memory charge above the measured drop on every board (Jetson 2,951 against 3,007, Pi 5 7,377 against 7,512);
- the Jetson bottomed at 599 MiB, above its reserve.

| Qwen3-30B-A3B Q3_K_M, rule with the reserve | measured | predicted | error |
|---|---|---|---|
| decode ms/token | **299.9** (3.34 tok/s; reps 3.27 / 3.38 / 3.36) | 266.9 | −11.0 % |
| prefill ms/token | 248.6 (reps **3.12 / 4.05 / 4.90** tok/s) | 124.6 | −49.9 %, *not steady* |

**Decode: the MoE's point, measured.** A 30.5 B-parameter model decodes at **3.34 tok/s** on these three boards. That is
2.9× the dense Qwen3-14B's best here (877 ms) and 2.3× Mistral-Nemo 12B on the Pi 5 alone (676 ms): a token reads 8 of
the 128 experts. The decode error (−11 %) is inside 09-27's range (up to −14.3 %, always optimistic).

**Prefill is a cold-start figure, not a steady one**, and it is reported as found. The Pi 4 host serves its own blocks
from the mmap (it does not repack). During loading, the whole 14.7 GB file streams through its 7.8 GB page cache to
the workers. The probe shows the process's file-backed memory climb to about 7.2 GB, then fall to **694 MiB** when
llama.cpp releases the uploaded ranges: only 0.7 of the Pi 4's own 3.9 GB of weights were still resident. They came
back during the repetitions, 694 → 3,280 MiB over about two minutes, at a rate that fits reads from the SD card, and
each prefill repetition ran faster than the one before. Decode ran after that, with the cache warmer, and its three
repetitions agree within 3.5 %.
- A steady prefill figure needs the host's weights resident before the measured repetitions: `--no-mmap` on the Pi 4,
  or a warm-up that touches every expert. Both change the protocol, so neither was done without a decision.
- This is also a placement fact. A host holding a model larger than its RAM gets its own weights evicted by the
  upload, and an MoE touches them unevenly, token by token.

**Cost:** about 50 minutes of board time, 16:14–16:51 and 17:05–17:18.

# 2026-09-28 (night) — placement's open items: the predictor's bias, the Pi 4 host's limit, the MoE's warm prefill

The user asked to finish placement's three open items (`Documents/HeteroServe_Remaining_Work.md`, item 4). They chose, for
each, to:
- find the cause of the predictor's optimism, fix it and re-test it;
- sweep the Pi 4 host to about 70 % and 95 % of the engine's limit, plus one forced cell past it;
- take the MoE's prefill with a warm-up test first.

Board time: 17:47–20:18.

## The bias came from the instrument, not the network

On the 15 cells of 09-27 and 09-28, **the single-board cells were the most optimistic**. Their mean decode error was
−10.6 %, against −6.3 % with one worker and −3.8 % with two, so the link model was not the cause.

The predictor prices a board from TinyLlama rows taken with the characterisation protocol (`-p 512 -n 128 -r 3`,
nothing else running). Every pool cell ran `-p 64 -n 16 -r 3 -v` with `memprobe.sh` polling at 1 Hz on every board.
`testbed/scripts/protocol_ab_pi5.sh` separated the two on the Pi 5 alone. It alternated three conditions, two rounds
per model:
- **R**, the row protocol;
- **P**, the pool protocol;
- **PQ**, the pool protocol with the probe running.

The run is `testbed/results/protocol_ab_20260928_174753`: every run exited 0, read `0x0`, and stayed at 35–50 °C.

| Pi 5 alone, ms/token | TinyLlama decode | TinyLlama prefill | Nemo 12B decode | Nemo 12B prefill |
|---|---:|---:|---:|---:|
| R (pp512 / tg128) | 55.25 * | 12.47 * | 628.1 | 152.3 |
| P (pp64 / tg16) | 54.6 | 11.75 | 634.8 | 149.8 |
| PQ (P + the probe) | 56.6 | 11.84 | **663.2** | 152.0 |

\* Round 2 only. Round 1 read 62.69 / 13.15: see the artefact below.

- **The probe cost 4.5 % of Nemo's decode** (3.6 % of TinyLlama's) and 0.8–1.4 % of prefill. It started about five
  processes a second on a board whose four cores llama.cpp keeps in lock-step.
- **The lengths:** about ±1 % in decode. In prefill, pp64 is 6 % faster per token than pp512 on TinyLlama, and 1.6 % on
  Nemo.
- **Model size:** under R, Nemo's decode is 1 % above what TinyLlama's per-byte rate predicts, and its prefill is **8.5 %
  above** the per-parameter rate. Under P the gaps are 3.2 % and 13.2 %, which are the factors 1.0324 and 1.1323 used
  below.
- **No drift in the row itself:** TinyLlama tg128 read 55.25 ms, against 55.16 on 09-27.
- **An artefact, reported as found.** The first TinyLlama run after the 7.5 GB Nemo file had been read ran **13–22 %
  slow**: 62.69 and 66.54 ms/token, in two separate runs. Its within-run spread was tight, a 60 s pause did not prevent
  it, and only a repeat shows it. The Pi 5 rows below were taken after reading the TinyLlama files into the page cache,
  then repeated (`pi5_prime_*.log`). The two runs agree within 0.02 % in prefill and 0.4 % in decode.

## The fix: a probe that starts no process

- **`memprobe.sh` v2** reads `/proc` with bash builtins and sleeps on a pipe nobody writes. It starts 2 processes in 3 s,
  against v1's 52, and its log format is identical. v1 is kept as `memprobe_v1.sh`.
- **On Nemo, v2 costs +0.4 % in decode and +0.5 % in prefill** (`protocol_ab_20260928_183623`). That is inside the
  run-to-run spread of ±0.4 %. TinyLlama's second check was too noisy to read, because of the artefact above.
- **Every pool cell from 2026-09-28 night on runs v2.** The decode figures of all earlier pool cells include the v1
  probe's cost: about 4 % on a Pi 5 host, not re-measured on the other boards.

## Rows at the pooled protocol

- **18 new rows**, at pp64 / tg16 and `-r 3`, none flagged:
  - the Pi 5 as host and as worker, at Q4_K_M and Q3_K_M (`quant_campaign_20260928_185947`, `worker_role_20260928_190009`;
    the repeat is `quant_campaign_20260928_190034`);
  - the Pi 4 as host at both, and as worker at Q4_K_M (`quant_campaign_20260928_183626`, `worker_role_20260928_184042`);
  - the Jetson's Vulkan at both (`quant_campaign_20260928_183626`; guard 0 trips).

  They come from the `PP` / `TG` knobs added to `run_quant_campaign.sh`, `bench_worker_role.sh` and
  `jetson_bench_tinyllama.sh`, whose defaults are unchanged.
- **`characterisation.csv` is rebuilt** (`9d8504f9…`). Every earlier row is byte-identical, and the previous file is
  `characterisation.20260928.csv`. `results.csv` now keeps only pp512 / tg128, because `profiles.py` keys rows by
  phase, and it is unchanged (`9c54706d…`). `predict_pool.py --rows pool` selects the new rows; the default reproduces
  every earlier registration.
- **The Jetson's GPU runs a 64-token prompt 13–16 % slower per token than a 512-token one** (Q4_K_M: 63.4 against
  73.0 t/s). Its decode is unchanged. The Pi 4's rows move within ±3 %.

## The fix, re-tested on pre-registered cells

- **The registration** is `testbed/dataset/pool_predictions_20260928b.json`, 19:02:33, `ce07518c…`, read-only. It was
  written by `.handoff/assets/placement-v3-2026-09-28/register_v3.py`, and `score_night.py` scores it into
  `pool_results_20260928b.json`.
- **Each cell carries three predictions**, with the decision rules registered alongside them:
  - **v3**, from the pool-protocol rows;
  - **char rows**, the rows every earlier registration used;
  - **v3s**, v3 with the Pi 5 host's Q4_K_M costs scaled by the size factors.
- **One cell was replaced before registering.** `nemo12b.even` became `qwen3_14b.even`: under the Jetson's 450 MiB
  reserve, its split with the head on the Jetson had a −60 MiB margin.

| cell | pipeline | decode ms | v3 | char rows | v3s | prefill ms | v3 | char rows | v3s | same split under v1 (decode / prefill) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v3.nemo12b.rule` | Pi 5 40 | **629.8** | −2.8 % | −1.4 % | +0.4 % | **147.0** | −10.2 % | −5.7 % | +1.7 % | 676.1 / 154.5 (`nemo12b.rule`) |
| `v3.qwen3_14b.rule` | Pi 5 35 + Jetson 5 | **823.8** | +1.8 % | +2.9 % | +4.1 % | **183.8** | −8.9 % | −7.0 % | +1.2 % | 877.3 / 193.9 (`qwen3_14b.rule`) |
| `v3.qwen14b.rule` | Pi 5 43 + Jetson 5 | **834.3** | −0.8 % | +0.3 % | +1.5 % | **183.4** | −9.3 % | −7.0 % | +1.0 % | 855.9 / 188.2 (`qwen14b.rule`) |
| `v3.qwen3_14b.even` | Pi 5 14 + Jetson 13 + Pi 4 13 | **1529.7** | −1.4 % | −0.6 % | −0.9 % | **487.6** | −3.6 % | −3.2 % | −2.1 % | 1588.4 / 488.0 (`qwen3_14b.even`) |

Every cell exited 0 and every Pi read `0x0`. llama.cpp's buffers matched the planned split within 3 MiB on every board.

**The registered verdicts:**
- **FIX: keep the characterisation rows in both phases.** Their mean |error| was 1.30 % against v3's 1.70 % in decode,
  and 5.72 % against 8.00 % in prefill.
- **SIZE:** in prefill, **v3s replaces v3**: 1.10 % against 9.10 % on the two out-of-sample Qwen cells. In decode,
  **v3 stays**: 1.30 % against 2.80 %.
- *Not a registered comparison:* over the four cells, v3s's prefill mean |error| is **1.5 %**, against the
  characterisation rows' 5.7 %. v3s is in-sample on Nemo.

**What it means:**
- **The decode bias is gone, and the fix was the probe.** With v2, the predictor every registration used decodes within
  −1.4 % to +2.9 % on four fresh cells, a mean |error| of 1.3 % against the 4–6 % before. The same splits measured 2–7 %
  faster than under v1.
- **Prefill keeps a model-size bias of about −6 %** on Pi 5-hosted cells under the characterisation rows. The size
  factor removes it on two out-of-sample models, but it has been tested only against v3, not against the rows it would
  be applied to. That test, the characterisation rows plus the prefill factor, is the next registration.

## The Pi 4 host's limit

The model is Qwen2.5-14B. The Pi 4 hosts k blocks from its mmap and does not repack; the Pi 5 is the worker and holds
the head. The engine's limit is 42 blocks: 7,300 MiB charged against 7,413 available at rest.

| cell | host blocks | of the limit | decode ms (v3; reps t/s) | prefill ms (v3; reps t/s) | host RssFile while decoding |
|---|---:|---:|---|---|---|
| `p4.qwen14b.k28` | 28 = 4,379 MiB | 67 % | 1705.2 (1617.7, −5.1 %; 0.598 / 0.567 / 0.594) | 766.2 (735.3, −4.0 %) | 4,405–4,406 MiB |
| `p4.qwen14b.k40` | 40 = 6,227 MiB | 95 % | 1947.7 (1997.8, +2.6 %; 0.513 / 0.514 / 0.514) | 1068.0 (928.2, −13.1 %; **0.733** / 1.037 / 1.040) | 6,638 MiB |
| `p4.qwen14b.k45_forced` | 45 = 7,040 MiB | 107 %, forced | 2105.2 (2165.1, +2.8 %; 0.474 / 0.475 / 0.476) | 1207.5 (1008.6, −16.5 %; **0.704** / 0.908 / 0.872) | 7,226–7,229 MiB |

"RssFile while decoding" is not a registered test. It is read from the copied probe log, over the tg16 repetitions only;
an unbounded window takes in the teardown after the test, when the mappings go.

- **No thrash, even 3 blocks past the limit.** By the registered rule none of the three thrashed: their decode ran
  1.05×, 0.97× and 0.97× the prediction, and steady. The host's weights stayed fully resident while decoding.
- **The limit is conservative, as registered.** The engine charges a Pi 4 host for the largest tensor it uploads, here
  the Pi 5's 609 MiB output head. That is a load-time copy, which on this host can only evict page cache, and it is freed
  before decoding. That buys 3–4 blocks of this model. The real thrash point is predicted at 46–47 blocks, where blocks
  plus 137 MiB pass the 7.4 GB available. It is not tested.
- **A new placement fact: the first prefill repetition is cold** on a Pi 4 host whose blocks and the streamed upload
  together overflow its page cache. That happened at 40 blocks and at 45, not at 28. The upload evicts the host's own
  weights, as the MoE showed, and llama-bench's one-pass warm-up does not bring them back. Most of those cells' prefill
  error is that first repetition; the steady ones are within 3.5 % (k40) and 10.3 % (k45) of v3.
- **Proposed, not done:** charge a Pi 4 host its blocks plus overhead, and check the upload copy against anonymous memory
  alone. Test it with a pair of cells at 46 and 47 blocks.

## The MoE, warm

- **The cell** re-ran the registered `qwen3moe.rule_reserve` split: Pi 4 13 (host) + Jetson 10 + Pi 5 25. An unscored
  pp512 test ran ahead of the scored pp64 / tg16 in the same llama-bench run (the cell's `bench.protocol`).
- **Registered rule MOE WARM: warm.** At the start of the scored pp64 test, and all through it, the host held 95–96 % of
  its 13 blocks resident: 3,582–3,598 of 3,751 MiB. Its three repetitions agree within 1.5 %: 5.40 / 5.35 / 5.32 t/s. The
  pp512 test absorbed the refill, starting from almost nothing after the upload and running at 6.17 → 6.42 t/s.
- **Steady prefill is 186.6 ms/token (5.36 t/s)**, against the cold 248.6 (4.02 t/s) at 17:05, a quarter faster. It is
  still **34 % slower than predicted** (123.3). The predictor prices an MoE prompt by its active parameters, but a 64-token
  batch sends about 4 tokens to each of the 128 experts, so each expert's matmul runs at a tiny batch. This is an open
  pricing gap for MoE prefill.
- **Decode is 290.0 ms (3.45 t/s)**, against 299.9 in the v1-probe cell, and predicted 263.3 (−9.2 %).
- **The rest:** exit 0 and `0x0` on both Pis. The Jetson had 0 trips, stayed at `pmode:0000`, and received 3.3 GB over
  eth0.

## Late night: the prefill size factor against the rows in use, and the Pi 4's thrash point (user: "go on")

**The registration** is `testbed/dataset/pool_predictions_20260928c.json`, 20:27:36, `53a8d538…`, read-only, by
`register_c.py`. It is scored by `score_c.py` into `pool_results_20260928c.json`.

The chain was paused 20:44–21:17 so the user could chat with the MoE; see "Serving" below. The four unrun cells then ran
under the same registration. Every cell exited 0, every Pi read `0x0`, and the guard recorded 0 trips.

**SIZE2 tests the characterisation rows with the Pi 5 host's Q4_K_M prefill scaled by 1.0806.** The factor is Nemo's
pp64 in the A/B against what the characterisation row predicts. The test used five fresh splits that no earlier cell had
measured.

| cell | pipeline | prefill ms | char rows | char + factor | v3s | decode ms | char rows |
|---|---|---:|---:|---:|---:|---:|---:|
| `s.qwen3_14b.p5j8` | Pi 5 32 + Jetson 8 | **184.8** | −6.7 % | −0.8 % | +2.2 % | 845.1 | +4.1 % |
| `s.qwen14b.p5j8` | Pi 5 40 + Jetson 8 | **184.7** | −7.0 % | −0.9 % | +1.6 % | 860.4 | +0.5 % |
| `s.nemo12b.p5j10` | Pi 5 30 + Jetson 10 | **155.8** | −8.0 % | −2.6 % | +1.0 % | 732.5 | +2.1 % |
| `s.qwen3_14b.p5p4` | Pi 5 26 + Pi 4 14 | **499.0** | −2.3 % | −0.6 % | −2.4 % | 1390.5 | +2.0 % |
| `s.nemo12b.p5p4` | Pi 5 28 + Pi 4 12 | **370.9** | −1.6 % | +0.5 % | −1.3 % | 1088.3 | +2.8 % |

- **Registered verdict: adopt.** The factor's prefill mean |error| is 1.08 %, against the characterisation rows' 5.12 %.
- **Where it lives:** `predict_pool.py` now carries it as `PI5_HOST_PREFILL_SIZE`, applied with `--size` (`costs(..., size=True)`).
  Without the flag, every earlier registration reproduces unchanged; with it, the registered char + factor predictions of
  all seven cells reproduce exactly.
- **Decode is not a SIZE2 test.** The characterisation rows now decode **0.5–4.1 % pessimistic**, with a mean of +2.3 %
  over these five cells. The old optimism has not just shrunk but reversed slightly. It is largest where the Jetson holds
  the head. The Jetson's RPC cost factor (+2.5 %, from the 09-27 smoke test under the v1 probe) is a suspect, not
  tested.

**THRASH2 finds the Pi 4 host's thrash point.** The model is Qwen2.5-14B, forced past the engine's 42-block limit, on
the registered short protocol `-p 16 -n 2 -r 3`.

| cell | host blocks | steady state (blocks + 137 MiB) | decode (tg2) | against its prediction | host RssFile while decoding |
|---|---:|---|---|---:|---|
| `p4.qwen14b.k46_forced` | 46 = 7,206 MiB | 7,343 of 7,413 MiB: fits by 70 | **0.469 tok/s** (0.468 / 0.469 / 0.470) | 0.97× | 7,229 MiB, all resident |
| `p4.qwen14b.k47_forced` | 47 = 7,373 MiB | 7,510 of 7,413 MiB: 97 short | **0.026 tok/s** (0.028 / 0.026 / 0.023) | **17.6×** | 7,185–7,234 MiB: **98 % resident** |

- **The steady-state model placed the cliff exactly**, between 46 and 47 blocks, and it is a cliff: one more 157 MiB block
  cut decode **17.6-fold**, from 2.13 s to 39.2 s per token.
- **98 % resident is not enough.** Decode reads every block in order on every token, so the missing 2 % keeps evicting the
  pages needed next. At the SD card's ~47 MB/s, 39 s per token is about 1.8 GB re-read per token, about a quarter of the
  host's weights.
- **Registered verdict ACCOUNT: adopt.** A Pi 4 host's charge should be its blocks plus its host overhead, with the
  largest-upload copy checked against the anonymous part alone. That allows 46 blocks of this model, not 42. Wherever that
  charge overflows, the engine must refuse: 47 blocks would run at 1/17 of the speed.
- **Implemented on the user's go** (2026-09-28 late night): `cluster.json` Pi 4 `host_weights_mapped: true`, and
  `DeviceProfile.host_weights_mapped`.
  - `charged_bytes` drops the upload copy for such a host, and `memory_cap` checks overhead + copy against the budget alone.
  - The flag is read in `config.py` and passed through by `profiles.py` and `predict_pool.py`.
  - The Pi 4 host's cap for the 14B is now 46 (it was 42). The Pi 5 is unchanged.
  - Five new tests; ThinkPad 214, Pi 5 357 and Pi 4 158 pass. A mutation check caught 5 of 5 breaks, each by its own test.
- **Prefill:** both cells repeat the cold first repetition (0.313 → 0.916 tok/s at k46).

**Serving** (not a measurement). `testbed/scripts/serve_pool.sh` served the MoE for the user's chat. It used the engine's
split for a 4096-token context: Pi 4 15 (host) + Jetson 9 + Pi 5 24 (head), predicted 3.65 tok/s, checked from the
workers' memory. The Pi 5 was left with 284 MiB free and the Jetson with 800 MiB, above its reserve. The first request
after loading read 0.26 tok/s in prefill and 0.76 in decode while the Pi 4 re-read its experts, the same cold start as in
the cells. The Jetson's guard recorded 0 trips, with a lowest input of 4,848 mV.

## MoE prefill, priced at the per-expert batch (user: "Model it, then a registered test")

**The hypothesis.** In an n-token prompt, each of Qwen3-30B-A3B's 128 routed experts sees only about n × 8 / 128 tokens: 4 of
a 64-token prompt. Each expert's matmuls therefore run at the efficiency of that tiny batch. A layer's dense parameters
(attention, norms, the router: 19.1 M per layer) run at batch n. The active expert parameters (37.7 M per layer) run at the
per-expert batch.

**The inputs are TinyLlama Q3_K_M prefill rows at prompt lengths 1–512**, in the MoE's roles: the Pi 4 as host, the Pi 5 as
worker, and the Jetson's Vulkan (`quant_campaign_20260928_220032`, `worker_role_20260928_220322`; guard 0 trips). None of
the inputs comes from the MoE.

| TinyLlama Q3_K_M prefill, tok/s | pp1 | pp4 | pp16 | pp64 | pp256 |
|---|---:|---:|---:|---:|---:|
| Jetson GPU (Vulkan) | 7.97 | **12.96** | 17.34 | 57.89 | 69.72 |
| Pi 4 CPU (host) | 6.74 | 9.83 | 10.35 | 10.42 | 10.32 |
| Pi 5 CPU (RPC worker) | 23.03 | 33.68 | 36.94 | 37.25 | 36.67 |

The CPUs flatten from about 4 tokens up. **The Jetson's GPU is 4.5× slower per token at a 4-token batch than at a 64-token one.**

**The test.** It is registered as `pool_predictions_20260928moe.json`, 22:04:51, `c8f05dd1…`, by `register_moe.py`, and scored by
`score_moe.py` into `pool_results_20260928moe.json`. It has two fresh splits. An unscored pp512 warm-up ran first, then pp16 /
pp64 / pp256 were scored, each with the batch-aware and the active-parameter prediction at the same rows.

| cell | pipeline | pp16 | pp64 | pp256 | tg16 |
|---|---|---|---|---|---|
| `m.a` | Pi 4 15 (host) + Jetson 9 + Pi 5 24 | 261.9 ms: batch-aware −14.2 %, active −40.7 % | 189.4: −15.9 %, −30.3 % | 164.8: −10.9 %, −20.1 % | 292.5 (−6.4 %) |
| `m.b` | Pi 4 18 (host) + Jetson 8 + Pi 5 22 | 255.3: −7.1 %, −35.1 % | 194.3: −12.5 %, −25.5 % | 173.8: −9.1 %, −16.8 % | 300.2 (−4.7 %) |

- **Registered verdict MOEPRE: adopt the batch-aware pricing.** Its mean |error| is 11.6 %, against 28.1 %.
- **It gets the shape right.** The measured prefill is 1.59× (m.a) and 1.47× (m.b) slower at pp16 than at pp256. The
  batch-aware pricing predicts 1.53× and 1.50×; the active-parameter pricing predicts 1.18× and 1.15×.
- **About 10 % optimism remains.** This MoE's expert matrices are narrow (768 wide, against TinyLlama's 5,632), and the
  routing is uneven.
- **Reported as found:**
  - pp16 was not steady in either cell (5.2 % and 6.5 % spread, the third repetition slower both times); it is still scored,
    as registered;
  - the Pi 4 host held 91–95 % of its blocks through the scored tests;
  - the guard recorded 0 trips.
- **In the predictor:** `predict_pool.py --moe-batch` (`moe_prefill`, `pp_curve`, `rate_at`, `rows_by_test`). Without the flag,
  every earlier registration reproduces. For the registered `qwen3moe.rule` split the flag gives 152.6 ms instead of 124.6;
  that split measured 186.6 warm.

## Two qwen35 hybrids the predictor was never fitted on: Qwen3.8-27B and Qwen3.5-9B

**The models.** Both are pinned and verified on their board (`testbed/models.md`; upstream licence Apache-2.0).
- **Qwen3.8-27B Q3_K_M**, 13.40 GB, on the Pi 4. The user: "let's download the 27B and let's benchmark it out".
- **Qwen3.5-9B Q4_K_M**, 6.17 GB, on the Pi 5. It is the prefill/decode separation model; the user: "ok let's go with Qwen3.5-9B".

Both are `qwen35` hybrids, which the pin supports. Three of every four layers are recurrent (Gated DeltaNet) and every
fourth is full attention, and each model has one multi-token-prediction (MTP) block (`nextn_predict_layers` 1). Every cost
the predictor uses was measured on standard transformers.

**The Jetson cannot run them at the pin.** On the engine's 27B rule split (Pi 4 10 + Jetson 17 + Pi 5 38), an unregistered
smoke run loaded fine. It reported "offloaded 56/66 layers", so the MTP block is counted in the offload. **The Jetson's
Vulkan worker then died silently on the first graph** (`testbed/results/pool_smoke.qwen38_27b.rule_20260928_231251.log`,
EXIT_BENCH 134). There was no error line, no dmesg record and no guard trip, and it had 515 MiB free at its lowest. The cause
is not established; a qwen35 operation on the Maxwell GPU is the suspect. The Pis-only smoke ran clean. No figure from
either smoke run is reported.

**The registered test** is `testbed/dataset/pool_predictions_20260928hyb.json`, 23:38:52, `64ec6bbb…`, by `register_hyb.py`. It
is scored by `score_hyb.py` into `pool_results_20260928hyb.json`. Its predictions use the adopted predictor: characterisation
rows, SIZE2 (`--size`), and the MTP block not computed.

| cell | pipeline | decode ms (tok/s) | predicted | prefill ms | predicted | notes |
|---|---|---|---|---|---|---|
| `h.q27.pis_rule` | Pi 4 27 (host) + Pi 5 38 | **2318.1 (0.431)** | 2094.3 (−9.7 %) | 1628.1 | 1446.4 (−11.2 %) | prefill not steady: cold first repetition (0.548 → 0.647 tok/s) |
| `h.q27.pis_b` | Pi 4 32 (host) + Pi 5 33 | **2466.2 (0.405)** | 2265.8 (−8.1 %) | 1923.7 | 1585.3 (−17.6 %) | prefill not steady: cold first repetition (0.385 → 0.588) |
| `h.q9.pi5alone` | Pi 5 33 | **544.1 (1.838)** | 466.3 (−14.3 %) | 106.3 | 95.5 (−10.2 %) | steady |
| `h.q9.p5p4` | Pi 5 20 (host) + Pi 4 13 | **1017.2 (0.983)** | 966.2 (−5.0 %) | 307.6 | 273.5 (−11.1 %) | steady |

Every cell exited 0 and every Pi read `0x0`.

- **Registered verdict TRANSFER: the predictor does not transfer to the hybrid.** Prefill is outside ±10 % on all four cells,
  and decode is outside ±5 % on three of the four. The errors all have the same sign: **the hybrids run slower than their
  weight bytes and parameters explain**, 5–14 % in decode.
- **An exploratory reading, not a registered test.** Spread over the recurrent layers, the decode shortfall is about 3.2 ms per
  recurrent layer per token for the 9B on the Pi 5 alone, and 4.2–4.7 ms on the 27B's Pi 4 + Pi 5 splits. The 9B across both
  Pis leaves only about 2.1 ms, so no single per-layer constant fits all four. A recurrent-layer cost term needs its own rows
  (a small qwen35 model on each board) and its own registered test.
- **The 9B on the Pi 5 alone decodes at 1.84 tok/s**, just under QR2's 2 tok/s; it was predicted at 2.14.
- **The MTP block is not even loaded.** On the board holding the last blocks, llama.cpp's buffer is 228–275 MiB (about one
  block) smaller than the plan in every cell, although "offloaded n/66" counts the block. The engine's memory charge for it is
  therefore cautious by one block on that board.
- **A Pi 4 host's cold first prefill repetition appears again** at 27 and 32 host blocks (4.9–5.8 GB).

## Prefill/decode separation: the first working hand-off (QR4, Project 2)

**The prototype.** `testbed/scripts/pd_handoff.py` prefills on one board and decodes on another, using llama-server's slot
save / restore (`--slot-save-path`). At the pin this is hybrid-aware: `llama_memory_hybrid` writes both the attention KV and
the recurrent state. The model is Qwen3.5-9B Q4_K_M, with a copy on each Pi, a 98-token prompt, greedy decoding, and
ctx 4096.

**A limit found on the way: a plain restore silently re-prefills a hybrid model.** llama-server matches a new prompt against
the slot's cached tokens. When the two are *equal*, it steps back one token to recompute the last logits, which a recurrent
state cannot do, so it re-processes the whole prompt. After a restore, the first check's continuation processed all 98
tokens (`testbed/results/pd_check_20260929_003221.log`); its identical output proved nothing. **The protocol that works:**
the prefill side also produces the first token t1, and the decode side sends prompt + t1. That prompt is longer than the
cache, so nothing rolls back and only t1 is processed. With it, on one board (`pd_check_20260929_003611.log`):
- 32 of 32 continuation tokens were identical to a run that never saved;
- 1 prompt token was processed after the restore.

**The hand-off, Pi 5 → Pi 4, 3 repetitions** (`pd_handoff_20260929_004018.log`), against the Pi 4 doing both itself, 3
repetitions (`pd_alone_20260929_005010.log`):

| step | repetitions | |
|---|---|---|
| Pi 5 prefill, 98 tokens | 10.87 / 10.64 / 10.62 s | 109 ms/token |
| save slot, 55,905,784 bytes | 0.04 / 0.04 / 1.52 s | |
| transfer, Pi 5 → laptop → Pi 4 (two ssh hops, onto the Pi 4's SD card) | 1.92 / 4.61 / 3.47 s | **12–29 MB/s** |
| restore on the Pi 4 | 0.07 s each | |
| **time to the first decode on the Pi 4** (prefill + save + transfer + restore) | **12.90 / 15.36 / 15.68 s, mean 14.6** | |
| the Pi 4 prefilling itself | 61.01 / 60.96 / 61.01 s | 622 ms/token |
| Pi 4 decode after restore / alone | 1523.5 / 1520.9 ms/token | the same |

- **The output survives the move.** In every repetition, t1 and all 32 decoded tokens matched the Pi 4's own run, and the Pi 4
  processed 1 prompt token. *(For this prompt. Token-for-token agreement with one board's own run does not hold at every
  length; see the sweep below.)* The state file is portable between the two boards, although the Pi 5 repacks its weights and the
  Pi 4 does not.
- **The hand-off gets the Pi 4 decoding 4.2× sooner** (3.9–4.7×), because the Pi 5 prefills this model 5.7× faster than the
  Pi 4.
- **The state is mostly fixed.** 55.9 MB for 98 tokens is the recurrent state, about 50 MB whatever the prompt, plus ~32 KB
  per token of attention KV. A long prompt adds little.
- **The bottleneck is the transfer**, 3.3 s of the 14.6, at a fraction of the 938 Mbit/s link. The file goes through the
  laptop and is written to an SD card. Direct board-to-board transfer into RAM (tmpfs), or restoring from a stream rather than
  a file, is the obvious next step (the plan's "upstream extension").
- **What it does not show.** For one user the Pi 5 alone is still fastest: it decodes this model at 544 ms/token (llama-bench,
  `h.q9.pi5alone`; 540–546 in the checks' four server decodes), against the Pi 4's 1,521. Separation pays when the Pi 5's time is worth more elsewhere, for example with a second user's prompt waiting.
  That needs a concurrency experiment.
- **Reported as found: a network event.** The Pi 4's Ethernet link dropped at 00:48:26 (kernel "eth0: Link is Down"; up
  again at 00:48:43; DHCP re-done by 00:49:25), 30 s after the hand-off run ended. No figure was affected; the next run
  was simply retried. A carrier loss is physical: the cable, the router port or the NIC.
- **Corrected 2026-09-29 ~01:15** against the raw logs: the per-repetition range, first written as 12.8–16.3 s (3.7–4.8×), the
  Pi 5's 108 ms/token, the 5.8× and the Pi 5's decode figure (541 was one server decode). The mean, 14.6 s, stood.
- Both Pis read `get_throttled` `0x0` at ~01:05, with no reboot since 2026-09-28 10:18 (Pi 5) and 10:24 (Pi 4), so the
  sticky bits cover every run above.

## The hand-off by prompt length: 128–1024 tokens, the state sent straight into the Pi 4's RAM (user: "continue our work")

**What changed from the prototype.** The slot files live in tmpfs (`/dev/shm`) on both boards. The state goes directly from
the Pi 5 to the Pi 4 over the wired overlay (192.168.10.x, 938 Mbit/s in `link.json`): one TCP connection, timed by the sender
up to the receiver's acknowledgement that the file is written, and sha256-checked on both ends. There is no laptop hop, no ssh
encryption and no SD card. A test on the prototype's 55.9 MB file (`pd_xfertest_20260929_010803.log`) took 0.49–0.50 s,
against 1.9–4.6 s via the laptop.

**The run.** `pd_handoff.py sweep`, 01:08–02:56 (`testbed/results/pd_sweep_20260929_010837.log`; the server logs, with no error
line, are beside it).
- **Prompt:** the GPL-3 text (`/usr/share/common-licenses/GPL-3`, sha256 in the log) cut to exactly n tokens, then a fixed
  question.
- **Design:** for each length, 3 repetitions, each running three arms in turn: the Pi 4 alone and the Pi 5 alone (its own
  prefill and 33 tokens), then the hand-off (Pi 5 prefill and t1; the Pi 4 restores and decodes 32).
- **Result:** 36 of 36 arms ran and none needed a make-up. Every Pi read `throttled=0x0` at each length and at the end (Pi 4
  ≤ 66.2 °C, Pi 5 ≤ 38.4 °C). Spreads are max − min over the 3 repetitions.

| prompt tokens | Pi 4 alone: prefill | Pi 5 alone: prefill | hand-off: Pi 4 decoding after | sooner than Pi 4 alone | state file (predicted) | transfer |
|---|---|---|---|---|---|---|
| 128 | 80.01 s (0.08) | 13.71 s (0.52) | **14.09 s** (0.04) | 5.68× | 56,889,664 (56,885,248) | 0.51 s |
| 256 | 159.58 s (0.07) | 27.37 s (0.16) | **27.92 s** (0.07) | 5.72× | 61,087,552 (61,079,552) | 0.54 s |
| 512 | 321.29 s (0.13) | 55.08 s (0.10) | **55.80 s** (0.18) | 5.76× | 69,483,328 (69,468,160) | 0.61 s |
| 1024 | 648.48 s (0.22) | 111.50 s (0.42) | **112.24 s** (0.29) | 5.78× | 86,274,880 (86,245,376) | 0.76 s |

| prompt tokens | Pi 4 decode, alone / after restore | Pi 5 decode, alone | Pi 5 ÷ Pi 4 speed: prefill / decode |
|---|---|---|---|
| 128 | 1515.6 / 1515.6 ms/token | 558.4 | 5.84× / 2.71× |
| 256 | 1522.5 / 1521.9 | 560.7 | 5.83× / 2.72× |
| 512 | 1538.6 / 1538.8 | 564.6 | 5.83× / 2.73× |
| 1024 | 1568.9 / 1568.6 | 578.4 | 5.82× / 2.71× |

- **The hand-off costs under a second.** Save 31–43 ms, transfer 0.51–0.76 s at 112–114 MB/s (about 90 % of the link) and
  restore 68–112 ms, for 0.60–0.92 s in all. The Pi 5's prefill is the rest. The Pi 4 decodes after a restore exactly as fast
  as after its own prefill, and its first step (t1, giving t2) takes 1.60–1.65 s.
- **The state size is known exactly.** The formula stated before the run was 52,690,944 bytes of recurrent state (24 gated
  delta-net layers, f32) plus 32,768 per token of attention KV (8 layers, f16), "plus a small header". The header turned out
  to be 832 + 28 bytes per token. **The file is 52,691,776 + 32,796 × n bytes, exact at all five lengths measured** (98 in
  the prototype, and 128–1024). Even at 1024 tokens, 61 % of the file is the fixed recurrent state.
- **The board ratios are flat.** The Pi 5 is 5.82–5.84× faster than the Pi 4 at prefill and 2.71–2.73× faster at decode, at
  every length. Prefill cost per token hardly moves (Pi 4 623–633 ms, Pi 5 107–109 ms), and decode slows ~3.5 % from 128 to
  1024 tokens on both.
- **An exploratory reading, not a registered rule:** the Pi 4's comparative advantage is decode. It is 5.8× behind at prefill
  but only 2.7× behind at decode, so if it does any of the work, the split that loses least gives it the decoding. For a
  single user the Pi 5 alone is still best: its first token comes at the same moment as the hand-off's t1 (both are the Pi
  5's prefill), and every later token is 2.7× faster. The hand-off only pays when the Pi 5's time is needed for other
  requests; the concurrency experiment decides that.

**Reported as found: greedy output depends on the board, in both phases.** Each arm is repeatable: all 3 repetitions gave
identical tokens, and t1 was the same in every arm. But the three arms do not always agree:

| prompt tokens | Pi 4 alone vs Pi 5 alone first differ at token | the hand-off follows |
|---|---|---|
| 128 | 14 | the Pi 4 for all 33 |
| 256 | 1 | the Pi 4 for all 33 |
| 512 | 19 | the Pi 5 for all 33 |
| 1024 | 1 | the Pi 5 up to token 27, then neither |

(Tokens are counted from 0, and t1 is token 0.)
- **The two boards' arithmetic differs, and the difference can come from either phase.** Where the hand-off follows the Pi 4,
  the split came from decode arithmetic. Where it follows the Pi 5, it came from the prefill state. At 1024 tokens both
  happen. The Pi 5 repacks the 9B's weights for its own kernels (`CPU_REPACK`, `q4_K_8x4`, in
  `pool_h.q9.pi5alone_20260929_001500.log`). A Pi 4 host maps them unchanged (`CPU_Mapped` only, in
  `pool_h.q27.pis_rule_20260928_234032.log`). So the two boards run different kernels, and the results differ at
  floating-point level. The server logs do not print the buffer lines.
- **The divergence can be large in kind.** At 256 and 1024 tokens, one side opens a `<think>` block while the other answers
  directly:
  - at 256, the Pi 4 alone and the hand-off think;
  - at 1024, only the Pi 4 alone does.
- **What this means for checking.** A hand-off cannot be verified by matching one board's own run token for token. The
  integrity evidence is this:
  - the same-board check (`pd_check_20260929_003611.log`: 32 of 32 against a run that never saved);
  - a byte-identical state on both ends (sha256);
  - `n_restored` = n;
  - exactly 1 prompt token processed after the restore.
  Pooled pipelines split layers across these same boards, so they cannot be expected to match either board exactly either.

**Next for separation:** two users at once, the experiment that decides whether the hand-off earns its place.
- **First, cheaply:** batched decode on each board (`llama-batched-bench` at 1, 2 and 4 sequences). Whether a second sequence
  costs a memory-bound board almost nothing sets how much the Pi 5 alone can already serve.
- **Then:** the Pi 5 alone with 2 slots, against Pi 5 prefill → Pi 4 decode for the second user.

## The ThinkPad stack's first runs on the cluster: heteroctl, laptop-driven (user: "let's run the whole thinkpad thing")

**What it took.** heteroctl (`ThinkPad/control`) had never run on the real cluster. The user chose "Full bring-up, step by step",
then "Laptop-driven runs, no agents". NOTEs are in `.handoff/nodes/{ThinkPad,Pi5,Pi4,Jetson}/`.
- **Network facts corrected.** `cluster.json`, both Pis' `node.env` and both `cluster.hosts` lists still held the proposal's
  192.168.50.x and user `pi`. The Jetson is not corrected yet: that needs its own `node.env`.
- **The agents can't start RPC servers.** The Pi 5 agent has no `/rpc/start`. The Pi 4's needs a systemd unit and sudoers rule
  that are not installed. The Jetson has none. So the workers were started over ssh, the testbed's way
  (`testbed/scripts/heteroctl_run_pis.sh`), and heteroctl records its agent calls as unreachable.
- **The laptop build.** llama.cpp at the pin, built by `01_build_llama_cpp.sh` into `/opt/heteroserve` (the user created the
  directory).
- **Three crash fixes in `ThinkPad/control`**, each with the user's approval, a test, a mutation check and a NOTE:
  1. **The pinned `llama-cli` rejects `-no-cnv`,** so the harness now calls `llama-completion`.
  2. **`Placement.describe()` crashed on latency-only (None) energy,** which every profile has until the DAQ exists.
  3. **The harness never passed the context it had planned KV for.** At the pin, llama.cpp then sizes the cache for the model's
     training context: 40,960 tokens for Qwen3-14B, about 6.7 GB. The first 14B run OOM-killed the Pi 5's `ggml-rpc-server`
     89 s in (anon-rss 7.87 GB). The harness now passes `-c`.
- **Inputs** (`testbed/results/heteroctl_20260929/`):
  - latency-only profiles built by `heteroctl profiles` from the measured TinyLlama rows (`results.csv` already has
    heteroctl's columns; one quantisation per profile);
  - `link.json` for the links;
  - a run-only config with the Jetson disabled.

**Qwen3-14B Q4_K_M through heteroctl**, `run --strategy rule`.
- The plan was Pi 4 6 + Pi 5 34 layers, with the laptop as host (embedding lookup and sampling only). The output layer is on the
  Pi 5, a worker here.
- 128 tokens at ctx 2048, 3 repetitions (`runs_20260929_122501.log`; records under `runs/`). Every Pi read `0x0`: Pi 5 ≤ 48.3 °C,
  Pi 4 ≤ 57.4 °C.

| rep | load (9 GB of weights sent over RPC) | prefill, 8 tokens | decode, 127 tokens |
|---|---|---|---|
| 1 | 88.4 s | 3.65 s | 966.3 ms/token |
| 2 | 89.4 s | 3.65 s | 965.3 ms/token |
| 3 | 88.4 s | 3.60 s | 965.2 ms/token |

- **Decode is 965.6 ms/token (1.04 tok/s), with a 0.12 % spread across the three repetitions.**
- **heteroctl's own prediction was 141.6 ms/token (7.06 tok/s), 85 % too fast.** Its profile builder
  (`profiles.per_layer_times`) takes a median ms-per-layer from the profiled model and applies it to any model's layers, so a
  14B layer is priced as a TinyLlama layer. The split it chose follows the boards' speed ratio. The absolute figure has no
  meaning, which is why the testbed prices per byte (`predict_pool.py`).
- **The memory plan held.** With the output layer on a worker (a first) and the launch at the planned context, nothing ran out
  of memory. The worker's transient copy of the 609 MiB output tensor, never measured, did not tip it over.
- **Context only, not a comparison.** The testbed's Pi-hosted 14B pools measured:
  - Pi 5 35 + Jetson 5: 823.8 ms/token;
  - Pi 5 26 + Pi 4 14: 1,390.5 ms/token.
  Both used llama-bench tg16, with the Pi 5 as host. The split, instrument, host role and generation length all differ.
- **The load is paid on every run.** With the laptop as host, every run first ships all 9 GB to the workers, about 88 s at
  ~100 MB/s. The testbed's Pi-hosted runs read the host's share from its own disk.

## The Jetson's CUDA build at the pin, and CUDA against Vulkan on the same GPU (user: "can we do cuda part right now?")

**The build** (CUDA 10.2, sm_53, `GGML_RPC=ON`, into `~/llama.cpp-cuda`). The user: "Micro-USB, build with guard", "Yes, install
g++-8", "Yes, fetch and install".
- **The power guard** was restarted. It had stopped at a clean shutdown at 07:26. 0 trips from then to the end of the benchmarks.
- **g++-8** (CUDA 10.2's newest allowed host compiler) was not installed, and **the Jetson has no internet** (the router has no
  WAN). The laptop fetched the 7 packages apt asked for (21.5 MB; sizes and md5 checked against `apt-get --print-uris`). apt
  installed them offline from its cache.
- **Two build failures, both in code patch v1 had never compiled** (it passed `git apply --check`, but no build with RPC on had
  been run). Both are fixed in **patch v2**, `testbed/patches/jetpack46-cuda102-b23701f.patch`, sha256 `1f43a3e4…2283c6`. v1 is kept
  in `provenance/`, and the README there has the details.
  1. **8 %: `ggml-rpc.cpp` needs C++17** (`std::filesystem`), while the port holds ggml at C++14 for nvcc 10.2. The fix gives C++17
     to that target alone.
  2. **Built, but every tool segfaulted at start-up** in `std::filesystem::path::_M_split_cmpts`.
     - `libggml-cuda.so` is linked by g++-8, which pulled in gcc 8's static `libstdc++fs.a` and **exported 329 of its
       `std::filesystem` symbols**. That copy then interposed over gcc 11's libstdc++ for the whole process.
     - The fix is `--exclude-libs,libstdc++fs.a` on that target: 0 exported, and `CUDA0: NVIDIA Tegra X1` is found.
  - `pin + v1` and `pin + v2`, each freshly applied, differ by exactly those two insertions.
- **The benchmarked binaries** came from the script (stopped at 8 %) plus two in-place continuations, on a tree that is
  `pin + v2` exactly. `~/heteroserve/build_info_cuda.json` says so (`built_how`). One clean script run with v2 is still to do.

**CUDA against Vulkan, A-B-A** (CUDA, Vulkan, CUDA, 13:56–14:06).
- `jetson_bench_tinyllama.sh`, TinyLlama 1.1B Q4_K_M (sha `9fecc3b3…`, byte-identical to the Pis' copy), pp512 / tg128, 5
  repetitions, `-ngl 99`.
- Conditions: MAXN, clocks locked (CPU 1479 MHz, GPU 921.6 MHz), swap off, every case started ≤ 33 °C and ended ≤ 38 °C.
- **Fan PWM 0 throughout.** It is under automatic control, and `jetson_clocks` on L4T 32.7.6 only forces it with `--fan`, which
  the script does not pass. The user noticed the fan never runs.
- Raw output: `testbed/results/jetson_tinyllama-q4km-aba{1_cuda,2_vulkan,3_cuda}_*`, chain log
  `jetson_cuda_vs_vulkan_20260929_135639.log`.

| run | backend | pp512 t/s (5 reps) | tg128 t/s (5 reps) | whole-run power, mean / peak |
|---|---|---|---|---|
| A1 | CUDA | 115.34 (115.24–115.44) | 6.148 (6.146–6.149) | 6.47 / 7.32 W |
| B | Vulkan | 73.20 (73.13–73.25) | 11.494 (11.492–11.497) | 7.30 / 8.47 W |
| A2 | CUDA | 115.61 (115.54–115.68) | 6.152 (6.151–6.153) | 6.55 / 7.32 W |

- **CUDA is 1.58× faster at prefill; Vulkan is 1.87× faster at decode.** The two CUDA runs, ten minutes apart, agree within
  0.23 % (pp) and 0.07 % (tg). Vulkan matches its 2026-09-25 figures (73.02 / 11.46) within 0.3 %.
- **CUDA decode is slower than the Jetson's own CPU.** The dataset's CPU-only rows are tg128 6.33–6.46 and pp512 8.4–8.7. So on
  this board, prefill ranks CUDA > Vulkan > CPU (115.5 / 73.2 / ~8.6) and decode ranks Vulkan > CPU > CUDA (11.5 / ~6.4 / 6.15).
  - **Cause not established.** sm_53 has no `__dp4a` (sm_61+), so the CUDA quantised mat-vec path may be falling back. That is a
    hypothesis, not a measurement.
- **For placement and separation (exploratory):** the best backend depends on the phase on this one board. That is the Pi 5 /
  Pi 4 prefill-decode story again, inside one device.
- **Power is not compared per phase.** The whole-run means mix different pp/tg time shares (CUDA spends 76 % of its window in tg,
  Vulkan 56 %), so they are not per-token figures.
- **Reported as found about the summariser** (frozen at `44456e39`, so not changed):
  - A1 shows "pins: verified" with "patched files: 0". That is an artefact: its tree's changes were staged when patch v2 was
    generated, and the script counts unstaged changes only. Unstaged before A2.
  - A2 shows "BUILD NOT VERIFIED (patched files 34)", which is its known wording for any patched CUDA tree (r13b finding #0).
  - The provenance is in `build_info_cuda.json`.

## The Jetson's CUDA build, finished: a clean build, CPU parity, pooled runs, and one revision again (user: "finish all the Next CUDA steps, leave the fan alone")

**1. One clean script run with patch v2.**
- 14:12–15:04, `jetson_build_llama_cuda.sh`, exit 0, 0 error lines, 0 guard trips. The benchmarked tree was moved aside first, to
  `~/llama.cpp-cuda.aba-20260929`, not deleted.
- It took 52 min because llama.cpp's default `GGML_CCACHE` served the objects the earlier builds had compiled from identical
  inputs. The tree, configure, patch and links were all fresh.
- The script wrote its own `build_info_cuda.json`. `libggml-cuda.so` exports 0 `std::filesystem` symbols, and `CUDA0: NVIDIA Tegra
  X1` is found.
- **The clean binaries perform like the benchmarked ones**: pp512 115.15 t/s, tg128 6.146, within 0.4 % and 0.1 % of A1/A2
  (`jetson_tinyllama-q4km-clean_cuda_20260929_150619`).

**2. CPU-backend parity on the same node** (the pin note's obligation for patches that touch kernel math).
- Script: `testbed/scripts/jetson_parity_kld.sh`; log `testbed/results/jetson_parity_20260929_151001/run.log`.
- Setup: TinyLlama 1.1B Q4_K_M; the board's `/usr/share/common-licenses/GPL-3` (sha `8ceb4b9e…`); ctx 512, 8 chunks.
- The reference is the UNPATCHED pin on the CPU (`-ngl 0 -dev none`): PPL 4.3480 ± 0.2510, with its logits saved (130.6 MB).

| scored against the CPU logits | PPL(Q)/PPL(base) | mean KLD | median KLD | same top token | RMS Δp |
|---|---|---|---|---|---|
| **CUDA**, pin + patch v2 | 0.9982 ± 0.0047 | 0.01475 ± 0.00068 | 0.00518 | 95.34 ± 0.47 % | 4.67 % |
| **Vulkan**, the unpatched pin | 1.0005 ± 0.0048 | 0.01490 ± 0.00068 | 0.00516 | 95.74 ± 0.45 % | 4.61 % |

- **The patched CUDA backend drifts from the CPU no more than the unpatched Vulkan backend does.** Both PPL ratios are within
  their error bars of 1, and the KLD and top-token figures are equal within error. The drift is ordinary GPU half-precision
  arithmetic, not the patch.

**3. The Jetson as a CUDA RPC worker in a pooled run.**
- Qwen3-14B Q4_K_M, the engine's rule split `v3.qwen3_14b.rule` copied verbatim: Pi 5 35 (host) + Jetson 5 + the output
  layer. pp64 / tg16 × 3.
- `run_pool.sh` gained a per-cell `backend` (the CUDA tree, `-d CUDA0`). Its gate proves that tree is the pin plus exactly patch
  v2: HEAD, the patch sha, a reverse-apply, and the file set.
- The cells are in `testbed/dataset/pool_cells_20260929cuda.json`, which is **not a registration**: no prediction is under test.
- Run C-V-C 15:21–15:39. Every cell exited 0 and both Pis read `0x0`.

| cell | Jetson worker | prefill pp64 | decode tg16 |
|---|---|---|---|
| C1 `pool_cu.q14.p5j5.cuda_20260929_152111.log` | CUDA | 187.8 ms/token (0.54 %) | 1,024.3 ms/token (5.33 %: rep 1 cold) |
| V `pool_cu.q14.p5j5.vulkan_20260929_152721.log` | Vulkan | 185.8 (0.71 %) | **840.1** (0.24 %) |
| C2 `pool_cu.q14.p5j5.cuda_20260929_153320.log` | CUDA | 188.2 (0.34 %) | 996.7 (5.33 %: rep 1 cold) |

- **It works: the Jetson joins pooled runs as a CUDA worker at the pin.**
- **In this split Vulkan is the better worker backend.** Decode is 16–18 % faster, because the Jetson's CUDA decode is 1.87×
  slower. Prefill is equal within 1.3 %: with 5 of 40 layers on the Jetson, the Pi 5 host dominates prefill, so CUDA's standalone
  1.58× cannot show.
- **Both CUDA cells' first decode repetition is cold.** 0.942 and 0.971 t/s against 0.99–1.03 after, a 5.33 % spread, over the
  5 % STEADY bar. Reported as found. The Vulkan cell is within 2 % of this split's 2026-09-28 figures (823.8 / 183.8).

**4. The two-revision exception is retired.**
- Its own dissolving condition ("a CUDA-10.2-buildable ref dated after 2025-04-18") is met by 1–3.
- `cluster.json`: `llama.exceptions` is now `{}`. `_exceptions_note` records the retirement, the evidence, and the one obligation
  left: every Jetson CUDA figure names patch v2. `_pin_note` has an update line.
- `Jetson/config/node.env` and `.example`: `LLAMA_CPP_REF` is the pin, and their exception comments are rewritten. Note:
  `Jetson/scripts/20_build_llama_cpp.sh` with `BUILD_CUDA=1` does not apply the patch, so the CUDA build at the pin is
  `jetson_build_llama_cuda.sh`.
- All four suites are green (ThinkPad 217, Pi 5 357, Pi 4 158, Jetson 23). A mutation that puts b5050 back undeclared fails
  `test_every_pin_location_agrees_unless_the_split_is_declared`.

**Power, reported as found.** On micro-USB the input sagged to **4.84 V at 8.8 W (≈ 1.8 A)** at 14:02:34, during the Vulkan
benchmark. There were 346 one-second samples under 5.0 V since the guard started. The guard trips at 3 consecutive samples under
4.75 V, so the margin was 90 mV, and it never tripped. The barrel supply (J25 + J48) is still the fix. The fan was left alone,
as asked: PWM 0 throughout.

## The ThinkPad T580 characterised like the boards (user: "let's finish The ThinkPad as a prefill node")

**The rows.** `testbed/scripts/run_t580_rows.sh` on the laptop's pinned build (`/opt/heteroserve/llama.cpp`, b23701f: AVX2 FMA F16C,
NATIVE off, BLAS off).
- TinyLlama at the four quantisations every board has, the same bytes (each re-hashed). `-t 4`, the boards' protocol, at pp512 /
  tg128 and at pp64 / tg16.
- Taken twice: r3 at a 65 °C gate (16:03, 16:07), then r5 at a 60 °C gate (16:17, 16:24).
- The laptop has no `vcgencmd`, so its STATE lines carry cpu0's **Intel thermal-throttle event counters**, and `build_dataset.py`
  flags a row `throttled` when they rise.
- **Reported as found: the counters register a throttle a moment after the load.** 24 → 52 and 59 → 80/85 both appeared in the idle
  gap after the hottest rows, never between a row's own before and after. So a row's window runs to the next row's STATE_BEFORE, or
  a STATE_END taken 30 s after the last row. That is conservative: it can only add flags. 6 of 32 rows are flagged, all in the r5
  retake.
- Conditions: AC, power profile `performance`, EPP `performance`, turbo on. All were recorded; none was changed (the user's system
  settings).

| TinyLlama | pp512 t/s | tg128 t/s | pp64 t/s | tg16 t/s |
|---|---|---|---|---|
| Q4_K_M | 93.2 / 87.2 | 29.6 / 28.2 | 102.5 / *63.7 thr* | 34.7 / *28.2 thr* |
| Q8_0 | 61.8 / 61.5 | 18.1 / 18.6 | 79.6 / 78.1 | 21.8 / 20.7 |
| Q4_0 | 74.7 / *75.0 thr* | 30.0 / *28.0 thr* | 94.6 / 89.2 | 36.2 / 34.8 |
| Q3_K_M | 47.7 / *49.6 thr* | 26.9 / *27.5 thr* | 68.7 / 66.3 | 35.6 / 34.7 |

(Each cell is first run / retake; *thr* = flagged throttled.)
- **The laptop is the fastest decoder in the cluster.** TinyLlama Q4_K_M decodes at 29.6 tok/s, against the Pi 5's 18.1, and
  prefills at 93 against the Pi 5's 81.
- **It is also the noisiest node, and thermally limited.** Clean rows still spread 5–26 % within one invocation (turbo decaying to
  the 15 W sustained limit, which the counters do not see). Sustained runs throttle, reaching 83 °C.
- Its figures depend on how long it has been working: a prefill node that works in short bursts gets the first repetition's
  speed, and one under sustained load gets less.
- The Pis are steady at 0.1–0.5 % within a run.

## heteroctl prices per byte (user: "Yes, price per byte (Recommended)")

- `ThinkPad/control/profiles.py` now stores unit costs on each profile: ms per byte read (decode) and ms per block parameter
  (prefill), from the profiled model's GGUF geometry. `for_model()` turns them into the planned model's per-layer figures, the
  testbed predictor's convention. The engine already prices the head on the last node, so the profiled model reproduces its own
  ms/token exactly (a test pins this).
- **Qwen3-14B, the same split heteroctl ran at 12:25** (Pi 4 6 + Pi 5 34): predicted **976.0 ms/token** against **965.6 measured
  (+1.1 %)**; it was 141.6 (−85 %). This is a post-hoc check, not a registration.
- 5 new tests, mutations 6/6; suites 222 / 357 / 158 / 23. NOTE: `.handoff/nodes/ThinkPad/2026-09-29-1623-NOTE-per-byte-profiles.md`.
- Not ported: role-specific costs, SIZE2, batch-aware MoE prefill, the Jetson RPC factor.

## The Jetson and the qwen35 hybrids: memory, not an operation

> **Corrected 2026-09-29 evening — this section's conclusions are wrong; see "The Jetson and the qwen35 hybrids, corrected" below.**
> The smoke tests read `-ngl` as blocks. At the pin, `-ngl k` gives the workers k-1 blocks **plus the output head**. So "5 blocks"
> and "9 blocks" of the 27B were 4 and 8 blocks plus its 834 MiB head, and "`-ngl 17`" was 16 blocks plus the head (3,667 MiB, over
> the 3,539 free). The "1,321 MiB + 0.99 × blocks" fit, the "~1.25 GB optimistic" engine charge and the "~13-block limit" are
> withdrawn: against the real shares, the engine charged 238–248 MiB MORE than the drop. The 09-28 cell's worker died of a
> **segfault** (apport.log: signal 11), not OOM, and it is not resolved here. The measured table below stands as data, with its
> "blocks" column misread.

- **The 09-28 "crash" is resolved.** That cell (Pi 4 host) gave the Jetson 17 blocks of Qwen3.8-27B Q3_K_M, charged 3,030 MiB, 65 MiB
  over its reserve.
- **Smoke tests, laptop host** (`testbed/scripts/jetson_qwen35_smoke.sh`; no figure reported):
  - the 9B at `-ngl 6` and the 27B at `-ngl 5` ran clean on **both** the Vulkan and the CUDA worker;
  - the 27B at `-ngl 17` (3,000 MiB) killed **both** workers (EXIT 134).
  - dmesg: the OOM killer reaped `ggml-rpc-server`, and `systemd-journal` with it. The board recovered (guard alive, journald
    restarted, no failed unit). Swap was off, because the bench script's `swapoff` had never been undone.
- **Measured with memprobe.sh v2** (Vulkan worker, `memprobe_qwen35smoke_vulkan_*.log`):

| Jetson share | blocks | lowest MemAvailable (from ~3,550) | worker peak VmRSS / RssAnon |
|---|---|---|---|
| 5 blocks | 970 MiB | 1,278 MiB (drop 2,278) | 1,491 / 844 MiB |
| 9 blocks | 1,647 MiB | 600 MiB (drop 2,946) | 2,183 / 844 MiB |

- **The need is ≈ 1,321 MiB + 0.99 × the blocks.** At 17 blocks that is ≈ 4,280 MiB, against 3,546 free: it predicts the OOM. The
  engine's charge (blocks + 44 MiB worker overhead + one repaid tensor) is **~1.25 GB optimistic** for this hybrid on the Jetson.
  That accounting was validated on dense models, where its worst error was 110 MiB.
- **The Jetson's real 27B limit** is ≈ 2,250 MiB of blocks (~13 blocks), or ~10 blocks keeping the 450 MiB reserve.
- **Cause of the fixed 1.3 GB: not established.** A hint, not a finding: the worker's anonymous memory is 844 MiB at both shares,
  about the size of the 27B's output head (834 MiB), which should not be on the Jetson at `-ngl` below n_layer + 1.
- **Next:** the same probe with an ARM host, and a dense model, before the engine's Jetson charge is changed. Changing it would be a
  change to what the engine computes, so it goes to the user.

## A recurrent-layer term for the qwen35 hybrids, registered and tested (RECUR)

**Why.** The registered rule TRANSFER (2026-09-28) found the predictor does not transfer to the hybrids: they run 5–14 % slower in
decode than their bytes explain. The open item was a recurrent-layer cost term measured on its own rows, then registered.

**The rows** (`quant_campaign_20260929_161727`, `…_165219`): Qwen3.5-9B Q4_K_M on each Pi alone, `-t 4 -r 3`.

| board | pp512 | tg128 | pp64 | tg16 |
|---|---|---|---|---|
| Pi 5 | 8.83 t/s | 1.88 | 9.42 | 1.85 (540 ms/token) |
| Pi 4 | 1.59 *(flagged: soft limit `0x80000` during the 31-min row)* | 0.66 *(flagged)* | 1.61 | 0.66 (1,515 ms/token) |

The Pi 5 rows agree with its 2026-09-28 figures (106.3 / 544.1 ms) within 0.7 %.
- Both Pis had Wi-Fi connected for these rows, as they did not for earlier rows. That is a condition change, recorded.

**The term.** `predict_pool.py`'s new optional `recur` argument adds, for each recurrent block a node holds, a per-board cost:
(measured − predicted) for the board alone, over the 24 recurrent blocks. Recurrent blocks are marked as llama.cpp marks them at the
pin: `(i + 1) % full_attention_interval != 0`, and not the MTP block.
- Pi 5: decode +3.06 ms, prefill +0.44 ms per recurrent block per token.
- Pi 4: decode +1.37 ms, prefill +2.10 ms.

**The registration.** `testbed/dataset/pool_predictions_20260929recur.json`, 17:00:26, sha256 `c2a02499…`, read-only, by
`.handoff/assets/placement-2026-09-29/register_recur.py`.
- Cells: three fresh Pi 5 + Pi 4 splits of the same 9B, both host orders.
- Stated limitation: out of sample in splits, not in models.
- Rule RECUR: adopt if every cell has |decode error| ≤ 5 % and |prefill error| ≤ 10 % with the term.

| cell | decode, measured / predicted | with term (without) | prefill, measured / predicted | with term (without) |
|---|---|---|---|---|
| `r.q9.p5h24_p4` Pi 5 24 (host) + Pi 4 9 | 986.1 / 925.3 ms | **−6.2 %** (−12.6 %) | 243.9 / 235.2 ms | −3.6 % (−12.0 %) |
| `r.q9.p4h13_p5` Pi 4 13 (host) + Pi 5 20 | 962.2 / 869.1 | **−9.7 %** (−15.5 %) *not steady, 6.0 %* | 362.5 / 351.8 | −3.0 % (−10.5 %) |
| `r.q9.p5h12_p4` Pi 5 12 (host) + Pi 4 21 | 1,227.4 / 1,227.2 | −0.0 % (−3.9 %) | 433.6 / 426.7 | −1.6 % (−9.8 %) |

- **Registered verdict: do NOT adopt.** Two cells miss the ±5 % decode bar.
- **The term does most of the work anyway.** Mean |decode error| is 5.30 % with it against 10.67 % without. Mean |prefill error| is
  2.73 % against 10.77 %, every prefill within 3.6 %.
- A side check, not registered: h.q9.p5p4 (2026-09-28), which the term was not fitted on, comes to +0.7 % / −2.8 % with it and
  −5.0 % / −11.1 % without.
- **An exploratory reading.** The shortfall that remains is in decode, where the Pi 5 holds many blocks (24 as host, 20 as worker).
  With the Pi 4 holding most of the model as a worker, the cell lands exactly.
  - The term was fitted on each board as a host holding everything. A role-aware term (9B rows with each Pi as an RPC worker) is the
    next candidate, and it needs its own rows and registration.
  - The Pi 4-hosted cell's decode repetitions spread 6.0 % (1.009–1.071 t/s): reported as found.
- All three cells exited 0; the Pi 5 read `0x0`, and the Pi 4 kept the earlier sticky `0x80000`, with no new bits.

## The ThinkPad as the prefill node: laptop prefill → Pi decode (Qwen3.5-9B, 128–1024 tokens)

**The run.** `pd_handoff.py sweep --prefill t580 --decode pi4,pi5 --alone t580`, 17:14–18:02 (`pd_sweep_20260929_171413.log`,
server logs beside it).
- The laptop's llama-server runs on loopback, from the same pinned revision and the same model bytes, at `-t 4`.
- The state goes from its overlay address (192.168.10.5) directly into each Pi's tmpfs.
- Prompts are identical to the 01:08 sweep (the same GPL-3 text and question, and both Pis tokenize the same), so the Pi-alone
  arms come from that sweep.
- 36 of 36 arms, no make-up. The laptop's throttle counters rose 86/91 → 153/164 across the run, so it throttled on the longer
  prompts.

| prompt | laptop prefill | Pi 5 prefill (01:08) | laptop → Pi 4, decoding after | Pi 5 → Pi 4 (01:08) | Pi 4 alone | laptop decode alone | Pi 5 decode alone |
|---|---|---|---|---|---|---|---|
| 128 | 10.4–12.0 s | 13.71 s | **12.64 s** | 14.09 s | 80.01 s | 262.8 ms/token | 558.4 |
| 256 | 21.1–25.6 | 27.37 | **26.27** | 27.92 | 159.58 | 277.4 | 560.7 |
| 512 | 47.0–51.3 | 55.08 | **50.87** | 55.80 | 321.29 | 287.5 | 564.6 |
| 1024 | 96.9–103.4 | 111.50 | **100.81** | 112.24 | 648.48 | 279.5 | 578.4 |

(Laptop prefill: the range of the three arms' means, each over 3 repetitions. Within-arm spreads reach 15–31 s at 512–1024 tokens,
where it throttled.)

- **The hand-off works from x86 to ARM.** State files are byte-for-byte the size the Pi 5 makes (52,691,776 + 32,796 × n). The
  transfer takes 0.49–0.76 s at 113–116 MB/s, and every restore processed 1 prompt token. The Pi 4 decodes after it at 1,529–1,589
  ms/token, the Pi 5 at 529–550, as after their own prefill.
- **As a prefill node for the Pi 4, the laptop gets it decoding 6.1–6.4× sooner than the Pi 4 prefilling itself,** and 6–10 %
  sooner than the Pi 5 route.
- **But against the Pi 5, the laptop is a better decoder than a prefiller.** At 15 W it prefills this 9B only 1.1–1.3× faster than
  the Pi 5, and least so on the longest prompts, where it throttles. It decodes 2.0–2.1× faster (263–288 against 558–578 ms/token).
  - So between these two, the laptop's comparative advantage is decode. Against the Pi 4 it is the opposite: prefill 6.2–7.7×,
    decode 5.4–5.8×.
  - An exploratory reading, not a registered rule: for a laptop + Pi 5 pair the split that loses least has the Pi 5 prefill and the
    laptop decode; for a laptop + Pi 4 pair, the laptop prefills.
- **Greedy output across three CPU architectures, reported as found.** The laptop alone agrees with the Pi 4 alone at 128 and 512
  tokens, and with the Pi 5 alone at 256 and 1024. After a hand-off, the tokens follow either the decoding board or the laptop,
  depending on the prompt. Exact agreement is only guaranteed on the same board, as at 01:08.
- **Energy is out of scope.** The laptop is not on the DAQ, as `cluster.json` says: the report must say so.

## The Jetson and the qwen35 hybrids, corrected: a driver segfault above a weights cap, not memory (user: "let's finish placement Two optional refinements and dig deep")

**What was wrong at 16:30.** `jetson_qwen35_smoke.sh` read `-ngl` as blocks. At the pin, `-ngl k` gives the workers k-1 blocks and the
output head (`llama-model.cpp`: `i_gpu_start = n_layer_all + 1 - ngl`; the engine's `ModelProfile.head_bytes` and `bench_args` always
knew it). Against the shares the Jetson really held, the engine's own charge was *conservative*:

| smoke share (27B) | what the Jetson held | model buffer | engine charge | measured drop (from rest) |
|---|---|---|---|---|
| `-ngl 5` | blocks 61–64 + head | 1,637.3 MiB | 2,516.1 MiB | 2,278 MiB |
| `-ngl 9` | blocks 57–64 + head | 2,313.9 MiB | 3,194.0 MiB | 2,946 MiB |
| `-ngl 17` | blocks 49–64 + head | 3,667.2 MiB (over the 3,539 free: OOM at allocation) | 4,549.8 MiB (refused) | — |

The 844 MiB of anonymous memory in the worker was the head's own receive buffer (ggml-rpc receives each tensor whole into a
`std::vector`), which the engine charges as the "repaid" tensor.

**The 09-28 cell was a segfault.** Its Jetson held blocks 10–26 of the 27B and no head: 2,922.05 MiB, charged 3,030 against a
3,034 MiB drop, 515 MiB left. On the board, `/var/log/apport.log.1` records **signal 11** for that worker (pid 2459) at 23:23:22, and
`kern.log` has no OOM line. L4T 32.7.6 logs nothing for a segfault (`debug.exception-trace` 0, `ulimit -c` 0), which is why it looked
"silent". The MoE's 09-28 failure is signal 6 in the same log, matching its recorded Vulkan allocation error.

**Replayed and located** (`testbed/scripts/jetson_rpc_memdiag.sh`, laptop host; each worker under a wrapper that records its exit, and
`memprobe_tegra.sh`, which adds nvmap and CMA to the probe):
- the same split, Pi 5 as the tail: SIGSEGV (`WORKER_EXIT=139`, `jetson_memdiag_r1_replay0928_20260929_184451.log`);
- under gdb (`…_r2_replay0928_gdb_…`): the fault is in `libnvidia-eglcore.so.32.7.6`, inside `vkAllocateDescriptorSets`, called by
  ggml-vulkan's `ggml_pipeline_allocate_descriptor_sets` ← `ggml_vk_matmul` ← `ggml_vk_mul_mat_q_f16` ← `ggml_vk_build_graph`, at
  the first graph. The faulting load is from `x0 + 8`, `x0 = 0x28`. A NVIDIA driver fault, not a missing qwen35 operation;
- with a laptop-loopback tail (only the Jetson's share on the wire), blocks 10 onward at pp16:

| Jetson blocks | model buffer | lowest MemAvailable | NvMapMemUsed peak | outcome |
|---|---|---|---|---|
| 12 | 2,105.7 MiB | 1,332 MiB | 2,163 MiB | ran |
| 14 | 2,433.4 | 996 | 2,494 | ran |
| 15 | 2,596.3 | 831 | 2,660 | ran |
| 16 | 2,759.2 | 663 | 2,826 | ran |
| 17 | 2,922.1 | 497 | 2,989 | **SIGSEGV** (4 of 4, counting 09-28) |

**A registered test separated graph size from memory** (`testbed/dataset/jetson_graph_limit_20260929.json`, 19:11:55, `9478f2fe…`, and
`_addendum.json`, 19:18:34, `1165ae51…`, both read-only):

| cell | what varies | model buffer | lowest MemAvailable | outcome |
|---|---|---|---|---|
| m1: Qwen3.5-9B, blocks 10–26 | the same 17-block graph, much less weight | 2,214.6 MiB | 1,223 MiB | ran |
| m2: 27B, 16 blocks + a 160 MiB filler | the passing graph at the crash's memory | 2,759.2 | 501 | ran |
| g1: 27B, 16 blocks + a 260 MiB filler | the passing graph, below every crash | 2,759.2 | **423** | ran |

- By the first registration's rule, m1 and m2 support **neither** GRAPH nor MEMORY alone. m2 landed 4 MiB above the crash's minimum,
  so it could not exclude a memory edge at 497–501 MiB; the addendum's g1 did.
- **Registered verdict: GPU_CAP.** What separates every Jetson cell ever run is the size of its model buffer. Of the 25 pooled cells
  with a Jetson share (2026-09-27..29), the 23 that ran (21 on Vulkan, 2 on CUDA; six models, four architectures) all held at most
  **2,829.46 MiB** (the MoE's 10 blocks), and so did every qwen35 diagnostic that ran (up to 2,759 MiB). The two that failed held
  2,922.05 MiB (SIGSEGV) and 3,112.41 MiB (the MoE's 11 blocks, `ErrorOutOfDeviceMemory` at upload). Counting the compute buffers as well, the
  warm MoE cell's total (2,932 MiB) is within 3 MiB of the crash (2,935), so the cap is set on the model buffer, which separates by
  93 MiB.
- **The fix, as registered:** `cluster.json` jetson `max_weights_mb: 2845`. It caps blocks + a head the node holds, in every role. 2,845
  is the registered figure; it came from the MoE's average block, and the logs give that share as 2,829.46 MiB. Both are below the
  first failure (2,922).
  - The engine applies it in `memory_cap`, and the predictor reports the margin with real block sizes. It reproduces the boundary:
    27B Jetson 16 blocks +86 MiB, 17 blocks −77; MoE 10 blocks +16, 11 blocks −267.
  - `run_pool.sh` refuses a cell registered over the cap unless forced.
  - Suites: ThinkPad 227 (5 new), Pi 5 357, Pi 4 158, Jetson 23. Mutations 6/6.
  - NOTEs: `.handoff/nodes/ThinkPad/2026-09-29-1934-NOTE-jetson-weights-cap.md`, `.handoff/nodes/Jetson/2026-09-29-1935-NOTE-…`.
- **What it means for placement:** the Jetson **can** serve the qwen35 hybrids as a worker, up to 16 middle blocks of the 27B, where
  memory alone would have allowed the 17 that crash. The 450 MiB reserve's evidence is superseded (the MoE failure it cited is the
  cap); it is kept as headroom.
- **The memory accounting, checked once more:** on the middle shares above, the Jetson's drop was its model buffer + 88–106 MiB,
  against the engine's buffer + 44 + the largest block tensor (58 MiB) + KV.

**Caveats.**
- The driver's internal limit is not identified (a GPU address-space or nvmap ceiling are candidates). The cap is the envelope of
  what was observed, not a mechanism.
- The CUDA worker's own ceiling is not measured; the cap applies to it too.
- Every cell here is a one-repetition diagnostic: no throughput figure from them is reported.

**Side result, the RPC worker's default thread count.** `ggml-rpc-server` defaults to `hardware_concurrency()/2` threads, 2 on a Pi.
Every instrument passes `-t 4` (`run_pool.sh` and `serve_pool.sh` through `WEXTRA`, `bench_worker_role.sh`, `heteroctl_run_pis.sh`,
`run_14b_pooled.sh`), so no registered figure is affected. A first reading of `run_pool.sh` missed `WEXTRA` and said otherwise; it was
retracted within the hour. What omitting `-t` costs, measured A-B-A-B on the Pi 4 (`rpc_worker_threads_ab.sh`,
`worker_threads_pi4_20260929_184920`; TinyLlama Q4_K_M, loopback, `-r 3`):

| Pi 4 worker | pp64 | tg16 | worker threads (peak) |
|---|---|---|---|
| `-t 4` | 12.90 / 12.94 t/s | 5.66 / 5.71 t/s | 4 |
| default | 6.61 / 6.62 t/s (**−49 %**) | 5.55 / 5.53 t/s (−2.4 %) | 2 |

## The recurrent-layer term, role by role (RECUR2), and why decode cannot be held to ±5 % (user: "finish placement ... dig deep")

**The worker-role rows** (`bench_worker_role.sh`, `worker_role_20260929_190749`; Qwen3.5-9B Q4_K_M, each Pi as a loopback RPC worker
at `-t 4`, pp64 / tg16, `-r 3`, spreads 0.06–0.44 %):

| board | tg16 as host | tg16 as worker | pp64 as host | pp64 as worker |
|---|---|---|---|---|
| Pi 5 | 539.7 ms/token | 542.0 | 106.1 | **181.6** (1.71× slower: no repacked kernels) |
| Pi 4 | 1,515.4 | 1,546.7 | 622.9 | 634.2 |

The term per recurrent block, in each role (measured − the adopted predictor's price of the board alone in that role, over 24):
Pi 5 host +3.06 ms decode / +0.44 prefill, worker +3.90 / **+1.10**; Pi 4 host +1.37 / +2.10, worker +2.15 / +2.78.
In sample, on the four 9B splits already scored, the role-aware term put every prefill within 2 % (RECUR's was within 3.6 %) and left
decode where RECUR had it (−5.7 and −8.4 % on the same two cells).

**The same split does not decode the same twice.** Repeats under new ids (`pool_cells_20260929repeat.json`, not a registration):

| split | decode ms/token (prefill) |
|---|---|
| Pi 5 20 (host) + Pi 4 13 | **1,017.2** (307.6) on 2026-09-29 00:18; **1,112.8** (307.7), **1,127.3** (306.3), **1,047.9** (308.1) at 19:26–20:05 |
| Pi 5 24 (host) + Pi 4 9 | **986.1** (243.9) at 17:00; **931.4** (245.6) at 19:31 |

- Prefill repeats within 1 %. Decode moves 5–11 % between runs, and three of the four slow runs were unsteady within themselves
  (3–8 %).
- **Six causes were ruled out, each by measurement:**
  - **thermal clock:** the Pi 4 holds 1,800 MHz from 70 to 82 °C (`pi_thermal_clock_probe.sh`, `thermal_clock_pi4_20260929_194345`);
    its trip points drive only its fan;
  - **background CPU:** 0.67 % (Pi 4: Tailscale, the desktop panel) and 0.19 % (Pi 5) of the CPUs at idle;
  - **the link:** 1 Gbps full duplex on both, no link event since boot;
  - **TCP loss on the RPC path:** `tcp_rr_probe.py` sent a decode step's shapes (16 KB out, 993 KB of logits back) 900 times over the
    overlay: p99 9.0 ms, none over 100 ms, and not one retransmission. The Pi 4's 19,710 timeouts since boot are elsewhere;
  - **the governor:** a 10 Hz probe (`freqprobe.sh`, bash builtins) shows both Pis at full clock through decode;
  - **transparent huge pages:** neither kernel has them.
- The cause is not established. The consequence is: **a ±5 % decode bar per cell cannot be met reliably on Pi 5 + Pi 4 pipelines**,
  whatever term is fitted. That is why RECUR missed on decode while its prefill was fine.

**RECUR2, registered before any of its cells ran** (`testbed/dataset/pool_predictions_20260929role.json`, 20:11:11, `f103d68e…`,
read-only; `.handoff/assets/placement-2026-09-29/register_recur2.py`, scored by `score_recur2.py`):
- **ROLE-PRE:** adopt the role-aware prefill term if every cell is within ±5 % and its mean |error| is below RECUR's.
- **ROLE-DEC:** the same for decode, at **±10 %**, the bar set from the repeats above before any cell ran.

| cell | decode measured / role-aware | error: role (RECUR, none) | prefill measured / role-aware | error: role (RECUR, none) |
|---|---|---|---|---|
| `rr.q9.p5h16_p4` Pi 5 16 (host) + Pi 4 17 | 1,134.9 / 1,132.9 | **−0.2 %** (−1.0, −5.7) | 370.1 / 371.0 | **+0.2 %** (−1.9, −10.2) |
| `rr.q9.p4h8_p5` Pi 4 8 (host) + Pi 5 25 | 844.4 / 758.1 | **−10.2 %** (−12.0, −19.5), *not steady: 9.2 %* | 291.8 / 291.9 | **+0.0 %** (−4.0, −11.0) |
| `rr.q9.p4h20_p5` Pi 4 20 (host) + Pi 5 13 | 1,079.8 / 1,054.0 | **−2.4 %** (−3.1, −7.6) | 458.3 / 457.1 | **−0.3 %** (−1.5, −9.3) |

- **Mean |error|:** prefill **0.17 %** role-aware, 2.47 % RECUR, 10.17 % none; decode **4.27 %**, 5.37 %, 10.93 %.
- **Registered verdicts:**
  - **ROLE-PRE: adopt** the role-aware prefill term;
  - **ROLE-DEC: do not adopt** — one cell missed by 0.2 points, the unsteady one, in which the Pi 5 is the 25-block worker.
- **In the predictor:** `predict_pool.predict(recur=, recur_worker=)` and `recurrent_terms()`. Both reproduce RECUR's and RECUR2's
  registered predictions exactly. The adopted form passes only each term's `"pre"`. Earlier calls are unchanged.
- **Reported as found:** `compare_pool.py` marks the first cell "pre-registered=False". The registration was written in the same
  second the run began (20:11:11), by a command that registers and only then runs, so the order is certain but the one-second
  timestamps cannot show it.
- All three cells exited 0; the Pi 5 read `0x0`; the Pi 4 kept its sticky `0x80000` (it held full clock throughout, above).
- **Limitation, as registered:** fitted on the 9B alone and tested on 9B splits. It is out of sample in splits, not in models.

## Prefill/decode separation, step 1: what a second user costs each board (batched decode)

**Why first.** "Whether a second sequence costs a memory-bound board almost nothing sets how much the Pi 5 alone can already serve"
(remaining work, item 6). `testbed/scripts/run_batched_decode.sh` ran `llama-batched-bench` on each board alone:
- Qwen3.5-9B Q4_K_M (sha256 checked on each node), 128-token prompts, 32 generated tokens per sequence, 1, 2 or 4 sequences at
  once, prompts not shared, `-t 4`;
- the whole bench three times, since it has no repetitions of its own;
- cold gates, and `get_throttled` plus the ARM clock (Pis) or the Intel throttle counters (laptop) around every round;
- logs: `batched_decode_20260929_202511` (all three boards), `…_211526` (the Pi 4's round cut off when the session ended, retaken),
  `…_211529` (the Pi 5 `--no-repack`).

| board | 1 sequence: step (per-user ms/token) | 2 sequences: step | 4 sequences: step | total decode t/s at 1 / 2 / 4 | prefill t/s |
|---|---|---|---|---|---|
| Pi 5 (default: repacked) | 571.9 | **1,086.5 (×1.90)** | 940.8 (×1.64) | 1.75 / **1.84** / 4.25 | 8.86–8.99 |
| Pi 5 `--no-repack` | **500.1** | **641.9 (×1.28)** | 1,158.0 (×2.32) | 2.00 / 3.12 / 3.45 | 5.69–5.72 |
| Pi 4 | 1,536.3 | 1,811.1 (×1.18) | — | 0.65 / 1.10 / — | 1.60 |
| laptop | 338.8 | 527.8 (×1.56) | 823.5 (×2.43) | 2.95 / 3.79 / 4.86 | 7.6–9.1 |

(Round-to-round spreads: Pi 5 ≤ 1.8 %, Pi 4 ≤ 1.0 %, the laptop 12 %, its usual turbo noise. Pi 4 at 4 sequences not taken: four
128-token prompts are 5 minutes of prefill a round.)

- **The Pi 4 behaves as a memory-bound board should:** a second sequence costs 18 % of a step, so two users get 1.7× the decode
  throughput.
- **The Pi 5, as it has been run, does not.** Its repacked kernels (`CPU_REPACK`, `q4_K_8x4`) make a 2-sequence step cost 90 % more
  than a 1-sequence one. Two users get only 5 % more throughput than one, and each decodes at half speed. Four sequences are cheaper
  per step than two (940.8 against 1,086.5 ms).
- **Without repacking** (`--no-repack`), the same board decodes one sequence 13 % faster (500.1 ms) and two for 28 % more, like
  the Pi 4. It prefills 1.55× slower. On this model at the pin, repacking is a **prefill** optimisation that costs decode. The same
  Pi 5 wants a different configuration for each phase: prefill/decode separation inside one board.
- **Not confirmed with `llama-bench`:** it has no switch for this at the pin. `-ot '.*=CPU'` still allocated `CPU_REPACK`
  (5,052.83 MiB in `-v`), so the A-B (`pi5_repack_ab_20260929_213130.log`) is void and not reported. Its four runs are repacked
  repeats, whose decode drifted 553.7 → 572.4 ms/token in 8 minutes: the known warm-up drift. The two-user test's `shared_nr` arm
  checks it in `llama-server`, which does take `--no-repack`.
- **For placement:** the Pi 5 host rows the predictor prices decode from are the repacked configuration. A Pi 5 used only to decode
  would be about 13 % faster than they say.
- **Decided 2026-09-30 (user): the engine is left alone for now.** Home use has little concurrency, and continuous batching is not
  this study's contribution, as it is not prima.cpp's. The engine keeps the repacked Pi 5, as llama.cpp runs it by default.
  - The condition that goes with the rule, derived from the rows above (exploratory; the 9B, 128-token prompts):
    - For one user, `--no-repack` costs 63 ms more per prompt token (175.3 against 112.0) and saves 72 ms per generated token.
    - So it takes less total time only when the reply is longer than about **0.9× the prompt**.
    - To first order this does not depend on how many blocks the Pi 5 holds, since both terms scale with its share.
  - Home chat prompts that carry history are usually longer than the replies, which is the case the repacked default serves.

## Prefill/decode separation, step 2: two users at once — the hand-off earns its place (TWOUSER, QR4)

**The test.** `testbed/scripts/pd_twouser.py run --reps 3 --n 128 --gen 32 --arms shared,handoff,static,shared_nr --stagger 0`,
22:00–22:23 (`pd_twouser_20260929_215703.log`, both servers' logs beside it).
- Two users, each a different 128-token prompt (the GPL-3 text from token 0 and from token 2000, then the same question), both
  waiting at t = 0, 32 greedy tokens each. Every token was streamed and timestamped on the laptop.
- Both boards ran 2-slot servers (`-np 2 -c 8192`). A 1-slot server refuses a 2-slot server's saved state (`n_stream mismatch`,
  found by the smoke run).
- Arms, interleaved per repetition:
  - **shared:** the Pi 5 serves both, as it has always run (repacked);
  - **handoff:** the Pi 5 prefills user 2 first, and the state goes straight into the Pi 4's RAM, which decodes user 2 (prompt + t1);
    the Pi 5 then serves user 1 whole;
  - **static:** a board per user;
  - **shared_nr:** as shared, with the Pi 5's server `--no-repack` (restarted for it, and back).

**Registered before any of it ran** (`testbed/dataset/twouser_predictions_20260929.json`, 21:57:03, `1985f56a…`, read-only;
`.handoff/assets/pd-2026-09-29/register_twouser.py`, scored by `score_twouser.py` into `twouser_results_20260929.json`). Every prediction
is priced from single-board rows measured before: the 128-token sweep of 01:08 (each board alone, and the hand-off's parts), and what a
second sequence costs the Pi 5 (the batched-decode ratios above).

| arm (mean of 3) | user 1: first token / done | user 2: first token / done | mean done: measured / predicted | makespan | mean per-token gap |
|---|---|---|---|---|---|
| shared (Pi 5, repacked) | 27.8 / 61.5 s | 27.8 / 61.5 s | **61.45** / 59.85 s (−2.2 %) | 61.5 s | 1,086 ms |
| **handoff** (Pi 5 prefill → Pi 4 decode for user 2) | 28.1 / 46.0 s | **13.7** / 64.1 s | **55.06** / 53.67 s (−2.5 %) | 64.1 s | 572 / 1,625 ms |
| static (a board per user) | 13.8 / 31.6 s | 80.4 / 127.9 s | **79.74** / 79.01 s (−0.9 %) | 127.9 s | 575 / 1,531 ms |
| shared_nr (Pi 5, `--no-repack`) | 45.1 / 65.1 s | 45.1 / 65.1 s | **65.13** / 62.05 s (−4.5 %) | 65.1 s | **648 ms** |

**The registered verdicts, applied as written:**
- **HANDOFF holds.** The hand-off's mean completion (55.06 s) is below both shared arms (61.45 s repacked, 65.13 s without): 10.4 %
  and 15.5 % less. Its mean first-token time is 20.9 s against 27.8 (−25 %). Its makespan is 4 % longer (64.1 against 61.5 s),
  because the Pi 4 decodes user 2 slowly.
- **NOREPACK holds.** Without repacking, the Pi 5's two-user decode is 40 % faster per token (648 against 1,086 ms), in
  `llama-server` as in `llama-batched-bench`. Its 1.6× slower prefill still makes it the slowest shared arm at 32-token replies.
- **ACCURACY holds on all four arms** (errors −0.9 to −4.5 %, all optimistic). The two-user outcome was predicted from single-board
  measurements alone.
- **STATIC-CHECK holds.** The 2-slot servers served one user as the sweep's 1-slot ones did: the Pi 5 +0.5 % prefill / +3.1 % decode,
  the Pi 4 +0.5 % / +1.0 %.
- **STEADY:** every arm's repetitions agree within 1.9 %.

**Integrity of every hand-off:** the Pi 4 processed exactly 1 prompt token after the restore and restored all 128 cells. The state was
56,889,668 bytes: 4 bytes more than a 1-slot server's (52,691,776 + 32,796 × 128), the 2-slot header. The transfer took 0.50–0.52 s.
0 failed arms.

**What it answers (QR4).**
- For two users at once, handing the second user's decode to the Pi 4 **pays**: both users are done sooner on average, and the
  second user's first token comes twice as early (13.7 against 27.8 s).
- It pays because of what the batched decode found. The Pi 5 **as it has been run** gives a second user almost no extra decode
  throughput, so moving one user's decode off it is nearly free for the other user.
- The static assignment (each user on their own board) is worst on every aggregate: the Pi 4 prefills 5.8× slower than the Pi 5.
- **Exploratory, not tested:** the same model predicts the answer reverses for longer replies. At 128 tokens each, the shared Pi 5
  without repacking would finish both users in ~122 s on average against the hand-off's ~153, because the Pi 4's slow decode then
  dominates. That is the next registration.

**Caveats that travel with it.** One prompt length, one reply length, one arrival pattern (together). The laptop orchestrates and
timestamps; it computes nothing. No energy figure: the rig does not exist yet.

## Prefill/decode separation, step 3: longer replies reverse the answer (TWOUSER128b, QR4)

**The test.** TWOUSER's design with one change: **128 tokens back per user** instead of 32 (`pd_twouser.py run --reps 3 --n 128
--gen 128 --arms shared,handoff,static,shared_nr --stagger 0`, 22:52–23:40, `pd_twouser_20260929_225221.log`). The model priced it
from the same pre-TWOUSER single-board rows, so this tests the model at a new reply length.

**A first run was stopped, reported as found.**
- Registration `twouser_predictions_20260929_gen128.json` (22:40:32, `b52227c7…`), run `pd_twouser_20260929_224035.log`: stopped at
  22:48 after two arms. User 1's prompt (a three-sentence summary) ended its answer at **61 tokens** (`stop_type eos`) while user 2's
  reached the 128 limit. The arms still did equal work, but not the work the predictions priced. At 32 tokens nobody reached an
  end-of-answer.
- The tool now sets `ignore_eos`. It was checked on the laptop with user 1's own prompt: 60 predicted tokens and `eos` without it,
  80 and `limit` with it.
- **Re-registered before the run** (`twouser_predictions_20260929_gen128b.json`, 22:52:19, `c7fd3007…`, read-only; it records what
  it supersedes). The run began 2 s later.
- Also reported: the command's automatic scoring step read the newest `pd_twouser_*.log`, which was the Pi 4's server log. The run
  was re-scored on its own log; the result file holds that score.

| arm (mean of 3; 128 tokens each, verified) | user 1: first / done | user 2: first / done | mean done: measured / predicted | makespan |
|---|---|---|---|---|
| **shared_nr** (Pi 5, `--no-repack`) | 44.5 / 126.6 s | 45.4 / 127.0 s | **126.81** / 122.20 s (−3.5 %) | 127.0 s |
| handoff | 27.1 / 96.9 s | 13.3 / 210.5 s | **153.71** / 153.22 s (−0.3 %) | 210.5 s |
| shared (Pi 5, repacked) | 27.1 / 158.8 s | 27.1 / 158.8 s | **158.80** / 161.68 s (+1.7 %) | 158.8 s |
| static (a board per user) | 13.4 / 83.1 s | 80.1 / 274.9 s | **179.01** / 178.56 s (−0.2 %) | 274.9 s |

**The registered verdicts, applied as written:**
- **REVERSAL holds.** The Pi 5 alone, without repacking, finishes two users 17.5 % sooner on average than the hand-off (126.8 against
  153.7 s), and its makespan is 40 % shorter (127.0 against 210.5 s).
- **HANDOFF does not hold, as the model predicted.** The hand-off still beats the Pi 5 as it has always run (repacked) by 3.2 %, but
  not the no-repack configuration.
- **NOREPACK holds:** a two-user decode step of 645 against 1,037 ms.
- **ACCURACY holds on all four arms**, within −3.5 to +1.7 %. The alternative scaled by each arm's TWOUSER miss did no better
  (0.7–4.5 %).
- **STATIC-CHECK holds:** the Pi 5 −2.5 % prefill / −1.7 % decode, the Pi 4 +0.1 % / +1.2 %.
- **STEADY:** every arm within 1.7 %.
- **Hand-off integrity:** 3 of 3 processed exactly 1 prompt token and restored 128 cells, state 56,889,668 bytes, transfer
  0.50–0.52 s. 0 failed arms.

**What steps 2 and 3 answer together (QR4).**
- Whether the hand-off earns its place depends on the reply length.
- **Short replies (32 tokens):** the hand-off wins, 10–16 % lower mean completion. The two users' prefills contend for the Pi 5, and
  the hand-off's second user starts decoding early.
- **Long replies (128 tokens):** the Pi 5 alone wins, *if* it runs without repacking. User 2's 127 tokens on the Pi 4 at 1.53 s each
  outweigh the Pi 5's prefill contention.
- **Exploratory, not measured:** the crossover, from the same model, lies near 50–60 generated tokens per user.
- **What decides it is how each board decodes, not the hand-off itself.** The hand-off costs under 4 s, and it was the same in both
  tests. The Pi 5's repacked kernels, which cost a second user 90 % of a step, are why the hand-off also beats the repacked Pi 5 at
  both lengths.
- **The predictive side of the study holds.** Eight arm-by-length outcomes were predicted from single-board rows, all within 4.5 %.
  Two-user serving on this cluster can be planned without running it.

**Caveats.** One prompt length (128), both users arriving together, one model. The laptop orchestrates and timestamps only. No energy
figure.

## The engine's placement against llama.cpp's own default split (DEFAULT) (user: "run the llama.cpp default split comparison")

**Why.** The study needs a baseline, and the honest one is what llama.cpp gives on this cluster with no placement rule. Given `-ngl 99`
and no `--tensor-split`, llama.cpp splits the layers across its devices in proportion to the free memory each one reports
(`llama-model.cpp` at the pin, "default split, by free memory"; `tools/rpc/README.md`).
- A CPU `ggml-rpc-server` reports the board's whole RAM as free: Pi 5 8,062 MiB, Pi 4 7,819.
- The Jetson's Vulkan server reports its free GPU heap: 3,289–3,292 MiB in these runs, 3,250–3,591 seen before.
- A CPU-only host is not a split device. Its own cores compute nothing unless it also runs a worker.
- So the default hands out blocks by memory, not by speed. The Pi 4 reports almost as much as the Pi 5.

**Registered before any cell ran** (`testbed/dataset/pool_predictions_20260930default.json`, 00:16:34, `a4aad01c…`, read-only).
- Written by `.handoff/assets/placement-2026-09-30/register_default.py`; scored by `score_default.py`.
- Setup: Qwen3-14B Q4_K_M; the Pi 5 hosts; `llama-bench -t 4 -p 64 -n 16 -r 3 -v` through `run_pool.sh` (memprobe v2 on every board).
- `run_pool.sh` gained one ability for this: a host that is also a worker, probed as both processes.

The three cells:
- **`d.q3_14b.rule`, the engine's plan** (`v3.qwen3_14b.rule`, 2026-09-28): the Pi 5 host holds 35 blocks, the Jetson 5 and the head
  (`-ngl 6`).
- **`d.q3_14b.default`, llama.cpp's default with every board a device** (`--rpc pi5,pi4,jetson -ngl 99`; the Pi 5 hosts and runs its
  own worker).
  - `llama_split()` replicates llama.cpp's formula. It gives 17/17/6, 18/16/6 or 18/17/5, depending on the free memory the Jetson
    reports.
  - Each variant was priced, and the cell is scored on the one llama.cpp actually made, read from its `-v` layer assignment.
- **`d.q3_14b.readme`, the RPC README verbatim** (`--rpc jetson,pi4 -ngl 99`; the Pi 5 hosts and computes nothing). The possible splits
  are 12/28, 13/27 and 14/26. It was priced at 13/27, and only that split has a prediction.
- **Not run: `readme_pi4_first`** (`--rpc pi4,jetson`). The engine's accounting gives the Jetson 11 blocks and the head. That is
  2,720.5 MiB of weights (under the 2,845 cap), but a **−282 MiB** margin after its 450 MiB reserve. The cell was registered as
  infeasible, with its figures recorded untested.

The registered rules:
- **DEFAULT:** the engine's placement wins if its decode ms/token is below both llama.cpp cells'. Prefill is compared the same way.
- **ACCURACY:** every cell within ±10 % of the prediction for the split llama.cpp made. The bar is 10 % in both phases because a host
  that is also a worker is untested.
- **SPLIT:** each llama.cpp cell's actual split must be one that was registered.
- **STEADY:** three repetitions within 5 %.

| cell | split llama.cpp made (read from `-v`) | decode ms/token: measured / predicted | prefill ms/token: measured / predicted | spread (decode / prefill) |
|---|---|---|---|---|
| **rule** | Pi 5 host CPU 35 + Jetson 5 + head | **834.4** (1.20 t/s) / 847.1 (+1.5 %) | **184.8** (5.41 t/s) / 182.7 (−1.1 %) | 0.13 / 0.23 % |
| default | Pi 5 worker 18 + Pi 4 16 + Jetson 6 + head (Jetson reported 3,290 MiB: the 18/16/6 variant) | **1,453.1** (0.69 t/s) / 1,445.1 (−0.6 %) | **602.0** (1.66 t/s) / 592.6 (−1.6 %) | 2.76 / 0.12 % |
| readme | Jetson 13 + Pi 4 27 + head (Jetson reported 3,289 MiB) | **1,992.9** (0.50 t/s) / 2,049.3 (+2.8 %) | **802.1** (1.25 t/s) / 790.0 (−1.5 %) | 1.74 / 0.17 % |

**The registered verdicts, applied as written:**
- **DEFAULT holds.**
  - Decode: the engine's placement takes **42.6 %** less time per token than llama.cpp's default split, and **58.1 %** less than the
    README setup. That is 1.74× and 2.39× the tokens per second.
  - Prefill: **69.3 %** and **77.0 %** less (3.26× and 4.34×).
- **ACCURACY holds on all three cells**, within −1.6 to +2.8 %. The predictor priced llama.cpp's own splits as well as its own plan.
- **SPLIT holds:** the default made the registered 18/16/6, and the README cell the priced 13/27.
- **STEADY holds:** every phase within 2.76 %.
- **Reproducibility:** the rule's cell repeats its 2026-09-28 measurement (823.8 / 183.8 ms) within +1.3 % in decode and +0.6 % in
  prefill.
- All three cells exited 0. The Pi 5 read `0x0`. The Pi 4 kept its sticky `0x80000`, with no new bits, and peaked at 63.3 °C at cell
  end. No clock probe ran in these cells.

**Memory, measured** (lowest MemAvailable per board):

| cell | Pi 5 | Pi 4 | Jetson |
|---|---|---|---|
| rule | **277 MiB** (registered margin: 228) | — | 1,295 MiB |
| default | 3,412 MiB (its worker's RssAnon 3,449) | 4,381 MiB | 1,095 MiB |
| readme | 6,586 MiB | 1,617 MiB | 959 MiB |

The engine's plan runs the Pi 5 host within 277 MiB of full, as its accounting said it would (within 49 MiB). The default leaves
memory unused on every Pi.

**Why the default loses, as the predictor prices it** (a decomposition of the registered predictions, not a new measurement; the
predictor matched all three cells within 3 %):
- Per block, the Pi 4 is the slowest board by far:

  | board | decode, ms per block | prefill, ms per block |
  |---|---|---|
  | Pi 4 | 55.1–55.6 | 26.9–27.2 |
  | Jetson | 27.3–28.0 | 4.7 |
  | Pi 5, host | 17.3 | 4.5 |
  | Pi 5, worker | 16.7 | 7.4 |

  The Pi 5 as a worker runs without the repacked kernels, which is why its prefill is slower.
- The default split gives the Pi 4 16 of the 40 blocks because it reports 7,819 MiB free. Those blocks are **869 of the 1,445 ms**
  predicted per decoded token (60 %) and **431 of the 593 ms** of prefill (73 %).
- In the README cell the Pi 4 holds 27 blocks: **1,676 of 2,049 ms** of decode (82 %).
- The engine leaves the Pi 4 out entirely. It gives the Pi 5 host as many blocks as its memory allows (35), and gives the Jetson the
  remaining 5 and the head.
- The links are not what separates the cells. Their largest share is the Jetson's 6.9 ms per token, which is the logits on the last
  hop.

**Caveats.**
- One model (Qwen3-14B, dense), one prompt length (64 tokens), a 16-token decode, one host.
- The baseline is llama.cpp *as configured by its defaults and README*. A user who hand-tunes `--tensor-split` could do better than
  either llama.cpp cell. Finding that split is what the engine does.
- No energy figure.

## QR3: time per output token at the 95th percentile, one user, on the engine's serving plan (user: "yes demote them, then start QR3")

**What QR3 asks.** The proposal: "TPOT P95 ≤ 500 ms (MVP) / ≤ 250 ms (Stretch)", with TPOT taken as decode streams the tokens
back to the client. No figure existed, because `llama-bench` gives only a mean per repetition.

**The instrument** (`testbed/scripts/qr3_tpot.py`, new).
- **Serving:** `serve_pool.sh` plans each model with the engine for a 4096-token serving context. It gates each board's memory and
  the Jetson's guard, starts `ggml-rpc-server` on the workers and `llama-server` on the host (`-np 1 -t 4`), and stops what it
  started.
- **Before the requests:** the model file is SHA-256-checked on its host. One unscored warm-up follows (a 512-token prompt, 32
  tokens).
- **The requests:** 10, one after another. Each has its own 128-token prompt (GPL-3 from token 500 × i, then a question) and returns
  128 tokens: greedy, `ignore_eos`, `cache_prompt` false.
- **Every token is timed twice:**
  - on the laptop, as its SSE event arrives. This is the primary figure: what a user on the LAN sees;
  - by `llama-server`'s own clock (`timings_per_token`: `predicted_ms`, stamped right after each token is sampled, in µs).
- **The statistics:**
  - A TPOT sample is the gap between two consecutive tokens of one request: 127 per request, 1,270 per config.
  - P95 is taken by linear interpolation. The 95 % CI comes from a bootstrap over whole requests (10,000 resamples), because tokens
    of one request are not independent.
- **Between requests:** `get_throttled`, temperature and MemAvailable on each Pi; the Jetson's guard line and clocks. No probe runs
  while scored tokens are timed.

**Two unscored pilots, and what they changed** (`qr3_pilot_qwen3_14b_*`):
1. **01:28: the memory gate refused the 14B's plan** (Pi 5 33 + Jetson 7).
   - The Pi 5 had 7,435 MiB available and the plan needed 7,530. The engine had planned against `memory_at_rest.json`: 7,619 MiB,
     read 2026-09-27 19:05.
   - A dated snapshot (`memory_snapshot_20260930_013353.json`, the lowest of 10 reads, by the new `memory_snapshot.py`) has the
     **Pi 5 127 MiB below its 09-27 figure**: 7,492 MiB. The Pi 4 (7,403) and the Jetson (3,540) are within 10 MiB of theirs.
   - Cause not established. It is not `/dev/shm` (empty), and not the page-cache watermarks (~70 MiB, charged whether the cache is
     full or not). Both Pis have had Wi-Fi connected since 09-29.
   - **Explained at 04:40, measured.** The Pi 5 was not rebooted (up since 09-29 11:40), but its page cache had been emptied, and
     MemAvailable read **7,616 MiB**, within 3 MiB of 09-27's (`memory_snapshot_20260930_043954.json`).
   - Reading the 14B's file (9.0 GB, 109 s) back into the cache took it to **7,518** (Cached 663 → 7,336 MiB): the reading drops
     ~100 MiB with a full page cache. That is MemAvailable's accounting of a warm cache, not a leak and not the Wi-Fi.
   - So the at-rest figures (cold cache) run ~100 MiB above what the gate reads once models are cached. A plan with less margin
     than that can be refused at serving time, which is why the dated snapshot (taken warm) is the right thing to plan a serving
     start against.
   - The at-rest file is left unchanged, because every earlier registration hashes it. `serve_pool.sh` gained `MEMFILE=` to plan
     against a snapshot; the default is unchanged.
   - Reading a model file lowers the host's MemAvailable for a while (7,435 MiB just after the SHA-256 pass, 7,492 settled), so the
     tool now waits for it to settle.
2. **01:35: the snapshot plan (Pi 5 32 + Jetson 8) ran end to end.**
   - The client and server gaps agreed within 0.2 ms.
   - The lowest MemAvailable was 620 MiB on the Pi 5 and 1,045 on the Jetson. The engine charges the whole 4096-token KV cache; a
     short chat touches only part of it.
   - But decode ran at **~1,050 ms/token against 878.5 predicted**. The cause: `serve_pool.sh` leaves the Jetson's clocks to its
     governors, and nvhost_podgov idled the GPU at 76.8 of 921.6 MHz. `run_pool.sh` locks them for every benchmark cell, and those
     cells are the rows the predictor is built on.
   - `qr3_tpot.py --clocks locked` now uses `run_pool.sh`'s own commands: stored, locked, and restored after.

**Registered before any scored request.**
- **`qr3_predictions_20260930.json`** (01:31:06, `56c0f10a…`) fixed the definitions, the rules and the predicted verdicts. It was
  drafted while the first pilot ran and registered after that pilot had failed, before its output was read.
- **`qr3_predictions_20260930b.json`** (01:50:27, `8fd7dadd…`) supersedes its plans. It adds the snapshot plans, locked clocks, a
  cool-host gate (Pi 5 < 50 °C, Pi 4 < 57 °C), and a secondary config: the 14B with the clocks as found. The requirement,
  definitions, rules and predicted verdicts are copied verbatim from the first.
- Run by `qr3_chain.sh`, 01:50–04:07.

The registered rules:
- **QR3-MVP:** the client P95 is ≤ 500 ms. "Met" if the whole CI is below 500.
- **QR3-STRETCH:** the same at 250 ms.
- **TAIL:** P95 / mean ≤ 1.10.
- **DELIVERY:** the client and server P95 within 2 %.
- **ACCURACY:** the server mean within ±10 % of the predictor.
- **PLAN**, **COMPLETE** and **PRIMARY** (the verdicts are taken on the client samples).

| config (the engine's plan at 4096) | mean TPOT | P50 | **P95** [95 % CI] | P99 | max | P95 / mean | TTFT, 128-token prompt | predicted mean (error) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-9B, Pi 5 alone | 537.8 ms (1.86 t/s) | 536.9 | **541.3** [540.6, 542.9] | 554.5 | 577.4 | 1.007 | 13.5 s | 466.3 (−13.3 %) |
| Mistral-Nemo 12B, Pi 5 39 + Jetson 1 | 668.3 (1.50 t/s) | 667.2 | **672.1** [671.4, 672.7] | 679.6 | 925.5 | 1.006 | 19.4 s | 665.9 (−0.4 %) |
| Qwen3-14B, Pi 5 32 + Jetson 8 | 851.6 (1.17 t/s) | 850.4 | **854.9** [854.6, 856.5] | 865.4 | 1,102.7 | 1.004 | 23.1 s | 878.5 (+3.2 %) |
| Qwen3-30B-A3B MoE, Pi 4 15 + Jetson 9 + Pi 5 24 | 302.4 (3.31 t/s) | 298.7 | **331.5** [324.2, 340.8] | 369.4 | 387.3 | 1.096 | 23.7 s | 273.7 (−9.5 %) |
| *secondary:* the 14B, Jetson clocks as found | 1,053.7 (0.95 t/s) | 1,041.1 | **1,172.8** [1,162.4, 1,194.5] | 1,325.3 | 1,749.3 | 1.113 | 23.5 s | 878.5 (−16.6 %) |

**The registered verdicts, applied as written** (`qr3_results_20260930.json`, by `score_qr3.py`):
- **QR3-MVP is met by the MoE alone**, as predicted. Its whole CI lies below 500 ms. The 9B, Nemo and the 14B do not meet it:
  their P95 is 541–855 ms.
- **QR3-STRETCH is met by none**, as predicted. The MoE's P95 is 331.5 ms against 250.
- **TAIL holds on all four primary configs:**
  - P95 within 0.4–0.7 % of the mean for the dense models and the 9B;
  - 9.6 % for the MoE, whose first decode step after the prompt averages 363 ms against 302.
  - It fails on the secondary: 11.3 %.
- **DELIVERY holds everywhere:** the client and server P95 are within 0.04 %. The LAN and HTTP add nothing measurable.
- **ACCURACY:**
  - holds for Nemo (−0.4 %), the 14B (+3.2 %) and the MoE (−9.5 %, its known −9.2 % bias);
  - fails for the 9B (−13.3 %). That is the hybrid's known bias: the predictor has no adopted recurrent decode term. The Pi 5's own
    `llama-bench` row of the model (539.7 ms) is within 0.4 %;
  - fails for the secondary (−16.6 %).
- **PLAN, COMPLETE and PRIMARY hold:** 10 × 128 tokens everywhere, and 0 merged events.
- **The runs:** all exited 0. The Pi 5 read `0x0`, and the Pi 4 had only its sticky `0x80000`. The guard recorded 0 alarms, and the
  Jetson's clocks were restored after each locked config.

**What it shows.**
- **For a dense model, QR3 is QR2.** Per-token time is so steady (P95 within 0.7 % of the mean) that the P95 target is a mean-speed
  target. No dense model, and not the 9B, reaches 500 ms: the fastest, the 9B on the Pi 5 alone, is at 541.
- **The MoE is the only model that meets QR3 (MVP), and the only one that meets QR2** (3.31 t/s). Its tail is wider, but well
  inside 500 ms.
- **Serving matches benchmarking when the conditions match.** With the Jetson locked, `llama-server`'s TPOT agrees with the
  predictor within 3.2 % on the dense models. The 9B at a 4096 context matches its `llama-bench` row within 0.4 %.
- **The Jetson's governors cost 24 % of mean TPOT and 37 % at the P95** (the 14B: 1,053.7 / 1,172.8 ms against 851.6 / 854.9). A
  serving start that includes the Jetson should lock its clocks.
  - **Decided 2026-09-30 (user: "yes, lock the Jetson clocks in serve_pool.sh").** `serve_pool.sh` now locks them for the session
    with `run_pool.sh`'s own commands, and `stop` restores them. It restores only what its start locked, and only in MAXN.
    `CLOCKS=asfound` keeps the old behaviour.
  - Checked on a Nemo start: the clocks were locked, one request decoded at 664.0 ms/token (the QR3 mean, locked, is 668.3), and
    `stop` restored them.
  - Both scripts changed after the campaign. The registration's `inputs_sha256` are of the versions that ran, and those versions
    are kept byte-exact, their hashes matching the registration: `.handoff/assets/qr3-2026-09-30/serve_pool.as_run_20260930.sh`
    and `qr3_tpot.as_run_20260930.py`. Today's `qr3_tpot.py --clocks locked` reaches the same condition through `serve_pool.sh`'s
    lock, which happens before the load instead of after it.
- **TTFT for a 128-token prompt is 13.5–23.7 s.** It has no requirement, but for a short reply it is most of the wait.
- **The requirements' tension, reported as found** (CLAUDE.md: take it to the advisor). The models that need the cluster (≥ 13B
  dense) meet QR1 and miss QR2 and QR3. The one model that meets QR2 and QR3 is the mixture of experts.

**Caveats.**
- One prompt length (128), 128-token replies, one user, a 4096 context. A longer conversation fills more of the KV cache and would
  raise TPOT a little.
- Each config is one server session of 10 requests. The proposal's full evaluation asks for at least 5 independent runs.
- No energy figure.

## The full evaluation's prima.cpp comparison (PRIMA) (user: "download the prima.cpp", "after downloaded do the comparison in the full evaluation", "i insist to go with qwen3-14B")

**Why.** The proposal's Phase 5 asks for a "comparison against prima.cpp under identical conditions". prima.cpp (ICLR 2026) is the
closest prior work: distributed llama.cpp-based inference on home clusters of mixed devices, with its own scheduler, Halda.

**What it took to run it** (`testbed/prima.md`).
- **Source:** OpenCPIL/prima.cpp @ `288df6a2`, the address the paper's v3 gives, MIT. Its late-2024 llama.cpp base has no Qwen3
  architecture, and its distributed path supports only llama and qwen2.
- **The Qwen3 port** (`testbed/patches/prima-288df6a-qwen3-v1.patch`): prima.cpp's own qwen2 code, with upstream's per-head Q/K RMS
  norms and no Q/K/V biases. Its scheduler's profiler was taught the two new tensors; otherwise it stops counting early.
- **Built on both Pis.** The user installed the ZeroMQ headers. HiGHS v1.15.1 (official prebuilt, SHA-256 matching GitHub's digest)
  is linked into the Pi 5, prima.cpp's head device.
- **The Jetson is out:** prima.cpp cannot build there (gcc 7.5, ZeroMQ 4.2.5, no internet). The comparison is therefore **Pi 5 +
  Pi 4**, the boards both systems can use.
- **The model** was copied to the Pi 4 and verified (513 s). prima.cpp loads a local copy on every device.

**Registered before prima.cpp's first run** (`testbed/dataset/prima_predictions_20260930.json`, 05:05:28, `f074fd61…`; scored by
`.handoff/assets/prima-2026-09-30/score_prima.py` into `prima_results_20260930.json`).
- **Common to both arms:** the QR3 protocol and client (`qr3_tpot.py`: a 512/32 warm-up, then 10 requests of 128-token prompts × 128
  tokens, greedy, timed per token on the laptop), a 4096-token context, `-t 4`, the wired overlay, and a cool Pi 5 (< 50 °C).
- **A:** the engine's plan on the Pi 5 + Pi 4, served by llama.cpp at the pin (`serve_pool.sh NODES="pi5 pi4"`): Pi 5 32 + Pi 4 8,
  with the head on the Pi 4. Predicted at 1,193.5 ms/token.
- **B:** prima.cpp as its README runs it. The Pi 5 is rank 0 (`llama-server`, the head, which also holds the input and output
  layers), the Pi 4 is rank 1, with `--prefetch` and no `-lw`, so Halda profiles both boards and picks the split.
- **D (reference, not re-run):** QR3's 14B on the whole cluster, the engine's Pi 5 32 + Jetson 8.
- An **unscored smoke** run of B came first. Its first launch hung on an ssh detail in the new launcher (fixed, and the orphaned
  worker stopped).

| arm | split | mean TPOT | P50 | **P95** [95 % CI] | P99 | TTFT (128-token prompt) | Pi 5 lowest MemAvailable |
|---|---|---|---|---|---|---|---|
| **A: engine + llama.cpp, Pi 5 + Pi 4** | Pi 5 32, Pi 4 8 + head | **1,188.6 ms** (0.84 t/s) | 1,184.8 | **1,216.1** [1,207.8, 1,221.5] | 1,240.0 | **45.2 s** | 296 MiB |
| B: prima.cpp + Halda, Pi 5 + Pi 4 | Pi 5 24 + head + embedding, Pi 4 16 | 1,339.3 (0.75 t/s) | 1,335.4 | 1,356.1 [1,354.9, 1,357.3] | 1,538.9 | 81.6 s | 6,859 MiB |
| *D: engine + llama.cpp, whole cluster (QR3)* | Pi 5 32, Jetson 8 + head | 851.6 (1.17 t/s) | 850.4 | 854.9 [854.6, 856.5] | 865.4 | 23.1 s | — |

**The registered verdicts, applied as written:**
- **CORRECTNESS holds.** prima.cpp's greedy output on request 0 matches llama.cpp's for **all 128 tokens** of B's scored run (16 of
  16 in the smoke). The port computes what upstream computes.
- **DECODE: A is faster, as predicted.** 11.2 % less time per token than prima.cpp (1,188.6 against 1,339.3 ms; B is 1.13× A).
- **TTFT: A is faster, as predicted.** 44.6 % less (45.2 against 81.6 s).
- **QR3:** neither arm meets 500 ms, as predicted (P95 1,216 and 1,356).
- **HALDA-SPLIT holds:** 24 blocks on the Pi 5 (≤ 30), the same split at the smoke run and at B.
- **ACCURACY holds for A:** the server mean is 1,188.7 against 1,193.5 predicted (+0.4 %), on the plan as registered.
- **COMPLETE:** 10 × 128 tokens in both arms (B: `tokens_predicted` 128 every time), 0 merged events.
- **The runs:** both Pis read `0x0` throughout (the Pi 4 was rebooted at 04:22, so its sticky flag is cleared). Peaks were 44.4 °C on
  the Pi 5 and 62.3 °C on the Pi 4.
- **REFERENCE:** with the Jetson, which prima.cpp cannot use, the engine's plan is 36.4 % faster per token than prima.cpp, and 71.7 %
  faster to first token.

**Why prima.cpp is slower here** (exploratory: from its own log and our cost model, not a registered rule).
- **Halda measured the boards correctly.** Its profile rates the Pi 4 3.8× slower in Q4_K compute (23.5 against 89.8 GFLOPS) and
  3.4× slower in RAM bandwidth (8.25 against 28.41 GB/s).
- **Its split is memory-bound.** 24 is exactly the most blocks that fit the Pi 5 budget Halda profiled (5.85 GiB), once the output
  layer (609 MiB), the input embedding (417 MiB) and a 4096-token KV cache are counted: 24 blocks need 5,919 MiB, 25 need 6,112.
  - prima.cpp's ring pins the output layer and the embedding to its head device, and its budget is well under the ~7.4 GiB the Pi 5
    has free.
  - So the slow Pi 4 gets 16 blocks instead of the engine's 8. Both schedulers fill the fastest board to its memory limit. They
    differ in what they must put on it, and in how much memory they will use.
- **The runtime is not what separates them.** Our cost model prices Halda's split at 1,348–1,366 ms/token against the measured
  1,339, and the engine's at 1,172–1,193 against 1,188.6. Prefill prices at 608 ms/token for Halda's split, with the Pi 5 unrepacked
  as prima.cpp runs it (77.8 s per 128 tokens against 81.6 measured). The prefill gap also carries the Pi 5's repacked kernels, which
  prima.cpp's base lacks.
- **The trade-off from its paper's table, measured.**
  - prima.cpp kept ~6.9 GB "available" on both boards, because its weights are file-backed page cache: the low memory stress it
    designs for (devices also in daily use).
  - The engine ran the Pi 5 with 296 MiB free, by design, on a board dedicated to the job.
  - On this cluster that buys 11 % in decode and 45 % in time to first token.

**Caveats.**
- Two boards: the Jetson cannot run prima.cpp. One model (Qwen3-14B), which needed our port; one prompt length; one user; 10
  requests per arm.
- prima.cpp's strengths are models larger than total memory (disk offload with prefetch), CUDA GPUs and many devices, and none of
  them applies here.
- Its server exposes no per-token server timings, so B has client-side samples only. In QR3 the client and server P95 agreed within
  0.04 %.
- No energy figure.

---

# 2026-10-07 — Halda on our runtime (offline): the engine against prima.cpp's scheduler, and what that changed in the engine

**Why** (user: "enhance our scheduler by aiming to compete with prima.cpp + halda scheduler directly"; scope chosen: "same
runtime + plans", offline, no downloads). The 2026-09-30 head-to-head ("PRIMA") compared two whole systems:

- the engine with llama.cpp: 1,188.6 ms/token;
- prima.cpp with its Halda scheduler: 1,339.3 ms/token.

This section compares the two **schedulers** on the same executor and the same inputs, across every model and every board set
with measured links.

**Halda, re-implemented** (`ThinkPad/control/halda.py`, from prima.cpp @ 288df6a2: `common/common.cpp` 860–1650 and
`src/llama.cpp` 22243–22410).
- **What it is.** Per device, an integer program over the layers (HiGHS) that minimises one per-token time. That time is built
  from FLOPS micro-benchmarks: its RAM-read term is commented out (line 921), so decode is priced as compute. Its network term is
  0 (line 971).
- **Fixed structure.** The ring order is fixed by rank. The head device (rank 0) holds the output layer and an embedding row.
  Every device holds at least one layer, and every device left with exactly one is dropped before re-solving.
- **Memory.** Each device is charged its layers, each with a full-context KV cache, plus its compute buffer, against the memory
  available when Halda profiled it.
- **Validated against Halda's one measured decision.** Fed the profile prima.cpp logged on 2026-09-30, the port gives:
  - a 368.0 MiB compute buffer, exactly prima.cpp's;
  - caps of 24 / 26;
  - windows **[24, 16]**, Halda's decision;
  - an objective within 0.5 ms of the logged 689.9 (`ThinkPad/tests/test_halda.py`).

**The comparison** (`testbed/scripts/compare_halda.py` → `testbed/dataset/halda_offline_20261007.json`; reads files only).
- **Halda's inputs, the most favourable it could have:**
  - our measured per-layer decode cost for each board, in the host role;
  - each board's MemAvailable at rest. On 09-30 prima.cpp itself profiled only 5.85 GiB free on the Pi 5, against 7.4 GiB at
    rest.
  - every ring order, with the host first.
- **Common to both:** a 4096-token context.
- **Pricing:** both plans are priced by the registrations' predictor (`predict_pool.predict`), with Halda's output layer on the
  host, where Halda puts it.
- **Memory:** both are checked against the engine's validated accounting (`predict_pool.memory`), including the Jetson's
  measured reserve and GPU cap.
- **Coverage:** the laptop is out, because no laptop↔board link is measured, and a link is never borrowed. The laptop holds
  every model alone in any case.

## Results: decode, predicted (ms/token)

| model | boards (host first) | engine: plan | decode ms | head last | Halda's rule: plan | decode ms | Halda vs engine |
|---|---|---|---:|---:|---|---:|---:|
| qwen14b | pi5 + pi4 + jetson | pi5 38 + jetson 10 (head on the host) | 841.4 | 871.8 | pi5 38 + jetson 10 | 841.4 | +0.0 % |
| qwen14b | pi5 + pi4 | pi5 38 + pi4 10 (head on the host) | 1078.1 | 1172.8 | pi5 38 + pi4 10 | 1078.1 | +0.0 % |
| qwen14b | pi5 + jetson | pi5 38 + jetson 10 (head on the host) | 841.4 | 871.8 | pi5 38 + jetson 10 | 841.4 | +0.0 % |
| qwen14b | pi4 + jetson | pi4 31 + jetson 17 (head on the host) | 1992.1 | 2063.4 | pi4 30 + jetson 18: **does not fit** the Jetson's measured limits | (1970.9) | would fail |
| tinyllama | pi5 + pi4 + jetson | pi5 22 | 55.2 | 55.2 | pi5 22 | 55.2 | +0.0 % |
| tinyllama | pi5 + pi4 | pi5 22 | 55.2 | 55.2 | pi5 22 | 55.2 | +0.0 % |
| tinyllama | pi5 + jetson | pi5 22 | 55.2 | 55.2 | pi5 22 | 55.2 | +0.0 % |
| tinyllama | pi4 + jetson | pi4 1 + jetson 21 | 95.4 | 95.4 | pi4 1 + jetson 21 | 101.7 | +6.6 % |
| mistral7b | pi5 + pi4 + jetson | pi5 30 + jetson 2 (head on the host) | 647.6 | 656.3 | **none**: drops one-layer boards, then needs disk offload | - | cannot plan |
| mistral7b | pi5 + pi4 | pi5 30 + pi4 2 (head on the host) | 709.8 | 737.3 | pi5 30 + pi4 2 | 709.8 | +0.0 % |
| mistral7b | pi5 + jetson | pi5 30 + jetson 2 (head on the host) | 647.6 | 656.3 | pi5 30 + jetson 2 | 647.6 | +0.0 % |
| mistral7b | pi4 + jetson | pi4 20 + jetson 12 (head on the host) | 1686.0 | 1699.5 | pi4 19 + jetson 13: **does not fit** the Jetson's measured limits | (1654.6) | would fail |
| nemo12b | pi5 + pi4 + jetson | pi5 39 + jetson 1 (head on the host) | 631.6 | 665.9 | **none**: drops one-layer boards, then needs disk offload | - | cannot plan |
| nemo12b | pi5 + pi4 | pi5 39 + pi4 1 (head on the host) | 655.1 | 765.7 | **none**: drops one-layer boards, then needs disk offload | - | cannot plan |
| nemo12b | pi5 + jetson | pi5 39 + jetson 1 (head on the host) | 631.6 | 665.9 | **none**: drops one-layer boards, then needs disk offload | - | cannot plan |
| nemo12b | pi4 + jetson | pi4 23 + jetson 17 (head on the host) | 1593.3 | 1653.7 | pi4 22 + jetson 18: **does not fit** the Jetson's measured limits | (1569.5) | would fail |
| qwen3_14b | pi5 + pi4 + jetson | pi5 32 + jetson 8 (head on the host) | 838.7 | 868.4 | pi5 32 + jetson 8 | 838.7 | +0.0 % |
| qwen3_14b | pi5 + pi4 | pi5 32 + pi4 8 (head on the host) | 1065.2 | 1157.6 | pi5 32 + pi4 8 | 1065.2 | +0.0 % |
| qwen3_14b | pi5 + jetson | pi5 32 + jetson 8 (head on the host) | 838.7 | 868.4 | pi5 32 + jetson 8 | 838.7 | +0.0 % |
| qwen3_14b | pi4 + jetson | pi4 26 + jetson 14 (head on the host) | 2000.5 | 2077.3 | pi4 25 + jetson 15: **does not fit** the Jetson's measured limits | (1971.7) | would fail |
| qwen3moe ¹ | pi4 + pi5 + jetson | pi4 15 + jetson 9 + pi5 24 | 273.7 | 273.7 | pi4 12 + pi5 25 + jetson 11: **does not fit** the Jetson's measured limits | (311.9) | would fail |
| qwen3moe ¹ | pi4 + pi5 | pi4 24 + pi5 24 | 282.2 | 282.2 | pi4 23 + pi5 25 | 322.6 | +14.3 % |
| qwen38_27b ¹ | pi4 + pi5 + jetson | pi4 11 + pi5 39 + jetson 15 (head on the host) | 2101.4 | 2133.1 | pi4 9 + pi5 39 + jetson 17: **does not fit** the Jetson's measured limits | (2078.6) | would fail |
| qwen38_27b ¹ | pi4 + pi5 | pi4 26 + pi5 39 (head on the host) | 2227.0 | 2231.4 | pi4 26 + pi5 39 | 2227.0 | +0.0 % |
| qwen35_9b ¹ | pi5 + pi4 + jetson | pi5 33 | 466.3 | 466.3 | pi5 33 | 466.3 | +0.0 % |
| qwen35_9b ¹ | pi5 + pi4 | pi5 33 | 466.3 | 466.3 | pi5 33 | 466.3 | +0.0 % |
| qwen35_9b ¹ | pi5 + jetson | pi5 33 | 466.3 | 466.3 | pi5 33 | 466.3 | +0.0 % |
| qwen35_9b ¹ | pi4 + jetson | pi4 14 + jetson 19 (head on the host) | 1137.8 | 1218.5 | pi4 11 + jetson 22: **does not fit** the Jetson's measured limits | (1083.3) | would fail |

¹ prima.cpp cannot run this architecture (its distributed path takes llama and qwen2, plus qwen3 with our port), so Halda's
rule is applied hypothetically. The full set, with prefill, three request workloads and 0 / 1 / 2 GiB headroom, is in the
JSON.

## What it found

**1. The engine had a blind spot: where the output head goes.** The engine always put the output layer on the last board,
where llama.cpp's `-ngl` sends it. Halda keeps it on the host.
- **Why it matters.** The head is read on every decoded token. For Qwen3-14B it is 638 MB: about 53 ms on a Pi 5 host, against
  about 168 ms on a Pi 4 at the end of the pipeline.
- **The size of the gap.** With the Pi 5 hosting, Halda's layout beats the engine's old one by **3.4–8.1 %** (Qwen3-14B,
  Qwen2.5-14B, Mistral-7B, Nemo 12B).
- **Why 09-30 still went our way.** Halda's run-time budget was smaller, so it could put only 24 blocks on the Pi 5.

**The fix:** `plan_optimal(head_placement="best")`. For every order the engine now tries the head on the last board and on the
host, and keeps the cheaper. At the pin the host placement is `-ot "output\.weight=CPU"`; the pin's `llama-bench` and server
accept it.
- **Gains, predicted:**
  - Nemo 12B on Pi 5 + Pi 4: **−14.4 %** per token (765.7 → 655.1 ms);
  - Qwen3-14B and Qwen2.5-14B on Pi 5 + Pi 4: −8 %;
  - Qwen3-14B on Pi 4 + Jetson: **−3.7 %** (2,077.3 → 2,000.5). There, moving the head off the Jetson frees room for 6 more
    blocks on it.
- **Checks:**
  - on 1,600 random clusters, `"best"` is never slower, never infeasible where `"last"` is feasible, and strictly faster in
    8 %;
  - the default stays `"last"` until a measured run checks the host placement on llama.cpp.

**2. With that fix, the engine matches or beats Halda's rule in every case Halda can run.**
- **It ties wherever Halda's best ring fits and the host is the fastest board.** The engine now finds the same layout.
- **It wins where Halda's fixed rules hurt:**
  - a slow host still keeps the head: TinyLlama on Pi 4 + Jetson, **+6.6 %** for Halda (the engine puts the head on the faster
    Jetson);
  - the MoE: **+14.0–14.3 %**.
- **It plans where Halda cannot:**
  - **One-layer drops, then disk.** Halda drops every one-layer board at once, and the host alone cannot hold the rest:
    Mistral-7B on three boards, and Nemo 12B on every Pi 5 set. prima.cpp would then read the overflow from the SD card on
    every token, at 0.04–0.07 GB/s.
  - **The Jetson's limits.** Halda knows nothing of the Jetson's measured Vulkan reserve and GPU cap. On Pi 4 + Jetson, all of
    its rings for the dense models put 13–22 blocks there and would fail at load, as the first MoE cell did on 09-28. The engine
    plans within the limits.

**3. Planning for the request, not one phase, changes little here.** `plan_optimal(phase={"prefill": P, "decode": G})`
minimises P × prefill + G × decode. On these boards the decode-optimal plan is usually also request-optimal. Against Halda, the
gains are:
- +1–2 % for 1,024-token prompts on Pi 5 + Pi 4;
- +3–6 % on TinyLlama (Pi 4 + Jetson);
- +1–12 % on the MoE (chat +8 %, 1,024-token prompts +1 %, 512-token answers +12 %).
It matters most when one board is much better at one phase: the Pi 5 repacks only as host.

**4. Headroom has a price, and Halda pays it worse.** Keeping memory free on every board (`with_headroom`, prima.cpp's
"devices stay usable" design point) costs the engine, on Qwen3-14B with three boards:
- **+19 %** per token with 1 GiB kept free;
- **+58 %** with 2 GiB.
At the same headroom, Halda often has no ring that fits at all (its compute buffer and full-context KV charge more than the
engine's measured overheads).

**The exception.** Qwen3.5-9B at 2 GiB of headroom, where Halda's ring is predicted 2–6 % faster than the engine's. The engine
sizes caps by the average layer. This model ends in a 247 MiB multi-token-prediction block that inflates that average, so the
planner believes the Pi 5 host holds only 29 of the smaller early layers, when 30 fit. The engine's own estimate of Halda's plan
is lower than of its own, so the planner simply cannot reach it.
- **Proposed fix:** an exact fill over real per-layer sizes, a dynamic program over layer boundaries, as the predictor already
  uses.
- **Not done yet.** It only bites on hybrids, which prima.cpp cannot run, and only at 2 GiB of headroom.

## Caveats

- **These are predictions, not measurements.** The predictor is the one validated against measured pools (decode within −11 to
  +0.3 % on 2026-09-28), but **the head-on-host layout has never been run on llama.cpp.** It was measured only inside
  prima.cpp, on 09-30, where our cost model priced it within 2 %.
- **Halda was given our measured costs and the full memory at rest**, both better than prima.cpp's own run-time view. This is the
  rule at its best, not prima.cpp as it ran.
- **The boards are the 2026-09-27/28 set.** The Pi 5 was returned on 2026-10-05.

## The next step: a pre-registered head-to-head on today's boards (Pi 4 + Jetson, the laptop as control)

| cell | what it tests | predicted decode |
|---|---|---|
| Qwen3-14B, engine, head last (Pi 4 32 + Jetson 8) | the old layout | 2,077.3 ms |
| Qwen3-14B, engine, head on the host (Pi 4 26 + Jetson 14) | the fix, at the pin with `-ot` | 2,000.5 ms (−3.7 %) |
| TinyLlama, engine (Pi 4 1 + Jetson 21, head on the Jetson) | a slow host, head last | 95.4 ms |
| TinyLlama, Halda's layout (Pi 4 1 + Jetson 21, head on the Pi 4) | Halda's head rule on a slow host | 101.7 ms (+6.6 %) |

Halda's own rings for the dense models on Pi 4 + Jetson are predicted not to fit, so they are not cells. The Jetson needs its
power guard back first, and its SD card is the suspect in three hangs.

---

# 2026-10-07 (evening) — the head-to-head on the boards, the exact fill, and sequential vs piped ring

User's answers at pickup:
- run on the Jetson's current SD card;
- build the exact fill now;
- "sequential" means our pipeline against prima.cpp's piped ring.

## The head-to-head, run (Pi 4 host + Jetson, pre-registered)

**Setup.** The Jetson's power guard was restarted per its NOTE (clock synchronized, the script's sha256 matching the repo's,
then started; 0 trips all session). Both models were re-hashed on the Pi 4 (`OK`), and the cells ran through `run_pool.sh`
(`-t 4 -p 64 -n 16 -r 3 -oe jsonl -v`, memory probe, every gate).

**The first registration is void.** `pool_predictions_20261007h2h.json` (16:33:28) passed `-ot "output\.weight=CPU"`.
llama.cpp matches `-ot` with `std::regex_search` (`llama-model-loader.cpp:1239` at the pin), so the unanchored pattern also
matched every `blk.N.attn_output.weight`: 23 tensors on TinyLlama, not 1. Each block's attention projection was moved to the
host. The two cells it touched are kept in the record and not counted:
- `x.tiny.halda` measured 166.0 ms, against 101.7 predicted;
- `x.q3_14b.engine_host` was stopped mid-cell, once its log showed `blk.36.attn_output.weight … overridden to CPU`.

The fix is the anchor, `-ot "^output\.weight=CPU"`, re-registered as `pool_predictions_20261007h2h_v2.json` (16:43:21,
`e139ae16…`; same plans and predictions). Every v2 head-on-host cell logged exactly one override, `output.weight`. The source
also shows that an override to CPU still goes through `select_weight_buft`, so a repacking host would get its repacked head.
The earlier belief that it lands in plain "CPU" was wrong; it does not matter on the Pi 4, which never repacks.

| cell (v2) | pipeline (blocks) | head on | decode ms, measured / predicted | prefill ms, measured / predicted |
|---|---|---|---|---|
| TinyLlama, engine | Pi 4 1 + Jetson 21 | Jetson | 95.6 / 95.4 (−0.2 %) | 19.4 / 16.9 |
| TinyLlama, Halda's layout | Pi 4 1 + Jetson 21 | **Pi 4** (`-ot`) | **101.7 / 101.7 (0.0 %)** | 19.3 / 17.1 |
| TinyLlama, engine (repeat) | Pi 4 1 + Jetson 21 | Jetson | 95.9 / 95.4 | 19.3 / 16.9 |
| **Qwen3-14B, engine, head on the host** | **Pi 4 26 + Jetson 14** | **Pi 4** (`-ot`) | **1,926.3 / 2,000.5** (predicted 3.9 % high) | **803.6 / 773.5** |
| Qwen3-14B, engine, head last (llama.cpp's default) | Pi 4 32 + Jetson 8 | Jetson | 2,043.5 / 2,077.3 (1.7 % high) | 931.7 / 907.3 |
| Qwen3-14B, head on the host (repeat) | Pi 4 26 + Jetson 14 | Pi 4 | 1,941.8 / 2,000.5 | 802.7 / 773.5 |

Every split matched the plan to 0.1 MiB, with the head where it was registered: on the Halda cell the Jetson's buffer is
523.14 MiB, the engine's 574.41 less the 51.27 MiB head. The Pis read `get_throttled 0x0` throughout.

**What the head-to-head found:**
- **Where the head goes is a real lever, and the engine chooses it correctly on both models.**
  - **Qwen3-14B:** putting the head on the host frees 609 MiB on the Jetson, which takes 6 more blocks. Decode is **5.4 %
    faster** (1,934.1 ms, the mean of the A-B-A pair, against 2,043.5; predicted 3.7 %), and prefill **13.8 % faster**
    (predicted 14.7 %). The repeat drifted 0.8 %, so the gap is real.
  - **TinyLlama:** the head belongs on the faster Jetson. Halda's rule (head on the host, here the slow Pi 4) costs **6.2 %**
    (predicted 6.6 %).
- **Halda's own 14B plan (Pi 4 25 + Jetson 15) was not run.** It fails the engine's validated memory accounting on the Jetson
  by 126 MiB. On these boards, the engine with `head_placement="best"` is the measured best, and Halda's head rule is right
  only when the host is the faster board.
- **The predictions held:**
  - decode within −0.5 to +3.9 % on all six cells (the 14B pessimistic by 1.7–3.9 %);
  - prefill optimistic by 2.6–3.7 % on the 14B, and by 11–13 % on TinyLlama (the pp64-vs-pp512 batch effect seen before).
- **Thermal flag.** The Pi 4 read `get_throttled 0x80000` after the first 14B cell: bit 19, the soft temperature limit
  occurred, and it stays set until reboot. All three 14B cells ran the same profile (55–59 °C at the start, 68–69 °C at the
  end), so their comparison holds, but their absolute times may include soft-limit capping. The TinyLlama cells read `0x0`.
  `run_pool.sh` takes snapshots only; polling bit 3 during a cell would certify it.
- **The Jetson's guard** recorded 0 trips (minimum 4,832 mV, peak 9.2 W). It was stopped after the runs, as its NOTE keeps
  it, to spare the suspect card.

## The exact per-layer fill (built: `placement.plan_optimal(fill="exact")`)

**What it is.** The average fill caps each board at its budget over the AVERAGE block and fills greedily. The exact fill takes
each block's real bytes (`ModelProfile.layer_bytes`, from the tensor table) and chooses the boundaries between the boards'
contiguous runs with a dynamic programme. It uses the same `charged_bytes`, reserve, weights cap and objective. It is
**opt-in**: the default stays `"average"`, so every registration reproduces.

**Checks:**
- **The default is unchanged:** a differential against the pre-edit `placement.py` over 1,600 random clusters × 2 phases × λ
  {0, 0.01} × 3 head placements gave 19,200 identical plans.
- **On 3,200 random models with unequal blocks:**
  - every exact plan passes a real-bytes memory check (2,481 of 2,481);
  - it is never slower than an average plan that is valid;
  - with equal blocks it equals the average fill (3,200 of 3,200).
- **5 unit tests** (`TestExactFill`); the suite went from 243 to 248.

**What it found:**
- **The average fill over-commits.** In 296 of those 3,200 random cases, its plan does not fit the board's real blocks: an
  OOM the accounting did not see. On the real models, Qwen3.5-9B on Pi 4 + Jetson at 2 GiB headroom is one such case: the
  average plan (27 + 6) is 104 MiB over, and the exact plan (28 + 5, 48 MiB to spare) is the valid one.
- **The engine's one miss against Halda is fixed.** For Qwen3.5-9B at 2 GiB headroom, the exact fill gives 483.8 / 524.5 ms,
  equal to Halda's best fitting plan, where the average fill was 1.7–5.7 % behind.
- **9 of 32 decode plans improve** at 0 headroom, by up to 3.4 %. For example, Qwen3-14B on Pi 5 + Pi 4 goes from 1065.2 to
  1029.3 ms, with one more block on the host.

**Caveat, and why it is not the default.** The exact fill packs each board to the accounting's edge: the plans it changes
have 4–37 MiB of margin. The accounting has been off by a few hundred MiB before, so the exact fill should go with a
keep-free headroom (`with_headroom`), and be measured once before it becomes the default.

## Sequential vs prima.cpp's piped ring (`testbed/scripts/ring_vs_sequential.py`)

The question, from `meeting.txt`: "try sequential: is it faster?".

- **Our runtime is sequential:** each token visits each board once (k = 1).
- **prima.cpp's piped ring visits the ring k times**, each board holding k windows. Halda prices the network as zero
  (`common.cpp:971`), so in memory every k ties and it keeps k = 1; its 09-30 log shows the ties. The ring exists to overlap
  disk prefetch with compute when the model does not fit.

**In memory, sequential is fastest.** On 22 pooled plans, each extra ring pass costs +0.6–2.4 ms per token with our link
terms, or +2.2–4.4 ms with the measured RPC visit (2.2 ms, smoke test 09-27):
- 0.1–0.35 % on the dense 7–27B models;
- 1.6 % on the MoE;
- 2.3 % on TinyLlama.

A single stream gives the ring nothing to overlap.

**Not in memory, the SD card decides.** Streaming the overflow per token from the measured SD reads (Pi 5 93.2, Pi 4 44.7 MB/s)
costs 10–69 s per token. Those are lower bounds, since the Jetson's card is unmeasured, and they compare with 0.3–2 s of
compute. A perfect overlap hides at most the compute, so ring and sequential are both unusable. An MoE reads only its active
experts, so its figure is about 10× too high, and still far above its compute. **The answer for these boards: plan in memory,
sequentially, and add memory or a board rather than stream.**

A measured piped-ring run needs prima.cpp's head, which was the Pi 5 (returned 2026-10-05). The Jetson cannot build prima.cpp.

## The exact fill becomes the default, with a 256 MiB headroom (user's decision, 2026-10-07 ~17:40)

**What changed in `placement.plan_optimal`:**
- `fill="auto"` is the default: the exact fill when the model carries its block sizes, the average fill when it does not.
- With the exact fill, every board keeps `EXACT_FILL_HEADROOM_MB` = **256 MiB** free on top of its own reserve.
  `headroom_mb` overrides it; 0 turns it off.
- `fill="average"` reproduces every plan made before.
- Both model builders now pass the real block sizes (`profiles.model_profile` from the GGUF, `predict_pool.model_profile`
  from `model_shapes.json`). The live planner and every **new** registration therefore plan this way. Registrations on file
  are unaffected, and the scripts that reproduce them pin `fill="average"`.

**Why 256 MiB.** Across the 83 board-cells measured with the memory probe since the board-aware accounting (09-28 to 10-07),
the accounting **under**-estimated a board's MemAvailable drop 6 times, by at most **133 MiB** (then 117, 77 and 76). 256
MiB is about twice the worst measured miss. The exact fill alone packed boards to 4–37 MiB.

**Checks:**
- **Models without block sizes plan exactly as before:** 19,200 of 19,200 plans identical against the pre-change file.
- **Models with them get exactly "the exact fill under a 256 MiB headroom":** 19,200 of 19,200.
- 4 new tests (`TestDefaultFill`); the suite went from 248 to 252. All four suites are OK.

**The trade, from the offline comparison's new `engine_default` column** (n_ctx 4096; the published columns are unchanged):

| plan | average fill (before) | default (exact + 256 MiB) |
|---|---|---|
| Qwen3-14B, Pi 5 + Pi 4 | Pi 5 32 + Pi 4 8, 1,065.2 ms | Pi 5 31 + Pi 4 9, 1,105.8 ms (+3.8 %), 429 MiB free |
| Mistral-7B, Pi 5 + Pi 4 | 30 + 2, 709.8 ms | 29 + 3, 754.0 ms (+6.2 %), 333 MiB free |
| Qwen2.5-14B, Pi 4 + Jetson | 31 + 17, 1,992.1 ms | 33 + 15, 2,037.1 ms (+2.3 %), 372 MiB free |
| TinyLlama, Qwen3.5-9B on the Pi 5 | unchanged | unchanged |
| Qwen3 MoE, Pi 5 + Pi 4 | 24 + 24, 282.2 ms | infeasible: it does not fit two Pis with 256 MiB free each |

Where a plan was tight, the default trades 0–6 % of decode speed for at least 268 MiB of free memory on every board. Against
Halda, compare at equal headroom (the 0 / 1 / 2 GiB columns): the default keeps memory free that Halda's inputs do not.

## The default exact plan, measured (17:42–18:12, pre-registered, A-B-A)

Qwen3-14B on Pi 4 host + Jetson. The registration is `pool_predictions_20261007exact.json` (17:42:55, `dd5b5094…`). The
default plan reproduced the offline comparison exactly before it was written. The guard ran throughout (0 trips, minimum
4,848 mV), and a new 1 Hz poll on the Pi 4 (`testbed/scripts/throttle_poll.sh` → `testbed/results/throttle_poll_pi4_20261007exact.log`)
logged `get_throttled`, the temperature and the actual ARM clock through every cell.

| cell | pipeline (blocks; head on the Pi 4) | decode ms (predicted) | prefill ms (predicted) | Jetson's minimum free memory |
|---|---|---|---|---|
| **default** (exact + 256 MiB) | Pi 4 27 + Jetson 13 | **1,966.4** (2,025.9) | 835.3 (796.0) | **957 MiB** (planned margin 284) |
| previous plan (average, no headroom) | Pi 4 26 + Jetson 14 | 1,934.4 (2,000.5) | 809.3 (773.5) | 780 MiB (planned margin 90) |
| default, repeat | Pi 4 27 + Jetson 13 | 1,971.7 (2,025.9) | 925.1 (796.0) | 957 MiB |

Results are in `pool_results_20261007exact.json`; every split matched to 0.1 MiB.

- **The headroom works on the board that needed it.** The old plan left the 4 GB Jetson 90 MiB of planned margin. The default
  moves one block (177 MiB) to the Pi 4, and the Jetson's measured minimum free memory rose from 780 to 957 MiB. (It was
  planned at a 4096-token context and benchmarked at 80 tokens, so the run kept more free than planned.)
- **Its cost is +1.6–1.9 % in decode, against +1.3 % predicted.** The two default runs agree within 0.3 %. The predictions
  were 2.7–3.4 % pessimistic in decode, consistent with the afternoon's 14B cells.
- **The Pi 4 thermally throttles under sustained 14B work.** The snapshots could not show it.

  | cell | peak temperature | seconds the soft limit was active (bit 3) | seconds below 1.75 GHz | lowest clock |
  |---|---|---|---|---|
  | default | 82.7 °C | 2 | 32 of 494 | 0.60 GHz |
  | previous plan | 82.7 °C | 5 | 46 of 498 | 0.60 GHz |
  | default, repeat | 83.2 °C | 9 | 91 of 568 | 0.60 GHz |

  The before/after snapshots read about 68 °C, because the board cools as a cell ends, and `0x80000` had been sticky since the
  afternoon. Throttling grew over the run. It explains the repeat's 10.7 % slower prefill (prefill is compute-bound); decode,
  which is memory-bound, moved 0.3 %. **The afternoon's 14B cells were very likely throttled the same way:** their decode
  comparison should hold, but their prefill figures are suspect. This is the house note's warning, now measured: snapshots
  bracket a cell and cannot see its peak.
- **What follows:**
  1. `run_pool.sh` should run `throttle_poll.sh` on every Pi in a cell, and the scorer should flag cells that throttled.
  2. The Pi 4 needs better cooling (a heatsink and fan) before more sustained runs. That is a hardware decision; it is flagged
     in `.handoff/nodes/Pi4/`.

## 2026-10-07 (night) — the throttle poll in `run_pool.sh`, and the evening's throttling re-read

User: "yes add the throttle poll and continue everything by your best decision" (the Next action of
`.handoff/2026-10-07-1821-exact-fill-pi4-throttle.md`). The laptop was off the testbed LAN, so the Pi 4 was reached over
Tailscale (`100.121.93.63`), its host key checked against the one known for `192.168.100.106`. The Jetson was not touched,
and its guard stayed down.

### The evening's poll, read by phase

The v1 log (`throttle_poll_pi4_20261007exact.log`) was read against each cell's own llama-bench timings (`test_time`,
`samples_ns`) and its memory probe:

| cell | phase | seconds | °C | seconds with bit 3 | seconds below 1.75 GHz |
|---|---|---:|---|---:|---:|
| default | load + warm-up | 234 | 52.1–68.1 | 0 | 31 |
| | timed prefill | 160 | 57.9–82.7 | 2 | 1 |
| | timed decode | 94 | 69.6–75.0 | 0 | 0 |
| previous plan | load + warm-up | 245 | 56.4–71.5 | 0 | 40 |
| | timed prefill | 155 | 65.2–82.7 | 5 | 6 |
| | timed decode | 92 | 69.1–74.5 | 0 | 0 |
| default, repeat | load + warm-up | 286 | 55.0–70.1 | 0 | 86 |
| | timed prefill | 182 | 60.8–83.2 | 9 | 5 |
| | timed decode | 94 | 70.1–75.0 | 0 | 0 |

**Corrections to the evening's reading** (the section above is left as written):
- **The clock below 1.75 GHz, down to 0.60 GHz, was the ondemand governor idling, not a thermal cap.** 31, 40 and 86 of the
  32, 46 and 91 seconds fell in the load and the warm-up, which wait on the SD card and the RPC upload, at 52–70 °C. The other
  12 fell in timed prefill: 9 with no bit set, pipeline gaps while the Pi 4 waits for the Jetson, and 3 under the soft limit
  at 1,677–1,726 MHz. Every second with bit 3 set read 1,677–1,800 MHz.
- **Decode never came near the limit:** 69–75 °C and 1.8 GHz through every timed decode repetition. The decode figures, and
  the default's +1.6–1.9 %, stand.
- **Prefill reached the soft limit (about 80 °C on this board) for 2–9 s per cell, with no visible effect.** The repeat's
  third repetition spent 8 of its seconds under the limit (mean 1,772 MHz) and took 53.43 s, against 53.21 s for the second.
- **The repeat's 10.7 % slower prefill was a cold first repetition, not heat.** All of it is in the first timed repetition:
  76.0 s against 53.2 and 53.4 s. That repetition ran at full clock (mean 1,779 MHz) with no bit set, while the Pi 4's
  `RssFile` grew from 5,390 to 5,834 MiB: 444 MiB of mmapped weights read from the SD card inside a timed repetition. In the
  first default cell, the warm-up had already brought `RssFile` to 5,962 MiB, and its first repetition took 53.5 s. This is
  the known Pi 4 host fact of 2026-09-28: "the first prefill repetition is cold" when the host's blocks and the streamed
  upload overflow its page cache. (The clock stays up through a page-in because llama.cpp's other threads spin at the
  barrier.)
- **So the afternoon's 14B prefill figures are not suspect on thermal grounds.** Their repetitions are steady (51.38–51.50,
  58.98–60.25 and 51.18–51.52 s), and the two head-on-host cells agree within 0.1 %. Under the new rule they read
  "not polled", which is what the scorer now prints for every cell before tonight.
- **The board already has a PWM fan** (`dtoverlay=pwm-gpio-fan,fan_gpio=18`; trip points 55 / 60 / 67.5 / 75 °C, full speed
  above 75 °C), and the peaks were reached with it at full speed. Better cooling is optional; so far the soft limit has cost
  nothing measurable.

### A poll that costs a cell (almost) nothing: `throttle_poll.sh` v2

- **v1 started five processes a second** (`date` and three `vcgencmd`): 101 in 20 s on the Pi 4. That is the rate at which
  memprobe v1 cost 4.5 % of decode on the Pi 5. v1 is kept as `throttle_poll_v1.sh` (sha256 `9158dacb…`), the instrument of
  the evening's log.
- **A user has no fork-free route to the bits on this board.** The firmware driver's sysfs `get_throttled` node does not
  exist on the Pi 4's 6.18 kernel. The mailbox, `/dev/vcio`, is root-only (`crw------- root`): a Python poller that asked
  the firmware through it, as `vcgencmd` asks, failed with EACCES and was dropped. `vcgencmd` itself goes through
  `/dev/vchiq` (group `video`).
- **v2 starts one process a second**, `vcgencmd get_throttled`: 24 in 20 s, for 21 samples. Everything else is a bash
  builtin:
  - the time (`$EPOCHREALTIME`);
  - the temperature (`thermal_zone0`, the sensor `measure_temp` reads: both read 51.1 °C side by side);
  - the clock the governor asks for (cpu0 `scaling_cur_freq`);
  - the sleep, which is timed to the next whole second, so no second is skipped (v1 drifted: 1,686 samples in 1,726 s).

  The measured ARM clock (`vcgencmd measure_clock arm`) is taken only in a second with a bit set.
- **Found while counting:** bash 5.2 forks twice for `$(cmd 2>/dev/null)` and once for `$(cmd)`: 400 against 201 process
  starts per 200 substitutions. v2 has no redirection inside a command substitution, and a test keeps it that way.

**What it costs, measured** (`throttle_poll_ab_pi4.sh`, `testbed/results/throttle_poll_ab_20261007_205507/`, `summary.txt`).
This was the Pi 4 alone, running TinyLlama Q4_K_M at `-t 4 -p 64 -n 128 -r 5`. The arms ran R V1 V2, then V2 V1 R, behind a
55 °C cold gate: every arm started at 54.0–54.5 °C and ended at 70.6–73.0 °C. The A/B was stopped after round 2 on the
user's word, and round 3 never started.

| condition | process starts a second | decode ms/token, rounds 1 / 2 | decode against R | prefill (pp64) ms/token | prefill against R |
|---|---:|---|---|---|---|
| R, nothing beside llama-bench | 0 | 176.01 / 176.37 | — | 77.52 / 77.60 | — |
| V1, `throttle_poll_v1.sh` | 5 | 178.76 / 179.70 | **+1.56 / +1.89 %** | 78.10 / 78.19 | +0.74 / +0.76 % |
| V2, `throttle_poll.sh` v2 | 1 | 175.78 / 177.65 | −0.13 / +0.73 % (pooled +0.30 %) | 77.66 / 77.61 | +0.18 / +0.01 % |

- **v1 costs 1.7 % of decode and 0.75 % of prefill on the Pi 4.** R's two rounds agree within 0.21 %.
  - v1 ran beside the three evening cells. Their A-B-A comparison is unaffected, since all three had it.
  - Their absolute decode figures carry about 1 % of it (the Pi 4's share of each token). That is an estimate, not
    measured.
- **v2 costs +0.30 % of decode (pooled) and +0.10 % of prefill: at the noise.** memprobe v2's +0.4 % was accepted the same
  way.
- **Reported as found:** both V2 arms end with one slow repetition (177.8 and 180.7 ms, against 175.2–177.3), and no R or V1
  arm shows one. It is unexplained. Without those two repetitions, v2 is within ±0.4 % of R. A third round was not run.
- Every polled arm's THROTTLE line read clean (no bit, peak 75.4–76.4 °C). The two polls' logs are kept beside `run.log`.

### In the runner and the scorer

- **`run_pool.sh`** starts `throttle_poll.sh` on every Pi, next to its memory probe, from before `state before` to after
  `state after`. The polls are held in their own list, `TPROBES`; putting them in `PROBES` would have made the MEMPROBE loop
  report every Pi twice. The EXIT trap stops them with the probes. After the bench, each poll is stopped (noting whether it
  was still alive), and its log is copied back as `throttle_pool_<cell>_<stamp>_<node>.log`. Then one line per Pi:
  `THROTTLE <node> poll=v2 samples=… max_temp_c=… active_s=… now_bits=… sticky_new=… active_min_clock_mhz=… … verdict=…`.
  A Pi with no poll, or whose log could not be copied, gets `verdict=UNKNOWN` and says why.
- **`throttle_summary.py`** holds the rule:
  - **THROTTLED:** a now-bit (0–3) was set in some second, or a sticky bit (16–19) came on between two seconds;
  - **UNKNOWN:** the poll does not cover the cell (no samples, a failed read, a gap over 3 s, an unreadable line, or a poll
    no longer running when stopped);
  - **clean:** otherwise.

  The clock is reported but never judged. On the evening's v1 log it reproduces the published per-cell figures exactly:
  2 / 5 / 9 s, 32 / 46 / 91 s, 82.7 / 82.7 / 83.2 °C.
- **`compare_pool.py`** reads the THROTTLE lines and recomputes each Pi's verdict with that rule, so the verdict logged
  beside the fields is not trusted. It writes `thermal: {verdict, per_node}` per cell, and a cell takes the worst of its
  Pis. Logs from before tonight read "not polled". Re-scoring the exact campaign gives the published results unchanged
  apart from the new key.
- **Tests:** `testbed/tests/test_throttle_poll.py`, 16 tests. The testbed suite went from 35 to 51.

### The shakedown: the edited runner, run end to end (2026-10-08 00:37)

`shake.tiny.pi4_poll` ran: TinyLlama on the Pi 4 alone, at the pool protocol, the laptop back on the LAN. Its registration is
`.handoff/assets/throttle-poll-2026-10-07/shakedown_registration.json` (2026-10-07 20:58:38, `c2534c24…`). The runner's four
files still matched the sha256 recorded in it. The log is `testbed/results/pool_shake.tiny.pi4_poll_20261008_003721.log`,
and the score is `shakedown_results.json` beside the registration. It is a test of the instrument, not a reported figure.
- **The poll ran as designed.**
  - It started beside the memory probe: `probes  pi4:17649   throttle polls pi4:17673`.
  - It sampled 30 s with no gap over 1 s, no failed read and no unreadable line. It was still alive when stopped.
  - Its log came back as `throttle_pool_shake.tiny.pi4_poll_20261008_003721_pi4.log`, with v2's header.
  - The line read `THROTTLE pi4 poll=v2 samples=30 … max_temp_c=71.0 active_s=0 … alive=1 verdict=clean`.
- **The scorer** printed `thermal=clean[pi4 peak 71.0 C]`.
  - Decode measured 176.7 ms against 175.4 predicted (−0.7 %), prefill 78.2 against 79.8 (+2.0 %).
  - The split matched (636.18 MiB of a planned 636.2), the cell was registered before it ran, and it exited 0.
- **The Pi 4 was left clean:** no llama-bench, and no probe or poll still running.

## 2026-10-08 — the output head goes where it is cheaper, by default; `serve_pool.sh` places it

User: "make head-on-host the default and add it to serve_pool". Implemented as `plan_optimal(head_placement="best")`, the
default since 2026-10-08. It weighs the head on the host against the head on the last board, for every order, and keeps the
cheaper; always-on-the-host would be wrong where the last board is faster. Both were measured on the boards on 2026-10-07:
- **Qwen3-14B, Pi 4 + Jetson:** the head on the host decoded 5.4 % faster and prefilled 13.8 % faster than head last.
- **TinyLlama, Pi 4 + Jetson:** the head stays on the Jetson. Halda's head-on-host layout measured 6.4 % slower (101.7 against
  95.6 ms).

**What changed:**
- **`ThinkPad/control/placement.py`:**
  - The default `head_placement` is `"best"`, and the docstring cites the measurements above.
  - `"last"` reproduces every plan made before.
  - `Placement.llama_args()` now refuses a plan with a head on the host. Its layout runs llama.cpp on the control node with
    every participant an RPC device, so `-ot` there would put the head on the laptop, not where the plan priced it.
  - Unreachable today: no model in `cluster.json` has `host_in_pipeline`, so the CLI and harness plans are unchanged.
- **`testbed/scripts/predict_pool.py`:** a registration's rule plan is now priced, charged and launched where the plan put the
  head: `head_at` passes into `predict`, `memory` and `bench_args` (with `-ot`).
  - `head_on`, `head_at` and the engine's `head_node` are recorded.
  - `--moe-batch` refuses a head on the host, because `moe_prefill` prices it last only. That is a refusal rather than a
    wrong prefill.
  - A dry run of qwen3_14b, tinyllama and nemo12b plans `qwen3_14b.rule` as Pi 5 34 + Jetson 6 with the head on the Pi 5 and
    `-ot "^output\.weight=CPU"`.
- **`testbed/scripts/serve_pool.sh`:**
  - **`SERVE_HOST`** names the board that runs llama-server. The default host for most models is still the Pi 5 (returned
    2026-10-05), and a missing host is refused by name.
  - **The plan's own head placement** flows into the gate, the prediction and `-ot`.
  - **After loading**, each worker's model buffer, as llama.cpp reports it, must match the plan within 1 MiB, or the server is
    stopped. llama-server does not print the debug-level "buffer type overridden" line; a head on the wrong board shows as
    ~600 MiB.
  - The planned Jetson buffer for the 14B (2,489.6 MiB) equals the one measured in the 2026-10-07 default cell (2,489.60).
- **`testbed/scripts/validate_placement.py`:** pinned to `"last"`, because it checks the pick against a sweep run in llama.cpp's
  own layout. It still picks the measured fastest, Pi 5 43 / Pi 4 5.

**Checks:**
- **ThinkPad 257** (with `-W error::SyntaxWarning`): +5 in `TestDefaultHead`, and four head-accounting tests now ask for
  `"last"` by name. Pi5 357, Pi4 158 and testbed 51, all OK.
- **`TestDefaultHead`:**
  - the default equals `"best"` in decode, prefill and request planning;
  - on 400 random three-board host-in-pipeline clusters (seed 20261008), the default is never slower and never infeasible
    where `"last"` was feasible, and it moves the head in some of them;
  - a model without `host_in_pipeline` plans exactly as before.

**`serve_pool.sh` on the boards** (Qwen3-14B, `SERVE_HOST=pi4 NODES="pi4 jetson" CTX=4096`; plan Pi 4 27 + Jetson 13, head on the
Pi 4, `-ot "^output\.weight=CPU"`). It took three starts, each finding something, all reported as found:
1. **00:48: the split check found no buffer lines,** and stopped the server, as it is meant to (the server, the worker and the
   Jetson's clocks were all restored). At the pin llama-server logs the library's INFO lines as TRACE, level 4
   (`common_log_get_verbosity`), above its default threshold of 3. So no serve log since 09-30 holds a "model buffer size"
   line, and the post-READY grep that was meant to show them has printed nothing all along. **Fixed:** `-lv 4`.
2. **00:53: the split matched,** with the Jetson's RPC buffer at 2,489.60 MiB, the plan's 2,489.6, and the Pi 4's
   `CPU_Mapped` at 6,089.47 MiB. **But serving thrashed.** The first request (5 prompt tokens) produced 4 tokens in 5.5 min.
   The second read the SD card at 19–31 MB/s with 65–75 % iowait throughout, and was stopped. With the weights
   memory-mapped, the page cache did not keep the Pi 4's 6,089 MiB resident beside the server's 0.7 GB of KV and compute
   buffers. The benchmark cells, at an 80-token context, never showed this, and the Pi 4 had never been a 14B serving host
   (QR3 used the Pi 5, which copies its weights out of the mapping when it repacks).
   - **Fixed:** `-lm none` (`--load-mode`), so the weights are read into the server's memory, which is what the engine charges
     for. mlock is not an option, because `ulimit -l` is 8 MiB on the Pis.
   - **`stop` now sends a second TERM after 20 s.** The server twice failed to exit on the first one: once mid-token on the
     card, and at 02:48 while idle. At the pin, llama-server takes the first signal as a graceful shutdown and a second as
     "terminating immediately" (`tools/server/server.cpp:29-32`). Both times the second TERM ended it.
3. **01:06: the plan served, at benchmark speed.**
   - READY 01:10:59, and the split check passed.
   - Two greedy requests each returned " Paris. What is the capital of the United States? The capital of the United", at
     **1,967.1 and 1,967.3 ms per generated token**. That equals the 2026-10-07 benchmark cells of this plan (1,966.4 /
     1,971.7 ms); the prediction was 2,025.9.
   - The Pi 4 read 0.2 MB/s from the card, with iowait at most 3 %.
   - Stopped cleanly. The Jetson's guard was started by its NOTE's procedure before the first load.

## 2026-10-08 — energy against prima.cpp and Halda, and what beats both

User: "benchmark energy compare to the prima and halda using what we have, and then analyze why halda + prima is more
sufficient then apply any techniques to overcome it". What we have:
- the boards: Pi 4 + Jetson, since the Pi 5 was returned on 10-05;
- prima.cpp: built on the Pi 4 only, without HiGHS, so it cannot run Halda there; the Jetson cannot run it, and the laptop has
  no ZeroMQ headers;
- Halda's plans: our port, `ThinkPad/control/halda.py`;
- the strip, and every energy run since 10-05.

### An energy model, validated on every measured pooled run

`testbed/scripts/energy_model.py` is the planner's own model: each board draws its measured active power while it computes
its share of a token, and its idle power while it waits. The power comes from the strip's single-board TinyLlama runs, and
the share from `predict_pool.predict`'s own terms. Against all 16 measured pooled phase-runs of 10-05/06:
- **gross mean |error| 3.4 %, net 5.2 %;**
- the 14B cells are within 3 %, so a board's full-load power hardly depends on the model;
- the one miss over 6 % is the run that ended in the Jetson's hang (12.8 W against 10.2 in the same cell).

`python3 testbed/scripts/energy_model.py validate` prints the table, and `testbed/tests/test_energy_model.py` keeps it under 5 %.

### Against prima.cpp: modeled at the 2026-09-30 measured speeds

The registered head-to-head of 09-30 (Qwen3-14B, Pi 5 + Pi 4, the QR3 protocol) measured speed only, since the strip came on
10-04. Priced at its measured times, with prima.cpp's Pi 5 at its unrepacked rate (prima.cpp's base predates the repacked
kernels):

| | decode, gross J/token (net) | a 128-token prompt, gross J (net) | board power while decoding |
|---|---|---|---|
| **engine + llama.cpp** (Pi 5 32, Pi 4 8) | **11.34** (5.76) | **478** (265) | Pi 5 5.05 W (busy 46 %), Pi 4 4.49 W (busy 53 %) |
| prima.cpp + Halda (Pi 5 24, Pi 4 16) | 12.46 (6.17) | 847 (463) | Pi 5 4.30 W (busy 33 %), Pi 4 5.00 W (busy 66 %) |

**The engine uses 9.0 % less energy per decoded token (6.7 % net) and 43.6 % less per prompt.** Halda moved 8 blocks onto the
Pi 4, which is twice as slow and half as efficient per byte (0.72 against 0.36 J per TinyLlama token, net), and both boards are
powered for the longer time. This is modeled, not measured: prima.cpp cannot run on today's boards.

### Against Halda: modeled offline on every board set, and the one Halda plan that fits today's boards, measured

**Offline** (`energy_model.py offline`): every model x board set of `halda_offline_20261007.json`, our default plan against
Halda's fastest ring that fits our memory accounting, priced at the predicted times.
- **Where Halda has a plan and the plans differ, Halda's uses less energy in most cases:**
  - Qwen2.5-14B and Qwen3-14B on Pi 5 + Pi 4: +2.2 / +3.0 % decode, +4.7 / +5.8 % prefill for us;
  - Mistral-7B on Pi 5 + Pi 4: +4.7 % decode, +8.6 % prefill for us;
  - with the Jetson: +1.4 to +2.8 % decode for us.
  - It is the headroom (next section); at equal headroom ours ties or wins.
- **On today's Pi 4 + Jetson, Halda has a plan only for TinyLlama.** For every other model, all of its rings fail our memory
  accounting (it knows nothing of the Jetson's Vulkan reserve or weights cap), so there is nothing of Halda's to run.

**Measured** (registered 01:13:35, `energy_h2h_predictions_20261008.json`, `f4244f65…`; run
`testbed/results/energy_pool_20261008_011447/`; `score.json`, `h2h_score.json`).
- The setup: TinyLlama on Pi 4 + Jetson, the same split (Pi 4 1 + Jetson 21). Our plan has the head on the Jetson, Halda's on
  the Pi 4.
- Run as engine, Halda, engine, Halda, engine inside one strip log. The Jetson's clocks were locked for the whole session,
  idle windows included.
- All five cells exited 0. Every throughput check held (within 1.5 % of the prediction), and every Pi 4 THROTTLE line read
  clean.

| plan | runs | decode t/s (predicted) | decode gross J/token, trapezoid (counter) | prefill gross J/token, trapezoid | decode net, counter | board W while decoding |
|---|---|---|---|---|---|---|
| **engine** (head on the Jetson) | 3 | **10.44** (10.49) | **1.3327** (1.3876), spread 0.6 % | 0.2032 | 0.7838 | Pi 4 3.42–3.47, Jetson 10.40–10.50 |
| Halda (head on the Pi 4) | 2 | 9.77 (9.83) | 1.3589 (1.4142), spread 0.3 % | 0.2042 | 0.7693 | Pi 4 3.85–3.87, Jetson 9.41–9.42 |

**The registered verdicts, applied as written:**
- **H-DEC holds:** the engine's decode costs 1.93 % less at the wall by the trapezoid and 1.88 % less by the counter. The
  prediction was −2.56 %.
- **H-PRE holds:** prefill differs by −0.49 % (trapezoid) and −0.25 % (counter), inside ±2 %. The prediction was −0.53 %.
- **ACC holds:** every run is within 10 % of the model, the worst at −6.7 %. Here the model is uniformly 5–7 % low. Most of
  that is the Pi 4's host power (3.4–4.1 W measured against 2.7–3.5 modeled): the throttle poll and the memory probe now run
  beside every cell, and the model prices the host's network and sampling work at idle.
- **Reported as found: above idle, Halda's plan is the cheaper one.** Its net decode energy is 1.9 % below ours. Its Jetson
  waits while the slow Pi 4 computes the head, so it draws less above its floor (9.4 W against 10.4). Both boards are powered
  for 6.4 % longer, though, so at the wall the engine's plan is the cheaper one. With boards that stay on, energy follows time.

### Why Halda (and prima.cpp) can look more efficient

- **Memory headroom, by design.** Our default keeps 256 MiB free on every board (user's decision, 10-07), and Halda's rule as
  modeled offline uses every byte.
  - On tight Pi 5 plans that headroom moves one block to a slower, hungrier board: +1.4 to +4.7 % decode energy and up to
    +8.6 % prefill against Halda.
  - **At equal headroom (0), our exact plan ties or beats Halda in every case:** −2.7 to 0 % decode, −5.8 to 0 % prefill.
  - The headroom is earned. Per board, the accounting under-estimated only on the Pi 5: 6 of 33 cells, by up to 133 MiB.
    The Pi 4 (0 of 33) and the Jetson (0 of 23) never did. Halda's full packing is exactly where the Pi 5 has run short.
  - prima.cpp itself did the opposite on 09-30. Its runtime budget (5.85 GiB on the Pi 5) was far below the board's free
    memory, which is why it measured slower.
- **Memory stress.** prima.cpp keeps ~6.9 GB "available" on each board (mmap and page cache, devices still usable). The engine
  runs a dedicated board near full. That is a design point, not energy. Our `with_headroom` buys it at a measured price:
  +19 % per token at 1 GiB kept free, +58 % at 2 GiB (14B, three boards).

### Techniques that go past both

1. **Energy-weighted planning with the strip's power** (`placement.py`, 2026-10-08).
   - `DeviceProfile.power_active_prefill_w`: prefill priced at prefill power, so the Jetson's GPU is charged 8.9 W prefilling
     and 11.2 W decoding.
   - `plan_optimal(unused="off"|"idle")` says whether a board the plan leaves out is switched off or idles; `evaluate(idle_on=)`
     charges it.
   - `predict_pool.profiles(power=)` feeds in the measured table.
   - The defaults plan exactly as before (ThinkPad 257 → 261: `TestEnergyPolicy`, 4 tests).
2. **What it finds on today's boards** (`energy_model.py techniques`, modeled):
   - **With every board left powered, the fastest plan is already the least-energy plan** for TinyLlama, Qwen3.5-9B and
     Qwen3-14B. Each board's idle floor is paid for the whole token, so time dominates.
   - **With the boards a plan leaves out switched off, consolidation pays:**
     - Qwen3.5-9B on the Pi 4 alone uses 24.6 % less energy, at +26 % time;
     - TinyLlama on the Pi 4 alone uses 10.8 % less, at +84 % time;
     - Qwen3-14B needs both boards.
   - Neither Halda nor prima.cpp ever switches a device off. The strip can, but that is the user's call per action.
3. **The host, chosen.** TinyLlama on the Jetson ALONE (GPU, all 22 blocks): the planner, allowed to pick the host among the
   boards holding the file, predicts 87.0 ms/token against 93.4 pooled, at 0.97 J/token (Pi 4 off) or 1.18 (Pi 4 idling)
   against 1.23. That is faster and cheaper at once. The 2026-10-05 single-board run measured 0.982 J/token on its outlet
   (1.193 with the Pi 4 idling). prima.cpp's head is whatever device the user starts as rank 0, and Halda never picks it.
**Measured** (registered 01:54:32, `energy_host_predictions_20261008.json`, `8c678420…`; run
`testbed/results/energy_pool_20261008_015432/`). Two runs, the same night and the same locked clocks as the head-to-head, with
the Pi 4 powered and idle on its own outlet (2.77–2.81 W):

   | plan | decode t/s | decode gross J/token, trapezoid (counter) | prefill t/s | prefill gross J/token |
   |---|---|---|---|---|
   | engine, pooled (Pi 4 1 + Jetson 21) | 10.44 | 1.3327 (1.3876) | 57.7 | 0.2032 |
   | Halda (the same, head on the Pi 4) | 9.77 | 1.3589 (1.4142) | 57.7 | 0.2042 |
   | **the Jetson alone, the Pi 4 idling** | **11.46** (pred. 11.49) | **1.2429** (1.2936) | **73.1** | **0.1615** |
   | the Jetson alone, the Pi 4 switched off | 11.46 | 1.0001 (1.0412) | 73.1 | 0.1232 |

   - **H-HOST holds:** with both boards powered, 6.7 % less energy per decoded token than the pooled engine plan (8.5 % less
     than Halda's) and 9.8 % faster. Prefill is 26.7 % faster and 20.5 % cheaper.
   - **Switch the idle Pi 4 off and the saving is 25.0 % (26.4 % against Halda).**
   - The model was within 2.8 % for the Jetson's outlet. With the Pi 4 it was 5.1 % low: the Pi 4's idle floor was 2.81 W
     tonight against 2.39 W on 10-05 (the Jetson's 3.28 against 3.14). Two candidates, not separated:
     - its Wi-Fi and Tailscale are up now (`wlan0` 192.168.1.107), while `node-pi4.md` says wired only;
     - its PWM fan sat at level 1 while idle tonight.
     Reported as found.
4. **Serving from memory, not from the SD card,** on a host that does not repack (`serve_pool.sh -lm none`): the mapped 14B
   thrashed at 4 tokens in 5.5 min, and served at 1,967 ms/token once resident. That is an energy fix too: every second the
   card stalls, both boards draw power for nothing.
5. **The Jetson's GPU clock (DVFS), measured** (registered 02:12:18, `energy_host_predictions_20261008_gpu614.json`,
   `bf08026f…`, exploratory; run `testbed/results/energy_pool_20261008_021218/`).
   - The setup: the same Jetson-alone cell, twice, with the GPU held at 614.4 MHz (devfreq min = max, EMC at 1,600 MHz),
     against the 921.6 MHz session above. Afterwards the clocks were restored from the store.

   | at 614.4 against 921.6 MHz | decode | prefill |
   |---|---|---|
   | speed | **−31.7 %** (7.83 against 11.46 t/s) | −31.6 % (50.0 against 73.1) |
   | the Jetson's power | −35.2 % (7.43 against 11.46 W) | −31.7 % (6.16 against 9.01 W) |
   | J/token, the Jetson alone (the rest switched off) | **−5.1 %** (0.949 against 1.000, trapezoid) | 0.0 % |
   | J/token with the Pi 4 left idling | **+4.9 %** (1.303 against 1.243) | +10.6 % |

   - **Decode on the Nano's GPU is compute-bound,** not memory-bound: speed fell almost in proportion to the clock (−31.7 %
     for −33.3 %).
   - **H-DVFS holds on the cell's own outlet** (−5.1 %, at a third less speed). It **reverses as soon as any other board stays
     powered**, because the longer time costs every idle floor.
   - The registered throughput checks fail, as registered: the predictions are the full-clock ones.
   - **So race-to-idle, the full clock, stays the default.** A lowered clock is worth it only for the Jetson alone with
     everything else off, and only in decode.

### Against the pipeline-parallel systems prima.cpp evaluates and cites (PP only; user, 2026-10-08)

Source: `.handoff/assets/pp-literature-2026-10-08/pp_systems_report.md`. Every published number there is copied with its table
or figure; ≈ marks a value read from a plot. Tensor-parallel systems (distributed-llama, TPI-LLM, Galaxy) and
speculative-only ones (SLED) are left out. prima.cpp's own llama.cpp baseline is always a single device, so exo is its only
outside PP baseline.

| system (PP scheme) | devices | model | decode ms/token | first token | memory | energy | our nearest measured figure |
|---|---|---|---|---|---|---|---|
| prima.cpp v3 (piped ring + Halda) | Mac M1, i9 + RTX 3070 laptop, i9 + RTX 2080 Ti desktop, Mate40Pro (37 GiB) | Llama 14B / 70B Q4K | 20 / 674 (T4) | 25 / 1,793 ms | ≤ 6.3 % per device (T5) | 12.6 Wh per 1K tokens, on another four-device testbed, model unstated, software counters (Fig. 13) | prima.cpp on **our** boards (09-30, measured): 1,339.3 ms, Qwen3-14B, Pi 5 + Pi 4; the engine 1,188.6 (−11.2 %), and −9 % J/token (modeled) |
| exo (memory-proportional ring), as prima.cpp ran it | M1, laptop, desktop | Llama 3 8B Q4K | 263 (T4) | 960 ms | 20–51 % per device | none | — |
| EdgeShard (DP over device selection and layers) | 12 AGX Orin 32 GB + 2 Orin NX + RTX 3090 | Llama2-13B FP16 | 173.4 (T IV) | not reported | words only | none | — |
| EdgeShard, re-run by Jupiter's authors | Xavier NX + 2 TX2 + **Jetson Nano** | Llama2-13B INT4 | ≈ 1,150 (Fig. 11) | figure only | — | none | **Qwen3-14B Q4_K_M on Pi 5 + Jetson Nano: 851.6** (P95 854.9; QR3, 09-30), two boards; on Pi 4 + Nano 1,967 |
| Jupiter (pipelined prefill; outline + speculative decoding, which changes the output) | the same four Jetsons | Llama2-13B INT4 | ≈ 370 (Fig. 11) | figure only | — | none (rated 10–20 W per device) | not exact decoding, so not comparable as such |
| LinguaLinked (MILP segments + load balancer) | 3 Pixel 7 Pro + 1 CUBOT X30, CPU | BLOOM-3B int8 | ≈ 1,580 (Fig. 4) | not reported | not reported | none | TinyLlama 1.1B: 95.6 ms pooled (Pi 4 + Jetson), 88 on the Jetson alone |
| llama.cpp RPC (the runtime we pool on) | — | — | **none published** | none | none | none | every pooled cell in this file since 09-27 |

**What the comparison says**
- **Against Jetson-class pipelines, our numbers are competitive with less hardware.** The closest published setting is
  EdgeShard on four Jetsons including a Nano, about 1,150 ms per token for a 13B INT4 model. Ours is 851.6 ms for a 14B Q4_K_M
  on two boards, a Pi 5 and a Nano. The models, quantisations and networks differ, so this is indicative, not a controlled
  result.
- **prima.cpp's headline speeds come from RTX-class GPUs** (20 ms/token at 14B). On our boards prima.cpp measured 1,339 ms,
  and the engine was faster in a registered head-to-head.
- **Energy is almost never reported for PP systems.**
  - Only prima.cpp v3 does: software power counters, on a different testbed, with the model not stated. So its 12.6 Wh per
    1K tokens (45 J/token) cannot be set against ours per token.
  - Ours is wall power per outlet, pre-registered, modeled within 3–7 %:
    - Qwen3-14B on Pi 4 + Jetson: 20.4 J/token gross in decode, 5.7 Wh per 1K tokens (10-06);
    - TinyLlama pooled: 1.33 J/token, 0.37 Wh per 1K;
    - TinyLlama on the Jetson alone: 0.98 J/token.
- **Memory pressure is prima.cpp's real advantage, by design.** It stays at or below 6.3 % of each device, because its
  weights live in the page cache of devices still in daily use. Our boards are dedicated and packed. In the 14B default cell
  the Jetson's MemAvailable fell from 3,541 to 957 MiB of 3,963, a pressure of 65 % by prima.cpp's definition (the drop over
  the total). `with_headroom` buys
  prima-like stress at a measured price (+19 % per token at 1 GiB kept free, +58 % at 2 GiB).
- **Rigour.**
  - prima.cpp's main tables give no repetitions, spreads or prompt lengths.
  - EdgeShard's throughput is batched, not single-stream, and it reports no time to first token.
  - LinguaLinked's and Jupiter's per-token values are in plots only.
  - Ours carry pre-registered predictions, at least 3 repetitions with spreads, byte-exact split checks, pinned builds and
    models, and (from tonight) a throttle certificate per Pi cell.
- **What no cited PP work measures:**
  - a Raspberry Pi in a pipeline;
  - llama.cpp RPC itself;
  - wall-power energy per token.
  That is the gap this project fills.

## 2026-10-08 (early morning) — the host chosen by default; a page-in flag; the load-mode A/B

User: "go on with 1 and 2 and all". Item 1 (the documents) went to a background agent. These are items 2–5.

### The host, chosen by default (item 2)
- **`testbed/scripts/model_inventory.py`** takes a dated snapshot of what each board holds: `testbed/dataset/model_inventory_<stamp>.json`,
  with names and sizes only (the runners check sha256 before reported figures). The first snapshot (03:06) shows TinyLlama on
  the Pi 4 and the Jetson; Qwen3-14B, Qwen3.5-9B and the 30B MoE on the Pi 4 only; the Pi 5 not reachable.
- **`predict_pool.plan_host_choice()`** plans each candidate host and keeps the fastest predicted decode.
  - The candidates are every board in the pool that holds the file, or `SERVE_HOST` if given, or the old default.
  - A GPU board hosts only a plan on itself alone: with RPC workers, its own GPU and theirs would need a device split that
    `bench_args` does not write.
- **`predict_pool.bench_args`** now puts a lone GPU board on its GPU (`-ngl 99`). The old `-ngl 0` would have run a
  Jetson-alone plan on its CPU while pricing it from its GPU row. CPU boards are unchanged.
- **`serve_pool.sh`** uses `plan_host_choice`, prints every host it considered, and serves TinyLlama on the Jetson alone
  (87.0 ms/token predicted, against 95.4 hosted on the Pi 4). The 14B models still go on the Pi 4, their only holder.
- **Fixed in both runners:** the Jetson's gates (MAXN, a live un-tripped guard, eth0 only) and its clock lock fired only when
  it was a WORKER. A Jetson-hosted cell (tonight's `eh.tiny.jetson`) was therefore not gated; its guard happened to be up.
  They now fire whenever the Jetson takes part.
- **Tests:** `testbed/tests/test_host_choice.py`, 6 tests.

### A page-in flag in the scorer (item 3, the scorer half)
`compare_pool.py` now reads the host's memory probe beside each cell's log. It reports how far the host's `RssFile` grew
inside each scored test's timed repetitions, and flags a cell over 64 MiB. The survey of every probed cell: normal cells
grew at most 23 MiB, the page-ins 169–992 MiB.
- On the 10-07 exact campaign it flags exactly `x.q3_14b.default_rep` (pp +446 MiB), the cell whose first prefill
  repetition took 76.0 s.
- The re-score test checks this.

### The load-mode A/B (item 3; registered 03:09:37, `pool_predictions_20261008loadmode.json`, `15536064…`)
- **The setup:** Qwen3-14B, the default plan (Pi 4 27 + Jetson 13, the head on the Pi 4), run mmap, none, mmap, none. `none`
  is `llama-bench -lm none`: the weights read into memory, as `serve_pool.sh` now serves. All four cells exited 0 and every
  split matched.
- **The four runs, each on its own** (`compare_pool.py` keys its JSON by cell id, so it keeps only the last run of each;
  `pool_results_20261008loadmode.json` holds those, and the runs below were read from their own logs):

  | run | prefill repetitions, s | decode repetitions, s | page-in during timed repetitions |
  |---|---|---|---|
  | mmap 03:09 | 53.03 / 52.89 / 52.93 | 31.41 / 31.36 / 31.59 | pp +9, tg +3 MiB (none) |
  | none 03:20 | 52.53 / 52.83 / 53.31 | 31.93 / 31.94 / 31.85 | 0 / 0 |
  | mmap 03:29 | 53.42 / 53.19 / 53.70 | 31.73 / 31.59 / 31.33 | pp +6, tg +3 MiB (none) |
  | none 03:41 | 52.62 / 52.74 / 53.09 | 32.08 / 31.46 / 31.04 | 0 / 0 |

- **H-LM-COLD holds:** in both `none` cells every scored test's repetitions agree within 3.4 %, and nothing paged in.
- **H-LM-SPEED holds:** `none` decodes at +0.7 % against `mmap` (1,982.3 against 1,968.8 ms/token, the means of the six
  repetitions), and prefills at −0.1 % against `mmap`'s fastest repetition (825.8 against 826.4 ms/token).
- **Reported as found:** neither `mmap` run had a cold first repetition this time. The A/B shows that reading the weights
  into memory costs nothing and rules the page-in out by construction; it did not catch a cold repetition in the control.
- **The memory accounting, checked literally.** With `-lm none` the Pi 4's MemAvailable fell by 6,228 MiB (anon 6,209),
  against the engine's charge of 6,241 MiB. Mapped, it fell by 190, because the weights count as page cache.
  **Re-read 10-09:** the match was a coincidence. llama-bench allocates KV only for its test context, and the charge's KV at a
  4,096 context (432 MiB) stood in for the embedding the charge missed (417). See "engine improvements", section 1, below.
- **The scorer's split check** reads the host's buffer as `CPU_Mapped` or `CPU_REPACK`; under `-lm none` it is `CPU`, so the
  Pi 4 printed 0/6089.5. Fixed below.
- All four cells reached the Pi 4's soft limit in prefill (THROTTLE: 4 / 29 / 17 / 16 s, 82.7–83.2 °C). Decode is unaffected,
  as on 10-07.
- **Decision (the user's "go on ... all"):** a Pi 4-hosted registration from now on loads the weights into memory (see the
  `bench_args` change below).

### The host choice, served (03:53)
`MODEL=tinyllama NODES="pi4 jetson" serve_pool.sh start`, with no host named.
- It considered both holders: the Jetson alone at 87.0 ms, Pi 4-hosted at 95.4.
- It served on the Jetson. The Jetson's gate ran for a Jetson HOST (the gate fix): MAXN, guard live, eth0 only.
- READY at 03:54:11; a greedy request decoded at 98.5 ms/token over 32 tokens (a short sample; the energy session's
  llama-bench measured 87.3); stopped cleanly. The guard was stopped at 03:54:20.

### The throttle poll's cost, round 3 (03:54–04:08, R V1 V2; `throttle_poll_ab_pi4.sh ROUNDS=1`)
- Decode: V1 +0.99 %, V2 −0.69 % against R. Prefill: V1 +0.42 %, V2 +0.06 %.
- **Over all three rounds:** v1 costs +1.48 % in decode (+1.56, +1.89, +0.99) and v2 −0.03 % (−0.13, +0.73, −0.69). v2's cost
  is indistinguishable from zero, and the poll that runs in every Pi cell is v2.

### Applied after the A/B
- **`predict_pool.bench_args(..., load_mode=None)`:** a registration passes `"none"` to add `-lm none`. New Pi 4-hosted
  registrations do so; the default is unchanged, so every registration on file still reproduces.
- **`compare_pool.py`'s split check** counts the host's `CPU` buffer (the `-lm none` case) beside `CPU_Mapped`. The `none`
  cell now reads 6,089.46 against 6,089.5 MiB.
- **Documents (item 1),** by a background agent stopped by the account's session limit:
  - `HeteroServe_Results_Comparison.docx` gained the energy and PP-literature sections (7 → 12 pages);
  - `HeteroServe_ECTICON_paper_draft.docx` gained Sections III-F (energy) and IV (PP literature), with the abstract and
    contributions tightened, still 4 pages;
  - report 4 was not reached.
  - Backups: `.handoff/assets/documents-2026-10-08/previous/`.

## 2026-10-08 (afternoon) — the network study registered; the hand-off's network crossover; the reproducibility check

User: "you run the command and finish what's left". The testbed LAN was down all afternoon: the laptop's and the Pi 4's
Ethernet had no carrier, and from the Pi 4 (over Tailscale, on home Wi-Fi) the router, the Jetson and the strip did not
answer. So the pilot waits in one background job (`.handoff/assets/full-eval-pilot-2026-10-08/wait_and_run_pilot.sh`, armed
14:44 for up to 6 h, under an idle-sleep inhibitor), and everything below was done without the boards.

### `full_eval.py` now polls every Pi (a house rule it broke)
The harness served through `serve_pool.sh`, which never polled `get_throttled`. It now does what `run_pool.sh` does:
- every Pi in the pool runs `throttle_poll.sh` v2 for the whole session, and its log is copied into the run dir;
- the summary certifies each request's whole window, keeping the worst verdict per placement. A window the poll does not
  reach within 3 s of either edge is UNKNOWN, so a poll that died mid-session cannot pass on the seconds it covered.
- The pilot's smoke round now stops if the pooled request's Pi 4 verdict is UNKNOWN.
- Tests: `test_full_eval.py` +1 (clean, throttled, past the poll's end, died inside, started late, a 5 s hole).

### The network study (topic-form scope 6), registered and ready
**Why.** At gigabit the link terms are 1–2 % of a token, so no campaign has tested them. Yet they decide one choice outright.
- With the head on the last board, every decoded token brings the logits back to the host (TinyLlama: 128 KB).
- With the head on the host, only the hidden state comes back (8 KB).
- At gigabit the head belongs on the Jetson (measured 10-07); below some bandwidth it belongs on the host.

**The shaping** (`testbed/scripts/netshape.sh`) is done on the Pi 4 alone; the Jetson's suspect card is not touched.
- **Egress to the Jetson:** a prio root whose third band carries only what a u32 filter on the Jetson's address puts there,
  with netem on that band.
- **Ingress from the Jetson:** a u32 filter redirects it through ifb0, which has the same netem.
- **Other traffic** (ssh, the strip, the laptop) is not shaped.
- **Tested 14:47 on the Pi 4's eth0, which was down:** apply, show, refuse to apply twice, clear (back to `mq`), and check.

**Registered before it ran** (`pool_predictions_20261008netshape.json`, 14:54:34, `736fcafa…`). The runner and the scorer are
pinned in it.
- TinyLlama, Pi 4 1 + Jetson 21, with the head last or on the host.
- Eight settings, the layout order alternating: unshaped, 200 / 100 / 50 / 20 Mbit/s, +2 / +10 ms RTT, then unshaped again.
- Each setting's link is measured on the shaped path, and the cells are scored against predictions re-priced with it.

| setting | head last (ms/token) | head on the host | predicted faster |
|---|---|---|---|
| unshaped (938 Mbit/s, 0.61 ms) | 95.4 | 101.7 | last, 6.6 % |
| 200 Mbit/s | 99.7 | 102.2 | last, 2.5 % (not judged: under 3 %) |
| 100 Mbit/s | 105.1 | 102.9 | host, 2.1 % (not judged) |
| 50 Mbit/s | 116.0 | 104.2 | host, 11 % |
| 20 Mbit/s | 148.7 | 108.1 | host, 38 % |
| +2 ms RTT | 99.4 | 105.7 | last |
| +10 ms RTT | 115.4 | 121.7 | last |

- **Predicted crossover: 130.4 Mbit/s.**
- **Hypotheses:**
  - NS-FLIP: the faster layout flips as predicted wherever the gap is at least 3 %;
  - NS-CROSS: the measured crossover is within a factor 1.5 of 130 Mbit/s;
  - NS-ACC: decode within 10 % of the re-priced predictions;
  - NS-RTT: decode rises by 1.5–2.5 round trips per token per hop (the predictor charges 2).
- **The scorer was run end to end on synthetic data first.** It recovered a crossover of 130.0 against 130.4.

### The prefill/decode hand-off's network crossover (topic-form scopes 4 and 6), MODELED
`testbed/scripts/pd_network_crossover.py` → `testbed/dataset/pd_network_crossover_20261008.json`.
- **Inputs:** the 09-29 hand-off runs (Qwen3.5-9B; state = 52,691,776 + 32,796 × n bytes, measured exact). Only the transfer,
  S / B, depends on the link.
- **Two users:** only user 2 waits for the transfer. User 1's first token in the hand-off arm (27.8–28.5 s) equals the
  shared arm's (27.8 s), so the transfer overlapped user 1's prefill.

| scenario | the hand-off pays above (TCP goodput) |
|---|---|
| one user, the Pi 4 decoding, Pi 5 or laptop prefill: 128 / 256 / 512 / 1,024-token prompts | 6.9 / 3.7 / 2.1 / 1.3 Mbit/s (laptop: 6.6 / 3.5 / 2.1 / 1.3) |
| two users, 32 tokens back, against the Pi 5 alone (repacked / without repacking) | **34.2 / 22.0 Mbit/s** |
| two users, 128 tokens back, against the Pi 5 alone repacked / without repacking | 42.6 Mbit/s / never: it loses even at gigabit |

- **On these boards the network is not what limits the hand-off.** Wi-Fi 5 GHz measured 167 Mbit/s of raw `nc` goodput
  (20.9 MB/s, Pi 5; "Link tiers measured" above), about four times the highest crossover (42.6 Mbit/s).
- **What decides it is the reply length,** as TWOUSER128b measured.
- This is modeled from measured parts, not measured end to end. The network study's shaped links will show whether a shaped
  link delivers its rate to a bulk transfer, which is the model's one link assumption.

### The reproducibility check (topic-form scope 8): 83 checks, 0 failing
`testbed/scripts/reproduce.py verify` re-runs every scorer on scratch copies and compares each published result leaf by
leaf. Afterwards it checks that nothing in `testbed/dataset/` or `testbed/results/` changed. It covers:
- the dataset csvs, byte for byte;
- 12 pool campaigns, from each cell's own log;
- 16 energy runs and 5 pooled energy sessions, each scored on a copy of its run dir;
- both two-user results;
- the three 09-30 results;
- 43 registrations: read-only, and every quoted sha256 resolves.

**Its first run found three things, all fixed and re-checked:**
1. **The dataset README's `Rebuild:` line was incomplete.**
   - It listed the 10 run dirs of 09-28 (62 rows). The 13 dirs added on 09-29 (the Jetson CUDA and ABA rows, the laptop, the
     9B and pooled-protocol rows) were never added.
   - The order matters: ties keep run order, and `results.csv` keeps run order. It was recovered from the published files
     (each row's `source`, the tie groups, and `results.csv`'s order).
   - The 23-dir command now rebuilds both csvs byte for byte.
2. **`pool_results_20261008loadmode.json` predated the split-check fix applied after the A/B.** It was re-scored from the same
   two logs. One field changed: the `none` cell's Pi 4 buffer, from 0 to 6,089.46 MiB.
3. **The prima.cpp scorer read the GGUF vocabulary with prima.cpp's vendored `gguf-py`**, which needs NumPy < 2; the laptop
   now has 2.3. It now uses the pinned llama.cpp's `gguf-py`, and the result is unchanged.

Also new:
- `reproduce.py export`: `dataset/export/pool_cells.csv`, 57 cells with their metadata. Every cell has board temperatures
  before and after, and the polled cells have the peak. For `lm.q3_14b.none` the snapshots read 55.5 → 67.2 °C, while the
  poll's peak was 83.2 °C.
- `reproduce.py export` also writes `dataset/export/all_figures.csv`: 17,396 figures.
- `reproduce.py manifest`: `dataset/MANIFEST.json`, 523 files.
- `testbed/REPRODUCE.md`.
- Tests: `test_reproduce.py` (2) and `test_netshape.py` (5). The testbed suite is at 70.

### A live dashboard (topic-form scope 10)
`python3 testbed/scripts/dashboard.py [--run DIR]`, at `http://127.0.0.1:8765/` (localhost only), or `dashboard` in
`.claude/launch.json`.
- **It is passive:** it reads the run's own files on the laptop (`strip.csv`, `run.jsonl`) and never contacts a board or the
  strip, so it cannot perturb what it shows.
- **Without `--run`** it follows the newest run dir.
- **Every 3 s it shows:**
  - each outlet's power over the last 15 minutes;
  - each served request's server TPOT (P50) by placement;
  - the run's last event, and per placement the requests and median TPOT so far.
- **Checked** on the 10-08 energy session's strip log, in the browser pane: four outlets, named from `strip.json`.
- Tests: `test_dashboard.py` (2). The testbed suite is at 72.

---

# 2026-10-08 (evening) — the network study, measured: the head placement flips with bandwidth, earlier than predicted

**What ran.** The design is in "The network study (topic-form scope 6), registered and ready" above.
- **Registration:** `testbed/dataset/pool_predictions_20261008netshape_r2.json`, 18:07:02, `f21b9228…`, read-only. It supersedes
  the 14:54 registration, which never ran: its `-rpc` pointed at the dead router's 192.168.100.111.
- **Cells:** TinyLlama, Pi 4 1 + Jetson 21, with the head on the last board or on the host, at eight link settings. Shaping is
  netem on the Pi 4's path to the Jetson.
- **Run:** `testbed/results/netshape_20261008_182208/`, 18:22–18:53. All 16 cells exited 0, all thermally clean, with no page-in.
  `score.json` says no code changed since registration.
- **Scoring:** `compare_pool.py` → `testbed/dataset/pool_results_20261008netshape_r2.json`, and
  `score_netshape_20261008.py` → `score.json` (results `2053c46f…`). Every cell is scored against its prediction re-priced with
  that setting's measured link.
- **A first attempt** (`netshape_20261008_180746/`) was aborted and is kept, not scored. The strip's factory reset cut the
  boards' power.

| setting | link measured: Mbit/s, RTT ms | head last: measured / re-priced | head on the host: measured / re-priced | faster, measured |
|---|---|---:|---:|---|
| unshaped | 939.6, 1.143 | 97.6 / 96.4 (-1.2 %) | 101.4 / 102.8 (+1.3 %) | last, by 3.8 ms |
| 200 Mbit/s | 193.7, 1.200 | 103.1 / 101.0 (-2.0 %) | 102.8 / 103.4 (+0.6 %) | host, by 0.3 ms |
| 100 Mbit/s | 97.9, 1.192 | 108.6 / 106.5 (-1.9 %) | 103.6 / 104.1 (+0.4 %) | host, by 5.0 ms |
| 50 Mbit/s | 50.1, 1.232 | 118.4 / 117.2 (-1.0 %) | 105.5 / 105.4 (-0.1 %) | host, by 12.9 ms |
| 20 Mbit/s | 21.5, 1.271 | 158.1 / 146.2 (-7.5 %) | 113.3 / 109.0 (-3.8 %) | host, by 44.8 ms |
| +2 ms RTT | 930.5, 3.247 | 99.5 / 100.7 (+1.2 %) | 102.8 / 107.0 (+4.1 %) | last, by 3.3 ms |
| +10 ms RTT | 895.4, 11.296 | 107.2 / 116.8 (+9.0 %) | 112.7 / 123.1 (+9.2 %) | last, by 5.5 ms |
| unshaped, again | 941.1, 1.341 | 95.9 / 96.8 (+1.0 %) | 101.3 / 103.2 (+1.8 %) | last, by 5.4 ms |

prefill error range: -24.3 to -11.5 %

The drift between the two unshaped settings, s0r against s0, is −1.7 % (head last) and −0.1 % (head on the host).

**The registered verdicts:**

| hypothesis | rule | result |
|---|---|---|
| **NS-FLIP** | wherever the re-priced gap is ≥ 3 %, the layout predicted faster measures faster | **HOLDS** at all 6 judged settings (200 and 100 Mbit/s were under 3 %, not judged) |
| **NS-ACC** | every cell's decode within 10 % of its re-priced prediction | **HOLDS**: worst 9.2 % |
| **NS-CROSS** | measured crossover within a factor 1.5 of the re-priced 130.4 Mbit/s | **FAILS**: 205.6 Mbit/s, ×1.58 |
| **NS-RTT** | decode rises by 1.5–2.5 ms per ms of added RTT (the predictor charges two round trips) | **FAILS**: slopes 0.94 (last), 1.09 (host) |

**What it answers.**

**1. The best place for the output head depends on the network, and the engine must price the link to choose it.**
- **At gigabit, the head belongs on the last board,** the Jetson: 3.3–5.5 ms faster per token.
- **Below about 200 Mbit/s it belongs on the host:** 5.0 ms faster at 100 Mbit/s, 12.9 ms at 50, and 44.8 ms (28 %) at 20.
- **Why:** with the head last, every token brings the logits back to the host. For TinyLlama that is 128 KB, against 8 KB for
  the hidden state.
- **The logits size scales with the vocabulary,** so the crossover moves with the model. A Qwen-sized vocabulary (152k, about
  608 KB per token) would put it far higher. That is an inference, not measured.

**2. The flip comes earlier than predicted.** Under shaping, the head-last layout's 128 KB per token costs more than its size
divided by the rate.
- **At 20 Mbit/s** it measured 158.1 ms against 146.2 re-priced (+11.9 ms). The head-on-host layout, which moves 8 KB, was only
  +4.3 ms over.
- **A candidate cause,** not tested: TCP restarting its congestion window after each idle gap between tokens. A 128 KB burst then
  needs several round trips to ramp up, while 8 KB fits in the first window.
- **For the engine:** on links below ~200 Mbit/s (Wi-Fi, for example), plan the head on the host.

**3. A worker costs one network round trip per decoded token, not two.**
- **The cause, in the pin's RPC client** (`ggml/src/ggml-rpc/ggml-rpc.cpp`, lines 945–1047): it sends `SET_TENSOR` (the hidden
  state in) and `GRAPH_COMPUTE` one-way (`send_async` with no reply), and waits only on `GET_TENSOR`. The predictor's two round
  trips per worker are a model of an older protocol.
- **Re-pricing with one round trip** (exploratory, not a registered rule):
  - the +10 ms cells' errors go from +9.0 / +9.2 % to −1.6 / −0.8 %;
  - the worst cell error goes from 9.2 % to 8.3 %, and that remaining worst is point 2's low-bandwidth logits.
- **At gigabit** the second round trip is about 1 ms per worker per token, about 1 % on these boards.
- **The fix,** one round trip, has to be registered before a prediction uses it.

**4. Decode accuracy held across the shaped links** (NS-ACC, worst 9.2 %). Prefill was −11.5 to −24.3 % optimistic, the known
TinyLlama pp64 bias; it was not among the hypotheses.

**Caveats.**
- One model (TinyLlama, a 32k vocabulary) and one pair of boards.
- The links are netem-shaped on the Pi 4, not real slow links.
- One session.
- The cluster is on the home router: the isolation deviation recorded for the advisor.

# 2026-10-09 — the full-evaluation pilot, measured: the engine's pick wins in time and energy; the sequential ring, documented

## The pilot (`full_eval.py`, TinyLlama on Pi 4 + Jetson, pre-registered)

**What ran.**
- **Registration:** `testbed/dataset/full_eval_tiny_pilot_20261009.json`, 19:02:42, `918a369a…`, read-only. Its plans,
  predictions, hypotheses and round order are 10-08's (`full_eval_tiny_pilot_20261008.json`, whose run was cut off when its
  session ended); only the hashes of three scripts differ, for the Jetson's new address.
- **Run:** `testbed/results/full_eval_tiny_pilot_20261009_20261009_190242/`, 19:02–19:40, as the transient systemd unit
  `heteroserve-pilot-20261009_185952` (via `run_pilot.sh`, after a passing smoke).
  - Four placements, five rounds in rotated order, four requests per placement per round: 128-token prompts, 96 tokens
    each, greedy, served by llama-server and timed per token.
  - The Jetson's clocks locked for the whole session. The strip logged at 5 Hz, and the Pi 4 was polled every second.
- **Complete:** 80 of 80 requests, each of 96 tokens. No serve failure, the model hash checked on both hosts, the Pi 4's
  poll alive to the end, and no code changed since registration.
- **Scored** by `testbed/scripts/score_full_eval.py` → `score.json`. The scorer was written while round 1 ran, before any
  summary existed.
- **The DHCP reservations were applied during round 1** (19:06–19:10, the user's "Now"). The strip's log has no error and no
  gap over 0.4 s there, and round 1 is in line with the other rounds (below).

| placement | plan | head | predicted | mean | P50 [95 % CI] | P95 [95 % CI] | err | J/token, all boards (predicted) | J/token, own boards | Pi 4 thermal |
|---|---|---|---:|---:|---|---|---:|---|---:|---|
| engine | Jetson 22 | last | 87.0 | 90.6 | 90.5 [90.41, 90.48] | 91.0 [90.93, 91.02] | −4.0 % | 1.2137 (1.1801) | 0.9802 | not used |
| pooled | Pi 4 1 + Jetson 21 | last | 96.3 | 100.8 | 100.8 [100.71, 100.83] | 105.8 [105.28, 106.36] | −4.4 % | 1.3062 (1.2479) | 1.3062 | clean, 20 of 20 |
| halda | Pi 4 1 + Jetson 21 | host | 102.6 | 107.2 | 107.0 [106.84, 107.26] | 111.4 [111.19, 111.63] | −4.3 % | 1.3365 (1.2805) | 1.3365 | clean, 20 of 20 |
| even | Pi 4 11 + Jetson 11 | last | 132.3 | 136.4 | 136.1 [135.1, 136.9] | 140.4 [139.14, 141.6] | −3.0 % | 1.5408 (1.4357) | 1.5408 | clean, 20 of 20 |

Server TPOT in ms; CIs from the two-level bootstrap (rounds, then requests); err = 100 × (predicted / measured mean − 1).
Gross J per decoded token at the wall, over each request's decode window: "all boards" sums both outlets (a board the
placement leaves out still idles), "own boards" only the placement's.

**Round to round** (mean TPOT per round, ms): engine 90.7 / 90.6 / 90.5 / 90.7 / 90.6 (spread 0.2 %), pooled 100.6–101.0
(0.3 %), halda 107.0–107.4 (0.4 %), even 135.2–137.5 (1.7 %).

**The registered verdicts:**

| hypothesis | rule | result |
|---|---|---|
| **FE-TPOT** | the placement predicted fastest has the lowest P50 and P95, its CIs clear of every other's | **HOLDS**: the engine; its CIs end 10.2 ms (P50) and 14.3 ms (P95) below the next placement's |
| **FE-ENERGY** | the placement predicted cheapest (all boards powered) measures the lowest | **HOLDS**: the engine, 1.2137 J; lowest in 5 of 5 rounds |
| **FE-ACC** | every placement's mean TPOT within 10 % of its prediction | **HOLDS**: worst −4.4 % |

**What it answers.**
1. **The engine's own pick, TinyLlama on the Jetson alone, is the fastest and the cheapest when served.** It takes 10.3 % less
   time per token than the pooled plan (P50), with 7.1 % less energy while the Pi 4 idles and 25.0 % less without it. This
   repeats the 10-08 host-choice cell (llama-bench: 9.8 %, 6.7 % and 25.0 %) under serving, with 20 requests per placement.
2. **At gigabit the head belongs on the last board:** the pooled plan takes 5.8 % less time per token than Halda's (P50), as
   in the network study's unshaped cells and the 10-07 head-to-head.
3. **Every placement ran 3.0–4.4 % slower than predicted, in the same direction,** so the ranking held. A candidate cause,
   untested: llama-server's per-token work (sampling and streaming), which the llama-bench rows behind the predictions do
   not include.

**Caveats.** One model, one pair of boards, one session; 128-token prompts only; the cluster is on the home router (the
isolation deviation recorded for the advisor).

## The sequential ring, documented (user: "add sequencial ring both in ieee and progress report")

The 10-07 analysis ("Sequential vs prima.cpp's piped ring", above) is now in the paper (Sections II-C, III-D and IV) and
in report 4 (details items 4 and 5). The facts were re-checked at the pins today:
- **Pipeline parallelism** (`src/llama-context.cpp:428–453` at `b23701f7`) needs more than one device, every layer
  offloaded, the layer split, offloaded KQV and no tensor override, and async and events on every non-CPU device. The RPC
  device reports both (`ggml-rpc.cpp:2184–2190`). So it was enabled in exactly two logged runs, the 09-30 llama.cpp default
  and RPC-README cells (`pool_d.q3_14b.{default,readme}_20260930_*.log`, both all-offloaded). There it had nothing to
  overlap: it overlaps the micro-batches of one batch, and those runs had `n_batch` = `n_ubatch` (64 for pp64, 16 for
  tg16) with one sequence. Every pooled pass in this work is one micro-batch.
- **Between two RPC workers, activations pass through the host:** a copy between different servers returns false
  (`ggml-rpc.cpp:752–761`), so the scheduler reads from one and writes to the other. The ring is logical; physically it is a
  star around the host.
- **prima.cpp's ring is physical:** each rank sends to `--next-node-ip`, and `-k/--n-cycles` laps it k times per token
  (`common/arg.cpp:749, 773` at `288df6a2`).

# 2026-10-09 (evening) — "do all 5": RTT1 registered and adopted; the ring laps measured

## 1. The predictor rule RTT1, registered (`testbed/dataset/rule_rtt1_20261009.json`, 20:08:34, `f5f2b768…`)
- **The rule:** an RPC worker costs one network round trip per token, not two. `predict_pool.predict`, `moe_prefill`,
  `plan_host_choice`, `plan_fixed` and `energy_model` take `rtt_trips`. The default stays 2, so every earlier registration
  and scorer reproduces unchanged: `reproduce.py verify` gave 85 checks, 0 failing, after the change.
  `predict_pool.py --rtt1` and `full_eval.py register --rtt-trips 1` apply it. Tests: `testbed/tests/test_rtt1.py`.
- **Its registered test:** the ring-lap study below, statistic c (decode's increase per Jetson visit, divided by the RTT
  increase), with RTT1 adopted for c in [0.75, 1.25].
- **The caveat it carried:** at gigabit the second round trip had been absorbing real per-visit overhead. RTT1 corrects the
  round-trip coefficient only.

## 2. The ring laps: is sequential faster? Measured, not modeled (`ring_laps_predictions_20261009.json`, 20:16:31, `a6bfa8f1…`)
**What ran** (`testbed/results/ring_laps_20261009_201631/`, 20:16–20:29, `ring_laps.sh` as a systemd unit, scored by
`score_ring_laps_20261009.py`):
- **The model and split:** TinyLlama Q4_K_M, the Pi 4 holding 4 blocks and the Jetson 18, cut into 1, 2, 3 or 4 alternating
  stages, so each token visits the Jetson k times.
- **Every stage an RPC device:** the Pi 4's as ggml-rpc-servers on its loopback, the Jetson's on its eth0 (8 servers). llama-bench
  on the Pi 4 holds only the embedding.
- **The same bytes in every layout:** the Pi 4's four blocks are two of each of TinyLlama's two block sizes (26.6 and 23.6 MiB),
  so only the visits change. Every cell's device buffers matched the design within 0.5 MiB (checked by the scorer).
- **Settings:** unshaped, and +5 ms RTT on the Pi 4 <-> Jetson path only (`netshape.sh apply 0 2.5`; ping 0.70–0.73 against
  5.72–5.73 ms, +5.02 ms).
- **The schedule:** 3 rounds, both settings in each, the layouts rotated. `-t 4 -ngl 99 -p 64 -n 32 -r 3`. A smoke cell checked
  the plumbing first (`ring_laps_smoke_20261009_201508/`, one repetition, not scored).
- **Complete:** 24 of 24 cells. The Pi 4 was clean throughout (peak 64.2 °C), and no code changed since registration.

| laps (Jetson visits) | decode unshaped, ms/token: mean (round range) | predicted RTT1 / RTT2 | decode +5 ms RTT | prefill unshaped, ms/token |
|---|---|---|---|---|
| 1 | 109.78 (108.70–111.16) | 106.13 / 107.26 | 115.33 | 27.99 |
| 2 | 113.80 (112.17–114.63) | 107.41 / 109.65 | 124.67 | 28.25 |
| 3 | 117.42 (115.86–118.74) | 108.68 / 112.05 | 136.29 | 28.57 |
| 4 | 123.41 (123.08–123.69) | 109.96 / 114.45 | 145.53 | 29.14 |

**The registered verdicts:**

| hypothesis | rule | result |
|---|---|---|
| **RL-SEQ** | one lap fastest in every unshaped round, and a positive slope over laps | **HOLDS**: one lap fastest in 3 of 3 rounds; **+4.45 ms per extra lap (4.1 %)** |
| **RL-RTT** | c in [0.75, 1.25] for RTT1, [1.75, 2.25] for two trips | **RTT1 HOLDS**: c = 1.147 (rounds 1.183, 1.191, 1.067) |
| **RL-ACC** | every unshaped layout within 10 % of its RTT1 prediction | **FAILS**: −3.3, −5.6, −7.4, −10.9 % (two trips: −2.3 to −7.3 %) |

**What it answers.**
1. **Sequential is faster, measured.** With the model in memory, each extra lap of the ring costs 4.45 ms per token at
   gigabit, and the single lap won every round. That answers the 10-05 meeting's "try sequential", and it is the answer the
   10-07 model gave. The model's range was 0.6–2.4 ms by link terms and 2.2–4.4 ms with the 09-27 measured RPC visit; the
   measurement sits at the top of the second.
2. **RTT1 is adopted.** Each Jetson visit costs 1.15 round trips of added latency, close to one and far from two. New
   registrations pass `rtt_trips=1`; the default stays 2 for reproducibility.
3. **The predictor misses a per-visit cost.** A lap costs 4.45 ms, and its link terms only 1.28 ms (RTT1), so about 3.2 ms per
   lap is fixed overhead per worker visit: the Jetson's visit plus a loopback one. Two visits of the 2.2 ms per-visit cost
   measured on 09-27 come to 4.4 ms (an inference, not a separate test). Hence RL-ACC's growing miss: the predictor carries one
   visit's overhead only inside the Jetson's 2.5 % RPC factor.
   - **For single-lap pools,** the usual layout, that error stays small (−3.3 % at one lap).
   - **A per-visit term** would fix multi-visit layouts. It has to be registered and tested on fresh cells before use.
4. **Prefill pays per lap too:** +0.38 ms per prompt token per lap. Each visit moves the whole batch's hidden states
   (64 × 8 KB) both ways.

**Caveats.** One model and one pair of boards; laps built from RPC servers on one host's loopback, the closest llama.cpp gets
to prima.cpp's piped ring (prima.cpp itself cannot run on today's boards). The unshaped rounds spread 2.2–2.5 %, below the
3.6 % lap step, and the single lap won every round.

## 3. The TCP diagnostic: why the logits seemed to cost more on a slow link (`pool_predictions_20261009tcpdiag.json`, 20:27:17, `854f1e4a…`)
**What ran** (`testbed/results/tcpdiag_20261009_202921/`, 20:29–20:45, `tcpdiag_campaign_20261009.sh` as a systemd unit):
- **The cells:** the network study's two layouts (TinyLlama, Pi 4 1 + Jetson 21, the head last or on the host), at 20 Mbit/s and
  unshaped, `-p 64 -n 64 -r 3`, through `run_pool.sh` (every gate, memory probe and throttle poll).
- **The sampler:** around each cell, `ss_sample.sh` read the RPC connection's tcp_info on both boards (`ss -tin`, about 17 Hz,
  read-only).
- **Nothing reconfigured** beyond the study's shaping. Both boards keep slow start after idle (`tcp_slow_start_after_idle` = 1,
  CUBIC). The Jetson has no pacing qdisc (`pfifo_fast`) and a 16 KB initial send buffer.
- **Scored** by `score_tcpdiag_20261009.py` within each cell's decode window, taken from llama-bench's `test_time`. All four
  cells exited 0, and the Pi 4 was clean.

| cell | decode ms/token | re-priced at measure_link's rate | Jetson cwnd in flight | retransmits | Pi 4 rcv_wnd | Jetson delivery rate |
|---|---:|---:|---|---:|---:|---:|
| 20 Mbit/s, head last | 159.68 | 146.1 (−8.5 %) | 40–87, median 87 | 0 | 213 KB | 19.2 Mbit/s |
| 20 Mbit/s, head on the host | 116.01 | 109.0 (−6.0 %) | 17 | 0 | 1.06 MB | — (8 KB messages) |
| unshaped, head on the host | 104.45 | 102.8 (−1.6 %) | — | 0 | 1.07 MB | — |
| unshaped, head last | 97.33 | 96.5 (−0.9 %) | 92 | 0 | 1.05 MB | 673 Mbit/s |

**The registered verdict.** **TD-SSAI HOLDS:** slow start after idle is not what slows the logits.
- A token (159.7 ms) is shorter than the Jetson's RTO on the connection (median 248 ms), so no idle gap between tokens reaches
  it.
- No sample with data in flight shows the restart window: in-flight cwnd 40–87, never 10.
- There were no retransmissions, and the Pi 4's advertised window (213 KB) exceeds a 128 KB burst.

**What the data say instead** (EXPLORATORY, post hoc; `testbed/scripts/explore_tcpdiag_20261009.py`):
1. **The link figure overstated the shaped rate.**
   - **How:** `measure_link.sh` times `dd` on the sender, so bytes still in socket buffers count as delivered. That is under 1 %
     for 1 GiB at gigabit, but the campaigns send only `rate` MiB at each shaped rate.
   - **By how much:** against the TCP payload rate netem allows (rate × 1448 / 1514), it read **+1.3 % at 200 Mbit/s, +2.4 % at
     100, +4.8 % at 50 and +12.4 % at 20** (10-08), and +12.9 % at 20 today. The Jetson's own delivery rate in the logits bursts,
     19.2 Mbit/s, matches the payload rate (19.1), not `measure_link` (21.6).
2. **At the payload rate the logits cost nothing extra.**
   - The 20 Mbit/s cells re-priced at 19.1 Mbit/s: head last −3.5 % (10-08) and −4.5 % (10-09), head on the host −3.1 % and
     −5.3 % (rtt_trips 2).
   - Both layouts are now similarly optimistic, as the pilot's placements were (3–4.4 %). The 10-08 head-last miss of −7.5 % was
     mostly the overstated link.
3. **The crossover follows the gigabit gap, which varies by session.**
   - **The mechanism:** with the transfer as modeled, the head placement flips where the logits' extra 958,464 bits a token cost
     as much as the head's gap at gigabit, g0 (host − last).
   - **How far g0 moved:** 3.80 ms in the 10-08 study's unshaped cells, 5.40 in its unshaped repeat, 6.21 in the 10-09 pilot
     (served) and 7.12 in today's unshaped cells. Those put the crossover at 199, 149, 133 and 118 Mbit/s.
   - **The model's own g0, 6.33 ms,** gives exactly its 130.4 Mbit/s, inside that range.
   - **So NS-CROSS's miss** (205.6 against 130.4, registered and standing as found) came from the 10-08 session's small gigabit
     gap, not from a mispriced link term.

**What changes.**
- Shaped links should be measured at the receiver, or with transfers much larger than the socket buffers. A change to
  `measure_link.sh`, to make before the next shaped campaign.
- Below 200 Mbit/s, plan the head on the host as before; the crossover is better quoted as 120–200 Mbit/s than as one figure.

**Caveats.** One model, two layouts, one session for this diagnostic. The payload-rate correction assumes netem charges whole
Ethernet frames, which the Jetson's measured delivery rate supports at 20 Mbit/s.

## 4. Qwen3-14B, the full evaluation's next model (`full_eval_q14b_20261009.json`, 20:45:31, `687d1a1c…`)
**What ran** (`testbed/results/full_eval_q14b_20261009_20261009_204531/`, 20:45–23:08, `run_14b.sh` as the systemd unit
`heteroserve-q14b-204524`, RTT=1):
- **The model and placements:** Qwen3-14B Q4_K_M on Pi 4 + Jetson, three placements fixed by memory before registering.
- **The schedule:** 5 rounds in rotated order, 2 requests per placement per round, at a 4,096 context.
  - Each request was a 128-token prompt and 32 tokens, greedy.
  - The warm-up was 64 + 8, and the weights were read into memory.
- **Conditions:** the Jetson's clocks locked for the session, the strip at 5 Hz, and the Pi 4 polled every second.
  - The model hash was checked on the Pi 4.
  - No code changed since registration.
- **Scoring:** `score_full_eval.py` → `score.json`.

| placement | plan | head | predicted | mean | P50 [95 % CI] | P95 [95 % CI] | P99 | err | J/token, all boards (predicted) | TTFT P50 |
|---|---|---|---:|---:|---|---|---:|---:|---|---:|
| engine | Pi 4 27 + Jetson 13 | host | 2,025.7 | 1,976.5 | 1,990.1 [1,945.9, 1,993.7] | 1,999.9 [1,996.5, 2,001.7] | 2,003.1 | +2.5 % | 21.11 (20.67) | 102.9 s |
| p32last | Pi 4 32 + Jetson 8 | last | 2,077.2 | 2,039.5 | 2,042.9 [2,037.7, 2,045.5] | 2,055.2 [2,049.7, 2,070.7] | 2,084.4 | +1.8 % | 21.86 (20.94) | 118.1 s |
| p32host | Pi 4 32 + Jetson 8 | host | 2,159.4 | — | — | — | — | — | — (21.37) | — |

Server TPOT in ms (310 tokens per placement); err = 100 × (predicted / measured mean − 1); gross J per decoded token at the
wall, both outlets. p32host never served: see point 2.

**The registered verdicts.** All three are **INCOMPLETE**, because one registered placement produced no data. On the data present
all three hold:
- **FE-TPOT:** the engine's pick has the lowest P50 and P95, its CIs 44.1 ms (P50) and 48.0 ms (P95) clear of p32last's.
- **FE-ENERGY:** the engine's pick is the cheapest, 21.11 J per decoded token against 21.86, and the lowest in 4 of 5 rounds.
- **FE-ACC:** both served placements are within 2.5 % of prediction, both slightly faster than predicted.

**What it answers.**
1. **The engine's pick wins again.** Its 27 + 13, with the head on the host, beats the only head-last split that fits the Jetson
   (32 + 8).
   - **Time:** 2.6 % less per token (P50) and 12.9 % sooner to the first token (102.9 against 118.1 s).
   - **Energy:** 3.4 % less per token. It was predicted at 2.5 % and 1.3 %; measured energy ran 2.1–4.4 % above prediction.
2. **p32host does not fit: the memory accounting was optimistic here.**
   - The Pi 4's kernel OOM-killed llama-server during load in every round (dmesg: `Killed process … (llama-server)`,
     anon-rss 7.58 GB).
   - The load logged a 7,022 MiB CPU model buffer and a 512 MiB KV cache at the 4,096 context.
   - The accounting gave the plan 7,254 of the Pi 4's 7,413 MiB, so it under-estimated by at least 148 MiB: 7,402 MiB
     anonymous at the kill, still loading. That is beyond the worst registered error (110 MiB).
   - The live gate of `serve_pool.sh` passed the plan because it checks the same accounting.
   - **The engine's default 256 MiB headroom would have refused this plan** (7,254 + 256 > 7,413). The registration chose it as
     the tighter alternative.
   - **To find:** what the accounting misses for a Pi 4 host serving at a long context with the head on the host. Candidates
     are the output and compute buffers at llama-server's n_batch 2048 / n_ubatch 512, which the llama-bench cells behind the
     accounting never used.
   - Until then, do not plan a Pi 4 host below the default headroom.
   - **Found the same night:** the token embedding (417 MiB) of a host that reads its weights into memory, which the charge
     missed. Fixed, and the registration now refuses p32host: "engine improvements", section 1, below.
3. **A 14B on the Pi 4 + Jetson decodes at about 0.5 tokens/s:** 1.98 s per token, against 0.85 s on Pi 5 + Jetson (09-30).
   The Pi 4 holds most of the model and is the slowest board per block.
4. **Thermal: the Pi 4 hit its soft temperature limit in prefill only.**
   - Its peak was 83.7 °C, with the ARM clock down to 1,629 MHz. The limit was active for 75 s of the engine's 1,036 s of
     prefill (7 %) and 208 s of p32last's 1,193 s (17 %).
   - It was **never active during decode** (0 of 614 and 631 s).
   - So the request windows read THROTTLED (6 of 10 and 10 of 10), but the decode figures the hypotheses judge are clean.
   - The TTFT figures carry it. p32last puts more blocks on the Pi 4, so its larger share is part of the plan.

**Caveats.** One model, one pair of boards, one session; 2 requests per placement per round (a 14B at about 2 s a token); the
home router (the isolation deviation).

# 2026-10-09 (late) — engine improvements: the loaded host's embedding, receiver-timed links, VISIT tested

## 1. The memory accounting: a host that reads its weights into memory pays its token embedding
**What was wrong.** Qwen3-14B's `p32host` (Pi 4 32 + Jetson 8, head on the host) was OOM-killed in all five rounds while the
accounting gave it 7,254 of the Pi 4's 7,413 MiB (section 4 above).
- **The cause:** the host's charge was fitted mapped (09-27/28). A mapped host holds only the token-embedding rows a prompt
  touches, about 30 MiB, and `host_overhead_mb` absorbed them. Since 10-08, `serve_pool.sh` serves with `-lm none`, which reads
  the weights into memory, so every tensor the host keeps is anonymous memory, the embedding included: 417 MiB for Qwen3-14B
  (437,575,680 B).
- **The evidence:** the Pi 4's CPU buffers, as llama-server logged them, sat 427 MiB above the charge in all three placements,
  and 417 of that is the embedding.

**The fix** (`ThinkPad/control/placement.py`):
- **The flag:** a new `ModelProfile.host_loads_weights`, a launch property like `context_len`. It defaults to False, so every
  registration on file reproduces.
- **The charge:** `charged_bytes` adds `embd_bytes` to a host that holds the embedding and loads its weights.
- **The upload copy:** `_copy_fits` counts it twice for such a host: the loader's staging buffer (`llama-model-loader.cpp:1739`
  at the pin) plus RPC's serialized message.
  - It is still checked apart from the steady footprint, because both copies are freed before the next tensor, and before the
    KV cache and compute buffers exist.
  - All seven model files on the Pi 4 store `output.weight` first and the embedding before block 0, so the head's copy comes
    and goes before any block is read.
- **The plumbing:** `predict_pool.model_profile`, `plan_host_choice` and `plan_fixed` take `load_mode`. `serve_pool.sh` and
  `full_eval.py` always price `"none"`, the way they serve.

| placement (4,096 context) | Pi 4 charge before | Pi 4 charge now | Pi 4 buffers logged by llama-server | Pi 4 available at rest | now |
|---|---:|---:|---:|---:|---|
| engine, 27 + 13, head on the host | 6,241 | 6,658 | 6,668.05 | 7,413 | fits, 755 MiB to spare |
| p32last, 32 + 8, head last | 6,645 | 7,062 | 7,072.02 | 7,413 | fits on the Pi 4 (351 MiB); the Jetson keeps 149 MiB beyond its 450 MiB reserve |
| p32host, 32 + 8, head on the host | 7,254 | 7,671 | 7,022 + 512 KV, killed loading | 7,413 | **refused, 258 MiB short** |

MiB. Logged means the CPU model, KV cache, compute and output buffers; a served placement now lands within 10 MiB of them, below.

**The registration gate** (`full_eval.py register`):
- It records each placement's memory on every board, and the load mode.
- It refuses a placement that does not fit at rest.
- It refuses one inside the engine's 256 MiB headroom unless `--below-headroom` is given, which the registration records.
- So 10-09's p32host would now be refused at registration, and p32last would need `--below-headroom` (the Jetson's 149 MiB).

**Unchanged:** the engine's pick and every timing prediction. `reproduce.py verify`: 89 checks, 0 failing.
Tests: `ThinkPad/tests/test_placement.py::TestLoadedHost` (8) and `testbed/tests/test_loaded_host.py` (5).

**Not yet seen live:** `serve_pool.sh` now prints a `mem` line after READY: each board's MemAvailable at READY against the gate,
and the charge. No real serve has run it yet.

**A correction to 10-08's "checked literally"** (the load-mode A/B, above): that compared a 6,228 MiB MemAvailable drop with a
6,241 MiB charge, and the match was a coincidence.
- llama-bench allocates the KV cache only for its test context, so the charge's KV at a 4,096 context (432 MiB) stood in for the
  uncharged embedding (417).
- A charge has to be compared at the cell's real context.

## 2. `measure_link.sh` times at the receiver
**Why:** the TCP diagnostic (section 3 above) found that sender-side `dd` overstated shaped rates, by +12.4 % at 20 Mbit/s.

**The change:**
- The receiving board runs a small python3 reader, sent base64-encoded. It prints `RECV a->b N bytes, first F, S s from the
  first byte to EOF`.
- `link_to_json.py` takes (N − F) × 8 / S where RECV lines exist, and keeps the sender's figures beside them
  (`sender_mbps_each`).
- It refuses a log whose RECV lines do not match its transfers one for one.
- All 26 older logs parse identically: the old and new `parse` were compared on each.

**Checked on the boards:**

| run | receiver, Mbit/s (pi4→jetson / jetson→pi4) | sender, Mbit/s | expected |
|---|---|---|---|
| unshaped, 256 MiB × 3 each way (`link_pi4-jetson_20261009_234258.log`) | 935.7 / 938.1 | 910–946 | gigabit |
| 20 Mbit/s (`netshape.sh apply 20 0`, then cleared; `check` exit 0; `…_234358.log`) | 19.1 / 19.1 | 22.0 / 21.6 | 19.13, the TCP payload rate netem allows (20 × 1448 / 1514) |

At 20 Mbit/s the receiver reads the payload rate, and the sender still reads 13–15 % high. `link.json` was not rebuilt from these
logs; they check the tool, and the pair's figure is unchanged.

## 3. VISIT: a fixed cost per worker visit — registered, tested on fresh cells, NOT adopted (`visit_predictions_20261009.json`, 23:53:45, `b815e419…`)
**The rule under test:**
- Each worker visit costs a fixed 1.587 ms of decode per token beyond RTT1's link terms (`predict_pool.VISIT_MS`, applied
  with `predict(..., visit_ms=VISIT_MS)`).
- It was fitted in sample on the ring laps: a lap of two visits, one loopback and one Jetson, cost 3.173 ms beyond its link
  terms, split evenly here.

**The cells** (`register_visit_20261009.py`):
- TinyLlama Q4_K_M, with the Pi 4 holding blocks 0–3 and the Jetson blocks 4–21 and the output layer in every layout, so only
  the stages change.
- The two kinds of visit, which the ring laps always added together, are separated, and the layout real pools use is
  included: the host computing its own blocks.
- `ring_laps.sh` at s0 (the link as it is), 3 rounds, the layouts rotated, `-t 4 -p 64 -n 32 -r 3`.

**What ran:**
- `testbed/results/ring_laps_20261009_235345/`, 23:53–00:02, as the systemd unit `heteroserve-visit-235345`.
- 18 of 18 cells; every device buffer matched the design within 0.5 MiB.
- The Pi 4 was clean (peak 64.2 °C, the soft limit never active), and no code changed since registration.
- A smoke cell checked the plumbing first (`ring_laps_smoke_20261009_235216/`, h4, not scored).

| layout | stages (H the Pi 4 in llama-bench, P a Pi 4 loopback server, J a Jetson server) | Jetson / loopback visits | decode ms/token: mean (round range) | RTT1 | VISIT | err RTT1 | err VISIT | prefill ms/token |
|---|---|---|---|---:|---:|---:|---:|---:|
| h1 | H4 · J18 | 1 / 0 | 108.37 (107.87–109.15) | 105.83 | 107.41 | −2.35 % | −0.89 % | 27.92 |
| h2 | H4 · J9 · J9 | 2 / 0 | 111.20 (110.79–111.58) | 107.05 | 110.22 | −3.73 % | −0.88 % | 28.34 |
| h4 | H4 · J5 · J5 · J4 · J4 | 4 / 0 | 112.49 (111.37–113.86) | 109.49 | 115.83 | −2.67 % | +2.97 % | 29.18 |
| a1 | P4 · J18 | 1 / 1 | 108.99 (108.81–109.29) | 106.13 | 109.31 | −2.63 % | +0.29 % | 27.94 |
| p4 | P1 · P1 · P1 · P1 · J18 | 1 / 4 | 117.68 (117.37–118.19) | 106.30 | 114.23 | −9.67 % | −2.93 % | 28.21 |
| j3 | P4 · J6 · J6 · J6 | 3 / 1 | 112.06 (112.00–112.08) | 108.57 | 114.92 | −3.11 % | +2.56 % | 28.47 |

err = 100 × (predicted / measured − 1). Ping Pi 4 → Jetson by round: 0.711, 1.732, 1.163 ms. a1 is the ring laps' one-lap
layout: 108.99 ms here against 109.78 three and a half hours earlier.

**The registered verdicts** (`score_visit_20261009.py` → `score.json`):

| hypothesis | rule | result |
|---|---|---|
| **VISIT-J** | the slope of decode over Jetson visits in h1, h2, h4, minus RTT1's predicted slope, in [1.087, 2.087] ms | **FAILS**: 1.269 − 1.220 = **0.049 ms** |
| **VISIT-P** | the same over loopback visits in a1 and p4 (informative only) | **FAILS**: 2.894 − 0.057 = **2.837 ms**, above the band |
| **VISIT-MAE** | VISIT's mean absolute decode error over the six layouts below RTT1's | **HOLDS**: 1.75 % against 4.03 % |

**Adoption (VISIT-J and VISIT-MAE): NOT ADOPTED.** New registrations keep RTT1 with `visit_ms` 0.

**What it answers.**
1. **A Jetson visit costs what RTT1's link terms say.** In the layout real pools use, with the host computing its own blocks,
   each extra Jetson visit added 1.27 ms per token, against RTT1's predicted 1.22. There is no fixed per-visit cost to add.
2. **The per-visit cost the ring laps found belongs to the loopback servers.**
   - Each Pi 4 loopback visit cost 2.89 ms per token, against 0.06 ms of link terms.
   - The two slopes add to 4.16 ms per lap of one of each, against the 4.45 ms per lap the ring laps measured.
   - In prefill a loopback visit costs 0.09 ms per prompt token (p4 against a1): a fixed cost per call, spread over the batch.
   - Loopback servers exist only in these experiments, where they stand in for a ring's host windows.
3. **So the ring laps overstate what an extra lap costs a real ring** (a reading of the slopes, not a registered test).
   - A ring whose host computes its windows in-process pays the Jetson visit, about 1.3 ms (1.2 %) per extra lap, not 4.45
     (4.1 %).
   - Sequential still wins, by less. The ring laps' numbers stand as measured, for their loopback layout.
4. **RTT1 stays the rule, and its remaining error is a constant** (exploratory).
   - Fitted on all 18 cells, the residual over RTT1 is 2.24 ms + 0.13 ms per Jetson visit + 2.08 ms per loopback visit.
   - RTT1 under-predicts every layout by about 2.2 ms (2 %), whatever its Jetson visits. That is the size of the cost table's
     known drift, and it is not attributed.
   - VISIT-MAE held because VISIT's per-visit term happened to cover that constant at one or two visits. It over-predicts at
     three and four Jetson visits (+2.6 to +3.0 %).
5. **Prefill pays per Jetson visit:** 0.42 ms per prompt token per visit in the h series, close to the ring laps' 0.38 per lap.

**Caveats.**
- One model, one pair of boards, one session, and the link as it is (no shaping).
- The loopback visit's cost is measured, not explained. One candidate, untested: a Pi 4 ggml-rpc-server's four threads share
  the cores with llama-bench's four.

# 2026-10-10 — scheduler upgrade 1 (speculative decoding), step 1: what checking k tokens costs the pooled 14B

**Why.** The survey (`Documents/HeteroServe_Scheduler_Research.html`, upgrade 1) put speculative decoding first. A small
model drafts tokens, and the pooled Qwen3-14B checks several in one pass. Its estimate of 1.2–1.7× hinged on one unknown:
does checking k tokens cost one pass, or k passes, on the Pi 4?

**The registration** (`spec_pass_predictions_20261010.json`, 01:12:29, `f9ea92f3…`; `register_spec_pass_20261010.py`).
- **Two models from single-board rows bound the answer:**
  - ADD(k) = decode + (k − 1) × the plan's compute per token: memory and compute never overlap;
  - ROOF(k) = Σ per board max(decode, k × compute) + link: weights are read once, and compute hides under it.
- **Logits are charged for every checked token,** because a check needs them all. On this plan the 638 MB output head is on
  the Pi 4.

**What ran** (`testbed/results/spec_pass_20261010_011447/`, 01:14–01:29, the systemd unit `heteroserve-specpass-011447`).
- **The plan:** the engine's own, as served on 10-09: Pi 4 host 27 blocks plus the head, Jetson 13 on Vulkan, weights read
  into memory.
- **The tool:** `run_pool.sh` with the new `bench.tool` `llama-batched-bench`. After a shared 8-token prompt, 8 generation
  steps each decode a batch of k tokens with logits for all of them, the work of a check. llama-bench's pp tests compute
  only the last token's logits.
- **The schedule:** k = 1..5 listed three times, so the repetitions interleave.
- **Conditions:** the Pi 4 was clean (peak 78.8 °C, the soft limit never active), the bench exited 0, and no code changed
  since registration.

| k (tokens per check) | T(k) ms per pass: mean (range of 3) | × T(1) | ADD | ROOF | E(k − 1; 0.7), tokens a check yields |
|---:|---|---:|---:|---:|---:|
| 1 | 1,928.4 (1,928.1–1,929.0) | 1.000 | 2,025.7 | 2,025.7 | 1.00 |
| 2 | 2,367.2 (2,365.9–2,368.2) | 1.228 | 2,884.3 | 2,026.1 | 1.70 |
| 3 | 3,416.3 (3,384.6–3,476.7) | 1.772 | 3,742.8 | 2,763.4 | 2.19 |
| 4 | 4,472.7 (4,428.8–4,558.5) | 2.319 | 4,601.4 | 3,561.0 | 2.53 |
| 5 | 5,566.6 (5,565.8–5,567.0) | 2.887 | 5,460.0 | 4,358.7 | 2.77 |

**The registered verdicts** (`score_spec_pass_20261010.py` → `score.json`):

| hypothesis | rule | result |
|---|---|---|
| **SP-K1** | T(1) within 5 % of the predicted decode step (2,025.7 ms) | **HOLDS**: 1,928.4 ms (−4.8 %) |
| **SP-BAND** | every T(k), k = 2..5, within [0.95 × ROOF, 1.05 × ADD] | **HOLDS** |
| **SP-PAYS** | T(k) / T(1) below E(k − 1; 0.7) for some k | **HOLDS** for k = 2, 3, 4 (not 5) |

**What it answers.**
1. **The first extra token is cheap, and the rest are not.**
   - Checking 2 tokens costs 439 ms more than 1, half of the plan's predicted compute per token (859 ms).
   - Each token after that costs 1,049–1,094 ms, about 1.25 × that compute.
   - ADD fits overall (mean absolute error 8.3 %, the overlap φ = 0.99); ROOF is 14–22 % optimistic. Why batch 2 is cheap
     and batches of 3–5 are dearer per token than the 64-token prefill rate is not tested. One candidate: small-batch
     kernels on the Pi 4, which has no dot-product instructions.
2. **One drafted token per check is best at every acceptance rate from 0.5 to 0.9.** The ceiling, with a free drafter,
   is 1.22× at 50 % acceptance, 1.30× at 60 %, 1.39× at 70 %, 1.47× at 80 % and 1.55× at 90 %.
3. **So the survey's 1.3× bar needs two things:**
   - first-position acceptance near 0.6 or better;
   - a drafter that costs little beside a 2.4 s check. On the Pi 4 a 0.6B draft token would cost about 180 ms (an estimate
     from its decode rate), and the bar then needs about 0.72.

   Where the drafter runs is therefore a real placement decision, as the survey framed it.

**Caveats.**
- One model, one plan, one session, and a shallow context (8–16 tokens deep).
- `llama-batched-bench` checks k parallel sequences, not one sequence's consecutive positions. The weight reads and matmuls
  are the same; the attention differs and is small at this depth.
- **Not checked:** the registered 0.5 MiB buffer check. `llama-batched-bench` logs only warnings by default, so the Jetson's
  buffer line is absent; the next such cell should pass `-lv 3`. The memory probes are consistent with the plan: the
  Pi 4's process peaked at 6,308 MiB anonymous (about 6,320 expected), and the Jetson's server at 2,645 MiB (2,490 MiB of
  blocks plus compute).
- The first round's k = 3 and k = 4 passes ran 2.7–2.9 % slower than the next two. That is reported as found; one
  candidate is the Jetson compiling its Vulkan pipelines on first use of those batch sizes.

# 2026-10-10 — scheduler upgrade 2: a correction layer on the predictor, registered and tested — NOT adopted

The survey's upgrade 2 (`Documents/HeteroServe_Scheduler_Research.html`): keep the analytic predictor, and add a linear
correction of its relative decode error on five features named in advance, wrapped in conformal intervals. Registered as
`testbed/dataset/rule_correction_20261010.json` (2026-10-10 02:01:36, `6d6b771f…`) before any fit, with the table embedded
and the scorer's code fixed. Laptop only: nothing ran on the boards.

**The table** (`correction_layer.py`): 118 decode measurements, each against the prediction registered for its campaign.
- 70 pool cells (`pool_results*.json`; two of them registered in addenda), 24 ring-lap cells (RTT1), 18 VISIT cells (RTT1, the
  predictor without the rule under test) and 6 served placements (the 10-09 pilot and the Qwen3-14B evaluation).
- 7 cells were left out: the crashed MoE cell (exit 134), the two forced-thrash cells and the TCP diagnostic's four. None of
  these has a measured decode in its results file.
- Features: `hybrid` and `moe` (0/1); `kv_frac`, the KV bytes read per token at the mean context depth over the weight bytes
  read (full-attention blocks only for the qwen35 hybrids, one in four); `pi_temp`, max(0, T − 60 °C) for the hottest Pi
  (the 1 Hz poll's peak, else its end-of-cell STATE temperature; 0 without a Pi); `visits`, the RPC stages per token.
- The predictions mix versions as registered: two trips per worker before 10-09, RTT1 after.

**The registered test, CORR-LOFO** (leave one model family out; the family-mean |error| must fall by at least 30 %): **FAILS.**

| family | rows | |error| uncorrected | corrected (fit without the family) |
|---|---:|---:|---:|
| tinyllama-1.1b | 67 | 4.98 % | 9.96 % |
| qwen3-14b | 19 | 2.56 % | 2.68 % |
| qwen2.5-14b | 10 | 3.48 % | 3.16 % |
| qwen3.5-9b | 8 | 6.66 % | 4.98 % |
| mistral-nemo-12b | 5 | 3.38 % | 4.53 % |
| qwen3-30b-a3b | 4 | 8.58 % | 5.27 % |
| mistral-7b | 3 | 11.26 % | 10.40 % |
| qwen3.8-27b | 2 | 9.77 % | 2.36 % |
| **family mean** | 118 | **6.33 %** | **5.42 %** (ratio 0.856; needed ≤ 0.70) |

- Row-weighted, it got worse: 4.87 → 7.37 %.
- Without Qwen2.5-14B, the figures for `Documents/`: family mean 6.74 → 5.74 % (ratio 0.852), row-weighted 5.00 → 7.76 %.
- **Not adopted.** CORR-COVER, the prospective coverage test of the intervals, applies only after adoption, so it does not
  run. For the record, the CV+ 90 % half-width would have been 5.2 %, and 21.3 % for a family the table has never seen.
- The all-rows fit, for reference only: intercept −0.12 %; hybrid +6.4 %; MoE +5.4 %; `kv_frac` × 6.8; `pi_temp`
  −0.25 % per °C; visits +1.5 % per visit.

**Why it failed** (exploratory, post hoc):
- **A visit has a fixed cost in milliseconds**, so as a share of the prediction it depends on model size: about 1.4 % of
  TinyLlama's ~110 ms per token, 0.08 % of a 14B's ~2 s.
- **Without TinyLlama, the fit sees only big models with 0–3 visits.** There it learned −2.6 % per visit, from the two-trip
  era's over-priced round trips.
- **On TinyLlama's 4–8-visit lap rows that sign is wrong:** their corrected error rises to +12, +22 and +33 % at 4, 6 and
  8 visits.
- **What a next registration would need:** the visit and constant terms entered as milliseconds over the prediction
  (visits ÷ predicted ms), not as counts. That is a new registration and a new test, not a re-score of this one.

**Files:** `testbed/scripts/correction_layer.py`, `register_correction_20261010.py`, `score_correction_20261010.py`;
`testbed/results/correction_20261010/{score.json,table_lofo.csv}`; tests in `testbed/tests/test_correction_layer.py` (12).

# 2026-10-10 — scheduler upgrade 3, offline: the two-parameter Pi 4 thermal model does not fit (exploratory; nothing registered)

The survey's upgrade 3 proposed a two-parameter thermal model of the Pi 4, fitted from the 1 Hz polls, to predict a plan's
steady temperature from its duty cycle. `testbed/scripts/thermal_model.py` fits it:
- **The model:** dT/dt = (T_a + k u − T)/τ, with u = 1 when the poll saw the governor at 1.8 GHz.
- **The parameters:** k and τ are shared; T_a is each log's idle level.
- **The method:** a τ grid, with least squares for the rest.
- **The code checks out:** on synthetic polls it recovers τ and k exactly (`testbed/tests/test_thermal_model.py`).

| logs | steps | τ | k | RMSE | per-log idle level T_a |
|---|---:|---:|---:|---:|---|
| all 42 (33 pool cells, 9 run dirs) | 18,690 | 44 s | 28.6 °C | **5.8 °C** | 26–68 °C (implausible) |
| the 7 Qwen3-14B logs | — | 41 s | 32.7 °C | **6.9 °C** | 31–46 °C |
| the 35 TinyLlama logs | — | 23 s | 13.3 °C | 2.4 °C | 41–68 °C |

Two variants, exploratory: a second, slow time constant (RMSE 6.2 / 7.5 °C), and the 14B run's own compute windows as the
load (5.5 °C). Neither fits.

**Why** (the 10-09 Qwen3-14B evaluation's trace, binned by minute):
- **The governor is not a power signal.** For the ~3 minutes each placement loads from the SD card it sits at 60–95 % of
  samples at top clock, while the board stays at 53–58 °C.
- **Real compute heats fast.** At the warm-up prefill the board climbs from 55 to 74 °C within a minute.
- **Sustained 14B load saturates.** The firmware throttles at 80–83 °C (the soft limit), so no linear steady state forms.

**So nothing is registered.** A prospective 30-minute test of a model that fails in sample would say nothing.

**What a model that fits needs:**
- a power-proportional load signal: CPU utilisation from /proc/stat, which a v3 poll could add, its cost measured as v2's
  was;
- the soft limit as a clock cap (the 10-09 run fell to 1,629 MHz);
- time per token under that cap as the quantity to predict, rather than a steady temperature.

# 2026-10-10 — scheduler upgrade 1, step 2: how often Qwen3-0.6B's drafts are what Qwen3-14B would say

**The registration** (`spec_accept_predictions_20261010.json`, 01:38:36, `a73ee502…`; `register_spec_accept_20261010.py`).
- **The prompts:** 50, fixed in the registration: 40 English and 10 Thai, across explanations, how-to, code, arithmetic,
  writing and technical questions.
- **The decoding:** Qwen3's non-thinking chat format, greedy, up to 128 tokens each.
- **The bar (SA-PAY1):** alpha1, the share of checks whose first drafted token is accepted, must be at least 0.60. That is
  where one drafted token per check reaches the survey's 1.3× with a free drafter, by step 1's T(2) / T(1).

**What ran** (`testbed/results/spec_accept_20261010_013837/`, 01:38–02:40, the unit `heteroserve-specaccept-013836`).
- **Where:** the pinned `llama-server` on the laptop, both models from its verified cache. Acceptance belongs to the
  models, the prompts and the decoding, not to the hardware, and the boards stayed free.
- **How:** the drafter proposed 4 tokens per check (`--spec-draft-n-max 4`, `--spec-draft-p-min 0`), and the server logged
  every check with `LLAMA_TRACE=1`.
- **Coverage:** 50 of 50 prompts and 2,010 checks, all assigned to their requests. 1,964 checks drafted the full 4 tokens.
- A smoke run checked the plumbing first (`spec_accept_smoke_20261010_013715/`, one prompt, never scored). The first one
  showed that the per-check lines need `LLAMA_TRACE`.

**The registered verdict.** **SA-PAY1 HOLDS:** alpha1 = **0.728** (bar 0.60). No code changed since registration.

| d (tokens drafted per check) | tokens one check yields, tau(d) | ceiling speedup, free drafter (step 1's T) |
|---:|---:|---:|
| 1 | 1.728 | 1.408 |
| 2 | 2.280 | 1.287 |
| 3 | 2.704 | 1.166 |
| 4 | 3.036 | 1.052 |

tau(d) = 1 + E[min(L, d)], with L the run accepted of a 4-token check: with greedy drafting, a shorter draft accepts the
same prefix.

**What it answers.**
1. **The drafter is good enough.** Seven checks in ten accept its first token, close to the survey's assumed 0.7.
2. **Acceptance is not independent per position.** Each later position is accepted a little more often once the earlier
   ones were (0.728, 0.757, 0.769, 0.784). One run in three of the 4-token checks accepts all four.
3. **One drafted token per check is still the best setting** on the pooled 14B. Checking a third token adds 1,049 ms
   (step 1), more than half a one-token pass, and the third token's yield is too small to pay for it. At d = 1 the drafter can cost up to 196 ms per
   token and still reach 1.3×.
4. **Thai is not worse:** alpha1 is 0.765 on the 10 Thai prompts and 0.719 on the 40 English ones (exploratory).

**Caveats.**
- **Measured on the laptop's CPU numerics.** Step 3 measures acceptance again on the boards.
- **Greedy decoding only.** Sampling at a temperature would accept less.
- 46 of the 50 answers reached the 128-token limit, and 4 ended by themselves.

# 2026-10-10 — scheduler upgrade 1, step 3: speculative decoding served, the draft placed by prediction: 1.40×

**The registration** (`spec_serve_predictions_20261010.json`, 02:40:39, `76631175…`; `register_spec_serve_20261010.py`,
written after step 2's score). The predictor, from measured parts only:
- **The pass times:** step 1's T(k).
- **The draft's yield:** step 2's tau(d).
- **One drafter step on each board:** its Q8_0 per-byte decode rate × the 633.5 MB a Qwen3-0.6B step reads. On the laptop,
  plus a round trip (0.86 ms, pinged from the Pi 4) and the 608 KB of logits back.
- **The draft reading each prompt once:** about 29 tokens at its prefill rate.

The resulting S(d, board) = T(1) / ((T(d + 1) + d · t) / tau(d) + ingest / 63).
- **The engine's choice:** the laptop, d = 1, predicted 1.381×; the Pi 4 at d = 1, 1.292×. Longer drafts were lower on
  both boards (d = 2: 1.256 and 1.152).
- **The Jetson was not a candidate:** it keeps about 440 MiB at this context, and the draft needs about 800.
- **The context:** 1,024 tokens, because at 4,096 the Pi 4 has no room for a draft beside the target. With the draft the Pi 4
  keeps 279 MiB by the accounting; it logged 348 MiB at READY.

**What ran** (`testbed/results/spec_serve_20261010_024045/`, 02:40–04:18, the unit `heteroserve-specserve-024045`).
- **The plan:** the engine's own (Pi 4 27 + Jetson 13, head on the host), served by `serve_pool.sh` with its new default-off
  draft options.
- **The schedule:** base, the Pi 4 draft arm, the laptop draft arm (a `ggml-rpc-server` on the laptop, the target pinned to
  the Jetson with `-dev RPC0`, the draft on `-devd RPC1`), base again.
- **Per session:** 8 of step 2's prompts, 64 tokens each, greedy, one warm-up request first.
- **Conditions:** the strip at 5 Hz and the Pi 4 polled at 1 Hz throughout. Every split matched the plan (Jetson
  2,489.60 MiB), all 32 requests were served, and no code changed since registration.
- **Smoke runs first** (`spec_serve_smoke_20261010_014826/` and `…_015821/`, plumbing only, never scored). The first found
  that `-dev` takes the device name `RPC<n>`, numbered across all `--rpc` servers, not the `RPC0[host:port]` buffer name.

| arm | decode ms per token (Σ server ms / Σ tokens) | speedup | predicted | paired per prompt: mean (min–max) | drafted / accepted | J per decoded token, Pi 4 + Jetson outlets |
|---|---:|---:|---:|---|---|---:|
| base (2 sessions: 1,910.7 and 1,923.1) | 1,916.9 | — | — | — | — | 20.42 |
| **laptop draft, d = 1 (the choice)** | **1,369.1** | **1.400** | 1.381 | 1.407 (1.286–1.543) | 283 / 216 | **15.87 (−22 %)** |
| Pi 4 draft, d = 1 | 1,538.9 | 1.246 | 1.292 | 1.252 (1.141–1.376) | 283 / 216 | 17.50 (−14 %) |

**The registered verdicts** (`spec_serve_20261010.py score` → `score.json`):

| hypothesis | rule | result |
|---|---|---|
| **SS-BAR** | the engine's choice at least 1.30× the base arm | **HOLDS**: 1.400× |
| **SS-ACC** | every draft arm within 10 % of its prediction | **HOLDS**: laptop −1.4 %, Pi 4 +3.7 % (prediction optimistic) |
| **SS-ORDER** | the arm predicted faster measured faster | **HOLDS**: the laptop |
| **SS-MOE** | the survey's MoE leg | **NOT TESTABLE**: Qwen3-30B-A3B (14.7 GB) does not fit Pi 4 + Jetson |

**What it answers.**
1. **Speculative decoding speeds the pooled 14B by 1.40×** (0.52 → 0.73 tokens/s), with greedy output that is the target's
   own. It is the first change in this work that makes a pooled dense model faster, not just placed better.
2. **Where the draft runs is a real placement decision, and the predictor got it right.**
   - The two draft arms accepted exactly the same tokens (283 drafted, 216 accepted in each, as greedy decoding must), so
     their whole difference is the drafter's board.
   - The laptop's 37 ms step beat the Pi 4's 173 ms. The predictor ranked them right and priced both within 4 %.
3. **It also saves energy at the boards:** 22 % less per decoded token at the Pi 4 and Jetson outlets with the laptop draft,
   14 % with the Pi 4's.
   - The boards draw a little more power (the Pi 4 5.8 → 6.6 W), but each token takes 29 % less time.
   - **The laptop's outlet is not a clean figure.** It also feeds the controller and the session writing this up: 24–28.5 W
     during the first base session, about 20 W otherwise. In the laptop draft arm it drew 19.4–21.8 W, the same as the
     closing base session, so drafting on the laptop cost at most about 1 W there.
4. **Acceptance on the boards was in line with the laptop's.**
   - alpha1 = 0.767 on these 8 prompts (288 checks), against 0.728 over step 2's 50 prompts. The predictions used the
     50-prompt figure.
   - Higher acceptance on these prompts lifted both arms about equally, so the ranking does not depend on it.

**Caveats.**
- One model, one plan, 8 prompts × 64 tokens, greedy only, and a 1,024-token context.
- **Thermal:** the Pi 4 draft arm touched the Pi 4's soft limit in 2 of its 8 requests (1 s and 3 s, peak 83.2 °C). Drafting
  adds CPU work on the board that is already the hottest. Its 3.7 % shortfall against prediction includes that.
- **The laptop draft needs the laptop on, on the same switch, running a `ggml-rpc-server`** on its LAN address (unsecured, as
  every RPC server here). That makes it part of the pool.
- **Not the default:** `serve_pool.sh` serves without a draft unless `SERVE_DRAFT` is set. **Changed 10-10 (the user:
  "Laptop draft"):** a person serving Qwen3-14B from a terminal now gets the laptop draft by default; harnesses serve as
  registered (`serve_defaults.sh`; JOURNAL 2026-10-10 12:11).

# 2026-10-10 — the draft's placement under shaping (registered): the choice flips, but at 64 Mbit/s, not 30 — and why the link costs 3×

The user chose the next experiment ("Draft under shaping"): where the Qwen3-0.6B draft belongs as the Pi 4 <-> laptop link
slows. It was registered as `testbed/dataset/spec_shape_predictions_20261010.json` (12:55:33, `fa6a9652…`) before any data,
and run 12:55–15:12 as `testbed/results/spec_shape_20261010_125542/` (all 44 requests). A smoke ran first on a scratch
copy (`spec_shape_smoke_20261010_124238/`, never scored).

**The setup:** step 3's plan and server (Qwen3-14B, Pi 4 27 + Jetson 13, head on the host, ctx 1,024, one drafted token per
check), with 4 of its prompts and 64 tokens each.
- **The shaping:** only the draft's RPC port and a measurement port were shaped, on the Pi 4 (`netshape.sh` with
  NETSHAPE_PEER=192.168.1.108, NETSHAPE_PORT=50053,5202).
- **The links:** each setting was measured receiver-timed on the measurement port, with nothing loaded. Every setting came
  in at its payload rate:
  - unshaped: 933.9;
  - 100 → 95.60;
  - 50 → 47.80;
  - 35 → 33.50;
  - 25 → 23.90;
  - 15 → 14.30;
  - unshaped again: 933.7 Mbit/s.
- **The order:** base, Pi 4 draft, laptop draft through all seven settings in one server load, Pi 4 draft, base.

**Measured** (ms per token, the server's timings; speedup against base, 1,903.1 ms):

| setting | goodput | laptop draft | speedup | registered prediction | Pi 4 draft | faster |
|---|---:|---:|---:|---:|---:|---|
| s0 | 933.9 | 1,358.2 | 1.401× | 1.382× | 1,496.1 (1.272×) | laptop |
| b100 | 95.6 | 1,456.5 | 1.307× | 1.355× | 1.272× | laptop |
| b50 | 47.8 | 1,536.2 | 1.239× | 1.327× | 1.272× | **Pi 4** (predicted laptop) |
| b35 | 33.5 | 1,614.4 | 1.179× | 1.303× | 1.272× | Pi 4 (predicted laptop, by 0.7 %) |
| b25 | 23.9 | 1,717.6 | 1.108× | 1.273× | 1.272× | Pi 4 |
| b15 | 14.3 | 1,977.7 | 0.962× | 1.207× | 1.272× | Pi 4; the laptop draft is slower than no draft |
| s0r | 933.7 | 1,361.5 | 1.398× | 1.382× | 1.272× | laptop (drift 0.25 %) |

**The registered verdicts:**
- **SH-FLIP holds:** at the four settings with a predicted gap of 3 % or more (s0, b100, b15, s0r), the engine's choice was
  measured faster.
- **SH-CROSS fails:** the measured crossover is 63.9 Mbit/s, against 29.8 predicted, outside the factor 1.5 band.
- **SH-ACC fails:** the error grows as the link slows: −1.4, +3.7, +7.1, +10.6, +14.9 and +25.5 %.
- **SH-SLOPE fails:** c = 3.06. The measured slowdown is about 3× the registered link term.
- **So the placement rule is NOT confirmed under shaping as registered.**

**Why the link costs about 3×** (exploratory, post hoc; `explain_spec_shape_20261010.py` → `explain.json`; the socket count
`explore_draft_bytes_20261010.py` → `testbed/results/draft_bytes_20261010_152629/`):
1. **Two draft decodes per check, not one.** After every target decode the server re-decodes the verified batch on the
   draft (`tools/server/server-context.cpp:3741-3747`, its own TODO: "for now, always re-evaluate for simplicity"). The
   draft-simple `process()` leaves that batch's logits null, which outputs the last token's logits
   (`src/llama-batch.cpp:120-128`), so a full 594.5 KiB row comes back unused. `draft()` then decodes the last token and
   brings back a second row.
2. **Each decode re-sends its whole graph.** ggml-rpc skips re-sending only when the graph's uid equals the device's last
   one (`ggml-rpc.cpp:1031-1050`). The two decodes alternate between 2-token and 1-token graphs, so every decode uploads its
   serialized graph: about 1,400 tensors of 296 bytes. The laptop's socket received 414,930 and 414,596 bytes per draft
   decode in two 64-token requests.
3. **So a check moves 2 × (405 KiB + 594.5 KiB) ≈ 2.0 MB,** against the model's 0.6 MB: 3.3×. With that count:
   - each setting's measured extra time per check is within 3–5 % at 47.8, 33.5, 23.9 and 14.3 Mbit/s (+13 % at 95.6,
     where the extra is smallest);
   - the crossover, from the corrected bytes and the arms' measured gap at gigabit (243.5 ms per check), comes to
     **62.9 Mbit/s against 63.9 measured.**
4. **At gigabit all of this costs about 17 ms per check,** which is why step 3 matched. The server also caches each finished
   request's state on the host, the draft's KV included (about 9 MB per request), but outside the timed generation.

**What it means for the placement rule:** below about 60 Mbit/s the draft belongs on the Pi 4, not 30. The link term for an
RPC draft must count both decodes and the graph uploads.

**Three ways to shrink that traffic:**
- sample on the draft's board (`--spec-draft-backend-sampling`; the survey's upgrade 6 is the target-side twin), which
  sends only the token back;
- skip the re-decode's logits (upstream's TODO);
- cache both graph shapes in the RPC backend.
Only the first is a switch at the pin.

**Also measured** (exploratory; `explore_spec_shape_20261010.py` → `explore.json`):
- **Energy follows time** (gross J per decoded token at the Pi 4 + Jetson outlets):
  - base 20.20; Pi 4 draft 17.04;
  - laptop draft by setting: 15.59 / 16.43 / 17.52 / 17.95 / 19.00 / 21.16, then 15.75 unshaped again;
  - so at 14.3 Mbit/s the laptop draft costs more energy per token than no draft.
  - The laptop's own outlet also feeds the controller, so it is not judged.
- **Thermal:** every request clean.
- **The user's sync timer:** 7 aunsync passes overlapped 6 requests (one each at s0, b50, b35 and b15, one base, one Pi 4
  draft), as the registration said would be reported. They left no visible mark: the overlapped c01 at s0 and the clean
  c01 at s0r differ by −0.21 %, within the 0.02–0.56 % spread of the other prompts between the two unshaped settings.

**Tried next: `--spec-draft-backend-sampling`** (the user: "try the backend sampling option then"; exploratory,
`explore_draft_bs_20261010.py` → `testbed/results/draft_bs_20261010_160336/`). **It does not apply to our draft type at the
pin:**
- **The source:** `llama_set_sampler`, which moves a draft's sampling onto its backend, is called only by the EAGLE-3,
  DFlash and MTP implementations (`common/speculative.cpp:499, 1031, 1406`). draft-simple (179–425) never calls it. The
  option also defaults to on (`common/common.h:332`), so every run today already had it.
- **On the boards:** the same 64-token request, flag explicitly on and explicitly off, in two server loads.
  - The draft's socket moved byte-for-byte the same in both: 49,430.1 KB down (625.7 KB per draft decode) and 32,011.2 KB
    up.
  - Both drafted 38 tokens and accepted 24, and gave identical output.
- **To get the saving:** the EAGLE-3 head (a download not yet approved) or a patch to the pinned llama.cpp. Neither was
  done. The user's instruction: if it does not work, move on to upgrade 5.

# 2026-10-10 (evening) — scheduler upgrade 5: the Jetson's power mode as a plan variable — all six registered tests hold; the saving is small

**The question** (the survey's upgrade 5; the user: "if it's not [working] moving on to Upgrade #5"). In the served 14B
plan the Jetson computes about 0.35 s of every 1.9 s token and waits the rest. Every run since 09-30 waits with its clocks
locked. Would a low-power idle (the governors free) or the 5 W mode save energy per token? At what cost in time? And
can a per-mode characterisation predict both within 5 %?

**Step 1, the characterisation** (`power_modes_char_20261010.py` → `testbed/results/pm_char_20261010_165919/`, not
scored).
- **The instrument:** the same 13-layer block on the Jetson (`-d Vulkan0`), hosted by the laptop instead of the Pi 4,
  with the served `llama-server` arguments.
  - One forward pass per request on a cached prompt, then a sleep G: G = 0, 1.1, 1.3, 1.5 s.
  - Each cell opens with a prefill burst, as a served request does, then runs 64 passes.
  - Two blocks per mode, in the order L U 5 5 U L, the second block taking the gaps in reverse.
  - The modes are switched live and verified from sysfs. The Jetson was restored and checked equal to its snapshot.
    0 guard trips.
- **The laptop was a noisy host while this session worked on it.** In blocks 0–2 the forward-pass medians ran up to
  200 ms above the ~610 ms undisturbed pass. Blocks 3–5, run after the session went quiet, are clean.
  - So the model reads each block-cell at its own measured cycle, and the block's time from the lower decile:
    contention only ever lengthens a pass.
  - That estimator replaced the interpolation first written into the registration script. The change came before the
    registration was written and before any served data.
- **Per mode:**

| mode | block time vs locked (lower decile, 4 gaps) | waiting power Pw (fit) | block energy above waiting A | idle window of 45 s, strip / INA3221 |
|---|---|---|---|---|
| locked: MAXN + jetson_clocks | — | 3.21 W | 2.98 J | 3.25 / 2.19 W |
| unlocked: MAXN, governors free | **+9.1 ms** | 3.31 W | 2.59 J | **2.40–2.55 / 1.56 W** |
| 5 W, jetson_clocks at its maxima | **+156 ms** | 2.97 W | 2.03 J | 2.88–2.96 / 1.91–1.95 W |

  - The fit is the Jetson's outlet power over a cycle of length T, P = Pw + A / T, across each mode's 8 block-cells
    (R² 0.991 / 0.961 / 0.993). So its energy per token is Pw × T + A.
  - **The emulation reproduces the served Jetson at full clocks:** Pw × T0 + A = 9.09 J per token, against 9.09 J
    measured at the Jetson's outlet in the shaping run's base arm.
- **Why the low-power idle does not happen:**
  - The GPU never railgates while the RPC server holds its Vulkan context: its runtime status is "active" throughout.
  - nvhost_podgov holds 921.6 MHz (and the EMC 1,600 MHz) for about 13 s after the last work, then drops to 153, then
    76.8 MHz. Both unlocked idle windows show it: 12.8 and 13.6 s.
  - A board that waits 1.5 s per token never gets there: none of the polls inside the unlocked cells found the GPU below
    921.6 MHz.
  - What the governors do save inside the cycle is the CPU's idle states and clocks. That is about 0.7 W at full load
    (G = 0), and 0.1–0.2 W at the served cycle.
- **5 W:** the GPU is capped at 614.4 MHz (cores 2–3 offline, the CPU at 921.6 MHz). The block takes 156 ms longer,
  about 45 % of its ~350 ms against a clock ratio of 1.5. The Jetson draws ~0.85 W less at the served cycle.

**Step 2, registered** (`power_modes_predictions_20261010.json`, 17:58:21, `6f219e44…`, by
`register_power_modes_20261010.py`).
- The prior is the shaping run's base arm: T0 = 1,903.1 ms and E0 = 20.195 J per token at the Pi 4 + Jetson outlets,
  recomputed from its raw data.
- Per mode: T(m) = T0 + dT(m), and E(m) = E0 + [Ej(m) − Ej(locked)] + 2.391 W × dT(m), with Ej = Pw × T + A.
- Hypotheses: PM-T and PM-E, each mode within 5 % (six tests); PM-PICK, the least-energy mode right, judged only if its
  margin is at least 2 % (it was 2.19 %).

**Step 3, served and scored** (`power_modes_20261010.py` → `testbed/results/power_modes_20261010_175827/`, 17:58–19:13).
- **The run:** the served plan, one load, the modes switched live in the order L U 5 5 U L.
  - 8 requests per mode: the shaping run's four prompts × 64 tokens, greedy.
  - Every switch was verified, and the guard never tripped. The text was identical across modes, the Jetson was
    restored equal to its snapshot, and no code changed after the registration.

| mode | ms per token, measured / predicted | gross J per token (Pi 4 + Jetson), measured / predicted | against locked, within the run |
|---|---|---|---|
| locked | 1,938.9 / 1,903.1 (+1.9 %) | 20.567 / 20.195 (+1.8 %) | — |
| unlocked | 1,963.8 / 1,912.2 (+2.7 %) | 20.425 / 20.050 (+1.9 %) | **+1.3 % time, −0.7 % energy** (predicted −0.73 %) |
| 5 W | 2,101.3 / 2,059.3 (+2.0 %) | 20.323 / 19.620 (+3.6 %) | **+8.4 % time, −1.2 % energy** (predicted −2.85 %) |

- **The verdicts:** all six PM-T and PM-E tests hold, and PM-PICK holds (5 W measured least, as predicted). **Upgrade 5
  holds as registered.**
- **What the run says beyond the verdicts:**
  - **The day's offset.** Every mode ran 1.9–2.7 % slower than the prior. The first request (e01, the Pi 4 at 77.9 °C)
    decoded at 1,889 ms/token, as in the prior; the rest at 81–83 °C ran ~1,948.
    - The ARM clock stayed at 1,800 MHz: the soft-limit bit was active 1–4 s in 5 of the 24 requests, with a minimum
      of 1,726 MHz.
    - So the slowdown is temperature, not throttling. Only the deltas within the run are free of it.
  - **The low-power idle:** −0.142 J per token, against −0.146 predicted. Its time cost is +25 ms, against +9 predicted.
    - One request (block 4, e01: 2,022 ms per token, the GPU at 921.6 MHz throughout and the Pi 4 clean) is unexplained.
    - Without it the cost is about +16 ms.
    - The Jetson's CPU ran at 250–410 MHz under schedutil in those requests.
  - **5 W:** −0.244 J per token, less than half the predicted −0.576.
    - The Jetson's outlet saved what was emulated, about 0.8 J (9.248 → 8.445; emulated 9.089 → 8.141).
    - **The Pi 4 cost more while it waited:** 0.56 J over the 162 ms added, which is 3.4 W, not the 2.391 W idle the
      model used.
    - A host whose worker computes is not idle. A 1-W-higher Pi 4 waiting power is within the energy model's validation
      error at today's busy fractions, but it decides a comparison this close.
  - **The survey's estimate does not survive.** It was "about 1.75 J per token for each watt cut from the Jetson's
    waiting power". The watt is never cut during a token: the governors keep the clocks, and 5 W slows the block instead.
- **As a plan variable:**
  - For time, locked stays.
  - For energy, 5 W saves 1.2 % at 8.4 % more time, and the low-power idle 0.7 % at 1.3 %.
  - Neither is worth changing the serving default. Locked stays, as decided 09-30.
  - The real low-power idle (2.40–2.55 W at the strip, against 3.21 W waiting locked) would need an explicit control:
    drop the GPU's clock while waiting, and restore it before the block. That is up to ~1.2 J per token (6 %) if the
    wake-up were free. Not built.

# 2026-10-10 (evening) — scheduler upgrade 6: sampling on the head board — it works over RPC and saves exactly the logits; as registered (b), 5 of 6 tests hold

**The question** (the survey's upgrade 6). When the output head sits on a worker, every token's logits (n_vocab × 4 =
607,744 bytes for Qwen3) cross back to the host, which then samples. llama.cpp's backend sampling (`backend_sampling`,
per request, in the pin) samples on the board that holds the logits. Does it work over RPC, and does it save exactly
the logits' transfer?

**Read in the pinned source before any data:**
- the RPC device reports every op as supported, so the sampling ops go to the Jetson, whose Vulkan backend has
  TOP_K, SOFT_MAX, CUMSUM and ARGMAX;
- with a sampler on the sequence, the raw logits are copied out only when needed;
- the chain must be backend-capable end to end, so the requests name `"samplers": ["top_k"]` with top_k 1 in both arms.

**The setup:** Qwen3-0.6B Q8_0 (the same vocabulary, so the same logits), Pi 4 14 layers + Jetson 14 + the head on the
Jetson. llama-server and ggml-rpc-server ran with serve_pool.sh's command lines and gates.
- The 14B was not used. With its head it fits the Jetson only at 7 layers, which decodes slower than the served plan,
  and its ~2 s token would bury a 5 ms saving.
- Settings: s0 (unshaped) and netem b100 and b50 on the RPC port. Each link was measured first with nothing loaded:
  940.1, 95.6 and 47.8 Mbit/s from the Jetson to the Pi 4.
- The four prompts × 64 tokens in ABBA order per setting, one server load.

**Registered:**
- `head_bs_predictions_20261010.json` (18:10, `3e64c40e…`), then **registration b** (19:20, `0c12e687…`),
  superseding it before the full run.
- The smoke (plumbing only) showed that the server's ms per token spreads n − 1 decode passes over n tokens; the first
  token comes from the prompt's pass. So b prices the saving at 63/64 × V / g. Nothing else changed.
- Hypotheses:
  - HB-BYTES: the Jetson's bytes per token fall by ≥ 0.9 V.
  - HB-SAVE: at each setting, the saving within 1 ms of the prediction.
  - HB-SLOPE: the slope of the saving against 1 / g within 10 % of its prediction.
  - HB-SAME: identical greedy text.

**The run** (`head_bs_20261010.py` → `testbed/results/head_bs_20261010_192112/`, 19:21–19:35; the Jetson restored equal,
0 guard trips):

| setting (Jetson → Pi 4 goodput) | ms per token off / on | saving, measured / predicted (b) | the Jetson's bytes per token off / on |
|---|---|---|---|
| s0 (940.1 Mbit/s) | 138.01 / 132.60 | 5.41 / 5.09 ms | 679,996 / 80,020 |
| b100 (95.6) | 184.75 / 133.85 | 50.90 / 50.06 ms | the same |
| b50 (47.8) | 236.17 / 135.04 | **101.13 / 100.13 ms** | the same |

- **The verdicts (b):**
  - HB-BYTES holds: the drop is 599,976 bytes per token, 0.987 V.
  - HB-SAME holds: 12 of 12 prompt and setting pairs are identical.
  - HB-SLOPE holds: 602,562 bytes against 598,248 (+0.72 %), with an intercept of 0.35 ms.
  - HB-SAVE holds at s0 (+0.32 ms) and b100 (+0.84 ms), and **fails at b50 by 0.008 ms** (+1.008 against a 1 ms
    tolerance).
  - **Upgrade 6 FAILS as registered, on that one test. The option works over RPC** (HB-BYTES and HB-SAME).
- **Registration a, superseded before the run, would have held all six** (`score_registration_a.json`: +0.24, +0.04 and
  −0.58 ms). Its missing 63/64 happened to offset the constant the run found. b stands; a is reported for the record.
- **What it means:**
  - The time saved is the logits' transfer, to within 0.7 % of its size, plus a constant 0.35 ms (the host's own
    sampling, now on the Jetson).
  - With backend sampling, a worker-held head returns only the token, so the head's placement stops depending on the
    link.
  - For the served 14B nothing changes: its head sits on the host for memory reasons.
- **Found on the way (exploratory):** every request sent several logits' worth of extra bytes from the Jetson in both arms
  (about 5 MB). The server log explains it: "idle slots will be saved to prompt cache upon starting a new task"
  (`--cache-ram`, 8,192 MiB by default). Each new request copies the previous slot's KV, including the worker's layers,
  back over RPC to the host.
  - For the served 14B that is about 4.5 MB per request at today's lengths, and it grows with the context.
  - `--cache-ram 0` turns it off. Not changed: a serving decision for later.

# 2026-10-10 (evening) — scheduler upgrade 4, the KV term: a KV byte does not cost a weight byte — 3.7× on the Pi 4's CPU, 0.5× on the Jetson's GPU (registered, FAILS)

**The question** (the survey's upgrade 4, its first term). The predictor prices a decoded token by the weight bytes each
board reads. At context depth d every layer also reads K and V for the d earlier positions, which the model leaves
out: 7.8 % more bytes for Qwen3-14B at 4,096.

**Registered** (`kv_depth_predictions_20261010.json`, 19:25, `47cb47d7…`, by `register_kv_depth_20261010.py`):
- the extra time per token is n_layers × 4,096 bytes × d × the board's weight-byte cost (predict_pool.costs; Q8_0: Pi 4
  0.273 ns, Jetson 0.134 ns per byte);
- the fresh cells: Qwen3-0.6B Q8_0 whole on one board, `llama-bench -p 0 -n 32 -d 0,1024,2048,4096 -r 3`, three
  invocations per board (the Jetson on Vulkan, clocks locked);
- HK-PI4 and HK-JET: each board's mean slope within 15 %. Predicted: 31.3 ms (Pi 4) and 15.4 ms (Jetson) per 1,000
  tokens of depth.

**The run** (`kv_depth_20261010.py` → `testbed/results/kv_depth_20261010_193538/`, 19:35–20:11): both boards at once,
each alone. The model was copied to the Jetson from the laptop's pinned copy, sha256 checked. The Jetson was restored
equal, with 0 guard trips. Flash attention was llama-bench's default (auto), as in serving.

| board | ms per token at d = 0 / 1,024 / 2,048 / 4,096 (mean of 3) | slope per 1,000 tokens, measured (each run) | predicted | error |
|---|---|---|---|---|
| Pi 4 (CPU, -t 4) | 186.2 / 302.1 / 418.8 / 654.6 | **114.4** (116.6, 112.0, 114.6; spread 4.0 %) | 31.3 | **+266 %** |
| Jetson (Vulkan) | 91.9 / 100.6 / 108.8 / 125.0 | **8.06** (8.00, 8.12, 8.06; spread 1.5 %) | 15.4 | **−48 %** |

- **The verdicts:** HK-PI4 fails and HK-JET fails, so **the KV term as registered fails.**
- **What it says** (exploratory below this line):
  - On the Pi 4 a KV byte costs **1.00 ns, 3.7× a weight byte**. Its A72 has no FP16 arithmetic, and attention does
    more work per byte read than a weight matrix-vector product.
  - On the Jetson a KV byte costs **0.07 ns, half a weight byte**: its GPU does F16 natively and attention is cheap.
  - The response is linear in d on both boards. The term's form holds; its price per byte is wrong in both directions.
- **For the served 14B** (Pi 4 27 + Jetson 13): the registered term added 38 ms per 1,000 tokens (+157 ms at 4,096). The
  measured costs give at least **+467 ms per token at 4,096 (+25 %)**, almost all of it the Pi 4's attention.
  - The 0.6B has 2 query heads per KV head; the 14B has 5. If the Pi 4's attention scales with query heads rather than
    with KV bytes, the 14B pays up to 2.5× that.
  - So the next registration must measure the 14B itself at depth. Its Pi 4 prefill (~0.9 s per token) makes that a
    long run.
- **What it changes:**
  - The planner should price attention per board, separately from weights.
  - At long contexts every Pi 4 layer gets dearer (+16.7 ms at 4,096 for a 14B layer, at the KV-byte rate) while a
    Jetson layer hardly moves (+0.6 ms). The best split moves layers to the Jetson as the context grows, as far as its
    memory allows.
  - Not built yet: a per-board KV cost is a new registration.
- **The rest of upgrade 4, not done tonight:**
  - the MoE term cannot be tested: Qwen3-30B-A3B does not fit the Pi 4 + Jetson;
  - the hybrid term needs a per-operation model first;
  - the planner's network model (the logits in its link term) is a change to `control/placement.py` that would move
    every recorded planner output. It needs an opt-in design, so the reproducibility checker keeps matching.

**Then the discriminating test on the served 14B** (registered `kv14_predictions_20261010.json`, 20:16, `dcabcac6…`, by
`register_kv14_20261010.py`; run by `kv14_20261010.py` → `testbed/results/kv14_20261010_201701/`, 20:17–21:32).
- **The two models, fixed before any 14B data**, both priced from the 0.6B's measured KV costs (Pi 4 1.00 ns, Jetson
  0.070 ns per KV byte):
  - A: the cost follows the KV bytes, 114.1 ms per 1,000 tokens;
  - B: on the Pi 4 it follows the query heads, 279.6 ms. The 14B has 5 per KV head against the 0.6B's 2.
  - HK14-A and HK14-B: each within 15 %.
- **The cell:** the served plan through run_pool.sh (every gate; `-lm none`), `llama-bench -t 4 -p 0 -n 16 -d 0,1024 -r 3`,
  three runs, each exit 0.

| run | d = 0 | d = 1,024 | slope per 1,000 tokens |
|---|---|---|---|
| 1 | 1,934.2 ms | 2,273.1 ms | 331.0 |
| 2 | 1,947.9 | 2,259.6 | 304.4 |
| 3 | 1,923.5 | 2,261.5 | 330.1 |
| mean | | | **321.8** (spread 8.3 %) |

- **The verdicts:**
  - HK14-A fails (+182 %).
  - HK14-B fails by a hair (+15.1 % against 15 %).
  - The morning's registered term (38.3 ms) was 8.4× too small.
- **A thermal confound, found in the logs after scoring:**
  - In every run the d = 0 decode ran first, at 56–81 °C with the ARM clock at 1,800 MHz.
  - The d = 1,024 decode ran after ~15 min of depth prefill, at 84–85 °C, with the soft limit active about 70–80 % of
    those seconds and the clock at 1,531–1,775 MHz.
  - So part of the slope is heat. If the Pi 4's share (~1,550 ms of a token) scaled fully with its clock, heat would
    account for up to ~140 ms. The heat-free slope lies roughly between 180 and 320 ms per 1,000 tokens.
  - **Model A is excluded either way. B is consistent with that range, but its registered verdict is FAILS.**
  - A clean test needs both depths decoded hot, e.g. `-d 1024,0`. Not run.
- **What it means as served** (heat included, as a long chat would heat it): **+322 ms per token for every 1,000 tokens of
  context**, about 17 %. Linear to 4,096 tokens, that is about +1.3 s per token.
  - The Pi 4's CPU attention, not its weights, would dominate a long chat.
  - The serving default context is 4,096, so a long conversation on this plan decodes far slower than the planner says.
  - The planner needs a per-board attention term with the query heads in it. It should also weigh the Jetson's share of
    attention against its memory cap.

**And again with both depths decoded hot: model B holds** (registered `kv14b_predictions_20261010.json`, 21:35, `072f8dec…`,
by `register_kv14b_20261010.py`; the same cell, models, tolerance and scorer, with the protocol `-d 1024,0`; run by
`kv14_20261010.py` → `testbed/results/kv14_20261010_213514/`, 21:35–23:03, three runs, each exit 0).

| run | d = 1,024 | d = 0, seconds after | slope per 1,000 tokens |
|---|---|---|---|
| 1 | 2,263.1 ms | 1,987.7 ms | 269.0 |
| 2 | 2,260.7 | 1,995.1 | 259.4 |
| 3 | 2,265.2 | 1,990.4 | 268.3 |
| mean | | | **265.6** (spread 3.6 %) |

- **The verdicts:**
  - **HK14-B holds (−5.0 %): on the Pi 4, attention costs per query head, not per KV byte.**
  - HK14-A fails (+133 %).
- **The heat's own cost:** d = 0 decoded hot here (1,987.7–1,995.1 ms) against cool in the first test (1,923.5–1,947.9),
  about +56 ms. That is what separates the first test's 321.8 from this 265.6.
- **The term the planner should use** (measured, Qwen3 attention shapes, F16 cache, llama.cpp's default flash-attention
  setting):
  - per layer and per token of depth: Pi 4 = 4,096 bytes × 0.499 ns × (query heads per KV head); Jetson = 4,096 bytes ×
    0.070 ns;
  - for the served 14B, 265.6 ms per token for every 1,000 tokens of context;
  - at the 4,096-token serving default, about **+1.09 s per token (+57 %)**, before heat.
  - The original weight-byte term (38.3 ms) was 6.9× too small. The Jetson's share was measured only at the 0.6B's GQA.
- **What it changes:**
  - Long contexts are the Pi 4's weakness, and the Jetson cannot take more layers (memory).
  - So a long-context plan wants a third board for layers, the laptop, or a shorter context.
  - Building the term into the planner is the next registration.

# 2026-10-11 — the long-context plan: attention priced per board, and the laptop takes 39 of the 14B's 40 layers — 485 ms per token, 4.1× the served plan (registered b; all six hold)

**The question** (the user, 2026-10-11): "build the measured context-length cost into the planner (registered), then try a
long-context plan where the laptop holds some layers. The Jetson has no memory to take more".

**What the planner now knows** (`predict_pool.py`; opt-in, so every earlier registration reproduces):
- **The attention term.** At context depth d, each block's decode costs d × attn(board) more, in both roles
  (`predict(depth=, attn=)`, `profiles(depth=, attn=)`), so `plan_optimal` weighs attention when it splits. The laws
  (`attention_cost_20261011b.json`):
  - Pi 4: 1.894 ns per query-head dimension (n_head × head_dim), from kv14b's 265.6 ms per 1,000 tokens less the Jetson's
    share. The 0.6B gives 1.995 as a check;
  - Jetson: 0.0703 ns per KV byte (kv_depth's 0.6B). How it scales with query heads is not measured;
  - laptop: 0.547 ns per query-head dimension (below).
  - For Qwen3-14B (40 query heads of 128; 4,096 KV bytes per layer per token), per block and per 1,000 tokens of context:
    Pi 4 9.70 ms, laptop 2.80 ms, Jetson 0.29 ms.
- **The laptop as an RPC worker** (`laptop`; cluster.json `t580`; `ggml-rpc-server -H 192.168.1.108 -p 50053 -t 4`), passed
  to `plan_host_choice` and `plan_fixed` through `extra_costs` and `extra_pairs`:
  - its host row is characterisation.csv's t580 CPU row. Its worker row is measured on its own, because the x86 build
    repacks Q4_K for AVX2 as a host but not as a worker;
  - its link to the Pi 4: 933.9 Mbit/s, RTT 0.846 ms (measured 10-10, 12:59);
  - its memory: MemAvailable at registration, less 8 GiB kept for the desktop.

**The laptop's worker row, measured twice:**
- **a, on loopback** (`laptop_worker_char_20261010.sh` → `testbed/results/laptop_worker_20261010_232946/`, 23:29–00:05;
  `attention_cost_20261011.json`, `laptop_worker_20261011.json`): llama-bench on the laptop itself, the worker on
  127.0.0.1.
  - TinyLlama Q4_K_M decode: 50.3 ms per token (47.8, 53.1, 50.0).
  - The 0.6B's attention: 36.3, 40.3 and 46.2 ms per 1,000 tokens, so 0.714 ns.
- **The flaw:** llama-bench's own four threads poll while they wait (`"poll": 50`), on the same four cores as the worker's.
  A one-sample smoke of a's plan (plumbing only, never scored) decoded far faster than a priced it. That traced to this row.
- **b, with the Pi 4 as host over the link** (`laptop_worker_remote_20261011.py` →
  `testbed/results/laptop_worker_remote_20261011_002313/`, 00:23–00:35; `attention_cost_20261011b.json`,
  `laptop_worker_20261011b.json`):
  - TinyLlama decode: 36.7 ms per token net of the network (raw 36.6, 39.9, 39.7; spread 8.4 %; 2.0 ms of link time taken
    off). Prefill: 18.8 ms.
  - The 0.6B's attention: 31.7, 29.0 and 33.4 ms per 1,000 tokens (spread 14.1 %), so 0.547 ns.
- So a priced the laptop's decode 37 % too high and its attention 31 % too high. Both b files re-derive from the raw logs.

**Registered:**
- a: `longctx_predictions_20261011.json` (00:13:53, `1b6281c4…`, by `register_longctx_20261011.py`). It was superseded
  before any scored data.
- b: `longctx_predictions_20261011b.json` (00:37:03, `fb404132…`, by `register_longctx_20261011b.py`). It keeps a's planner,
  boards, planning depth, protocol, runs, tolerances and hypotheses; only the laptop's costs change.
- **The planner's pick.** The host is fixed on the Pi 4, which holds the model file and serves today. The boards are the
  Pi 4, the laptop and the Jetson; RTT1; `-lm none`. With b's costs the pick is **Pi 4 1 + laptop 39, the head on the
  laptop, at every depth up to 4,096**.
  - As a worker, a laptop block decodes in 11.5 ms, a Jetson block in 28.0 ms and a Pi 4 block in 55.6 ms. So the Jetson is
    dropped. The Pi 4 keeps one block because every participant holds at least one.
  - With a's costs the pick was the same up to 2,048. At 4,096 it was Pi 4 1 + Jetson 13 + laptop 26.
- **Two cells, three runs each:**
  - **pick:** `-rpc 192.168.1.108:50053 -ngl 40 -lm none`, with `llama-bench -t 4 -p 0 -n 16 -d 4096,0 -r 3`. So d = 0
    decodes seconds after d = 4,096, at the same heat;
  - **jetson_forced**, the plan that keeps the Jetson (a's pick, and the cluster the question assumed):
    `-rpc 192.168.1.114:50052,192.168.1.108:50053 -ngl 40 -ts 12.5/27.5`, at d = 0 only.
- **The hypotheses:**
  - LC-0 and LC-4096: the pick at each depth within 15 %;
  - LC-SLOPE: its rise from d = 0 to 4,096 within 25 %;
  - LC-GAIN: the served plan's measured 1,991.1 ms per token (kv14b, hot) over the pick's d = 0, within 15 % of the
    predicted factor;
  - LC-JET: the forced plan within 15 %;
  - LC-RANK: the pick measures faster than the forced plan.

**The run** (`longctx_20261011.py` → `testbed/results/longctx_20261011_003704/`, 00:37–02:16):
- every run exited 0;
- the model's sha256 was checked on the Pi 4;
- the Pi 4 was never at its soft limit or capped (5,782 polls);
- the Jetson was restored equal, with 0 guard trips.

| run | pick, d = 4,096 | pick, d = 0 | with the Jetson, d = 0 |
|---|---|---|---|
| 1 | 935.6 ms | 488.5 ms | 679.7 ms |
| 2 | 917.5 | 479.4 | 674.9 |
| 3 | 1,045.1 | 488.1 | 698.6 |
| mean (spread) | **966.1** (13.2 %) | **485.4** (1.9 %) | **684.4** (3.5 %) |
| predicted, b | 1,038.0 | 551.1 | 767.8 |
| predicted, a (superseded) | not priced | 730.7 | 891.8 |

| test | measured | predicted | error | verdict |
|---|---|---|---|---|
| LC-0 | 485.4 ms | 551.1 ms | −11.9 % | **holds** |
| LC-4096 | 966.1 ms | 1,038.0 ms | −6.9 % | **holds** |
| LC-SLOPE (the rise) | 480.7 ms | 487.0 ms | −1.3 % | **holds** |
| LC-GAIN | 4.10× | 3.61× | +13.5 % | **holds** |
| LC-JET | 684.4 ms | 767.8 ms | −10.9 % | **holds** |
| LC-RANK | forced 1.41× slower | 1.39× | | **holds** |

- **The verdicts: all six hold.** The score re-derives (`reproduce.py verify --only rundirs,regs`: 89 checks, 0 failing).
- **a's verdicts, as the protocol requires.** a's own plan is the forced cell. It was measured at d = 0 only, and alone
  (`-d 0`, not seconds after a 4,096 decode):
  - LC-0 would fail: 684.4 against 891.8 (−23.3 %);
  - LC-GAIN would fail: 2.91× against 2.23× (+30.3 %);
  - LC-4096 and LC-SLOPE are untested.
- **Caveats:**
  - Run 3's d = 4,096 repetitions (1,008, 1,151 and 990 ms, against 910–941 in runs 1–2) coincide with desktop activity
    on the laptop logged at 01:55–01:57: a file manager and app windows opening. Without run 3, LC-4096 is −10.7 % and the
    rise −9.1 %; the verdicts stand.
  - The laptop's attention law (per query head) is the Pi 4's CPU law applied by analogy. It was fitted on the 0.6B
    (2 query heads per KV head) and is tested on the 14B (5) only through LC-SLOPE.
  - The served plan was not measured at 4,096: its Pi 4 prefill to 4,096 takes about an hour a run. Its measured slope
    (kv14b) puts it at about 3,079 ms.
  - Energy was not measured. Time per token is all this run claims.

**What it says** (exploratory below this line):
- **The attention term works on a plan the planner chose.** The rise with context is within 1.3 % of the term. Without the
  term the planner prices the pick at 551 ms at 4,096, and the measurement is 75 % above that.
- **The laptop should hold the layers.** It is the cluster's fastest board per block, 2.4× the Jetson. With it in the pool
  the Jetson's memory limit stops mattering: the planner drops the Jetson, and keeping it costs 41 % (39 % predicted).
- **Against the served plan** (Pi 4 27 + Jetson 13):
  - at d = 0, 4.1× faster: 485 against 1,991 ms per token, 2.06 against 0.50 tokens/s;
  - at 4,096, about 3.2×: 966 against ~3,079 ms, 1.04 against ~0.32 tokens/s.
- **Beyond 4,096** (the planner, unregistered):
  - the Jetson comes back at about 6,600 tokens (between 6,528 and 6,656), where its cheaper attention outweighs its slower
    weights;
  - at 8,192 the pick is Pi 4 1 + Jetson 12 + laptop 27;
  - the Jetson's share then falls as its KV cache fills its memory: 10 blocks at 16,384, 8 at 32,768;
  - the Jetson's attention at the 14B's grouping (5 query heads per KV head) is not measured.
- **The rule taken to its end:**
  - the host was fixed on the Pi 4, so a laptop-only plan was not in the plan space;
  - priced with the same costs, the laptop alone as host decodes in 482 ms at d = 0 and 940 ms at 4,096, 13 % and 9 %
    below the pick's prices. Not measured;
  - with the laptop in the cluster, the boards add the host role and little else.
- **What it changes:**
  - A long-context deployment should give the laptop the layers, when the laptop is on and quiet. It is also the user's
    desktop, and desktop use disturbs the worker (run 3).
  - The serving default is unchanged (Pi 4 27 + Jetson 13, the laptop draft). Making the laptop hold the model is a
    decision for the user.
  - Open:
    - this plan's energy per token (the strip; the laptop is on outlet 1);
    - the laptop alone, measured;
    - the Jetson's attention at 5 query heads per KV head;
    - a draft beside the laptop plan.

**Reproduce:**
- `python3 testbed/scripts/longctx_20261011.py score testbed/results/longctx_20261011_003704 --reg
  testbed/dataset/longctx_predictions_20261011b.json`;
- the costs: `python3 testbed/scripts/reproduce.py verify --only costs` rebuilds all four cost files from their raw logs;
- the tests: `testbed/tests/test_longctx.py`.

# 2026-10-20 — the replacement Pi 5, as found: slower than the old board, most where the most bytes move (registered; RB-PP, RB-TG and RB-E fail, RB-MEM holds)

**Why.** The Pi 5 came back on 2026-10-19 on a replacement board (serial `b8ed9049ada12a03`; the returned board was
`b266e1877252f91e`) running the old SD card. Every Pi 5 row in the dataset is the old board's. Registered before any row
(`testbed/dataset/pi5b_predictions_20261019.json`, 2026-10-20 00:00:47, `53f12a70…`): the old board's 34 rows re-taken by
the same scripts and protocols, each within 3 % (RB-PP prefill, RB-TG decode); MemAvailable at rest within 1.5 % (RB-MEM);
TinyLlama Q4_K_M energy per token within 10 % (RB-E); and, stated in advance, the bootloader as the first suspect if decode
fails (the new board's 2024-09-23 against the old one's 2026-01-21).

**Run.** `testbed/results/pi5b_session_20261020_000050/` (unit `heteroserve-pi5b`, 00:00–01:47): host rows at pp512/tg128 and
pp64/tg16 (`quant_campaign_20261020_000124`, `_003008`), worker rows at both protocols and the prefill-size sweep
(`worker_role_20261020_004250`, `_010345`, `_011758`), the Q4_K_M energy run (`energy_phase_pi5b_q4km_20261020_012156`).
Every run exit 0, `get_throttled` 0x0 throughout, the cold gate below 50 °C before each model. **Stopped by plan after step 7**:
the user's yes to the bootloader update came after the registration, and the four remaining energy runs (sanity bands only)
would have described a firmware state the update ends; they run after it (the next entry). Step 8's folder
(`energy_phase_pi5b_q4_0_20261020_014741`) holds only its preflight and an ABORTED note.

**Result** (`score.json` / `score.txt` in the session folder):

| test | verdict | |
|---|---|---|
| RB-PP | **FAILS** | 19 rows; mean \|diff\| 3.1 %, max 13.3 % (host Q8_0 pp512 −13.3 %, 9B pp512 −9.2 %, worker Q3_K_M pp1 −10.9 %) |
| RB-TG | **FAILS** | 15 rows; mean \|diff\| 8.7 %, max 15.1 % (worker Q4_0 tg128 −15.1 %, host Q8_0 tg128 −13.8 %); 14 of 15 outside 3 % |
| RB-MEM | holds | 7,641 MiB at rest against 7,619 (+0.3 %) |
| RB-E | **FAILS** | net J/token +10.4 % prefill (0.0975 against 0.0883), +13.8 % decode (0.3852 against 0.3385); decode 15.89 t/s (−9.6 %); mean power 10.51 / 9.00 W against 9.46 / 8.26 |

- **The pattern is memory traffic.** Prefill rows that are compute-bound match within 1–3 % (TinyLlama Q3_K_M, Q4_K_M at both
  protocols, every worker prefill from pp4 up). The rows that move the most bytes per unit of work fall furthest: decode at
  every quantisation (5–15 %), Q8_0's prefill (twice Q4's bytes, −13.3 %), the 9B's pp512 (−9.2 %), and pp1, which is a decode
  in all but name (−10.9 %).
- **The board also draws more power** at the same work (+1.0 W prefilling, +0.7 W decoding), so its energy per token is worse
  than its time alone explains.
- **The kernel command line differed:** as found, it carried `numa_policy=interleave` but not the firmware-added
  `numa=fake=8 iommu_dma_numa_policy=interleave` that the old board ran with (its 10-05 snapshot): the 2024-09-23 bootloader
  does not add the NUMA emulation.
- New rows without an old counterpart (Q6_K at both protocols and roles; Q4_0 and Q8_0 pooled): see `score.txt`.

**What it means:** the old board's rows do not describe this one; every Pi 5 result before 2026-10-06 is the old board's, and
the planner must price the board on the bench from its own rows (the merge rule). The firmware's share is measured next.

# 2026-10-22 — scheduler v2: precision as a plan variable (Qwen3-32B, four nodes): 2.25× faster, −45 % J/token

**Registered** `testbed/dataset/sched2_predictions_20261022.json` (01:26, `f74b0917…`) before anything ran. ctx 2048, 5 requests of
128 + 64 tokens per arm-run, A V A V, energy at all four outlets (the strip at 5 Hz). The user's rule: Q6_K or better servable.

| arm | model / plan | client TPOT P50 | server mean (pred) | TTFT P50 | J/token, all four outlets |
|---|---|---:|---:|---:|---:|
| A, v1 | Q8_0: laptop 35 + Pi 5 14 + Jetson 5 + Pi 4 10 | 3,384.0 ms | 3,384.1 (3,506.7, −3.5 %) | 124.5 s | 117.7 |
| **V, v2** | **Q6_K: laptop 46 + Jetson 2 + Pi 5 16 (no Pi 4)** | **1,506.0 ms** | 1,498.5 (1,596.3, −6.1 %) | **92.2 s** | **64.2** |

| quality (laptop alone, llama-perplexity vs Q8_0 logits, GPL-3, 8 × 512) | PPL ratio | mean KLD | same top token |
|---|---:|---:|---:|
| Q6_K | 1.0062 | 0.0118 | 99.41 % |
| Q4_K_M (reference, below the floor) | 1.0788 | 0.0950 | 97.16 % |

| hypothesis | result |
|---|---|
| S2-TPOT (V ≥ 1.8× A) | **HOLDS: 2.247×** (predicted 2.197×) |
| S2-ACC (each within 10 %) | HOLDS: −3.5 %, −6.1 % |
| S2-TTFT (V below A) | HOLDS: −26 % |
| S2-ENERGY (V below A) | HOLDS: −45 % |
| Q6-TOP (≥ 97 %) / Q6-PPL (≤ 1 %) / Q-ORDER | HOLD |
| **Q6-KLD (≤ 0.01)** | **FAILS: 0.0118** |

Runs: `testbed/results/sched2_20261022_032239/` (session, strip, score) and `sched2_quality_20261022_012629.log`. Conditions: the
laptop quiet (Wi-Fi off, no apps); Claude desktop's sandbox VM (qemu, ~3 % CPU) and aunsync.timer present, as in every laptop run.

# 2026-10-23 — the engine against prima.cpp + Halda on all four nodes (Qwen3-32B Q8_0): prima.cpp 12 % faster, 6 % less energy

**Registered** `testbed/dataset/prima4b_predictions_20261023.json` (03:09, `44d4b756…`, supersedes the 2026-10-21 registration)
before anything ran: ctx 2048, 5 requests of 128 + 64 tokens per arm-run, order B A B A with a 07:35 deadline (the fourth
arm-run, A, could not end by it and was not started), energy at all four outlets (the strip at 5 Hz). Arm A uses the user's
7 GiB laptop reserve for Qwen3-32B runs (8 GiB no longer fit at 25.8 GB free). Both systems run the same file, sha-pinned; the
boards run prima.cpp from range-checked partial copies.

| arm | plan | client TPOT P50 (P95) | predicted | TTFT P50 | J/token, all four outlets |
|---|---|---:|---:|---:|---:|
| A, the engine | laptop 34 + Pi 5 14 + Jetson 5 + Pi 4 11, head on the Pi 4 | 3,564.4 ms (3,599.4) | 3,624.2 (server mean −1.7 %) | 131.1 s | 114.17 |
| **B, prima.cpp + Halda** | **Pi 5 11 + Pi 4 11 + Jetson 3 (CPU) + laptop 39 (head), both runs** | **3,131.0 ms (3,172.7)** | 3,379.2 as it ran (−7.3 %) | 165.0 s | **107.47** |

| hypothesis | result |
|---|---|
| P4-ALL (every node gets layers in every B run) | HOLDS (39/11/11/3 twice; the 02:58 decision run said 38/11/11/4) |
| P4-TPOT (B, predicted faster, is faster) | **HOLDS: B/A 0.878** |
| P4-ENERGY (B, predicted cheaper, is cheaper) | **HOLDS: B/A 0.941** |
| P4-FOLLOW (energy ratio within 0.05 of the TPOT ratio) | FAILS: 0.941 against 0.878 |
| P4-ACC (A within 10 %) | HOLDS: −1.7 % |
| P4-CORRECT (request 0's first 16 tokens equal) | HOLDS (all five requests equal) |

B's two runs: 3,128.9 and 3,134.4 ms (0.18 % apart). Per board, A → B, J/token: laptop 69.14 → 68.66, Pi 5 16.40 → 14.08,
Pi 4 15.04 → 13.11, Jetson 13.60 → 11.61. **Why B won:** the memory rule. Halda keeps 20 % of MemAvailable free (5.2 GB on
the laptop), while the engine keeps a 7 GiB desktop reserve, so Halda gives the laptop 39 layers and the engine 34. Under
Halda's rule the planner picks laptop 38 + Pi 5 14 + Jetson 5 + Pi 4 7, at a predicted 3,153.9 ms decode and 733.2 ms/token
prefill (predicted, not measured; the equal-limits run is next).
Runs: `testbed/results/prima4_20261023_030920/` (session, strip, battery, score). Conditions in the registration: the laptop on
its charger and its battery full throughout, Wi-Fi off, no apps; a second Claude desktop instance idle; aunsync.timer present.

# 2026-10-23 (afternoon) — equal limits against prima.cpp; speculative decoding on v2; Qwen3.6-35B-A3B's ring order

**Equal laptop limits** (`prima4c_predictions_20261023.json`, 11:10, `35ceb8d9…`; A B A). The engine was given Halda's laptop
budget, 20 % of MemAvailable kept free (laptop=5179, the user's OK for this run); the boards kept the engine's reserves.

| arm | plan | client TPOT P50 | predicted | TTFT P50 | J/token, all four outlets |
|---|---|---:|---:|---:|---:|
| **A, the engine** | laptop 38 + Pi 5 14 + Jetson 5 + Pi 4 7, head on the Pi 4 | **3,063.7 ms** | 3,153.9 (server mean −2.9 %) | **102.8 s** | **105.74** |
| B, prima.cpp + Halda | Pi 5 11 + Pi 4 11 + Jetson 3 + laptop 39 | 3,173.5 ms | 3,379.2 as it ran (−6.2 %) | 166.2 s | 109.55 |

All six hypotheses hold (P4-ALL, P4-TPOT B/A 1.036, P4-ENERGY 1.036, P4-FOLLOW, P4-ACC, P4-CORRECT on request 0). With the
morning (`prima4_20261023_030920`): **the memory rule decides the comparison**. At its own 7 GiB laptop reserve the engine is
12 % slower than prima.cpp; at Halda's laptop budget it is 3.5 % faster, with 3.5 % less energy and a 38 % shorter TTFT.

**Speculative decoding on scheduler v2** (`spec4_predictions_20261023.json`, 13:41; V S V S; Qwen3-32B Q6_K on laptop 44 +
Pi 5 18 + Jetson 2, the draft's 800 MiB inside the 7 GiB laptop reserve):

| arm | mean time per token | predicted | TTFT P50 | J/token, all four outlets |
|---|---:|---:|---:|---:|
| V, v2 alone | 1,481.5 ms | 1,597.1 (−7.2 %) | 95.3 s | 60.80 |
| **S, + Qwen3-0.6B draft** (laptop CPU, 1 token per check) | **1,150.5 ms (1.288×)** | 1,152.8 (1.385×) | 98.4 s | **51.44 (−15.4 %)** |

SP-FASTER, SP-PRED (within 15 %), SP-ENERGY, SP-ACC and SP-TEXT hold. The draft acceptance was 69.7 % (129 of 185, from the
laptop's server logs after scoring), so 1.70 tokens per check. A two-token check costs about 1.30 × T(1) on this plan,
against the 1.23 measured on 2026-10-10's Pi 4 + Jetson plan, and that difference is the gap from the prediction.

**Qwen3.6-35B-A3B Q6_K, the ring order** (`order35b_predictions_20261023.json`, 15:48; P F P F; 8 requests each). Every
standard baseline fails here in the engine's accounting; `order_audit_20261023.py` found this the one model where the
predictor prefers an order the engine did not pick.

| arm | plan | client TPOT P50 | predicted | TTFT P50 | J/token |
|---|---|---:|---:|---:|---:|
| P, the engine's pick | laptop 26 + Pi 4 2 + Pi 5 10 + Jetson 3 | 233.5 ms | 197.0 | 10.4 s | 9.20 |
| **F, the predictor's best order** | laptop 26 + Jetson 3 + Pi 5 10 + Pi 4 2 | **224.9 ms (P/F 1.038)** | 188.1 (1.047) | **9.4 s** | **8.67** |

OR-FASTER, OR-ENERGY and OR-TEXT hold; **OR-ACC fails** (+20.2 %, +19.6 %): the predictor is 20 % optimistic on this model
while ranking the orders right. Runs: `testbed/results/{prima4_20261023_111055, spec4_20261023_134113, order35_20261023_154835}/`.

# 2026-10-23 (evening) — the draft length on v2: one drafted token per check is best

`spec4b_predictions_20261023.json` (17:03): the same plan in every arm (Qwen3-32B Q6_K on laptop 47 + Jetson 1 + Pi 5 16),
the Qwen3-0.6B draft on the laptop's CPU. INCOMPLETE as registered: the charger came out 17:06–17:51, so the session started
measuring at 18:44, and the 19:55 deadline left one run each of S1 and S2; S3 never ran.

| arm | mean time per token | predicted | draft acceptance | J/token, all four outlets |
|---|---:|---:|---:|---:|
| **S1, 1 drafted token** | **1,186.9 ms** | 1,214.7 | 62.6 % | **53.27** |
| S2, 2 drafted tokens | 1,355.3 ms (+14.2 %) | 1,142.6 | 46.6 % | 60.59 |

Both arms gave the same text. Two drafted tokens lose on this laptop-heavy plan: acceptance falls and the longer check costs
more than linear. Also adopted on 2026-10-23: `plan_host_choice(order_check=True)`. Measured by order35b (3.8 %) and off by
default; `testbed/tests/test_order_check.py`.

# 2026-10-24 (night) — the draft length on v2, registered again: INCOMPLETE, one arm-run refused for the laptop's memory

`spec4c_predictions_20261024.json` (00:41): spec4b's design, complete (S1 S2 S3 S3 S2 S1, three requests of 128 + 64 per
arm-run), on the plan the engine gave that night: laptop 43 + Pi 5 18 + Jetson 3, the head on the Jetson. The laptop had
1.5 GB less free memory than on 23 October. Run: `testbed/results/spec4c_20261024_004103/`.

**INCOMPLETE as registered.** Arm-run 4 (S3's second) was refused at 02:05:22 by `serve_pool.sh`'s memory gate: 24,521 MiB
free against the 25,674 the plan needs with the 7 GiB reserve. A third Claude desktop app had been opened at ~01:53, and the
plan's margin at rest was 377 MiB. S3 has one run of two.

| arm | arm-runs (mean ms/token) | mean time per token | predicted | draft acceptance | J/token, all four outlets |
|---|---|---:|---:|---:|---:|
| **S1, 1 drafted token** | 1: 1,261.9; 6: 1,235.7 | **1,248.8 ms** | 1,253.8 | 66.1 % (74 / 112 per run) | **55.67** |
| S2, 2 drafted tokens | 2: 1,288.4; 5: 1,346.7 | 1,317.6 ms (+5.5 %) | 1,453.8 | 57.1 % (100 / 175 per run) | 56.06 |
| S3, 3 drafted tokens | 3: 1,556.3 | 1,556.3 ms (+24.6 %) | 1,655.8 | 45.2 % (107 / 237) | 65.86 |

- **As found, not as a verdict:** the order is spec4b's. One drafted token is fastest and cheapest, and each longer draft
  is slower; acceptance falls from 66 % to 45 %.
- **The runs:** S2's two runs differ by 4.5 %, S1's by 2.1 %. Request 0's text was the same in both runs of S1 and of S2,
  and the same across the arms (an observation).
- **Errors against the registered predictions:** −0.4, −9.4 and −6.0 %. The model over-priced the longer checks.
- **A defect in the published score:** `score.json` gives S3's acceptance as 0.5024.
  - `server_log()` takes the first laptop server log started after an arm-run's qr3 log. The refused run 4 had no server,
    so it got run 5's log (S2's), which was pooled into S3.
  - S3's own run gives 0.4515 (107 / 237). No hypothesis uses the figure.
  - The published file stays as its registered scorer wrote it. `score_spec4d_20261024.py` is that scorer with this one
    fix, for the next registration.
- **Next:** spec4d (`register_spec4d_20261024.py`), queued after the four-node chain as `heteroserve-chain1024b`
  (`chain2_20261024.sh`).
  - It plans with 1,024 MiB of laptop headroom beyond the serve-time gate.
  - Its predictions are drawn from all eight measured arm-runs, stated as such: spec4 S, spec4b S1 and S2, and spec4c's
    five completed runs.

# 2026-10-24 (night) — Qwen3.6-35B-A3B Q6_K characterised where it runs: per-board multipliers, every test holds

`c35_predictions_20261024.json` (02:57:25, snapshot `memory_snapshot_20261024_025643.json`). The model (30.95 GB) fits no single
node, so it was served on all four in five splits on one ring order (laptop host, Jetson, Pi 5, Pi 4: order35b's F), head on
the host. Order A B C D E A, six requests of 128 + 64 per arm-run, ctx 4096. Run: `testbed/results/c35_20261024_025725/`
(chain job 1, 02:57–04:14).

| split | plan (layers) | predicted (predictor as it is) | server mean | client P50 / P95 | TTFT P50 | J/token, all four outlets |
|---|---|---:|---:|---:|---:|---:|
| A (the engine's pick; first and last) | laptop 25 + Jetson 3 + Pi 5 10 + Pi 4 3 | 202.3 ms | **246.2 ms** (245.1, 247.2) | 243.1 / 259.8 | 10.8 s | 9.34 |
| B (3 from the laptop to the Pi 4) | laptop 22 + Jetson 3 + Pi 5 10 + Pi 4 6 | 242.5 | 292.2 | 290.4 / 302.6 | 13.4 s | 10.47 |
| C (3 from the Pi 5 to the Pi 4) | laptop 25 + Jetson 3 + Pi 5 7 + Pi 4 6 | 236.5 | 282.7 | 281.5 / 291.2 | 12.6 s | 10.42 |
| D (2 from the Jetson to the Pi 4) | laptop 25 + Jetson 1 + Pi 5 10 + Pi 4 5 | 221.6 | 266.4 | 264.7 / 275.6 | 11.0 s | 9.87 |
| E (held out) | laptop 23 + Jetson 2 + Pi 5 10 + Pi 4 6 | 237.7 | 285.9 | 284.2 / 293.8 | 12.6 s | 10.21 |

**The fit** (`c35_fit_20261024.py`: one multiplier per board on its predicted busy time, from A–D, residuals 0): laptop
**1.172**, Jetson **1.278**, Pi 5 **1.344**, Pi 4 **1.152**. Per block at A, predicted against fitted: laptop 3.64 → 4.27 ms,
Jetson 7.36 → 9.40, Pi 5 4.97 → 6.68, Pi 4 11.03 → 12.70.

**All five tests hold:**
- C35-RAW: the predictor as it is under-predicts every split by 19.5–21.7 %.
- C35-DRIFT: A's two arm-runs agree within 0.8 %.
- C35-HOLDOUT: the fit predicts E at 285.7 against 285.9 ms (−0.04 %).
- C35-PAST: the same multipliers predict order35b's runs of 23 October within 1.0 % (P: 239.3 against 236.9) and 2.2 % (F:
  229.9 against 225.0), on another day's memory and plans.
- C35-PLAUSIBLE: every multiplier lies in [0.8, 2.0].

**What it means:** the 20 % optimism on this MoE model is per-board and stable, not noise. Its expert reads cost every board
15–34 % more per byte than the dense-model rows it was priced from. The Pi 5's blocks are the most under-priced (1.34), so
a plan priced with the multipliers would move weight off it; that re-plan is untested.

**Not yet adopted:** using the multipliers in the planner is a change to `predict_pool.py`, which every registration of
tonight's chain pins; it waits until the chain and spec4d are scored. pm4 already uses the fitted Jetson multiplier for its
predictions, as registered.

# 2026-10-24 (night) — the network study on four nodes (ns4): INCOMPLETE, nothing measured

`ns4_predictions_20261024.json` (04:17:50): TinyLlama Q4_0 on laptop 11 + Pi 4 11, the output head on the host (H) or last on
the Pi 4 (L), the Pi 4's RPC link unshaped, at 20 and 5 Mbit/s, and +10 ms round trip; ten arm-runs. Run:
`testbed/results/ns4_20261024_041750/`.

**Every arm-run failed before its first request**, so every hypothesis is INCOMPLETE (NS4-SHAPED fails for want of a scored
request). The server came up each time. Then `qr3_tpot.py`'s check that the served host is the one it expects stopped the
run: it expected the Pi 5, because its per-model host table lists only the large models and TinyLlama fell back to the Pi 5.
`qr3_tpot.py` now takes the host from the fixed plan when one is given. The study is registered again as ns4b, unchanged,
to run after the chain (`chain2b_20261024.sh`). Predicted, for the record: HU 94.7, LU 108.2, H20 101.1, L20 161.5, H5 120.8,
L5 325.0, HD 104.7, LD 118.2 ms.

# 2026-10-24 (night) — "try sequential" on the four-node cluster's boards (ring4): one lap is fastest; R4-SEQ holds, R4-ACC fails

`ring4_predictions_20261024.json` (04:40:45). This is the advisor's "try sequential: is it faster?" (5 October), answered on
10-09 with the Pi 4 and the Jetson and now on the laptop + Pi 5. TinyLlama Q4_K_M, the laptop holding 4 blocks and the Pi 5
18, cut into k = 1–4 alternating stages, so each token visits the Pi 5 k times while each board holds the same bytes
throughout. Every stage is an RPC server: the laptop's on its loopback, the Pi 5's on its router address. llama-bench runs
on the laptop. Three rounds, the layouts rotated by round; time only. Run: `testbed/results/ring4_20261024_044045/`.

| layout | Pi 5 visits per token | decode, mean of 3 rounds | rounds | predicted (one round trip per visit) | error |
|---|---:|---:|---|---:|---:|
| **k1** | 1 | **51.95 ms** | 51.92, 52.18, 51.76 | 53.12 | −2.2 % |
| k2 | 2 | 54.37 ms | 54.32, 54.30, 54.49 | 54.25 | +0.2 % |
| k3 | 3 | 57.59 ms | 57.68, 57.10, 57.98 | 55.37 | +4.0 % |
| k4 | 4 | 62.81 ms | 63.09, 62.00, 63.33 | 56.49 | +11.2 % |

- **R4-SEQ holds:** k = 1 is fastest in every round, and decode rises by **3.58 ms per extra lap** (least squares).
- **R4-ACC fails:** k4 is 11.2 % over its prediction, against a 10 % bar.
- **Exploratory:** the link terms price a lap at 1.12 ms (two round trips: 2.09 ms). The other ~2.5 ms per lap is a
  per-visit overhead the model leaves out, as on 10-09 (~3.2 ms with the Pi 4 and the Jetson).
- Prefill moves little: 19.9–20.7 ms per token.
- **Thermal:** the Pi 5 peaked at 66.6 °C with no throttle flag (216 one-second polls); its clock sat below 1,750 MHz for
  37 s of them.

**What it answers:** on this cluster, as on the boards alone, the sequential ring (one visit per board per token) is the
fastest way to split a model across devices for one user. Every extra lap is a cost, not a speed-up.

# 2026-10-24 (morning) — the Jetson's power mode on the four-node 35B plan (pm4): locked MAXN is best; a 5 W Jetson costs the cluster energy

`pm4_predictions_20261024.json` (04:50:51), registered from the c35 run (its split A, its fitted Jetson multiplier). Qwen3.6-35B-A3B
Q6_K on laptop 25 + Jetson 3 + Pi 5 10 + Pi 4 3, head on the host; order L U W W U L, six requests of 128 + 64 per arm-run,
ctx 4096. Run: `testbed/results/pm4_20261024_045051/` (unit `heteroserve-after1024`, after the chain; the chain's own pm4 never
registered: a "%" format bug, fixed).

| arm | the Jetson | server mean | predicted | client P50 / P95 | TTFT P50 | J/token, all four outlets | of which the Jetson's outlet |
|---|---|---:|---:|---:|---:|---:|---:|
| **L** | MAXN, clocks locked (every run since 09-30) | **245.0 ms** | 246.2 | 242.7 / 259.4 | 10.8 s | **9.06** | 0.95 |
| U | MAXN, governors free | 254.7 ms (+3.9 %) | 246.6 | 250.9 / 271.0 | 11.4 s | 9.16 (+1.1 %) | 0.90 |
| W | 5 W mode (2 cores at 922 MHz, GPU 614 MHz) | 254.9 ms (+4.0 %) | 253.3 | 252.2 / 270.2 | 11.9 s | 9.65 (+6.6 %) | 0.81 |

**Verdicts:** PM4-T5, PM4-E5, PM4-ACC, PM4-EACC and PM4-SWITCH hold; **PM4-TU fails**. Freeing the governors moved the token by
3.9 %, outside the registered ±2 %.

**What it shows:**
- **A low-power Jetson raises the cluster's energy.** The 5 W mode saves 14 % at the Jetson's own outlet (0.81 against 0.95
  J/token). But every token takes 4 % longer, and the other three nodes draw power throughout, so the cluster's energy per
  token rises 6.6 %. On this ring, energy follows time.
- **Locked clocks, as every run since 30 September used, are fastest and cheapest.** Free governors cost 3.9 % in time: more
  than the 10 October study measured (+1.5 % on the Jetson's own block), and the reason PM4-TU fails.
- Predictions informed by c35's fit were within 0.5–3.3 % (time) and 1.6–3.1 % (energy).
- The power mode stays fixed: MAXN, clocks locked.

# 2026-10-24 (morning) — the network study registered again (ns4b): the head's place measured; the shaping missed the RPC flow

`ns4b_predictions_20261024.json` (06:06; ns4's design unchanged after the qr3 host fix). TinyLlama Q4_0 on laptop 11 + Pi 4 11, head
on the host (H) or last on the Pi 4 (L), the Pi 4's link nominally unshaped, at 20 and 5 Mbit/s, and +10 ms round trip.
Run: `testbed/results/ns4b_20261024_060648/`. Every arm-run served its six requests.

| setting (as registered) | H: head on the host | L: head on the Pi 4 | L − H | predicted H / L |
|---|---:|---:|---:|---:|
| unshaped (two runs each) | 95.44 ms | 109.34 ms | 13.9 | 94.7 / 108.2 |
| 20 Mbit/s | 95.37 | 109.48 | 14.1 | 101.1 / 161.5 |
| 5 Mbit/s | 95.11 | 108.72 | 13.6 | 120.8 / 325.0 |
| +10 ms round trip | 96.01 | 108.76 | 12.8 | 104.7 / 118.2 |

**The shaping never touched the RPC traffic.** `netshape.sh` filters the Pi 4's packets by peer address. It takes only a
router-LAN peer, and the registration named the laptop's 192.168.1.108. But `serve_pool.sh` connects to the Pi 4's wired-overlay
address (`--rpc 192.168.10.12:50052`), so the flow ran 192.168.10.5 ↔ 192.168.10.12 on the same NICs: checked with `ip route get`
on both ends. The qdiscs were installed and verified (NS4-SHAPED holds), but no RPC packet matched their filter. Every "shaped" arm
is an unshaped measurement: 5 Mbit/s would have cost 128 KB of logits ≈ 200 ms a token on L, and L5 is 108.7 ms.

**What does hold, as measured:**
- **The head's place:** on the host it is faster than last on the Pi 4, at every arm (NS4-HOST-U/20/5/D hold, the
  20/5/D arms being unshaped repeats).
- **The gap is the Pi 4's compute plus the logits' transfer at gigabit:** 12.8–14.1 ms, against a predicted 13.5.
- The unshaped arms match their predictions within 0.8 and 1.0 %; drift between first and last arm-runs is 0.14 %.

NS4-ACC, NS4-RTT and NS4-GAP fail because the settings were not applied to the measured flow. Those three verdicts are about the
instrument, not the network. **Next:** a third registration that shapes the overlay flow and checks, from the filter's own packet
counters, that the RPC traffic was shaped.

# 2026-10-24 (morning) — the context-depth term on four nodes (lc4charb, lc4planb): the best placement changes with the context

Scheduler upgrade 4 again: TinyLlama Q4_K_M, byte-identical on every node, so no cost has to transfer between models. Run by
`lc4_20261024.py` (unit `heteroserve-after1024`; the chain's own lc4char stopped at its gate, an ssh warning read as a process).

**Step 1, the cost of depth per node** (`lc4charb_predictions_20261024.json`, 06:50:32; `testbed/results/lc4charb_20261024_065032/`).
llama-bench `-p 0 -n 32 -d 0,2048,4096,8192 -r 3` on the laptop (`-t 4`), the Pi 5 (`-t 4`) and the Jetson (`-ngl 99`, Vulkan,
clocks locked), the three at once (they share nothing).

| node | depth 0 | 2,048 | 4,096 | 8,192 | slope (ms per 1,000 tokens) | predicted by its law | error |
|---|---:|---:|---:|---:|---:|---:|---:|
| laptop | 29.95 | 57.15 | 100.40 | 232.60 | 25.29 | 24.64 | +2.6 % |
| Pi 5 | 55.35 | 79.21 | 123.11 | 251.74 | 24.61 | 26.33 (provisional) | −6.5 % |
| Jetson (GPU) | 86.87 | 112.48 | 104.80 | 121.46 | 3.52 | 1.58 | +122.5 % |

**LC4C-LAWS fails:** the Jetson's KV-byte law under-prices its depth cost 2.2×, while the CPUs' query-head laws hold within
2.6 and 6.5 %. On both CPUs the cost of depth is steep (~25 ms per 1,000 tokens); on the Jetson's GPU it is a seventh of that.
At 8,192 tokens the Jetson decodes TinyLlama about twice as fast as the laptop. The Pi 5 ended at 58.2 °C, no throttle flag.

**Step 2, the plans without and with the depth term** (`lc4planb_predictions_20261024.json`, 07:00:22, registered from step 1's
measured slopes; `testbed/results/lc4planb_20261024_070022/`). A: the planner's pick without the term (the laptop alone). B: its
pick with the term at 8,192 tokens (all 22 layers on the Jetson's GPU over RPC, the head on the laptop). Each a laptop-hosted
llama-bench at `-d 0,8192 -r 3`, order A B B A.

| plan | depth 0 | predicted | depth 8,192 | predicted |
|---|---:|---:|---:|---:|
| A, the laptop alone | **31.34 ms** | 32.37 | 238.22 ms | 239.53 |
| B, the Jetson's GPU (head on the laptop) | 83.37 ms | 83.18 | **131.61 ms** | 120.13 |

**All three tests hold:**
- LC4P-SHORT: A is faster at depth 0, 31.3 against 83.4 ms.
- LC4P-LONG: B is faster at 8,192 tokens of depth, **1.81×** (131.6 against 238.2 ms).
- LC4P-ACC: every bench is within 15 % of its prediction, at most +9.6 % (B at 8,192).

**What it means:** with the depth term, the engine's best placement depends on the context. For a long conversation it moves
the model off the fastest CPU and onto the GPU whose attention cost barely grows. Priced from measured slopes, the move is
predicted within 10 % and measured 1.81× faster.

# 2026-10-24 (morning) — the draft length on v2, complete (spec4d): one drafted token per check is best; all five tests hold

`spec4d_predictions_20261024.json` (07:19:09, `66ba47cd…`): spec4c's design (S1 S2 S3 S3 S2 S1, three requests of 128 + 64
per arm-run), planned with 1,024 MiB of laptop headroom beyond the serve-time gate, on the plan the engine gave: laptop 45 +
Pi 5 18 + Jetson 1, the head on the Jetson. The Qwen3-0.6B Q8_0 draft on the laptop's CPU, 1, 2 or 3 drafted tokens per
check. Its predictions come from all eight earlier measured arm-runs (spec4 S, spec4b S1 S2, spec4c's five), scaled to this
plan. Run: `testbed/results/spec4d_20261024_071909/` (07:19–09:51, unit `heteroserve-after1024`, `chain2b_20261024.sh`).
Every arm-run served its three requests; the laptop's margin against the gate was 1,373 MiB. Code changed since registration: none.

| arm | arm-runs (mean ms/token) | mean time per token | predicted | draft acceptance | J/token, all four outlets |
|---|---|---:|---:|---:|---:|
| **S1, 1 drafted token** | 1: 1,169.6; 6: 1,155.0 | **1,162.3 ms** | 1,206.4 (−3.7 %) | 70.6 % (77 / 109 per run) | **51.65** |
| S2, 2 drafted tokens | 2: 1,259.3; 5: 1,259.3 | 1,259.3 ms (+8.3 %) | 1,313.9 (−4.2 %) | 56.5 % (100 / 177) | 53.90 (+4.4 %) |
| S3, 3 drafted tokens | 3: 1,390.9; 4: 1,391.2 | 1,391.0 ms (+19.7 %) | 1,499.3 (−7.2 %) | 48.5 % (111 / 229) | 59.24 (+14.7 %) |

**All five tests hold:** SC-ORDER (S1 < S2 < S3), SC-BEST (S1, as predicted), SC-ENERGY (the fastest arm is the cheapest),
SC-ACC (each within 10 %: −3.7, −4.2, −7.2 %) and SC-TEXT (request 0's text the same in both runs of each arm; across the
arms too, an observation). TTFT P50 was 97.0–97.5 s in every arm: the draft does not change the prefill.

- **The same order in three registrations:** spec4b, spec4c and spec4d all find one drafted token fastest and cheapest. Each
  extra token lowers the acceptance (70.6 → 56.5 → 48.5 %), and a longer check on this laptop-heavy plan costs more than the
  tokens it saves.
- **Repeatable:** each arm's two runs drafted and accepted the same tokens (greedy, seed 1), and their times differ by 1.3 %
  (S1), 0.002 % (S2) and 0.02 % (S3).
- **Against no draft, not measured in this session:** spec4 measured v2 alone at 1,481.5 ms on 23 October's plan (laptop 44 +
  Pi 5 18 + Jetson 2) and one drafted token at 1,150.5 ms there (1.29×). The registration's estimate for this plan without a
  draft is 1,533.5 ms (23 October's measured time scaled by the predictor, ASSUMED), which would make S1 1.32× faster.
- **The predictions were 4–7 % pessimistic** in every arm, after spec4c's −0.4 to −9.4 %. The ratios to the no-draft time
  (0.787, 0.857, 0.978 by draft length) were drawn from runs on two other plans.
- **Kept:** the engine's default draft length, one token per check (`serve_defaults.sh`, `--spec-draft-n-max 1` since
  2026-10-10), now agrees with three four-node registrations. Qwen3-32B is not in `DRAFT_MODELS`: its draft is passed as
  `SERVE_DRAFT` arguments, as in these runs.

# 2026-10-24 (morning) — the network study, third registration (ns4c): the shaping reaches the RPC flow; the head belongs on the host, and its penalty grows as the link slows

`ns4c_predictions_20261024.json` (09:59:25, `e7ed68cf…`): ns4b's design and predictions, with the shaping on the flow the RPC
actually uses (the Pi 4's eth0 ↔ the laptop's overlay address 192.168.10.5, port 50052) and two extra checks:
`ns4c_extra_20261024.py`'s NS4-SHAPED (from netem's own packet counters) and NS4-GAP. Registered after
`smoke_ns4c_20261024.sh` passed at 09:57 (its first attempt at 09:53 failed on its own instrument: JOURNAL 24 Oct, item 23).
TinyLlama Q4_0 on laptop 11 + Pi 4 11; six requests of 128 + 64 per arm-run after one warm-up of 512 + 32; order HU LU H20 L20
H5 L5 HD LD LU HU. Run: `testbed/results/ns4c_20261024_095925/` (10:03–10:45). Every arm-run served; code changed since
registration: none.

| setting | H: head on the host | predicted | L: head last, on the Pi 4 | predicted | L − H | L / H, time | L / H, J per token |
|---|---:|---:|---:|---:|---:|---:|---:|
| unshaped (two runs each) | **96.67 ms** | 94.69 | 110.22 ms | 108.21 | 13.55 | 1.14 | 1.10 |
| 20 Mbit/s | **108.20** | 101.11 (+7.0 %) | 172.18 | 161.52 (+6.6 %) | 63.98 | 1.59 | 1.42 |
| 5 Mbit/s | **146.31** | 120.77 (**+21.2 %**) | 361.78 | 324.95 (+11.3 %) | 215.47 | **2.47** | **2.05** |
| +5 ms each way (+10 ms round trip) | **107.96** | 104.69 | 120.98 | 118.21 | 13.02 | 1.12 | 1.04 |

J per token at all four outlets: HU 2.97, LU 3.26, H20 3.21, L20 4.57, H5 4.36, L5 8.94, HD 3.32, LD 3.47. TTFT P50: 5.8 s
unshaped, 6.7–6.8 s at 20 Mbit/s, 9.7–9.9 s at 5 Mbit/s.

**Every shaped arm-run was shaped** (NS4-SHAPED holds). Netem passed 6,800–7,600 packets out and 25,400–25,900 in on each H
run; on each L run it passed 40,100–41,100 out (55.9 MB, the logits) and 26,400–26,700 in.

**Six of the seven registered tests hold, and both extra checks:**
- NS4-HOST-U / 20 / 5 / D: the head on the host is faster at every setting.
- NS4-RTT: +10 ms of round trip adds 11.29 ms per token to H (the band 7–13): one round trip per token, RTT1 on four nodes.
  It adds 10.76 to L.
- NS4-DRIFT: HU's first and last runs within 1.46 % (95.96, 97.37).
- NS4-GAP: the head-last penalty grows as the rate falls, 13.6 → 64.0 → 215.5 ms.
- **NS4-ACC FAILS on one arm:** H5 is 21.2 % slower than predicted (tolerance 15 %). The other seven arms are within 1.9–11.3 %.

**Why the shaped arms run slow (post hoc, not registered).** The slopes give the bytes a token moves over the link:
- On H, about 29–31 KB per token, at 20 and at 5 Mbit/s alike. The predictor counts 16 KB: the 8 KB hidden state, once
  each way.
- On L, about 155–157 KB per token, against 133–135 predicted.
- L − H is 126 KB, the 128,000 bytes of logits. The predictor's logits term is right.
- What is missing is a constant of roughly 15–22 KB a token, on top of the tensors. At gigabit that is 0.1–0.2 ms,
  invisible; at 5 Mbit/s it is ~25–35 ms.
- The archived 9 October diagnostic showed that netem charges whole frames (payload rate = rate × 1448 / 1514). That
  explains under 5 % of it.
- The rest is most likely the RPC's own per-compute messages (the graph and its small inputs), not yet attributed.
- The counters agree that more crosses than the tensors alone: 33.6 MB laptop → Pi 4 per arm-run. But that total includes
  the warm-up and the prompts, so the per-token share is not separated here.
- **A candidate predictor term:** a per-compute message size, measured per worker. It would have to be registered before
  use.

**What it means for the planner:** its default since 2026-10-08 weighs the head on the host against the head on the last
board and keeps the cheaper (`head_placement="best"`). Here the host is cheaper at every link speed tested, as priced. The
price of the wrong choice rises from 14 % at gigabit to 2.5× the time and 2× the energy per token at 5 Mbit/s.

# 2026-10-24 (late morning) — sampling on the head's board on four nodes (bs4): the logits stay on the Pi 4; at 5 Mbit/s 2.37× faster and half the energy

Scheduler upgrade 6. `bs4_predictions_20261024.json` (10:48:38, `22e0e418…`), registered after `smoke_bs4_20261024.sh` passed
at 10:46. `serve_pool.sh` now takes an opt-in `SERVE_EXTRA` (`apply_serve_extra_20261024.py`; the testbed suite passed, 189
tests; `serve_pool.sh` is `9f315d70…`). TinyLlama Q4_0 on laptop 11 + Pi 4 11, the head last on the Pi 4, with and without
`--backend-sampling`, the Pi 4's RPC link unshaped and at 5 Mbit/s (ns4c's overlay shaping). Six requests of 128 + 64 per
arm-run after a 512 + 32 warm-up; order LU LBU L5 LB5 LB5 L5 LBU LU. Run: `testbed/results/bs4_20261024_104838/` (10:53–11:29).
Every arm-run served; code changed since registration: none.

| arm | the Pi 4's link | backend sampling | mean ms per token (runs) | predicted | J per token, all four outlets | TTFT P50 |
|---|---|---|---:|---:|---:|---:|
| LU | unshaped | no | 110.51 (111.06, 109.95) | 108.21 (+2.1 %) | 3.38 | 5.79 s |
| LBU | unshaped | **yes** | 110.35 (110.26, 110.44) | 107.12 (+3.0 %) | 3.35 | 5.80 s |
| L5 | 5 Mbit/s | no | 362.36 (362.66, 362.05) | 324.95 (+11.5 %) | 8.91 | 9.88 s |
| **LB5** | 5 Mbit/s | **yes** | **153.01** (152.27, 153.74) | 120.16 (**+27.3 %**) | **4.28** | 9.73 s |

**The logits stayed on the Pi 4.** Netem's counters, on the Pi 4's sending side:

| arm-runs | packets out | megabytes out |
|---|---:|---:|
| L5 (each) | 40,600–40,700 | 55.9 |
| LB5 (each) | about 9,450 | 0.65 |

Into the Pi 4, LB5 carried 35.2 MB against L5's 33.7.

**Six of the seven tests hold, plus BS4-SHAPED:**
- BS4-FASTER-5: at 5 Mbit/s LB5 is 2.37× faster than L5.
- BS4-SAVE-5: the saving is 209.4 ms, against a predicted 204.8 (band 163.8–245.7): the logits' transfer.
- BS4-SAVE-U: unshaped it saves 0.16 ms (band −1 to +3, predicted 1.1).
- BS4-ENERGY-5: 4.28 against 8.91 J per token, −51.9 %.
- BS4-TEXT: one text across every arm-run, with and without the option.
- BS4-DRIFT: LU's runs 1.0 % apart.
- **BS4-ACC FAILS:** LB5 is 27.3 % slower than predicted (tolerance 15 %); L5 +11.5 %, the unshaped arms +2–3 %.

**Why LB5 misses (post hoc, the same cause as ns4c's H5).** LB5 − LBU = 42.7 ms at 5 Mbit/s is ~27 KB a token over the link,
where the predictor counts 8 KB (the hidden state in, the sampled token out). The extra ~19 KB a token is the same per-compute
RPC traffic ns4c's slopes showed (15–22 KB). With the option, LB5 runs only 6.7 ms slower than ns4c's head-on-host H5 (146.3
ms) on the same link. Without it, the head on the Pi 4 is 216 ms slower.

**What it means:** a head on a worker no longer costs the logits' transfer. Its penalty becomes the board's own compute of
the head, about 13.7 ms here. On today's four-node plans the head sits on the laptop, or on the Jetson at gigabit (v2's
Qwen3-32B plan, where Qwen3's 0.61 MB of logits cost about 5 ms of ~1,500), so the default gain is small. On a slow link it
is the difference between 2.5× and 1.05× of the head-on-host time.

**Adopted as an option, not a default:** `SERVE_EXTRA="--backend-sampling"` serves it, and every request stays greedy here
(llama-server turns it off for a request that asks for `n_probs`). The predictor does not yet price a head-last plan without
the logits. That term and the per-compute RPC message term (ns4c) go together, to be registered before use.

# 2026-10-24 (noon) — the full reproducibility check: 236 checks, none failing; the 35B's multipliers in the planner, as an option

**`reproduce.py verify`, no time limit** (11:31–12:19, unit `heteroserve-verify1024`; `testbed/results/verify_full_20261024_113117.log`).
Exit 0, **236 checks, 0 failing**, the record untouched:
- 2 dataset, 14 pool, 27 energy, 2 two-user, 3 for 30 September, 49 run-directory and 4 cost checks;
- 113 registrations, each read-only and quoted by its results;
- 22 session scores, the new family. It re-runs every four-node A/B session from its raw logs: spec4–spec4d, c35, pm4,
  ns4–ns4c, bs4, ring4, lc4charb/lc4planb, prima4 ×2, sched2, the replacement Pi 5's sessions and the correction
  layer's re-test. Each matched with 0 differing values.

The 23 October run with a 1,500 s limit had ended with no output (exit 124). Without the limit the check takes 48 minutes.

**From the release candidate alone** (`/mnt/data/KMITL/Final/HeteroServe_release_20261024/`, not published).
- **First attempt:** 233 checks, 9 failing. Two model files were missing. More importantly, only 13 of the laptop's
  server logs were shipped, and the A/B scorer's "first server log after the arm-run" lookup then picked up later draft
  logs in four sessions.
- **The fixes:** the release ships the whole `~/logs`, and its README names the two model files. `reproduce.py` now
  compares paths under a home folder relative to that home.
- **After them:** 236 checks, 0 failing, run with `HOME` set to the release's `laptop_home` (13:21–14:09).

**The 35B's per-board multipliers, now an option in the planner** (`predict_pool.NODE_MULT`, `apply_node_mult()`,
`plan_host_choice` / `plan_fixed(node_mult="measured")`; `testbed/tests/test_node_mult.py`; the testbed suite passes, 193
tests; `predict_pool.py` is now `ac54e03b…`):
- **What it does:** it multiplies each board's decode cost per byte by c35's fitted factor (laptop 1.1717, Jetson 1.2782,
  Pi 5 1.3436, Pi 4 1.1517), in both roles. Prefill is not scaled.
- **Off by default:** `None` reproduces every plan before. Unscaled, c35's five splits price exactly as registered
  (A 202.28 ms).
- **On:** they price at the measured server means of A–D (246.15, 292.21, 282.70, 266.36; the fit's residuals are 0), and
  the held-out E at 285.75 against 285.87.
- **What it changes today (a what-if, nothing measured):** on the 10:47 memory snapshot, with `order_check=True`, the
  planner's 35B plan is the same with and without the factors, laptop 25 + Pi 5 10 + Jetson 3 + Pi 4 3 with the head on
  the host. Only its predicted time moves, 202.4 → 246.2 ms. The layer counts are bound by memory, so the factors fix the
  prediction, not the choice.
- **Not done:** `serve_pool.sh` plans without them, and no registration has used them yet. Making them the default for this
  model is the advisor's question in the brief.

# 2026-10-24 (afternoon) — the RPC link's bytes per token, measured (rpc4, step 1): ~11 KB a token the predictor does not count

After the user's "let's go" (14:15): the cause behind ns4c's and bs4's two accuracy failures, measured directly, then priced.
`rpc4char_predictions_20261024.json` (14:25, `1480d897…`; `testbed/scripts/rpc4_char_20261024.py`); run
`testbed/results/rpc4char_20261024_142530/` (14:25–14:29).
- **The instrument:** the laptop's own TCP counters for the RPC connection (`ss -tin`, dst 192.168.10.12:50052, one
  socket). It reads `bytes_acked` (payload, laptop → Pi 4), `bytes_received` (payload, back) and the data segments each
  way. The link was unshaped and nothing on the Pi 4 was touched.
- **The plans:** TinyLlama Q4_0 on laptop 11 + Pi 4 11, served three ways. H has the head on the host, L the head last
  on the Pi 4, and LB the head last with `--backend-sampling`.
- **The method:** each plan ran three repetitions of a 16-token and a 144-token greedy completion. A token's bytes are the
  difference divided by 128, so the prompt's traffic cancels.
- **Disclosed in the registration:** a hand probe of plan H at 14:20, the instrument check run before queuing.

| plan | laptop → Pi 4, payload a token | Pi 4 → laptop | the predictor counts | data segments a token (out / in) | on the wire, both ways (+66 B a segment) |
|---|---:|---:|---:|---:|---:|
| H, head on the host | 19,452 B | 8,200 B | 8,192 + 8,192 | 22.3 / 7.0 | 29,583 B |
| L, head on the Pi 4 | 19,452 B | 128,008 B | 8,192 + 128,000 | 18.6 / 90.0 | 154,625 B |
| LB, the same with backend sampling | 21,369 B | **48 B** | 8,192 + 4 | 34.4 / 8.0 | 24,213 B |

**All four tests hold:**
- RC-REPEAT: the payload is identical to the byte in all three repetitions of every plan. Only the segment counts vary,
  by up to 19 % for L's.
- RC-LOGITS: L returns exactly 119,808 B a token more than H, the logits less the hidden state.
- RC-LB: with backend sampling 48 B a token come back.
- RC-EXTRA: the laptop sends 11,260 B a token beyond the 8,192-byte hidden state (H and L), and 13,177 B with backend
  sampling. That is the graph and its small inputs, each decode step: the RPC's own messages.

**What it explains:** priced at a rate, H's 29.6 KB a token cost 47.3 ms at 5 Mbit/s and 11.8 ms at 20. ns4c measured
49.6 and 11.5 over its unshaped arm, where the predictor's 16 KB gave 26.1 and 6.4. The missing term is the RPC's messages,
not the network model. Step 2 tests it blind, at 10 and 3 Mbit/s (`rpc4plan_predictions_20261024.json`, 14:30, `014cd451…`).

# 2026-10-24 (afternoon) — the RPC link term, tested blind (rpc4, step 2): every shaped arm within 5.8 %, against up to 40.8 % before

`rpc4plan_predictions_20261024.json` (14:30, `014cd451…`), registered from step 1's score (`testbed/scripts/register_rpc4plan_20261024.py`).
- **The arms:** the same three plans, on the Pi 4's RPC link shaped to 10 and 3 Mbit/s, two rates never measured before.
  The order was H10 L10 LB10 H3 L3 LB3 H10, with six requests of 128 + 64 per arm-run.
- **New term:** the predictor's unshaped decode, less the tensor transfer it already charges, plus step 1's bytes on the
  wire at the shaped rate. Wire bytes are the payload plus 66 B per data segment, both ways.
- **Old term:** the predictor as ns4c and bs4 registered it.
- **Run:** `testbed/results/rpc4plan_20261024_143036/` (14:30–15:12). Every arm-run served; code changed since
  registration: none.

| arm | measured | new term (error) | old term (error) | J per token, all four outlets |
|---|---:|---:|---:|---:|
| H10, head on the host | 122.97 ms | 118.22 (+4.0 %) | 107.66 (+14.2 %) | 3.48 |
| L10, head on the Pi 4 | 234.11 | 230.74 (+1.5 %) | 216.00 (+8.4 %) | 5.12 |
| LB10, the same with backend sampling | 129.09 | 126.42 (+2.1 %) | 113.60 (+13.6 %) | 3.19 |
| H3 | 180.07 | 173.44 (+3.8 %) | 138.24 (+30.3 %) | 4.16 |
| L3 | 529.30 | 519.38 (+1.9 %) | 470.22 (+12.6 %) | 10.19 |
| LB3 | 181.54 | 171.61 (+5.8 %) | 128.90 (+40.8 %) | 4.24 |

**All four registered tests hold, and both extra checks:**
- RP-ACC: every arm within 10 % (+1.5 to +5.8 %).
- RP-BETTER: every arm closer with the new term.
- RP-HOST-3: H3 < L3.
- RP-LB-3: LB3 < L3.
- RP-DRIFT: H10's two runs 1.8 % apart.
- RP-SHAPED: every arm-run's netem counters ≥ 6,900 packets each way.

**What it means:**
- **The cause is closed:** the predictor's network model was right; what it missed was the RPC's own per-compute messages,
  11–13 KB a token from the laptop. Counted from one unshaped measurement per plan, they predict shaped links within 6 %.
- **The errors left are all positive (+1.5 to +5.8 %),** largest at 3 Mbit/s with the most small segments (LB). That is
  consistent with the ACKs, which the model leaves free, costing a little.
- **Backend sampling makes the head's place nearly free:** at 3 Mbit/s, LB3 (181.5 ms) is within 1 % of H3 (180.1),
  against 529.3 for L3.
- **Next:** the term goes into `predict_pool.py` as an option. It waits until the switch-off session, which pins the
  planner, has ended.

# 2026-10-24 (afternoon) — the switch-off test (so4): cutting the idle Pi 4's outlet saves 3.41 J per token (5.7 %), speed unchanged; every test holds

The user's go-ahead (14:44, "do the switch-off after this"). `so4_predictions_20261024.json` (15:15, `85d82fb4…`).
- **The plan:** scheduler v2's Qwen3-32B Q6_K, planned as spec4d planned it with 1,024 MiB of laptop headroom: laptop 45 +
  Pi 5 18 + Jetson 1, head on the host, no draft. It leaves the Pi 4 out.
- **The arms:** A has the Pi 4 on and idle, as in every run so far. In B it is powered off cleanly and its strip outlet
  switched off before the arm-run (`strip_switch_20261024.py off-node`), then switched back on, with the board up again,
  after it (`on-node`).
- **Order and run:** A B B A, three requests of 128 + 64 per arm-run. Run: `testbed/results/so4_20261024_151550/`
  (15:15–17:02). Every arm-run served; code changed since registration: none.
- **The tool:** `strip_switch_20261024.py` is the only tool that switches an outlet.
  - It may switch only the Pi 4, and refuses outlet 3 (the laptop's, do_not_switch_off).
  - It checks the strip's MAC and the outlet's name, and cuts power only after a clean poweroff and 25 s of silence.
  - It was rehearsed by hand at 15:12 (`testbed/results/so4_rehearsal_20261024_151259/`).

| arm | time per token | predicted | J per token, all four outlets | predicted | the Pi 4's outlet |
|---|---:|---:|---:|---:|---:|
| A, the Pi 4 on and idle (runs 1, 4) | 1,447.8 ms | 1,566.3 (−7.6 %) | 59.99 | 56.19 | 3.52 J (2.4 W) |
| **B, the Pi 4 off at its outlet** (runs 2, 3) | 1,453.1 ms | 1,566.3 (−7.2 %) | **56.58** | 52.44 | **0.00 J** |

**All five registered tests hold, and both extra checks:**
- SO-SAVE: B saves 3.41 J per token against A, predicted 3.74 (the band 2.62–4.87).
- SO-TIME: B / A = 1.004: the Pi 4 is not in the plan, and cutting it changes nothing else.
- SO-ACC: both arms within 7.6 % of the predicted time.
- SO-DRIFT: A's runs 0.5 % apart.
- SO-TEXT: one text in every arm-run.
- SO-ZERO: all 5,615 strip samples in B's requests read the outlet off at 0 W.
- SO-BACK: both power cycles came back. The board was down 32 s after its poweroff, answered ssh in 61 and 53 s with a new
  boot id, and had its clock synchronized and its overlay up in 62 and 59 s.

**What it means:**
- **The cluster's energy per token falls 5.7 % for free,** whenever a plan leaves a board out: the outlet's idle draw is
  all that moves. The energy model priced it within 9 %.
- **A clean shutdown alone is not enough:** halted, the Pi 4 still drew 1.7–1.9 W at the wall, most of its 2.2–2.4 W
  idle. The saving needs the outlet cut.
- **The cost is a boot,** about one minute to come back. That is cheap for a long session and wrong between requests.
- **For the planner:** an idle board's outlet can be cut while it is out of the plan, and powered up when a plan needs it.

# 2026-10-24 (evening) — the RPC messages' shape (rpc5): not per block but per position; two tests fail as registered

`rpc5char_predictions_20261024.json` (17:05, `46c8a709…`; `testbed/scripts/rpc5_char_20261024.py`). The registration discloses
a hand probe of the two-worker ring (~17:05), which had already shown the FIRST worker receiving ~8.2 KB a token more than
the second. It registered a model with a per-block message term fitted from rpc4's H11 and that probe: to a worker =
8,192 + 8,192 for the first + 2,199 + 79 B per block. Run: `testbed/results/rpc5char_20261024_170616/` (17:06–17:12; the Jetson's power
guard started and stopped by the run, as `ab_session` does). TinyLlama Q4_0, the head on the host, unshaped, the laptop's TCP
counters per worker connection.

| split | to the first worker | to the middle worker | to the last worker | back from each |
|---|---:|---:|---:|---:|
| H6: laptop 16 + Pi 4 6 | 19,452 B (Pi 4, first and last) | – | – | 8,200 B |
| H11: laptop 11 + Pi 4 11 | 19,452 B | – | – | 8,200 B |
| H16: laptop 6 + Pi 4 16 | 19,452 B | – | – | 8,200 B |
| R2: laptop 8 + Pi 5 7 + Pi 4 7 | 19,134 B (Pi 5) | – | 10,946 B (Pi 4) | 8,200 B |
| R3: laptop 4 + Pi 5 6 + Jetson 6 + Pi 4 6 | 19,134 B (Pi 5) | 10,628 B (Jetson) | 10,946 B (Pi 4) | 8,200 B |

**Verdicts:**
- R5-REPEAT holds: H11 and R2 match rpc4 and the probe to the byte.
- R5-BACK holds: every worker returns 8,200 B a token.
- **R5-SLOPE FAILS:** 0.0 B per block. The single worker receives exactly 19,452 B a token whether it holds 6, 11 or 16
  blocks.
- **R5-MODEL FAILS:** H6 +2.07 % and H16 −1.99 % against the 2 % registered, because the model assumed a per-block slope.
  R3's three workers came within 0.4–2.2 %.

**The shape, as found:** the messages depend on a worker's place in the ring, not on its blocks. Four figures fit all six
measurements exactly:
- every worker receives the hidden state (8,192 B) plus 2,436 B of messages;
- the last worker 318 B more;
- the first worker 8,506 B more.

Every repetition repeated to the byte.

**Not settled:** the first worker's 8,506 B equals 8,192 + 314. That is TinyLlama's hidden state (2,048 × 4 B), but
equally n_ctx × 4 B at the 2,048-token context every run used. rpc6 tests that at a 4,096-token context and at a deep cache.

# 2026-10-24 (evening) — the first worker's extra is a hidden state, not the context; the messages grow with the cache (rpc6)

`rpc6char_predictions_20261024.json` (17:13, `4e21117a…`; `testbed/scripts/rpc6_char_20261024.py`). Run:
`testbed/results/rpc6char_20261024_171408/` (17:14–17:22). TinyLlama Q4_0, laptop 11 + Pi 4 11, the head on the host, unshaped.

| arm | context | prompt | to the Pi 4 a token | back |
|---|---:|---:|---:|---:|
| C2 | 2,048 | 22 tokens | 19,452 B | 8,200 B |
| C4 | 4,096 | 22 tokens | 19,452 B | 8,200 B |
| D2 | 2,048 | 778 tokens (the cache holds ~780–920 cells while the tokens are measured) | **20,988 B** | 8,200 B |

**Verdicts:**
- R6-REPEAT holds.
- R6-CTX holds: the context size changes nothing. So the first worker's extra 8,506 B is a second hidden state (n_embd × 4)
  plus 314 B, not a context-sized input.
- **R6-DEPTH FAILS:** a deeper cache adds 1,536 B a token.

**Read post hoc:** the 1,536 B is 2 B per cache cell between 256 cells (llama.cpp pads the cache it attends over to 256)
and 1,024 cells. That is an F16 attention mask over the cache, sent each decode step.

**The RPC term, as measured on TinyLlama:**
- every worker receives the hidden state, plus 1,924 B, plus 2 B per padded cache cell;
- the first worker receives one more hidden state plus 314 B;
- the last worker receives 318 B more;
- each worker returns the hidden state plus 8 B (or the logits, or 48 B with backend sampling);
- the header bytes come on top.

Whether a middle worker's mask grows with the cache too is assumed, not measured: D2 had one worker.

# 2026-10-24 (evening) — the RPC term in the planner, as an option: twelve shaped arms priced within +1.4 to +5.8 %

`predict_pool.py` (`3fd7d9ba…`; patcher and backup in `.handoff/assets/nodemult-2026-10-24/`):
- `RPC_MSG` and `rpc_bytes()` carry the shape rpc4–rpc6 measured.
- `predict(..., rpc_msg=RPC_MSG, backend_sampling=...)` prices each worker's link with it: the payload, plus 66 B of
  headers per 1,448 B, plus 10 small segments a token (27.4 with backend sampling).
- None, the default, reproduces every plan before. `testbed/tests/test_rpc_msg.py` adds 4 tests; the suite passes, 197
  tests.
- `rpc_bytes()` reproduces all eight measured byte figures exactly: rpc4 H / L / LB, rpc5's rings, and rpc6's deep cache.

Re-priced with it (post hoc for ns4c and bs4, whose arms motivated the term; rpc4plan was its blind test):

| arm | rate | measured | with the term | the predictor before |
|---|---:|---:|---:|---:|
| ns4c H5 | 5 Mbit/s | 146.31 ms | 141.87 (+3.1 %) | 120.77 (+21.2 %) |
| ns4c L5 | 5 | 361.78 | 354.79 (+2.0 %) | 324.95 (+11.3 %) |
| ns4c H20 / L20 | 20 | 108.20 / 172.18 | 106.38 / 168.98 (+1.7 / +1.9 %) | +7.0 / +6.6 % |
| bs4 LB5 | 5 | 153.01 | 145.77 (+5.0 %) | 120.16 (+27.3 %) |
| rpc4plan, all six | 10, 3 | — | +1.4 to +5.8 % | +8.4 to +40.8 % |

Every residual is positive. The model leaves the ACKs free; they are the likely remainder. Not yet used by
`plan_host_choice` (on gigabit the term is ~0.1–0.3 ms a token per worker, below any plan's margin). It is for pricing slow
or shaped links, which a registration names.

# 2026-10-24 (evening) — scope 4, the prefill/decode hand-off on today's boards: the network crossover, measured (pd4); every test holds

The user's "complete scope 4". The 29 September two-user test (TWOUSER, with the original Pi 5) was repeated on today's boards
(the replacement Pi 5 and the Pi 4). The 8 October modeled crossover (`pd_network_crossover.py`, never measured end to end) was
then tested on a shaped link. `testbed/scripts/pd4_cross_20261024.py` drives `pd_twouser.py`'s arms unchanged and can shape the
hand-off's state transfer per arm (netshape.sh on the Pi 4: peer the Pi 5's overlay 192.168.10.11, port 50091; nothing else is
shaped). `register_pd4_20261024.py` registers both steps, the second from the first's scores; the chain is `chain_pd4_20261024.sh`.
Qwen3.5-9B Q4_K_M, sha256 verified on both Pis before the runs. Two users of 128 tokens each want 32 back, together, greedy;
three repetitions, interleaved; every token timestamped on the laptop, which computes nothing.

**The instrument, checked by hand first:** a 10 MB file through the hand-off's own transfer code took 4.206 s with the port
shaped to 20 Mbit/s, against 4.182 s from the frame-aware model (S x 8 x 1514/1448 / R), and 0.095 s unshaped.

**Step 1, pd4base** (`pd4base_predictions_20261024.json`, 18:34, `4fcf94a4…`; `testbed/results/pd_pd4cross_20261024_183506.log`
and `_score.json`; 18:35–18:52):

| arm | mean completion | user 1 done | user 2: first token / done | makespan | 29 Sep (original Pi 5): mean completion |
|---|---:|---:|---:|---:|---:|
| shared_nr (the Pi 5 alone, no repacking) | 64.68 s | 64.59 s | 45.01 / 64.76 s | 64.76 s | 65.13 s |
| shared (the Pi 5 alone, repacked) | 59.44 s | 59.44 s | 26.94 / 59.44 s | 59.44 s | 61.45 s |
| **handoff** (Pi 5 prefill → Pi 4 decodes user 2) | **53.88 s** | 44.96 s | **13.61** / 62.80 s | 62.80 s | 55.06 s |

All five tests hold:
- PD4-WIN: the hand-off is 16.7 % sooner than shared_nr.
- PD4-WIN-R: it is 9.4 % sooner than shared.
- PD4-STATE: 56,889,668 B exactly, and exactly 1 prompt token processed after the restore, in every hand-off.
- PD4-XFER: the transfer took 0.509 s.
- PD4-STEADY: every arm's runs within 2.1 %.

The 29 September outcome replicates on the replacement board.

**Step 2, pd4cross** (`pd4cross_predictions_20261024.json`, 18:52, `71a015ea…`, registered from step 1's scores).
- **The crossover the model puts against shared_nr:** B* = S x 8 x 1514/1448 / (t0 + 2 (mean_nr − mean_ho)) = **21.52 Mbit/s**
  (against the repacked arm: 40.88).
- **Shaped arms:** at R_hi = 43 and R_lo = 11 Mbit/s, each arm's mean completion predicted as step 1's hand-off plus half the
  extra transfer time.
- **Run:** `testbed/results/pd_pd4cross_20261024_185248.log` and `_score.json` (18:52–19:18).

| arm | mean completion | predicted | user 2 done | the transfer: measured / model | netem packets (min each way) |
|---|---:|---:|---:|---:|---:|
| shared_nr | 64.84 s | — | 64.93 s | — | — |
| handoff, gigabit | 53.92 s | — (step 1: 53.88) | 62.71 s | 0.511 s | — |
| **handoff, 43 Mbit/s** | **59.15 s** | 59.154 (−0.00 %) | 72.96 s | 11.144 / 11.066 s (+0.7 %) | 10,307 |
| **handoff, 11 Mbit/s** | **75.12 s** | 75.251 (−0.18 %) | 105.23 s | 43.507 / 43.26 s (+0.6 %) | 10,276 |

All six tests hold:
- PD4X-ABOVE: at 43 Mbit/s the hand-off still beats the Pi 5 alone, 59.15 against 64.84 s.
- PD4X-BELOW: at 11 Mbit/s it loses, 75.12 s.
- PD4X-ACC: both within 0.2 %.
- PD4X-XFER: both transfers within 0.7 %.
- PD4X-SHAPED: ≥ 10,276 packets each way.
- PD4X-STEADY: ≤ 0.57 %.

The gigabit hand-off repeated step 1's within 0.08 %.

**What it means (scope 4):**
- **For two users at once, handing the second user's decode to the Pi 4 pays.** Both users finish 17 % sooner on average,
  and the second user's first token comes 3.3 times sooner. That holds as long as the link carries the 57 MB state faster
  than ~21.5 Mbit/s (~40.9 Mbit/s against the repacked Pi 5).
- **The 8 October model, priced from single-board parts and one bulk-transfer term, placed the crossover right.** Both
  sides of it measured within 0.2 %.
- **On wired gigabit** (938 Mbit/s here) and on 5 GHz Wi-Fi (167 Mbit/s measured earlier), the network is not what limits
  the hand-off. What limits it is the reply length: 29 September's 128-token replies reversed the answer.

**An observation, not registered: energy.** The strip logged all four outlets through both steps. Over each arm-run's makespan,
divided by the 64 decoded tokens:
- shared_nr 31.2 / 30.4 J per token;
- the hand-off 30.1 / 29.5;
- shared (repacked) 27.9;
- the hand-off at 43 and 11 Mbit/s 33.5 and 45.0.

The repacked Pi 5 alone finishes the pair soonest (makespan 59.4 s), so it uses the least energy. The hand-off finishes the
average user soonest but keeps the slow Pi 4 decoding to the end. Latency and energy disagree here, unlike every single-user
plan, where the fastest was also the cheapest. A registered energy comparison is left for later.

# 2026-10-24 (evening) — the RPC term on the Qwen3 models (rpc7): exact to the byte on the dense 32B; the hybrid MoE's first worker gets 9 KB a token more

`rpc7char_predictions_20261024.json` (19:56, `b329db57…`; `testbed/scripts/rpc7_char_20261024.py`, chain
`chain_rpc7_20261024.sh`). Run: `testbed/results/rpc7char_20261024_195615/` (19:56–20:07). The question: does the term rpc4–rpc6
measured on TinyLlama (`predict_pool.rpc_bytes()`, the `RPC_MSG` option) hold on the models the planner prices? Two arms,
served by serve_pool.sh, ctx 2,048, the head on the host, unshaped. Per worker, the bytes a decoded token, read from `ss` on
the laptop around a 16- and a 48-token greedy completion (the difference over 32), two repetitions:
- **D32:** Qwen3-32B Q6_K (dense; n_embd 5,120), laptop 45 + Pi 5 18 + Jetson 1 (so4's plan; the laptop's reserve 7 GiB).
- **M35:** Qwen3.6-35B-A3B Q6_K (hybrid: gated delta-net layers with attention every fourth block, experts with a gated
  shared expert; n_embd 2,048), laptop 25 + Jetson 3 + Pi 5 10 + Pi 4 3 (c35's split A).

| arm | worker (place) | to the worker a token: measured / predicted | back: measured / predicted |
|---|---|---:|---:|
| D32 | Pi 5 (1 of 2) | 43,710 / 43,710 B | 20,488 / 20,488 B |
| D32 | Jetson (2 of 2) | 23,234 / 23,234 | 20,488 / 20,488 |
| M35 | Jetson (1 of 3) | **28,288** / 19,134 (+47.8 %) | 8,200 / 8,200 |
| M35 | Pi 5 (2 of 3) | 10,958 / 10,628 (+3.1 %) | 8,200 / 8,200 |
| M35 | Pi 4 (3 of 3) | 11,276 / 10,946 (+3.0 %) | 8,200 / 8,200 |

**Verdicts:**
- R7-DENSE holds: both workers to the byte, on a model of another size and family than the one the term was measured on.
- **R7-HYBRID FAILS:** every worker received 330 B a token more than predicted, and the first worker 9,154 B more.
- R7-BACK holds: all five workers to the byte.
- R7-REPEAT holds: the two repetitions identical.

The Pis' throttle flags read `0x0` afterwards; they are sticky since boot (the Pi 5 since 23 Oct 02:20, the Pi 4 since 16:34
today), so they cover rpc5–rpc7 and pd4. **A gap against the house rule, recorded:** rpc4–rpc7 do not log the flags per
run. Byte counts cannot depend on heat, but the rule is per Pi run; rpc8 logs them before and after.

**Read post hoc, from llama.cpp at the pin** (`ggml/src/ggml-backend.cpp`, `src/llama-context.cpp`,
`src/models/qwen35moe.cpp`; found after the data, so untested here):
- **Why the first worker gets more.** While a batch has fewer than 32 tokens, `graph_get_cb` pins each layer's "norm" to that
  layer's device. The scheduler then expands each RPC worker's backend *backwards* from there ("expand gpu up"), through every
  weightless op, until it meets an op on the host's CPU. So the host's last block's trailing weightless ops run on the first
  worker, and every tensor they read crosses the link, each with a ~314 B `set_tensor` header.
- **The dense case.** A dense block ends in one add (FFN output + residual): one hidden state more than a later worker gets,
  which is the "first worker: one more hidden state plus 314 B" that rpc6 measured.
- **The 35B.** Its block ends in the routed experts plus the sigmoid-gated shared expert, then the residual. That is three
  hidden states and the one-value gate: 2 × (8,192 + 314) + (4 + 314) = 17,330 B more than a later worker. Measured:
  28,288 − 10,958 = 17,330 B.
- **The 330 B on every worker.** qwen35-family layers read two inputs that dense ones do not: the recurrent state's copy
  index (one int32 tensor, 4 + 314 B) and IMROPE's four positions a token instead of one (12 B more).

**What it changes:**
- The term's shape is not TinyLlama's alone. It is exact on a 64-block dense model with 2.5 times the hidden size.
- On the hybrid MoE it under-counts the first worker by 9,154 B a token: 0.08 ms at gigabit, 3.8 ms at 20 Mbit/s, 15 ms at
  5 Mbit/s.
- The reading predicts two models it was not fitted on, registered blind as rpc8 (queued behind c30):
  - Qwen3-30B-A3B (experts with no shared expert, so nothing on the CPU between the experts' weighting and the residual): its
    first worker should receive all eight experts' outputs before weighting, 57,690 B a token more than `rpc_bytes()`.
  - Qwen3.5-9B (a dense hybrid): +330 B on every worker.

# 2026-10-24 (evening) — the Jetson's attention law between two candidates (lc4j): the KV-byte law holds on its own model again; the query-head law fails

`lc4j_predictions_20261024.json` (20:07, `14413c76…`; `testbed/scripts/register_lc4j_20261024.py`, run and scored by
`lc4_20261024.py`, its new `slope_near` type). Run: `testbed/results/lc4j_20261024_200720/` (20:08–20:13). Qwen3-0.6B Q8_0 whole
on the Jetson's GPU (Vulkan, clocks locked, guard live), `llama-bench -ngl 99 -p 0 -n 32 -d 0,2048,4096,8192 -r 3`.

**Why:** lc4charb measured TinyLlama's decode on the Jetson growing 3.522 ms per 1,000 tokens of context, 2.2 times the board's
KV-byte law (0.070251 ns per KV byte, fitted on Qwen3-0.6B on 11 October). Qwen3-0.6B has the same query-head width per
layer as TinyLlama (2,048) and four times its KV bytes (4,096 B per layer per token), so the two candidate laws give it very
different slopes.

| depth | 0 | 2,048 | 4,096 | 8,192 | slope |
|---|---:|---:|---:|---:|---:|
| ms per token (3 repetitions; spread ≤ 0.36 %) | 92.08 | 109.44 | 125.77 | 159.27 | **8.18 ms per 1,000 tokens** |

**Verdicts:**
- **J-KV holds:** 8.18 against the KV-byte law's 8.057, +1.5 %.
- **J-QHEAD FAILS:** 8.18 against 4.483 (the query-head law, calibrated on TinyLlama's slope), +82.5 %.

**What it means, and what it does not:**
- The 11 October law replicates, 13 days later, on the model it was fitted on, within 1.5 %. This is a replication, not a
  transfer.
- A law that follows the query heads, as the CPUs' laws do, is ruled out for the Jetson's GPU.
- The TinyLlama slope the losing law was calibrated on is itself in doubt. lc4charb's Jetson series is not monotonic: 86.87,
  112.48, 104.80, 121.46 ms at 0, 2,048, 4,096, 8,192 tokens. Today's Qwen3-0.6B series is smooth.
- Whether the KV-byte law carries to TinyLlama is open. lc4k (registered, queued after rpc8) repeats lc4charb's Jetson bench
  alone. Its tests: every depth within 3 % of lc4charb's figures; the slope near the law's 1.583 or near lc4charb's 3.522.
- An observation, untested: per layer and per 1,000 tokens, TinyLlama costs 0.160 ms with 4 KV heads, and Qwen3-0.6B costs
  0.292 ms with 8. So per KV head the two agree within 9 % (0.040 and 0.037), for head sizes of 64 and 128.

# 2026-10-24 (night) — do the 35B's per-board multipliers carry to Qwen3-30B-A3B (c30)? No: they over-price it by 10–12 %; the predictor as it is misses it by only 4–7 %

`c30_predictions_20261024.json` (20:15:22, `a8e349b6…`, snapshot `memory_snapshot_20261024_201418.json`;
`register_c30_20261024.py`, scored by `ab_score_20261024.py` with `c30_fit_20261024.py`; chain `chain_c30_20261024.sh`). Run:
`testbed/results/c30_20261024_201522/` (20:15–21:08). **Why:** c35's multipliers (laptop 1.172, Jetson 1.278, Pi 5 1.344,
Pi 4 1.152) priced the 35B within 0.04 % on a held-out split. They are an option in the planner for that model only.
c30 asks whether they are a property of the boards reading MoE experts, or of that one model. Qwen3-30B-A3B is the other MoE
on file: the same hidden size, full attention in all 48 blocks, no shared expert, Q6_K like the 35B.

**The design:** c35's, on the 30B.
- One ring order (laptop host, Jetson, Pi 5, Pi 4), the head on the host.
- Split A is the best split on that order with at least 3 blocks on every board. The planner's own pick for this model uses
  two boards: laptop 37 + Pi 5 11, 143.7 ms predicted.
- B, C and D move 3, 3 and 2 blocks from the laptop, the Pi 5 and the Jetson to the Pi 4.
- Order A B C D A; six requests of 128 + 64 per arm-run; ctx 4096; the server's mean gap. Every arm-run served (rc 0), the
  battery full, no session notes.

| split | plan | server mean | the predictor as it is | with the 35B's multipliers | TTFT P50 | J/token, all four outlets |
|---|---|---:|---:|---:|---:|---:|
| A (first and last) | laptop 37 + Jetson 3 + Pi 5 5 + Pi 4 3 | **193.51 ms** (195.75, 191.27) | 181.36 (+6.7 %) | 215.96 (−10.4 %) | 10.5 s | 7.61 |
| B (3 from the laptop to the Pi 4) | laptop 34 + Jetson 3 + Pi 5 5 + Pi 4 6 | 224.54 | 213.90 (+5.0 %) | 253.31 (−11.4 %) | 12.6 s | 8.38 |
| C (3 from the Pi 5 to the Pi 4) | laptop 37 + Jetson 3 + Pi 5 2 + Pi 4 6 | 216.99 | 208.79 (+3.9 %) | 245.26 (−11.5 %) | 12.0 s | 8.04 |
| D (2 from the Jetson to the Pi 4) | laptop 37 + Jetson 1 + Pi 5 5 + Pi 4 5 | 203.01 | 195.73 (+3.7 %) | 231.02 (−12.1 %) | 10.6 s | 7.78 |

(In brackets: the measured mean against that prediction.)

**Verdicts (two registered, two in the extra):**
- **C30-TRANSFER FAILS:** every split runs 10.4–12.1 % faster than the 35B's multipliers predict (5 % allowed).
- **C30-RAW FAILS:** the predictor as it is under-predicts the 30B by only 3.7–6.7 %, not the 10–30 % registered from the 35B
  (19.5–21.7 %).
- **C30-DRIFT FAILS:** A's two arm-runs are 2.32 % apart (195.75 and 191.27 ms), with 2 % allowed.
  - **The likely cause is my own work on the laptop host during run 1.** From 20:21 to 20:27 I ran the switch-off wrapper's
    unit tests and dry runs (short Python processes, each planning with `predict_pool`). The laptop's one-minute load read
    2.1–2.8 in run 1 against 1.4–2.3 in run 5, when I was idle. That is likely, not established.
  - **The rule from now on:** during a session the laptop hosts, only text edits; no Python, not even a short test.
- C30-PLAUSIBLE holds: the 30B's own multipliers, fitted exactly on A–D, are laptop 1.031, Jetson 1.336, Pi 5 1.221,
  Pi 4 0.967.

**How far to trust the 30B's own multipliers:** 1 % on one split's mean moves the laptop's by at most 0.07, the Pi 4's by
0.05, and the Pi 5's and the Jetson's by up to 0.16 (the scorer's `dk_per_1pct_on_split`). With a 2.3 % drift in A, only the
laptop's is well determined: 1.03 on the 30B against 1.17 on the 35B. The same caveat applies to c35's own four.

**What it means:**
- **The 35B's multipliers belong to the 35B.** Applied to another MoE model on the same boards they over-price every split
  by 10–12 %, while the unscaled predictor misses the 30B by 4–7 % (it ranks the four splits right: A < D < C < B).
- So the 35B's 20 % miss is not the boards' cost of reading experts. It comes from what that model has and the 30B lacks.
  Post hoc and untested, the candidates are the gated delta-net layers (three blocks in four) and the shared expert. The
  per-byte rates priced from dense-model rows would not count their compute.
- **For the planner:** `node_mult="measured"` already looks the factors up by model key, so it leaves the 30B unscaled. c30
  says to keep it that way: a model with new block types needs its own characterisation (c35's design, about 1.3 h) before
  its factors exist.

# 2026-10-24 (night) — the first-worker rule, tested blind (rpc8): exact to the byte on two models it was not fitted on; the RPC term now by architecture

`rpc8char_predictions_20261024.json` (21:10, `02ae6685…`; `testbed/scripts/rpc8_char_20261024.py`, chain
`chain_rpc8_20261024.sh`). Run: `testbed/results/rpc8char_20261024_211007/` (21:10–21:14).

**What was tested.** The reading of llama.cpp's scheduler that rpc7's 35B suggested after the fact:
- The host's last block's weightless tail runs on the first worker, so every tensor it reads crosses the link.
- qwen35-family layers read 330 B more on every worker.

It predicts two models it was not fitted on. The arms, on the laptop + Pi 5 (first worker) + Pi 4 (last), head on the host,
ctx 2,048, unshaped; each model's sha256 checked on the laptop first; the Pis' throttle flags logged before and after
(`0x0` both times):
- **M30:** Qwen3-30B-A3B Q3_K_M (qwen3moe: experts without a shared expert; NEOX RoPE), laptop 36 + Pi 5 6 + Pi 4 6. Its
  block ends in `build_moe_ffn`'s weighting and sum, then the residual, with no CPU op between them. So the first worker
  should receive the 8 experts' outputs before weighting (2,048 × 8 × 4 B), the router weights (32 B) and the residual.
- **H9:** Qwen3.5-9B Q4_K_M (qwen35: a dense hybrid with IMROPE), laptop 25 + Pi 5 4 + Pi 4 4. Its block ends in one add,
  as a dense one. So it should get the dense first-worker extra, and every worker +330 B.

| arm | worker (place) | to the worker a token: measured | the reading (registered) | the term as it was |
|---|---|---:|---:|---:|
| M30 | Pi 5 (1 of 2) | **76,824 B** | 76,824 | 19,134 (−75 %) |
| M30 | Pi 4 (2 of 2) | 10,946 | 10,946 | 10,946 |
| H9 | Pi 5 (1 of 2) | 35,848 | 35,848 | 35,518 (−0.9 %) |
| H9 | Pi 4 (2 of 2) | 19,468 | 19,468 | 19,138 (−1.7 %) |

**All five tests hold:**
- R8-MOE-FIRST and R8-MOE-LAST: both to the byte.
- R8-HYB: both workers to the byte.
- R8-BACK: 8,200 and 16,392 B, to the byte.
- R8-REPEAT: the two repetitions identical.

**What it means:**
- **The RPC term is now exact on five models of five architectures** (llama, qwen3, qwen35, qwen3moe, qwen35moe) **and four
  quantisations:** TinyLlama, Qwen3-32B, Qwen3.5-9B, Qwen3-30B-A3B and
  Qwen3.6-35B-A3B. Two of those rules were predicted blind, and one (the 35B's) was fitted on the data it explains.
- **On an MoE model without a shared expert, the first hop carries 7× a later hop's bytes:** 76.8 KB a token. That is
  0.7 ms at gigabit, but 32 ms at 20 Mbit/s and 128 ms at 5 Mbit/s. The evening's term missed 57.7 KB of it.
- **An observation, not tested:** llama.cpp pins a layer's last op to its own device only where a model names it `l_last`
  (deepseek4 and qwen4exp at the pin; `llama_context::graph_get_cb`, with a FIXME to fix it in the scheduler). qwen3moe does
  not, so the eight experts' outputs cross instead of one hidden state.

**In the planner** (`predict_pool.py`, now `312c5985…`; backup in `.handoff/assets/nodemult-2026-10-24/`):
- `RPC_MSG` prices the first worker by architecture (`ARCH_TAIL`: dense, `experts` for qwen3moe, `shexp_gated` for
  qwen35moe) and adds the hybrid inputs (`HYBRID_ARCH`).
- `RPC_MSG_DENSE` keeps the evening's term, under which every model is dense. The two agree for every dense model,
  TinyLlama included, so every arm priced before is unchanged.
- `testbed/tests/test_rpc_msg.py` reproduces all nine rpc7 and rpc8 figures and the dense term. The testbed suite passes,
  211 tests, and so do ThinkPad 269, Pi 5 357 and Pi 4 158.
- The term is still an option (`rpc_msg=`), off by default: on gigabit it is below any plan's margin.

# 2026-10-24 (night) — lc4charb's Jetson bench repeated alone (lc4k): its 2,048-token point was the outlier; TinyLlama's slope is 4.22, and the KV-byte law under-prices it 2.7×

`lc4k_predictions_20261024.json` (21:15, `a63cbd2d…`; `register_lc4k_20261024.py`, run and scored by `lc4_20261024.py`).
Run: `testbed/results/lc4k_20261024_211512/` (21:16–21:23). TinyLlama Q4_K_M whole on the Jetson's GPU, alone, the same
bench as lc4charb (`-ngl 99 -p 0 -n 32 -d 0,2048,4096,8192 -r 3`, Vulkan, clocks locked, guard live).

| depth | 0 | 2,048 | 4,096 | 8,192 | slope |
|---|---:|---:|---:|---:|---:|
| lc4k, the Jetson alone (spread ≤ 0.4 %) | 86.87 | **96.47** | 104.83 | 121.70 | **4.22 ms per 1,000 tokens** |
| lc4charb (three nodes' benches at once) | 86.87 | 112.48 | 104.80 | 121.46 | 3.52 |

**All three registered tests fail:**
- K-REPEAT: three depths repeat within 0.2 %. The 2,048 point is 14.2 % lower, so lc4charb's 2,048 point was disturbed;
  what disturbed it is not known.
- K-KV: the slope is 2.67× the KV-byte law's 1.583 (+166.6 %).
- K-SAME: the slope is 19.8 % above lc4charb's 3.52, because the outlier had flattened that fit.

**What it means, with lc4j:**
- **The Jetson's KV-byte law is model-specific.** It holds on Qwen3-0.6B, its own model (lc4j, +1.5 %), and under-prices
  TinyLlama 2.7×.
- **LC4C-LAWS's failure (2.2×) stands, larger:** 2.7×.
- **For the planner,** the Jetson's depth cost needs a per-model figure, like the 35B's multipliers.
- **lc4planb's plan B at 8,192 tokens** measured 131.61 ms against 120.13 priced with the 3.52 slope (+9.6 %). With 4.22
  it would price about 5.7 ms higher, closer. That is a note, not a re-score.
- Two models cannot tell which mix of query width and KV bytes drives the Jetson's attention cost. A third model with
  another head layout would have to.

# 2026-10-25 (night) — the switch-off automation run live (so5): every test holds

The user's go-ahead (00:55, "Do the switch-off automation"). `so5_predictions_20261025.json` (01:01, `6c2458a7…`, snapshot
`memory_snapshot_20261025_010122.json`; `testbed/scripts/so5_trial_20261025.py`, chain `chain_so5_20261025.sh`). Run:
`testbed/results/so5_20261025_010213/` (01:02–01:06). The first live use of `serve_switch_20261024.py`, as a user would run
it, both of its paths, with the strip logging all four outlets at 5 Hz:
- **A:** the planner's own pick for Qwen3-30B-A3B Q6_K on the laptop, Pi 5 and Pi 4 (laptop 37 + Pi 5 11), which leaves
  the Pi 4 out. The wrapper served it, then shut the Pi 4 down and cut its outlet. Three greedy 32-token completions ran
  with the Pi 4 off. Then `stop`, the Pi 4 left off.
- **B:** TinyLlama Q4_0 on laptop 11 + Pi 4 11, which needs the Pi 4 while it is off. The wrapper powered it up first,
  then served one completion, then `stop`. The cluster ended as it began.

| step | measured |
|---|---|
| A served | laptop 37 + Pi 5 11, as registered; 32, 32, 32 tokens at 144.1, 143.9, 143.6 ms per token |
| the Pi 4 switched off | clean poweroff, 25 s of silence, outlet cut **38.3 s** after the command; it drew 1.9 W halted, just before the cut |
| during A's completions | all 88 strip samples of outlet 0: off, **0.0 W**; the Jetson 1.9 W, Pi 5 5.4 W, laptop 32.9 W on average |
| the Pi 4 powered up for B | outlet on; ssh after **58.1 s**, clock synchronized and overlay up at **58.9 s**, a new boot id (so4 by hand: 53–62 s) |
| B served | laptop 11 + Pi 4 11; 32 tokens |
| the other outlets | on in all 3,453 samples of the trial (1,151 polls × 3 outlets) |

**All five tests hold:**
- SO5-PLAN: both served plans are the registered ones.
- SO5-OFF: the switch log reads "outlet off", and outlet 0 read off at ≤ 0.1 W in every sample of A's completions.
- SO5-SERVE: all four completions returned their 32 tokens.
- SO5-UP: the switch log reads "back".
- SO5-ONLY: nothing else was switched.

No code changed since registration.

**What it means:** the planner's energy lever from so4 (5.7 % less energy per token) now applies itself.
- Serve through `serve_switch_20261024.py` instead of `serve_pool.sh`, and a board the plan leaves out is shut down and
  unplugged after the model is up.
- When a later plan needs the board, it is powered up first, about a minute.
- It switches only the Pi 4. It stays an opt-in: `serve_pool.sh` is unchanged, and a plan change that needs the Pi 4 costs
  that minute of boot.

# 2026-10-25 (night) — a middle worker's attention mask grows with the cache too (rpc9): as the RPC term charges; the repeat test fails on one warm-up repetition

The user's "complete everything lefts". `rpc9char_predictions_20261025.json` (03:13, `be8eab45…`;
`testbed/scripts/rpc9_char_20261025.py`, chain `chain_rpc9_20261025.sh`). Run: `testbed/results/rpc9char_20261025_031402/`
(03:14–03:18). rpc6 measured the mask's growth (2 B per padded cache cell) on a one-worker ring; `RPC_MSG` charges it to
every worker, and whether a middle worker gets it was assumed, not measured.

TinyLlama Q4_0 on rpc5's three-worker ring, laptop 4 + Pi 5 6 + Jetson 6 + Pi 4 6 (the Jetson in the middle), the head on
the host, ctx 2,048. One served model, two prompts: S3 (22 tokens; the cache ~38–166 cells, 256 padded) and D3 (778 tokens;
~794–922 cells, 1,024 padded). Three repetitions of 16- and 144-token completions; a token's bytes = difference / 128. The
Pis read `throttled=0x0` before and after.

| worker (place) | S3: measured / predicted | D3: measured (mean of 3) / predicted | D3 repetitions 1, 2, 3 |
|---|---:|---:|---|
| Pi 5 (1 of 3) | 19,134 / 19,134 | 20,635.95 / 20,670 (−0.16 %) | 20,567.8, 20,670, 20,670 |
| **Jetson (2 of 3)** | 10,628 / 10,628 | **12,129.95 / 12,164 (−0.28 %)** | 12,061.8, 12,164, 12,164 |
| Pi 4 (3 of 3) | 10,946 / 10,946 | 12,430.9 / 12,482 (−0.41 %) | 12,328.8, 12,482, 12,482 |

**Verdicts:**
- R9-MID holds: the middle worker's mask grows as the term charges (+1,536 B a token at 1,024 padded cells).
- R9-ALL holds: all six within 0.5 %.
- R9-BACK holds: 8,200 B.
- **R9-REPEAT FAILS:** a 1.24 % spread against the 1 % registered (rpc4–rpc8 repeated exactly). D3's first repetition came in
  102 B a token low on the Pi 5 and the Jetson and 153 B on the Pi 4. Repetitions 2 and 3, and all of S3, are exact to the byte.
- **Read post hoc:** the run warmed up on the short prompt only. The first long prompt then made the workers' graph grow, and
  its one-off messages landed in the 16-token baseline. A deep warm-up would have kept them out. Recorded as found.

**What it means:** the RPC term's last assumption is now measured. Every worker that holds an attention layer receives the
mask, and it grows with the cache at 2 B per padded cell.

# 2026-10-25 (night) — the RPC term and backend sampling as planner options: they change no plan materially on today's cluster

The user's "complete everything lefts" (follow-up 3). The predictor already priced a head-last plan whose board samples
(`predict(..., rpc_msg=RPC_MSG, backend_sampling=True)`), blind-tested by rpc4plan's LB arms (within 2.1 and 5.8 %). What was
missing was the planner's search using it.

`predict_pool.py` (now `58df0c3a…`; backup in `.handoff/assets/nodemult-2026-10-24/`):
- `plan_host_choice`, `plan_fixed` and `best_by_predictor` take `rpc_msg` and `backend_sampling` and price every plan as
  `predict()` does with them.
- The head's position follows them only with `order_check`, since `plan_optimal`'s own objective does not know them.
- None and False, the defaults, reproduce every plan.
- `testbed/tests/test_rpc_msg.py`: three new tests (off by default; `plan_fixed` prices as `predict()`; on bs4's 5 Mbit/s
  link the planner's pick is the fastest, and a two-board pick keeps the head on the host, as bs4 measured). The suites pass:
  testbed 214, ThinkPad 269, Pi 5 357, Pi 4 158.

**What-if, on the 25 October 01:01 snapshot** (all 14 models of the four-node evaluation, `order_check`, without and with
both options):
- 13 plans are identical.
- One near-tie ring order flips: Qwen3-32B Q8_0, where the Pi 4 and the Pi 5 swap places (3,525.3 against 3,525.6 ms).
- The predicted times move by at most 4.9 ms. That is v2's Qwen3-32B Q6_K plan, whose head sits on the last board (the
  Pi 5): sampling there keeps 608 KB of logits a token off the link (1,596.3 → 1,591.4 ms, 0.3 %).

A registered test of a changed decision is not possible on this cluster, because the fastest board hosts.

# 2026-10-25 (night) — the Jetson's attention law against a third model (lc4m): the query-width law holds on Qwen3-4B; read with care

`lc4m_predictions_20261025.json` (03:31, `771ed254…`; `register_lc4m_20261025.py`, run and scored by `lc4_20261024.py`).
Run: `testbed/results/lc4m_20261025_033221/` (03:32–03:39).
- **The model:** Qwen3-4B Q4_K_M, downloaded for this with the user's yes (official Qwen GGUF @ `bc640142`, SHA-256 checked
  on the laptop and on the Jetson; `testbed/models.md`).
- **Its layout:** 36 layers, 32 query heads × 128, 8 KV heads × 128. That is Qwen3-0.6B's KV bytes per layer (4,096) with
  twice its query dims (4,096 against 2,048).
- **The bench:** whole on the Jetson's GPU (Vulkan, clocks locked, guard live),
  `-ngl 99 -ub 128 -p 0 -n 32 -d 0,256,512,768,1024 -r 3`.

**The range was set by hand smokes before the registration** (single runs, disclosed in it):
- the OOM killer stopped llama-bench at 3,072 tokens of depth;
- with `-ub 128` the decode collapsed 30× at 2,048 (memory pressure, not attention) and again at 1,536;
- 0–1,024 was clean. Those single runs gave roughly 22.5 ms per 1,000 tokens. The three predictions come from lc4j's and
  lc4k's slopes only, fixed before any smoke.

| depth | 0 | 256 | 512 | 768 | 1,024 | slope (least squares) |
|---|---:|---:|---:|---:|---:|---:|
| ms per token (3 repetitions; spread ≤ 0.2 %) | 328.12 | 336.38 | 341.08 | 345.60 | 349.93 | **20.64 ms per 1,000 tokens** |

**Verdicts (at most one could hold):**
- **J4-QW holds:** the query-width law (per layer ∝ query dims, as Qwen3-0.6B's) predicted 21.03, −1.9 %.
- J4-KV fails: the KV-byte law of 11 October predicted 10.36, +99.3 %.
- J4-TWO fails: the two-term fit on lc4j and lc4k predicted 16.22, +27.3 %.

**Read with care:**
- The 4B could be measured only to 1,024 tokens of depth; TinyLlama and Qwen3-0.6B were measured to 8,192.
- Its first 256 tokens cost more (+8.26 ms) than each later 256 (+4.3 to 4.7 ms). Its slope over 256–1,024 alone is
  **17.6**, which no law predicts within 10 %.
- The other two models' first segments are steeper too, but diluted over 8,192 tokens.

**What the three models say together:**
- The KV-byte law fits only the model it was fitted on (Qwen3-0.6B). It under-prices TinyLlama 2.7× and Qwen3-4B 2×.
- The query-width law, calibrated on the 0.6B, fits the 4B (−1.9 %) but over-prices TinyLlama (6.43 against 4.22, +52 %).
- No single simple law fits all three. The planner needs the Jetson's depth cost per model; it is a 5-minute bench.

# 2026-10-28 — rpc9b, and the planner's defaults (PLANNER_V3) tested on new Qwen3.6-35B-A3B placements (def1, def2)

**rpc9b** (`rpc9bchar_predictions_20261028.json`; `testbed/results/rpc9bchar_20261028_015239`): rpc9 with one deep warm-up.
R9-MID, R9-ALL, R9-BACK hold, and R9-REPEAT now holds too: 0.0 % spread, every figure to the byte. The middle worker's deep
figure is 12,164 B, as predicted.

**def1** (`def1_predictions_20261028.json`): three new placements, server mean TPOT against two predictions:

| arm | placement (head) | measured | PLANNER_V3 (multipliers on blocks and head) | shipped |
|---|---|---:|---:|---:|
| H1 | laptop 25 + Pi 5 10 + Pi 4 6 (host) | 270.4 ms | 272.1 (−0.6 %) | 226.7 (+19.3 %) |
| H2 | laptop 25 + Pi 5 10 + Pi 4 3 + Jetson 3 (host) | 254.3 ms | 255.8 (−0.6 %) | 211.2 (+20.4 %) |
| H3 | laptop 25 + Jetson 3 + Pi 5 10 + Pi 4 3 (last: Pi 4) | 341.4 ms | 364.5 (**−6.3 %**) | 306.0 (+11.6 %) |

D1-V3 fails (H3 outside 5 %); D1-SHIP, D1-BETTER and D1-REPEAT hold. Found after the data: the multipliers had also scaled
the 417 MB output head.

**def2** (`def2_predictions_20261028.json`): the multipliers on the blocks only, three new head-last placements:

| arm | placement (head on) | measured | blocks only | blocks and head | shipped |
|---|---|---:|---:|---:|---:|
| L1 | laptop 25 + Jetson 3 + Pi 4 4 + Pi 5 9 (Pi 5) | 293.4 ms | 289.9 (+1.2 %) | 301.9 (−2.8 %) | 247.3 (+18.6 %) |
| L2 | laptop 25 + Pi 5 10 + Pi 4 4 + Jetson 2 (Jetson) | 309.1 ms | 303.1 (+2.0 %) | 317.7 (−2.7 %) | 261.2 (+18.3 %) |
| L3 | laptop 25 + Pi 5 10 + Jetson 3 + Pi 4 3 (Pi 4) | 343.1 ms | 347.1 (−1.2 %) | 364.6 (−5.9 %) | 306.0 (+12.1 %) |

All three hold: D2-BLOCKS (within 3 %), D2-BETTER, D2-REPEAT (0.06-0.28 %). **Adopted:** `PLANNER_V3 = {order_check, RPC_MSG,
node_mult="measured_blocks"}`, used by `serve_pool.sh`'s own planning and new registrations; `SERVE_PLANNER=shipped` plans as
before. On the gigabit cluster the RPC term moves predictions by only 0.0-0.4 %; its effect is on slow links (rpc4plan).

# 2026-10-28 — lc4n: the Jetson's depth cost is its flash-attention op, priced with no model loaded

The Jetson decodes on Vulkan with flash attention on auto. ggml-vulkan picks the attention kernel's layout per call from the
shape and the cache's length, so the depth cost has no single formula in the model's shape (lc4charb, lc4j, lc4k, lc4m).
The Vulkan per-op timer (exploration, `lc4nx_20261028_080414`) shows that of a decoded token's ops only FLASH_ATTN_EXT grows
with the cache. **fa_bench** (`fa_bench_20261028.cpp`) times that op alone for an attention shape: one graph of the model's
layers, chained, with no model loaded. v2 is `lc4nfa_20261028_081317`. lc4n (`lc4n_predictions_20261028.json`,
`testbed/results/lc4n_20261028_081724`) is llama-bench on the Jetson alone, three repetitions, F16 cache, against
fa_bench's predictions:

| bench | measured slope, long range (ms / 1,000 tokens) | fa_bench | depth 768 − depth 512 | fa_bench | 384 vs 256 |
|---|---:|---:|---:|---:|---:|
| TinyLlama Q4_K_M (1,024–4,096) | 4.022 | 4.035 (−0.3 %) | −1.296 ms | −1.535 | +1.06 % |
| TinyLlama Q8_0 (1,024–4,096) | 4.053 | 4.035 (+0.4 %) | −1.289 ms | −1.535 | +0.64 % |
| Qwen3-0.6B Q8_0 (1,024–4,096) | 7.945 | 7.994 (−0.6 %) | +2.008 ms | +2.040 | +0.88 % |
| Qwen3-4B Q4_K_M (256–768) | 18.306 | 16.616 (**+10.2 %**) | +4.969 ms | +4.259 | +0.59 % |

- **J5-SWITCH holds:** TinyLlama decodes faster at depth 768 than at 512, where its attention changes path at 1,024 cache
  cells. Qwen3-0.6B, with no change of path, decodes slower.
- **J5-QUANT holds** (Q8_0 within 0.77 % of Q4_K_M), and **J5-REPEAT holds** (at most 0.72 %).
- **J5-FA fails on Qwen3-4B** (+10.17 % against 10 %); the other three are within 0.7 %.
- **J5-STAIR fails:** the shader skips a block whose mask is all −inf (`flash_attn.comp`, found after the data), so the
  cost follows the real depth, not the padded cache.
- Per attention layer and 1,000 tokens of depth, fa_bench gives:
  - TinyLlama 0.18 (0.40 below 1,024 cells); Qwen3-0.6B 0.29; Qwen3-4B, Mistral-7B and Mistral-Nemo 0.47;
  - Qwen3-14B 0.51; Qwen3-32B 0.51; Qwen3-30B-A3B 0.25;
  - Qwen3.5-9B 1.13; Qwen3.6-35B-A3B 0.42; Qwen3.8-27B 0.60 (the hybrids per attention layer).
- In the planner as the Jetson's `fa_table` law (`attention_cost_20261028.json`, opt-in through `predict_pool.ATTENTION_FA`).

# 2026-10-28 — ph1: where to prefill and where to decode one request (the phase scheduler)

The scheduler (`phase_plan_20261028.py`) prices three plans for one request on Qwen3-0.6B Q8_0:
- L, the laptop alone;
- J, the Jetson's GPU alone;
- H, prefill on the laptop with the slot handed to the Jetson (`phase_serve_20261028.py`).

Its costs come from a calibration through the same serving path (`phase_costs_20261028.json`) and the energy model's power
table. It picks by energy at all four outlets or by time. ph1 (`ph1_predictions_20261028.json`,
`testbed/results/ph1_20261028_090849`) used 256 generated tokens, every plan twice, interleaved, with the strip at 5 Hz.
Each cell is the mean, with the error against the prediction in brackets:

| prompt | L: laptop alone | J: Jetson alone | H: hand-off | least energy / time (predicted) |
|---:|---|---|---|---|
| 1,024 | 22.1 s, 993 J (−4.9 %, +7.6 %) | 35.9 s, 1,129 J (−1.4 %, +3.3 %) | 36.8 s, 1,339 J (−0.6 %, +14.3 %) | L / L (L / L) |
| 2,048 | 41.3 s, 1,870 J (−1.7 %, +12.8 %) | 48.6 s, 1,554 J (−1.3 %, +6.2 %) | 51.6 s, 1,880 J (+0.5 %, +12.1 %) | J / L (J / L) |
| 4,096 | 88.9 s, 3,771 J (0.0 %, +8.4 %) | 82.2 s, 2,607 J (−1.0 %, +7.2 %) | 91.7 s, 3,321 J (+2.8 %, +9.4 %) | J / J (J / J) |
| 8,192 | 216.4 s, 8,835 J (−1.5 %, +3.7 %) | 181.7 s, 5,796 J (−0.8 %, +9.5 %) | 200.9 s, 7,717 J (−0.6 %, +7.8 %) | J / J (J / J) |

- **What held:** PH-PICK-E and PH-PICK-T (every pick, including the energy/time split at 2,048), PH-TIME (within 5 %) and
  PH-ENERGY (within 15 %; every energy 3–14 % above the power table's prediction).
- **What failed:**
  - PH-TTFT: H's first token came 8.2 and 6.0 % after L's at 2,048 and 4,096 tokens.
  - PH-REPEAT: H at 2,048 had a 6.0 % spread.
- **Why, found after the data:** the laptop prefills up to 18 % slower right after a request that kept it busy to the end.
  The CPU's short-term turbo budget refills while it idles during a Jetson decode.
- **The hand-off never pays for one user here:** by the length at which the Jetson decodes faster (~3,000 tokens of
  context), its GPU prefills about as fast as the laptop's CPU.
- **Text:** H's text matches L's for all 256 tokens at 1,024. At longer prompts it parts after 34–49 tokens, and J's after
  1–199 tokens (CPU against GPU numerics under greedy decoding).

# 2026-10-28 (afternoon) — why Qwen3-4B missed in lc4n; per-model power rows for the phase scheduler (ph2)

**Qwen3-4B's +10.2 % in lc4n, explored after the data** (`testbed/results/lc4nq_20261028_121358`; not registered). lc4n's
bench was repeated plain and with the Vulkan per-op timer, under a 1 Hz memory monitor:
- The wall slope repeats at 18.33 ms per 1,000 tokens.
- The attention op in the model costs 17.25 (fa_bench predicted 16.62: −3.8 %).
- The other GPU ops add 0.54: every weight product slows about 0.1 % per 512 cache cells.
- Time outside the GPU adds 0.54.
- The board fell to 507 MB free and swapped ~19 MB, with 1,253 major faults. TinyLlama, the control, kept 2.85 GB free and
  swapped nothing.
- So the miss is ~4 % fa_bench and ~6 % memory pressure near the 4 GB limit. On the Tegra the CPU and GPU share one DRAM.

**Power rows per model and phase** (`phase_power_20261028.py fit` on ph1's strip data → `phase_power_20261028.json`):

| row | laptop | Jetson | TinyLlama table (laptop / Jetson) |
|---|---|---|---|
| prefill | 40.5 W for the first 20.4 s of load, then 30.6 W | 8.3 W | 29.3 / 8.9 W |
| decode | 35.6 W inside that window, then 33.1 W | 9.4 W | 32.1 / 11.2 W |
| idle while serving | 17.1 W | 3.2 W | 14.1 / 3.1 W |
| during a hand-off | 24.4 W | 4.2 W | (idle assumed) |

**ph2** (`ph2_predictions_20261028.json`, `testbed/results/ph2_20261028_122445`): held-out prompts of 1,536 and 6,144 tokens,
256 generated, L/J/H twice, **60 s between requests** (ph1: 10 s). **All seven hypotheses hold.**

| arm | total s (error) | energy J (rows' error / table's error) | first token s |
|---|---|---|---|
| 1,536 L | 30.9 (−4.1 %) | 1,423 (−2.1 % / +11.6 %) | 11.9 |
| 1,536 J | 42.0 (−1.0 %) | 1,339 (+0.3 % / +5.6 %) | 13.6 |
| 1,536 H | 43.2 (−1.4 %) | 1,598 (−1.0 % / +13.3 %) | 12.1 |
| 6,144 L | 148.5 (+0.3 %) | 6,067 (−1.2 % / +5.2 %) | 92.0 |
| 6,144 J | 126.7 (−0.7 %) | 3,924 (−1.0 % / +5.8 %) | 88.9 |
| 6,144 H | 139.7 (+0.1 %) | 5,290 (−0.8 % / +8.8 %) | 92.5 |

- **Picks:** J by energy and L by time at 1,536; J / J at 6,144.
- **Energy:** with the rows every arm is within 2.1 %, against 5.2–13.3 % with the table (PH-ENERGY, PH-BETTER).
- **First token:** with the turbo budget refilled, H's first token is within 1.3 % of L's (PH-TTFT). This confirms ph1's
  post-hoc explanation.
- **Repeats:** within 0.8 %.
- **Adopted:** `phase_plan_20261028.choose(..., rows=)` takes the rows, and the serving path's `ask` uses them by default.

# 2026-11-01 — the scheduler review's fixes: PLANNER_V4 (a hybrid's memory, depth on every node), the laptop's power refit (v4a)

The user: "do item 1 to 4 then 6-7", the 1 November review's items (`testbed/JOURNAL.md`). This section has the data behind
items 2-4; the code and items 1, 6 and 7 are in the journal.

**A hybrid's memory per block** (`hybrid_shapes_20261101.py` → `testbed/dataset/hybrid_shapes_20261101.json`). llama.cpp at the
pin gives a qwen35 / qwen35moe block a KV cache only when (i + 1) % full_attention_interval == 0, every other block below
n_layer a fixed recurrent state (f32, `llama-hparams.cpp`: (d_conv − 1) × (d_inner + 2 × n_group × d_state) +
d_state × d_inner values), and the MTP block neither. `model_shapes.json` carried no full_attention_interval, so the planner
charged every block a full-attention cache. From the three GGUF headers (interval 4 in all three):

| model | attention + recurrent blocks | state per recurrent block | at 4,096 cells: KV + state | the planner charged |
|---|---|---:|---:|---:|
| Qwen3.5-9B | 8 + 24 | 2.094 MiB | 128.0 + 50.3 MiB | 528.0 MiB |
| Qwen3.8-27B | 16 + 48 | 3.117 MiB | 256.0 + 149.6 MiB | 1,040.0 MiB |
| Qwen3.6-35B-A3B | 10 + 30 | 2.094 MiB | 80.0 + 62.8 MiB | 328.0 MiB |

Checked against the 35B's 28 October serve (laptop 25 + Jetson 3 + Pi 4 4 + Pi 5 9; llama.cpp's own buffer lines): every
device's KV and recurrent-state buffer within 0.003 MiB (laptop 48.00 + 39.78, Jetson 8.00 + 4.19, Pi 4 8.00 + 6.28, Pi 5
16.00 + 12.56 MiB).

**The attention table for every node** (`attention_cost_20261101.py` → `attention_cost_20261101.json`): the 28 October table,
the Pi 5's query-head law from lc4charb (24.605 ms per 1,000 tokens on TinyLlama → 0.5461 ns per query-head dimension; the
Pi 5 had none, so a plan with depth and the Pi 5 raised a TypeError), and the hybrids' attention fraction (one block in four)
on every board, where only the Jetson's law applied it.

**PLANNER_V4** = PLANNER_V3 + `hybrid_memory` + `attn_table` (`predict_pool.py`). Planned on 28 October's snapshot, the 35B at
4,096 cells gets a 25th laptop block (259.2 → 242.9 ms predicted); on 1 November's, where the laptop had ~900 MiB less, the
two planners agree at 4,096 and part at 16,384 (below). Dense models plan exactly as before.

**The laptop's power refitted** (`energy_refit_20261101.py` → `power_refit_20261101.json`), from the four-node evaluation's own
strip data: per placement, every outlet's mean W over the decode windows of its requests. The boards match their table
(median error +1.5 / −7.6 / +1.9 % for the Pi 5, Pi 4, Jetson); the laptop does not. In those sessions it idled at 20.47 W
(20.0–21.6; the table: 14.05) and decoded alone at 35–42 W (the table: 32.12). Fitted:
`P = idle + (active − idle) × f + per_token / T (the host) + pool_host (hosting workers)`, with idle 20.47 W, active 35.74 W,
per_token 0.217 J and pool_host 1.62 W; f is the laptop's busy fraction by the predictor's terms, T the time per token.

| all four outlets, J per decoded token, 52 placements | median error | worst | within 10 % |
|---|---:|---:|---:|
| the energy model as registered | −21.6 % | −32.3 % | 3 |
| refit, in sample | −0.1 % | −8.0 % | 52 |
| refit, each model left out of its own fit | −0.1 % | −8.0 % | 52 |

**v4a** (`v4a_predictions_20261101.json`, registered 06:15; `testbed/results/v4a_20261101_061538`): Qwen3.6-35B-A3B Q6_K at
16,384 cells, both arms planned on the 06:12 snapshot and priced by PLANNER_V4, served with `SERVE_PLANNER=v4` (the gate
charges as V4 does), order K3 K4 K4 K3, five requests of 128 + 64 each, the strip at 5 Hz.

| arm | plan (head on the laptop) | server mean TPOT (predicted) | J per token, all outlets | refit / old model |
|---|---|---:|---:|---:|
| K3, PLANNER_V3's pick | laptop 22 + Jetson 2 + Pi 5 10 + Pi 4 7 | 296.95 ms (299.1, −0.7 %) | 9.728 | 11.844 (+21.8 %) / 9.490 (−2.4 %) |
| K4, PLANNER_V4's pick | laptop 23 + Jetson 3 + Pi 5 10 + Pi 4 5 | 269.74 ms (274.5, −1.7 %) | 9.238 | 11.070 (+19.8 %) / 8.924 (−3.4 %) |

- **V4-KV holds:** V4's plan is **9.2 % faster** (predicted 8.2 %), and uses 5.0 % less energy per token.
- **V4-ACC holds** (−0.7 and −1.7 %); **V4-REPEAT holds** (0.06 and 0.20 %).
- **V4-MEM holds:** on K4 every board held less than V4 charged it (worst: the laptop 28 MiB under its charge, the Pi 5 24).
  So the accounting is right at 16,384 cells too.
- **E-REFIT fails** (+21.8 and +19.8 %), and **E-OLD fails** (the old model only 2.4 and 3.4 % low). All of the miss is the
  laptop's outlet: 20.33 W measured against 27.58 refit (19.71 old) on K3; the boards are within 0.4 W of both. Two reasons,
  found after the data:
  - the laptop's idle baseline moves between sessions: 17.1 W in this one's idle window (and in ph1/ph2's, 28 Oct), 20.5 W
    in the evaluation's (20-21 Oct). The refit fixed it at the evaluation's;
  - above that baseline the laptop drew only 3.2 W while computing about 31 % of each token of this hybrid MoE, against the
    refit's 7.1 W. Its dynamic power is model-dependent, as ph2's per-model rows found on the phase scheduler.
  The old model's near-match is two errors cancelling: an idle row 3 W below this session's, a busy row 3.6 W below the
  evaluation's. **The refit stays an option** (`energy_model.energy_refit`); it is not adopted.

**v4d** (`v4d_predictions_20261101.json`, registered 07:10; `testbed/results/v4d_20261101_071003`; lc4planb's design and
runner): context depth priced by default, on `serve_pool.sh`'s default model, Qwen3-30B-A3B Q3_K_M, at ctx 8,192. A is
PLANNER_V3's pick (the laptop alone), B PLANNER_V4's priced at half the context, 4,096 tokens (laptop 40 + Jetson 8, the
head on the laptop). Each is a laptop-hosted llama-bench `-p 0 -n 32 -d 0,4096 -r 3`, order A B B A, with the Jetson's guard
and locked clocks.

| plan | depth 0 | predicted | depth 4,096 | predicted | slope (ms / 1,000 tokens) |
|---|---:|---:|---:|---:|---:|
| A, the laptop alone | **89.9 ms** | 99.8 | **378.5 ms** | 540.2 | 70.5 (the law: 107.5) |
| B, laptop 40 + Jetson 8 | 146.6 ms | 145.1 | 393.7 ms | 520.4 | 60.3 |

- **D4-SHORT holds:** at depth 0 the laptop alone is faster.
- **D4-LONG fails:** at 4,096 tokens the plan the depth term chose is **4.0 % slower**.
- **D4-ACC fails:** both plans 24–30 % below their predictions at 4,096 (A at depth 0 is 9.9 % below: llama-bench has no
  sampling or HTTP, which the planner's rows carry).
- **D4-LAW fails:** the laptop's attention costs 70.5 ms per 1,000 tokens on this shape against its query-head law's 107.5
  (−34.5 %). The law is a least-squares line through depths whose cost curves upward: lc4charb's TinyLlama on the laptop,
  30.0 → 57.1 → 100.4 → 232.6 ms at 0 / 2,048 / 4,096 / 8,192, has a 0–4,096 secant 0.70 of its 0–8,192 slope, about this
  bench's ratio. Over-pricing the laptop's attention pushed 8 blocks onto the Jetson for nothing.
- **What changes:** interactive serving prices depth only when asked (`PLAN_DEPTH`, default 0;
  `serve_plan_20261101.plan_depth`). The Pi 5's law and the hybrid fraction stay in the table, so a plan with depth no
  longer crashes, and a long session can ask for it (lc4planb: at 8,192 tokens the depth-priced plan was 1.81× faster). A CPU
  law that bends with depth is work left.

**The refit on the sessions it was not fitted on** (`energy_holdout_20261101.py`, after v4a; post hoc, not registered;
`testbed/results/energy_holdout_20261101/holdout.json`). Every engine arm since the evaluation (draft arms and prima.cpp's own
left out), 24 arms in 10 sessions (sched2, prima4 ×2, order35, spec4, c35, c30, def1, def2, v4a), at each arm's measured time
per token, all four outlets:

| J per decoded token | median error | median \|error\| | worst | within 10 % |
|---|---:|---:|---:|---:|
| the old model (power_table) | −11.2 % | 11.2 % | −16.7 % | 7 of 24 |
| the refit (the evaluation's laptop idle, 20.47 W) | +10.4 % | 10.4 % | +20.9 % | 12 of 24 |
| the refit with the session's own laptop idle | −1.2 % | 2.6 % | −12.4 % | 21 of 24 |

The laptop idled at 13.0–17.7 W in these sessions' idle windows. Shifted to each session's own idle (its busy row by the same
amount: the baseline, display and desktop, adds to both), the refit fits; the worst are scheduler v2's 32B Q6_K plans (−12.4,
−10.2 %) and v4a (+10.6, +8.0 %). `energy_model.energy_refit(..., laptop_idle_w=)` takes it, opt-in. Not adopted on this
evidence alone: a registered test, with the laptop's idle read before the run, is next.

# 2026-11-01 — so6: every idle board switched off, live (the review's item 7): all five tests hold

`serve_pool.sh` now switches every board a plan leaves out (`strip_switch_20261101.py`: the Pi 4, the Pi 5 and the Jetson; the
Pis powered off, the Jetson halted, as `shutdown_all.sh` halts it; several at once; the Jetson's power guard started after it
boots), for a person at a terminal; a harness never switches. so6 (`so6_predictions_20261101.json`, registered 07:39;
`testbed/results/so6_20261101_073945`) ran `serve_pool.sh` as a person's serve runs (`SERVE_SWITCH=on`, `SERVE_PLANNER=v4`,
the four nodes, live memory), the strip at 5 Hz:
- **A:** the default model at ctx 4,096 on the laptop alone. After READY all three boards were shut down and cut at once
  (shutdown to outlet cut: Pi 5 31.3 s, Pi 4 32.3 s, Jetson 36.1 s); three 32-token completions, about 92 ms a token, with
  the wall at 39.6 W in all.
- **B:** Qwen3.6-35B-A3B at ctx 4,096, which needs every node. Planned on the newest snapshot for the three silent boards,
  it powered all three up at once (outlet on to back: Pi 5 45.8 s, Jetson 46.1 s, Pi 4 59.6 s), started the Jetson's guard
  (ok), gated them live and served laptop 24 + Jetson 2 + Pi 5 9 + Pi 4 6; one completion at 308 ms a token.

**All five hold:** SO6-PLAN, SO6-OFF (every strip sample of outlets 0-2 in A's completions at 0.0 W, 56 each), SO6-SERVE,
SO6-UP and SO6-LAPTOP (the laptop's outlet on throughout). The cluster ended with every board on.

# 2026-11-02 — e2: the laptop's refitted power with the session's own idle, on Qwen3-32B Q6_K: it fails; the laptop's state moves

`e2_predictions_20261102.json` (registered 04:13; `testbed/results/e2_20261102_041357`, 04:13–06:31). The refit
(`power_refit_20261101.json`) with the laptop's idle read from the session's own idle-before window, on scheduler v2's model
and context (ctx 2,048), outside the refit's data and where the post-hoc look missed most. The prediction was fixed at
registration as a formula with one measured input: J = T × (boards' W + idle_s + 15.27 × f_laptop + 1.62) + 0.217. Order
E1 E2 E2 E1, three requests of 128 + 64 each.

| arm | plan | server mean (predicted) | J/token, all outlets | at predicted T | at measured T | old table |
|---|---|---:|---:|---:|---:|---:|
| E1, V4's pick | laptop 42 + Pi 5 18 + Jetson 4 (head on the Jetson) | 1,540.7 ms (1,713.3, −10.1 %) | 63.34 | +10.0 % | −1.1 % | −12.1 % |
| E2, 6 GiB less on the laptop | laptop 25 + Pi 5 18 + Pi 4 15 + Jetson 6 (head on the laptop) | 2,890.4 ms (3,092.4, −6.5 %) | 93.48 | +20.4 % | +12.6 % | −1.4 % |

- **E2-REFIT, E2-POWER and E2-BETTER fail; E2-REPEAT holds** (0.10 and 0.05 %).
- **Why, read after the data:** the laptop's idle moved within the session. Its idle-before window read 19.4 W, minutes after
  someone was at the cluster (the strip's buttons had switched the boards on at 04:09: their outlets read "short_push"), and
  its idle-after window read 13.7 W; the quietest
  minute of every 10-minute span of the session was 13–15 W. Its display went off soon after the start, so the reading the
  formula took was not the run's baseline. On E2, where the laptop is busy 17 % of each token, it drew 18.6 W: below its own
  idle reading. Against a 13.7 W baseline the laptop drew 13.7 W above it on E1 (busy 48 % of each token, the dense Q6_K
  model), about three times what it drew above its baseline for v4a's hybrid MoE at a similar share. So its draw while
  computing depends on the model too.
- The boards match their table again. The time predictor is 6.5–10 % slow on this dense model, as the review found.
- **What it means:** neither one formula nor one session reading predicts the laptop's draw within 10 %. Its display state
  (about 6 W) and the model it computes decide it. The refit stays an option, not adopted. An energy prediction that holds
  needs the laptop's state controlled during the run (the display off) and per-model rows, as ph2 used.

# 2026-11-02 — q9: a request priced as prefill and decode (item 9), INCOMPLETE

`q9_predictions_20261102.json` (registered 06:44; `testbed/results/q9_20261102_064400`). Interactive serving plans by decode
time; for a request of P prompt tokens and G generated, `predict_pool.objective_ms` now prices P × prefill + G × decode
(before, `plan_host_choice` scored a workload on its prefill alone). On 2 Nov's memory only Qwen3-30B-A3B Q6_K changes plan
under it. Arms at ctx 4,096, three requests of 512 + 64 per arm-run, order D R R D:

| arm | plan | predicted TTFT / request / decode | measured (one arm-run) |
|---|---|---|---|
| D, by decode | laptop 33 + Pi 5 14 + Jetson 1 (head on the laptop) | 35.4 s / 45.2 s / 155.7 ms | 37.19 s / 48.79 s / 184.1 ms |
| R, by the request | laptop 33 + Pi 5 10 + Jetson 5 (head on the laptop) | 32.6 s / 42.9 s / 163.6 ms | 34.81 s / 46.63 s / 187.5 ms |

- **Every hypothesis INCOMPLETE.** Arm-runs 1 (D) and 2 (R) ran. Arm-run 3 waited on the session's quiet gate (the Zen
  browser, open from 07:02:41) until the unit was stopped at 08:13:33. Zen was also open through all three of arm-run 2's
  requests: the gate is checked before an arm-run, not during it.
- What the halves showed, untested:
  - R's request 4.42 % faster (predicted 5.26 %);
  - the trade the objective makes: the first token sooner, each token slower;
  - the time to the first token 4.9 and 6.9 % above prediction;
  - the time per token 18.2 and 14.6 % above it (predicted at depth 0, as serving plans; these requests decode at
    512–576 tokens of depth).
- Gross energy per request, all four outlets (reported, not tested): D 1,968 J, R 1,785 J.
- Re-run complete as q9b (below).

# 2026-11-03 — q9b: q9 again, complete: the request objective's plan is not faster (Q9-REQ fails)

`q9b_predictions_20261103.json` (registered 12:57; `testbed/results/q9b_20261103_125720`, 12:57–13:37, the laptop quiet in every
arm-run, Zen closed). q9's design, hypotheses and scorer on a fresh snapshot (`memory_snapshot_20261103_125622.json`; the laptop
1.1 GiB freer than on 2 Nov, so both plans moved). Qwen3-30B-A3B Q6_K at ctx 4,096, three requests of 512 + 64 per arm-run,
order D R R D.

| arm | plan | TTFT (predicted) | request (predicted) | server mean TPOT (predicted; at depth 544) | J per request |
|---|---|---:|---:|---:|---:|
| D, by decode | laptop 36 + Pi 5 12 (head on the laptop) | 32.79 s (34.27, −4.3 %) | 43.01 s (43.47) | 162.1 ms (146.0, +11.1 %; 204.5) | 1,551 |
| R, by the request | laptop 36 + Pi 5 7 + Jetson 5 (head on the laptop) | 31.90 s (30.77, +3.7 %) | 43.25 s (40.76) | 180.2 ms (158.5, +13.7 %; 211.5) | 1,590 |

- **Q9-REQ FAILS:** R's request was 0.56 % *slower* than D's (predicted 6.23 % faster).
- **Q9-TRADE HOLDS:** R's first token came sooner and each token slower, the trade the objective makes.
- **Q9-ACC FAILS:** the time to the first token is within 4.3 %; the time per token is 11.1 and 13.7 % above prediction
  (tolerance 10 %).
- **Q9-REPEAT HOLDS:** each arm's two runs within 0.19 and 0.04 %.
- **Where the gain went:** the predictor expected R's first token 3.50 s sooner and its 63 later tokens 0.79 s slower, a
  2.71 s gain. Measured: 0.90 s sooner and 1.14 s slower, a 0.24 s loss. The first-token gain was a quarter of the predicted.
  - In q9b, D has no Jetson, so R's gain rested on the Jetson's GPU prefilling five MoE blocks. Its prefill is priced from
    TinyLlama's Vulkan rows, a dense model.
  - In q9's halves, D already held one Jetson block, and R's four more gave 2.38 of the predicted 2.87 s.
  - Why q9b's gain was so much smaller is not established.
- **Decode with context depth (reported, not tested):**
  - priced at depth 0, as serving plans, the time per token came in 11–14 % low;
  - priced at the requests' mean decode depth (544) with PLANNER_V4's straight-line attention table, it came in 26.1 %
    (D) and 17.4 % (R) high.
  - The truth lies between the two, as v4d found for the CPUs' depth law.
- **What it means for item 9:**
  - The objective's form is right: it made the predicted trade twice (q9's halves and q9b).
  - The gains it chases are 5–6 %, smaller than the predictor's plan-to-plan prefill error where the plans differ in the
    Jetson's share (here an 8-point swing).
  - It stays an option for callers that pass a workload. It is not given to interactive serving (no `PLAN_WORKLOAD`).
  - Next would be the Jetson's MoE prefill rows and a depth law that bends.

# 2026-11-03 — item 10's exploration: speculation with no draft model, and what a check costs (not registered)

`explore_spec_20261102.py`, read by `spec_cost_20261103.py`. One server per setting (the pin's server takes no per-request
draft size), a fixed plan, `SERVE_PLANNER=v4`. Each serve with no speculation first runs a prompt-length scan: one batch of
m tokens, the server's prompt_ms, the median of three. Then three greedy requests of 128 tokens on each of two prompts:
- free: an explanation, little to copy;
- copy: repeat a passage it is given.
The serve_pool.sh split fix came first: with `--spec-type draft-mtp`, llama.cpp loads the MTP block on the board holding the
last blocks, which the split check had always left out.

**One batch of m tokens against one decode step**

| plan | decode step | m 2 | m 3 | m 4 | m 8 | m 16 | m 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| B: Qwen3.5-9B Q4_K_M, laptop alone | 270.0 ms | 1.38 | 1.85 | **1.69** | 3.47 | 6.08 | 24.9 |
| A: Qwen3.6-35B-A3B Q6_K, laptop 24 + Jetson 2 + Pi 5 9 + Pi 4 6 | 288.4 ms | **1.31** | 1.70 | 1.96 | 4.92 | 7.86 | 22.4 |

- On the laptop a batch of 4 costs less than one of 3. My reading, unconfirmed: the CPU kernels work in groups of 4 tokens.
- A's scan was predicted before it ran from dense curves (B's laptop, TinyLlama's board rows;
  `testbed/dataset/explspec_A_precommit_20261103.json`). Against T(1) it was 31–33 % high at m 2–4, right at m 8 and 59 %
  high at m 64. MoE blocks batch more cheaply than dense ones.

**Speculation against none** (time per token; tokens accepted / drafted)

| plan | prompt | MTP 1 | MTP 2 | MTP 3 | n-gram |
|---|---|---:|---:|---:|---:|
| B, laptop alone | free | 0.986× (75 %) | 0.862× (63 %) | 0.926× (53 %) | 0.86× (noisy) |
| B, laptop alone | copy | 1.044× (92 %) | 1.012× (87 %) | **1.103×** (74 %) | **1.156×** |
| A, four nodes | free | **1.180×** (84 %) | 1.154× (75 %) | 1.117× (70 %) | 1.62× (noisy) |
| A, four nodes | copy | 1.246× (95 %) | 1.299× (91 %) | **1.352×** (92 %) | 1.142× |

- **Checking is cheap where the plan is pooled.** On A every MTP setting won 12–35 %. On the laptop alone MTP won only on
  copy-heavy text. Two notes on the noisy n-gram figures:
  - on free text the greedy output repeated itself and n-gram drafted from that (spreads of 33 and 122 %);
  - its drafts are long (up to 64 tokens after a 24-token match), so it pays only when the text copies.
- **A check pass** costs the batch check plus a draft cost per drafted token: 75.4 ms on A, 99.3 on B.
  - Both fit two passes of the output head (vocabulary 248k) per drafted token, the draft's own and the check's extra row,
    plus the MTP block on the board that holds it.
  - That is the Pi 4 on A, since the MTP block lands with the last blocks.
- **The Pis' threads** (item 9; `explthreads_20261103_133920`, llama-bench `-r 3`, one invocation each):

  | board, model | decode t/s at 3 / 4 threads | prefill t/s at 3 / 4 threads |
  |---|---:|---:|
  | Pi 5 (replacement), TinyLlama Q4_K_M | 18.79 / 17.62 (+6.6 % at 3) | 61.70 / 81.24 |
  | Pi 5 (replacement), Qwen3.5-9B Q4_K_M | 1.96 / 1.80 (+8.9 %) | 7.31 / 9.14 |
  | Pi 4, TinyLlama Q4_K_M | 5.89 / 5.53 (+6.5 %) | 9.49 / 12.52 |
  | Pi 4, Qwen3.5-9B Q4_K_M | 0.68 / not measured | 1.22 / not measured |

  The Pi 4's 9B at 4 threads was stopped at 80.3 °C (throttled=0x80000) before item 10's config A; see the folder's
  `NOTE.txt`. September's knee holds on the replacement Pi 5 and at serving size. Pooled serving runs both Pis at `-t 4`,
  so a 3-thread decode would gain a few % of a pooled plan's step at most; not built.

# 2026-11-03 — sp10: the planner's checking cost, INCOMPLETE (two MTP serves hung with the MTP block on the Pi 5)

`sp10_predictions_20261103.json` (registered 15:13; `testbed/results/sp10_20261103_151444`).

**The model** (`spec_model_20261103.py`): a check pass with n drafted tokens costs dec × R(n+1) + n × d + i.
- R and i come from the same model's exploration plan; dec is the planner's decode, calibrated there.
- The draft d = κ(2H + b_last) + the visit + an RPC cost when the MTP block is on a worker. H is the head and b_last the MTP
  block on its node. κ = 1.0685 comes from B and the RPC cost, 10.32 ms per draft, from A.
- A pass yields the exploration's tokens per pass for the same model, setting and prompt.

**Two plans the model hadn't seen:**
- P2: the 35B with the Pi 5 last;
- P1: the 9B on laptop 24 + Pi 5 9.

| plan | R at m 2 / 3 / 4 (err) | MTP 1 free / copy | MTP 2 free / copy | MTP 3 |
|---|---|---|---|---|
| P2 | 1.308 / 1.716 / 2.101 (+0.2 / +0.8 / +7.0 %) | 1.248× / 1.320× (pred. 1.233 / 1.308) | 1.230× / 1.397× (pred. 1.186 / 1.344) | hung |
| P1 | 1.360 / 1.788 / 1.892 (−1.1 / −3.5 / **+12.3 %**) | **1.064×** / 1.138× (pred. 0.969 / 1.071) | hung | not run |

- **Every hypothesis INCOMPLETE.** Two serves hung, both with the MTP block on the Pi 5 and two or more drafted tokens.
  - P2's MTP 3 hung on its third request, after the second had slowed to 955 ms a token. The server aborted on its next
    send to the Pi 5 ("Remote RPC server crashed or returned malformed response").
  - P1's MTP 2 hung on its first request.
  - In both, the Pi 5's kernel gave up on its replies after 15.7 minutes of unacknowledged retransmission
    (TcpExtTCPAbortOnTimeout 2). The laptop held the connection established with nothing queued.
  - ~~Hypothesis, unconfirmed: the laptop's ufw drops packets its strict connection tracking marks INVALID.~~ Withdrawn
    at 19:45: ufw is disabled, and a reproduction inspected live showed the Pi 5's network driver holding the reply
    (`testbed/results/pi5_tx_stall_20261103/NOTE.md`).
  - Details and raw logs: the folder's `NOTE.txt`. sp10 was stopped before P1's MTP 3.
- What completed: on the 35B the model held within 4 % for every arm and within 7 % on the check cost. On the 9B it missed
  the batch of 4 (12 %): the laptop's discount there doesn't extend to the Pi 5's share. It also underrated MTP 1, which won
  6.4 % on free text where it predicted a 3 % loss.

# 2026-11-03 — sp10c: the checking cost, the MTP block off the Pi 5: holds on the 35B, fails on the 9B's pooled plans

`sp10c_predictions_20261103.json` (registered 16:43; `testbed/results/sp10c_20261103_164359`, 16:43–17:48, every arm complete,
no hang). sp10's model, prompts, scorer and hypotheses on two plans it hadn't seen, keeping the MTP block off the Pi 5:
- P3: the 35B with the Jetson last, laptop 24 + Pi 5 9 + Pi 4 6 + Jetson 2 (the MTP block on the Jetson's GPU);
- P4: the 9B on laptop 24 + Pi 4 9 (the MTP block on the Pi 4).

| plan | decode step | R at m 2 / 3 / 4 (err) | draft (pred.) | MTP 1 free / copy | MTP 2 free / copy | MTP 3 free / copy |
|---|---:|---|---:|---|---|---|
| P3 | 297.1 ms | 1.281 / 1.704 / 1.918 (−1.9 / +0.1 / −2.4 %) | 63.3 ms (65.8) | 1.184× / 1.300× (1.224 / 1.299) | **1.206×** / 1.368× (1.174 / 1.330) | 1.123× / **1.428×** (1.171 / 1.435) |
| P4 | 531.6 ms | 1.273 / 1.759 / 1.996 (−7.4 / −5.1 / **+18.4 %**) | 136.0 ms (147.3) | **1.102×** / **1.179×** (1.038 / 1.148) | 0.995× / 1.164× (0.917 / 1.112) | 0.886× / 1.106× (1.022 / 1.262) |

- **SP-SIGN and SP-DRAFT hold:** all 10 arms predicted ≥ 5 % from even landed on their side, and the draft cost was within
  3.8 % and 7.7 %.
- **SP-CHECK, SP-SPEEDUP and SP-BEST fail, on the 9B's P4 alone.**
  - The model carries the laptop's whole-plan batch curve (B's) onto a plan where the Pi 4 holds 9 of 32 blocks. The
    laptop's batch of 4 costs less than its batch of 3; the Pi 4's doesn't. So the model priced the batch of 4 18 % low,
    MTP 3 12–13 % too fast, and picked MTP 3 for copying where MTP 1 measured 6.2 % faster.
  - P1 in sp10 missed the same way (+12.3 % at m 4).
- **On the 35B it holds:**
  - P3: every speedup within 4.1 %, the batch cost within 2.4 %, the draft cost within 3.8 %;
  - P2's arms that completed in sp10 were within 3.9 %.
- **Across the 3 Nov runs, MTP with no draft model on pooled plans of the 35B gained 12–43 %** (A, P2, P3): 1.12–1.25× on
  free text and 1.25–1.43× on copying. The best setting was MTP 1–2 for free text and MTP 3 for copying.
- On the 9B, free text under MTP 1 won 6–10 % on the pooled plans (P1, P4) and lost 1.4 % on the laptop alone. Copying won
  under every MTP setting (1.01–1.18×).
- **Not done:**
  - per-node batch curves, which the 9B's pooled plans need (the laptop's from its own scan, each board's from its own);
  - a serving default that turns MTP on: two serves hung with the MTP block on the Pi 5 (sp10). The cause was found at
    19:45: the Pi 5's driver holds a reply (`pi5_tx_stall_20261103`). serve_pool.sh now refuses that placement under MTP.

# 2026-11-03 — the Pi 5 hang: its network driver held replies with hardware TSO on; TSO off, three of three rounds hold

`testbed/results/pi5_tx_stall_20261103/` (the diagnosis, 19:30–19:45, with `NOTE.md`) and
`testbed/results/pi5_tso_check_20261103_220804` (the check, 22:08–22:19).

**What hung:** sp10's two MTP serves, a deliberate reproduction (the 9B on laptop 24 + Pi 5 9, MTP 2: its first real
request), and later sp10d's three requests. Every one had the Pi 5 in the plan.

**The live state during the reproduction (19:43):**
- the Pi 5 held 2,896 bytes sent and never acknowledged, its rto at the 120 s maximum;
- TcpExtTCPSpuriousRtxHostQueues rose 41 → 62 while TcpRetransSegs rose only 1 → 3: each retransmission timer found the
  original segment "still in the host queue" and did not resend it;
- the qdisc held nothing and the transmit queue's byte queue limit showed 0 bytes in flight;
- the laptop had nothing unsent and was waiting for that reply.

So the Pi 5's driver (macb, hardware TSO on) reported the bytes sent while it still held the packet. After tcp_retries2 the
Pi 5's kernel abandoned the connection, which is sp10's 15.7 minutes. The Pi 4 (bcmgenet) runs with TSO off and never hung.
The earlier ufw hypothesis is withdrawn: ufw is disabled (ENABLED=no), and the reply never left the Pi 5.

**sp10d (below) showed it is not MTP-specific:** its three hangs had the Pi 5 as a middle worker, one of them in a serve
with no speculation, and SpuriousRtxHostQueues rose again, 62 → 103 (the Pi 4's stayed 0).

**The fix (the owner's, on the board):** `sudo ethtool -K eth0 tso off` at 22:00. At 22:21 it was made persistent with
`sudo nmcli connection modify hs-wired ethtool.feature-tso off` (the Pi 5 runs NetworkManager; `hs-wired` is eth0's
connection).

**The check** (`check_pi5_tso_20261103.py`; `SERVE_MTP_PI5=1`): three rounds of the serve that had hung on its first request
every time before the fix.

| round | warm-up | scan (12 short prompts) | 4 × 128-token requests | Pi 5 SpuriousRtxHostQueues |
|---|---|---|---|---|
| 1 | answered | answered | answered | 103 → 103 |
| 2 | answered | answered | answered | 103 → 103 |
| 3 | answered | answered | answered | 103 → 103 |

- **3 of 3 rounds hold.** So the stall was the Pi 5's hardware TSO.
- **The guard:** serve_pool.sh asks the Pi 5 (`ethtool -k eth0`). It allows MTP with the Pi 5 in the plan only when TSO reads
  off, and otherwise refuses, naming the folder (test extended; testbed 268 OK).

# 2026-11-03 — jmoe: the Jetson prefilling MoE blocks (item 2): pp512 within 6 %, but a block moved to it saves 112 ms, not 717

`jmoe_predictions_20261103.json` (registered 19:56, `029ee05a…`; `testbed/results/jmoe_20261103_195639`; scored by
`score_jmoe_20261103.py` into `score_prefill.txt`).

**Why:** q9b's request objective rested on the Jetson's GPU prefilling five MoE blocks. The planner priced that from
TinyLlama's dense Vulkan rows, and q9b's first-token gain came in at 0.90 s of the predicted 3.50.

**The runs:** lc4's bench on Qwen3-30B-A3B Q6_K, the laptop's 36 blocks fixed and the Jetson taking k of the Pi 5's 12.
llama-bench pp512 + tg16, -r 3; J0 run first and last (drift 0.52 %).

| arm | Jetson's blocks | pp512 (ms) | predicted | error | decode (ms / token) |
|---|---:|---:|---:|---:|---:|
| J0 | 0 | 32,260.2 | 34,268.8 | −5.86 % | 140.42 |
| J1 | 1 | 32,021.5 | 33,642.9 | −4.82 % | 140.65 |
| J3 | 3 | 31,798.2 | 32,208.3 | −1.27 % | 149.93 |
| J5 | 5 | 31,574.1 | 30,773.7 | +2.60 % | 155.27 |

- **JM-PRE holds:** every pp512 within 6 % of the planner.
- **JM-SHIFT fails:** J0 → J5 saved 686.1 ms, against 3,495.1 predicted.
- **The fit** over J1 / J3 / J5: a block moved from the Pi 5 to the Jetson saves 111.9 ms of a 512-token prefill, where the
  planner priced 717. Adding the Jetson as a stage costs nothing (the fitted step is −126.7 ms).
- **The correction:** with the Pi 5's price kept, the Jetson's prefill per parameter on this model is **2.3692×** its row.
  It is `predict_pool.PREFILL_MULT` (`{"qwen3moe_q6k": {"jetson": 2.3692}}`) via `apply_prefill_mult`, opt-in as
  `prefill_mult=` on plan_host_choice and plan_fixed (testbed 268 OK).
- **After the fact:**
  - on q9b's snapshot, the corrected planner predicts R 0.72 % slower (measured 0.56 % slower), and the request objective
    then picks the decode plan;
  - on q9's it predicts −0.10 % against q9's measured +4.4 %. Each arm is within 3.2 %, but the gain is smaller than that
    error.
  - Rescanned with the correction, the request objective moves plans only for predicted gains of 0.5–2.2 % (the 30B) and
    2.1–6.3 % (the dense 32B models). That is within the per-arm error, so no re-test was run.
- Reported, not tested: decode slows as the Jetson takes blocks, 140.4 → 155.3 ms a token from J0 to J5.

# 2026-11-03 — per-node batch curves (item 1), and sp10d: the per-node model holds on P6; INCOMPLETE (three hangs on the Pi 5)

**The curves:** the batch of m tokens against one, on Qwen3.5-9B Q4_K_M, llama-bench pp1–pp8, -r 3, sha256 checked on every
node.
- Host role: `explcurves_20261103_194648`, each node alone.
- Worker role: `worker_role_20261103_195154`, `bench_worker_role.sh` through a loopback RPC server (port 50053), which
  doesn't repack the weights.

| node, role | m 2 | m 3 | m 4 | m 6 | m 8 |
|---|---:|---:|---:|---:|---:|
| laptop, host | 1.46 | 1.89 | **1.79** | 2.76 | 3.19 |
| Pi 5, host | 1.65 | 2.26 | **1.15** | 2.29 | 1.83 |
| Pi 4, host | 1.09 | 1.47 | 1.86 | 2.63 | 3.39 |
| Pi 5, worker | 1.08 | 1.36 | 1.71 | 2.29 | 3.02 |
| Pi 4, worker | 1.13 | 1.51 | 1.89 | 2.76 | 3.59 |

- The Pi 5 as a host runs repacked kernels that work in multiples of 4, so its host curve dips at 4 and 8. As an RPC worker
  it has no repack. A board's share of a pooled plan follows its worker curve.
- **The model** (`spec_model_20261103.py`, per-node functions): each node's busy share from the planner, times its own curve.
  The one-token offset on the pooled plans is 1.02 (P1 1.0201, P4 1.0169), against the laptop alone's 0.955.
- **Checked on P1 and P4 (after the fact):** R within 0.5–3 % at m 2–3 and ~6 % low at m 4. With B's offset it was 7–14 % low.

**sp10d** (`sp10d_predictions_20261103.json`, registered 20:32, `30968ac0…`; `testbed/results/sp10d_20261103_203213`,
20:32–21:43). sp10's hypotheses on two three-node 9B plans none of it was fitted on, the intercept P4's 40.0 ms:
- P5: laptop 20 + Pi 5 7 + Pi 4 6;
- P6: laptop 12 + Pi 5 11 + Pi 4 10.

**INCOMPLETE on every hypothesis.** Three requests hung, each ended by the runner's 600 s timeout, all with the Pi 5 as a
middle worker:
- P5's no-speculation serve, on its scan's first batch of 8, so P5 has no baseline;
- P5's MTP 3, on copy request 1;
- P6's MTP 3, on free request 0.

The Pi 5's stall counters rose (SpuriousRtxHostQueues 62 → 103), so the guard then refused MTP with the Pi 5 anywhere in the
plan. The section above has the cause and the fix.

**What completed, P6** (decode step 637.6 ms):

| | m 2 | m 3 | m 4 | MTP 1 free / copy | MTP 2 free / copy |
|---|---:|---:|---:|---|---|
| predicted | 1.204 | 1.590 | 1.855 | 1.156 / 1.278 | 1.047 / 1.270 |
| measured | 1.204 | 1.601 | 1.974 | 1.136× / 1.295× | 1.010× / 1.270× |
| error | +0.02 % | +0.66 % | **+6.43 %** | −1.8 / +1.3 % | −3.5 / +0.0 % |

- Every completed speedup within 3.5 %. The setting the model picks (MTP 1) was the best measured on both prompts.
- So the per-node model holds where it could be measured. A complete test needed the Pi 5's fix: sp10e, below.
- **Deviation:** at about 20:33–20:34 a short Python string edit ran on the laptop (well under a second of CPU) during P5's
  scan. Plain file edits only after it.

# 2026-11-03 — a depth law that bends (item 3), and dl1: the slope error cut about five-fold; right on the dense 14B, high on the MoE

**The law** (`attention_bend_20261103.py` → `testbed/dataset/attention_cost_20261103.json`; `predict_pool.ATTENTION_V5`,
opt-in). On the laptop and the Pi 5 the per-dimension price bends with depth, law(d) = a (1 + d / d1) ns per query-head
dimension. It is fitted by least squares on lc4charb's TinyLlama depth curves alone:
- the laptop: a 0.211, d1 5,109 (the straight V4 law was 0.547), residuals within 0.4 %;
- the Pi 5: a 0.176, d1 4,013, residuals within 3.1 %.

The straight V4 laws equal the bent ones at about 8,192 tokens. The Pi 4 (no curve) and the Jetson (flat: its flash-attention
table) are unchanged. `planner_inputs` applies the bend.

**After the fact, on data it wasn't fitted on:**
- v4d's depth term at 4,096: +6.1 % (A) and +6.6 % (B), where V4 was +52.6 / +51.8 %;
- q9b's time per token at ~544: +4.9 % (D) and +0.5 % (R), where V4 was +26.1 / +17.4 % and depth 0 −9.9 / −12.1 %;
- priced at 4,096, the planner now picks the laptop-alone plan v4d measured faster (V4 picked laptop 40 + Jetson 8).

**dl1** (`dl1_predictions_20261103.json`, registered 21:47, `8fa720c3…`; `testbed/results/dl1_20261103_214735`, 21:47–22:07):
llama-bench `-p 0 -n 32 -r 3` at fresh depths, on the laptop:
- A: Qwen3-30B-A3B Q3_K_M on the laptop alone;
- B: the same, laptop 40 + Jetson 8 (Vulkan);
- C: Qwen3-14B Q4_K_M (dense) on the laptop alone, to 1,024 only (the 14B past 1,024 is a known dead end).

The Pi 5's bench was dropped before registering: about 45 minutes, and its stall was not yet fixed.

| bench | ms a token at depth | measured slope (ms / 1,000) | bent law (V5) | straight law (V4) |
|---|---|---:|---:|---:|
| A | 99.3 / 137.2 / 185.6 (0 / 1,024 / 2,048) | 42.2 | 58.1 (−27.5 %) | 107.5 |
| B | 140.0 / 179.4 / 222.4 (0 / 1,024 / 2,048) | 40.2 | 50.5 (−20.3 %) | 91.6 |
| C | 399.9 / 458.9 / 452.1 (0 / 512 / 1,024) | 51.0 | 51.9 (−1.6 %) | 112.0 |

- **DL-SLOPE fails** (15 %): A and B are outside, C within 1.6 %.
- **DL-PICK holds:** A faster than B at 2,048 (185.6 against 222.4 ms; bent law 218.9 against 248.5).
- **DL-ACC fails narrowly** (15 %): A at 2,048 is −15.2 %; every other point is within 10.5 %.
- C's 512 point sits above its 1,024 point, so its slope is a fit through three points that don't line up.
- **Reading:**
  - The bent law cuts the slope error about five-fold: the straight law was +119 to +155 % over the measured slope, the bent
    one +2 to +38 %.
  - It is right on the dense model but still over-prices the MoE at moderate depths. It was fitted on one dense model
    (TinyLlama), so a law per architecture comes next.
  - It stays an option (ATTENTION_V5); serving prices depth only when asked.

# 2026-11-03 — sp10e: sp10d again with the Pi 5's TSO off: complete, no hang; three tests hold, two fail on P5's free text

`sp10e_predictions_20261103.json` (registered 22:21, `e8ce52cd…`): sp10d's registration copied unchanged (its sha256
`30968ac0…`, so nothing re-derived). `testbed/results/sp10e_20261103_222119`, 22:21–23:14, the Pi 5's TSO off (persistent since
22:21).
- **Every arm on both plans completed, and no request hung.** The Pi 5 was a middle worker in both plans, through eight
  serves under MTP 1–3 and none, and its stall counters did not move (SpuriousRtxHostQueues 103, AbortOnTimeout 2, read at
  23:15). P5's scan, which hung sp10d at its batch of 8, completed.
- Thermal: the Pi 5 at 50.5–57.1 °C, throttled=0x0. The Pi 4 at 54.5–66.7 °C, throttled=0x80000 on every line: the sticky
  "soft limit has occurred" bit from 14:28's scan at 80.3 °C; its "now" bits stayed clear (as in sp10d).

| plan | decode step | R at m 2 / 3 / 4 (err) | draft (pred.) | MTP 1 free / copy (pred.) | MTP 2 free / copy (pred.) | MTP 3 free / copy (pred.) |
|---|---:|---|---:|---|---|---|
| P5: laptop 20 + Pi 5 7 + Pi 4 6 | 491.8 ms | 1.287 / 1.712 / 2.002 (+2.0 / +2.3 / +9.0 %) | 120.6 ms (148.4) | 1.058× / 1.157× (1.050 / 1.161) | **1.077×** / 1.129× (0.936 / 1.135) | **0.815×** / 1.084× (0.908 / 1.122) |
| P6: laptop 12 + Pi 5 11 + Pi 4 10 | 639.9 ms | 1.199 / 1.621 / 1.994 (−0.4 / +1.9 / +7.5 %) | 143.8 ms (146.4) | 1.137× / 1.292× (1.156 / 1.278) | 1.000× / 1.269× (1.047 / 1.270) | 0.964× / 1.176× (0.986 / 1.218) |

- **SP-CHECK holds** (10 %): the batch cost within 9.0 % (P5 at m 4); m 2–3 within 2.3 %.
- **SP-DRAFT holds** (20 %): P5's draft 18.7 % under the prediction, P6's 1.8 %.
- **SP-BEST holds** (3 %): the model picks MTP 1 for both prompts on both plans. It was the best measured on three of the
  four, and on P5's free text 1.8 % behind MTP 2.
- **SP-SPEEDUP fails** (7 %): 10 of 12 arms within 4.5 %, but P5's free text under MTP 2 (+15.1 %) and MTP 3 (−10.3 %).
- **SP-SIGN fails:** 9 of the 10 arms predicted ≥ 5 % from even landed on their side. P5's MTP 2 on free text was predicted a
  loss (0.936) and won 7.7 %.
- **Both misses are in the acceptance, not the cost.** The model carries each setting's tokens per pass (E) over from the
  exploration's 9B. On P5's free text MTP 2 accepted 2.54 a pass (2.23 carried) and MTP 3 2.33 (2.59). At the measured E the
  model gives 1.067 and 0.816, against 1.077 and 0.815 measured: within 1 %. Why the acceptance on free text moved on this
  plan is not established. Every other arm's E was within 3 % of the carried one, except P5's MTP 1 on free text (1.81
  against 1.74).
- **Item 1, where it stands:** with per-node batch curves the model prices the batch cost within 9 % and the draft within
  19 % on two plans it was not fitted on, and the speed-up within 4.5 % wherever the acceptance matches the borrowed one.
  MTP 1, the setting it picks, was always within 2 % of the best. On these pooled 9B plans MTP 1 gained 6–14 % on free text
  and 16–29 % on copying.
- **Deviation:** at about 22:24–22:25 a short Python string edit ran on the laptop (under a second of CPU), during P5's
  no-speculation free requests. Plain file edits only after it.
- Serving still keeps speculation off by default; MTP for pooled plans is a serving decision now open (the Pi 5 no longer
  hangs).
- **Decided 4 Nov (the owner: "Turn MTP 1 on"):** a person serving a pooled plan of Qwen3.5-9B or Qwen3.6-35B-A3B from a
  terminal gets MTP 1 (`serve_defaults.sh` mtp_mode, `MTP_MODELS`; `SERVE_MTP=off|on`). Not on a one-board plan, not for
  Qwen3.8-27B (it has the head, never measured with it), and not when the Pi 5 is in the plan with its TSO on (served
  without MTP then). Harnesses keep their registered settings.

# 2026-11-04 — the 4B's calibration (item 4): its hand-off thrashes the 4 GB Jetson from 1,024 tokens, so ph3 tests L and J

`testbed/results/phase_calib_20261104_004942.log` (`PHASE_MODEL=qwen3_4b phase_calib_20261028.py run`, 00:49–02:32; not a
registered test). Qwen3-4B Q4_K_M (sha256 7485fe6f…) on the laptop and on the Jetson (Vulkan, 2,560 cells). The first attempt
(3 Nov, `phase_calib_20261103_231753.log`) ended when the owner powered the Jetson off. Zen open, allowed by the owner.

**L and J, complete** (16 rows, two repetitions at 256 / 512 / 1,024 / 2,048 tokens), fitted into
`testbed/dataset/phase_costs_qwen3_4b_20261103.json`:

| node | prefill, ms (n prompt tokens) | max residual | decode, ms a token (d context) | max residual |
|---|---|---:|---|---:|
| laptop | 1,979.7 + 27.75 n + 0.00818 n² | 4.4 % | 108.3 + 0.0553 d | 7.6 % |
| Jetson | 6,970.3 + 43.04 n + 0.00544 n² | 7.6 % | 336.9 + 0.0198 d | 1.1 % |

**H, the hand-off, thrashed the Jetson's memory:**

| request | state | save | transfer (with ssh) | restore | decode, 17 tokens |
|---|---:|---:|---:|---:|---:|
| r1, n = 512 | 77.3 MB | 0.04 s | 1.7 s | 0.29 s | 5.5 s |
| r1, n = 1,024 | 152.8 MB | 0.08 s | 3.6 s | **213.9 s** | **3,208 s** |
| r1 n = 2,048; r2 n = 512, 1,024 | — | — | failed: "receiver on jetson did not listen ... banner exchange" | — | — |
| r2, n = 2,048 | 303.8 MB | 0.14 s | 46.6 s | never returned | — |

- **Why:** the receiver's slot file sits in /dev/shm (RAM), and the restore reads it into a second copy, on top of the 4B's
  weights, its 2,560 cells and its compute buffers on the 4 GB board. Its swap is zram (compressed RAM), so the board
  thrashed without writing to its SD card. ph1 and ph2's Qwen3-0.6B never came near the limit.
- The Jetson stopped answering ssh and HTTP (ping did); the owner power-cycled it at ~02:38.
- **The hand-off is not fitted** (`fit` skips it when any hand-off request of the run failed). It is infeasible on this
  board at these lengths. The phase planner has no memory check for H yet: it needs the receiver's free memory against
  twice the state, plus margin.
- **So ph3 tests L and J only** (registered 02:39:52; the changes and the evidence are in its registration's
  `handoff_excluded` and in the journal).

# 2026-11-04 — ph3: the 0.6B's power rows on Qwen3-4B: picks and times hold, the Jetson's energy misses on the laptop's idle

`ph3_predictions_20261103.json` (registered 02:39:52, `b352fab7…`); `testbed/results/ph3_20261104_024007` (02:40–03:10, complete;
code changed since registration: none). Qwen3-4B Q4_K_M, prompts of 768 and 1,536 tokens (held out), 256 generated, plans L
and J (no H: above), two repetitions interleaved, 60 s between requests, the strip at 5 Hz on all four outlets. Times from the
4B's own calibration; energies from the 0.6B's power rows (ph2's), and for comparison the TinyLlama table. Zen open
(QUIET_ALLOW_APPS=zen, in the conditions).

| arm | total, s (pred.) | error | energy, J (rows / table) | error: rows / table | first token, s (pred.) |
|---|---:|---:|---|---|---:|
| 768 L | 68.4 (69.0) | −0.96 % | 2,908.9 (3,005.7 / 2,727.5) | −3.22 % / +6.65 % | 27.7 (28.6) |
| 768 J | 136.6 (134.3) | +1.71 % | 3,738.7 (4,224.7 / 4,015.8) | −11.50 % / −6.90 % | 46.2 (43.9) |
| 1,536 L | 114.9 (115.8) | −0.78 % | 4,672.7 (4,862.7 / 4,529.3) | −3.91 % / +3.17 % | 63.2 (64.5) |
| 1,536 J | 180.9 (181.0) | −0.05 % | 5,078.3 (5,661.1 / 5,347.9) | −10.29 % / −5.04 % | 87.3 (86.6) |

- **3 of 6 hold:** PH-PICK-E and PH-PICK-T (L at both lengths), PH-TIME (within 1.7 %). PH-ENERGY fails (J −11.5 and −10.3 %
  against 10 %), PH-REPEAT fails (L's two runs 4.9 and 4.4 % apart; J's 1.3 and 0.5 %), PH-BETTER fails (better than the table
  on L, worse on J).
- **Found after the data:** the Jetson's draw was the same in every J request (prefill 8.1–8.4 W, decode 10.9–11.0 W), but the
  laptop idled at 9.7–9.9 W in the first two J requests (02:47–02:53; most likely its display off) and 12.2–18.4 W in the last
  two (03:00–03:07; the owner's message and my reply came during the run). The rows assume 17.1 W. So each J pair differs 17
  and 26 % in energy at the same time: the first two came 18–21 % under the prediction, the last two within 0.05 and 4.6 %.
  The rows' structure carries to the 4B; the laptop's idle state, which e2 found moving within a session, does not.
- **The 4B's own rows** (`testbed/dataset/phase_power_qwen3_4b_20261103.json`): laptop idle 14.3 W, prefill 37.3 W for the
  first 40 s (the fit's upper bound) then 24.3 W, decode 32.0 W; Jetson prefill 8.34 W, decode 10.93 W (the 0.6B's: 8.26,
  9.39); no hand-off rows.
- Deviation: the owner's message and my reply during the run.

# 2026-11-04 — the MTP 1 default served live (unregistered): it works as built; on one prompt it was 4–6 % slower, and its text differs

`testbed/results/mtp_default_check_20261104_031253` (`summary.json`). Qwen3.5-9B Q4_K_M on sp10e's P5 placement (laptop 20 +
Pi 5 7 + Pi 4 6, head on the host), through `serve_pool.sh` with SERVE_MTP=on and no draft or speculation arguments, then the
same with SERVE_MTP=off. One free-text prompt, 64 tokens, greedy, twice each; Zen open.
- The default's path worked: MTP 1 on the command line, the Pi 5's TSO read off, the split check matched with the MTP block
  on the Pi 4.
- MTP 1: 514.5 and 503.0 ms a token, 23 of 39 drafts accepted (59 %); without: 484.4 and 483.3 ms. **4–6 % slower on this
  prompt**, where sp10e's registered free-text prompt gained 5.8 % on this plan at 1.81 tokens a pass. The gain follows the
  prompt's acceptance.
- **The text is not bit-identical:** each setting repeated its own text exactly, but the two diverged after 78 characters.
  The batched check computes slightly different numbers from one-token decoding on these CPUs.
- Not a registered test; the default stays as the owner chose, with this reported to them.

# 2026-11-04 — dl2: the laptop's depth cost per attention shape: its attention op, timed with no model loaded, held on three new models (0.6–4.9 %); three other laws did not

**Why:** V5, the planner's CPU depth law (a (1 + d / d1) ns per query-head dimension, fitted on TinyLlama alone), priced the
30B MoE's slope 20–28 % high (dl1). The user: "do all four follow-up studies".

**The instrument** (`testbed/results/dl2fa_20261104_043524`, 04:35–04:51):
- `fa_bench_cpu_20261104.cpp` is fa_bench's graph on the CPU backend: -t 4, a 64 MiB read between graphs (the caches as a
  decoded token finds them), the median of 15 graphs.
- It ran every attention shape the cluster serves at 256–8,192 cells, in two passes (the second in reverse order), with the
  CPU at 56–78 °C.
- The table, `testbed/dataset/attention_cpu_20261104.json`, is the mean of the two passes. The passes differ by up to 25–44 %
  at single cells: an isolated op on a laptop CPU is noisy.

Per attention op at 2,048 cells (µs), shape = head size × query heads × KV heads:

| shape | 64×32×4 | 128×16×8 | 128×32×4 | 128×32×8 | 128×40×8 | 128×64×8 | 256×16×4 | 256×24×4 | 256×16×2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model | TinyLlama | Qwen3-0.6B | Qwen3-30B-A3B | Qwen3-4B | Qwen3-14B | Qwen3-32B | Qwen3.5-9B | Qwen3.8-27B | Qwen3.6-35B-A3B |
| µs | 1,284 | 1,713 | 1,886 | 2,963 | 3,672 | 6,108 | 2,637 | 4,136 | 1,692 |

**Checked before registering, on curves the table was not built from:** measured against the table, TinyLlama +3.1 %, the 30B
−18.3 % and the 14B −13.6 % (V5: −0.1, −27.5 and −1.6 %). Not settled, so four laws were registered against each other:
- V5, the planner's law;
- FA, the table;
- HD, a per-head cost beside the head size, fitted on TinyLlama and the 30B;
- KV3, three terms per layer (query heads, query dims, KV dims), fitted exactly on TinyLlama, the 30B and the 14B.

**dl2** (`dl2_predictions_20261104.json`, registered 04:53, `33cb936a…`; `testbed/results/dl2_20261104_045326`, 04:53–05:20):
- llama-bench -ngl 0 -t 4 -p 0 -n 32 -d 0,1024,2048,4096 -r 3 on the laptop alone, by lc4;
- its gate now takes `QUIET_ALLOW_APPS` (Zen open, as the owner allowed);
- three models whose attention shapes no depth curve had been measured on.

| bench | ms a token at 0 / 1,024 / 2,048 / 4,096 | measured slope | V5 | **FA** | HD | KV3 |
|---|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B Q8_0 (128×16×8, 28 layers) | 33.6 / 57.5 / 89.4 / 152.8 | 29.50 | 22.15 (+33.1 %) | **29.11 (+1.3 %)** | 16.07 (+83.6 %) | 44.71 (−34.0 %) |
| Qwen3-4B Q4_K_M (128×32×8, 36) | 126.7 / 179.7 / 259.7 / 430.7 | 75.62 | 56.97 (+32.7 %) | **75.16 (+0.6 %)** | 41.32 (+83.0 %) | 65.87 (+14.8 %) |
| Qwen3.5-9B Q4_K_M (256×16×4, 8 attention layers of 32) | 280.2 / 292.3 / 293.7 / 333.2 | 12.69 | 12.66 (+0.2 %) | **12.09 (+4.9 %)** | 7.44 (+70.5 %) | 12.90 (−1.7 %) |

Slopes are ms per 1,000 tokens of depth; each error is measured against the law (10 % registered).
- **FA holds on all three; V5 and KV3 on one each; HD on none.**
- The laptop's depth cost follows the whole attention shape: query heads, KV heads and head size.
  - The 0.6B (16 query heads, 8 KV heads) costs a layer about what the 30B (32 query, 4 KV) does.
  - Doubling the KV heads at 32 query heads (the 30B's shape to the 4B's) adds about 57 %.
  - One price per query dimension cannot carry that, nor did two- or three-term fits on three curves.
- The isolated op priced the real decode's slope within 5 %, though its two passes differ by up to 44 % at single cells: a
  slope integrates many cells.
- **In the planner as ATTENTION_V6, opt-in.** The laptop's law is "fa_curve" (`predict_pool.attn_ms` takes the depth: a token
  at depth d attends over 256 × ceil((d + 1) / 256) cells). The table is `testbed/dataset/attention_cost_20261104.json`, from
  `dl2_fa_cpu_20261104.py v6`; the other boards are as in V5. Serving still prices depth only when asked (PLAN_DEPTH). Tests:
  `test_attention_v6.py`.
- **Not done:**
  - the Pis' CPUs (fa_bench_cpu would build on each board against its pinned tree);
  - the 30B's and the 14B's 3 Nov curves sit 14–18 % under the table, and dl2 did not re-measure them.

# 2026-11-04 — hm1: a memory check before the hand-off: the planner and its guard keep the Jetson out of a thrash; 3 of 4 hold (near the line it swapped 42 MiB)

**Why:** the 4B's hand-off thrashed the 4 GB Jetson from 1,024 tokens (the calibration at 00:49, and the board needed a power
cycle). The receiver keeps the state in its tmpfs, which is RAM, through the whole decode.

**What was built:**
- **The check** (`phase_plan_20261028.handoff_fits`, `choose(receiver_avail=...)`): H is planned only when its state plus
  320 MiB fits the receiver's MemAvailable with its server loaded.
- **The guard** (`phase_serve_20261028.run(guard_headroom=...)`): once the file lands, the Jetson's MemAvailable is read.
  - Below the headroom, the file is deleted and the laptop, which still holds the prompt in its slot, decodes the rest
    ("H->L").
  - Otherwise the restore goes ahead and the file is deleted straight after.
- phase_serve's `ask` uses both.

**The headroom, from the evidence** (`hmem_20261104_052109` and `_052145`: the Jetson's MemAvailable after a warm-up):
- the 0.6B at 9,216 cells leaves 1,691 MiB, and the 4B at 2,560 cells 401 MiB;
- the 4B's hand-off thrashed with 255 MiB left after its 146 MiB state landed, and ran clean with 327 MiB left;
- the 0.6B's ran clean with 794 MiB left.

So 320 MiB, near the clean side. All five past hand-offs agree with it: the 4B's clean one by 7 MiB, and its thrash is refused
by 65 MiB. By the same rule, the 4B's server at 2,560 cells leaves room for about 570 prompt tokens of state. Its hand-off costs were never fitted (ph3), so the phase planner still does not offer it.

**hm1** (`hm1_predictions_20261104.json`, registered 05:25, `b7b7e5a0…`; `testbed/results/hm1_20261104_052536`, 05:25–05:32):
- Qwen3-0.6B at 16,384 cells, a context the Jetson had not served;
- predicted MemAvailable 907 MiB (the 9,216-cell figure less the KV's growth), so the check's limit is 5,369 prompt tokens;
- 17 tokens per request.

| case | prompt | live check | left after landing | what happened | restore | swap used |
|---|---:|---|---:|---|---:|---:|
| a1 | 2,684 | allow (+276 MiB) | 596 MiB | restored; the Jetson decoded at 119.5 ms a token (its law: 120.6) | 0.29 s | 0 |
| a2 | 4,563 | allow (+65 MiB) | 451 MiB | restored; decoded at 134.6 ms (its law: 135.8) | 1.46 s | **41.6 MiB** |
| g1 | 5,637 | refuse (−34 MiB) | 302 MiB | the guard refused; the laptop decoded, with tokens equal to L's | — | 0 |
| g2 | 1,036, headroom forced to 8 GiB | allow | 796.5 MiB | the guard refused (forced); tokens equal to L's | — | 0 |

- **HM-READY holds** (895 MiB against 907 predicted). **HM-AGREE holds:** every live verdict was the registered one.
  **HM-GUARD holds.**
- **HM-ALLOW fails on a2's swap:** 41.6 MiB against the < 32 MiB registered. Its restore (1.46 s) and decode (within 1 % of the
  Jetson's law) were normal, and memory came back to 938 MiB once the file was deleted.
- **Reading:**
  - 320 MiB keeps the board out of a thrash, but with about 450 MiB left after landing the Jetson still swaps a little (zram)
    during the restore; with 596 MiB left it swapped nothing.
  - The headroom stays at 320 MiB: a refusal costs the hand-off's saving, and this swap cost nothing measurable. A margin
    with no swap at all would be about 600 MiB.

# 2026-11-04 — sp11: the MTP 1 default across 12 prompts: faster on every one; the misses are one copy prompt and one text divergence; 4 of 6 hold

`sp11_predictions_20261104.json` (registered 05:35, `e791de91…`); `testbed/results/sp11_20261104_053515` (05:35–06:50, complete;
scored by `score_sp11_20261104.py` at 06:50, code changed since registration: none). Qwen3.5-9B Q4_K_M, 128 tokens, greedy, on
sp10e's two plans: Q5 = laptop 20 + Pi 5 7 + Pi 4 6, Q6 = laptop 12 + Pi 5 11 + Pi 4 10 (head on the host, ctx 4,096), each
prompt plain and with MTP 1; Q5's plain texts once more with every token's top-two probabilities (not timed). The 35B case was
dropped before registering: on the 05:33 snapshot no subset of the cluster holds it with the laptop's 8 GiB reserve. Zen open;
the laptop's quietness not gated (speed ratios within one plan's serves).

| prompt | Q5: none / MTP 1 ms a token | Q5 speedup (model at E; E) | Q6: none / MTP 1 ms | Q6 speedup (model; E) |
|---|---|---|---|---|
| free | 502.0 / 467.2 | 1.075× (1.095; 1.81) | 631.9 / 540.4 | 1.169× (1.141; 1.72) |
| copy | 494.9 / 402.8 | 1.229× (1.161; 1.92) | 632.1 / 483.4 | 1.308× (1.279; 1.92) |
| story | 479.4 / 408.1 | 1.175× (1.144; 1.90) | 631.0 / 485.4 | 1.300× (1.260; 1.90) |
| email | 471.7 / 435.0 | 1.084× (1.080; 1.79) | 631.1 / 493.0 | 1.280× (1.241; 1.87) |
| code | 472.1 / 388.1 | 1.216× (1.198; 1.98) | 630.6 / 479.1 | 1.316× (1.279; 1.92) |
| tips | 473.5 / 424.5 | 1.115× (1.095; 1.81) | 630.4 / 516.9 | 1.220× (1.189; 1.79) |
| hash | 471.0 / 408.0 | 1.155× (1.144; 1.90) | 632.0 / 494.9 | 1.277× (1.241; 1.87) |
| summary | 476.2 / 402.2 | 1.184× (1.179; 1.95) | 631.5 / 476.7 | 1.325× (1.298; 1.95) |
| thai | 480.1 / 405.9 | 1.183× (1.144; 1.90) | 630.3 / 486.4 | 1.296× (1.260; 1.90) |
| math | 472.1 / 393.9 | 1.199× (1.179; 1.95) | 630.6 / 471.4 | 1.338× (1.298; 1.95) |
| compare | 472.6 / 409.6 | 1.154× (1.127; 1.87) | 630.7 / 500.8 | 1.259× (1.223; 1.84) |
| chat | 472.5 / 429.3 | 1.101× (1.080; 1.79) | 634.3 / 524.3 | 1.210× (1.172; 1.76) |

- **MT-COST fails on one arm of 24:** Q5's copy prompt +5.8 % over the model at its measured E (5 % registered). Every other
  arm is within 3.4 % (Q5, mean |error| 2.1 %) and 3.2 % (Q6: all +2.0 to +3.2 %, mean 2.8 %).
- **MT-SIGN holds, 24 of 24:** MTP 1 was faster everywhere, and every prompt's E exceeded the break-even E* (1.657 on Q5,
  1.505 on Q6). **MT-MEAN holds:** geometric means 1.155 and 1.274. **MT-WORST holds:** the slowest 1.075 and 1.169 (free).
- **TX-TIE fails on one of 19 divergences:** 18 sit at near-ties in Q5's plain text (top-two gaps 0.0025–0.098); "compare"
  under Q5's MTP 1 left it at token 68, where the gap was 0.127 (0.521 against 0.394).
- **TX-E holds, 12 of 12:** where Q5's and Q6's MTP 1 acceptance differs by more than 3 % (free 5.7 %, email 4.2 %, code
  3.1 %) their texts differ.
- **Reading:** on these plans the default gained on every prompt tried, as the model's sign predicts. The 4 Nov 03:12 live
  check's prompt accepted 1.59 tokens a pass, under Q5's E*, and lost 4–6 %, as the model would say. P5's free-text
  acceptance moved in sp10e because its text moved at a near-tie (character 93), not because the cost did.
- Deviation: at about 05:36 a short Python string edit ran on the laptop during Q5's first plain request.

# 2026-11-04 — dl3fa: the Pis' attention op per shape, timed with no model loaded (dl2fa's method on each Pi)

**Why:** the owner, after dl2: "Time the Pis' CPUs per attention shape, so the per-shape depth table (ATTENTION_V6) covers more
than the laptop". The Pis are priced by V5's per-dimension laws: the Pi 5's bent law fitted on its TinyLlama curve (lc4charb),
the Pi 4's straight law from Qwen3-14B on the Pi 4 + Jetson (10 Oct).

**The instrument:** `fa_bench_cpu_20261104.cpp` (source sha256 `3aa37579…`) built on each Pi against its pinned tree
(`b23701f7`; Pi 5 binary `8f7a2880…`, Pi 4 `6efb1259…`), driven by `dl3_fa_pi_20261104.py run <board>`: dl2fa's shapes,
cache lengths (256–8,192), -t 4, the median of 15 graphs, two passes; a cold gate; the temperature and get_throttled logged
around every shape. Both boards at once from 22:40:20 (`testbed/results/dl3fa_pi5_20261104_224020`, to 23:05;
`dl3fa_pi4_20261104_224020`, to 23:36). 18 of 18 shape-passes each, rc 0, throttled 0x0 throughout (peaks 65.9 and 68.6 °C).
The table is `testbed/dataset/attention_cpu_pi_20261104.json`.

Per attention op (µs) at 1,024 / 2,048 / 4,096 / 8,192 cells:

| shape (model) | Pi 5 | Pi 4 |
|---|---|---|
| 64×32×4 (TinyLlama) | 497 / 1,101 / 2,937 / 8,926 | 1,971 / 5,779 / 13,859 / 28,881 |
| 128×16×8 (Qwen3-0.6B) | 1,067 / 3,265 / 8,393 / 18,360 | 3,269 / 7,413 / 15,732 / 31,293 |
| 128×32×8 (Qwen3-4B) | 1,730 / 6,101 / 16,176 / 35,928 | 6,361 / 14,739 / 31,276 / 62,387 |
| 128×40×8 (Qwen3-14B) | 2,039 / 7,433 / 20,684 / 45,746 | 7,867 / 18,606 / 38,901 / 76,749 |
| 128×32×4 (Qwen3-30B-A3B) | 1,085 / 4,066 / 13,652 / 32,239 | 7,249 / 16,892 / 35,137 / 70,831 |
| 128×64×8 (Qwen3-32B) | 3,042 / 11,800 / 33,173 / 74,021 | 12,563 / 29,417 / 61,726 / 124,577 |
| 256×16×4 (Qwen3.5-9B) | 2,056 / 6,909 / 17,170 / 36,318 | 6,654 / 15,283 / 32,032 / 64,720 |
| 256×24×4 (Qwen3.8-27B) | 2,788 / 10,146 / 25,592 / 54,229 | 9,754 / 22,872 / 47,738 / 97,764 |
| 256×16×2 (Qwen3.6-35B-A3B) | 1,035 / 4,121 / 10,763 / 24,645 | 7,390 / 17,106 / 35,004 / 70,381 |

- **Repeatable:** the two passes agree within 5.5 % at every cell on both Pis (the laptop's: up to 44 %).
- **The boards differ in shape:** the Pi 4's op grows about linearly with the cache past 512 cells; the Pi 5's bends up
  (the 0.6B's shape: ×3.1, ×2.6, ×2.2 per doubling from 1,024 cells), which is what V5's bend approximated.
- **The KV heads matter on the Pi 5 too:** at 2,048 cells the 0.6B's shape (8 KV heads) costs 3.0× TinyLlama's (4 KV heads,
  head size 64), where V5 prices them alike (both 2,048 query-head dimensions).
- **Checked before registering:** lc4charb's TinyLlama curve on the Pi 5 (24 Oct): measured slope 24.61 ms per 1,000 tokens,
  the table's 25.31 (−2.8 %), V5's 24.72 (−0.5 %; fitted on that curve). The Pi 4 has no four-node-era depth curve.

# 2026-11-04 — dl3: the Pis' per-shape tables against the planner's law on four benches: 5 of 8 hold; the table right on the dense models

`dl3_predictions_20261104.json` (registered 23:37:22, `193f1ce0…`); `testbed/results/dl3_20261104_233734` (23:37–00:06, lc4;
code changed since registration: none). llama-bench -t 4 -p 0 -n 32 -r 3 on the board alone, in two pairs (the Pis at once):
P5Q06 with P4Q06, then P5Q9 with P4TL. Qwen3-0.6B Q8_0 was copied to the Pi 5 for P5Q06 (sha256 checked on the board).

| bench | ms a token at its depths | measured slope | **FA** (the table) | V5 (the planner's law) |
|---|---|---:|---:|---:|
| P5Q06: Qwen3-0.6B, Pi 5 | 55.7 / 92.6 / 156.2 / 304.3 (0 / 1,024 / 2,048 / 4,096) | 62.14 | **62.50 (−0.6 %)** | 20.78 (+199.0 %) |
| P5Q9: Qwen3.5-9B, Pi 5 | 578.1 / 568.0 / 574.8 / 614.7 (0 / 512 / 1,024 / 2,048) | 19.90 | 32.30 (−38.4 %) | 8.82 (+125.5 %) |
| P4Q06: Qwen3-0.6B, Pi 4 | 189.5 / 298.5 / 427.4 / 660.9 (0 / 1,024 / 2,048 / 4,096) | 115.87 | **113.18 (+2.4 %)** | 108.61 (+6.7 %) |
| P4TL: TinyLlama, Pi 4 | 183.1 / 240.2 / 325.0 / 504.7 (0 / 1,024 / 2,048 / 4,096) | 79.92 | **80.01 (−0.1 %)** | 85.33 (−6.3 %) |

Slopes in ms per 1,000 tokens of depth; each error is measured against the law (10 % registered).
- **FA holds on three of four, V5 on two** (both on the Pi 4, where its straight law was fitted on shapes like these).
- On the Pi 5 the per-dimension law misses the 0.6B threefold: it prices the 0.6B like TinyLlama (both 2,048 query-head
  dimensions), but the 0.6B reads four times the KV per layer. The table, timed per shape, does not miss it.
- **The hybrid 9B on the Pi 5 misses both laws:** its depth cost is small against a 578 ms step and noisy at 0–2,048 (the
  512 point under the 0 point); the table prices it 38 % high, V5 56 % low. Why is not established.
- **Thermal:** the Pi 4 reached 84.2 °C in the second pair; its soft limit fired at 80.3 °C about 52 s into P4TL and was
  active for 379 of its ~823 s (0x80008), most likely over its deeper depths. P4TL matched FA anyway (−0.1 %). The first pair
  ran unthrottled; the Pi 5 stayed at 0x0.
- **In the planner as ATTENTION_V7, opt-in:** V6 with both Pis' laws replaced by their curves (`attention_cost_20261105.json`,
  `dl3_fa_pi_20261104.py v7`); the laptop and the Jetson as in V6. Serving still prices depth only when asked.

# 2026-11-05 — sp12: the MTP 1 default on the 35B across 12 prompts: faster on every one, the model within 3.3 %; 4 of 5 hold

`sp12_predictions_20261104.json` (registered 00:10:03, `8af80447…`; the 00:09 memory snapshot, the laptop at 27,722 MiB with the
owner's apps closed and the 8 GiB reserve kept); `testbed/results/sp12_20261105_001012` (00:10–00:37, `sp11_session_20261104.py`,
scored by `score_sp11_20261104.py`, which since 4 Nov scores TX-E and TX-TIE only when registered; code changed since
registration: none). Qwen3.6-35B-A3B Q6_K on the plan a person gets now (PLANNER_V4): laptop 25 + Jetson 3 + Pi 5 10 + Pi 4 3,
head on the laptop. sp11's 12 prompts, 128 greedy tokens, plain and MTP 1; then the plain texts' top-two probabilities (not
timed). The model: decode 250.1 ms, MTP 1 pass 395.5 ms, E* 1.581.

| prompt | none / MTP 1 (ms a token) | speed-up (model at E) | E |
|---|---|---|---:|
| free | 244.8 / 216.6 | 1.130 (1.164) | 1.841 |
| copy | 246.4 / 206.3 | 1.194 (1.236) | 1.954 |
| story | 246.7 / 244.7 | 1.008 (1.017) | 1.608 |
| email | 247.5 / 208.1 | 1.190 (1.217) | 1.924 |
| code | 245.7 / 206.2 | 1.191 (1.217) | 1.924 |
| tips | 245.5 / 208.0 | 1.180 (1.217) | 1.924 |
| hash | 249.6 / 209.7 | 1.190 (1.199) | 1.896 |
| summary | 249.2 / 218.1 | 1.143 (1.164) | 1.841 |
| thai | 246.1 / 219.0 | 1.123 (1.131) | 1.789 |
| math | 245.6 / 207.2 | 1.186 (1.217) | 1.924 |
| compare | 248.9 / 217.9 | 1.143 (1.147) | 1.814 |
| chat | 246.7 / 219.2 | 1.125 (1.147) | 1.814 |

- **MT-COST, MT-SIGN (11 of 11; story within the dead band), MT-MEAN (geometric mean 1.149) and MT-WORST (slowest 1.008)
  hold.** The model is within 3.33 % on every prompt, each slightly over the measurement.
- **TX-TIE fails:** of eight divergences from the plain text, six sit at near-ties (gaps 0.0011–0.054); chat's (token 12,
  gap 0.119) and code's (token 65, gap 0.197) do not. On this MoE the batched check can turn a choice that is not close.
- **With sp11:** MTP 1 was faster on all 36 prompt-plans tried (two 9B plans and the 35B), its gain following the prompt's
  acceptance against the model's break-even. The default stands as the owner kept it.
- Thermal: the Pi 5 0x0; the Pi 4 0x80000 throughout (dl3's sticky bit, now-bits clear, 55–56 °C). No timeouts.
