# Getting the Jetson into pooled runs *with* its GPU

**Raised 2026-09-20**, after the user confirmed that **pooled GPU runs matter to the
argument**. Desk research only — nothing here has been built. The Jetson is not
physically present yet.

## The constraint, stated exactly

Three things must hold at once, and until now we had only two:

1. **GPU acceleration on the Nano** — the node's whole reason to be in the study.
2. **The same RPC wire protocol as the Pi 5 + Pi 4**, which are on `b23701f7`.
3. **A build that exists**, i.e. someone has actually compiled it on this hardware.

The b5050 decision satisfies (1) and (3) but fails (2): b5050 (2025-04-04) predates
`RPC_CMD_HELLO` (2025-04-18), so a mixed run silently misexecutes. That is why the
Jetson was scoped to standalone characterisation.

**The good news: (2) is not actually blocked by CUDA 10.2.** The ceiling is later than
the b5050 note assumed.

## What the evidence says

| Source | Newest CUDA-10.2 build | Method | RPC? |
|---|---|---|---|
| [kreier/llama.cpp-jetson](https://kreier.github.io/llama.cpp-jetson/) | **b5050** (2025-04) | CUDA arch → 50/61, linker fix in `ggml/CMakeLists.txt`, strip `constexpr`, comment `__builtin_assume` in 3 files, `nv_bfloat16` → `half` | not mentioned |
| [Z841973620/llama.cpp-jetpack4.6](https://github.com/Z841973620/llama.cpp-jetpack4.6) | **b8056** | remove bf16 support | not mentioned |

**b8056 is the important number.** It is far newer than b5050 and therefore *well after*
2025-04-18 — so a CUDA-10.2 build carrying `RPC_CMD_HELLO` demonstrably exists. The
"newest CUDA-10.2-buildable ref" is not April 2025; b5050 was simply the newest one with
a *published step-by-step Nano recipe*, which is a different claim.

That maintainer also notes the CUDA path is no longer actively maintained and
**recommends the Vulkan backend instead**, because JetPack 4.6 / L4T r32.7 ships
**Vulkan 1.2**.

## Three paths, best first

### Path 1 — Vulkan backend at the cluster pin `b23701f7`

Build the Jetson with `-DGGML_VULKAN=ON` instead of CUDA, on the *same revision as the
Pis*. If it works this is strictly the best outcome: **one revision cluster-wide, GPU
acceleration, pooled runs safe, and nothing already measured has to be re-taken.**

Unknowns to settle on the hardware:
- Does the L4T r32.7 Vulkan 1.2 driver satisfy llama.cpp's Vulkan backend requirements
  on **Maxwell sm_53**? Vulkan 1.2 being *present* is not the same as the driver
  exposing what the shaders need.
- **FP16.** The proposal commits to FP16 on this node. CUDA gets it from sm_53's native
  `half2`; Vulkan needs `shaderFloat16` / `VK_KHR_16bit_storage` to be exposed by that
  driver. If it is not, the node is still GPU-accelerated but the "FP16 node" wording in
  the proposal has to change.
- Performance against the CUDA baseline. The published CUDA figure is ~20% faster than
  CPU on this board, so the bar is low but not zero.

### Path 2 — CUDA, with the b8056-class patch set applied to `b23701f7`

Keeps CUDA and FP16 exactly as the proposal specifies. The patch set is known and
bounded (arch limit, linker, `constexpr`, `__builtin_assume`, bf16 → half). The open
question is only whether it applies cleanly to `b23701f7` specifically — 2026 CUDA code
may have drifted further than the bf16 removal covers.

Same outcome as Path 1 if it lands: one revision, GPU, pooled runs.

### Path 3 — move the whole cluster back to b5050

**Only if 1 and 2 both fail.** It costs re-measuring everything taken on `b23701f7` —
the thread sweeps and the 14B pooling result — and it *still* does not remove the risk
below. Worst value for effort; listed for completeness.

## The risk common to all three, and it should be tested first

> **OPEN RISK (already recorded in `cluster.json`'s `_pin_note`): nobody has published a
> Nano build with `-DGGML_RPC=ON`. That is an untested experiment, not a documented
> path.**

Neither recipe above mentions RPC. Every path costs real work and then lands on this
same unverified step, so **de-risk it first**: on whichever ref, build with
`-DGGML_RPC=ON` and confirm `ggml-rpc-server` runs on the Nano *at all*, before
investing in the GPU backend question. It is CPU-side plumbing and should be the
cheapest thing to falsify.

Note the Pi-side work already done transfers: `testbed/scripts/run_14b_pooled.sh`
carries the two traps found on 2026-09-20 — `pkill -x` not `-f`, and never probing the
worker with a bare TCP connect (backlog is 1; a connect-then-close kills it).

## Recommendation

**Do not reverse the cluster pin now.** `b23701f7` is where the measurements are and the
14B pooling works on the Pi 5 + Pi 4 pair. The Jetson is not here yet, so nothing is
blocked by deferring.

When the board arrives, in order:
1. `-DGGML_RPC=ON` smoke test on the Nano — does the worker run at all?
2. Path 1 (Vulkan at `b23701f7`), checking `shaderFloat16` explicitly.
3. Path 2 (CUDA patch set at `b23701f7`) if Vulkan is unusable.
4. Re-open Path 3 only if both fail, and cost the re-measurement honestly.

If any of 2 or 3 succeeds, the `llama.exceptions` entry in `cluster.json` is deleted and
the cluster returns to a single revision — which is why that exception was written as a
declared, testable value rather than a loosened rule.


---

# Path 1 vs Path 2, decided on evidence (2026-09-20)

Further desk research materially changed the picture. The two paths do **not** trade off
on difficulty — they trade off on **which thing you are willing to lose**.

> **Path 1 buys a single revision and risks losing FP16.
> Path 2 buys FP16 and pays a patch-maintenance burden.**

## The finding that separates them: FP16 on the Nano's Vulkan driver is doubtful

The proposal does not call this node "the GPU node". It calls it **the FP16 node**, and
the reason is specific: Maxwell **sm_53 has native `half2` arithmetic** — it is the one
Maxwell part that does — which is exactly why a 2019 Nano is scientifically interesting
in a heterogeneity study at all. `docs/` already leans on this: "JetPack 4.6.6 supports
FP16 but not efficient INT8" is a stated project constraint.

Vulkan does not inherit that for free:

- `shaderFloat16` is an **optional** feature in core Vulkan 1.2. "L4T r32.7 supports
  Vulkan 1.2" therefore does **not** imply FP16 is available.
- NVIDIA's own forum carries an RFE, *"update Jetson Vulkan driver to include fp16
  support"*, opened against the Jetson Vulkan driver for missing
  `VK_KHR_shader_float16_int8`. NVIDIA acknowledged it in **May 2019** with "targeting
  the next release" and the thread ends there with **no confirmed resolution**, and
  related threads report Vulkan limitations persisting on Jetson well afterwards.

So Path 1's FP16 risk is **concrete and documented**, not speculative. If it lands that
way the node is still GPU-accelerated, but the study loses the FP16 claim, the
FP16-vs-INT8 constraint discussion loses its basis, and "the FP16 node" has to be
reworded throughout the proposal.

What Path 1 does have going for it: llama.cpp's Vulkan backend demonstrably runs on very
old hardware (a 2012 FirePro W8000, RX 580 clusters), so *Maxwell being old* is not by
itself a reason to expect failure — and it needs **no patches at all**, so it builds at
`b23701f7` and the `llama.exceptions` entry is simply deleted.

## Side by side

| | **Path 1 — Vulkan at `b23701f7`** | **Path 2 — CUDA patch set at `b23701f7`** |
|---|---|---|
| Cluster revisions | **one** — exception deleted | **one** — exception deleted |
| FP16 | **at risk** — optional in VK 1.2, Jetson RFE unresolved since 2019 | **kept** — native `half2` on sm_53, as the proposal assumes |
| Patch burden | **none** | 5 categories, re-applied on every ref bump |
| Precedent | none on a Nano that I can find | **two independent** — kreier at b5050, Z841973620 at b8056 |
| Perf vs CUDA | unknown on Tegra Maxwell | the known quantity (~20% over CPU) |
| Write-up cost | reword the FP16 node if fp16 is absent | classify each patch as build-only vs precision-touching |
| Fails if… | driver lacks the features llama.cpp's shaders need | patches don't port to a 2026 CUDA backend |

Both end in the same good place — one revision, GPU, pooled runs — so the choice is
purely about which failure you would rather absorb.

## Recommendation: Path 2, but run one 2-minute test first

**Path 2 protects the research claim, and the claim is why the node is in the study.**
A tidy build story is worth less than the FP16 result it would cost. Path 2 also has two
independent precedents; Path 1 has none on this hardware.

**But settle Path 1 before committing**, because one command decides it outright and
costs nothing:

    vulkaninfo | grep -i -A2 shaderFloat16        # on the Nano

If `shaderFloat16 = true`, Path 1 becomes strictly better than Path 2 — same destination,
zero patches, no maintenance — and should be taken. If it is false, Path 1 cannot deliver
the FP16 node and Path 2 is the answer.

That is the cheapest decisive test in the whole matrix, and it needs no build.

## Sequence when the board arrives

1. `vulkaninfo | grep shaderFloat16` — decides Path 1 vs Path 2 outright.
2. `-DGGML_RPC=ON` smoke test on the chosen backend. **Still the common blocker**: no
   published Nano build uses it, and both paths die here if it does not work.
3. Build the chosen path at `b23701f7`, confirm `ggml-rpc-server` runs.
4. Join a pooled run with the Pi 5 + Pi 4 and re-check the 14B result with three nodes.
5. Delete the `llama.exceptions` entry in `cluster.json`; the hygiene test then enforces
   the single revision automatically.

## If Path 2 is taken, the patch classification is not optional

`cluster.json`'s `_pin_note` already commits to it: *build/toolchain-only is a footnote,
anything touching kernel math or precision is a declared threat to validity and wants a
CPU-backend parity check on the same node.* Against that rule the known patch set sorts
as:

| Patch | Class |
|---|---|
| CUDA arch → 50/61 | build-only — footnote |
| linker fix in `ggml/CMakeLists.txt` | build-only — footnote |
| strip `constexpr` | build-only — footnote |
| comment `__builtin_assume` | build-only, but removes an optimiser hint — note it |
| **`nv_bfloat16` → `half` / remove bf16** | **precision-touching in principle** |

The bf16 one is the only one that needs an argument, and the argument is available: the
model ladder is Q4_K_M and Q8_0, neither of which uses bf16 paths, so the change should
be unreachable for every model under test. **Say that explicitly and back it with the
parity check** rather than leaving it implied.
