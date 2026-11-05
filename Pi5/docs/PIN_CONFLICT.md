# The pinned llama.cpp revision is not the one that produced the measurements

**Raised 2026-09-20. RESOLVED the same day** — see *Decision* at the bottom.
Kept as the record of why the cluster pin moved, because the reasoning is a threat-to-
validity note the write-up will need.

## The two revisions

| | SHA | Date | Status |
|---|---|---|---|
| **Pinned** (apparatus) | `23106f94ea2bc3da929afb7330655fd5515d08dc` (b5050) | 2025-04-04 | in `Pi5/config/node.env:47` and `ThinkPad/config/cluster.json` → `llama.pin_sha`. **Never built on any node.** |
| **Built** (measurements) | `b23701f77d47dad9de834d59ebfcbe25c9e8b46f` | 2026-09-19 | what is compiled on Pi 5 and Pi 4, and what produced **every figure** in `testbed/benchmarks.md`. |

## Why b5050 was pinned

From `Pi5/.handoff/2026-09-20-1157-survivors-closed-and-pin-set.md`:

- It is the newest revision that still satisfies the **Jetson's CUDA 10.2 ceiling**.
  The unpatched window is empty: b4419 (2025-01-06) adds `#include <cuda_bf16.h>` to
  `ggml-cuda/vendors/cuda.h`, which `common.cuh` pulls into every CUDA translation
  unit — nine days *before* the `-no-cnv` floor at b4475.
- It **predates all three upstream `rpc-server` hazards**: the `--mem` removal
  (41386cf, 2025-10-17), the `rpc-server` → `ggml-rpc-server` rename (c299a92,
  2026-06-27), and the `examples/rpc` → `tools/rpc` move (2025-05-02).

## Why this is not cosmetic

The built revision is **after** all three hazards, and one of them bears directly on
the memory-pooling experiment this project exists to run:

- **`--mem` is gone.** Verified on the bench 2026-09-20 — `ggml-rpc-server --help` on
  `b23701f7` offers only `-t -d -H -p -c -h`. There is no way to tell the worker what
  to advertise.
- Consequently the worker answers `GET_DEVICE_MEMORY` with the device's **real free
  RAM**, not the configured `RPC_MEM_MB=2600`. The "deliberate 6144 MB headroom" on the
  Pi 5 and "2600 MB headroom" on the Pi 4 are **not in effect** on the built revision.
- The binary is named `ggml-rpc-server`, i.e. post-rename — independent confirmation
  that the build is past the hazard window.

So the apparatus is configured for a revision whose safety properties the measurements
do not have.

## The choice (as put to the user)

1. **Rebuild both Pis on b5050** and re-take every measurement. Restores the memory
   headroom mechanism and keeps the Jetson reachable on one cluster-wide revision.
   Cost: two rebuilds (Pi 4 ≈ 29 min) and the whole measurement set re-run. Risk:
   b5050 is 2025-04; it predates the split-GGUF and `-dev`/`-ts` handling the current
   14B pooling test relies on, so that test may need reworking.
2. **Move the pin forward to `b23701f7`** and accept that the Jetson cannot join on
   the same revision (it is capped at CUDA 10.2), i.e. the cluster runs two revisions
   and the report says so.
3. **Keep both deliberately**: b5050 for the Jetson-inclusive runs, `b23701f7` for the
   ARM-only runs, and report per-revision. Honest but doubles the matrix.

**Not decided here.** Option 2 is the cheapest and matches what is already measured,
but it gives up the single-revision property that b5050 was chosen to protect — and
that trade is the advisor's to make, not this session's.

## What is safe to say meanwhile

Every figure in `testbed/benchmarks.md` is internally consistent: one revision, one
model checksum, both nodes. Nothing there is invalidated. The exposure is that the
*apparatus configuration* describes a different revision, so anyone following the
setup docs would build something the numbers do not describe.

---

# Decision — 2026-09-20: move the pin forward to `b23701f7`

**Chosen by the user: option 2.** The cluster pin is now
`b23701f77d47dad9de834d59ebfcbe25c9e8b46f`.

The pin now **follows the evidence**: it names the revision that is actually compiled
on both Pis and that produced every figure in `testbed/benchmarks.md`. Nothing has to
be re-measured, and the apparatus no longer describes a revision nobody has built.

## Applied in all four locations

| File | Key | Value |
|---|---|---|
| `ThinkPad/config/cluster.json` | `llama.pin_sha` | `b23701f7…` |
| `Pi5/config/node.env:47` | `LLAMA_PIN_SHA` | `b23701f7…` |
| `Pi4/config/node.env:39` | `LLAMA_PIN_SHA` | `b23701f7…` (was empty) |
| `Jetson/config/node.env` + `.example` | `LLAMA_CPP_REF` | `b23701f7…` (was empty) |

`ThinkPad/tests/test_repo_hygiene.py` asserts all four agree. It had been failing on
this exact inconsistency; **it now passes — full suite 165 tests, OK.**

## One thing got safer

The old `_pin_note` warned that a mixed-revision cluster **silently misexecutes**,
because the RPC opcode enum shifted between b4475 and b5050 and the `RPC_CMD_HELLO`
version handshake only lands 2025-04-18. `b23701f7` is long past that date, so a node
on a different ref is now **refused at the handshake** instead of quietly computing
the wrong thing. That is a real argument in favour of moving forward — though a
refused node is still a node that cannot join a pooled run.

## What it costs — and this is still open

**The Jetson cannot build the CUDA backend on this pin.** JetPack 4.6.x is CUDA 10.2;
`b23701f7` needs CUDA ≥ 11. The old pin existed precisely to keep the Jetson buildable,
and that constraint has now been inverted — the Pis' measured revision is the authority
and the Jetson has to adapt. It must pick one, and **this has not been decided**:

- **(a) `BUILD_CUDA=0`, CPU-only on `b23701f7`.** One revision across the whole cluster.
  But it gives up the FP16 GPU role that is the entire reason the Jetson is in the
  study — the proposal lists it as "the GPU-accelerated node", and QR-level claims about
  GPU placement would have nothing to stand on.
- **(b) Keep CUDA on its own older ref (b5050).** The cluster runs two revisions and the
  report must declare it as a threat to validity. The handshake makes this fail loudly
  rather than silently, but the Jetson then cannot participate in a pooled run at all.

Neither is free. (a) costs a research claim; (b) costs cluster membership. Flagged in
`Jetson/config/node.env` at the pin itself, where the Jetson session will hit it.

---

# Decision — 2026-09-20: the Jetson keeps FP16 GPU (option b)

**Chosen by the user.** The Jetson stays on **b5050 = `23106f94…`** with `BUILD_CUDA=1`,
so the cluster is deliberately **two-revision**.

The reasoning is that the Jetson's only reason to exist in this study is that it is the
one accelerator. CPU-only would have kept a tidy single revision, but it would have
deleted the FP16 GPU node from the design and with it every QR-level claim about GPU
placement — a tidy repo bought with a research contribution.

## Declared, not drifted

The split is registered in `ThinkPad/config/cluster.json` under `llama.exceptions` as a
mapping of location → the exact value that location may hold:

    "exceptions": {
      "Jetson/config/node.env":         "23106f94…",
      "Jetson/config/node.env.example": "23106f94…"
    }

`test_repo_hygiene.py::TestLlamaRevisionPin` was strengthened to match. It previously
only checked set-vs-unset for the cluster-wide invariant, so **four locations holding
four different SHAs would have passed** — the exact partial rollout its own docstring
described was not actually caught by that check. It now:

- fails on any **undeclared** divergence (verified: a bogus SHA in `Pi4/config/node.env`
  turns it red);
- fails if a **declared** exception holds anything other than its declared value
  (verified the same way) — so "the Jetson is special" cannot quietly become "the Jetson
  is on whatever it happens to be on";
- fails if `llama.exceptions` names a path that is not a checked pin location.

Suites after the change: ThinkPad **166**, Pi 5 **357**, Pi 4 **155** — all OK.

## The cost, and it is not small: no pooled runs on the Jetson

**b5050 (2025-04-04) predates `RPC_CMD_HELLO` (2025-04-18)**, so it has no version
handshake, and the RPC opcode enum shifted between b4475 and b5050. A `b23701f7` main
node driving a b5050 worker is precisely the mixed-revision case that **silently
misexecutes** rather than refusing to connect.

Note this inverts the reassurance recorded when the pin moved forward. Moving to
`b23701f7` made mismatches *safe* only because both ends would then have the handshake.
Keeping the Jetson on a pre-handshake ref puts it back in the dangerous window — the
protection lives on the OLD node, and the old node does not have it.

So the Jetson is now a **standalone-characterised node**:

- single-node prefill / decode / energy on its own revision — still a real contribution,
  and the per-device characterisation dataset is a deliverable in its own right;
- **not** a member of pooled-memory runs. Those stay on the Pi 5 + Pi 4 pair, which
  share `b23701f7` and have already demonstrated 14B Q4 pooling.

Every Jetson figure must carry its revision, and any cross-node comparison that mixes
revisions has to say so. That is a declared threat to validity, not a footnote.

## What would dissolve the trade-off — and the user says it matters

**Confirmed 2026-09-20: pooled GPU runs matter to the argument**, so this was scoped.
Full write-up in `Jetson/docs/50_pooled_gpu_options.md`; the short version is that the
trade-off is **probably not real**, because the CUDA-10.2 ceiling is later than assumed.

b5050 was "the newest ref with a published step-by-step Nano recipe" — which is a
different claim from "the newest ref that builds". A second fork,
`Z841973620/llama.cpp-jetpack4.6`, carries a CUDA-10.2 build at **b8056**, far newer than
b5050 and comfortably **after** the 2025-04-18 handshake. So a CUDA-10.2 build that
speaks the current RPC protocol demonstrably exists.

That maintainer additionally recommends the **Vulkan** backend over CUDA, because
JetPack 4.6 / L4T r32.7 ships **Vulkan 1.2** — and Vulkan builds from current master,
which would put the Jetson on `b23701f7` itself with GPU acceleration and no exception at
all.

Order of attack when the board arrives: first prove `-DGGML_RPC=ON` builds and runs on a
Nano *at all* (still nobody's published that, and every path lands on it); then Vulkan at
the cluster pin, checking `shaderFloat16` because the proposal's FP16 commitment depends
on it; then the CUDA patch set applied to `b23701f7`; and only then consider moving the
cluster back to b5050, which costs re-measuring everything and removes none of the risk.

**The pin is not being reversed now.** The Jetson is not physically present, so nothing
is blocked, and `b23701f7` is where the measurements and the working 14B pooling are. The
`llama.exceptions` entry is written as a declared, testable value precisely so it can be
*deleted* if one of those paths lands and the cluster returns to a single revision.
