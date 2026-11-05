# 03 — Building llama.cpp on the Pi 4, and the models this node holds

## Why the flags are written out

`scripts/02_build_llama_cpp.sh` builds with `-DGGML_NATIVE=OFF` and an explicit
`-march=armv8-a+crc+simd -mtune=cortex-a72`.

`GGML_NATIVE=ON` would let the compiler pick features from the host, which is convenient
and wrong for this project: the build then differs silently between nodes, and a Pi 4 vs
Pi 5 gap could be a compiler decision rather than a hardware property. Writing the flags
down makes the ISA difference an explicit, recorded input to the dataset.

What the Cortex-A72 has: ARMv8-A, NEON (`asimd`), CRC32, FP16 *storage*.
What it lacks, and the Pi 5's Cortex-A76 has: **dotprod** (`asimddp`).

It also lacks **i8mm** -- but so does the A76. i8mm is ARMv8.6 (Cortex-A710 and
later) and the A76 is ARMv8.2, so i8mm is *not* part of the Pi4/Pi5 contrast. It is
still recorded as absent because the dataset should say what was checked.

This is the single most important fact about this node. llama.cpp's quantised matrix
kernels for Q4_K_M and Q8_0 have dotprod paths that give the A76 a large advantage; on the
A72 they fall back to plain NEON. The Pi 4 is not merely "an older Pi" in this cluster —
it is missing the instruction that the workload is built around, which is exactly why the
proposal designates it the straggler and why the straggler study is interesting rather
than trivial.

`node/device_info.py` records the feature bits so this claim is evidence in the dataset
rather than an assertion in the report.

## Pinning the revision

The first build leaves `LLAMA_PIN_SHA` empty and builds the tip of the default branch;
the script prints the resolved SHA and writes it to
`$DATA_DIR/inventory/build_manifest.json`.

**Copy that SHA into `config/node.env` on every node before Phase 2 begins.** llama.cpp
moves fast and its ARM kernels in particular have changed measurably between releases.
A device-characterisation dataset gathered across two revisions is not a dataset — it is
two half-datasets that cannot be compared. `scripts/99_healthcheck.sh` fails while the pin
is empty, on purpose.

The manifest also records the compiler version, the cmake options, the kernel and the
CPU feature string, so a result can be reproduced years later from the file alone.

## Build notes for this board (8 GB, no swap)

- `-j2`, not `-j4`. This was chosen when the board was believed to be 4 GB, where a four-way
  build with swap disabled (by design) OOMs the compiler on the larger ggml translation units.
  On this 8 GB board `-j4` has not been tried, so `-j2` stays. It takes roughly 25–40 minutes.
- `ccache` is installed by the bootstrap; a rebuild after a pin change is much quicker.
- `BUILD_SHARED_LIBS=OFF` keeps `rpc-server` self-contained, so moving the binary or
  updating a library cannot change what is running mid-experiment.
- OpenBLAS is **neither installed nor enabled**. `GGML_BLAS=OFF` is passed explicitly and
  `libopenblas-dev` is deliberately absent from `00_bootstrap.sh`, so a stray
  `-DGGML_BLAS=ON` fails to configure instead of quietly producing a node whose CPU numbers
  are not comparable with the others'. On the Pi 4 it helps prefill slightly and can hurt
  decode, and enabling it on one node and not another would confound the cross-node
  comparison — which is why the whole cluster is BLAS-free (decided 2026-09-14). If you want
  it, install the headers, flip the flag, and measure it as a separate documented variant —
  do not fold it into the baseline.

## Models on this node

`config/models.txt` lists small models only (TinyLlama 1.1B, Qwen2.5 1.5B, Llama 3.2 3B,
at Q4_K_M and Q8_0). They exist for one purpose: the **isolated** characterisation runs of
Phase 2, where every node is benchmarked alone before the cluster is assembled.

This is an 8 GB board with about 7.4 GB free at rest (`testbed/dataset/memory_at_rest.json`),
so a 7B Q4_K_M (4.4–4.7 GB of weights) would fit it alone; the 14B and larger models are split
across boards. In a pool, the board that runs llama.cpp holds the GGUF and sends each worker
its layers over RPC. As a worker, this node has held up to 35 of the 14B's blocks plus its
output head (2026-09-25). It also hosts the Qwen3-30B-A3B MoE, whose only copy is here
(2026-09-28).

Q4_K_M and Q8_0 are both measured because the quantisation level changes the
compute/bandwidth balance: Q8_0 moves twice the bytes per weight, and on a
memory-bandwidth-bound decode phase that shows up immediately. Two points are the minimum
needed to say anything about that slope.

## rpc-server memory

`NODE_USABLE_MB=2600` was sized against the ~3200 MB a 4 GB board leaves free. This board is
8 GB and leaves about 7.4 GB free at rest, so the figure is stale (see
`.handoff/nodes/Pi4/2026-09-27-2100-NOTE-memory-budget-vs-measured.md`). The gap is deliberate
headroom: the node has no swap, the straggler experiments deliberately overload it, and an
OOM kill mid-run does not just lose a measurement — it can take the SSH session with it and
end the whole sweep.

**Whether `rpc-server` advertises anything depends on the build.** `--mem` was removed
upstream in 41386cf (2025-10-17), so on a current master build the option set is
`-H -t -d -p -c -h` and the server answers `GET_DEVICE_MEMORY` with the board's real free
memory. `scripts/lib.sh rpc_extra_args()` probes `--help` and passes `--mem $RPC_MEM_MB`
only if the flag exists, warning loudly when it does not — so on a modern build `RPC_MEM_MB`
reaches nothing, and on an older pinned build it still works.

Because that is a property of whichever SHA is pinned, the budget is enforced in the one
place the control node always reads: `NODE_USABLE_MB`, via the agent's `usable_budget_mb`,
with `systemd/heteroserve-rpc.service` (`MemoryHigh=2800M`, `MemoryMax=3G`) as a backstop
*above* it. Note the ceiling is itself conditional — see `make health`'s memory-cgroup gate.

Upstream flags drift, and this kit has already been bitten by it twice: the binary was
renamed to `ggml-rpc-server`, and `--mem` never existed at all. `scripts/lib.sh` therefore
resolves the binary with `bin_rpc()` and probes the accepted flags with `rpc_extra_args()`
rather than hard-coding either. Note that upstream `exit(0)`s on an unknown argument, so a
bad flag looks like a clean shutdown — never trust the exit code, check that the port is
listening. Record any change in the run log so the dataset says which invocation produced
it.

## Verifying the cluster path

From the control node, with `rpc-server` running here:

```bash
llama-cli -m models/llama3.2-3b-q4km.gguf \
  --rpc 192.168.50.13:50052,192.168.50.12:50052 \
  -ngl 99 -p "hello" -n 32
```

`scripts/07_rpc_smoke.sh` covers the node side and tells you whether the control node
actually allocated buffers here.
