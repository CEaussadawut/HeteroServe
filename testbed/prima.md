# prima.cpp — the comparison baseline (proposal Phase 5: "comparison against prima.cpp under identical conditions")

## Source, pinned

| | |
|---|---|
| Repository | [github.com/OpenCPIL/prima.cpp](https://github.com/OpenCPIL/prima.cpp), the one the paper gives. The paper is arXiv:2504.08791 v3, 2026-07-04: "Prima.cpp: Fast 30-70B LLM Inference on Heterogeneous and Low-Resource Home Clusters" (ICLR 2026 poster). The paper's earlier address, `github.com/Lizonghang/prima.cpp`, returns "Repository not found" (checked 2026-09-30) |
| Commit | `288df6a2cc2120387e12d6f40798f30cc525c9ea` (tree `d3310acde16f98a1d15f13dfdcaab774bb33adf4`), "Update README", 2026-09-01. The repository has 5 commits: the code was uploaded as files on 2026-06-30, so there is no llama.cpp history in it |
| Licence | MIT |
| Downloaded | 2026-09-30 04:4x (user: "then download the prima.cpp"), `git clone` to the laptop's `/opt/heteroserve/prima.cpp`, then checked out at the pin. 97 MB on disk; the GitHub API gives 32,138 KB |

## What it can run here (read from its code at the pin, 2026-09-30)

- **Its llama.cpp base predates Qwen3.** `src/llama*.cpp` knows `llama`, `qwen`, `qwen2`, `qwen2moe`, `gemma2`, `deepseek2`, `phi3` and others, but **no `qwen3`, `qwen3moe` or `qwen35`**. The tree still has `ggml/src/ggml.c`, `ggml-aarch64.c` and kompute, so it is a late-2024 base.
  - So it cannot load **Qwen3-14B, Qwen3-30B-A3B, Qwen3.5-9B or Qwen3.8-27B**.
  - It can load **Mistral-Nemo 12B** (architecture `llama`; the `tekken` pre-tokenizer is present), **TinyLlama**, and **Qwen2.5-14B** (`qwen2`). Qwen2.5-14B is in its README's supported list; it was dropped from this study on 2026-09-29 and archived on the laptop.
- **Backends:** CPU, CUDA and Metal. It has no Vulkan in its distributed path, and the Jetson's GPU is used here through Vulkan.
- **Prerequisites (README):** gcc ≥ 9, make ≥ 4.2.1, cmake ≥ 3.16.3, fio ≥ 3.16, ZeroMQ ≥ 4.3.2 on every device, and HiGHS ≥ 1.9.0 on rank 0 (the head: its Halda scheduler solves the layer assignment). It ships the cppzmq headers (`include/zmq.hpp`, `zmq_addon.hpp`), which need `<zmq.h>`.
- **Every device loads the model from its own local copy** (mmap, with `--prefetch`). A llama.cpp RPC worker instead receives its tensors from the host.

## The boards against its prerequisites (2026-09-30)

| | gcc | make | fio | ZeroMQ | internet |
|---|---|---|---|---|---|
| Pi 5 (Debian 13) | 14.2 | 4.4.1 | 3.39 | runtime 4.3.5; **no `libzmq3-dev` (headers)** | yes (Wi-Fi) |
| Pi 4 (Debian 13) | 14.2 | 4.4.1 | 3.39 | runtime 4.3.5; **no `libzmq3-dev`** | yes (Wi-Fi) |
| Jetson (Ubuntu 18.04) | **7.5** | **4.1** | **3.1** | runtime **4.2.5** | **none** |

The Pis need only the ZeroMQ headers (`sudo apt install libzmq3-dev`: the user's step). The Jetson falls short on four prerequisites and has no internet to fetch them. It is out of a prima.cpp run unless its toolchain is rebuilt.

## Made to run Qwen3-14B (user, 2026-09-30: "i insist to go with qwen3-14B")

**The port** (`testbed/patches/prima-288df6a-qwen3-v1.patch`, `b8359f2f…`; written by
`.handoff/assets/prima-2026-09-30/port_qwen3.py`): 323 lines added, one assertion changed, all in `src/llama.cpp`.
- **What Qwen3 dense is:** qwen2 without the Q/K/V biases, and with a per-head RMS norm of Q and K before RoPE (upstream llama.cpp
  at the cluster pin, `src/models/qwen3.cpp`). Every edit copies prima.cpp's own qwen2 code and makes only those changes.
- **The edits:**
  - the architecture and its tensor names;
  - the hyperparameters;
  - a loader, `llm_load_qwen3_tensors`: prima.cpp's per-rank qwen2 loader with the Q/K norms;
  - both loader call sites: the model load, and the scheduler's profiler;
  - a distributed graph, `build_qwen3`: prima.cpp's `build_qwen2` with the norms;
  - the graph dispatch, and the distributed-support assertion (llama/qwen2 only until now);
  - the NEOX RoPE type.
- **The profiler:** it counts FLOPs per tensor by name, and on an unknown name it logs "Uncaught tensor" and returns early. It is
  taught the two norm tensors (counted as `attn_norm` is), so Halda gets a complete profile.
- **Not changed:** the F32 attention-precision list (upstream does not force it for qwen3) and everything else.
- **Checked in the comparison's registered rule CORRECTNESS:** its greedy output against llama.cpp's.

**Built** on both Pis (`testbed/scripts/prima_build.sh`, 2026-09-30 04:57):
- `git archive` of the pin plus the patch; the patched `src/llama.cpp` (`ab8b99ce…`) is hash-checked on each board;
- then `make -j4 llama-cli llama-server`.
- **Pi 5, rank 0:** relinked with **HiGHS v1.15.1**, the official prebuilt `highs-1.15.1-aarch64-linux-gnu-static-mit.tar.gz`
  (7,044,260 bytes). Its sha256 `82a20506…` matches GitHub's published digest; the user approved the download. It is unpacked in
  `~/highs-1.15.1`, with nothing installed system-wide.
  - It links statically with `libhighs_extras` and zlib.
  - Binaries: `llama-server` `ca0de309…`, `llama-cli` `530943ae…`.
- **The ZeroMQ headers** (`libzmq3-dev` 4.3.5) were installed on both Pis by the user.

**Registered before prima.cpp's first run:** `testbed/dataset/prima_predictions_20260930.json` (05:05:28, `f074fd61…`).

## Result (2026-09-30, registered; `testbed/benchmarks.md`, "PRIMA")

| Pi 5 + Pi 4, Qwen3-14B, QR3 protocol | mean TPOT | P95 | TTFT (128 tokens) |
|---|---|---|---|
| the engine + llama.cpp (Pi 5 32, Pi 4 8) | 1,188.6 ms | 1,216.1 | 45.2 s |
| prima.cpp + Halda (Pi 5 24 + head + embedding, Pi 4 16) | 1,339.3 ms | 1,356.1 | 81.6 s |

- **The port is exact:** 128 of 128 greedy tokens match llama.cpp.
- **Halda's split is memory-bound.** 24 blocks is the most that fits its profiled 5.85 GiB Pi 5 budget, once the output layer, the
  embedding and the KV cache are counted.
- **The runtimes are equivalent:** our cost model prices its split within 2 %.
- **Binaries:**
  - Pi 5: `llama-server` `ca0de309…`, `llama-cli` `530943ae…` (with HiGHS);
  - Pi 4: `llama-server` `a54dda92…`.
