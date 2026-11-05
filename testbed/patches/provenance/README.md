# Provenance of `jetpack46-cuda102-b23701f.patch`

Preserved 2026-09-25 from the scratchpad of the session that built the patch, before
`/tmp` was cleared. Needed to answer "where did each change in the CUDA port come from?"

| File | What it is |
|---|---|
| `jetpack46_b10621.patch` | The source: the CUDA 10.2 / JetPack 4.6 port from the fork `Z841973620/llama.cpp-jetpack4.6`, as a diff against its own base, **b10621** |
| `resolve.py` | How the 13 conflicts from three-way merging that onto the cluster pin `b23701f7` were decided |

`resolve.py` decides every conflict explicitly — no blanket take-ours/take-theirs — by one
rule: **keep the fork's C++14 structure, carry upstream's new semantics into it.** Each
decision asserts the shape it expects, so it fails loudly rather than silently resolving the
wrong way if either side changes.

**It is not the complete record.** It resolves the conflict markers in `mmvf.cu`, `mmvq.cu`,
`fattn.cu`, `mmf.cuh`, `mmid.cu` and `fattn-tile.cuh`. Twice it says a helper body is
brought up to date "below" (the `mm_ids_helper_specialized` padding in `mmid.cu`, and
`use_sparse` inside `launch_fattn_tile_case`) — but the file ends there, so those two edits
were made separately and exist only in the final patch. The "13 conflicts" figure comes
from the handoff, not from this file; it counts markers at run time.

It is a record, not a re-runnable build step: it runs against a merged tree
(`python3 resolve.py <merged-tree>`) that was produced in that scratchpad and not kept.
The authoritative artefact is `../jetpack46-cuda102-b23701f.patch`.

## Versions

| Version | sha256 | What changed |
|---|---|---|
| v1 (2026-09-25), kept here as `jetpack46-cuda102-b23701f.v1.patch` | `69262c35…704b` | The port as merged; 3,444 lines, 34 files. It applied cleanly but was never built with `GGML_RPC=ON` |
| **v2 (2026-09-29), current** | `1f43a3e4…2283c6` | v1 plus two build fixes found by the first full build on the board (see below); 3,506 lines, 35 files (git format) |

v2's two fixes, each a CMake insertion with its reason in a comment:
1. **`ggml/src/ggml-rpc/CMakeLists.txt`**: `target_compile_features(ggml-rpc PRIVATE cxx_std_17)`. The port holds ggml at C++14 for
   nvcc 10.2, but `ggml-rpc.cpp` uses `std::filesystem` (its tensor cache). The RPC server is plain C++ that nvcc never compiles.
   Without it, the first build stopped at 8%.
2. **`ggml/src/ggml-cuda/CMakeLists.txt`**: `target_link_options(ggml-cuda PRIVATE "LINKER:--exclude-libs,libstdc++fs.a")`.
   - `libggml-cuda.so` is linked by nvcc's host compiler (g++-8). The port's `link_libraries(stdc++fs)` then pulls in **gcc 8's**
     static filesystem library, and the .so exported its 329 `std::filesystem` symbols.
   - At run time that copy interposed over gcc 11's libstdc++, and every tool segfaulted at start-up in
     `path::_M_split_cmpts` (from `ggml_backend_load_best`). With the fix it exports 0, and `llama-bench --list-devices`
     finds `CUDA0: NVIDIA Tegra X1`.

**Checked.**
- `pin + v1` and `pin + v2`, each applied to a fresh checkout of `b23701f7`, differ by exactly those two insertions (`diff -r`).
- v2 passes `git apply --check` on the laptop's clean pin and on the board's `~/llama.cpp`.
- The board's build that produced the benchmarked binaries was the script up to 8%, then two in-place continuations. Its tree
  is `pin + v2` exactly. A single clean run of `jetson_build_llama_cuda.sh` with v2 has not been done yet.
