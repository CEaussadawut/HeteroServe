# Audit findings — recovered from workflow run `wf_d276e2c1-96b`


> **Provenance and status.** These come from an audit workflow launched
> 2026-09-12 15:04 that was cut off before it finished. Its `Audit` and `Facts`
> phases completed (7 of 8 agents wrote results); its **`Verify` phase never
> ran**. The adversarial verifier that was supposed to kill weak findings
> therefore never executed.
>
> So: the **claims** below carry real verdicts, because the fact-check lenses
> produced verdicts directly and cited sources. The **findings** below are
> *unverified leads* — plausible, specific, but not adversarially checked.
> Confirm each against the code (and on hardware) before acting on it.
>
> Recovered from `journal.jsonl` rather than by re-running the audit; the
> workflow's own `resumeFromRunId` is same-session-only and could not replay.


## Fact-checked claims (13)

These have verdicts. `partly-wrong` means the kit's stated rationale is
inaccurate even where its behaviour may be fine — read the evidence.


### [PARTLY-WRONG] facts:llamacpp-rpc

**Claim.** RPC server binary name. scripts/lib/common.sh: `bin_rpc() { find_bin "$LLAMA_BIN" rpc-server llama-rpc-server; }`; scripts/20_build_llama_cpp.sh dies with "rpc-server was not built — GGML_RPC did not take effect on this ref"; scripts/50_bench_device.sh `pgrep -x rpc-server`; docs/20_troubleshooting.md `pgrep -a rpc-server` and the heading "`rpc-server` was not built".


**Evidence.** Upstream has used exactly two names, and `llama-rpc-server` is not one of them. Historically (examples/rpc/CMakeLists.txt, e.g. tag b4000) the target was `add_executable(rpc-server rpc-server.cpp)`. After the move to tools/ (PR #13249, May 2025) it stayed `rpc-server`, and it was renamed on 2026-06-27 by commit c299a92, "binaries : Improve rpc-server and export-graph-ops names. (#25045)": tools/rpc/CMakeLists.txt now begins `set(TARGET ggml-rpc-server)` and tools/rpc/README.md's examples all read `bin/ggml-rpc-server -p 50052`. Concrete failure: config/node.env.example ships `LLAMA_CPP_REF=""`, so the default build is master. The build succeeds and produces build/bin/ggml-rpc-server, then `bin_rpc` returns non-zero and 20_build_llama_cpp.sh aborts the whole build with a message blaming GGML_RPC — sending the operator to docs/20_troubleshooting.md, which tells them to walk CMake spellings backwards for a problem that does not exist. `pgrep -x rpc-server` likewise never matches the running worker, so 50_bench_device.sh's "an rpc-server is running and will steal CPU" guard silently passes while a live RPC worker contends for the 4 GB and 4 cores during the sweep — quietly corrupting the single-node baseline numbers.


**Correction.** scripts/lib/common.sh: `bin_rpc() { find_bin "$LLAMA_BIN" ggml-rpc-server rpc-server; }` (drop `llama-rpc-server`, which never existed upstream). scripts/50_bench_device.sh: replace `pgrep -x rpc-server` with `pgrep -f '(ggml-)?rpc-server'`. docs/20_troubleshooting.md: replace `pgrep -a rpc-server` with `pgrep -af '(ggml-)?rpc-server'` and retitle the section "the RPC server binary was not built (`ggml-rpc-server`, formerly `rpc-server`)". scripts/20_build_llama_cpp.sh: change the die message to name both binaries.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/CMakeLists.txt ; https://github.com/ggml-org/llama.cpp/commits/master/tools/rpc/CMakeLists.txt (c299a92, "binaries : Improve rpc-server and export-graph-ops names. (#25045)", 2026-06-27) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/examples/rpc/CMakeLists.txt ; scripts/lib/common.sh:63


**Files.** `scripts/lib/common.sh, scripts/20_build_llama_cpp.sh, scripts/50_bench_device.sh, docs/20_troubleshooting.md`


### [PARTLY-WRONG] facts:llamacpp-rpc

**Claim.** RPC server flags. scripts/40_run_rpc_server.sh: "-H/-p are accepted by every ref that has the rpc example; -m (free-mem override) only by newer ones, so probe rather than assume", then `ARGS=( -H "${RPC_BIND}" -p "${RPC_PORT}" )`, `if "$RPC" --help 2>&1 | grep -q -- '-m,'; then ARGS+=( -m "${RPC_MEM_MB}" )`, and `if "$RPC" --help 2>&1 | grep -q -- '-c,'; then ARGS+=( -c )`. config/node.env.example: `RPC_MEM_MB=2048` with "Advertise less than you have or the orchestrator will hand you a partition that OOMs mid-run"; docs/20_troubleshooting.md: "Lower `RPC_MEM_MB` ... so the node advertises less."


**Evidence.** `-H, --host` and `-p, --port` are real and stable. `-c, --cache` is real on master. `-m/--mem` is GONE from master: print_usage() in tools/rpc/rpc-server.cpp is exactly `-h, --help`, `-t, --threads N`, `-d, --device <dev1,dev2,...>`, `-H, --host HOST`, `-p, --port PORT`, `-c, --cache` — no memory option at all; the server reports real free device memory. Worse, BOTH grep probes are written against help text that never existed. On the old refs that do have `-m`, the help line is `"  -m MEM, --mem MEM     backend memory size (in MB)\n"` — the literal string `-m,` does not occur, so `grep -q -- '-m,'` fails and the kit prints "this ref's rpc-server has no -m flag" and silently drops the advertised-memory cap on exactly the refs where it works. Conversely on old refs there is no `-c` and on master the help is `"  -c, --cache ..."` so only the cache probe happens to match. Concrete failure: on a b3xxx/b4xxx CUDA-10.2 build — the ref family the kit itself tells you to pin — the Nano advertises its whole free 4 GB shared pool instead of RPC_MEM_MB=2048, the orchestrator assigns an oversized partition, and the run dies with a CUDA OOM mid-sweep; the documented remedy "lower RPC_MEM_MB" has no effect because the flag is never passed. On master the same knob is simply inert.


**Correction.** Probe on the LONG option, which is stable across every ref: `if "$RPC" --help 2>&1 | grep -q -- '--mem'; then ARGS+=( -m "${RPC_MEM_MB}" ); else warn "this ref has no --mem; the node will advertise its real free memory — cap the partition from the orchestrator instead"; fi` and `if "$RPC" --help 2>&1 | grep -q -- '--cache'; then ARGS+=( -c ); fi`. In config/node.env.example and docs/20_troubleshooting.md, state that RPC_MEM_MB only takes effect on refs that still have `--mem` (removed upstream) and that on master the size of the partition must be controlled client-side with `--tensor-split`. Optionally replace the `--cpu` path's `CUDA_VISIBLE_DEVICES=""` with master's supported `-d CPU` / `--device CPU`.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/rpc-server.cpp (print_usage) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/examples/rpc/rpc-server.cpp (print_usage: "  -m MEM, --mem MEM     backend memory size (in MB)") ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md ; scripts/40_run_rpc_server.sh:57-68


**Files.** `scripts/40_run_rpc_server.sh, config/node.env.example, docs/20_troubleshooting.md`


### [PARTLY-WRONG] facts:llamacpp-rpc

**Claim.** Client side. scripts/70_cluster_smoke.sh: `"$CLI" -m "$MODEL" --rpc "$RPCLIST" -ngl 99 -n 32 -p "..."` with the comment "-ngl 99 so layers are actually handed to the remote backends; without it the RPC servers connect and then sit idle, which looks like success"; docs/20_troubleshooting.md: "You forgot `-ngl`. Without layers assigned to the remote backend, the RPC [servers idle] ... `-ngl 99` (or a measured partial value) on the client."


**Evidence.** The flag spelling is right: upstream tools/rpc/README.md shows `llama-cli -hf ggml-org/gemma-3-1b-it-GGUF -ngl 99 --rpc 192.168.88.10:50052,192.168.88.11:50052`, and tools/server/README.md documents `--rpc SERVERS` as "comma-separated list of RPC servers (host:port)". The `-ngl` claim is only true for the older refs: on master `-ngl, --gpu-layers, --n-gpu-layers N` is documented as "max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)", i.e. omitting it no longer leaves remote backends idle — llama.cpp auto-fits layers across all devices, local and remote, in proportion to available memory. Passing `-ngl 99` is still accepted and still correct, so the smoke test works; the defect is the diagnostic advice. Concrete failure: on a master-built orchestrator a split run that is genuinely slow (network-bound, or the Nano advertising the wrong memory) is misdiagnosed by docs/20_troubleshooting.md as "you forgot -ngl", and the operator adds a flag that was already effectively in force instead of looking at the link or the partition.


**Correction.** docs/20_troubleshooting.md: replace "You forgot `-ngl`. Without layers assigned to the remote backend, the RPC servers sit idle" with "On refs older than the `-ngl auto` default, omitting `-ngl` leaves every layer local and the RPC servers idle. On current refs `-ngl` defaults to `auto` and layers are auto-fitted across local and remote devices, so always pass an explicit `-ngl` (99, or a measured partial value) anyway — the study needs the placement fixed and recorded, not chosen by the auto-fitter." Keep `-ngl 99` in 70_cluster_smoke.sh and add `--tensor-split` guidance for controlling how much lands on the Nano.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md ("-ngl, --gpu-layers, --n-gpu-layers N | max. number of layers to store in VRAM, either an exact number, 'auto', or 'all' (default: auto)") ; scripts/70_cluster_smoke.sh:70-75 ; docs/20_troubleshooting.md:55-57


**Files.** `scripts/70_cluster_smoke.sh, docs/20_troubleshooting.md`


### [REFUTED] facts:llamacpp-rpc

**Claim.** CMake options. scripts/20_build_llama_cpp.sh: "Both spellings are passed because the option was renamed LLAMA_* -> GGML_* in Aug 2024 and we may be sitting on either side of that rename", with `--no-warn-unused-cli ... -DGGML_RPC=ON -DLLAMA_RPC=ON` and, on the CUDA path, `-DGGML_CUDA=ON -DLLAMA_CUDA=ON -DLLAMA_CUBLAS=ON` plus `-DGGML_CUDA_F16=ON -DLLAMA_CUDA_F16=ON`. docs/20_troubleshooting.md: "older refs spell it `LLAMA_RPC` ... The build script passes both spellings".


**Evidence.** The rename was June 26, 2024 (PR #8006, "llama : reorganize source code + improve CMake": "build options that are relevant to the ggml library are now prefixed with GGML_"), not August 2024. Passing both spellings is not harmless: llama.cpp's top-level CMakeLists.txt keeps a deprecation shim, and on both b4000 and current master it contains `llama_option_depr(FATAL_ERROR LLAMA_CUBLAS GGML_CUDA)` alongside `llama_option_depr(WARNING LLAMA_RPC GGML_RPC)` and `llama_option_depr(WARNING LLAMA_CUDA GGML_CUDA)`. `--no-warn-unused-cli` only silences unused-variable notices; it cannot suppress an explicit `message(FATAL_ERROR ...)`. Concrete failure: the default, intended path — `./20_build_llama_cpp.sh` with BUILD_CUDA=1 on any ref from June 2024 onward, which is every ref the kit can plausibly use — aborts at the `cmake "${CMAKE_ARGS[@]}"` step with "LLAMA_CUBLAS is deprecated ... Use GGML_CUDA instead" before a single object is compiled. The script never reaches the compile step, so the Nano never gets a CUDA build at all, and the error text points at cuBLAS rather than at the kit. (`-DLLAMA_RPC=ON` and `-DLLAMA_CUDA=ON` are merely noisy warnings; only LLAMA_CUBLAS is fatal. Note also that on master `LLAMA_CURL` is deprecated-and-ignored, so `-DLLAMA_CURL=OFF` no longer disables the libcurl dependency.)


**Correction.** In scripts/20_build_llama_cpp.sh delete `-DLLAMA_CUBLAS=ON` outright (it is a hard build-stopper), and gate the other legacy spellings on the ref instead of passing them unconditionally — e.g. `if ! grep -q 'GGML_RPC' "${LLAMA_SRC}/CMakeLists.txt" "${LLAMA_SRC}/ggml/CMakeLists.txt" 2>/dev/null; then CMAKE_ARGS+=( -DLLAMA_RPC=ON ); else CMAKE_ARGS+=( -DGGML_RPC=ON ); fi` and likewise for CUDA/CUDA_F16. Fix the comment to read "renamed LLAMA_* -> GGML_* on 2024-06-26 (PR #8006)" in both the script and docs/20_troubleshooting.md. If targeting master, add an explicit curl option (libcurl is no longer disabled by LLAMA_CURL=OFF).


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt (llama_option_depr block) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/CMakeLists.txt (same FATAL_ERROR LLAMA_CUBLAS line) ; https://github.com/ggml-org/llama.cpp/pull/8006 (merged 2024-06-26) ; scripts/20_build_llama_cpp.sh:88-113


**Files.** `scripts/20_build_llama_cpp.sh, docs/20_troubleshooting.md`


### [PARTLY-WRONG] facts:llamacpp-rpc

**Claim.** llama-bench. bench/run_matrix.py build_cmd(): `[bench_bin, "-m", mpath, "-o", "csv", "-r", reps, "-t", threads, "-ngl", ngl]` plus `-p`/`-n` (always both, 0 when unused) and `-fa 1`; "llama-bench writes its CSV to stdout with -o csv"; the avg_ts scrape `re.search(r'"?avg_ts"?[,=]\s*"?([\d.]+)', line)`; telemetry/energy_window.py `pick(b, "model_filename", "model", "model_type")` and `pick(b, "test", "n_prompt")`; tokens_of() docstring "Tokens actually processed in the window, counting repetitions and the warm-up run llama-bench performs before each measured set" with `per_rep * repetitions`.


**Evidence.** Correct: `-o <csv|json|jsonl|md|sql>` exists and is "printed to stdout"; `-r, --repetitions <n>` is repetitions (upstream default 5); `-p 0`/`-n 0` are accepted (get_cmd_params_instances skips them with `if (n_prompt == 0) { continue; }` / `if (n_gen == 0) { continue; }`); a warm-up does run (`--no-warmup` skips it); and `avg_ts`, `model_filename`, `n_prompt`, `n_gen` are all real CSV fields. Wrong in three ways that corrupt the deliverable. (a) There is NO `test` column in the CSV — get_fields() is build_commit … n_prompt, n_gen, n_depth, test_time, avg_ns, stddev_ns, avg_ts, stddev_ts; `test` only exists in the markdown renderer. energy_window.py's `pick(b, "test", "n_prompt")` therefore always silently falls through to n_prompt, so every decode row (n_prompt=0) is labelled "0". (b) `-p 512 -n 128` is TWO separate tests (pp512 and tg128), not one mixed test. For the `partial_offload_sweep`, `quantisation_sweep` and `largest_local_model` cases the CSV has two rows with different avg_ts; the regex on line 273 never matches (in real CSV `avg_ts` appears only in the header, followed by `"stddev_ts"`, not a number), so the DictReader fallback runs and keeps the LAST row — the tg128 number — while the power window spans both tests and tokens_of() charges 640 tokens. The published joules_per_token for those cases is prefill+decode energy divided by a token count that assumes one fused test, paired with a decode-only throughput. (c) The warm-up is real but is NOT counted: upstream runs `test_prompt(ctx, t.n_prompt, t.n_batch, t.n_threads)` — the FULL prompt — plus `test_gen(ctx, 1, ...)` before the timed repetitions. So a pp512 case at -r 3 processes 512*4 = 2048 tokens inside the power window while tokens_of() returns 512*3 = 1536, inflating every prefill joules_per_token / wh_per_mtok figure by 33% (and by 1/(r+1) generally) — the headline energy numbers of the study.


**Correction.** bench/run_matrix.py: (1) delete the `avg_ts` regex block and parse the CSV with csv.DictReader only, emitting ONE record per llama-bench row (key on the row's own `n_prompt`/`n_gen`) instead of one record per invocation with a single avg_ts; (2) never emit a case with both n_prompt and n_gen non-zero — in expand(), split it into a `-p N -n 0` run and a `-p 0 -n M` run so each power window covers exactly one phase (this also makes phase_of() honest and removes "mixed"); (3) fix tokens_of to account for the warm-up: `per_rep = (n_prompt or 0) + (n_gen or 0); warm = (n_prompt or 0) + (1 if n_gen else 0); return per_rep * (repetitions or 1) + warm`, or pass `--no-warmup` and document that the first repetition is cold. telemetry/energy_window.py: replace `pick(b, "test", "n_prompt")` with a phase derived from the row's n_prompt/n_gen (`"pp%s" % n_prompt` / `"tg%s" % n_gen`) since llama-bench's CSV has no `test` column. docs/10_measurement_protocol.md: add `n_prompt`/`n_gen` to the column table and state that reported energy includes llama-bench's warm-up pass.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/README.md ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/llama-bench/llama-bench.cpp (get_cmd_params_instances, warmup block, get_fields) ; bench/run_matrix.py:131-146,271-283,397-402 ; telemetry/energy_window.py:184-187 ; docs/10_measurement_protocol.md:100


**Files.** `bench/run_matrix.py, bench/matrix.json, telemetry/energy_window.py, docs/10_measurement_protocol.md`


### [CONFIRMED] facts:llamacpp-rpc

**Claim.** Security posture. scripts/40_run_rpc_server.sh: "SECURITY: ggml's RPC server executes whatever the client sends it and has no authentication. Bind it to the testbed's isolated LAN only — never to an interface that is reachable from the internet." systemd/heteroserve-rpc.service: "the ggml RPC protocol is unauthenticated and the server executes whatever graph the client sends. Only enable this on the isolated testbed LAN."


**Evidence.** Upstream tools/rpc/README.md states it verbatim: "This example and the RPC backend are currently in a proof-of-concept development stage. As such, the functionality is fragile and insecure. **Never run the RPC server on an open network or in a sensitive environment!**" The server's own startup also warns "This is an experimental feature and is not secure!". There is no authentication, no TLS and no capability check in the protocol, and the deserializer has a documented history of memory-safety issues reachable by an unauthenticated client (public RCE write-up against the llama.cpp RPC server). The kit's wording is if anything understated — the risk is remote code execution on the Nano, not merely resource abuse. One thing worth stating alongside it: upstream's own default is `--host 127.0.0.1`, whereas config/node.env.example ships `RPC_BIND=0.0.0.0`, which is necessary for the testbed but means the warning is load-bearing rather than theoretical, and the systemd unit enables the worker at boot on every interface the Nano has (including Wi-Fi, which the kit's own wired-vs-Wi-Fi comparison requires be configured).


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/README.md ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/rpc/rpc-server.cpp (startup warning, default host 127.0.0.1) ; https://retr0.blog/blog/llama-rpc-rce ; scripts/40_run_rpc_server.sh:8-10 ; systemd/heteroserve-rpc.service


**Files.** `scripts/40_run_rpc_server.sh, systemd/heteroserve-rpc.service, config/node.env.example`


### [PARTLY-WRONG] facts:jetpack-cuda

**Claim.** Claim 1 — "Stock JetPack 4.6.x ships CUDA 10.2, and 4.6.x is the LAST JetPack line supporting the original Jetson Nano (JetPack 5/6 are Orin-only)." Stated in docs/00_flash_jetpack.md:5-11 ("JetPack 4.6.x is the **last** release line for the original Nano — JetPack 5/6 are Orin-only"), README.md:100, config/node.env.example:19-20.


**Evidence.** The two load-bearing parts are correct: JetPack 4.6 = L4T 32.7.x = Ubuntu 18.04 aarch64 + CUDA 10.2, and the original Nano (T210) never moved past the JetPack 4.6.x line — NVIDIA's EOL announcement confirms 4.6.6 / L4T 32.7.6 is the final release for Nano/TX1/TX2, so 4.6.x is indeed terminal for this board. The parenthetical is wrong: JetPack 5.x is not Orin-only, it supports Jetson AGX Xavier and Xavier NX as well as the Orin family (JetPack 6 is the Orin/Thor-only line). Minor, but it is a factual claim in the doc a reader may repeat in the thesis. Also worth pinning down in the doc: the kit says 4.6.x without saying which; 4.6.6 is the newest and all of 4.6.0–4.6.6 ship CUDA 10.2, so the range statement holds.


**Correction.** docs/00_flash_jetpack.md:9-10 — replace "JetPack 4.6.x is the **last** release line for the original Nano — JetPack 5/6 are Orin-only." with: "JetPack 4.6.x is the **last** release line for the original Nano (T210). NVIDIA declared JetPack 4 end-of-life with 4.6.6 / L4T 32.7.6; JetPack 5 supports only Xavier and Orin, JetPack 6 only Orin and Thor. Every 4.6.x release ships CUDA 10.2."


**Sources.** https://developer.nvidia.com/embedded/jetpack-sdk-46 ; https://forums.developer.nvidia.com/t/announcing-end-of-life-for-nvidia-jetpack-4-with-the-release-of-jetpack-4-6-6/314300 ; https://developer.nvidia.com/embedded/jetpack-archive ; D:/KMITL/Final/HeteroServe/Jetson/docs/00_flash_jetpack.md:5-11


**Files.** `docs/00_flash_jetpack.md`


### [PARTLY-WRONG] facts:jetpack-cuda

**Claim.** Claim 2 — "Current llama.cpp requires CUDA >= 11 and will NOT compile its CUDA backend against 10.2", plus the recovery advice "Walk the tags backwards (b3xxx era and earlier) until one configures". Stated in scripts/20_build_llama_cpp.sh:52-60, README.md:100-105, config/node.env.example:19-26, docs/20_troubleshooting.md:3-20.


**Evidence.** (a) The conclusion is right but there is no *stated* minimum: llama.cpp's docs/build.md gives no minimum CUDA version and ggml/src/ggml-cuda/vendors/cuda.h contains no `#error` guard — it silently degrades below CUDART 11.2. The de-facto floor is CUDA 11 because ggml/CMakeLists.txt:291 now sets `CMAKE_CXX_STANDARD 17` unconditionally (CMake propagates this to CMAKE_CUDA_STANDARD) and nvcc 10.2 caps at C++14 — C++17 device support arrived in CUDA 11.0; master additionally uses `nv_bfloat16` (CUDA 11+) and `__builtin_assume` in the fattn kernels. So "CUDA >= 11" is accurate as an observation, not as a documented requirement.
(b) **The script will never reach a CUDA error.** scripts/20_build_llama_cpp.sh:103 passes `-DLLAMA_CUBLAS=ON`. Every llama.cpp ref since the mid-2024 LLAMA_*→GGML_* rename contains `llama_option_depr(FATAL_ERROR LLAMA_CUBLAS GGML_CUDA)` in the root CMakeLists.txt (verified present at b4000:107, b5050:112, and master:186 — absent at b3000). Configure aborts with "LLAMA_CUBLAS is deprecated, use GGML_CUDA instead" before nvcc is ever invoked. Given the script's own scary CUDA-10.2 banner printed immediately above, the operator will read this as "CUDA 10.2 too old" and fall back to `--cpu-only` — the thesis then records "GPU build impossible on this board" for a reason that is entirely the kit's own flag. Note `-DLLAMA_RPC=ON` and `-DLLAMA_CUDA=ON` are only WARNING-level, so only LLAMA_CUBLAS is fatal.
(c) "Walk the tags backwards until one configures" does not converge. The ggml RPC backend was merged 2024-05-14 (PR #6829); any ref old enough to build cleanly against CUDA 10.2 without patches predates it and has no rpc-server, so the build then dies at scripts/20_build_llama_cpp.sh:133 ("rpc-server was not built"). The only two publicly reproduced Nano/CUDA-10.2 pins are: FlorSanders' gist → commit a33e6a0 / b2275 (Feb 2024) built via the old Makefile with `LLAMA_CUBLAS=1` — pre-RPC, and its commenters report `CUDA error: no kernel image is available for execution` because the gist uses sm_62; and kreier/llama.cpp-jetson → **commit 23106f9, build b5050 (April 2025)** — which *does* have RPC, but requires six source patches (CMAKE_CUDA_ARCHITECTURES limit, an extra linker line in ggml/CMakeLists.txt, dropping `constexpr` at ggml/src/ggml-cuda/common.cuh:455, commenting `__builtin_assume` in three fattn files, and replacing `nv_bfloat16` with `half`), plus `-DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true -DGGML_NATIVE=off -DGGML_CPU_ARM_ARCH=armv8-a`. There is **no reliable public pin that builds unpatched with both CUDA 10.2 and the RPC backend.**


**Correction.** 1) scripts/20_build_llama_cpp.sh:103 — change
    `-DGGML_CUDA=ON -DLLAMA_CUDA=ON -DLLAMA_CUBLAS=ON`
  to
    `-DGGML_CUDA=ON`
  (`LLAMA_CUBLAS` is a hard FATAL_ERROR on every ref that has the RPC backend; `LLAMA_CUDA` is redundant and only warns.)
2) scripts/20_build_llama_cpp.sh:93 — same reasoning, drop the legacy spelling: `-DGGML_RPC=ON` only, and note that refs older than 2024-05-14 have no rpc-server at all.
3) scripts/20_build_llama_cpp.sh:56-60 — replace "Walk the tags backwards (b3xxx era and earlier) until one configures" with: "Do NOT walk the tags backwards: the ggml RPC backend was merged 2024-05-14, so every ref old enough to build unpatched against CUDA 10.2 has no rpc-server and is useless to this cluster. The only reported CUDA-10.2 + RPC build is b5050 (commit 23106f9, April 2025) and it needs six source patches — see https://github.com/kreier/llama.cpp-jetson. Add `-DCMAKE_CUDA_STANDARD=14 -DCMAKE_CUDA_STANDARD_REQUIRED=true` for any CUDA 10.2 build."
4) config/node.env.example:26 — `LLAMA_CPP_REF=b5050` with a comment that the patch set from kreier/llama.cpp-jetson must be applied, and add `CUDA_STANDARD=14`.
5) README.md:100-105 and docs/20_troubleshooting.md:3-20 — replace "pin an older llama.cpp ref" with the b5050+patches statement.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt (line 186) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/CMakeLists.txt (line 107) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b5050/CMakeLists.txt (line 112) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/CMakeLists.txt (line 291) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/vendors/cuda.h ; https://github.com/ggml-org/llama.cpp/pull/6829 ; https://github.com/kreier/llama.cpp-jetson ; https://kreier.github.io/llama.cpp-jetson/ ; https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f ; D:/KMITL/Final/HeteroServe/Jetson/scripts/20_build_llama_cpp.sh:103,133


**Files.** `scripts/20_build_llama_cpp.sh, config/node.env.example, README.md, docs/20_troubleshooting.md`


### [CONFIRMED] facts:jetpack-cuda

**Claim.** Claim 3 — "CUDA 10.2 refuses host GCC newer than 8" and gcc-8 is installable from the stock archive. Stated in scripts/10_install_toolchain.sh:3,8-13, config/node.env.example:31 (`HOST_CXX=g++-8`), scripts/00_system_prep.sh apt line, docs/20_troubleshooting.md:26.


**Evidence.** CUDA 10.2's /usr/include/crt/host_config.h contains `#if __GNUC__ > 8 / #error -- unsupported GNU version! gcc versions later than 8 are not supported!`, so nvcc 10.2 rejects gcc 9+ outright (only `-allow-unsupported-compiler`, which does not exist in 10.2, or editing host_config.h bypasses it). NVIDIA's CUDA 10.2 Linux support matrix lists GCC 8.x as the maximum for Ubuntu 18.04; the practical range is roughly GCC 5–8. JetPack 4.6's default is GCC 7.5, which also works, so gcc-8 is an upgrade rather than a strict necessity — but it is the newest supported and is what the kit picks.
Availability confirmed directly against the Ubuntu archive: gcc-8 8.4.0-1ubuntu1~18.04 is Published for bionic/arm64 in **universe** (bionic-updates and bionic-security pockets); the release pocket carries 8-20180414-1ubuntu2. The stock JetPack 4.6 sources.list enables universe on ports.ubuntu.com, so `apt-get install gcc-8 g++-8` in scripts/00_system_prep.sh succeeds without adding a PPA. (The two well-known Nano guides build GCC 8.5 from source, ~3 h — that is a preference for 8.5 specifically, not a sign that apt gcc-8 is unavailable; 8.4 satisfies nvcc 10.2 and keeps libstdc++ 6.0.25, the same ABI as the stock 7.5.) The script's wiring is also correct: `HOST_CC="${HOST_CXX/g++/gcc}"` yields gcc-8, and it sets `-DCMAKE_CUDA_HOST_COMPILER`, `-DCMAKE_C_COMPILER`, `-DCMAKE_CXX_COMPILER` consistently.


**Sources.** https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html ; https://gist.github.com/ax3l/9489132 ; https://forums.developer.nvidia.com/t/gcc-support-in-cuda-10/65331 ; https://launchpad.net/ubuntu/bionic/+package/gcc-8 ; Launchpad API getPublishedBinaries binary_name=gcc-8 distro_arch_series=bionic/arm64 → 8.4.0-1ubuntu1~18.04, component=universe ; D:/KMITL/Final/HeteroServe/Jetson/scripts/10_install_toolchain.sh:8-13


**Files.** `scripts/10_install_toolchain.sh, scripts/00_system_prep.sh, config/node.env.example`


### [PARTLY-WRONG] facts:jetpack-cuda

**Claim.** Claim 4 — llama.cpp's CMake floor is above the 3.10 Ubuntu 18.04 ships, and `pip install cmake` supplies a working aarch64 wheel. Stated in scripts/10_install_toolchain.sh:2-3,19-25 (gate at 3.14.0, install `cmake>=3.22,<4` "aarch64 wheels exist"), docs/20_troubleshooting.md:41-43.


**Evidence.** The pip half is confirmed. PyPI's `cmake` package publishes `manylinux2014_aarch64` / `manylinux_2_17_aarch64` wheels continuously from 3.18.2 onward; Ubuntu 18.04's glibc 2.27 > 2.17, so they run. One detail the script gets right by accident and should keep: bionic's system pip is 9.0.1, which predates manylinux2014 tag support and would fall back to building CMake from an sdist — scripts/10_install_toolchain.sh:21 upgrades pip first (Python 3.6.9 tops out at pip 21.3.1, which does understand the tag), so the wheel is found. Resolution detail worth knowing: on Python 3.6 the newest satisfying release is **3.28.3** — every wheel from 3.29.0.1 up declares `Requires-Python >=3.7` (and 4.1+ needs >=3.8). 3.28.3 satisfies `>=3.22,<4`, so the pin works, and capping `<4` is correct.
The version floor is understated. Root CMakeLists.txt and ggml/CMakeLists.txt say `cmake_minimum_required(VERSION 3.14...3.28)`, but **ggml/src/ggml-cuda/CMakeLists.txt line 1 says `cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES`**. The toolchain script's gate is `3.14.0`, so a board that already has CMake 3.14–3.17 (e.g. someone who installed from a third-party PPA or a leftover /usr/local build) is declared "cmake new enough", the pip install is skipped, and the CUDA build then fails deep inside `add_subdirectory(ggml-cuda)` with a minimum-version error that looks nothing like a toolchain problem. Stock 3.10 happens to trip the gate anyway, so this only bites a non-pristine board.


**Correction.** scripts/10_install_toolchain.sh:19 — change the gate from 3.14.0 to 3.18.0:
  `if [ "$(printf '%s\n3.18.0\n' "${CMV:-0}" | sort -V | head -1)" != "3.18.0" ]; then`
and update the comment at line 2-3 to: "llama.cpp's root floor is CMake 3.14, but ggml/src/ggml-cuda/CMakeLists.txt requires 3.18 for CMAKE_CUDA_ARCHITECTURES — 3.18 is the real floor for a CUDA build. Ubuntu 18.04 ships 3.10."
Optionally note at line 22 that on Python 3.6.9 the resolver lands on cmake 3.28.3 (3.29+ require Python >= 3.7), which is expected and fine.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/CMakeLists.txt (line 1) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/CMakeLists.txt (line 1) ; https://pypi.org/pypi/cmake/json (aarch64 wheel/requires_python inventory, 3.18.2 … 4.4.3) ; D:/KMITL/Final/HeteroServe/Jetson/scripts/10_install_toolchain.sh:19-25


**Files.** `scripts/10_install_toolchain.sh, docs/20_troubleshooting.md`


### [PARTLY-WRONG] facts:jetpack-cuda

**Claim.** Claim 5 — "`CMAKE_CUDA_ARCHITECTURES=53` is the right value for the Nano's Maxwell GM20B; and GGML_CUDA_F16 is meaningful on sm_53." Stated in config/node.env.example:29, scripts/20_build_llama_cpp.sh:104-107 (`-DGGML_CUDA_F16=ON -DLLAMA_CUDA_F16=ON` with the comment "FP16 accumulation is the one arch-specific win available on sm_53"), README.md:17,24-27.


**Evidence.** The architecture value is correct: Tegra X1 / GM20B is compute capability 5.3, and `-DCMAKE_CUDA_ARCHITECTURES=53` is exactly what Nano builders use. (Getting it wrong is a real failure mode — the FlorSanders gist's `sm_62` produces `CUDA error: no kernel image is available for execution` at runtime, reported by several commenters.)
The F16 half is wrong three times over:
1. **The option no longer exists.** `GGML_CUDA_F16` was deleted from ggml by PR #10318 ("CUDA: remove DMMV, consolidate F16 mult mat vec", Nov 2024) — grep of master's ggml/CMakeLists.txt shows no `option(GGML_CUDA_F16 ...)`. On any ref newer than ~b4100, including the b5050 pin that actually works on this board, `-DGGML_CUDA_F16=ON` and `-DLLAMA_CUDA_F16=ON` are unknown cache entries, and scripts/20_build_llama_cpp.sh:86 passes `--no-warn-unused-cli`, so CMake does not even warn. The build_info.json provenance record will imply an FP16 build that does not exist.
2. **ggml does not consider sm_53 FP16-capable.** ggml/src/ggml-cuda/common.cuh:267-269 defines `FP16_AVAILABLE` only for `__CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL` (600); sm_53 = 530 is below it, so `warp_reduce_sum(half2)` compiles to `NO_DEVICE_CODE` and the runtime helpers `fp16_available(cc)` / `fast_fp16_available(cc)` return false for 530. This guard is identical at b4000 (common.cuh:123-129), i.e. it also held on every ref where the flag still existed. The hardware does have native FP16; ggml simply gates on Pascal and never uses it on 5.3.
3. **It would not touch this kit's models anyway.** At b4000, ggml/src/ggml-cuda/dmmv.cu's `#ifdef GGML_CUDA_F16` block converts src1 to half only for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/F16; the K-quant branches are passed `src1_ddf_i` (float) unconditionally. Every model in config/node.env.example MODELS is `q4_k_m`, so the flag is a no-op for the entire measurement matrix even on a ref that honours it.
Consequence for the thesis: README.md:24-27's framing ("FP16 yes, INT8 no … FP16 accumulation is the one arch-specific win available on sm_53") sets up an expectation the software cannot deliver, and a measured null result would be misattributed to the hardware instead of to ggml's Pascal gate.


**Correction.** 1) scripts/20_build_llama_cpp.sh:105-107 — delete the two F16 flags and the comment; keep only `-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"`. Replace the comment with: "# sm_53 (Tegra X1 GM20B). Note ggml gates all FP16 kernels on __CUDA_ARCH__ >= 600 (Pascal) in ggml-cuda/common.cuh, so sm_53 gets the FP32 paths regardless of what the hardware can do; GGML_CUDA_F16 was removed from ggml in PR #10318 (Nov 2024) and is silently ignored here."
2) README.md:24-27 — replace "FP16 accumulation is the one arch-specific win available on sm_53" with: "Maxwell GM20B has native FP16 units, but ggml's CUDA backend enables its FP16 kernels only from compute capability 6.0 (Pascal) upward, so sm_53 runs the FP32 paths. That is a software limit, not a hardware one, and it should be stated as such next to any FP16-related result."
3) Keep `CUDA_ARCH=53` in config/node.env.example — it is correct.


**Sources.** https://github.com/ggml-org/llama.cpp/pull/10318 ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/CMakeLists.txt (options GGML_CUDA_* — no F16) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/common.cuh (lines 267-273, 308-323) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/ggml/src/ggml-cuda/common.cuh (lines 123-129) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/b4000/ggml/src/ggml-cuda/dmmv.cu (lines 627-680) ; https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f ; D:/KMITL/Final/HeteroServe/Jetson/scripts/20_build_llama_cpp.sh:104-107


**Files.** `scripts/20_build_llama_cpp.sh, README.md, config/node.env.example`


### [PARTLY-WRONG] facts:jetpack-cuda

**Claim.** Claim 6 — "GGML_CUDA_ENABLE_UNIFIED_MEMORY is a real env var" that "lets an oversized partition spill to swap instead of dying". Stated in config/node.env.example:42-45, scripts/40_run_rpc_server.sh:37-41, docs/20_troubleshooting.md:63-66.


**Evidence.** The variable is real and the spelling is right. llama.cpp's docs/build.md:285-287 documents `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1`, and ggml/src/ggml-cuda/ggml-cuda.cu:140-144 implements it as: if the env var is set, `ggml_cuda_device_malloc` calls `cudaMallocManaged` instead of `cudaMalloc`.
The described *effect* is wrong on this board. Upstream's wording is "allows swapping to system RAM instead of crashing when the GPU VRAM is exhausted" — that is discrete-GPU VRAM overflowing into host RAM, not host RAM overflowing into a disk swapfile. On a Jetson Nano:
 • CPU and GPU already share one 4 GB LPDDR4 pool (the kit says so itself at README.md:31-33), so there is no separate VRAM to overflow — `cudaMallocManaged` and `cudaMalloc` draw from the same physical memory.
 • sm_53 is pre-Pascal: no page-migration engine, no 49-bit addressing, no on-demand faulting. On pre-Pascal hardware total managed allocation is capped at the device memory size and cannot be oversubscribed; the driver migrates all managed pages to the device before each kernel launch.
 • CUDA managed memory is not paged out to a swapfile by the kernel, so /var/swapfile from 00_system_prep.sh cannot absorb it.
Failure scenario: the orchestrator hands this node a partition above `RPC_MEM_MB`, the operator follows docs/20_troubleshooting.md:63-66 and sets `CUDA_UNIFIED_MEMORY=1`, and the allocation fails again (or succeeds no differently) — while scripts/40_run_rpc_server.sh:41 has already stamped the run "unified memory ON — not comparable to the default condition", so a run that is physically identical to the default gets labelled as a separate experimental condition. If any over-commit rows do land in the dataset they will be attributed to a mechanism that was never active.


**Correction.** 1) config/node.env.example:42-44 — replace the comment block with: "# GGML_CUDA_ENABLE_UNIFIED_MEMORY makes ggml allocate with cudaMallocManaged.
# Upstream's purpose is letting a discrete GPU spill out of VRAM into host RAM.
# On the Nano CPU and GPU already share one 4 GB pool, and sm_53 is pre-Pascal:
# no page migration, managed memory capped at device memory, never paged to the
# swapfile. So this does NOT rescue an over-committed partition here — lower
# RPC_MEM_MB instead. Left in place only so the flag can be measured, not used."
2) docs/20_troubleshooting.md:63-66 — delete the `CUDA_UNIFIED_MEMORY=1` recommendation; the only correct remedy on this board is lowering `RPC_MEM_MB` (or `-ngl`).
3) scripts/40_run_rpc_server.sh:38-39 — change the comment to "# cudaMallocManaged instead of cudaMalloc. On sm_53 this changes the allocator, not the capacity — it does not add headroom." and keep the label warning.


**Sources.** https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/build.md (lines 285-287) ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/ggml-cuda/ggml-cuda.cu (lines 140-144) ; https://developer.nvidia.com/blog/unified-memory-cuda-beginners/ ; https://forums.developer.nvidia.com/t/beyond-gpu-memory-limits-with-unified-memory-on-pascal/148516 ; D:/KMITL/Final/HeteroServe/Jetson/config/node.env.example:42-45 ; D:/KMITL/Final/HeteroServe/Jetson/scripts/40_run_rpc_server.sh:37-41


**Files.** `config/node.env.example, scripts/40_run_rpc_server.sh, docs/20_troubleshooting.md`


### [PARTLY-WRONG] facts:jetpack-cuda

**Claim.** Claim 7 — "this takes roughly 40-70 min on a cold SD-card build" and "-j4 with nvcc on 4 GB reliably invokes the OOM killer; -j2 plus the swapfile is the combination that finishes." Stated in scripts/20_build_llama_cpp.sh:122-124, scripts/10_install_toolchain.sh:36-37, config/node.env.example:30 (`BUILD_JOBS=2`), docs/20_troubleshooting.md:29-35.


**Evidence.** The -j2 + swap judgement is sound in direction; the time figure is optimistic and the parallelism is stated with more confidence than the evidence supports.
The only published end-to-end Nano CUDA timing is kreier/llama.cpp-jetson: **~85 minutes** for the llama.cpp compile on SD ("60 minutes on faster storage") — and that is a *serial* `cmake --build build --config Release` (no -j) on b5050 with the architecture list already cut to two virtual archs. A -j2 build that also compiles the RPC backend and the fattn/mmq template instantiations will not reliably come in under 70 minutes on an SD card; 40 min is not achievable on the configuration this kit describes. Understating it matters operationally because README.md:63 tells the operator to run long jobs in tmux but the build instruction carries a 40-70 min expectation, and because an operator who thinks the build has hung at minute 75 may kill it.
On parallelism: the OOM concern is real — several fattn/mmq translation units peak well above 1.5 GB RSS under nvcc, so two concurrent nvcc processes can exceed the ~3.2 GB of usable RAM on a 4 GB Nano, which is why the kit's own docs/20_troubleshooting.md:35 already says to fall back to `-j 1`. But the swapfile is on the SD card, so once -j2 starts swapping the build gets *slower* than -j1, not merely safer. "-j2 is the combination that finishes" is asserted as tested fact in a kit that has never run on hardware; it should be phrased as the starting point with -j1 as the documented fallback. Secondary point, also unverified-on-hardware: the script passes `-DGGML_NATIVE=ON` (line 94) while both independent Nano guides had to disable native CPU flags (kreier uses `-DGGML_NATIVE=off -DGGML_CPU_ARM_ARCH=armv8-a`; the FlorSanders gist deletes `MK_CXXFLAGS += -mcpu=native` from the Makefile) — worth trying `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a` before concluding the toolchain is broken.


**Correction.** 1) scripts/20_build_llama_cpp.sh:122-124 — replace with: "# nvcc peaks well over 1.5 GB RSS on the fattn/mmq template instances, so -j4 on
# 4 GB invites the OOM killer. Start at -j2 with the swapfile from 00_system_prep.sh;
# if the build is killed, drop to -j 1 (the swapfile is on SD, so a swapping -j2 is
# slower than a resident -j1)."
and change the info line to: `info "budget ~1.5-2 h on a cold SD-card build (the one published Nano CUDA build reports ~85 min serially); run this under tmux"`.
2) scripts/10_install_toolchain.sh:36-37 — change "~40-70 min" to "~1.5-2 h".
3) README.md:56 — the `make build` line should carry the same 1.5-2 h expectation, and the tmux instruction at README.md:63 should cover the build, not just the sweeps.
4) scripts/20_build_llama_cpp.sh:94 — add a fallback note: if the CPU backend fails to compile under gcc-8 on this A57, retry with `-DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8-a` (both published Nano guides needed this).


**Sources.** https://kreier.github.io/llama.cpp-jetson/ ; https://github.com/kreier/llama.cpp-jetson ; https://gist.github.com/FlorSanders/2cf043f7161f52aa4b18fb3a1ab6022f ; D:/KMITL/Final/HeteroServe/Jetson/scripts/20_build_llama_cpp.sh:122-125 ; D:/KMITL/Final/HeteroServe/Jetson/docs/20_troubleshooting.md:29-35


**Files.** `scripts/20_build_llama_cpp.sh, scripts/10_install_toolchain.sh, README.md, docs/20_troubleshooting.md`



## Unverified findings (30)

Sorted by claimed severity. Duplicates across lenses are kept: two lenses
reaching the same conclusion independently is itself weak evidence for it.


| # | sev | file:line | lens | title |
|---|---|---|---|---|
| 1 | critical | `bench/run_matrix.py:247` | audit:python | llama-bench stderr is merged into stdout, so the captured "CSV" has log lines before the header and avg_ts is never parsed |
| 2 | critical | `bench/run_matrix.py:244` | audit:methodology | Energy window spans the whole llama-bench process, so model load + CUDA init are billed to the tokens |
| 3 | critical | `bench/run_matrix.py:272` | audit:methodology | For mixed prefill+decode cases the row pairs decode-only throughput with prefill+decode energy |
| 4 | critical | `scripts/30_fetch_models.sh:32` | audit:ops | A truncated GGUF is treated as complete and its bad hash is written into SHA256SUMS as ground truth |
| 5 | critical | `scripts/30_fetch_models.sh:32` | audit:shell | A truncated .gguf is treated as "already present" and its wrong sha256 is written into SHA256SUMS |
| 6 | critical | `scripts/60_collect_node_facts.sh:32` | audit:shell | `timeout 6 tegrastats` under `set -o pipefail` kills the script before any JSON is printed |
| 7 | critical | `telemetry/energy_window.py:185` | audit:python | pick(b, "n_prompt", "n_gen", ...) returns the literal "0" on decode rows, so decode windows silently get no energy-per-token |
| 8 | high | `bench/run_matrix.py:401` | audit:python | tokens_of() omits llama-bench's warm-up run its own docstring claims to count, inflating J/token by ~33% at the default -r 3 |
| 9 | high | `bench/run_matrix.py:398` | audit:methodology | tokens_of() omits llama-bench's warm-up pass despite its docstring claiming to count it |
| 10 | high | `bench/run_matrix.py:290` | audit:methodology | Fixed 20 s cooldown plus a within-window temp_rise_c check cannot detect the thermal drift it is meant to catch |
| 11 | high | `docs/10_measurement_protocol.md:33` | audit:ops | The external inline power meter is declared ground truth but nothing in the kit reads, logs, or collects it |
| 12 | high | `scripts/00_system_prep.sh:38` | audit:shell | An interrupted swapfile allocation wedges 00_system_prep.sh permanently, despite the "idempotent" claim |
| 13 | high | `scripts/20_build_llama_cpp.sh:108` | audit:shell | Missing g++-8 aborts the build with exit 1 and no error message at all |
| 14 | high | `scripts/50_bench_device.sh:22` | audit:platform | Pre-flight thermal gate reads the PMIC-Die dummy zone and always reports 100 C |
| 15 | high | `scripts/60_collect_node_facts.sh:79` | audit:platform | POM_5V_IN is described as module-only power; on the Nano devkit it is the whole-board 5 V input |
| 16 | high | `scripts/60_collect_node_facts.sh:32` | audit:platform | The idle-baseline tegrastats capture kills the whole facts script under set -e/pipefail |
| 17 | high | `scripts/70_cluster_smoke.sh:61` | audit:ops | No way to capture a phase-marked power window on this node during an orchestrator-driven split run |
| 18 | high | `systemd/heteroserve-telemetry.service:23` | audit:ops | The always-on telemetry unit overwrites its one CSV on every restart and records no phase markers |
| 19 | high | `telemetry/energy_window.py:59` | audit:python | datetime.fromisoformat does not exist in Python 3.6.9, so --from/--to always raises AttributeError |
| 20 | medium | `bench/run_matrix.py:272` | audit:python | Mixed prompt+gen cases keep only the last (decode) avg_ts while tokens_of sums prompt and gen tokens, producing a self-contradictory row |
| 21 | medium | `bench/run_matrix.py:214` | audit:python | A telemetry logger that dies at startup (second tegrastats instance) is never detected; the sweep completes with silently empty energy columns |
| 22 | medium | `bench/run_matrix.py:342` | audit:methodology | idle_floor() pools the cold pre-sweep baseline with every hot cooldown, and disagrees with the idle floor the docs tell you to use |
| 23 | medium | `bench/run_matrix.py:398` | audit:ops | Every joules-per-token figure includes model-load and warm-up energy in the numerator but not in the token count |
| 24 | medium | `scripts/50_bench_device.sh:16` | audit:shell | `sudo nvpmodel -q \| grep` under pipefail aborts the whole sweep at pre-flight when sudo cannot prompt |
| 25 | medium | `scripts/50_bench_device.sh:29` | audit:shell | `pgrep -x rpc-server` misses the `llama-rpc-server` binary name the toolkit itself supports |
| 26 | medium | `scripts/60_collect_node_facts.sh:14` | audit:platform | GPU devfreq sysfs path does not exist on the Nano; gpu.max_freq_hz is always 0 |
| 27 | medium | `scripts/60_collect_node_facts.sh:13` | audit:platform | cpu.max_freq_khz reads cpuinfo_max_freq, which ignores the nvpmodel cap |
| 28 | medium | `systemd/heteroserve-rpc.service:20` | audit:ops | ExecStartPre's `sudo jetson_clocks` always fails under systemd and is swallowed, so the boot-time worker serves at unlocked clocks |
| 29 | medium | `telemetry/tegrastats_logger.py:148` | audit:methodology | 5 Hz point sampling beats against the token cadence, and the INA integral is never validated against the inline meter |
| 30 | medium | `telemetry/tegrastats_logger.py:85` | audit:platform | Parked cores counted as 0% halves cpu_util_pct in the 5 W power mode |


## Detail


### 1. [CRITICAL] llama-bench stderr is merged into stdout, so the captured "CSV" has log lines before the header and avg_ts is never parsed

> **Status:** FIXED 2026-09-12

`bench/run_matrix.py:247` — lens `audit:python`


**Detail.** `subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, ...)` merges llama-bench's stderr into the stream that is then (a) written verbatim to `case_csv` (line 268) and (b) fed to `csv.DictReader((out.stdout or "").splitlines())` (line 277). With BUILD_CUDA=1 (the default in config/node.env.example) the CUDA backend prints its banner on stderr *before* main() silences the llama log, e.g. `ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no` / `ggml_init_cublas: found 1 CUDA devices:` / `  Device 0: NVIDIA Tegra X1, compute capability 5.3`. DictReader therefore takes the first banner line as the header row, `fieldnames` becomes that one junk string, and `r.get("avg_ts")` is None for every row. The regex path above it cannot save this: llama-bench's CSV writer quotes every field, so the literal token `avg_ts` appears only in the header line `"avg_ts","stddev_ts"`, where `"?avg_ts"?[,=]\s*"?([\d.]+)` cannot match (the next field is `stddev_ts`, not digits). So the regex never fires on a real llama-bench CSV either, and `tps` stays None on every case. The same pollution makes each `<base>_NNN.csv` unparseable by energy_window.load_bench, and 90_collect_results.sh bundles those broken files as the raw evidence.


**Failure scenario.** Run `./scripts/50_bench_device.sh --tag wired-gbe` on the Nano with the CUDA build. Every case prints `    0.00 t/s, 41.3 s`, the `avg_ts` column in `<node>_<stamp>_wired-gbe_bench_energy.csv` is empty for all ~26 rows, and every `<base>_001.csv` starts with `ggml_init_cublas:` lines instead of a CSV header. Two to three hours of sweep produce joules with no throughput attached — the exact pairing the driver exists to create.


**Fix.** Capture the streams separately: `out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, check=False)`; write `out.stderr` to `logfh` and write only `out.stdout` to `case_csv`. Additionally harden the parse against any stdout preamble: before constructing the DictReader, drop lines until the first one that starts with `"build_commit"` (or, generically, the first line containing `avg_ts`), e.g. `lines = (out.stdout or "").splitlines(); start = next((i for i, l in enumerate(lines) if "avg_ts" in l), 0); rows = list(csv.DictReader(lines[start:]))`, and delete the dead regex loop at lines 271-275.


### 2. [CRITICAL] Energy window spans the whole llama-bench process, so model load + CUDA init are billed to the tokens

> **Status:** FIXED 2026-09-12

`bench/run_matrix.py:244` — lens `audit:methodology`


**Detail.** `t_start = time.time(); set_marker(label)` at line 244-251 opens the power window immediately before `subprocess.run(cmd, ...)` and closes it when the process exits. Everything llama-bench does outside token computation is therefore inside the joules figure: Python/loader startup, ggml CUDA context creation (CUDA 10.2 on Maxwell is seconds, not milliseconds), reading the GGUF off the SD card, and for `-ngl > 0` the host->device weight upload. `avg_ts` from llama-bench, by contrast, is timed *only* around the measured repetitions. So every row pairs a compute-only throughput with a load-inclusive energy, and `joules_per_token` / `wh_per_mtok` in the headline table are not energy per token of inference at all. The load phase is not idle-power either (SD read + memcpy is several watts), so the idle subtraction does not remove it. Worse, the overhead is not constant along the axes being compared: `-ngl 99` runs pay a CUDA context + weight-upload cost that `-ngl 0` runs do not, and `qwen2.5-0.5b-q8_0` (~530 MB) reads ~35% more bytes off the card than `q4_k_m` (~400 MB), so both the GPU-offload conclusion and the quantisation conclusion are contaminated by I/O and init that has nothing to do with the model math.


**Failure scenario.** `cpu_decode_sweep`, n_gen=32, repetitions=3 on a Nano at ~10 tok/s: 96 measured tokens = ~10 s of compute. The first invocation that touches the 400 MB q4_k_m file reads it cold from SD at 25-45 MB/s = 9-16 s, so the window is ~20-26 s and `joules_per_token` is inflated ~2-2.5x. The next run of the same model is page-cache warm (~2 s) and is inflated only ~1.2x. Nothing in the CSV records which happened, so the same case measured twice differs by ~2x for reasons unrelated to inference, and the ngl=0 vs ngl=99 decode comparison (the study's central GPU-offload claim) carries a fixed multi-second CUDA-init penalty on the GPU side only.


**Fix.** Stop treating the process lifetime as the measurement window. Concretely: (a) before each (model, n_gpu_layers) group, run one calibration invocation of the same binary/model with minimal work (`-p 1 -n 0 -r 1`) inside its own marker window and store `load_joules` / `load_s`; (b) add columns `compute_joules = joules - load_joules` and `compute_joules_per_token`, and make the joined CSV report those as the primary energy figures with raw `joules` kept alongside; (c) also parse `avg_ns`/`stddev_ns` out of each llama-bench CSV row into the joined row so the compute duration is on the record and `compute_joules / (sum of avg_ns * reps)` can be sanity-checked against `mean_mw`; (d) record a `page_cache_cold` flag (or force a consistent state by reading the GGUF with `cat > /dev/null` before every case) so load cost is at least uniform across the axis being compared.


### 3. [CRITICAL] For mixed prefill+decode cases the row pairs decode-only throughput with prefill+decode energy

> **Status:** FIXED 2026-09-13 — matrix split into single-phase cases, one record per bench row, `per_token_status` on every row. `expand()` refuses a mixed case by name at `--dry-run` time, and the driver refuses any window whose returned phase does not match the one it asked for. Then adversarially verified by five independent lenses, which found one further CRITICAL (a `-pg` row misreported as decode) and six HIGHs — all since fixed; see `docs/42_finding3_verification.md`. Regression: `bench/fixtures/run_regression.sh`, 30 passed. The cited line numbers are PRE-FIX; do not renumber them.

`bench/run_matrix.py:272` — lens `audit:methodology`


**Detail.** A case with both `n_prompt` and `n_gen` (partial_offload_sweep, quantisation_sweep) becomes ONE llama-bench invocation with `-p 512 -n 128`, which llama-bench runs as two separate tests and emits as two CSV rows (pp512, tg128). The driver gives the whole invocation a single marker window. The avg_ts extraction loops over all lines/rows with no `break` (lines 272-284), so `tps` ends up as the LAST row's value, i.e. the tg128 decode throughput. Meanwhile `tokens_of()` returns (512+128)*3 = 1920 and `energy_between()` returns the joules of prefill + decode + load together. The resulting row therefore states a decode tokens/s next to a joules_per_token computed from prefill-dominated energy over a prefill+decode token count. Note also that the first regex at line 273 never matches a real llama-bench CSV (data rows do not contain the literal string `avg_ts`; only the header does, and there the next field is `"stddev_ts"` which fails `([\d.]+)`), so the DictReader fallback is always the path taken — same last-row-wins behaviour.


**Failure scenario.** `partial_offload_sweep` (n_gpu_layers 0/8/16/24/99, p=512, g=128) is described in matrix.json as 'the placement question in miniature'. Every one of its 5 rows reports tg128 t/s as `avg_ts`, while `joules`/`joules_per_token` are dominated by the 512-token prefill (and the 1.1 GB model load). Raising ngl mostly speeds prefill, so the energy column moves a lot while the throughput column barely moves; a reader plotting the energy Pareto sees a large J/token improvement attributed to a decode throughput that did not change, and picks the wrong layer split. The same applies to both quantisation_sweep rows.


**Fix.** Make one window contain exactly one phase. In bench/matrix.json split `partial_offload_sweep` and `quantisation_sweep` into a prefill-only case (`"n_prompt":[512],"n_gen":[0]`) and a decode-only case (`"n_prompt":[0],"n_gen":[128]`) — `phase_of()` then returns prefill/decode instead of 'mixed'. In run_matrix.py, additionally parse the bench CSV with DictReader and emit one output record per bench row (carrying that row's own `n_prompt`/`n_gen`/`avg_ts`/`avg_ns`), and refuse to write `joules_per_token` when a window produced more than one bench row — write `joules` only and leave the per-token columns empty rather than a number that mixes phases.


### 4. [CRITICAL] A truncated GGUF is treated as complete and its bad hash is written into SHA256SUMS as ground truth

> **Status:** FIXED 2026-09-14 — downloads go to a `.part` sidecar and are promoted only on success, so a partial transfer is never left where a complete file is looked for. `config/models.tsv` gained a 4th sha256 column (all `-` until someone records real values); with a hash the file is verified against it, without one its size is checked against the server's `Content-Length`. SHA256SUMS entries are now RE-VERIFIED rather than skipped, `die` became `warn; continue`, and the script exits non-zero when a model genuinely failed.

`scripts/30_fetch_models.sh:32` — lens `audit:ops`


**Detail.** The presence gate is `if [ -s "$dest" ]; then ok "already present"`. `-s` is true for ANY non-empty file, including a half-downloaded one, so the `curl -L --fail --retry 5 -C - -o "$dest"` resume path on line 37 is unreachable once a partial file exists. Worse, the hashing block (lines 40-43) runs regardless: it sha256sums whatever bytes are on disk and appends that to `${MODEL_DIR}/SHA256SUMS`. Because `config/models.tsv` carries no expected-hash column, nothing ever compares that sum to an authoritative value, so the `sha256sum -c SHA256SUMS` the script prints on line 51 is circular and passes forever. Note also that `die` on line 38 aborts the whole loop, so one failed model silently skips the remaining ones in `$MODELS`.


**Failure scenario.** Student runs `make models` at 2am; the 1.1 GB Qwen2.5-1.5B download stalls after curl's 5 retries (or they Ctrl-C it), leaving a 600 MB partial at ${MODEL_DIR}/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf. They re-run `make models`: it prints "ok already present (600M)", hashes the truncated file, and appends that sum to SHA256SUMS. Every later verification passes. `50_bench_device.sh` then fails every partial_offload_sweep case with a gguf-load error that looks like an OOM, and the cross-node claim "all four nodes ran the same bytes" is false while appearing verified.


**Fix.** Add a 4th sha256 column to config/models.tsv and change the loop to: (1) always attempt `curl -L --fail --retry 5 --retry-delay 5 -C - -o "$dest" "$url"` when a file exists but its hash does not match the manifest (accept curl exit 33/HTTP 416 as 'already complete'); (2) verify the computed sum against the manifest column and `die` if it differs, deleting or renaming the bad file to *.partial; (3) only append to SHA256SUMS after a successful match. Replace the `die` on line 38 with `warn ...; continue` so one bad URL does not skip the rest of $MODELS.


### 5. [CRITICAL] A truncated .gguf is treated as "already present" and its wrong sha256 is written into SHA256SUMS

> **Status:** FIXED 2026-09-14 — same fix as #4; the two lenses found the same defect. Covered by `bench/fixtures/run_regression.sh` ("model fetcher"), which truncates a file to 3000 of 5000 bytes and asserts it is detected, completed and byte-identical to the source.

`scripts/30_fetch_models.sh:32` — lens `audit:shell`


**Detail.** The download is only attempted when `[ -s "$dest" ]` is false, i.e. when the file is absent or zero bytes. A partially transferred GGUF is non-empty, so on the next run the `then` branch fires and prints `ok "already present (...)"`. This makes the `-C -` resume flag on line 37 unreachable — the resume path can never run, because any file curl left behind is by definition non-empty. Worse, lines 40-44 then hash whatever bytes are on disk and append that hash to `${MODEL_DIR}/SHA256SUMS`, so the manifest that exists precisely to prove "all four nodes ran the same bytes" gets populated with the checksum of a corrupt file, and the `sha256sum -c SHA256SUMS` verification advertised at line 51 passes.


**Failure scenario.** Downloading Qwen2.5-3B-Instruct-Q4_K_M.gguf (~2 GB) over a home link: curl exhausts `--retry 5`, exits non-zero, `die` fires, and ~1.2 GB is left at ${MODEL_DIR}/Qwen2.5-3B-Instruct-Q4_K_M.gguf. (Same outcome if the operator Ctrl-Cs the multi-minute SD-card write, or the card fills.) Re-running `make models` reports `ok already present (1.2G)`, hashes the truncated file into SHA256SUMS, and prints the line as if verified. llama-bench later fails to load the model with a gguf magic/size error for every case in the sweep, or — if the truncation happens to land after the tensor data of a smaller file — the cluster runs mismatched bytes on different nodes while the checksum file claims they match.


**Fix.** Download to a sidecar and only promote it on success, so partial files are never mistaken for complete ones:

    part="${dest}.part"
    if [ -s "$dest" ]; then ok "already present ..."; else
      curl -L --fail --retry 5 --retry-delay 5 -C - -o "$part" "$url" || die "download failed: ${name}"
      mv -f "$part" "$dest"
    fi

and make the hash step re-verify rather than blindly append: if a line for `$file` already exists in SHA256SUMS, run `(cd "$MODEL_DIR" && grep " ${file}\$" SHA256SUMS | sha256sum -c -)` and `die` on mismatch instead of skipping.


### 6. [CRITICAL] `timeout 6 tegrastats` under `set -o pipefail` kills the script before any JSON is printed

> **Status:** FIXED 2026-09-14 — reproduced first: the script emitted 0 bytes and, through `make ... | tee`, reported success. The `timeout` pipeline is now guarded, and so is every other fact-gathering pipeline in the kit — this was systemic, not one line. `scripts/lib/common.sh` documents the rule next to `set -euo pipefail`, and `l4t_version` / `cuda_version` were self-fatal off-device too.

`scripts/60_collect_node_facts.sh:32` — lens `audit:shell`


**Detail.** common.sh sets `set -euo pipefail`, which every script inherits. Line 32-34 is a plain assignment:

    IDLE_MW="$(timeout 6 tegrastats --interval 1000 2>/dev/null | sed ... | tail -3 | awk ...)"

`tegrastats` never exits on its own, so `timeout` SIGTERMs it and returns 124. With `pipefail` the whole pipeline returns 124, the command substitution returns 124, so the assignment returns 124, and `set -e` terminates the script right there — *before* the `python3 - "$BUILDINFO" <<PY` heredoc that is the script's entire output. I verified this exact construct in bash: `set -euo pipefail; X="$(timeout 1 sh -c 'sleep 5' | cat)"; echo B` prints nothing after `A` and the script exits 124. This fires on every real Jetson (i.e. exactly when `command -v tegrastats` succeeds); it is silent off-device because the `if` guard skips it, which is why it would never show up in off-board testing. Note the author guarded the identical pattern in 00_system_prep.sh line 92 with `|| true` but missed it here.


**Failure scenario.** On a stock Nano with tegrastats present, `make facts` runs `scripts/60_collect_node_facts.sh | tee results/jetson-nano-01_facts.json`. The script dies at 6 s with status 124 having printed nothing. make's shell has no pipefail, so the pipeline's status is tee's 0 and make reports SUCCESS while leaving a zero-byte `results/jetson-nano-01_facts.json`. The same happens inside 90_collect_results.sh line 16, where stderr is sent to /dev/null and the failure degrades to `warn "could not collect node facts"` — so the tarball shipped to the analysis machine contains an empty node_facts.json, and the device-characterisation row every measurement is interpreted against is silently missing.


**Fix.** Make the timeout's non-zero status non-fatal and let an empty result fall through to the `${IDLE_MW:-0}` default already present at line 78:

    IDLE_MW="$( { timeout 6 tegrastats --interval 1000 2>/dev/null || true; } \
      | sed -n 's/.*POM_5V_IN \([0-9]*\)\/.*/\1/p' | tail -3 \
      | awk '{s+=$1; n++} END{if(n) printf "%d", s/n}' )" || IDLE_MW=""

(the trailing `|| IDLE_MW=""` also covers awk/sed printing nothing under pipefail).


### 7. [CRITICAL] pick(b, "n_prompt", "n_gen", ...) returns the literal "0" on decode rows, so decode windows silently get no energy-per-token

> **Status:** FIXED 2026-09-12

`telemetry/energy_window.py:185` — lens `audit:python`


**Detail.** A llama-bench CSV always contains BOTH `n_prompt` and `n_gen` columns; for a tg (decode) row the values are `n_prompt="0"`, `n_gen="128"`. `pick()` returns the first name whose value is not in `("", None)` — and `"0"` passes that test — so `n_tok` is `"0"` for every decode row. `if n_tok:` is true (non-empty string), `tokens = float("0") = 0.0`, which is not None, so the `bench_tps * duration_s` fallback at line 196 is skipped, and `if joules is not None and tokens:` at line 198 is False because 0.0 is falsy. Result: `joules_per_token`, `wh_per_mtok`, `tokens_per_joule` and `active_joules_per_token` are never set for any decode window, with no warning. Prefill rows are wrong in a different way: `n_prompt` from the CSV is the per-test prompt length, but llama-bench ran it `-r` times (plus a warm-up) inside the marker window, so dividing window joules by 512 instead of by ~512*3 inflates J/token by roughly the repetition count. `res["bench_test"] = pick(b, "test", "n_prompt")` has the same "0"-is-truthy quirk but is only cosmetic.


**Failure scenario.** Follow docs/10_measurement_protocol.md §5 exactly: log a decode window with `echo decode > /tmp/hs_marker`, then `python3 telemetry/energy_window.py run.csv --marker decode --bench bench.csv --out per_token.csv`. The table prints the power lines but never the `J/tok` line, and `per_token.csv` has no `joules_per_token` column at all — the headline number of the study is missing, and it looks like the join "just didn't apply" rather than a bug. With `--marker prefill` you do get a number, but it is ~3x too large at the default `-r 3`.


**Fix.** Treat "0" as absent and pick by test type: `def _tok(b): g = pick(b, "n_gen"); p = pick(b, "n_prompt"); g = float(g) if g else 0.0; p = float(p) if p else 0.0; return (p + g) or None` — using both, since a `-p N -n M` invocation emits separate rows. Then multiply by the repetition count: add `ap.add_argument("--reps", type=int, default=1, help="must match llama-bench -r for the window")` and use `tokens = per_row_tokens * args.reps` (plus the warm-up term, see the tokens_of finding). Also change `pick()` to skip values that are `"0"`/`0` when it is being used to find a token count, or give it an explicit `skip_zero=True` flag.


### 8. [HIGH] tokens_of() omits llama-bench's warm-up run its own docstring claims to count, inflating J/token by ~33% at the default -r 3

> **Status:** FIXED 2026-09-12

`bench/run_matrix.py:401` — lens `audit:python`


**Detail.** The docstring says tokens are counted "counting repetitions and the warm-up run llama-bench performs before each measured set", but the body is `per_rep = n_prompt + n_gen; return per_rep * repetitions`. llama-bench runs an unmeasured warm-up before the timed repetitions — a full-size prompt pass for a pp test and a 1-token generate for a tg test — and that warm-up happens inside the `t_start..t_end` window that `energy_between()` integrates. So the joules cover (reps+1) prompt passes while the divisor covers reps of them: for `cpu_prefill_sweep` / `gpu_prefill_sweep` (n_prompt 128..1024, `repetitions: 3`) `joules_per_token`, `wh_per_mtok` and `active_joules_per_token` are all about 33% too high, systematically and in the same direction for every prefill row. `largest_local_model` sets `repetitions: 2`, making its inflation ~50% — so the bias is not even constant across the sweep and the cases are not comparable. Relatedly, `matrix.json` advertises a `"warmup": true` knob in `defaults`, but `expand()` never reads it and `build_cmd()` never emits a corresponding flag, so it is a dead setting that implies the warm-up is under control when it is not. Model-load time also sits inside the window; that is a separate, smaller bias worth stating in the protocol.


**Failure scenario.** `./scripts/50_bench_device.sh` with the stock matrix. The `gpu_prefill_sweep` row for n_prompt=1024 reports joules for 4x1024 tokens divided by 3x1024 tokens. The prefill-vs-decode energy comparison — the thesis's central claim — is skewed by a factor that differs between prefill rows (+33%) and decode rows (+~1 token, negligible), i.e. prefill is made to look worse than it is by a third.


**Fix.** Make the accounting match what the pinned llama-bench ref actually runs: `n_p = rec.get("n_prompt") or 0; n_g = rec.get("n_gen") or 0; reps = rec.get("repetitions") or 1; total = n_p * (reps + 1) + n_g * reps + (1 if n_g else 0); return total or None` — and verify the warm-up shape against the ref recorded in build_info.json before trusting it. Either wire `warmup` through `expand()`/`build_cmd()` to the pinned binary's actual flag or delete the key from matrix.json defaults so it stops implying a control that does not exist. Also record `repetitions` and the warm-up assumption as columns in the joined CSV so the divisor is auditable.


### 9. [HIGH] tokens_of() omits llama-bench's warm-up pass despite its docstring claiming to count it

> **Status:** FIXED 2026-09-12

`bench/run_matrix.py:398` — lens `audit:methodology`


**Detail.** The docstring says 'Tokens actually processed in the window, counting repetitions and the warm-up run llama-bench performs before each measured set', but the body is `(n_prompt + n_gen) * repetitions` — no warm-up term. llama-bench performs a warm-up before the repetition loop of each test: a full `n_prompt`-token prefill for a pp test, and a 1-token generation for a tg test. That warm-up burns real energy that IS inside the marker window, but its tokens are not in the denominator, so `joules_per_token` and `active_joules_per_token` are systematically overstated for every prefill row by a factor (reps+1)/reps. The `"warmup": true` default in bench/matrix.json is also dead: `expand()` copies it into the merged dict but `build_cmd()` never emits any flag for it, so the knob advertises control the driver does not have.


**Failure scenario.** `cpu_prefill_sweep` p=1024, repetitions=3: the window contains 4*1024 = 4096 prefill tokens of work but the denominator is 3072, so every prefill J/token in the table is 33% too high. Because the error is (reps+1)/reps, following docs/10_measurement_protocol.md section 6 ('5 or more for any number that goes in a table') and re-running the same case at r=5 makes the identical hardware appear 11% more energy-efficient purely from bookkeeping — a spurious 'improvement' that the repetition-spread check in section 6 cannot catch, since it compares repetitions within a run, not across reps settings.


**Fix.** Compute the denominator from what llama-bench actually did: `tokens = n_prompt*(repetitions+1) + (n_gen*repetitions + 1 if n_gen else 0)` for the phase-separated cases, and confirm the warm-up shape against the exact llama.cpp ref pinned in build_info.json before trusting it (the warm-up is `test_prompt(n_prompt)` plus `test_gen(1)` in llama-bench's `main`). Better: derive it from the bench CSV rows' own `n_prompt`/`n_gen` plus an explicit `warmup_tokens` column so the assumption is visible in the data. Either delete `"warmup": true` from matrix.json defaults or wire it to a real flag.


### 10. [HIGH] Fixed 20 s cooldown plus a within-window temp_rise_c check cannot detect the thermal drift it is meant to catch

> **Status:** FIXED 2026-09-14 — the finding is right that `temp_rise_c` is backwards as a contamination test: it is a WITHIN-window delta and goes to ~0 exactly when the board has reached a hot steady state, so the worst case passed the docs' old "> 8 C" rule. Four changes. (1) `cooldown_s` is now a FLOOR: after it the driver polls `temp_thermal_c` from the tail of the power CSV — the same column every row is scored on, so gate and audit column cannot disagree — until the board is within `cooldown_target_c` (3 C) of the idle baseline, capped by `max_cooldown_s` (300 s). `tail_sample()` reads only the last 16 KB, because a gate polling every 2 s for 3 h would otherwise put the driver's own load inside the cooldown it is measuring. With `--no-power` the wait stays open loop and `cooldown_reached` is written BLANK, so no row claims a gate that did not run. (2) `temp_baseline_c` is captured once from the idle window and `temp_drift_c = temp_start_c - temp_baseline_c` is on every row — the between-run quantity `temp_rise_c` structurally cannot express. (3) `energy_between()` now carries `temp_start_c`/`temp_end_c` and `cpu_freq_start_mhz`/`cpu_freq_min_mhz`/`gpu_freq_*`, so a post-hoc throttling audit of the joined CSV is possible at all. (4) Case 1 is repeated unchanged as the final run (`drift_control=1`, `--no-drift-control` opts out) and the driver prints the delta: same case, hours apart, so that difference IS drift. The audit's other option — shuffling with a seed — was deliberately not taken: it fights `interleave_groups()`, which depends on a chosen order, and a sweep that cannot be re-run in sequence is harder to defend than one whose drift is published. Demonstrated with a thermally realistic fixture: with a 1 s floor the gate held the sweep 35-79 s per case until the board came down.

`bench/run_matrix.py:290` — lens `audit:methodology`


**Detail.** `time.sleep(run["cooldown_s"])` is an open-loop 20 s wait; nothing reads the temperature before starting the next case. A fanless Nano dev kit dissipating ~10 W has a heatsink time constant of minutes, so 20 s recovers almost nothing, and across the ~23 runs of the enabled matrix (1.5-3 h per scripts/50_bench_device.sh) the board ratchets monotonically upward and eventually hardware-throttles. The only contamination check the toolkit offers is `temp_rise_c = temps[-1] - temps[0]` (run_matrix.py:392, energy_window.py:123) with the docs' rule '> 8 C means the case is thermally contaminated' (docs/10_measurement_protocol.md:103). That is a *within-window delta*, which goes to ~0 precisely when the board has reached a hot steady state — the worst case passes the check. `temp_max_c` is on the row but `temp_start_c` is not, and neither `cpu_freq_mhz` nor `gpu_freq_mhz` at window start is copied onto the joined row, so a post-hoc throttling audit of the joined CSV is impossible. Cases are also run in strict matrix order with no randomisation, so thermal drift is perfectly confounded with case order.


**Failure scenario.** `cpu_prefill_sweep` runs first (board ~40 C, full clocks); `thread_sweep` — the same 512-token CPU prefill workload — runs last, ~2 h in, at ~85 C with the SoC clamping clocks. Both report temp_rise_c of roughly 1-2 C and pass the '<8 C' rule, but thread_sweep shows lower t/s and higher J/token. The thesis then reports 'the 4th core does not help / costs energy' when the effect is drift along run order, not thread count. The early cpu_prefill and late thread_sweep rows are not comparable even though both are 512-token CPU prefill.


**Fix.** Replace the fixed sleep with a closed-loop gate: after each case, poll `temp_thermal_c` (tail of the power CSV, or /sys/devices/virtual/thermal/thermal_zone*/temp) and block until it is within +3 C of the pre-sweep baseline captured in the 30 s idle window, with a `max_cooldown_s` ceiling; record the actual wait as `cooldown_actual_s`. Add `temp_start_c`, `cpu_freq_start_mhz`, `gpu_freq_start_mhz` and `run_index` to every joined row (all are already in the power CSV at the window's first sample). Shuffle run order with a seed recorded in the output, or at minimum re-run the first case as the final case and publish the two rows as an explicit drift control.


### 11. [HIGH] The external inline power meter is declared ground truth but nothing in the kit reads, logs, or collects it

> **Status:** ADDRESSED 2026-09-12 — there is no meter; across-device comparison is documented as deferred.

`docs/10_measurement_protocol.md:33` — lens `audit:ops`


**Detail.** README.md:21/115 and this protocol section make the inline meter the ground truth for whole-device energy and the ONLY instrument comparable across the four nodes (the Pis have no INA3221), and section 2 says "Report both; never silently substitute one for the other". Yet the folder contains no meter logger, no serial/USB read path, no wiring or model-specific instructions, no field for it in `60_collect_node_facts.sh`'s facts JSON (which only records `rail_source: INA3221 via tegrastats`), no column for it in run_matrix.py's joined CSV, and no copy step in `90_collect_results.sh`. The only trace is the commented `pyserial>=3.5` in telemetry/requirements.txt line 8, which nothing imports. There is also no procedure for aligning a low-rate meter reading to the sweep's UTC window boundaries.


**Failure scenario.** Student completes the entire documented bring-up and sweep, ships the tarball to the laptop, and finds every energy number is tegrastats POM_5V_IN — module-only. The whole-device figures and the cross-device (Nano vs Pi 4 vs Pi 5) energy-per-token comparison, which the thesis requires, have zero recorded data, and the sweep must be repeated once a meter capture path exists.


**Fix.** Add `telemetry/inline_meter_logger.py` (pyserial; one CSV with ts_unix/ts_utc/node_id/volts/amps/watts/energy_j and the same `--marker-file` mechanism as tegrastats_logger.py so its windows align), start it alongside the tegrastats logger in bench/run_matrix.py (guarded by a `METER_PORT=` knob added to config/node.env.example), copy its CSV in 90_collect_results.sh, and add a docs/ section naming the meter model, how it is inserted in the 5 V supply lead, and the manual-transcription fallback for meters with no serial output.


### 12. [HIGH] An interrupted swapfile allocation wedges 00_system_prep.sh permanently, despite the "idempotent" claim

> **Status:** FIXED 2026-09-14 — reproduced by reasoning through the state machine: `[ ! -f "$SWAPFILE" ]` is false for a partial file, so BOTH the allocation and the `mkswap` were skipped and `swapon` failed under `set -e`, identically on every later run. A new `swapfile_ready()` checks the file is present, the right SIZE, and carries a real `mkswap` header (`swaplabel`), rebuilding it otherwise; a failed `swapon` now warns and continues rather than killing a script whose remaining steps are still useful. All five decisions asserted in `bench/fixtures/run_regression.sh`.

`scripts/00_system_prep.sh:38` — lens `audit:shell`


**Detail.** `mkswap` is inside the `if [ ! -f "$SWAPFILE" ]` guard (lines 38-44) but `sudo swapon "$SWAPFILE"` is outside it (line 45). So the guard only protects re-allocation, not the swap-header creation. If /var/swapfile exists but was never `mkswap`ed, every subsequent run skips lines 39-44 and goes straight to `swapon`, which fails with `swapon: /var/swapfile: read swap header failed: Invalid argument`. Under `set -e` (inherited from common.sh) that aborts the script at line 45, before the power-mode, jetson_clocks, fan-PWM, chrony and network steps — all the things the measurement protocol depends on. Nothing in the script can recover from this state; the operator has to know to `sudo rm /var/swapfile` by hand. The header comment explicitly advertises "Idempotent — safe to re-run", so this is the state an operator will be least prepared for.


**Failure scenario.** On a Nano whose rootfs is on SD, `fallocate` is unsupported/fails, so the fallback `sudo dd if=/dev/zero of=/var/swapfile bs=1M count=8192 status=progress` starts — the script itself warns this is "slow on SD; be patient". The operator Ctrl-Cs it after a few minutes (or dd hits ENOSPC on a 32 GB card that already holds JetPack). /var/swapfile now exists with no swap signature. Every later `make prep` prints "Swap", skips mkswap because the file exists, dies on `swapon`, and never reaches `nvpmodel`/`jetson_clocks`/fan — so the operator proceeds to benchmark an unlocked, unfanned board while believing prep succeeded on an earlier run.


**Fix.** Allocate to a temp name and rename, and make mkswap unconditional (mkswap is idempotent):

    if [ ! -f "$SWAPFILE" ]; then
      sudo fallocate -l "${SWAP_GB}G" "${SWAPFILE}.part" 2>/dev/null \
        || sudo dd if=/dev/zero of="${SWAPFILE}.part" bs=1M count=$((SWAP_GB*1024)) status=progress
      sudo mv -f "${SWAPFILE}.part" "$SWAPFILE"
    fi
    sudo chmod 600 "$SWAPFILE"
    sudo swaplabel "$SWAPFILE" >/dev/null 2>&1 || sudo mkswap "$SWAPFILE"
    sudo swapon "$SWAPFILE"

and add a `trap 'sudo rm -f "${SWAPFILE}.part"' EXIT`-style cleanup or at least remove a stale `.part` at the top.


### 13. [HIGH] Missing g++-8 aborts the build with exit 1 and no error message at all

> **Status:** FIXED 2026-09-14 — confirmed the mechanism in isolation: `"$(command -v g++-8)"` inside an array assignment returns 1 and prints nothing, and `set -e` then exits silently. The host compiler is now checked BEFORE it is spliced into `CMAKE_ARGS`, naming the missing binary, the apt command that installs it, and the `BUILD_CUDA=0` escape. Asserted in the regression against a fixture with a fake `nvcc` and an absent `g++`.

`scripts/20_build_llama_cpp.sh:108` — lens `audit:shell`


**Detail.** Lines 108-110 append `-DCMAKE_CUDA_HOST_COMPILER="$(command -v ${HOST_CXX})"` etc. to the `CMAKE_ARGS` array. `command -v` returns 1 when the compiler is absent, and under `set -e` a compound array assignment whose command substitution fails terminates the shell. I verified this: `set -euo pipefail; A=(x); A+=( -DFOO="$(command -v definitely-not-here)" ); echo reached` exits 1 and never prints `reached`. The script's own preflight only does `need_cmd git` and `need_cmd cmake` (lines 23-24) — it never checks HOST_CXX/HOST_CC even though node.env's `HOST_CXX=g++-8` is a non-default compiler that only exists because 00_system_prep.sh apt-installs it.


**Failure scenario.** An operator who already has cmake and git (or who re-flashed and restored ~/heteroserve) runs `make build` directly — the Makefile's `build` target depends only on `config/node.env`, not on `prep` or `toolchain`, so nothing forces gcc-8/g++-8 to be installed first. The script prints the source step, the "CUDA feasibility" banner, `── Configure ───`, and then exits with status 1 printing nothing else. make reports `Error 1` with no cause. The operator has no way to tell this apart from a cmake/CUDA failure and will go hunting in the CUDA-10.2 rabbit hole described in the heredoc.


**Fix.** Resolve and validate the compilers before building the array, right after line 24:

    HOST_CC="${HOST_CXX/g++/gcc}"
    if [ "${BUILD_CUDA}" = "1" ]; then
      need_cmd "${HOST_CXX}" "apt install g++-8 — run scripts/00_system_prep.sh"
      need_cmd "${HOST_CC}"  "apt install gcc-8 — run scripts/00_system_prep.sh"
    fi

and quote the expansions in the array (`"$(command -v "${HOST_CXX}")"`) so a HOST_CXX containing spaces is not word-split. Delete the now-duplicate `HOST_CC=` at line 101.


### 14. [HIGH] Pre-flight thermal gate reads the PMIC-Die dummy zone and always reports 100 C

> **Status:** FIXED 2026-09-14 — the gate selected by MAX over all zones, and the hottest zone on a Nano is the PMIC-Die dummy reporting a constant 100 C, so it reported 100 C on a stone-cold board every time and its "settle below 45 C" warning always fired. A new `thermal_zone_c()` in `scripts/lib/common.sh` selects by the zone's `type` file — never by zone NUMBER, which is not stable across L4T releases — preferring `thermal-fan-est`, which is the same quantity tegrastats prints as `thermal@` and the logger records as `temp_thermal_c`, so the pre-flight reading and every sweep row are the same number. Falls back to the hotter of CPU-therm/GPU-therm, can never select PMIC-Die, prints nothing rather than aborting when no zone exists. Demonstrated against a fake Nano zone layout: 38 C from thermal-fan-est where the old rule said 100 C.

`scripts/50_bench_device.sh:22` — lens `audit:platform`


**Detail.** `T="$(cat /sys/devices/virtual/thermal/thermal_zone*/temp | sort -n | tail -1)"` takes the MAXIMUM over all thermal zones. On a Jetson Nano (L4T 32.7) the zones are AO-therm, CPU-therm, GPU-therm, PLL-therm and PMIC-Die, and PMIC-Die is a hard-coded dummy that always reads 100000 m°C — the kit's own tegrastats sample line in telemetry/tegrastats_logger.py:9 shows it (`PMIC@100C`). The max is therefore always 100000, never the real die temperature. NVIDIA confirms the PMIC driver reports a constant 100 C because it cannot read the PMIC die.


**Failure scenario.** Operator runs `./scripts/50_bench_device.sh` on a cold board that has been idle overnight. Pre-flight prints `temp start : 100 C` and warns `board is warm; let it settle below 45 C` on every single invocation. The warning is pure noise, so the operator learns to ignore it — and the one case it exists for (starting a sweep on a heat-soaked board after a previous 3 h run) is indistinguishable from the cold case. The recorded thermal starting point in docs/10_measurement_protocol.md's condition table is meaningless.


**Fix.** Select the zone by type instead of taking the max over all zones. Replace line 22 with a loop that reads each zone's `type` and keeps only CPU-therm/GPU-therm, e.g.:

```sh
T=""
for z in /sys/devices/virtual/thermal/thermal_zone*; do
  case "$(cat "$z/type" 2>/dev/null)" in
    CPU-therm|GPU-therm)
      v="$(cat "$z/temp" 2>/dev/null || echo 0)"
      [ "${v:-0}" -gt "${T:-0}" ] && T="$v" ;;
  esac
done
```

(Equivalently, exclude any zone whose type is `PMIC-Die`.) Note that `thermal@` in tegrastats — the zone telemetry/tegrastats_logger.py already maps to `temp_thermal_c` — is the thermal-fan-est estimate and is a fine alternative source for the same check.


### 15. [HIGH] POM_5V_IN is described as module-only power; on the Nano devkit it is the whole-board 5 V input

> **Status:** FIXED 2026-09-12 — POM_5V_IN documented as whole-devkit scope in five places.

`scripts/60_collect_node_facts.sh:79` — lens `audit:platform`


**Detail.** `"rail_source": "INA3221 via tegrastats POM_5V_IN (module only, excludes carrier board and PSU loss)"` — and the same claim in README.md:113-116, docs/10_measurement_protocol.md:27-31 and telemetry/tegrastats_logger.py:12-16 ("They cover the module only: the carrier board, any USB peripherals and the PSU conversion loss are NOT included"). That is a TX2/Xavier fact, where the INA3221 sits on the module. On the Nano B01 devkit the INA3221 is on the CARRIER board (i2c 6-0040, `/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0/`), and channel 0 (VDD_IN / POM_5V_IN) measures the 5 V input rail of the whole developer kit, downstream of the micro-USB/barrel-jack selector. It therefore includes the carrier board, the USB VBUS draw and HDMI — everything except the wall adapter's AC/DC conversion loss. The kit's own docs/20_troubleshooting.md:84-89 says third-party CARRIER boards omit the INA3221, which contradicts the module-only claim.


**Failure scenario.** The report claims the on-board rail 'systematically under-reports whole-device consumption' and uses the gap between POM_5V_IN and the external inline meter as the carrier-board + peripheral overhead. On the real board that gap is only the PSU conversion loss (~10-20%), and a USB SSD attached per docs/00_flash_jetpack.md:24 draws through POM_5V_IN — so the 'module' energy-per-token figures silently include SSD power, and the stated direction and magnitude of the bias between the two instruments are wrong in the thesis.


**Fix.** Correct the fact everywhere it appears. The string at line 79 should read: `"INA3221 on the Nano devkit carrier board via tegrastats POM_5V_IN = 5V input rail of the whole developer kit (module + carrier + USB/HDMI peripherals); excludes only the external supply's conversion loss"`. Update README.md:113-116, docs/10_measurement_protocol.md:27-31 and the telemetry/tegrastats_logger.py docstring the same way, and drop the claim that it under-reports whole-device consumption — with no peripherals attached POM_5V_IN is close to the inline meter reading minus PSU loss. If the study wants a module-only number, it does not exist on this board; POM_5V_CPU + POM_5V_GPU are the only sub-rails available.


### 16. [HIGH] The idle-baseline tegrastats capture kills the whole facts script under set -e/pipefail

> **Status:** FIXED 2026-09-14 — same root cause as #6, fixed with it.

`scripts/60_collect_node_facts.sh:32` — lens `audit:platform`


**Detail.** scripts/lib/common.sh:5 sets `set -euo pipefail` for every script that sources it. Line 32 assigns `IDLE_MW="$(timeout 6 tegrastats --interval 1000 2>/dev/null | sed ... | tail -3 | awk ...)"` with no `|| true`. `tegrastats` never exits on its own, so `timeout` always reaches its deadline and always exits 124; `pipefail` propagates 124 as the pipeline status, and a bare assignment whose command substitution returns non-zero takes the shell's exit status — so `set -e` terminates the script right there. Everything after line 32, including the python3 heredoc that actually emits the JSON, never runs.


**Failure scenario.** On any Nano where `command -v tegrastats` succeeds (i.e. every stock JetPack 4.6 board), `./60_collect_node_facts.sh > ../results/jetson-nano-01_facts.json` exits with status 124 after 6 seconds and leaves a zero-byte facts file. Since the redirection already created the file and the only output was on stderr, a user running it via `make facts` sees the step 'complete' with an empty JSON. Every later measurement is then interpreted against a non-existent node record.


**Fix.** Make the assignment an AND-OR list so set -e is not armed, and default the value:

```sh
  IDLE_MW="$(timeout 6 tegrastats --interval 1000 2>/dev/null \
    | sed -n 's/.*POM_5V_IN \([0-9]*\)\/.*/\1/p' | tail -3 \
    | awk '{s+=$1; n++} END{if(n) printf "%d", s/n}')" || IDLE_MW=""
```

Same pattern as scripts/00_system_prep.sh:92, which already guards its `timeout 3 tegrastats ... || true`. Harden further by writing to a file instead of a pipe (`tegrastats --interval 1000 --logfile "$tmp" &` / `kill` / parse `$tmp`), since tegrastats' stdout is block-buffered into a pipe and a 6-second capture may never flush a full 4 KiB block before SIGTERM, yielding an empty IDLE_MW even after the set -e fix.


### 17. [HIGH] No way to capture a phase-marked power window on this node during an orchestrator-driven split run

> **Status:** FIXED 2026-09-14 — `scripts/45_capture_window.sh start|mark|idle|stop|path|status`. The orchestrator drives it over ssh, because it is the only machine that knows which phase it asked for: this node is an RPC worker and is handed tensors, never phases. It writes the SAME marker file `run_matrix.py` uses and lands the CSV in `$RESULTS_DIR` under the house naming, so a cluster capture is read by exactly the same `energy_window.py` commands as a sweep and `90_collect_results.sh` bundles it without being told. It refuses to start when a tegrastats is already running and proves it is logging before the orchestrator commits a phase to it. The limit is written into the script and into docs/10 §5: an ssh round trip puts each marker edge within about one sample period, which is immaterial for a seconds-long leg and hopeless for the prefill/decode boundary inside a single request -- cluster runs must bracket whole legs.

`scripts/70_cluster_smoke.sh:61` — lens `audit:ops`


**Detail.** The study's central measurement is prefill-vs-decode placement across nodes, but every energy path in this folder assumes the benchmark is driven locally: only `bench/run_matrix.py` writes the marker file (`$RESULTS_DIR/.marker_$NODE_ID`) that `tegrastats_logger.py` stamps onto samples. When the run is driven from the laptop over ggml RPC, nothing on the Jetson starts a logger or sets markers — `70_cluster_smoke.sh`, the only cluster-facing script here, runs a two-node inference (line 74) with no power logging at all, and there is no Makefile target or script for "log this node's power for a remote run". The sole documented fallback is docs/10 section 5 (`echo prefill > /tmp/hs_marker` by hand, in a second terminal, at a path that does not even match the one run_matrix uses), which cannot resolve a prefill phase that lasts well under a second.


**Failure scenario.** Student runs the prefill/decode disaggregation experiment from the laptop against the Jetson's rpc-server, then tries `energy_window.py ... --marker prefill` on the Jetson's CSV: either no logger was running at all, or every row's marker is empty, so no per-phase joules can be attributed to this node and the distributed energy-per-token result — the deliverable — cannot be computed without redoing the runs.


**Fix.** Add a small node-side capture entry point, e.g. `scripts/45_capture_window.sh start|mark <label>|stop` that launches tegrastats_logger.py with `--marker-file $RESULTS_DIR/.marker_$NODE_ID` into $RESULTS_DIR with a timestamped filename and writes labels into that same marker file (so the orchestrator can drive it over ssh: `ssh jetson scripts/45_capture_window.sh mark prefill`), wire a `make capture` target to it, standardise docs/10 section 5 on that marker path instead of /tmp/hs_marker, and have 70_cluster_smoke.sh start/stop it around the two-node run.


### 18. [HIGH] The always-on telemetry unit overwrites its one CSV on every restart and records no phase markers

> **Status:** FIXED (both halves, one of them by decision rather than by code) 2026-09-14 — the truncation half is gone: `ExecStart` writes `continuous_power_$(date -u +%%Y%%m%%dT%%H%%M%%SZ).csv`, a fresh file per start, with the `%` doubled so systemd does not expand it before /bin/sh sees it. The "records no phase markers" half is NOT a defect: the unit's own header (lines 15-18) states that this CSV deliberately carries none, because nothing in an always-on logger knows when a prefill or decode window begins. It answers "what was the board doing between runs"; per-phase attribution comes from the sweep's own marked logger.

`systemd/heteroserve-telemetry.service:23` — lens `audit:ops`


**Detail.** ExecStart hard-codes `--out /home/%USER%/heteroserve/results/continuous_power.csv`, a single fixed filename, and `tegrastats_logger.py` opens it with `open(args.out, "w")` (line 182) — truncating. Combined with `Restart=on-failure` / `RestartSec=10` and `WantedBy=multi-user.target`, any crash, `systemctl restart`, or reboot destroys the entire previous log with no backup. The running trapezoidal integral (`energy_in_j`) also restarts at 0 after a restart, so any downstream edge-subtraction that straddles a restart yields a negative or meaningless joule figure. Separately, the unit passes no `--marker-file`, so `marker` is empty on every row (the arg defaults to `$HS_MARKER_FILE`, which nothing sets), which means `energy_window.py continuous_power.csv --marker prefill` exits with "no samples carry marker" — the unit's data can never be cut into prefill/decode windows. Finally the path is hard-coded to ~/heteroserve/results, so if the student follows docs/00_flash_jetpack.md line 24 and moves HS_ROOT to a USB SSD, the file is written outside $RESULTS_DIR and `90_collect_results.sh` never tarballs it.


**Failure scenario.** Student enables the unit, runs an unattended overnight cluster experiment, and the box reboots at 03:00 after a brown-out (or the student runs `systemctl restart` to pick up a config change). On boot the unit restarts, truncates continuous_power.csv, and the entire night's power log is gone with no error anywhere except a silently reset file.


**Fix.** Make the output path unique per start and append-safe: `ExecStart=/bin/sh -c 'exec /usr/bin/python3 /home/%USER%/HeteroServe/Jetson/telemetry/tegrastats_logger.py --out "$RESULTS_DIR/continuous_power_$(date -u +%%Y%%m%%dT%%H%%M%%SZ).csv" --hz 1 --node-id "$NODE_ID" --marker-file "$RESULTS_DIR/.marker_$NODE_ID"'` with `EnvironmentFile=/home/%USER%/HeteroServe/Jetson/config/node.env` so RESULTS_DIR/NODE_ID come from node.env instead of being hard-coded. Additionally change tegrastats_logger.py line 182 to open in append mode and write the header only when the file is empty.


### 19. [HIGH] datetime.fromisoformat does not exist in Python 3.6.9, so --from/--to always raises AttributeError

> **Status:** FIXED 2026-09-12

`telemetry/energy_window.py:59` — lens `audit:python`


**Detail.** `datetime.fromisoformat()` was added in Python 3.7. JetPack 4.6.x / Ubuntu 18.04 ships python3 == 3.6.9 and cannot be upgraded on this board, so `parse_time()` is guaranteed to blow up the moment the argument is not a bare float. The `try: return float(s)` guard only catches the unix-epoch case; any ISO stamp — including the exact format this toolkit's own logger writes (`2026-09-12T04:11:07.512+00:00` from `isoformat(timespec="milliseconds")`) — falls through to line 59 and dies with `AttributeError: type object 'datetime.datetime' has no attribute 'fromisoformat'`. Secondary issue in the same function: even on 3.7+, an ISO string with no timezone suffix is parsed as naive and `.timestamp()` then interprets it in the node's local zone; on a box set to Asia/Bangkok that shifts the window by 7 h and silently selects zero samples.


**Failure scenario.** After a cluster experiment, you want the window around a split run: `python3 telemetry/energy_window.py run.csv --from 2026-09-12T04:11:07Z --to 2026-09-12T04:13:40Z`. On the Nano it crashes with a traceback instead of printing joules; the same command works on the x86 orchestrator (3.8+), so it looks like a data problem rather than a version problem.


**Fix.** Replace the fromisoformat call with a 3.6-safe parser using strptime over a small format list, and force UTC for stamps lacking an offset: strip a trailing `Z`, split off a `+HH:MM`/`-HH:MM` offset manually, then `datetime.strptime(base, fmt)` for fmt in `("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S")`, attach `timezone.utc` (or the parsed offset) via `.replace(tzinfo=...)`, and return `.timestamp()`. Import `timezone` from datetime. Reject naive-without-offset input explicitly rather than defaulting to local time.


### 20. [MEDIUM] Mixed prompt+gen cases keep only the last (decode) avg_ts while tokens_of sums prompt and gen tokens, producing a self-contradictory row

> **Status:** PARTLY FIXED 2026-09-13 — the mixed row is gone (see #3). The side-by-side `pp_ts`/`tg_ts` columns this finding proposes were deliberately NOT taken: they put two phases back on one row against one joules figure, which is the defect itself. One row per phase instead.

`bench/run_matrix.py:272` — lens `audit:python`


**Detail.** For a case with both axes in play — `partial_offload_sweep` (n_prompt 512, n_gen 128, 5 runs), `quantisation_sweep` (2 runs), and `largest_local_model` when enabled — llama-bench emits two CSV rows, pp512 and tg128. The extraction loops (`for line in ... splitlines()` at 272 and `for r in rows` at 278) overwrite `tps` on every match with no break, so the recorded `avg_ts` is whichever row came last, i.e. the tg (decode) number; the pp throughput is dropped from the joined CSV entirely. Meanwhile `tokens_of()` returns `(512 + 128) * reps`, so the same record divides whole-window joules by prefill+decode tokens while labelling the row with a decode-only tokens/s. Anyone reading `<base>_bench_energy.csv` per docs/10_measurement_protocol.md §"Reading the output" ("`avg_ts` | llama-bench tokens/s for the case") will pair a decode rate with a mixed-phase energy figure.


**Failure scenario.** `./scripts/50_bench_device.sh --only partial_offload_sweep` produces 5 rows. Each shows e.g. `avg_ts=4.1` (the tg128 rate) with `tokens=1920` and `joules_per_token` derived from a window that is mostly prefill. The layer-partition curve — the placement question the case exists to answer — is plotted against the wrong throughput metric, and the prefill rate for each offload level is nowhere in the output.


**Fix.** Parse all rows and keep them separately instead of collapsing: build `per_test = [(r.get("n_prompt"), r.get("n_gen"), float(r["avg_ts"])) for r in rows]` and emit `pp_ts` and `tg_ts` as distinct columns (leaving `avg_ts` set only when exactly one test ran). Better still for the join, split mixed cases in `expand()` into two runs — one prompt-only, one gen-only — each with its own marker window, so throughput, tokens and joules on a row all describe a single phase.


### 21. [MEDIUM] A telemetry logger that dies at startup (second tegrastats instance) is never detected; the sweep completes with silently empty energy columns

> **Status:** FIXED 2026-09-14 — the driver now refuses to start when another `tegrastats` is already running (two instances cannot both read the INA3221; the second silences the first), and after the idle baseline it asserts the logger is alive AND that the CSV actually contains samples, aborting with a named cause instead of burning two hours to produce empty energy columns. It prints the ACHIEVED sample rate rather than the requested one, since that is what bounds every integral downstream. A logger that dies mid-sweep is reported at the end, and `idle floor: 0 mW` is no longer printed as though it were a measurement — the absence of idle samples now says so. The abort takes the logger down first, so it cannot leave a stray tegrastats that makes the NEXT run fail for a reason it did not cause.

`bench/run_matrix.py:214` — lens `audit:python`


**Detail.** The driver launches the logger and immediately `time.sleep(30)` without ever checking `logger.poll()`, and the join at line 304 only tests `os.path.exists(power_csv)`. `tegrastats_logger.py` creates and header-writes the CSV before it reads a single line, so the file exists even when tegrastats refuses to start — which is exactly what happens if `heteroserve-telemetry.service` is enabled (the unit file itself warns about this) or any other tegrastats is alive, since tegrastats permits only one instance and exits after printing its complaint. With 0-1 data rows: `idle_floor()` finds no `pow_in_mw` and returns 0.0, printing the reassuring-looking `idle floor: 0 mW`; `energy_between()` hits `if len(sel) < 2: return {}` for every record, so `rec.update({})` adds nothing and `joules`, `duration_s`, `mean_mw`, `temp_rise_c`, `swap_peak_mb`, `joules_per_token` and `active_joules` are absent from every row. Nothing in the run prints a warning, and the final line still says `26/26 runs ok`.


**Failure scenario.** Operator installs the telemetry unit per systemd/heteroserve-telemetry.service, forgets the `systemctl stop` note, and starts a 3-hour sweep in tmux. It reports `26/26 runs ok`, writes a bench_energy.csv with no energy columns whatsoever, and the whole sweep has to be redone.


**Fix.** After the 30 s baseline, assert the logger is alive and actually sampling: `if logger.poll() is not None: sys.exit("telemetry logger exited (is heteroserve-telemetry.service or another tegrastats running? systemctl stop heteroserve-telemetry)")`, and count data rows in `power_csv` (`len(_rows(power_csv)) < 10` after 30 s at >=1 Hz) before proceeding. In the join, replace the bare `os.path.exists(power_csv)` test with a row-count check and print a loud warning (and set `status="no_power"` on the records) when energy could not be computed, so a zero-energy sweep can never masquerade as a successful one.


### 22. [MEDIUM] idle_floor() pools the cold pre-sweep baseline with every hot cooldown, and disagrees with the idle floor the docs tell you to use

> **Status:** FIXED 2026-09-14 — all three definitions were live and mutually inconsistent, and a fourth problem the audit missed decided the shape of the fix: the GGUF pre-read ran BEFORE the run's marker was set, i.e. under the `idle` one, so the floor was measured partly over a multi-watt SD read. `precache()` now carries its own `precache:<case>:<NNN>` marker. The subtrahend is now LOCAL and per case — the mean over the idle interval that ended just before that case opened its window, first 3 s dropped for clock ramp-down — which for the first case IS the 30 s pre-sweep baseline, so there is one definition rather than two. Every row carries `idle_mw`, `idle_mw_source`, `idle_mw_pooled` and `idle_mw_baseline`, so the subtraction is auditable from the archived CSV and an older figure stays reproducible. `telemetry/energy_window.py` gained the twin definition as its `--idle-mw auto` default, and the regression asserts the two agree. Every `max(..., 0.0)` clamp is gone from both tools: a negative active energy means the floor is wrong, and clamping published a plausible 0.0 while discarding the evidence. The node-facts number keeps its own job — cold-board device-table figure — now sampled over 15 s with the first 3 s dropped and carrying `idle_board_mw_basis` that says so.

`bench/run_matrix.py:342` — lens `audit:methodology`


**Detail.** `idle_floor()` averages `pow_in_mw` over ALL rows whose marker is 'idle' in the whole file. That is the 30 s pre-sweep baseline plus every 20 s cooldown across the sweep — by sample count the cooldowns dominate. docs/10_measurement_protocol.md:55 states the floor is 'taken from the 30 s idle baseline the driver collects before the sweep', which is not what the code does. Meanwhile energy_window.py:18 tells the analyst to take `--idle-mw` from node facts `power.idle_module_mw`, which scripts/60_collect_node_facts.sh:32 derives from the last 3 instantaneous POM_5V_IN samples of a 6 s tegrastats run on a cold, otherwise-unloaded board. So the toolkit contains two different, silently inconsistent definitions of the same subtrahend. Physically they must differ: SoC leakage rises with junction temperature, so the pooled hot-cooldown mean sits several hundred mW above the cold baseline. A single scalar is then subtracted uniformly at line 315, over-subtracting from the cold early cases and under-subtracting from the hot late ones — a bias aligned with exactly the case-order axis the sweep varies. The `max(active, 0.0)` clamp at 316 hides the pathological end of this silently.


**Failure scenario.** Pooled floor comes out at ~2.05 W (dominated by cooldowns at 75-85 C); the node-facts floor from a cold board is ~1.60 W. The same power.csv analysed by run_matrix.py and by `energy_window.py --idle-mw 1600` gives active_joules differing by 0.45 W x ~40 s = ~18 J on a ~200 J window, about 9%. The GPU-vs-CPU decode efficiency gap on a 0.5B model is of the same order, so which instrument definition the student happened to use decides the sign of 'is the Maxwell GPU worth its watts' in the energy table.


**Fix.** Define the floor locally and per case: take the contiguous run of idle-marked samples immediately preceding each case's `t_start` (skip the first ~3 s to let clocks ramp down) and use its mean as `idle_mw_local` for that row's `active_joules`; keep the pooled value as `idle_mw_pooled` and the pre-sweep value as `idle_mw_baseline`, and write all three columns so the subtraction is auditable. Make energy_window.py's `--idle-mw` default to the same per-window definition when the CSV contains idle-marked neighbours. Correct docs/10_measurement_protocol.md:55 and energy_window.py's docstring to describe whichever definition survives, and drop the `max(active, 0.0)` clamp in favour of emitting the negative value (a negative active energy is a signal that the floor is wrong, not something to hide).


### 23. [MEDIUM] Every joules-per-token figure includes model-load and warm-up energy in the numerator but not in the token count

> **Status:** FIXED 2026-09-14 — superseded by the finding-#2 work. The numerator is no longer the whole process: it is the trailing compute slice, marker-anchored and interpolated to the exact bounds (`energy_at()`), and the divisor includes the warm-up tokens whose joules are inside that slice. Both sides of the ratio now cover the same work. `energy_basis` states which slice each row used, and `load_s`/`load_joules` report the excluded part rather than hiding it.

`bench/run_matrix.py:398` — lens `audit:ops`


**Detail.** `tokens_of()` returns `(n_prompt + n_gen) * repetitions`, while the energy window is `t_start`/`t_end` around the whole `llama-bench` subprocess (lines 244-252), which covers process start, GGUF load from SD, llama-bench's warm-up iteration, and all repetitions. The function's own docstring claims it is "counting repetitions and the warm-up run llama-bench performs before each measured set" — it is not. The `"warmup": true` key in bench/matrix.json defaults is also never read by `expand()`, so it is dead config. The resulting inflation is not a constant: at repetitions=3 the warm-up alone adds ~33% to the numerator, and model load adds a fixed multi-second, multi-watt block that is a large fraction of a 32-token decode case and a small one of a 1024-token prefill case, and is far larger for qwen2.5-1.5b than for qwen2.5-0.5b.


**Failure scenario.** The cpu_decode_sweep (n_gen=32, r=3, 96 tokens) window includes ~20 s of SD-card model load at ~2-3 W plus a warm-up decode, so joules/token comes out several times the true value, while gpu_prefill_sweep at n_prompt=1024 is barely affected — the prefill-vs-decode energy gap that is the study's headline result is manufactured by the measurement harness rather than the hardware.


**Fix.** Bound the energy window to measured work: emit the marker only around the measured portion (simplest defensible version: run one bench invocation per case with a short pre-pass to absorb model load, or record a separate `load_joules` by marking the interval from t_start until llama-bench's first result line and subtracting it), and make tokens_of account for llama-bench's warm-up iteration explicitly (`per_rep * (repetitions + 1)` when warm-up is on). Either honour `defaults.warmup` in `expand()`/`build_cmd()` or delete the key from bench/matrix.json, and fix the tokens_of docstring to match whichever is implemented.


### 24. [MEDIUM] `sudo nvpmodel -q | grep` under pipefail aborts the whole sweep at pre-flight when sudo cannot prompt

> **Status:** FIXED 2026-09-14 — same bug class as #6. `sudo nvpmodel -q | grep` in `50_bench_device.sh` is now guarded, so a sudo that cannot prompt no longer aborts a two-hour sweep at pre-flight.

`scripts/50_bench_device.sh:16` — lens `audit:shell`


**Detail.** `MODE="$(sudo nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1)"` is a plain assignment under `set -euo pipefail`. If `sudo` cannot authenticate non-interactively it writes nothing to stdout (its diagnostic goes to the discarded stderr), `grep` finds no match and exits 1, pipefail propagates 1 to the assignment, and `set -e` kills the script. The `${MODE:-unknown}` default on line 17 is never reached — the script dies one line earlier. The author guarded the very next sudo call (`sudo jetson_clocks 2>/dev/null || warn` on line 19) correctly, so the inconsistency is clearly unintentional.


**Failure scenario.** The operator follows the script's own advice to survive an ssh drop and launches the 1.5-3 h sweep detached — `nohup make bench &` or from a cron/systemd context rather than inside an interactive tmux pane. There is no tty for sudo's password prompt, `sudo nvpmodel -q` produces no stdout, and the script exits 1 having printed only `── Pre-flight ───`. The operator comes back hours later to an empty results directory and a log containing no error text explaining why.


**Fix.** Make the probe non-fatal: `MODE="$(sudo -n nvpmodel -q 2>/dev/null | grep -A1 'NV Power Mode' | tail -1 || true)"`. The `-n` also makes sudo fail fast instead of hanging on a prompt, and `|| true` lets the existing `${MODE:-unknown}` fallback on line 17 do its job.


### 25. [MEDIUM] `pgrep -x rpc-server` misses the `llama-rpc-server` binary name the toolkit itself supports

> **Status:** FIXED 2026-09-14 — `RPC_BIN_NAMES` in `scripts/lib/common.sh` is now the single list, read by both `bin_rpc()` (binary lookup) and a new `rpc_running()` (contention guard), so a future rename is one edit rather than two that can drift. `rpc_comm_re()` cuts each name to the 15 characters `/proc/<pid>/comm` keeps — verified against real processes: a 16-char `llama-rpc-server` really does report as `llama-rpc-serve`. `rpc_running()` also falls back to `RPC_PORT` via `ss -ltn`, which is the only check that survives the next rename. Deliberately NOT `pgrep -f`: that matches the operator's own `vim 40_run_rpc_server.sh` and the ssh command that started the sweep, and a false warning during ordinary work trains people to dismiss the confirm prompt.

`scripts/50_bench_device.sh:29` — lens `audit:shell`


**Detail.** common.sh line 65 deliberately resolves the RPC binary as either `rpc-server` *or* `llama-rpc-server` (`bin_rpc() { find_bin "$LLAMA_BIN" rpc-server llama-rpc-server; }`), but the contention guard here only tests `pgrep -x rpc-server`. `-x` requires an exact match against `comm`, which the kernel truncates to 15 characters — so a running `llama-rpc-server` appears as `llama-rpc-serve` and matches neither `rpc-server` nor even a literal `llama-rpc-server`. The guard silently returns false and the sweep proceeds with a competing process on a 4 GB, 4-core board.


**Failure scenario.** The operator builds a llama.cpp ref whose example is installed as `llama-rpc-server`, installs systemd/heteroserve-rpc.service (Restart=on-failure, WantedBy=multi-user.target, so it comes back on every boot), then runs `make bench`. The RPC worker is resident, holding its CUDA context and competing for the 4 cores and the shared 4 GB pool, but line 29 does not detect it and no warning or confirm prompt appears. Every throughput and joules-per-token number in that sweep is contaminated by a second inference process, and nothing in the output CSV records that it was running — the contaminated run is indistinguishable from a clean one at analysis time.


**Fix.** Match both names and the port, e.g.:

    if pgrep -f '(^|/)(llama-)?rpc-server' >/dev/null \
       || ss -ltn 2>/dev/null | grep -q ":${RPC_PORT}\b"; then

and consider also checking `systemctl is-active --quiet heteroserve-rpc heteroserve-telemetry` here, since systemd/heteroserve-telemetry.service's own header says the telemetry unit must be stopped before a sweep but nothing enforces it.


### 26. [MEDIUM] GPU devfreq sysfs path does not exist on the Nano; gpu.max_freq_hz is always 0

> **Status:** FIXED 2026-09-14 — the lookup now walks a candidate list (the platform device `57000000.gpu`, the host1x-nested spelling, the `/sys/class/devfreq` symlink, the legacy `gpu.0` path this kit had, then `*.gpu` / `*.gm20b` / `*.gp10b` globs) and takes the first that answers. The resolved path is recorded as `gpu.max_freq_source`, so a 0 Hz reading is distinguishable from a lookup failure, and the uncapped ceiling from `available_frequencies` sits beside it as `gpu.hw_max_freq_hz`.

`scripts/60_collect_node_facts.sh:14` — lens `audit:platform`


**Detail.** `GPU_MAXF="$(cat /sys/devices/gpu.0/devfreq/57000000.gpu/max_freq 2>/dev/null || echo 0)"`. On Tegra X1 / L4T 32.7 the devfreq node lives under the platform device itself: `/sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq` (NVIDIA's Nano power-management guide for L4T 32.7.5 lists exactly this path for min_freq/max_freq/cur_freq). `/sys/devices/gpu.0/` exists on the Nano but carries the legacy nvgpu knobs (railgate_enable, aelpg_enable), not a `devfreq/` subtree. The `2>/dev/null || echo 0` swallows the failure.


**Failure scenario.** `make facts` on a correctly configured Nano writes `"gpu": {..., "max_freq_hz": 0, ...}` into jetson-nano-01_facts.json. The device-characterisation dataset records a 0 Hz GPU for the only GPU node in the cluster, and the 5 W-vs-10 W condition (GPU capped at 640 MHz vs 921.6 MHz) is not captured anywhere in the facts record.


**Fix.** Use the documented path, with a glob so it survives a different device-tree address:

```sh
GPU_MAXF="$(cat /sys/devices/57000000.gpu/devfreq/57000000.gpu/max_freq 2>/dev/null \
  || cat /sys/devices/*.gpu/devfreq/*.gpu/max_freq 2>/dev/null \
  || echo 0)"
```

Expected values on a Nano: 921600000 Hz in nvpmodel mode 0, 640000000 Hz in mode 1.


### 27. [MEDIUM] cpu.max_freq_khz reads cpuinfo_max_freq, which ignores the nvpmodel cap

> **Status:** FIXED 2026-09-14 — `cpu.max_freq_khz` is now `scaling_max_freq`, the cap nvpmodel actually enforces, with the silicon ceiling kept beside it as `cpu.hw_max_freq_khz` (same pair for the GPU, see #26). Demonstrated against fake 5 W and 10 W sysfs trees: the OLD field is identical in both modes, the new one separates 1479000/918000 kHz and 921.6/640 MHz. The live power mode is read from `/var/lib/nvpmodel/status` as `power.nvpmodel_active` beside the configured `power.nvpmodel`, so intent and reality can be compared — deliberately NOT via `sudo nvpmodel -q`, which cannot prompt from a detached sweep (the #24/#6 bug class). The script takes its sysfs root from `HS_SYSFS_ROOT`, which is what makes all of this testable without a Jetson.

`scripts/60_collect_node_facts.sh:13` — lens `audit:platform`


**Detail.** `CPU_MAXF="$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq ...)"`. `cpuinfo_max_freq` is the silicon limit and reads 1479000 on a Nano regardless of power mode; nvpmodel enforces its cap through `scaling_max_freq` (918000 in mode 1), which is the file NVIDIA's Nano power-management guide documents for the per-mode CPU frequency. The facts JSON also records `power.nvpmodel` from the env file (`${NVP_MODEL}`), not from the board, so nothing in the record reflects the mode actually in force when the script ran.


**Failure scenario.** The operator runs the 5 W condition: `sudo nvpmodel -m 1`, then `make facts`. The facts JSON says `"max_freq_khz": 1479000` while the CPU is actually capped at 918 MHz, and `"nvpmodel": "0"` if config/node.env still holds the default NVP_MODEL=0 from config/node.env.example:52. The 5 W energy-per-token rows are then interpreted against a 1.48 GHz, mode-0 device — a wrong recorded experimental condition in the characterisation dataset.


**Fix.** Record both frequencies and read the live power mode from the board:

```sh
CPU_MAXF="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq 2>/dev/null || echo 0)"
CPU_HWMAXF="$(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq 2>/dev/null || echo 0)"
NVP_ACTIVE="$(sudo nvpmodel -q 2>/dev/null | tail -1 | tr -dc '0-9')"
```

Emit `max_freq_khz` (scaling), `hw_max_freq_khz` (cpuinfo) and `power.nvpmodel_active` alongside the configured `power.nvpmodel`, so a mismatch between intent and reality is visible in the dataset.


### 28. [MEDIUM] ExecStartPre's `sudo jetson_clocks` always fails under systemd and is swallowed, so the boot-time worker serves at unlocked clocks

> **Status:** FIXED (verified 2026-09-14) — `ExecStartPre=+/bin/sh -c '/usr/bin/jetson_clocks || echo "CLOCKS-NOT-LOCKED: ..." >&2'`. The `+` prefix runs it as root regardless of `User=` (systemd 231+; L4T R32.7 ships 237), so there is no sudo and no tty to fail on. It is deliberately non-fatal — the `|| echo` means sh always exits 0 — but no longer silent: the warning reaches the journal, and `systemd-analyze verify` passes on the unit. Note that verify exits 0 even for an unescaped `%`, so it is not sufficient evidence on its own.

`systemd/heteroserve-rpc.service:20` — lens `audit:ops`


**Detail.** The unit runs as `User=%USER%` and calls `/bin/sh -c '/usr/bin/sudo /usr/bin/jetson_clocks || true'`. A systemd service has StandardInput=null and no tty, so unless the student has added a NOPASSWD sudoers rule (nothing in this kit does), sudo cannot prompt and exits non-zero; `|| true` hides it and the unit starts anyway. The same applies inside ExecStart's `40_run_rpc_server.sh` lines 30-31, whose `sudo jetson_clocks 2>/dev/null || warn` and `sudo tee /sys/devices/pwm-fan/target_pwm` also fail silently when launched by the unit. jetson_clocks does not survive a reboot (docs/20 says so explicitly), and README line 120 calls it mandatory before any timed run. The dataset then lies about it: `60_collect_node_facts.sh` line 77 emits `clocks_locked` straight from the LOCK_CLOCKS config knob rather than from the board's actual state.


**Failure scenario.** Student enables heteroserve-rpc so the worker comes back after reboots, reboots the Nano, and runs the cluster sweep from the laptop. The Nano serves with a wandering CPU/GPU governor and the fan at default PWM; repetitions disagree by 10-20%, throughput drifts as the board heats, and node_facts.json still records clocks_locked: true — so the contaminated runs look valid in the archive.


**Fix.** Drop sudo and use systemd's root-privilege prefix: `ExecStartPre=+/usr/bin/jetson_clocks` and `ExecStartPre=+/bin/sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm'` (the `+` prefix runs as root and is supported by systemd 237 on Ubuntu 18.04), or ship a sudoers drop-in granting NOPASSWD for jetson_clocks/nvpmodel and have 00_system_prep.sh install it. Also change 60_collect_node_facts.sh to report measured state (e.g. `scaling_min_freq == scaling_max_freq` on cpu0 plus the GPU devfreq min/max) instead of echoing LOCK_CLOCKS.


### 29. [MEDIUM] 5 Hz point sampling beats against the token cadence, and the INA integral is never validated against the inline meter

> **Status:** FIXED 2026-09-14 — the sampling bound now travels on every joined row instead of being underivable: `power_samples`, `sample_gap_max_s` (the longest hole the trapezoid crossed on one straight line) and `pow_in_plateau_frac` (how often the rail repeated its previous reading, i.e. whether the requested rate is real). The shipped default rose from 5 Hz to 10 Hz, `_rows()` is memoised because the join was O(records x rows) and would otherwise get worse as the rate rises, and docs/10 §2 carries a two-minute on-board calibration whose result goes in the experiment log as the stated sampling uncertainty. Two things are now said plainly in both the logger and the protocol rather than left to be assumed: a sample rate does NOT average out with repetitions (every repetition is sampled on the same grid, so a tight spread is not evidence the rate sufficed), and the INA3221's absolute accuracy is unvalidated and unvalidatable here — `joules_vs_mean_pct` compares two arithmetics over the SAME samples, so it catches a bookkeeping error and proves nothing about the samples, and tegrastats' running average is not a second instrument. The aliasing half of the finding is NOT solved and is not solvable with one instrument; it is stated as a limitation instead of being papered over.

`telemetry/tegrastats_logger.py:148` — lens `audit:methodology`


**Detail.** `interval_ms = 1000/hz` with TELEMETRY_HZ=5 (config/node.env.example:48) samples the INA3221 *instantaneous* register every 200 ms, and the trapezoid at lines 199-206 assumes power varies linearly between those points. Decode on this board is a train of short compute bursts at the per-token cadence — 8-15 tok/s means a 67-125 ms period, i.e. the same order as the 200 ms sample interval — so the integral is an aliased estimate whose error is a slowly drifting beat, stable within a run. Repetitions therefore do NOT expose it as variance. Short prefill bursts fare worse: a GPU pp128 repetition lasts on the order of 1-2 s, giving roughly 5-10 samples for the entire measured burst, and the GPU DVFS ramp (tens of ms) is invisible, so `peak_mw` systematically understates the true peak and `pow_gpu_mw` cannot distinguish 'GPU at 40% duty cycle at full power' from 'GPU drawing 40% power'. Compounding this, the running-average field tegrastats already provides is parsed and thrown away (`for name, inst, _avg in RE_POM.findall(line)`, line 102), and nothing anywhere in the toolkit ever compares the INA integral against the external inline meter that docs/10_measurement_protocol.md:33 calls ground truth — so the instrument uncertainty on every joule in the thesis is unmeasured and unreported.


**Failure scenario.** gpu_decode_sweep at n_gen=32 produces ~10 s of decode sampled 50 times. If the sampler happens to land preferentially on the inter-token gaps for that run, window joules come in low; the next case, whose token rate differs slightly, lands preferentially on the bursts and comes in high. The two rows differ by a double-digit percentage with tight within-run repetition spread, so the protocol's 'if repetitions differ by more than a few percent something is wrong' check passes and the aliasing is averaged into the table as a real GPU-vs-CPU difference.


**Fix.** Raise the default to TELEMETRY_HZ=20 (interval 50 ms — already the floor the code enforces at line 148) and, on the first run, check the CSV for long runs of byte-identical `pow_in_mw` values, which would show the rail itself is not updating that fast and would bound the achievable rate. Stop discarding the second POM field: add `pow_in_avg_mw` / `pow_gpu_avg_mw` / `pow_cpu_avg_mw` columns from `_avg` and report window energy computed from the averaged field alongside the point-sample trapezoid; the two agreeing is evidence the sampling is adequate, the two disagreeing quantifies the aliasing. Finally, run one calibration once per campaign — a fixed 10-minute load logged simultaneously by tegrastats and by the inline meter — and record the INA-to-meter ratio and residual in the node facts as the stated instrument uncertainty on every joule figure.


### 30. [MEDIUM] Parked cores counted as 0% halves cpu_util_pct in the 5 W power mode

> **Status:** FIXED 2026-09-14 — an offline core reads as `off` in tegrastats' CPU list, and counting it as 0% busy while dividing by the printed core count halved `cpu_util_pct` in the 5 W mode, which parks two of the four cores: a fully saturated 2-core board reported 49.5%. Utilisation is now averaged over the cores that exist, with `cpu_cores_online` and `cpu_cores_seen` carried so the figure is interpretable, and the per-core string keeps the literal `off` so the power mode stays legible in the row rather than only in `raw`. All cores parked yields `cpu_util_pct=None` — the absence of a measurement, not a measured zero.

`telemetry/tegrastats_logger.py:85` — lens `audit:platform`


**Detail.** In `parse()`, a core field that is not `N%@F` (i.e. `off`) appends `0.0` to `utils`, and `cpu_util_pct` is `sum(utils)/len(utils)` over all four entries. The comment claims this keeps the mean 'comparable across power modes that park cores' — the opposite is true. nvpmodel mode 1 (5 W) on the Nano leaves only 2 of 4 cores online (NVIDIA's mode table: mode 0 = 4 cores @ 1479 MHz, mode 1 = 2 cores @ 918 MHz), and tegrastats prints `CPU [98%@918,97%@918,off,off]`. A fully saturated 5 W board therefore logs cpu_util_pct ≈ 49, while the same saturated workload in mode 0 logs ≈ 98.


**Failure scenario.** The sweep is run twice as the protocol requires (`--tag 5W` and `--tag 10W`). The 5 W rows show ~50% CPU utilisation against ~100% for 10 W, and the write-up concludes the 5 W mode leaves half the CPU idle / is bandwidth-starved, when in fact both conditions ran the CPU flat out on the cores that existed. `cpu_freq_mhz` (max over the online cores) is unaffected, so nothing else in the row contradicts the wrong reading.


**Fix.** Average only over online cores and record how many there were. In `parse()`, drop the `else: utils.append(0.0)` branch (count the core as offline instead), and add an `cpu_cores_online` column:

```python
offline = 0
for core in m.group(1).split(","):
    mm = RE_CORE.match(core.strip())
    if mm:
        utils.append(float(mm.group(1))); freqs.append(int(mm.group(2)))
    else:
        offline += 1
d["cpu_cores_online"] = len(utils)
```

Keep the per-core string reporting `off` for the parked cores so the power mode stays visible in the raw column.

