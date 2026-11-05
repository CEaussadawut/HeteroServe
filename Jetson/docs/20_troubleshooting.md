# Troubleshooting — the failures this hardware actually produces

## The CUDA build fails with errors inside `ggml-cuda`

**Cause.** Stock JetPack 4.6 ships CUDA 10.2; current llama.cpp targets CUDA ≥ 11
and uses intrinsics and APIs that nvcc 10.2 does not have.

**What to do.** Walk llama.cpp's tags backwards until one configures and
compiles, then pin it:

```bash
cd ~/heteroserve/llama.cpp && git tag --sort=-creatordate | head -50
cd ~/HeteroServe/Jetson
./scripts/20_build_llama_cpp.sh --ref <tag> --clean
```

Record the ref that works in `config/node.env` as `LLAMA_CPP_REF`. It ends up in
`build_info.json` and therefore in every result file.

If nothing recent builds, go CPU-only (`--cpu-only`). Say so in the report: the
Nano then participates as a Cortex-A57 worker, and the reason is a real
constraint of legacy edge hardware, which is the study's subject matter.

## `nvcc` refuses the host compiler

CUDA 10.2 supports host gcc up to 8. `00_system_prep.sh` installs `gcc-8`/`g++-8`
and `node.env` sets `HOST_CXX=g++-8`. If you changed it, change it back.

## The build is killed part-way through

The OOM killer. `nvcc` on 4 GB cannot sustain `-j4`.

```bash
free -h                      # confirm the swapfile from 00_system_prep.sh is on
./scripts/20_build_llama_cpp.sh -j 1
```

Also make sure you rebooted after the desktop was disabled — the GUI holds
several hundred megabytes that the compiler wants.

## `cmake` says the version is too old

Ubuntu 18.04 ships CMake 3.10. `10_install_toolchain.sh` installs a current one
into `~/.local/bin`. Open a new shell, or `export PATH="$HOME/.local/bin:$PATH"`.

## `rpc-server` was not built

`GGML_RPC` did not take effect on the ref you pinned — older refs spell it
`LLAMA_RPC`, and some very old ones have no RPC example at all. The options were
renamed `LLAMA_*` -> `GGML_*` on 2024-06-26 (PR #8006). The build script detects
which era the checked-out source is from and passes **one** spelling; it logs the
choice as `llama.cpp CMake option era: ggml|llama`. If the binary is still
missing, the ref predates the RPC backend and you need a newer one.

Do not "just pass both spellings" as a shortcut. Post-rename `CMakeLists.txt`
carries `llama_option_depr(FATAL_ERROR LLAMA_CUBLAS GGML_CUDA)`, and
`--no-warn-unused-cli` cannot suppress a `message(FATAL_ERROR ...)` — so adding
`-DLLAMA_CUBLAS=ON` stops configure dead on any modern ref.

## The orchestrator connects but the Nano stays idle

You forgot `-ngl`. Without layers assigned to the remote backend, the RPC
servers connect and then do nothing — which looks exactly like success. Use
`-ngl 99` (or a measured partial value) on the client.

## The distributed run dies with an allocation failure on the Nano

The orchestrator handed this node a partition larger than its share of the 4 GB.

- Lower `RPC_MEM_MB` in `config/node.env` so the node advertises less.
- Or, for the deliberate over-commit experiments only, set
  `CUDA_UNIFIED_MEMORY=1` so the partition spills into swap instead of failing.
  **Label those runs** — they are not comparable to the default condition.

## Throughput drops as a sweep progresses

Thermal throttling. Check `temp_rise_c` and `temp_max_c` in the sweep output.
Fit a fan, raise `cooldown_s` in `bench/matrix.json`, and re-run. A table whose
first rows ran at full clock and whose last rows ran throttled is not a table.

## Repetitions of the same case differ by more than a few percent

`jetson_clocks` was not applied (it does not survive a reboot), or another
process — commonly a leftover `rpc-server` — is competing.

```bash
sudo jetson_clocks --show
pgrep -a rpc-server
```

## `tegrastats` shows no `POM_*` fields

The carrier board's INA3221 is not being read. Some third-party carrier boards
omit it entirely. This node then has **no power signal at all** — there is no
external meter to fall back on — so every energy result in the study depends on
this working. Check `ls /sys/bus/i2c/drivers/ina3221x/` (the B01 devkit enumerates
`6-0040`). If it is genuinely absent, set `power.rail_source` to `none` in the
node facts so the gap is visible in the dataset, and treat the run as
timing-only: no joules, no joules-per-token.

## `make models` says a file is truncated, or a sha256 changed

Both are the fetcher working. A `.part` file means a download stopped partway and
was never promoted to the real name — re-run and it resumes. `sha256 CHANGED
since it was recorded` means the bytes on disk differ from what `SHA256SUMS` says,
which is either a corrupt SD card or a file replaced by hand; delete it and
re-fetch. A `.bad` file is one that failed verification against the manifest and
was quarantined rather than deleted, so you can inspect it.

If you see `present ... but UNVERIFIED`, the model has no sha256 in
`config/models.tsv` and the server could not be reached for a size check, so
nothing confirmed it. Re-run online before trusting a sweep made with it.

## `make facts` succeeds but the JSON is empty

Fixed, but worth knowing the shape: the script gathers facts through shell
pipelines under `set -euo pipefail`, and any stage that fails — `timeout` on a
command that never exits, `ip route get` with no default route, `ethtool` on an
unknown interface — used to abort it before it printed anything. Piped into
`tee`, that looks like success. Every such pipeline is now guarded; see the note
at the top of `scripts/lib/common.sh` before adding another.

## A late case is slower than an early one for no visible reason

Check `temp_drift_c` and `cpu_freq_min_mhz` on the two rows before blaming the
axis under test. A fanless Nano ratchets upward across a long sweep, and a case
that ran clamped looks exactly like a case that is genuinely slower. The
drift-control pair — the first case repeated as the last run, `drift_control=1` —
is the direct measurement: if those two rows differ, the difference is drift, and
it is an upper bound on how much of any other delta is drift too.

`cooldown_reached=0` on a row means the cooldown gate hit `max_cooldown_s` with
the board still above `cooldown_target_c`. A run of those means the sweep
outpaced the heatsink; raise `max_cooldown_s`, or accept and report it.

## iperf3 reports ~95 Mbit/s on a "gigabit" link

The port negotiated to 100 Mbit — nearly always a bad cable or a switch port.
Over 100 Mbit, RPC tensor transfer dominates every split run and your
wired-vs-Wi-Fi comparison becomes meaningless.

```bash
ethtool eth0 | grep -i speed
```

## The SD card fills up

GGUFs, a llama.cpp build tree and months of CSVs. Move `HS_ROOT` to a USB 3.0
SSD and re-run `make build`.
