# HeteroServe

**Where should the prefill, decode and layers of a large language model run across a cluster of cheap, mismatched
devices, so that latency, throughput and energy per token come out right together?** HeteroServe answers by measurement.
This repository holds its code and every measurement. It is a senior project in Computer Engineering at King Mongkut's
Institute of Technology Ladkrabang (KMITL), project CE69-43.

The cluster is a ThinkPad T580 laptop, a Raspberry Pi 5 (8 GB), a Raspberry Pi 4 (8 GB) and a Jetson Nano (4 GB). They
are pooled with llama.cpp's RPC backend at one pinned revision (`b23701f7`). Energy is measured at the wall on all four
outlets, at 5 Hz, by a metering power strip.

This release is from 6 November 2026 (tag `v2026.11.06`) and holds every study through 5 November 2026. The project
continues until April 2027.

## How a study works

Each study is registered before it runs. A read-only prediction file, pinned by SHA-256, says what the planner expects and
what counts as holding. The run follows under `systemd-run`, with its raw output kept. A score then marks each test HOLDS,
FAILS or INCOMPLETE. Failed predictions stay in the record (`testbed/benchmarks.md`) as found.

## Some results

Each result below is a study in `testbed/benchmarks.md`. Its registration is in `testbed/dataset/` and its raw data in
`testbed/results/`.

- **Speculative decoding without a draft model (sp12, 5 Nov).** Qwen3.6-35B-A3B Q6_K is 31 GB, more than any one of the
  devices has free. It ran pooled across all four, with its own multi-token-prediction head drafting one token per check.
  That was faster on all 12 test prompts, by 1.01–1.19× (geometric mean 1.149). The cost model was within 3.3 % on
  every prompt. Across the 3 Nov runs on pooled 35B plans (sp10, sp10c), the best drafting setting for each plan gained
  1.12–1.25× on free text and 1.25–1.43× on copying.
- **The same on Qwen3.5-9B (sp11, 4 Nov):** 1.07–1.23× and 1.17–1.34× faster on its two pooled plans (geometric means
  1.155 and 1.274).
- **Switching idle boards off (so4, 24 Oct):** cutting the idle Pi 4's outlet saved 3.41 J per token (5.7 %), with speed
  unchanged.
- **Against prima.cpp + Halda (23 Oct):** this was on all four nodes with Qwen3-32B Q8_0.
  - At this engine's own memory reserves, prima.cpp was 12 % faster and used 6 % less energy.
  - Given Halda's memory budget on the laptop, this engine was 3.5 % faster, with 3.5 % less energy.
- **A Raspberry Pi 5 network stall (3 Nov):** pooled serving hung because the Pi 5's network driver (macb), with hardware
  TCP segmentation offload on, held RPC replies it had already reported as sent. With TSO off, three of three rounds held
  (`testbed/results/pi5_tx_stall_20261103/NOTE.md`).

## What is here

| Path | What it is |
|---|---|
| `testbed/README.md` | Start here for the cluster: how it runs and where everything is |
| `testbed/REPRODUCE.md` | How to reproduce, at three levels: re-derive, re-run on boards, reuse the data |
| `testbed/benchmarks.md` | The record: every study, with its verdicts as found |
| `testbed/models.md` | Every model file, pinned by upstream revision and SHA-256 |
| `testbed/dataset/` | The registrations (`*predictions*.json`, read-only, each pinning its inputs by SHA-256), the characterisation dataset, and the link and memory measurements |
| `testbed/results/` | Every run's raw output (logs, power at 5 Hz, throttle polls) and its score |
| `testbed/scripts/`, `testbed/tests/`, `testbed/patches/` | The harness, the planner (`predict_pool.py`), the scorers, their tests, and the patches applied at the pin |
| `ThinkPad/`, `Pi5/`, `Pi4/`, `Jetson/` | Per-node scripts, configuration, docs and tests (`ThinkPad/config/cluster.json` defines the cluster) |
| `.handoff/assets/{pd-2026-09-29,placement-2026-09-30,prima-2026-09-30,qr3-2026-09-30}/` | Four scorer folders that `reproduce.py verify` runs |
| `laptop_home/logs/` | The laptop's `~/logs`: its llama-server and rpc-server logs, from which the A/B scorers read each arm-run's server log |
| `laptop_home/.cache/heteroserve-models/README.txt` | The model files `verify` reads: named and pinned, not shipped |
| `make_release.py` | The script that built this folder from the project |
| `READ_ONLY_FILES.txt` | The files that are read-only in the record (the registrations and three others) |

Left out:
- the working journal (it names personal accounts); `testbed/benchmarks.md` is the public record;
- backups, the project's documents (thesis, reports, paper) and the planning notes.

## Re-derive every published result (no boards needed)

Tested with Python 3.14.4 and numpy 2.3.5. Some node scripts also use `lxml`, `matplotlib` and `pyserial`.

1. Git doesn't keep read-only modes, and `verify` checks that every registration is read-only. After cloning, make the
   227 files listed in `READ_ONLY_FILES.txt` read-only again:

   ```bash
   while IFS= read -r f; do chmod a-w "$f"; done < READ_ONLY_FILES.txt
   ```

2. Download the model files that `laptop_home/.cache/heteroserve-models/README.txt` names, each at its pinned revision.
   Two go in that folder. The three hybrid models go in `laptop_home/models/`.
3. From this folder, run:

   ```bash
   HOME=$PWD/laptop_home PYTHONUSERBASE=$(python3 -m site --user-base) python3 testbed/scripts/reproduce.py verify
   ```

It re-runs every scorer on scratch copies of the raw data and compares each published value. On the project laptop, run
from this release with all five model files, it gave **328 checks, 0 failing**, with the record untouched (6 November
2026, 02:47–03:49). That equals the project folder's own count on 5 November. A fresh clone of this repository, on another
filesystem and with step 1 done, matched on all 328 as well (6 November 2026). That was after two fixes the clone turned
up, both in this release: a rebuilt CSV's row order, and scratch paths when the clone sits in `/tmp`.

Without the three hybrid models, the check of `testbed/dataset/hybrid_shapes_20261101.json` is skipped. Without the two
other files, two checks report an error. Every other check stands.

Notes on the command:
- `HOME` points the scorers at the laptop's logs and model cache, as they were when the results were scored.
- `PYTHONUSERBASE` keeps your own installed packages visible.
- Paths under a home folder are compared relative to it, so the different `HOME` is not a difference.

`reproduce.py export` writes the CSVs: software version, settings, temperatures and time per run. `reproduce.py manifest`
pins the files by SHA-256 (`testbed/dataset/MANIFEST.json`, 2,230 files).

## Re-run on hardware

See `testbed/REPRODUCE.md`. Each study has three steps, each a script:
1. register: a read-only prediction file, written before anything runs;
2. run: a session under `systemd-run`;
3. score: HOLDS / FAILS / INCOMPLETE.

Boards of the same model can differ: the replacement Pi 5 decoded 5–15 % slower than the original. Characterise your own
boards first.

## Licence

- **Code** (`*.py`, `*.sh`, the patches, configuration, tests and service files): the MIT licence, in `LICENSE`.
- **Data and documentation** (`testbed/dataset/`, `testbed/results/`, `testbed/energy/`, `laptop_home/logs/` and the
  Markdown documents): Creative Commons Attribution 4.0 International (CC BY 4.0), in `LICENSE-DATA`.

The patches in `testbed/patches/` modify llama.cpp and prima.cpp, which keep their own licences. The model files are not
included; each keeps its publisher's licence.

## What was changed for publication

The home Wi-Fi network names and the routers' MAC addresses were replaced, in 64 files, with placeholders. The networks read
`WIFI-A`, `WIFI-B` and `WIFI-C`, and the routers' MACs `02:00:00:00:00:01`–`06`. Public Wi-Fi maps can turn either into a
location.

Two kinds of SHA-256 no longer match their files:
- `testbed/energy/strip.json` is one of the changed files. The 14 registrations that pin it by SHA-256 pin the original,
  which the author keeps.
- Two registrations, `energy_predictions_20261005.json` and `energy_predictions_20261005b.json`, had a network name in a
  note. Logs of earlier verify runs in `testbed/results/` quote their original hashes.

No other file of the record was edited, apart from `testbed/dataset/MANIFEST.json`, regenerated for this release. Private
(192.168.x.x) addresses, device MAC addresses and user names are as recorded. They mean nothing outside the home network.

## Citing

`CITATION.cff` has the details; GitHub shows them under "Cite this repository".

Aussadawut Phathong, *HeteroServe: code and data*, KMITL senior project CE69-43, release v2026.11.06, 6 November 2026.
https://github.com/CEaussadawut/HeteroServe
