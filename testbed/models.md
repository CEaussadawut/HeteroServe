# Models on the testbed

Every model is pinned by upstream revision and verified by SHA-256 after transfer.

> **2026-10-19: the Pi 5 is back on a replacement board with the old SD card. All seven files on it were re-hashed there and match their pins** (Mistral-Nemo 12B, Qwen3-14B, Qwen3.5-9B, TinyLlama Q3_K_M / Q4_0 / Q4_K_M / Q8_0; `testbed/results/pi5b_models_sha_20261019_234724.log`, 23:47–23:52). Its tables below say "Pi 5" for the card, which is the same. TinyLlama Q6_K is copied to it from the laptop's cache, and re-hashed, by `run_quant_campaign.sh` on 10-20.
A model that has not been checksummed on the node is not used for a reported figure.

## TinyLlama-1.1B-Chat-v1.0, Q4_K_M

| | |
|---|---|
| Purpose | the fast comparability model — thread sweeps, ISA ablation, link A/B |
| Size | 636.18 MiB (1.10 B params) |
| Repo, revision | **not recorded** (found 2026-10-05). The other TinyLlama files are `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645b…`; this one's revision was never written down. Its bytes are pinned by the SHA-256 below, which matches on every board |
| SHA-256 (first 16) | `9fecc3b3cd76bba8` |
| SHA-256 (full) | `9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0` — `sha256sum` run on all three nodes 2026-09-25, all match |
| Present on | Pi 5, Pi 4, Jetson — **byte-identical**, so any difference between nodes is hardware |

## TinyLlama-1.1B-Chat-v1.0, Q8_0 and Q4_0  *(added 2026-09-27)*

| | Q8_0 | Q4_0 |
|---|---|---|
| Purpose | the quantisation axis of the characterisation dataset (`testbed/dataset/`) | same |
| Repo, revision | `TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` at `52e7645ba7c309695bec7ac98f4f005b139cf465` | same |
| Upstream file | `tinyllama-1.1b-chat-v1.0.Q8_0.gguf` | `tinyllama-1.1b-chat-v1.0.Q4_0.gguf` |
| On the boards as | `~/models/tinyllama-1.1b-q8_0.gguf` | `~/models/tinyllama-1.1b-q4_0.gguf` |
| Size | 1,170,781,568 bytes (1,116.5 MiB) | 637,699,456 bytes (608.2 MiB) |
| SHA-256 (full) | `a4c9bb1dbaa372f6381a035fa5c02ef087aaa1ff1f843a56a22328114f03fc59` | `da3087fb14aede55fde6eb81a0e55e886810e43509ec82ecdc7aa5d62a03b556` |
| Present on | Pi 5, Pi 4, Jetson — byte-identical | same |

The boards have no internet (the router has no WAN), so both were fetched on the laptop into
`~/.cache/heteroserve-models/`, checked there against the SHA-256 Hugging Face publishes for each file
at that revision, copied by `scp`, and **re-hashed on every board** by
`testbed/scripts/run_quant_campaign.sh` before the 2026-09-27 campaign (`campaign.log`: `sha256 OK` for
all three quantisations on all three boards). The Q4_K_M file above was re-hashed in the same pass.

## Downloaded 2026-10-25: a third head layout for the Jetson's attention law

- **Qwen3-4B Q4_K_M** (the user, ~03:10: "yes to the qwen download"): official Qwen GGUF, Apache-2.0, ungated. Study lc4m: it
  has Qwen3-0.6B's KV bytes per layer (8 KV heads x 128) and twice its query dims (32 x 128), so the Jetson's candidate
  attention laws part on it. Fetched 03:14 (`fetch_20261025.log` in the cache): 23 s, its SHA-256 equal to the LFS oid at the
  revision; then copied to the Jetson (55 s) and SHA-256-checked there.

| upstream file (source @ revision) | on the laptop as | size (bytes) | SHA-256 (full) | present on |
|---|---|---:|---|---|
| `Qwen3-4B-Q4_K_M.gguf` (`Qwen/Qwen3-4B-GGUF` @ `bc640142c66e1fdd12af0bd68f40445458f3869b`, Apache-2.0) | `~/models/qwen3-4b-q4_k_m.gguf` (links into the cache) | 2,497,280,256 | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` | laptop cache; Jetson `~/models/qwen3-4b-q4_k_m.gguf` (verified 2026-10-25 03:20) |

## Downloaded 2026-10-21: the newest Qwen prima.cpp can run, its precision variants, and a model that needs every node

- **Qwen3.6-35B-A3B Q6_K** (user, ~17:15: "Q6_K, 31.0 GB"): the four-node evaluation's model that needs every node; a qwen35moe
  hybrid with one MTP block (not loaded at the pin). Fetched 20:01-20:28 (`.handoff/assets/models-2026-10-21/fetch_20261021.sh`).
- **Qwen3-32B Q8_0** (user, ~20:45: "Yes, download it"): dense qwen3, the newest Qwen prima.cpp's distributed path can run;
  for the engine against prima.cpp + Halda on all four nodes. Fetched 20:35-20:58 (`fetch_qwen3_32b_q8_20261021.sh`).
- **Qwen3-32B Q6_K and Q4_K_M** (user, ~23:40: "Both files"; "near-lossless only"): scheduler v2's precision choice (Q6_K
  servable, Q4_K_M the quality reference). Fetched from 23:45 to the laptop's data drive (`/mnt/data/heteroserve-models`: `/`
  had 24 GB free) by `fetch_qwen3_32b_q6k_q4km_20261021.sh`; `~/models` links to them.
- **The boards' partial copies of the Q8_0** (`~/models/qwen3-32b-q8_0.prima-partial.gguf`, prima.cpp only; the user's choice,
  ~20:45, "Partial copies"): sparse files holding the header and the layers Halda can give each board, every present range
  checked against the laptop's verified file. **An exception to the whole-file rule above:** they are verified range by range,
  manifest `testbed/dataset/prima_copies_qwen3-32b-q8_0.json` (Pi 5 layers 0-12, Pi 4 9-22, Jetson 19-26).

| upstream file (source @ revision) | on the laptop as | size (bytes) | SHA-256 (full) | present on |
|---|---|---:|---|---|
| `Qwen_Qwen3.6-35B-A3B-Q6_K.gguf` (`bartowski/Qwen_Qwen3.6-35B-A3B-GGUF` @ `5c2410d71524f4f72b023ce8daf7a80528226d5f`) | `~/models/qwen3.6-35b-a3b-q6_k.gguf` | 30,947,309,184 | `a97ef79439194d8ec722bc0b6c316f0c41c0b477b35bbae8177c772ecbb0eb91` | laptop cache |
| `Qwen3-32B-Q8_0.gguf` (`Qwen/Qwen3-32B-GGUF` @ `938a7432affaec9157f883a87164e2646ae17555`, Apache-2.0) | `~/models/qwen3-32b-q8_0.gguf` | 34,817,718,912 | `2c50eb8aad05047dbf24fa014eb621adf552e14176cabe0c5db4ef38c91e2169` | laptop cache; partial copies on the boards (above) |
| `Qwen3-32B-Q6_K.gguf` (the same repo and revision) | `~/models/qwen3-32b-q6_k.gguf` | 26,883,306,112 | `c4c7c3cb6da260df1fe1d3cfd090a32dc7cc348f1278158be18e301f390d6f6e` | laptop data drive (see the fetch log) |
| `Qwen3-32B-Q4_K_M.gguf` (the same repo and revision) | `~/models/qwen3-32b-q4_k_m.gguf` | 19,762,149,024 | `efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689` | laptop data drive (see the fetch log) |

## Downloaded 2026-10-11: a model that needs the pool even with the laptop as a node

The user made the laptop a node on 2026-10-11 and asked for the full evaluation to start "with a model that needs the pool
even with the laptop". No model on file does: with 8 GiB kept for the desktop the laptop holds each one alone (the largest,
the 30B MoE at Q3_K_M, needs 14.4 GiB). The user chose the same MoE at Q6_K (~03:44, "Qwen3-30B-A3B at Q6_K"), 23.4 GiB,
and with it the TinyLlama Q6_K file, because the planner prices a model by the TinyLlama row at its quantisation.
- **Fetched:** on the laptop into `~/.cache/heteroserve-models/` by
  `.handoff/assets/models-2026-10-11/fetch_20261011.sh`, each from its pinned revision's `resolve` URL; size and SHA-256
  checked against the LFS oid and size Hugging Face publishes at that revision before the file got its final name
  (`fetch_20261011.log` in the cache; its "~03:55" approval time is a slip for ~03:44).
- **The laptop's `~/models`** (made 2026-10-11): links into the cache under the boards' short names, so every
  `predict_pool.MODELS` path holds on the laptop as on a board.

| upstream file (source @ revision) | on the boards as | size (bytes) | SHA-256 (full) | present on |
|---|---|---:|---|---|
| `tinyllama-1.1b-chat-v1.0.Q6_K.gguf` (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645ba7c309695bec7ac98f4f005b139cf465`, the other TinyLlama rows' revision, Apache-2.0) | `~/models/tinyllama-1.1b-q6_k.gguf` | 904,385,920 | `9fb2be7c8bbc0ca358ae88cef6d9e6844b81929eaffa5016a543e6f56b22cb6d` | laptop cache (verified 03:47:49); Pi 4 and Jetson (copied and re-hashed by `run_quant_campaign.sh`, 2026-10-11) |
| `Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf` (`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` @ `eea7b2be5805a5f151f8847ede8e5f9a9284bf77`, the Q3_K_M file's source and revision, Apache-2.0) | `~/models/qwen3-30b-a3b-q6_k.gguf` | 25,092,532,640 | `b1a3ed3bc584d58d25253c408cbbf894f974dffa2df74e016469289e686ef701` | laptop cache (see the fetch log) |

## Downloaded 2026-10-10: the speculative-decoding draft model (scheduler upgrade 1)

The user approved one draft model for upgrade 1 on 2026-10-10 at 01:01 ("Qwen3-0.6B only"; the EAGLE-3 head was not
approved).
- **Fetched:** on the laptop into `~/.cache/heteroserve-models/`, from the pinned revision's `resolve` URL. Its size and
  SHA-256 were checked against the LFS oid and size Hugging Face publishes for the file at that revision before it got its
  final name (`fetch_20261010.log` in the cache).
- **Copied:** to the Pi 4 by `testbed/scripts/copy_model_from_laptop.sh` and re-hashed there. It goes on the Pi 4 because
  llama-server loads a draft model on its host, whichever device then computes it.
- **Why it can draft for Qwen3-14B:** its vocabulary matches Qwen3-14B's, which llama.cpp requires
  (`common/speculative.cpp`: within 128 tokens, with the same token text from id 5).

| upstream file (source @ revision) | on the boards as | size (bytes) | SHA-256 (full) | present on |
|---|---|---:|---|---|
| `Qwen3-0.6B-Q8_0.gguf` (`Qwen/Qwen3-0.6B-GGUF` @ `23749fefcc72300e3a2ad315e1317431b06b590a`, Apache-2.0) | `~/models/qwen3-0.6b-q8_0.gguf` | 639,446,688 | `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031` | laptop cache (verified 2026-10-10 01:06); Pi 4 (verified 01:11, 12 s copy) |

## Downloaded 2026-09-28 (the picks approved 2026-09-27)

At the user's request ("let's download all the aimed model"), every pick in `testbed/dataset/model_manifest.json` was
fetched on the laptop into `~/.cache/heteroserve-models/` by `.handoff/assets/models-2026-09-28/fetch_models.py`. Each
file came from its pinned revision's `resolve` URL, and its size and SHA-256 were checked against the manifest's
recorded values before it got its final name (the log is `fetch_20260928.log` in the cache). The boards have no
internet, so each file was then copied over the wired LAN, by `testbed/scripts/copy_model_from_laptop.sh` or by
`run_quant_campaign.sh` for the TinyLlama file, and **re-hashed on the board** before it was renamed into place. All are
Apache-2.0.

| upstream file (source @ revision) | on the boards as | size (bytes) | SHA-256 (full) | present on |
|---|---|---:|---|---|
| `Mistral-Nemo-Instruct-2407-Q4_K_M.gguf` (`bartowski/Mistral-Nemo-Instruct-2407-GGUF` @ `a2dd64a0a76ea1bdb2bb6ab6fa5496b003c7c908`) | `~/models/mistral-nemo-12b-q4_k_m.gguf` | 7,477,208,192 | `7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46` | Pi 5 (verified 2026-09-28 15:36) |
| `Qwen3-14B-Q4_K_M.gguf` (`Qwen/Qwen3-14B-GGUF` @ `530227a7d994db8eca5ab5ced2fb692b614357fd`) | `~/models/qwen3-14b-q4_k_m.gguf` | 9,001,752,960 | `500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0` | Pi 5 (verified 15:41); **Pi 4** (copied from the Pi 5 2026-09-30 05:06, sha256 OK on arrival, `copy_pi5-pi4_20260930_045738.log`: prima.cpp loads a local copy on every device) |
| `Qwen3-30B-A3B-Instruct-2507-Q3_K_M.gguf` (`unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF` @ `eea7b2be5805a5f151f8847ede8e5f9a9284bf77`) | `~/models/qwen3-30b-a3b-q3_k_m.gguf` | 14,711,847,328 | `e145c9d2f5d11c9583eb099aa75100b7ab943e77d5240c9a2cd936f81c89ef43` | Pi 4 (verified 2026-09-28 15:58, `testbed/results/copy_laptop-pi4_20260928_155136.log`: the first copy's on-board check timed out while the Pi 4 was still writing the file to its SD card, so the full-size `.part` was re-hashed there instead of copied again) |
| `tinyllama-1.1b-chat-v1.0.Q3_K_M.gguf` (`TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF` @ `52e7645ba7c309695bec7ac98f4f005b139cf465`, the other TinyLlama rows' revision) | `~/models/tinyllama-1.1b-q3km.gguf` | 550,819,200 | `ec461c4d2b60896152bca3978aa49cd70edad7298715e4222e12af1eefea2125` | Pi 5, Pi 4 (verified 15:27), Jetson (verified 15:41) |
| `Qwen3.8-27B-Q3_K_M.gguf` (`bartowski/Qwen3.8-27B-GGUF` @ `0c92138c51f112d2f0dc84d5f6d1ebe4b9912b9a`; added 2026-09-28 night, user: "let's download the 27B and let's benchmark it out") | `~/models/qwen3.8-27b-q3_k_m.gguf` | 13,404,051,168 | `14865aaef0b6c26a013538782d366443c7298b11d5df9f0fb829d34d1963f869` | laptop cache (verified). It was on the Pi 4 (verified 2026-09-28 23:12, `testbed/results/copy_laptop-pi4_20260928_225907.log`) and was removed there by the user on 2026-09-29, after its benchmark, to make room for Qwen3.5-9B |
| `Qwen_Qwen3.5-9B-Q4_K_M.gguf` (`bartowski/Qwen_Qwen3.5-9B-GGUF` @ `182be2fd6c7bc44887d88a91cb03ff009cc9f549`; added 2026-09-28 night, user: "ok let's go with Qwen3.5-9B") | `~/models/qwen3.5-9b-q4_k_m.gguf` | 6,169,341,984 | `d784ce9eda1a5a7b51e8f705a9e6310844bf4f173654d115823c775fdea56d43` | Pi 5 (verified 2026-09-28 23:4x, `testbed/results/copy_laptop-pi5_*.log`); laptop cache |

**Storage**, as the manifest planned: the Pi 5 holds Mistral-Nemo and Qwen3-14B, the Pi 4 the MoE. No file was deleted.
Our pin supports `qwen3` and `qwen3moe`.

**What the tensor tables show** (`testbed/dataset/model_shapes.json`, read with `ThinkPad/control/gguf.py`):
- **Mistral-Nemo 12B:** 40 blocks of 156 MiB, plus a 525 MiB head and a 360 MiB embedding. The manifest called it "just
  too big for one Pi 5", but by the table **it fits the Pi 5 alone**: about 6,985 MiB needed against 7,619. The pick was
  sized from its file size.
- **Qwen3-14B:** 40 blocks of 188.8 MiB, plus a 609 MiB head. It needs two or more boards, as planned.
- **Qwen3-30B-A3B:** 48 blocks of 284.5 MiB with all 128 experts, which is what memory must hold. A decoded token reads
  only **27.1 MiB** per block: the attention, the router and 8 of the 128 experts. It has about 3.3 B active parameters
  of 30.5 B.

## Mistral-7B-Instruct-v0.2, Q8_0

| | |
|---|---|
| Purpose | the single-node capacity probe |
| Size | 7.70 GB / **7.17 GiB** |
| Present on | **laptop cache** (`~/.cache/heteroserve-models/mistral-7b-q8_0.gguf`, sha256 re-verified 2026-09-29 19:07, `testbed/results/copy_boards-laptop_20260929_190251.log`). **Dropped from the study 2026-09-29** (user: "delete them both you recommended"): Mistral-Nemo 12B covers "fits the Pi 5 alone" for a dense model. The Pi 5's copy was archived first, and the user removed it: absent from the Pi 5's `~/models` on 2026-09-30 01:30. Every result below stays valid and re-runnable from the laptop copy |
| Result | **fits one Pi 5** with ~250 MiB to spare, no swap, zero OOM records, at 1.51 tok/s decode — so it cannot serve as the pooling proof, and it sits below the QR2 floor of 2 tok/s |
| Repo, revision | `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` at `3a6fbf4a41a1d52e415a4958cde6856d34b2db93` (Apache-2.0) *(added 2026-09-27)* |
| Upstream file | `mistral-7b-instruct-v0.2.Q8_0.gguf`, on the Pi 5 as `~/models/mistral-7b-q8_0.gguf` |
| Size | 7,695,857,952 bytes |
| SHA-256 (full) | `f326f5f4f137f3ad30f8c9cc21d4d39e54476583e8306ee2931d5a022cb85b06` |
| Verified | 2026-09-27: `sha256sum` on the Pi 5 matches, to the byte, the LFS oid and size Hugging Face publishes for the file at that revision (read from its API, not re-downloaded). Until then the file had no recorded provenance, and the 1.51 tok/s above predates the `-r 3` rule: treat it as indicative, not reportable |

## Qwen2.5-14B-Instruct, Q4_K_M  *(added 2026-09-20)*

| | |
|---|---|
| Purpose | **the pooling proof** — the smallest configuration that fits no single node |
| Repo | `Qwen/Qwen2.5-14B-Instruct-GGUF` |
| Revision | `b466e1f8c07172155743e8e1307507d8a4f91fbd` |
| Licence | **Apache-2.0** — matches the proposal's model policy |
| Layout | split GGUF, 3 parts, loaded by passing part 1 |
| Size | 8,988,110,496 bytes = **8,572 MiB = 8.37 GiB** |
| Present on | **laptop cache** (`~/.cache/heteroserve-models/qwen2.5-14b-instruct-q4_k_m-0000{1,2,3}-of-00003.gguf`, all three sha256 re-verified 2026-09-29 19:07 against the lines below, copied from the Pi 4, `testbed/results/copy_boards-laptop_20260929_190251.log`). **Dropped from the study 2026-09-29** (user: "should i cut out qwen2.5 ... since we already have 3-14b?", then "delete them both you recommended"): Qwen3-14B covers the dense-14B case, plans the same (Pi 5 + Jetson 5) and measures within 1.3 %. The Pi 5's and the Pi 4's copies were archived first, and the user removed them: absent from both boards' `~/models` on 2026-09-30 01:30 (the Pi 5's disk 87 % → 59 %). The QR1 proof, the 2026-09-27 campaign and the Pi 4 host limit stay valid and re-runnable from the laptop copy |

Why this model: it is the QR1 **stretch** target ("14B at Q4") and the ladder in the
proposal is Qwen2.5 0.5B / 1.5B / 7B / 14B, all Apache-2.0. 7B Q8_0 was the intended
MVP proof but was measured to fit one node, so 14B Q4 is the smallest honest case.

Capacity arithmetic, measured at rest with swap off:

| node | MemAvailable | holds 8.37 GiB alone? |
|---|---:|---|
| Pi 5 | 7,629 MiB (7.45 GiB) | **no** |
| Pi 4 | 7,373 MiB (7.20 GiB) | **no** |
| pooled | 15,002 MiB (14.65 GiB) | yes, with headroom |

### SHA-256, verified on the Pi 5 against upstream LFS metadata

    a09ea5e7b1eafb1b30b241726c3cc3c905c96f14ad41e246ffa5f44e53904f68  ...-00001-of-00003.gguf
    21b9457d079680d284e90ef69607c4b2d8ef64a09d4729cb7b5e1357bdba41ae  ...-00002-of-00003.gguf
    c8d37006760a387a35216e070e6664d7da927f10be8eb870fef2e3d4833d9976  ...-00003-of-00003.gguf

All three `OK` on 2026-09-20. Sizes also match upstream exactly.

Re-verified **2026-09-25**, before the `-r 3` re-run: all three `OK` on the Pi 5 again
(`sha256sum -c`, 1 m 50 s), and the upstream LFS oids at `b466e1f8…` re-fetched from the
Hugging Face tree API — identical to the three lines above. The node's
`~/models/qwen14b.sha256` was generated from the downloaded files, so checking against it
alone would only prove the bytes have not changed since download; the upstream fetch is
what makes the check non-circular.
