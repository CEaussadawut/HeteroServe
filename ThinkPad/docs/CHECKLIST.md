# Control node — bring-up and per-session checklist

## A. One-time bring-up

- [ ] Ubuntu LTS, lid-close suspend disabled, running on mains
- [ ] hostname `t580`, static `192.168.50.10`, wired into the managed switch
- [ ] `make bootstrap` — venv, `heteroctl` installed, `dialout` group, chrony **as server**
- [ ] logged out and back in (dialout group takes effect)
- [ ] SSH keys copied to every node; `ssh pi4 true` works without a prompt
- [ ] `chronyc clients` lists the workers
- [ ] `make build` — llama.cpp with AVX2 + RPC; SHA recorded
- [ ] **llama.cpp revision pinned everywhere** — `llama.pin_sha` in `cluster.json`, `LLAMA_PIN_SHA` in Pi4/ and Pi5/ `node.env`, `LLAMA_CPP_REF` in Jetson/ `node.env` (three spellings, one value; choose it on the Jetson first — CUDA 10.2)
- [ ] `make models` — GGUFs downloaded, checksums saved, shapes printed
- [ ] `make deploy` — every node folder pushed, exec bits restored
- [ ] `make status` — every agent answers
- [ ] `make flash-daq` — Pico 2 running the DAQ firmware
- [ ] collector unit enabled
- [ ] `make test` — all tests pass
- [ ] `make preflight` — READY

## B. Power instrumentation (once, then re-verify monthly)

- [ ] four INA226 high-side on 0.01 Ω shunts, addresses 0x40/41/44/45
- [ ] `daq.channels` in `cluster.json` matches the wiring **and** each node's `ina_channel`
- [ ] one node loaded at a time confirms the right channel moves
- [ ] every channel calibrated at 2–3 points (`I = V/R`, both values measured)
- [ ] `calibration/ch*.json` committed with the data
- [ ] idle and load readings inside the sanity bands in `docs/03_energy_collector.md`
- [ ] measurement boundary written into the report (fan included; T580 excluded)

## C. Before every measurement session

- [ ] `make preflight` → READY
- [ ] collector running; CSV growing
- [ ] `heteroctl daq sync` — offset spread ~1 ms, not tens
- [ ] `heteroctl tier verify` — no leftover shaping
- [ ] idle baseline recorded, marked, cluster up and idle
- [ ] link tier set explicitly and surveyed
- [ ] run log started: date, SHA, tier, ambient temperature

## D. After every session

- [ ] `heteroctl tier clear`
- [ ] `heteroctl rpc down`
- [ ] node data pulled (`rsync -av pi4:/opt/heteroserve/data/ ./data/pi4/`)
- [ ] energy joined; flag counts reviewed **before** reading the numbers
- [ ] `heteroctl report characterisation` / `report sweep` regenerated
- [ ] prediction error checked for a systematic sign
- [ ] anything unusual written into the run log

## E. Before quoting a result in the report

- [ ] every row behind it is flag-free, or the flags are explained
- [ ] the llama.cpp SHA is identical across every node in the comparison
- [ ] the energy convention (total vs marginal) is stated and consistent
- [ ] the measurement boundary is stated (which rails, what is inside)
- [ ] the link tier is the *measured* bandwidth, not the requested one
- [ ] the number of repetitions and the spread are reported, not just the mean
- [ ] negative results are included, not filtered out
