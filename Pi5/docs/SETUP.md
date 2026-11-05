# Pi 5 runbook

From a blank microSD to a node that `06_verify_node.sh` passes. Everything below is
done over SSH from the control node; nothing needs a monitor on the Pi.

---

## 0. Image and first boot

- **Raspberry Pi OS Lite (64-bit)**, Bookworm or later. Lite, not Desktop: a
  compositor and a browser on the same rail are measured power that has nothing to
  do with inference.
- Flash with Raspberry Pi Imager and preset hostname `pi5`, your SSH key, locale,
  and **no Wi-Fi credentials yet** — the wireless tier is configured deliberately in
  step 4, not left up by default.
- Prefer an NVMe HAT or a USB 3.0 SSD over microSD if one is available. Model load
  time is dominated by storage and long sweeps are hard on SD cards. If you stay on
  microSD, use an A2 card — see `../sdcards/` for the cards already characterised.

First login:

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a        # firmware; under-voltage reporting depends on it
sudo reboot
```

Then clone this repo to the Pi. The systemd units expect it at
`/opt/heteroserve/Pi5`, so:

```bash
sudo install -d -o "$USER" /opt/heteroserve
git clone <repo> /opt/heteroserve/HeteroServe
sudo ln -s /opt/heteroserve/HeteroServe/Pi5 /opt/heteroserve/Pi5
cd /opt/heteroserve/Pi5
```

---

## 1. Edit `config/node.env`

It is the only file you edit. Check at minimum:

- `ETH_ADDR` — the Pi 5's address on the wired tier (see the conflict note in
  `config/cluster.hosts`).
- `ETH25_IFACE` — run `ip -br link` with the USB 2.5GbE adapter plugged in and use
  the name it actually enumerates as. Blank it out if there is no adapter.
- `CONTROL_HOST` / `CONTROL_NTP` — the ThinkPad.
- `INA_CHANNEL` / `INA_I2C_ADDR` — must match the Pico 2 firmware's channel map.

`LLAMA_PIN_SHA` is **already set** and must not be changed — see step 3.

---

## 2. OS preparation and dependencies

```bash
sudo ./scripts/00_prepare_os.sh
sudo ./scripts/01_install_deps.sh
```

`00` turns swap off, pins the governor to `performance`, pins the active cooler to
full, forces UTC, points systemd-timesyncd at the control node, and writes
`node_baseline.txt`. **These pins do not survive a reboot on their own** —
`heteroserve-prepare.service` (installed in step 5) re-applies them at boot.

---

## 3. Build llama.cpp — the pinning ritual

```bash
./scripts/02_build_llama_cpp.sh
```

**The revision is already pinned, so there is no paste-back step.** `node.env` ships

```
LLAMA_PIN_SHA=b23701f77d47dad9de834d59ebfcbe25c9e8b46f        # 2026-09-19
```

and `02_build_llama_cpp.sh` checks that revision out directly (it `die`s if the SHA is not
in upstream history). One run is all you need; `build_manifest.json` records it.

Do **not** blank the pin to "see what master does". A mixed-revision cluster is still a
way to get bad numbers into the dataset — though on this pin it fails more honestly than
it used to: `b23701f7` is long past the `RPC_CMD_HELLO` version handshake (2025-04-18),
so a node on a different revision is **refused at the handshake** rather than silently
misexecuting. That was not true of the previous pin.

**Pin changed 2026-09-20** from b5050 (`23106f94…`, 2025-04-04) to `b23701f7`. The old
pin was chosen from the Jetson's CUDA 10.2 ceiling and had never been built on any node,
while every measurement in `testbed/benchmarks.md` was taken on `b23701f7` — so the pin
was moved to follow the evidence. The cost is that the Jetson can no longer build the
CUDA backend on the cluster pin; see `docs/PIN_CONFLICT.md`. `ThinkPad/config/cluster.json` → `llama.pin_sha` is the authoritative copy and its
`_pin_note` carries the full rationale. Changing it means changing all five locations in
one commit (Pi4/ and Pi5/ `node.env`, Jetson/ `node.env` and `node.env.example`,
ThinkPad/ `cluster.json`) — `ThinkPad/tests/test_repo_hygiene.py` fails a partial rollout.

From here the SHA never changes. Every number in the characterisation dataset must
come from one revision on every node; `06_verify_node.sh` fails if the built SHA and
the pinned SHA disagree, and `10_isolated_bench.sh` refuses to run unpinned.

> The Jetson is the constraint here. JetPack 4.6.x ships CUDA 10.2 and current
> llama.cpp master will not compile against it, so the cluster-wide SHA is whichever
> revision the Jetson can still build — pick it there first, then bring the Pi 5 and
> Pi 4 to it. See `../Jetson/config/node.env.example`.

---

## 4. Models and network

```bash
./scripts/03_fetch_models.sh
sudo ./scripts/05_network_setup.sh addr
     ./scripts/05_network_setup.sh show
```

Only the models this node runs **in isolation** live here. Sharded 7B Q8 / 14B Q4
placements are streamed from the control node over RPC and are never resident.

Link tiers, used throughout Phase 4:

```bash
sudo ./scripts/05_network_setup.sh tier gbe          # exactly one tier carries traffic
sudo ./scripts/05_network_setup.sh tier 2.5gbe
sudo ./scripts/05_network_setup.sh shape 100mbit 5   # synthesise an intermediate rate
sudo ./scripts/05_network_setup.sh clear
```

`netem` shapes egress only. For a symmetric link, apply the mirrored qdisc on the
peer — otherwise the return path is unshaped and the measured transfer cost is half
of what you think it is.

---

## 5. Services

```bash
sudo ./scripts/04_install_services.sh
```

Installs four units:

| Unit | Role |
|---|---|
| `heteroserve-prepare` | re-applies the measurement pins at boot, before `rpc` |
| `heteroserve-rpc` | the rpc server on `:50052`, `MemorySwapMax=0`, `CPUAffinity=0-3`. Launched through `/opt/heteroserve/bin/rpc-server.sh`, which `04` generates against the binary that was actually built — upstream renamed `rpc-server` to `ggml-rpc-server` (`c299a92`, 2026-06-27) and removed `--mem` (`41386cf`, 2025-10-17), so the flags depend on the pinned SHA. `cat` that file to see exactly what is launched. |
| `heteroserve-agent` | `/health` and `/ready` (readiness verdict, 200/503), `/info`, `/stat` = `/metrics`, `/events?n=`, `POST /run-tag`, `POST /mark` on `:9410` |
| `heteroserve-telemetry` | 5 Hz thermal/clock/memory CSV on the harness time base |

The agent is read-only apart from the run tag and the phase-marker log, neither of
which changes how the node computes anything. That is deliberate: a node that can
be reconfigured mid-sweep over HTTP is a node whose measurements cannot be trusted.
`ThinkPad/control/nodes.py` also drives `POST /rpc/start` and `/rpc/stop`; this node
answers 404 to both, on purpose. That disagreement is open — see the README.

`/health` used to answer 200 with a liveness body carrying no `ready` and no
`blocking`, which are the only two keys `nodes.py` reads. The effect was that this
node contributed no blocking reasons however throttled, hot or swapped it was, and
the cluster preflight gate passed it unchecked. Both paths now serve the same verdict.

Check from the control node:

```bash
curl -s http://192.168.50.12:9410/ready | jq
```

`503` lists exactly which gate failed.

---

## 6. Acceptance

```bash
./scripts/06_verify_node.sh
```

Must exit 0 before any measurement is recorded. It checks the board identity, the
A76 feature bits, governor and clocks, `vcgencmd get_throttled` since boot, swap,
available memory against the advertised figure, the pinned SHA, the binaries, the
models, every link tier, clock sync, all four services, and peer reachability.

Paste the output into the lab notebook. It is the provenance record for the session.

---

## 7. Phase 2 sweep

```bash
./scripts/10_isolated_bench.sh --reps 3
```

Per cell it gates on temperature, tags the agent (and through it the DAQ) with the
cell id, runs `llama-bench`, records start/end temperature, and cools down before
the next. Output goes to `/opt/heteroserve/results/pi5_isolated_<RUN_ID>/`, with the
sweep file, the `node.env` snapshot, the build manifest and the model checksums
copied in alongside — provenance travels with the numbers.

Then:

```bash
./bench/parse_bench.py /opt/heteroserve/results/pi5_isolated_<RUN_ID>
```

which writes `dataset.csv`, and exits non-zero if any row lacks `c_per_layer_ms`.
The column that matters downstream is
**`c_per_layer_ms`** — decode time per layer per token. That is the `c_i` in Rule A
of proposal Appendix B.6:

```
L_i = round( L_total × (1/c_i) / Σ_j (1/c_j) )   subject to  mem_i ≤ M_i
```

Sanity-check `bw_util_gbs` against the Pi 5's 17 GB/s peak. A value above it means
the model was not actually resident and the cell is void.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `vcgencmd get_throttled` non-zero | Under-voltage or capping has occurred **since boot**. Results from this boot are suspect — use the official 27 W PD supply, reboot, re-verify. |
| `/ready` fails `no_swap_in_use` | A reboot re-enabled swap. Re-run `00_prepare_os.sh` or check `heteroserve-prepare`. |
| `/ready` fails `governor_pinned` | Same cause. The governor reverts to `ondemand` at boot. |
| Cells disagree run to run | Check `temp_start_c` in `cells.csv`. If the gate is being hit repeatedly, the cool-down is too short for the ambient temperature. |
| `rpc-server` killed mid-run | `RPC_MEM_MB` is too close to usable. It must stay well below `NODE_USABLE_MB`; the whole point of advertising less is that the placement engine never hands you a partition that OOMs. |
| 2.5GbE tier skipped | `ETH25_IFACE` does not match what the adapter enumerated as. Check `ip -br link`. |
| `heteroserve-rpc` inactive, no failure, nothing restarted | The server was passed a flag this build does not know. It prints `error: unknown argument` and then calls **`exit(0)`**, which systemd cannot distinguish from a clean shutdown — so `Restart=on-failure` never fires. Check the tail of `/opt/heteroserve/data/rpc-server.log`, then re-run `04_install_services.sh` to regenerate `/opt/heteroserve/bin/rpc-server.sh` against the binary you actually built. |
| `parse_bench.py` exits 1 with "no c_per_layer_ms" | `cells.csv` has no `n_layers` column, so `c_i` cannot be computed and the rows are invisible to Rule A. The run predates the fix; re-run the sweep with the current `10_isolated_bench.sh`, which reads the layer count from the GGUF header (`node/gguf_meta.py`) because llama-bench does not report one. |
