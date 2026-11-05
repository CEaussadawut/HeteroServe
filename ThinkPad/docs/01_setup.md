# 01 — Setting up the control node

Target state: Ubuntu LTS on the ThinkPad T580, serving as the cluster's NTP source, holding
the models, running `heteroctl`, and recording the power stream from the Pico 2.

## 1. OS

Ubuntu LTS (22.04 or 24.04), 64-bit. Nothing exotic is required of it, but two settings
matter for measurement quality:

- **Disable suspend on lid close.** The control node must stay awake through a four-hour
  sweep with the lid shut. `sudo systemctl mask sleep.target suspend.target` or set
  `HandleLidSwitch=ignore` in `/etc/systemd/logind.conf`.
- **Keep it on mains.** On battery, the i5-8350U clocks down, which changes prefill timing
  when the T580 itself serves layers in the disaggregation experiment.

The T580 is *not* on the DAQ — there are four INA226 channels and they belong to the four
measured compute nodes. The control node's own consumption sits outside the measurement
boundary, and the report must say so rather than leave a reader to assume otherwise.

## 2. Static address and hostname

```bash
sudo hostnamectl set-hostname t580
sudo nmcli con mod "Wired connection 1" \
  ipv4.addresses 192.168.50.10/24 ipv4.gateway 192.168.50.1 ipv4.method manual
sudo nmcli con up "Wired connection 1"
```

The addresses in `config/cluster.json` must match reality on every node. Nothing else
reconciles them — a node reachable at a different address simply looks down.

## 3. Bootstrap

```bash
sudo install -d -o $USER -g $USER /opt/heteroserve
git clone <this repo> /opt/heteroserve/src      # or rsync the folder across
cd /opt/heteroserve/src/ThinkPad
make bootstrap
```

`scripts/00_bootstrap.sh` installs the toolchain, creates `.venv`, installs the `control`
package in editable mode (so `heteroctl` and `python -m control.cli` are the same code),
adds you to `dialout` for serial access to the Pico, and configures chrony **as a server**
for the cluster subnet.

Log out and back in afterwards — the `dialout` group is not applied to a running session,
and the DAQ will read "permission denied" until you do.

## 4. SSH keys to every node

```bash
ssh-keygen -t ed25519 -C heteroserve-control
for h in 192.168.50.12 192.168.50.13 192.168.50.15; do ssh-copy-id "$h"; done
```

Key-only access is what makes `heteroctl tier` and `heteroctl survey` work unattended;
`control/nodes.py` runs ssh with `BatchMode=yes` and will fail rather than prompt.

## 5. Verify the time service before anything else

```bash
chronyc clients          # the workers should appear here once they are up
ssh pi@pi4 chronyc tracking
```

Every joules-per-token figure in this project depends on the workers' clocks agreeing with
this machine. The node-side healthcheck treats an offset above 5 ms as blocking, and
`heteroctl preflight` refuses to start a run while any node is out of sync.

## 6. Build llama.cpp and fetch models

```bash
make build         # AVX2 + FMA, RPC on; writes control_build_manifest.json
make models        # the 7B/14B GGUFs; prints each model's shape as the planner reads it
```

After the first build, put the printed SHA into `llama.pin_sha` in `config/cluster.json`
**and** into every node's `config/node.env`. Mind the spelling — it is not the same key on
every node:

| file | key |
|---|---|
| `ThinkPad/config/cluster.json` | `llama.pin_sha` |
| `Pi4/config/node.env`, `Pi5/config/node.env` | `LLAMA_PIN_SHA` |
| `Jetson/config/node.env` | `LLAMA_CPP_REF` |

Pick the revision **on the Jetson first**: JetPack 4.6.x ships CUDA 10.2, current llama.cpp
master will not build against it, and whatever the Jetson can build therefore constrains all
four nodes.

Preflight fails while these disagree, which is the intended behaviour: rows produced by two
different llama.cpp revisions cannot be compared with each other. Note that `heteroctl
preflight` checks this over the network, so it needs the cluster up (and the Jetson ships no
node agent, so it cannot self-report); `make test` checks that these files agree offline.

## 7. Deploy the node folders

```bash
make deploy        # rsyncs Pi4/, Pi5/, Jetson/... to each node and restores exec bits
make status        # every agent should answer
```

`scripts/03_deploy_nodes.sh` reads the node list from `cluster.json` and uses each node's
`script_dir`. A node whose folder does not exist locally is skipped with a warning rather
than failing the deploy.

## 8. The DAQ

```bash
make flash-daq     # copies firmware/pico2/*.py to the Pico (MicroPython must be on it)
heteroctl daq status --seconds 10
```

Full wiring, calibration and error budget: `docs/03_energy_collector.md`.

## 9. Collector as a service

```bash
sudo install -m0644 systemd/heteroserve-collector@.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now heteroserve-collector@$USER
```

Leave it running for a whole measurement session. The analysis cuts run windows out of a
continuous trace, so a collector that starts and stops per run risks losing the first
seconds of each one.

## 10. Confirm the whole chain

```bash
make test          # unit tests, no hardware needed
make preflight     # cluster readiness + DAQ + calibration, in one command
```

`make preflight` is the gate. It is meant to be run before every session, and its failures
are the ones that would otherwise show up as quietly wrong numbers three weeks later.
