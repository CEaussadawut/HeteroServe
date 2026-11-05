# HeteroServe — Jetson Nano node.
# Run these ON the Jetson, in order. `make all` does the whole bring-up.
SHELL := /bin/bash
S     := scripts

.PHONY: help all prep toolchain build models facts rpc bench smoke collect clean check install-units

help:
	@echo "HeteroServe / Jetson Nano node"
	@echo
	@echo "  make prep        system prep: headless, swap, clocks, packages (sudo)"
	@echo "  make toolchain   cmake + gcc-8 + python deps"
	@echo "  make build       build llama.cpp with RPC (+CUDA if possible)"
	@echo "  make models      download the GGUFs in config/models.tsv"
	@echo "  make facts       print this node's characterisation record"
	@echo "  make all         prep -> toolchain -> build -> models -> facts"
	@echo
	@echo "  make rpc         run the RPC worker in the foreground"
	@echo "  make bench       full single-node throughput+energy sweep"
	@echo "  make smoke       cluster reachability / RPC / two-node check"
	@echo "  make collect     tar up results for the analysis machine"
	@echo "  make check       lint the scripts (shellcheck + py_compile)"
	@echo "  make install-units  install+parameterise the systemd units (sudo)"

all: prep toolchain build models facts

config/node.env:
	cp config/node.env.example config/node.env
	@echo "created config/node.env — edit NODE_ID and paths before continuing"

prep: config/node.env      ; $(S)/00_system_prep.sh
toolchain: config/node.env ; $(S)/10_install_toolchain.sh
build: config/node.env     ; $(S)/20_build_llama_cpp.sh
models: config/node.env    ; $(S)/30_fetch_models.sh
rpc: config/node.env       ; $(S)/40_run_rpc_server.sh
bench: config/node.env     ; $(S)/50_bench_device.sh
smoke: config/node.env     ; $(S)/70_cluster_smoke.sh
collect: config/node.env   ; $(S)/90_collect_results.sh

facts: config/node.env
	@mkdir -p results
	$(S)/60_collect_node_facts.sh | tee results/$$(grep -m1 '^NODE_ID=' config/node.env | cut -d= -f2)_facts.json

check:
	@command -v shellcheck >/dev/null && shellcheck -S warning $(S)/*.sh $(S)/lib/*.sh || echo "shellcheck not installed, skipping"
	@python3 -m py_compile telemetry/*.py bench/*.py && echo "python ok"
	@python3 -c "import json;json.load(open('bench/matrix.json'));print('matrix.json ok')"

clean:
	rm -rf __pycache__ telemetry/__pycache__ bench/__pycache__
	@echo "build artefacts live in \$$LLAMA_BUILD; remove with: scripts/20_build_llama_cpp.sh --clean"

# --- systemd ---------------------------------------------------------------
# The units ship with no paths in them. systemd expands ${VAR} only in Exec*=
# lines, never in User= or WorkingDirectory=, so this target generates both an
# EnvironmentFile (for the command lines) and a per-unit drop-in (for the rest).
# Re-run it after moving the repo or changing HS_ROOT in config/node.env.
install-units: config/node.env
	@set -eu; \
	repo="$(CURDIR)"; \
	run_user="$${SUDO_USER:-$$USER}"; \
	hs_root="$$(bash -c 'set -a; . "$$1"; printf %s "$$HS_ROOT"' _ config/node.env)"; \
	results="$$(bash -c 'set -a; . "$$1"; printf %s "$$RESULTS_DIR"' _ config/node.env)"; \
	node_id="$$(bash -c 'set -a; . "$$1"; printf %s "$$NODE_ID"' _ config/node.env)"; \
	echo "repo=$$repo"; echo "user=$$run_user"; echo "workspace=$$hs_root"; \
	printf 'HS_REPO=%s\nHS_ROOT=%s\nHS_RESULTS=%s\nNODE_ID=%s\n' \
	  "$$repo" "$$hs_root" "$$results" "$$node_id" \
	  | sudo tee /etc/default/heteroserve >/dev/null; \
	for u in heteroserve-rpc heteroserve-telemetry; do \
	  sudo install -m 0644 systemd/$$u.service /etc/systemd/system/; \
	  sudo mkdir -p /etc/systemd/system/$$u.service.d; \
	  printf '[Service]\nUser=%s\nWorkingDirectory=%s\n' "$$run_user" "$$hs_root" \
	    | sudo tee /etc/systemd/system/$$u.service.d/10-local.conf >/dev/null; \
	done; \
	sudo mkdir -p "$$results"; sudo chown "$$run_user" "$$results"; \
	sudo systemctl daemon-reload; \
	echo "installed. enable with: sudo systemctl enable --now heteroserve-rpc"
