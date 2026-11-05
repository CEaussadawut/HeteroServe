# HeteroServe -- Pi 4 worker node
# Run on the Pi itself. Targets are ordered the way bring-up actually happens.
SHELL := /bin/bash
.DEFAULT_GOAL := help

help: ## show this help
	@grep -hE '^[a-z0-9_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

bootstrap: ## one-time OS bootstrap (apt, chrony, dirs) -- needs sudo
	sudo scripts/00_bootstrap.sh

tune: ## apply the fixed measurement state -- needs sudo, re-run after every reboot
	sudo scripts/01_system_tune.sh

build: ## build the pinned llama.cpp revision for Cortex-A72
	scripts/02_build_llama_cpp.sh

models: ## download the small GGUF models used for isolated characterisation
	scripts/03_fetch_models.sh

install: ## install and enable the systemd units
	# Both long-lived units use StandardOutput=append: into this directory. systemd opens
	# that file before running any Exec* line, so an ExecStartPre cannot create it -- the
	# unit would fail first, for a reason loggable only to the journal. Create it here.
	sudo install -d -o pi -g pi -m0755 /opt/heteroserve/data/logs
	sudo install -m0644 systemd/heteroserve-*.service /etc/systemd/system/
	# Validate the sudoers file BEFORE it goes live. `visudo -c` after `install` is too
	# late: the malformed file is already in /etc/sudoers.d/ by the time visudo objects,
	# and a broken file there can make sudo refuse everything -- on a headless Pi that
	# means physical access with a keyboard to recover. `-c -f` checks the source in place.
	sudo visudo -c -f systemd/heteroserve-agent.sudoers
	sudo install -m0440 systemd/heteroserve-agent.sudoers /etc/sudoers.d/heteroserve
	sudo visudo -c
	sudo systemctl daemon-reload
	sudo systemctl enable --now heteroserve-tune heteroserve-agent
	@echo "rpc-server is left disabled on purpose: the control node starts it per run."

health: ## readiness check -- run before EVERY measurement run
	scripts/99_healthcheck.sh

info: ## print this node's device inventory and save it
	python3 node/device_info.py --save

test: ## run the off-target test suite (no Pi, no models, no network needed)
	python3 -m unittest discover -s tests -t .

rpc: ## run rpc-server in the foreground
	scripts/04_rpc_server.sh foreground

bench: ## full Phase 2 characterisation sweep (long: hours)
	scripts/06_bench_local.sh full

bench-quick: ## short smoke sweep to prove the pipeline end to end
	scripts/06_bench_local.sh quick

bench-validate: ## R=1,3,5 cross-check that the energy window is right (separate output dir)
	scripts/06_bench_local.sh validate

net-verify: ## check link speed / RTT / throughput against the control node
	scripts/05_net_tier.sh verify

clean-shaping: ## remove any leftover tc netem shaping
	sudo scripts/05_net_tier.sh clear

journal: ## dump this boot's unit journals into data/logs (they are otherwise uncollectable)
	@mkdir -p /opt/heteroserve/data/logs
	@for u in heteroserve-rpc heteroserve-agent heteroserve-tune; do \
	    f=/opt/heteroserve/data/logs/$$u.journal.log; \
	    journalctl -u $$u -b --no-pager > $$f 2>/dev/null || true; \
	    if [ -s $$f ] && ! grep -qx -- "-- No entries --" $$f; then \
	      echo "  wrote $$u.journal.log ($$(wc -l < $$f) lines)"; \
	    else \
	      echo "  $$u: NO JOURNAL ENTRIES this boot -- unit never started, or journald is volatile"; \
	    fi; \
	done
	@echo "OOM kills, exit codes and 'start request repeated too quickly' live ONLY here."

collect: journal ## print the rsync command the control node should run
	@echo "rsync -av $(shell hostname):/opt/heteroserve/data/ ./data/$(shell hostname)/"

.PHONY: help bootstrap tune build models install health info test rpc bench bench-quick bench-validate journal net-verify clean-shaping collect
