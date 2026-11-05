# HeteroServe -- control node (ThinkPad T580)
SHELL := /bin/bash
PY := .venv/bin/python
# The unit tests import only the standard library plus this repo's own `control`
# package -- pyserial and matplotlib are runtime-only. So `make test` must work on a
# fresh checkout, before `make bootstrap` has built the venv; otherwise the first
# command a new user runs dies with "/bin/bash: .venv/bin/python: No such file or
# directory" and exit 127, which reads as a broken repo rather than an un-bootstrapped
# one. The runtime targets keep $(PY) on purpose: they DO need pyserial, and failing
# on a missing venv is the right answer there.
TEST_PY := $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)
.DEFAULT_GOAL := help

help: ## show this help
	@grep -hE '^[a-z0-9_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

bootstrap: ## one-time control-node setup (apt, venv, chrony NTP server)
	scripts/00_bootstrap.sh

build: ## build the pinned llama.cpp for x86-64 AVX2
	scripts/01_build_llama_cpp.sh

models: ## download the GGUF models the cluster serves
	scripts/02_fetch_models.sh

deploy: ## push each node's folder to that node
	scripts/03_deploy_nodes.sh

flash-daq: ## copy the DAQ firmware onto the Pico 2
	scripts/04_flash_pico.sh

calibrate: ## guided calibration for one channel: make calibrate CH=2
	scripts/05_calibrate.sh $(CH)

collector: ## run the energy collector in the foreground
	scripts/06_collector.sh

preflight: ## everything that must be true before measuring
	scripts/99_preflight.sh

status: ## cluster summary and live node metrics
	$(PY) -m control.cli status

inventory: ## collect device_info from every agent
	$(PY) -m control.cli inventory --save

rpc-up: ## start rpc-server on every worker
	$(PY) -m control.cli rpc up

rpc-down: ## stop rpc-server everywhere
	$(PY) -m control.cli rpc down

survey: ## measure bandwidth and RTT to every node
	$(PY) -m control.cli survey --save

idle: ## record the marked idle baseline (cluster up, no load)
	$(PY) -m control.cli idle-baseline --seconds 120

test: ## run the unit tests (no hardware needed, no venv needed)
	$(TEST_PY) -m unittest discover -s tests

clean-shaping: ## remove netem shaping everywhere
	$(PY) -m control.cli tier clear

.PHONY: help bootstrap build models deploy flash-daq calibrate collector preflight status \
        inventory rpc-up rpc-down survey idle test clean-shaping
