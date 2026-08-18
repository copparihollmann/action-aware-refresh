SHELL := /bin/bash
# `--no-sync` on purpose: a bare `uv run` re-resolves the environment to the
# project's default dependency set. In third_party/cosmos-framework that
# silently replaced torch 2.10.0+cu130 with 2.13.0+cu130 and broke flash-attn's
# ABI; in third_party/RoboLab it would uninstall isaacsim. Never let a *run*
# step mutate the environment being measured. Use `make venv` to (re)create it.
PY := uv run --no-sync python
export PYTHONPATH := $(PWD)/src

.PHONY: help venv audit tasksets setup test test-model smoke baseline profile profile-wire pilot report clean sources contract

help:
	@echo "Targets:"
	@echo "  audit     - inventory machine, write docs/environment_report.md"
	@echo "  sources   - clone/pin third_party repos, populate reproducibility/source_manifest.json"
	@echo "  contract  - derive docs/baseline_contract.md from checked-out Cosmos + client source"
	@echo "  setup     - build cosmos + robolab environments (native uv; see decision_log)"
	@echo "  test      - run unit tests (repo venv; torch-dependent tests skip)"
	@echo "  test-model- run unit tests in the cosmos venv (includes torch tests)"
	@echo "  smoke     - run baseline smoke test (server + one RoboLab task)"
	@echo "  baseline  - run official Cosmos3 baseline configuration"
	@echo "  profile   - produce compute-anatomy profiler traces"
	@echo "  pilot     - PILOT-task benchmark (later sessions only)"
	@echo "  report    - regenerate results/reports/*"

venv:
	uv sync --extra dev

audit:
	bash scripts/audit_machine.sh

tasksets:
	$(PY) scripts/select_task_sets.py

sources:
	bash scripts/clone_sources.sh

contract:
	$(PY) scripts/derive_baseline_contract.py

setup:
	bash scripts/setup_cosmos.sh
	bash scripts/setup_robolab.sh

test:
	uv run --extra dev --frozen pytest tests/unit

# Tests that need torch. The repo venv deliberately has none (torch lives in the
# third_party venvs and must not be re-resolved), so those tests skip under `make
# test` and run here instead, against the SAME interpreter that runs the experiments.
# `.venv/bin/python -m pytest`, never `uv run`: a bare `uv run` in cosmos-framework
# re-resolves and replaces the measured torch build.
#
# Which interpreter that is now depends on the substrate, so it is resolved rather
# than hardcoded — `COSMOS_SUBSTRATE=efficient_imagination make test-model` runs the
# same tests against the fork's stack. It prints the substrate it picked, because
# "the tests passed" means little without knowing where.
test-model:
	@set -euo pipefail; \
	REPO_ROOT=$(PWD); source scripts/lib/common.sh; \
	resolve_repo_py; resolve_substrate; \
	echo "test-model: substrate=$$COSMOS_SUBSTRATE  interpreter=$$COSMOS_PY"; \
	PYTHONPATH=$(PWD)/src "$$COSMOS_PY" -m pytest tests/unit -q

smoke:
	bash scripts/smoke_test.sh

# The server takes minutes to load 33 GB of weights, so the client must not be
# launched until /healthz answers. Backgrounding with a bare `&` raced the
# client's own health gate and usually lost. `smoke_test.sh` already sequences
# server startup, health-wait, client run and teardown correctly — reuse it
# rather than duplicating a second, subtly different launch path here.
baseline:
	bash scripts/smoke_test.sh

# M2 compute anatomy. Uses the IN-PROCESS probe: the server exposes no per-stage
# timings and num_steps/decode_video are server-start args, so a wire-level
# sweep can neither attribute stages nor vary configs. One config per process
# (the model is ~31 GiB resident and is not reliably freed).
profile:
	bash scripts/run_anatomy_sweep.sh

# Legacy wire-level driver: end-to-end client latency only, no stage attribution.
# Superseded by `profile` for M2; kept because round-trip cost is still a real
# number we need once the closed-loop client runs.
profile-wire:
	bash scripts/profile_baseline.sh

# `--method` is required by run_sweep.py; a task set alone is not a run.
pilot:
	$(PY) scripts/run_sweep.py --method $(or $(METHOD),baseline_full) --set pilot

report:
	$(PY) scripts/build_report.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
