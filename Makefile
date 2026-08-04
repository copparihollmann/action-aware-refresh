SHELL := /bin/bash
PY := uv run python
export PYTHONPATH := $(PWD)/src

.PHONY: help audit setup test smoke baseline profile pilot report clean sources contract

help:
	@echo "Targets:"
	@echo "  audit     - inventory machine, write docs/environment_report.md"
	@echo "  sources   - clone/pin third_party repos, populate reproducibility/source_manifest.json"
	@echo "  contract  - derive docs/baseline_contract.md from checked-out Cosmos + client source"
	@echo "  setup     - build cosmos + robolab environments (native uv; container path is blocked)"
	@echo "  test      - run unit tests"
	@echo "  smoke     - run baseline smoke test (server + one RoboLab task)"
	@echo "  baseline  - run official Cosmos3 baseline configuration"
	@echo "  profile   - produce compute-anatomy profiler traces"
	@echo "  pilot     - PILOT-task benchmark (later sessions only)"
	@echo "  report    - regenerate results/reports/*"

audit:
	bash scripts/audit_machine.sh

sources:
	bash scripts/clone_sources.sh

contract:
	$(PY) scripts/derive_baseline_contract.py

setup:
	bash scripts/setup_cosmos.sh
	bash scripts/setup_robolab.sh

test:
	uv run pytest tests/unit

smoke:
	bash scripts/smoke_test.sh

baseline:
	bash scripts/start_cosmos_server.sh &
	bash scripts/run_robolab.sh

profile:
	bash scripts/profile_baseline.sh

pilot:
	$(PY) scripts/run_sweep.py --set pilot

report:
	$(PY) scripts/build_report.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache build dist *.egg-info
