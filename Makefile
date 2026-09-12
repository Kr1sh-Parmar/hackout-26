# Aliases for the commands in this repo's README. Every target is one line and
# shells out to a script that works on its own -- `make` is a convenience here,
# never a build system, and nothing depends on having it installed.

REGION ?= BE

.PHONY: setup test lint format train tune backtest deep cycle live api snapshot verify demo clean

setup:     ## install the package and the dev toolchain
	pip install -e ".[dev]"

test:      ## the full suite
	python -m pytest tests/ -q

lint:      ## what CI enforces
	python -m ruff check src tests scripts
	python -m ruff format --check src tests scripts

format:    ## apply the formatter
	python -m ruff format src tests scripts

train:     ## fit and promote both technologies for $(REGION)
	python scripts/train.py --region $(REGION)

tune:      ## Optuna search; writes artifacts/tuning/$(REGION)_<tech>.json
	python scripts/tune.py --region $(REGION)

backtest:  ## 22-fold walk-forward; writes artifacts/backtest.json
	python scripts/backtest.py --region $(REGION)

deep:      ## rungs 6-7 benchmark (needs the [deep] extra; ~10 CPU-min)
	python scripts/train_deep.py

cycle:     ## one forecast cycle replaying the latest run in the training matrix
	python scripts/run_cycle.py --region $(REGION)

live:      ## one forecast cycle against live weather
	python scripts/run_cycle.py --region $(REGION) --live

api:       ## serve on :8000, docs at /docs
	uvicorn src.api.main:app --reload --port 8000

snapshot:  ## freeze the latest cycle for offline replay
	python scripts/snapshot_replay.py --region $(REGION)

verify:    ## re-check the built dataset against its manifest
	python scripts/verify_dataset.py

demo:      ## the offline path: frozen data, no network
	REPLAY_MODE=true uvicorn src.api.main:app --port 8000

clean:     ## caches only -- never data/, artifacts/ or gold
	rm -rf .pytest_cache .ruff_cache .hypothesis
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
