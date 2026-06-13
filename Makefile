.PHONY: install test lint smoke clean

install:
	python -m pip install -e ".[dev]"

lint:
	ruff check src tests scripts

test:
	pytest -q

smoke:
	python scripts/make_smoke_dataset.py --output-dir data/smoke
	python -m insect_context_ai.train --config configs/smoke.yaml

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info reports/smoke data/smoke
