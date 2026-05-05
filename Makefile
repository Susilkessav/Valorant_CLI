.PHONY: install lint format format-check test check run

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

format-check:
	ruff format --check .

test:
	pytest -v

check: lint format-check test

run:
	valocoach coach "Attacking B on Bind, full buy, lost first half 4-8"
