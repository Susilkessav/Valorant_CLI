.PHONY: install lint format test check run

install:
	pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest -v

check: lint test

run:
	valocoach coach "Attacking B on Bind, full buy, lost first half 4-8"
