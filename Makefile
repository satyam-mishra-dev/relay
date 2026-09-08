setup:
	uv sync

data:
	uv run python -m relay.data

eval:
	uv run python -m relay.evaluate

reproduce:
	uv run python -m relay.evaluate

test:
	uv run pytest -q

lint:
	uv run ruff check .

.PHONY: setup data eval reproduce test lint
