cov_path_html := "/tmp/" + `whoami` + "-coverage-html"

# Run all tests (unit + integration, excludes slow) with coverage
test *args:
    uv run pytest -m "unit or integration" --cov --cov-report=term-missing {{args}}

# Run all tests including slow (e.g. MobileCLIP) with coverage
test-all *args:
    uv run pytest -m "unit or integration or slow" --cov --cov-report=term-missing {{args}}

# Run only unit tests with coverage
test-unit *args:
    uv run pytest -m unit --cov --cov-report=term-missing {{args}}

# Run only integration tests with coverage
test-int *args:
    uv run pytest -m integration --cov --cov-report=term-missing {{args}}

# Run tests matching a keyword expression
test-k expr *args:
    uv run pytest -k "{{expr}}" {{args}}

# Generate HTML coverage report and serve it on localhost:8000
coverage-html *args:
    @just --justfile {{justfile()}} test-all --cov-report=html:{{cov_path_html}} {{args}}
    @echo "Coverage report at http://localhost:8000 — Ctrl-C to stop"
    cd {{cov_path_html}} && python -m http.server 8000

# lint source + tests
lint:
    -uv run ruff format --check src tests
    -uv run ruff check src tests
    -uv run mypy src tests

# format source + tests
format:
    uv run ruff format src tests
    uv run ruff check --fix src tests
