# https://github.com/casey/just

# List justfile recipes
help:
    just --list

# Update the environment with the latest version of the dependencies
dev-sync:
    uv sync --all-extras --cache-dir .uv_cache

# Update the environment but not with the development dependencies
prod-sync:
    uv sync --all-extras --no-dev --cache-dir .uv_cache

# Fix the code style and format
fix: dev-sync
    uv run ruff check --fix
    uv run ruff format
    uv run scripts/license_headers.py src --fix
    uv run scripts/license_headers.py tests --fix

# Execute the ruff code linter and format checker
check: dev-sync
    uv run ruff check
    uv run scripts/license_headers.py src
    uv run scripts/license_headers.py tests

# Install within the nvtexlive container
nvtexlive-install:
    pip install -e '.[test]'

# Execute the tests within the nvtexlive container
nvtexlive-test:
    pytest tests/regression_tests/ -v

# Build the release package
build: dev-sync
    rm -rf dist
    uv build --wheel
    uv build --sdist
