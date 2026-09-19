.DEFAULT_GOAL := help

.PHONY: fmt fmt-check lint lint-fix check fix test test-cov build help

# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

fmt:  ## Auto-format source code with black
	uv run black src/ tests/

fmt-check:  ## Check formatting without modifying files
	uv run black --check src/ tests/

# ---------------------------------------------------------------------------
# Linting
# ---------------------------------------------------------------------------

lint:  ## Run ruff linter
	uv run ruff check src/ tests/

lint-fix:  ## Run ruff and auto-fix safe issues
	uv run ruff check --fix src/ tests/

# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------

check: fmt-check lint  ## Run all checks (no modifications)

fix: fmt lint-fix  ## Format and auto-fix everything

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:  ## Run the test suite
	uv run pytest

test-cov:  ## Run tests with coverage report
	uv run pytest --cov=kagglex --cov-report=term-missing

# ---------------------------------------------------------------------------
# Documentation
# ---------------------------------------------------------------------------

docs-serve:  ## Run local MkDocs development server
	uv run --group docs mkdocs serve

docs-build:  ## Build documentation site into site/
	uv run --group docs mkdocs build --clean

# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

build:  ## Build source and wheel distribution packages
	uv build

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
