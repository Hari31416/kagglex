# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-19

### Added

- MkDocs documentation site with Material theme, search, and automated GitHub Pages deployment workflow.
- Complete documentation guides for quickstart, batch execution, interactive REPL, datasets, quota tracking, configuration, workspaces, CLI, and Python API.
- Support for user global configuration in `~/.kagglex/config.toml` (and `~/.kagglex/kagglex.toml`) with project-level overrides in `pyproject.toml` and `kagglex.toml`.
- Hierarchical merging of configuration dictionaries for environment variables across global, project, and CLI tiers.

### Changed

- Centralized experiment run history storage exclusively in `~/.kagglex/runs.json` with project provenance (`project_path`), eliminating project-local directory dual writes.

## [0.3.0] - 2026-09-19

### Added

- Automated large payload offloading (`--auto-dataset`, `--auto-dataset-slug`) to package and upload projects or local assets exceeding 5 MB as private Kaggle datasets.
- Remote bootstrap orchestrator support for detecting and extracting package and data payloads from mounted dataset inputs.
- Declarative project configuration support via `[tool.kagglex]` in `pyproject.toml`, project-level `kagglex.toml`, and user-level `~/.kagglex/config.toml`.
- GPU and TPU quota tracking command (`kagglex quota`) aggregating rolling 7-day accelerator consumption against Kaggle weekly limits.
- Pre-flight quota warnings before dispatching kernel runs when quota is nearing exhaustion (> 90% used or < 2 hours remaining).
- Safe TOML loader support using `tomllib` (Python 3.11+) and `tomli` fallback (Python < 3.11).

## [0.2.0] - 2026-09-17

### Fixed

- Convert shell glob patterns to regular expressions for Kaggle API server-side filtering, resolving silent regex syntax errors when pulling wildcard patterns (e.g. `*.json`).
- Prevent unneeded network downloads of large `.pt` checkpoints and `.pyc` files during default output pulls via server-side negative lookaheads.
- Raise an explicit `ValueError` when `pkg_payload.zip` or `data_payload.zip` exceeds 5 MB instead of silently dropping the payload.

### Changed

- Set payload size limit to 5 MB during project and local data packaging to prevent oversized inline kernel payloads.

## [0.1.0] - 2026-09-16

### Added

- Initial release of `kagglex` CLI and Python toolkit.
- Automatic packaging of local code and dependencies for Kaggle kernels.
- Remote execution on Kaggle GPUs and CPUs.
- Live output streaming and artifact synchronization.
- Pre-flight workspace validation and `.kaggleignore` filtering.
- Dataset upload and staging management.
