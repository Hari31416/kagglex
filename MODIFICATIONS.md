# Kaggle Runner PyPI Package Modifications and Roadmap

This document outlines the required refactorings, architectural changes, and feature enhancements needed to convert the internal `kaggle_runner` scripts into an independent, production-grade PyPI package (`kagglerun`).

## Architecture and Project Structure

Currently, the code resides in a flat `kaggle_runner/` folder copied directly from the monorepo scripts directory.

### Target Repository Layout

To follow modern Python packaging standards, adopt a `src/` layout:

```text
kagglerun/
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .kaggleignore
├── MODIFICATIONS.md
├── src/
│   └── kagglerun/
│       ├── __init__.py
│       ├── api.py
│       ├── bootstrap.py
│       ├── cli.py
│       ├── client.py
│       ├── config.py
│       ├── dataset.py
│       ├── ignore.py
│       └── packager.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_cli.py
    ├── test_config.py
    ├── test_ignore.py
    └── test_packager.py
```

### Build and Distribution Configuration

Create `pyproject.toml` with `hatchling` as the build system and entry points for the CLI:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "kagglerun"
version = "0.1.0"
description = "Execute local Python code, modules, and experiments seamlessly on Kaggle GPUs."
readme = "README.md"
license = "MIT"
requires-python = ">=3.9"
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
keywords = ["kaggle", "gpu", "machine-learning", "deep-learning", "cli"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "kaggle>=1.6.0",
    "pathspec>=0.11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "black>=24.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]

[project.scripts]
krun = "kagglerun.cli:main"
kagglerun = "kagglerun.cli:main"
```

## Core Code Decoupling and Refactoring

The current implementation contains couplings specific to the originating monorepo. The following changes are required in each module.

### 1. Configuration (`config.py`)

- **Remove monorepo root detection:**
  `find_repo_root()` currently searches for `context.md`. Replace this with general project root detection (checking for `.git`, `pyproject.toml`, `setup.py`, `setup.cfg`, or `requirements.txt`), falling back to `Path.cwd()`.
- **Remove mandatory subproject requirement:**
  `subproject: str = "embeddings"" and `resolve_subproject_path()` assume the codebase is partitioned into monorepo subprojects containing `src/` and `pyproject.toml`.
  Instead, allow executing:
  - The current working directory directly (root project mode).
  - A specified directory (`--dir` or `--project-dir`).
  - A single standalone script file (`--file train.py`).
- **Support TPU accelerators:**
  Extend `gpu_type` choices beyond `t4-2x` and `p100` to support Kaggle TPU VM options or flexible accelerator identifiers.

### 2. Packaging and Staging (`packager.py`)

- **Dynamic project structure detection:**
  `package_subproject()` currently raises `FileNotFoundError` if `src/` or `pyproject.toml` is absent.
  Update the packaging logic to detect project types:
  - Standard `src/` package layout.
  - Flat package layout (e.g. `mypackage/__init__.py` at root).
  - Standalone script with `requirements.txt`.
  - Poetry / uv lockfiles (`poetry.lock`, `uv.lock`).
- **Standard ignore patterns (`.gitignore` and `.kaggleignore`):**
  Replace hardcoded `ROOT_EXCLUDES` with a dedicated ignore parser using `pathspec`.
  The packager should automatically read `.gitignore` and optional `.kaggleignore` files, ensuring virtual environments, local datasets, checkpoints, and caches are not bundled into the upload.
- **Eliminate base64 script inlining:**
  Currently, `generate_bootstrap_script()` embeds base64-encoded zip archives directly into the text of `kaggle_bootstrap.py`.
  Kaggle enforces strict size limits on script files. Large base64 strings risk rejection or truncation.
  Instead:
  - Place `pkg_payload.zip` and `data_payload.zip` directly into the staged directory.
  - In `kaggle_bootstrap.py`, extract the zip files from `/kaggle/working/` or use Kaggle API dataset attachment if payload exceeds 20 MB.

### 3. Remote Execution Bootstrap (`bootstrap.py`)

- **Flexible dependency installation:**
  Currently executes `pip install -e pkg_src`.
  Update remote installation to handle diverse project configurations:
  - If `pyproject.toml` exists: `pip install -e pkg_src` or `pip install pkg_src`.
  - If `requirements.txt` exists: `pip install -r requirements.txt`.
  - If `setup.py` exists: `pip install -e pkg_src`.
- **Customizable environment defaults:**
  The current bootstrap sets `HF_HOME=/tmp/huggingface` and `PYTHONUNBUFFERED=1`. Make environment variables easily extensible via configuration without modifying bootstrap template code.

### 4. Client and API Wrapper (`client.py`)

- **Decouple experiment logging:**
  `append_experiment_record()` appends entries in a specific monorepo format to an experiment `README.md`.
  Abstract this into an optional hook or post-run callback (e.g. `--log-format markdown|json`).
- **Robust error parsing:**
  Improve error extraction from Kaggle API responses when kernels fail during preparation or validation.

## High-Impact Features to Add

### 1. Job Cancellation (`krun cancel` / `krun stop`)

Implement a subcommand to terminate running jobs:

```bash
krun cancel <kernel-id>
```

Under the hood, call `api.kernels_cancel(kernel_id)`. This prevents wasted GPU quota if a run diverges or hangs.

### 2. Run History and Status Dashboard (`krun list`)

Implement a subcommand to list recent Kaggle runs:

```bash
krun list [--limit 10] [--mine]
```

Display a clean tabular output showing Kernel Slug, Status, Accelerator Type, Submission Time, and Execution Duration.

### 3. Real-Time Streaming by Default

In `cli.py`, `run` currently polls every 20 seconds and prints status changes, while log streaming is separated under `status --stream`.
Unify this workflow: when a user launches `krun run ...`, automatically connect to `api.kernels_logs_stream(kernel_id)` once the kernel starts running. Local terminal output should match Kaggle stdout in real time.

### 4. Secrets and Environment Variable Injection

Add options to inject API keys and secrets securely:
- CLI flags: `--env KEY=VALUE` and `--env-file .env`.
- Kaggle Secrets support: `--kaggle-secrets HF_TOKEN WANDB_API_KEY`. The bootstrap script automatically retrieves these on Kaggle via `kaggle_secrets.UserSecretsClient`.

### 5. Chained Execution via Kernel Sources (`--parent-kernel`)

Support Kaggle kernel output chaining:
- CLI flag `--parent-kernel <username/slug>`.
- Generates `kernel_sources` in `kernel-metadata.json`, allowing multi-stage workflows (e.g. Step 1: Data preprocessing -> Step 2: GPU training consuming Step 1 outputs).

### 6. Programmatic Python API

Expose a clean Python API alongside the CLI so users can trigger Kaggle jobs from Python scripts or orchestrators:

```python
from kagglerun import KaggleRunner

runner = KaggleRunner()
job = runner.run(
    command="python -m mypkg.train --epochs 5",
    gpu="t4-2x",
    multi_gpu=True,
    wait=True,
)
job.pull_outputs(destination="./results")
```

### 7. Pre-flight Validation and Health Checks

Before submitting jobs and consuming weekly GPU quotas:
- Validate that Kaggle API credentials exist and authenticate properly.
- Calculate and display total upload payload size, warning if it approaches limits.
- Confirm required datasets exist on Kaggle before kernel submission.

## PyPI Publishing Workflow

1. Configure GitHub Actions workflow for PyPI Trusted Publishing (OIDC).
2. Set up automated linting and formatting with `ruff` and `black`.
3. Set up unit and integration test suite with `pytest`.
4. Release version `0.1.0` on PyPI upon tag push (`v0.1.0`).
