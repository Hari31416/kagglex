# kagglerun

Execute local Python code, modules, and experiments seamlessly on Kaggle GPUs and TPUs.

## Overview

`kagglerun` enables machine learning practitioners and researchers to transparently package and dispatch local Python code, standalone scripts, or complete packages to Kaggle's cloud GPU and TPU environments without tedious manual uploading or notebook maintenance.

## Features

- Flexible project detection for standalone scripts, flat packages, or `src/` layout projects
- Pre-flight validation with local AST syntax checking and credential verification
- Automatic ignore filtering powered by `pathspec` supporting `.gitignore` and `.kaggleignore`
- Dual-tier packaging preventing bloated uploads and base64 truncation
- Automated multi-GPU execution using `torchrun`
- Secure Kaggle Secrets integration for WandB, HuggingFace, and custom credentials
- Local experiment history repository for tracking past runs, statuses, and durations
- Programmatic Python SDK alongside the `kagglerun` CLI

## Installation

Install via uv or pip:

```bash
uv pip install kagglerun
```

## CLI Usage

### Run a Standalone Script

```bash
kagglerun run --file train.py --gpu t4-2x --title "Pilot Training"
```

### Run a Module with Multi-GPU

```bash
kagglerun run "python -m mypkg.train --epochs 10" --gpu t4-2x --multi-gpu
```

### Stream Remote Logs in Real-Time

```bash
kagglerun run "python train.py" --stream
```

### List Recent Runs

```bash
kagglerun list
```

### Check Status or Cancel a Run

```bash
kagglerun status my-experiment
kagglerun cancel my-experiment
```

### Pull Downloaded Outputs

```bash
kagglerun pull my-experiment --output-dir ./results --include-outputs "*.json" "checkpoints/*"
```

### Push a Kaggle Dataset

```bash
kagglerun dataset push --data-dir ./data/embeddings --title "Embeddings Dataset"
```

## Python SDK Usage

```python
from kagglerun import KaggleRunner, RunConfig

runner = KaggleRunner()
job = runner.run(
    command="python -m mypkg.train --batch-size 64",
    title="Fine Tuning Run",
    gpu="t4-2x",
    multi_gpu=True,
    wait=True,
)

print(f"Status: {job.status}")
job.pull_outputs(destination_dir="./results")
```
