# Quickstart

This guide will help you install `kagglex`, configure your Kaggle credentials, and execute your first remote GPU experiment.

## Prerequisites

- Python 3.9 or newer
- A verified Kaggle account with phone verification completed (required by Kaggle to enable cloud GPU/TPU access)

## 1. Installation

Install `kagglex` from PyPI using your preferred package manager:

```bash
pip install kagglex
```

Or with `uv`:

```bash
uv add kagglex
```

## 2. Kaggle API Credentials

`kagglex` uses the standard official Kaggle API client. Ensure your API token is placed in `~/.kaggle/kaggle.json`:

1. Navigate to [kaggle.com/settings](https://www.kaggle.com/settings).
2. Scroll to the **API** section and click **Create New Token**.
3. Move the downloaded `kaggle.json` file to `~/.kaggle/kaggle.json`:

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Verify your authentication status:

```bash
kagglex run --dry-run
```

## 3. Run Your First Script on Kaggle GPU

Create a minimal training script named `train.py`:

```python
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device Count: {torch.cuda.device_count()}")

    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print(f"Matrix multiplication result shape: {y.shape}")
```

Dispatch this script to a dual Nvidia T4 instance on Kaggle:

```bash
kagglex run --file train.py --gpu t4-2x --stream
```

`kagglex` will:

- Detect your standalone script and package it
- Stage bootstrap files in `.kagglex/staging`
- Push the execution kernel to Kaggle
- Stream remote logs directly to your terminal
- Download any generated output files to `./outputs` upon completion

## 4. Next Steps

- Explore [Running Experiments](user-guide/running-experiments.md) to learn about multi-file packages and accelerator configurations.
- Learn how to interactively debug models in real-time with the [Interactive REPL](user-guide/interactive-repl.md).
- Check your compute balance with [Quota Tracking](user-guide/quota.md).
