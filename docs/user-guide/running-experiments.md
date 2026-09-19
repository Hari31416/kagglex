# Running Experiments

The `kagglex run` command packages, uploads, and executes machine learning workloads on Kaggle infrastructure.

## Basic Execution

To run a standalone script:

```bash
kagglex run --file train.py
```

To run a full project module or arbitrary shell command:

```bash
kagglex run --dir . --command "python -m my_module.train --epochs 10"
```

## Accelerator Selection

Kaggle provides several hardware accelerators. You can specify the target accelerator using the `--gpu` flag:

| Accelerator Type | CLI Option | Memory & Specs |
|---|---|---|
| Dual Nvidia T4 | `--gpu t4-2x` | 2x 16 GB VRAM (32 GB total) |
| Nvidia P100 | `--gpu p100` | 1x 16 GB VRAM |
| Google TPU v3-8 | `--gpu v3-8` | 8 cores, 128 GB HBM |
| Standard CPU | `--gpu none` | 4 cores, 30 GB RAM |

### Distributed Multi-GPU Training

When using `--gpu t4-2x`, both GPUs are exposed to PyTorch. Enable distributed data parallel (DDP) by adding the `--multi-gpu` flag or using `torchrun`:

```bash
kagglex run \
  --dir . \
  --command "torchrun --nproc_per_node=2 -m my_package.train" \
  --gpu t4-2x \
  --multi-gpu
```

## Monitoring and Log Streaming

By default, `kagglex run` submits the job, polls status until completion, and downloads results:

```bash
# Stream remote stdout/stderr in real-time
kagglex run --file train.py --stream

# Asynchronous submission (returns immediately after kernel push)
kagglex run --file train.py --no-wait
```

### Inspecting Running and Completed Jobs

You can query recent runs recorded across your machine:

```bash
kagglex list --limit 10
```

To cancel an active remote run:

```bash
kagglex cancel <kernel-slug-or-id>
```

## Managing Output Artifacts

Generated files written to Kaggle's `/kaggle/working` directory are automatically downloaded when the run completes.

### Custom Output Directory

Direct artifacts to a specific local folder:

```bash
kagglex run --file train.py --output-dir ./results/experiment_1
```

### Selective Artifact Filtering

Avoid downloading large intermediate checkpoints by specifying include or exclude glob patterns:

```bash
# Download only evaluation metrics and final model
kagglex run --file train.py \
  --include-outputs "metrics.json" \
  --include-outputs "best_model.pt"

# Exclude raw checkpoint dumps
kagglex run --file train.py \
  --exclude-outputs "checkpoint_epoch_*.pt"
```

## Environment Variables and Kaggle Secrets

### Passing Environment Variables

Pass inline environment variables or specify an environment file:

```bash
kagglex run --file train.py \
  --env WANDB_PROJECT=vision-transformer \
  --env LEARNING_RATE=0.0001 \
  --env-file .env.production
```

### Kaggle Secrets

Inject Kaggle user secrets (such as API keys stored in your Kaggle account settings):

```bash
kagglex run --file train.py \
  --secret WANDB_API_KEY \
  --secret HF_TOKEN
```
