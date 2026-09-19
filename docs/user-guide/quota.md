# Accelerator Quota Tracking

Kaggle provides standard weekly quotas for hardware accelerators:

- **GPU Quota**: 30 hours per rolling 7-day window (shared between T4 and P100 instances)
- **TPU Quota**: 20 hours per rolling 7-day window (for TPU v3-8 instances)

Because Kaggle's public API does not expose an endpoint to query live quota balances, `kagglex` computes usage locally from run records stored in `~/.kagglex/runs.json`.

## Viewing Quota Dashboard

Inspect your rolling accelerator usage with `kagglex quota`:

```bash
kagglex quota
```

Sample output:

```text
Kaggle Accelerator Quota Usage (Past 7 Days):

ACCELERATOR    USED HOURS    QUOTA LIMIT    REMAINING    UTILIZATION    RUNS
GPU            12.45 hrs     30.00 hrs      17.55 hrs    41.5%          4
TPU            2.30 hrs      20.00 hrs      17.70 hrs    11.5%          1

Recent Runs in Quota Window:
- 2026-09-18 14:20:10 UTC [GPU: t4-2x] username/exp-bert-finetune (duration: 3.50 hrs, status: complete)
- 2026-09-17 09:12:44 UTC [GPU: t4-2x] username/exp-lora-training (duration: 8.95 hrs, status: complete)
```

## Customizing Rolling Windows and Limits

You can adjust the time window or custom quota thresholds using CLI flags:

```bash
# Check past 14 days with custom 40-hour limit
kagglex quota --days 14 --gpu-limit 40.0 --tpu-limit 30.0
```

## Low Quota Pre-Flight Warnings

When submitting a new experiment via `kagglex run`, `kagglex` automatically estimates remaining quota. If your available balance falls below 2 hours, a warning is logged before execution:

```text
WARNING: Estimated GPU quota is low (1.20 hours remaining of 30.00 hours in past 7 days).
```

## Centralized Storage

All run records are stored in `~/.kagglex/runs.json`. This ensures that runs submitted from different repositories on your machine are aggregated under your account's single global quota.
