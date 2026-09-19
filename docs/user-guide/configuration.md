# Configuration Hierarchy

`kagglex` supports a multi-tier configuration system that enables setting machine-wide defaults while allowing individual projects and CLI invocations to override them.

## Precedence Order

When executing commands, configuration values are resolved in the following order (later sources override earlier ones):

1. **User Global Configuration**: `~/.kagglex/config.toml` (or `~/.kagglex/kagglex.toml`)
2. **Project pyproject.toml**: `[tool.kagglex]` table in the project root
3. **Project kagglex.toml**: `kagglex.toml` file in the project root
4. **CLI Arguments**: Explicit flags supplied on the command line

## Global Configuration

Create `~/.kagglex/config.toml` to establish user-wide defaults across all your machine learning repositories:

```toml
# ~/.kagglex/config.toml
gpu = "p100"
quota_days = 7
gpu_weekly_limit_hours = 30.0
tpu_weekly_limit_hours = 20.0
kaggle_secrets = ["WANDB_API_KEY", "HF_TOKEN"]

[env]
WANDB_ENTITY = "my-research-lab"
```

## Project Configuration

### Using pyproject.toml

Add a `[tool.kagglex]` section to your repository's `pyproject.toml`:

```toml
# pyproject.toml
[tool.kagglex]
command = "python -m my_package.train --batch-size 32"
title = "Vision Transformer Pretraining"
gpu = "t4-2x"
multi_gpu = true
auto_dataset = false
include_data = ["./data/tokenizer"]
include_outputs = ["*.json", "checkpoints/best.pt"]

[tool.kagglex.env]
WANDB_PROJECT = "vit-pretraining"
LR = "3e-4"
```

### Using kagglex.toml

Alternatively, define settings in a standalone `kagglex.toml` file in your project directory:

```toml
# kagglex.toml
gpu = "v3-8"
title = "TPU Training Run"
include_outputs = ["eval_metrics.csv"]
```

## Dictionary Merging

For structured keys such as environment variables (`[env]` or `[tool.kagglex.env]`), `kagglex` merges dictionaries hierarchically. Variables defined in `~/.kagglex/config.toml` are preserved in child projects unless specifically overridden by the project configuration.
