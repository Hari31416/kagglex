# Python API Reference

`kagglex` exposes a programmatic Python API for embedding remote Kaggle compute into automated pipelines and notebooks.

## KaggleRunner

The primary interface for dispatching and monitoring batch jobs.

```python
from pathlib import Path
from kagglex import KaggleRunner, RunConfig

runner = KaggleRunner(repo_root=Path("."))

config = RunConfig(
    command="python train.py --epochs 20",
    title="ResNet Baseline Experiment",
    gpu_type="t4-2x",
    multi_gpu=True,
    dataset_slugs=["username/image-dataset"],
    extra_pip_deps=["timm>=1.0.0", "wandb"],
    env_vars={"WANDB_PROJECT": "image-classification"},
    kaggle_secrets=["WANDB_API_KEY"],
)

# Submit, stream logs, and retrieve results
job = runner.run(
    config=config,
    wait=True,
    stream=True,
    pull=True,
)

print(f"Run completed with status: {job.status}")
print(f"Outputs downloaded to: {job.config.output_dir}")
```

### Methods

#### `KaggleRunner.stage(config: RunConfig) -> Path`

Packages the target project and creates local staging files without submitting to Kaggle.

#### `KaggleRunner.run(config: RunConfig, wait: bool = True, stream: bool = False, pull: bool = True) -> Job`

Submits the kernel to Kaggle and optionally waits for completion, streams logs, and pulls output files.

#### `KaggleRunner.list_runs(limit: int = 20) -> list[RunRecord]`

Returns recent run history records from `~/.kagglex/runs.json`.

#### `KaggleRunner.cancel(kernel_id_or_slug: str) -> bool`

Cancels a remote Kaggle kernel.

## InteractiveClient

Connect programmatically to an active Kaggle Jupyter Proxy session for rapid execution.

```python
from kagglex import InteractiveClient

client = InteractiveClient(
    url="https://kkb-production.jupyter-proxy.kaggle.net/k/12345/abc?token=xyz",
    timeout=60,
)

# Verify health
is_connected, msg = client.test_connection()
print(f"Connected: {is_connected} ({msg})")

# Execute code snippet
result = client.execute_code("import torch; print(torch.cuda.is_available())")
print(result["stdout"])

# Query GPU specs
gpu_info = client.get_gpu_info()
print(gpu_info)

# Transfer files
client.upload_file("checkpoint.pt")
client.download_file("results.json", local_path="results.json")
```

## Configuration Classes

### RunConfig

```python
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class RunConfig:
    command: str
    title: str
    slug: str | None = None
    project_dir: Path | None = None
    target_file: Path | None = None
    gpu_type: str = "t4-2x"
    enable_tpu: bool = False
    multi_gpu: bool = False
    enable_internet: bool = True
    dataset_slugs: list[str] = field(default_factory=list)
    include_data: list[str] = field(default_factory=list)
    extra_pip_deps: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    kaggle_secrets: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    auto_dataset: bool = False
    auto_dataset_slug: str | None = None
```

### DatasetConfig

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass
class DatasetConfig:
    title: str
    data_dir: Path
    slug: str | None = None
    is_public: bool = False
    license_name: str = "CC0-1.0"
```
