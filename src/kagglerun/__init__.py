"""kagglerun: Execute local Python code, modules, and experiments seamlessly on Kaggle GPUs."""

from kagglerun.api import Job, KaggleRunner
from kagglerun.config import DatasetConfig, InteractiveConfig, RunConfig
from kagglerun.history import RunRecord
from kagglerun.interactive import JupyterProxyClient

__version__ = "0.1.0"
__all__ = [
    "Job",
    "JupyterProxyClient",
    "KaggleRunner",
    "RunConfig",
    "DatasetConfig",
    "InteractiveConfig",
    "RunRecord",
    "__version__",
]
