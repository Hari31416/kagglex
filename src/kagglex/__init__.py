"""kagglex: Execute local Python code, modules, and experiments seamlessly on Kaggle GPUs."""

from kagglex.api import Job, KaggleRunner
from kagglex.config import DatasetConfig, InteractiveConfig, RunConfig
from kagglex.history import RunRecord
from kagglex.interactive import JupyterProxyClient

__version__ = "0.2.0"
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
