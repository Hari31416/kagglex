"""Configuration models and validation rules for kagglerun."""

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

SUPPORTED_ACCELERATORS: Set[str] = {
    "t4-2x",
    "p100",
    "v3-8",
    "none",
}


def slugify(value: str) -> str:
    """Convert arbitrary string into a valid Kaggle slug.

    Kaggle slugs must be lowercase, alphanumeric with hyphens, and 5-50 chars.
    """
    slug = re.sub(r"[^\w\s-]", "", value.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")
    if len(slug) < 5:
        slug = f"{slug}-exp"
    return slug[:50]


@dataclass
class RunConfig:
    """Configuration for a remote Kaggle experiment execution."""

    command: str
    title: str
    slug: Optional[str] = None
    project_dir: Optional[Path] = None
    target_file: Optional[Path] = None
    gpu_type: str = "t4-2x"
    enable_tpu: bool = False
    multi_gpu: bool = False
    enable_internet: bool = True
    dataset_slugs: List[str] = field(default_factory=list)
    include_data: List[str] = field(default_factory=list)
    extra_pip_deps: List[str] = field(default_factory=list)
    env_vars: Dict[str, str] = field(default_factory=dict)
    kaggle_secrets: List[str] = field(default_factory=list)
    parent_kernels: List[str] = field(default_factory=list)
    output_dir: Optional[Path] = None
    record_file: Optional[Path] = None
    include_outputs: List[str] = field(default_factory=list)
    exclude_outputs: List[str] = field(default_factory=list)
    poll_interval: int = 20
    timeout_sec: int = 43200

    def __post_init__(self) -> None:
        """Validate and normalize configuration parameters."""
        if not self.command or not self.command.strip():
            raise ValueError("Command cannot be empty.")
        self.command = self.command.strip()

        if not self.title or not self.title.strip():
            raise ValueError("Title cannot be empty.")
        self.title = self.title.strip()

        if not self.slug:
            self.slug = slugify(self.title)
        else:
            self.slug = slugify(self.slug)

        self.gpu_type = self.gpu_type.lower()
        if self.gpu_type not in SUPPORTED_ACCELERATORS:
            raise ValueError(
                f"Invalid gpu_type '{self.gpu_type}'. Must be one of {sorted(SUPPORTED_ACCELERATORS)}"
            )

        if self.gpu_type == "v3-8":
            self.enable_tpu = True

        if self.project_dir and isinstance(self.project_dir, str):
            self.project_dir = Path(self.project_dir).resolve()

        if self.target_file and isinstance(self.target_file, str):
            self.target_file = Path(self.target_file).resolve()

        if self.output_dir and isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir).resolve()

        if self.record_file and isinstance(self.record_file, str):
            self.record_file = Path(self.record_file).resolve()


@dataclass
class DatasetConfig:
    """Configuration for Kaggle dataset creation and versioning."""

    title: str
    data_dir: Path
    slug: Optional[str] = None
    is_public: bool = False
    license_name: str = "CC0-1.0"

    def __post_init__(self) -> None:
        """Validate and normalize dataset parameters."""
        if not self.title or not self.title.strip():
            raise ValueError("Dataset title cannot be empty.")
        self.title = self.title.strip()

        if not self.slug:
            self.slug = slugify(self.title)
        else:
            self.slug = slugify(self.slug)

        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir).resolve()
        else:
            self.data_dir = self.data_dir.resolve()

        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.data_dir}"
            )
