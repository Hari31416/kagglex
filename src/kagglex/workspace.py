"""Workspace inspection, project type detection, and pre-flight validation."""

import ast
from enum import Enum
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class ProjectType(str, Enum):
    """Supported workspace project structures."""

    STANDALONE_SCRIPT = "standalone_script"
    SRC_LAYOUT = "src_layout"
    FLAT_PACKAGE = "flat_package"
    SIMPLE_DIRECTORY = "simple_directory"


def find_repo_root(start_dir: Optional[Path] = None) -> Path:
    """Find the root of the project or repository.

    Searches upward for .git, pyproject.toml, setup.py, setup.cfg, or requirements.txt.
    Falls back to current working directory.
    """
    current = (start_dir or Path.cwd()).resolve()
    marker_files = {
        ".git",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt",
    }

    for parent in [current, *current.parents]:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    return current


def detect_project_type(target_path: Path) -> Tuple[ProjectType, Path]:
    """Identify project structure and canonical root path.

    Args:
        target_path: File or directory path to inspect.

    Returns:
        Tuple of (ProjectType, root_path).
    """
    resolved = target_path.resolve()

    if resolved.is_file():
        if resolved.suffix == ".py":
            return ProjectType.STANDALONE_SCRIPT, resolved.parent
        return ProjectType.STANDALONE_SCRIPT, resolved.parent

    # Directory checks
    if (resolved / "src").is_dir() and (
        (resolved / "pyproject.toml").is_file() or (resolved / "setup.py").is_file()
    ):
        return ProjectType.SRC_LAYOUT, resolved

    # Check for flat package with __init__.py in a top-level folder
    has_init = any(
        sub.is_dir() and (sub / "__init__.py").is_file()
        for sub in resolved.iterdir()
        if not sub.name.startswith((".", "_"))
    )
    if (
        has_init
        or (resolved / "pyproject.toml").is_file()
        or (resolved / "setup.py").is_file()
    ):
        return ProjectType.FLAT_PACKAGE, resolved

    return ProjectType.SIMPLE_DIRECTORY, resolved


def validate_python_syntax(file_path: Path) -> None:
    """Validate Python syntax locally before dispatching to remote GPU.

    Raises:
        SyntaxError: If the script contains syntax errors.
        FileNotFoundError: If the target script does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Target script does not exist: {file_path}")

    source_code = file_path.read_text(encoding="utf-8")
    try:
        ast.parse(source_code, filename=str(file_path))
    except SyntaxError as e:
        logger.error(
            "Syntax validation failed for %s on line %s: %s",
            file_path.name,
            e.lineno,
            e.msg,
        )
        raise


def extract_script_path_from_command(command: str, base_dir: Path) -> Optional[Path]:
    """Attempt to locate target Python script from command string for syntax validation.

    Handles 'python train.py', 'python3 path/to/script.py', etc.
    """
    parts = command.strip().split()
    if not parts:
        return None

    # Check for 'python <script.py>'
    for i, token in enumerate(parts):
        if token in {"python", "python3"} and i + 1 < len(parts):
            candidate = parts[i + 1]
            if candidate.endswith(".py") and not candidate.startswith("-"):
                script_path = (base_dir / candidate).resolve()
                if script_path.exists():
                    return script_path
        elif token.endswith(".py") and not token.startswith("-"):
            script_path = (base_dir / token).resolve()
            if script_path.exists():
                return script_path
    return None


def detect_dependencies(project_dir: Path) -> List[str]:
    """Detect declared dependencies in project directory."""
    deps: List[str] = []
    req_file = project_dir / "requirements.txt"
    if req_file.exists():
        for line in req_file.read_text(encoding="utf-8").splitlines():
            clean = line.strip()
            if clean and not clean.startswith("#"):
                deps.append(clean)
    return deps
