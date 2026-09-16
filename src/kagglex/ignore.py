"""Ignore rule parser and file exclusion matcher using pathspec."""

import logging
from pathlib import Path
from typing import List, Optional, Set

import pathspec

logger = logging.getLogger(__name__)

DEFAULT_IGNORE_PATTERNS: List[str] = [
    # VCS
    ".git",
    ".git/**",
    # Virtual environments
    ".venv",
    ".venv/**",
    "venv",
    "venv/**",
    "env",
    "env/**",
    # Caches and bytecode
    "__pycache__",
    "__pycache__/**",
    "*.py[cod]",
    ".pytest_cache",
    ".pytest_cache/**",
    ".mypy_cache",
    ".mypy_cache/**",
    ".ruff_cache",
    ".ruff_cache/**",
    # Build artifacts
    "build",
    "build/**",
    "dist",
    "dist/**",
    "*.egg-info",
    "*.egg-info/**",
    # IDE / OS
    ".DS_Store",
    ".idea",
    ".idea/**",
    ".vscode",
    ".vscode/**",
    # Local runs and staging
    ".kagglerun",
    ".kagglerun/**",
    ".kagglex",
    ".kagglex/**",
    "artifacts",
    "artifacts/**",
    # Large ML checkpoints and data (default to prevent accidental multi-GB uploads)
    "*.pt",
    "*.pth",
    "*.ckpt",
    "*.safetensors",
    "*.bin",
]


class IgnoreFilter:
    """Evaluates whether files or directories should be excluded from packaging."""

    def __init__(
        self,
        base_dir: Path,
        custom_patterns: Optional[List[str]] = None,
        use_gitignore: bool = True,
        use_kaggleignore: bool = True,
    ) -> None:
        self.base_dir = base_dir.resolve()
        patterns: List[str] = list(DEFAULT_IGNORE_PATTERNS)

        if use_gitignore:
            gi = self.base_dir / ".gitignore"
            if gi.is_file():
                patterns.extend(self._read_patterns_file(gi))

        if use_kaggleignore:
            ki = self.base_dir / ".kaggleignore"
            if ki.is_file():
                patterns.extend(self._read_patterns_file(ki))

        if custom_patterns:
            patterns.extend(custom_patterns)

        self.spec = pathspec.PathSpec.from_lines("gitignore", patterns)

    @staticmethod
    def _read_patterns_file(path: Path) -> List[str]:
        """Read ignore patterns from file."""
        lines: List[str] = []
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    lines.append(stripped)
        except Exception as e:
            logger.warning("Could not read ignore file %s: %s", path, e)
        return lines

    def should_ignore(self, path: Path) -> bool:
        """Check if path relative to base_dir matches any ignore pattern."""
        try:
            rel_path = path.resolve().relative_to(self.base_dir)
        except ValueError:
            rel_path = path

        rel_str = str(rel_path)
        if rel_str == ".":
            return False

        if path.is_dir():
            rel_str_dir = rel_str + "/"
            return self.spec.match_file(rel_str) or self.spec.match_file(rel_str_dir)

        return self.spec.match_file(rel_str)
