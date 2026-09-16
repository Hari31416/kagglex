"""Unit tests for workspace detection and pre-flight validation."""

from pathlib import Path
import pytest

from kagglex.workspace import (
    ProjectType,
    detect_project_type,
    extract_script_path_from_command,
    find_repo_root,
    validate_python_syntax,
)


def test_detect_project_type_standalone_script(tmp_path: Path) -> None:
    """Test detecting standalone Python script."""
    script = tmp_path / "train.py"
    script.write_text("print('hello')", encoding="utf-8")

    proj_type, root = detect_project_type(script)
    assert proj_type == ProjectType.STANDALONE_SCRIPT
    assert root == tmp_path


def test_detect_project_type_src_layout(sample_src_project: Path) -> None:
    """Test detecting src/ layout project."""
    proj_type, root = detect_project_type(sample_src_project)
    assert proj_type == ProjectType.SRC_LAYOUT
    assert root == sample_src_project


def test_validate_python_syntax(tmp_path: Path) -> None:
    """Test pre-flight Python syntax verification."""
    valid_script = tmp_path / "valid.py"
    valid_script.write_text("def foo():\n    return 42\n", encoding="utf-8")
    # Should not raise
    validate_python_syntax(valid_script)

    invalid_script = tmp_path / "invalid.py"
    invalid_script.write_text("def foo(:\n    return 42\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        validate_python_syntax(invalid_script)


def test_extract_script_path_from_command(tmp_path: Path) -> None:
    """Test extracting script path from command string."""
    script = tmp_path / "train.py"
    script.write_text("pass", encoding="utf-8")

    extracted = extract_script_path_from_command(
        "python train.py --epochs 10", tmp_path
    )
    assert extracted == script

    extracted2 = extract_script_path_from_command("python3 train.py", tmp_path)
    assert extracted2 == script

    extracted_none = extract_script_path_from_command("pytest tests", tmp_path)
    assert extracted_none is None


def test_find_repo_root(tmp_path: Path) -> None:
    """Test locating repo root by marker files."""
    repo = tmp_path / "my_repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("", encoding="utf-8")
    sub = repo / "subfolder" / "deep"
    sub.mkdir(parents=True)

    found = find_repo_root(sub)
    assert found == repo
