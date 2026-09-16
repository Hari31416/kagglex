"""Unit tests for ignore filters and pattern matching."""

from pathlib import Path

from kagglerun.ignore import IgnoreFilter


def test_ignore_filter_defaults(tmp_path: Path) -> None:
    """Test default ignore patterns (.git, .venv, __pycache__, checkpoints)."""
    filter_engine = IgnoreFilter(base_dir=tmp_path)

    assert filter_engine.should_ignore(tmp_path / ".git")
    assert filter_engine.should_ignore(tmp_path / ".venv" / "bin" / "python")
    assert filter_engine.should_ignore(tmp_path / "src" / "__pycache__" / "foo.pyc")
    assert filter_engine.should_ignore(tmp_path / "model.pt")
    assert filter_engine.should_ignore(tmp_path / "weights.safetensors")

    assert not filter_engine.should_ignore(tmp_path / "train.py")
    assert not filter_engine.should_ignore(tmp_path / "src" / "my_pkg" / "module.py")


def test_ignore_filter_custom_files(tmp_path: Path) -> None:
    """Test reading custom .gitignore and .kaggleignore rules."""
    (tmp_path / ".gitignore").write_text("local_temp/\n*.secret\n", encoding="utf-8")
    (tmp_path / ".kaggleignore").write_text("big_data/\n", encoding="utf-8")

    filter_engine = IgnoreFilter(base_dir=tmp_path)

    assert filter_engine.should_ignore(tmp_path / "local_temp" / "file.txt")
    assert filter_engine.should_ignore(tmp_path / "api.secret")
    assert filter_engine.should_ignore(tmp_path / "big_data" / "corpus.csv")
    assert not filter_engine.should_ignore(tmp_path / "README.md")
