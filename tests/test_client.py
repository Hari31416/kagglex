"""Unit tests for Kaggle client interactions, status polling, and pattern matching."""

from pathlib import Path
from unittest.mock import MagicMock

from kagglex.client import (
    _match_pattern,
    cancel_kernel,
    check_kaggle_health,
    get_authenticated_username,
    get_kernel_status,
    poll_kernel,
    pull_kernel_output,
    push_kernel,
    _build_server_fetch_pattern,
    _glob_to_regex,
)


def test_match_pattern() -> None:
    """Test output glob pattern matching."""
    assert _match_pattern("outputs/eval.json", ["outputs/**"])
    assert _match_pattern("outputs/eval.json", ["*.json"])
    assert _match_pattern("cache/huggingface/blob", ["cache/**"])
    assert not _match_pattern("outputs/eval.json", ["cache/**"])


def test_glob_to_regex() -> None:
    """Test translation of shell glob patterns to regex."""
    import re

    regex_json = _glob_to_regex("*.json")
    compiled_json = re.compile(regex_json)
    assert compiled_json.search("outputs/metrics.json")
    assert compiled_json.search("metrics.json")
    assert not compiled_json.search("outputs/model.pt")

    regex_dir = _glob_to_regex("outputs/**")
    compiled_dir = re.compile(regex_dir)
    assert compiled_dir.search("outputs/metrics.json")
    assert compiled_dir.search("outputs/sub/model.pt")

    regex_slash = _glob_to_regex("eval/")
    compiled_slash = re.compile(regex_slash)
    assert compiled_slash.search("outputs/eval/metrics.json")


def test_build_server_fetch_pattern() -> None:
    """Test server-side regex pattern construction."""
    import re

    default_pattern = _build_server_fetch_pattern()
    compiled_default = re.compile(default_pattern)
    assert compiled_default.search("outputs/metrics.json")
    assert compiled_default.search("outputs/summary.csv")
    assert not compiled_default.search("outputs/model.pt")
    assert not compiled_default.search("outputs/checkpoints/best.pt")
    assert not compiled_default.search("outputs/__pycache__/mod.pyc")

    single_pattern = _build_server_fetch_pattern(["*.json"])
    compiled_single = re.compile(single_pattern)
    assert compiled_single.search("outputs/metrics.json")
    assert not compiled_single.search("outputs/summary.csv")

    multi_pattern = _build_server_fetch_pattern(["*.json", "checkpoints/**"])
    compiled_multi = re.compile(multi_pattern)
    assert compiled_multi.search("outputs/metrics.json")
    assert compiled_multi.search("outputs/checkpoints/model.pt")
    assert not compiled_multi.search("outputs/summary.csv")


def test_pull_kernel_output_default(mock_kaggle_api: MagicMock, tmp_path: Path) -> None:
    """Test pull_kernel_output with default settings."""

    def fake_kernels_output(kernel_id: str, path: str, **kwargs: object) -> None:
        p = Path(path)
        (p / "outputs").mkdir(parents=True, exist_ok=True)
        (p / "outputs" / "metrics.json").write_text('{"score": 0.9}', encoding="utf-8")
        (p / "outputs" / "model.pt").write_bytes(b"dummy weight")
        (p / "pkg_payload.zip").write_bytes(b"dummy zip")

    mock_kaggle_api.kernels_output.side_effect = fake_kernels_output
    dest = tmp_path / "outputs"

    saved = pull_kernel_output("testuser/test-exp", destination_dir=dest)
    saved_rel = [str(f.relative_to(dest)) for f in saved]

    assert "metrics.json" in saved_rel
    assert "model.pt" not in saved_rel
    assert "pkg_payload.zip" not in saved_rel
    mock_kaggle_api.kernels_output.assert_called_once()


def test_pull_kernel_output_include_patterns(
    mock_kaggle_api: MagicMock, tmp_path: Path
) -> None:
    """Test pull_kernel_output with explicit include patterns."""

    def fake_kernels_output(kernel_id: str, path: str, **kwargs: object) -> None:
        p = Path(path)
        (p / "outputs").mkdir(parents=True, exist_ok=True)
        (p / "outputs" / "metrics.json").write_text('{"score": 0.9}', encoding="utf-8")
        (p / "outputs" / "summary.csv").write_text("a,b\n1,2", encoding="utf-8")

    mock_kaggle_api.kernels_output.side_effect = fake_kernels_output
    dest = tmp_path / "results"

    saved = pull_kernel_output(
        "testuser/test-exp",
        destination_dir=dest,
        include_patterns=["*.json"],
    )
    saved_rel = [str(f.relative_to(dest)) for f in saved]

    assert "outputs/metrics.json" in saved_rel
    assert "outputs/summary.csv" not in saved_rel


def test_authenticated_username(mock_kaggle_api: MagicMock) -> None:
    """Test fetching authenticated Kaggle username."""
    user = get_authenticated_username()
    assert user == "testuser"

    healthy, name = check_kaggle_health()
    assert healthy is True
    assert name == "testuser"


def test_push_kernel(mock_kaggle_api: MagicMock, tmp_path: Path) -> None:
    """Test submitting kernel using mocked API."""
    meta_file = tmp_path / "kernel-metadata.json"
    meta_file.write_text('{"id": "testuser/test-experiment"}', encoding="utf-8")

    kernel_id, url = push_kernel(tmp_path)
    assert kernel_id == "testuser/test-experiment"
    assert "https://www.kaggle.com/code/" in url
    mock_kaggle_api.kernels_push.assert_called_once_with(str(tmp_path))


def test_get_kernel_status(mock_kaggle_api: MagicMock) -> None:
    """Test retrieving kernel status."""
    info = get_kernel_status("testuser/test-exp")
    assert info["status"] == "complete"
    assert info["id"] == "testuser/test-exp"


def test_poll_kernel_terminal(mock_kaggle_api: MagicMock) -> None:
    """Test polling kernel that immediately reaches complete status."""
    info = poll_kernel("testuser/test-exp", poll_interval_sec=1)
    assert info["status"] == "complete"


def test_cancel_kernel(mock_kaggle_api: MagicMock) -> None:
    """Test cancelling kernel."""
    ok = cancel_kernel("testuser/test-exp")
    assert ok is True
    mock_kaggle_api.kernels_cancel.assert_called_once_with("testuser/test-exp")
