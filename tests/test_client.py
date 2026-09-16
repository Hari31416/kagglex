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
    push_kernel,
)


def test_match_pattern() -> None:
    """Test output glob pattern matching."""
    assert _match_pattern("outputs/eval.json", ["outputs/**"])
    assert _match_pattern("outputs/eval.json", ["*.json"])
    assert _match_pattern("cache/huggingface/blob", ["cache/**"])
    assert not _match_pattern("outputs/eval.json", ["cache/**"])


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
