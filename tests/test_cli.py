"""Unit tests for CLI argument parsing and commands."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kagglerun.cli import create_parser, main


def test_cli_parser_run() -> None:
    """Test parsing 'run' arguments."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "python train.py --epochs 5",
            "--title",
            "Training BERT",
            "--gpu",
            "t4-2x",
            "--multi-gpu",
            "--datasets",
            "user/dataset1",
            "--env",
            "BATCH_SIZE=32",
            "--kaggle-secrets",
            "WANDB_API_KEY",
            "--parent-kernels",
            "user/step1-clean",
            "--stream",
        ]
    )

    assert args.subcommand == "run"
    assert args.command_pos == "python train.py --epochs 5"
    assert args.title == "Training BERT"
    assert args.gpu == "t4-2x"
    assert args.multi_gpu is True
    assert args.datasets == ["user/dataset1"]
    assert args.env == ["BATCH_SIZE=32"]
    assert args.kaggle_secrets == ["WANDB_API_KEY"]
    assert args.parent_kernels == ["user/step1-clean"]
    assert args.stream is True


def test_cli_parser_cancel() -> None:
    """Test parsing 'cancel' subcommand."""
    parser = create_parser()
    args = parser.parse_args(["cancel", "my-experiment"])
    assert args.subcommand == "cancel"
    assert args.kernel == "my-experiment"


def test_cli_parser_list() -> None:
    """Test parsing 'list' subcommand."""
    parser = create_parser()
    args = parser.parse_args(["list", "--limit", "5"])
    assert args.subcommand == "list"
    assert args.limit == 5


def test_cli_dry_run(
    mock_kaggle_api: MagicMock, sample_script: Path, tmp_path: Path
) -> None:
    """Test CLI dry-run staging execution."""
    exit_code = main(
        [
            "run",
            "--file",
            str(sample_script),
            "--title",
            "Dry Run Test",
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_cli_parser_exec() -> None:
    """Test parsing 'exec' subcommand arguments."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "exec",
            "print('test')",
            "--url",
            "https://proxy.kaggle.net?token=123",
            "--timeout",
            "45",
        ]
    )
    assert args.subcommand == "exec"
    assert args.code == "print('test')"
    assert args.url == "https://proxy.kaggle.net?token=123"
    assert args.timeout == 45


def test_cli_exec_missing_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test 'exec' exits with error code 1 when no URL is provided or configured."""
    monkeypatch.delenv("KAGGLE_JUPYTER_URL", raising=False)
    exit_code = main(["exec", "print(123)"])
    assert exit_code == 1


def test_cli_exec_test_connection() -> None:
    """Test 'exec --test' command invokes client.test_connection."""
    with patch(
        "kagglerun.interactive.JupyterProxyClient.test_connection", return_value=True
    ):
        exit_code = main(["exec", "--url", "https://proxy.kaggle.net/proxy", "--test"])
        assert exit_code == 0
