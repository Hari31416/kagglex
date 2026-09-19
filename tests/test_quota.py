"""Tests for GPU and TPU quota tracking and CLI dashboard."""

import time
from pathlib import Path
from unittest.mock import patch

from kagglex.cli import create_parser, handle_quota, handle_run
from kagglex.history import (
    RunRecord,
    get_quota_usage,
    parse_timestamp,
    record_run,
)


def test_parse_timestamp() -> None:
    """Verify parse_timestamp handles standard UTC formats."""
    ts_str = "2026-09-19 10:00:00 UTC"
    epoch = parse_timestamp(ts_str)
    assert epoch is not None

    ts_iso = "2026-09-19T10:00:00"
    epoch_iso = parse_timestamp(ts_iso)
    assert epoch_iso == epoch

    assert parse_timestamp("") is None
    assert parse_timestamp("invalid-date") is None


def test_get_quota_usage_empty(tmp_path: Path) -> None:
    """Verify get_quota_usage with no runs."""
    usage = get_quota_usage(repo_root=tmp_path)
    assert usage["gpu"]["used_hours"] == 0.0
    assert usage["gpu"]["remaining_hours"] == 30.0
    assert usage["gpu"]["used_pct"] == 0.0
    assert usage["gpu"]["run_count"] == 0

    assert usage["tpu"]["used_hours"] == 0.0
    assert usage["tpu"]["remaining_hours"] == 20.0
    assert usage["runs"] == []


def test_get_quota_usage_aggregated(tmp_path: Path) -> None:
    """Verify get_quota_usage aggregates recent runs within the time window."""
    now = time.time()

    # Recent GPU run: 2 hours (7200 sec)
    r1 = RunRecord(
        slug="gpu-run-1",
        kernel_id="testuser/gpu-run-1",
        title="GPU Run 1",
        command="python train.py",
        accelerator="t4-2x",
        submitted_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now - 3600)),
        status="complete",
        duration_sec=7200.0,
    )
    record_run(r1, repo_root=tmp_path)

    # Recent TPU run: 1.5 hours (5400 sec)
    r2 = RunRecord(
        slug="tpu-run-1",
        kernel_id="testuser/tpu-run-1",
        title="TPU Run 1",
        command="python train_tpu.py",
        accelerator="v3-8",
        submitted_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now - 7200)),
        status="complete",
        duration_sec=5400.0,
    )
    record_run(r2, repo_root=tmp_path)

    # Old run outside 7-day window: 10 hours
    r3 = RunRecord(
        slug="old-run",
        kernel_id="testuser/old-run",
        title="Old Run",
        command="python train.py",
        accelerator="t4-2x",
        submitted_at=time.strftime(
            "%Y-%m-%d %H:%M:%S UTC", time.gmtime(now - (10 * 86400))
        ),
        status="complete",
        duration_sec=36000.0,
    )
    record_run(r3, repo_root=tmp_path)

    usage = get_quota_usage(repo_root=tmp_path, window_days=7, current_time=now)
    assert usage["gpu"]["used_hours"] == 2.0
    assert usage["gpu"]["remaining_hours"] == 28.0
    assert usage["gpu"]["used_pct"] == round((2.0 / 30.0) * 100.0, 1)
    assert usage["gpu"]["run_count"] == 1

    assert usage["tpu"]["used_hours"] == 1.5
    assert usage["tpu"]["remaining_hours"] == 18.5
    assert usage["tpu"]["run_count"] == 1
    assert len(usage["runs"]) == 2


def test_handle_quota_cli(tmp_path: Path, capsys) -> None:
    """Verify handle_quota CLI formatting."""
    parser = create_parser()
    args = parser.parse_args(["quota", "--days", "7", "--gpu-limit", "30.0"])

    with patch("kagglex.cli.find_repo_root", return_value=tmp_path):
        ret = handle_quota(args)
        assert ret == 0

    captured = capsys.readouterr().out
    assert "Kaggle Accelerator Quota Usage" in captured
    assert "GPU" in captured
    assert "TPU" in captured


def test_handle_run_quota_warning(tmp_path: Path) -> None:
    """Verify handle_run emits a warning when GPU quota is nearly exhausted."""
    now = time.time()
    # 29 hours used out of 30
    r1 = RunRecord(
        slug="heavy-gpu-run",
        kernel_id="testuser/heavy-gpu-run",
        title="Heavy GPU Run",
        command="python train.py",
        accelerator="t4-2x",
        submitted_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now - 1000)),
        status="complete",
        duration_sec=29.0 * 3600.0,
    )
    record_run(r1, repo_root=tmp_path)

    (tmp_path / "train.py").write_text("print(1)\n")

    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "--dir",
            str(tmp_path),
            "--file",
            str(tmp_path / "train.py"),
            "--gpu",
            "t4-2x",
            "--dry-run",
        ]
    )

    with (
        patch("kagglex.cli.logger.warning") as mock_warn,
        patch("kagglex.api.KaggleRunner.stage", return_value=tmp_path / "staged"),
        patch("kagglex.cli.check_kaggle_health", return_value=(True, "testuser")),
    ):
        ret = handle_run(args)
        assert ret == 0
        # Check that warning was logged
        warning_calls = [
            c[0][0]
            for c in mock_warn.call_args_list
            if "Estimated GPU quota" in c[0][0]
        ]
        assert len(warning_calls) > 0
