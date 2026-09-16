"""Unit tests for programmatic Python API (KaggleRunner, Job)."""

from pathlib import Path
from unittest.mock import MagicMock

from kagglerun.api import KaggleRunner
from kagglerun.config import RunConfig


def test_kaggle_runner_stage(
    mock_kaggle_api: MagicMock, sample_src_project: Path, tmp_path: Path
) -> None:
    """Test staging a run via KaggleRunner."""
    runner = KaggleRunner(
        repo_root=sample_src_project, staging_dir=tmp_path / "staging"
    )
    cfg = RunConfig(
        title="API Staging Test",
        command="python -m my_ml_project.train",
    )

    staging_dir = runner.stage(cfg)
    assert staging_dir.exists()
    assert (staging_dir / "pkg_payload.zip").exists()
    assert (staging_dir / "kaggle_bootstrap.py").exists()
    assert (staging_dir / "kernel-metadata.json").exists()


def test_kaggle_runner_run(
    mock_kaggle_api: MagicMock, sample_src_project: Path, tmp_path: Path
) -> None:
    """Test submitting and tracking a Job via KaggleRunner."""
    runner = KaggleRunner(
        repo_root=sample_src_project, staging_dir=tmp_path / "staging"
    )

    job = runner.run(
        command="python -m my_ml_project.train",
        title="API Run Test",
        wait=False,
        pull=False,
    )

    assert job.kernel_id == "testuser/api-run-test"
    assert job.status == "complete"

    # Test listing runs
    runs = runner.list_runs()
    assert len(runs) >= 1
    assert any(r.slug == "api-run-test" for r in runs)
