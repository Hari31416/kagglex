"""Tests for auto-dataset payload offloading and bootstrap resolution."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kagglex.api import KaggleRunner
from kagglex.bootstrap import generate_bootstrap_script
from kagglex.cli import create_parser
from kagglex.config import RunConfig
from kagglex.packager import package_local_data, package_project


def test_packager_allow_large_payload(tmp_path: Path) -> None:
    """Verify packager allows payloads > 5 MB when allow_large_payload is True."""
    proj_dir = tmp_path / "large_proj"
    proj_dir.mkdir()
    (proj_dir / "train.py").write_text("print('hello')\n")
    # Write 6 MB uncompressible random bytes so zip size exceeds 5 MB
    (proj_dir / "large_weights.dat").write_bytes(os.urandom(6 * 1024 * 1024))

    out_zip = tmp_path / "output.zip"

    # Should raise error without allow_large_payload
    with pytest.raises(ValueError, match="exceeds maximum limit"):
        package_project(proj_dir, out_zip, allow_large_payload=False)

    # Should succeed with allow_large_payload
    res = package_project(proj_dir, out_zip, allow_large_payload=True)
    assert res.exists()
    assert res.stat().st_size > 5 * 1024 * 1024


def test_package_local_data_allow_large_payload(tmp_path: Path) -> None:
    """Verify package_local_data allows data > 5 MB when allow_large_payload is True."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    large_file = data_dir / "large_data.dat"
    large_file.write_bytes(os.urandom(6 * 1024 * 1024))

    out_zip = tmp_path / "data.zip"
    with pytest.raises(ValueError, match="exceeds maximum upload limit"):
        package_local_data([large_file], tmp_path, out_zip, allow_large_payload=False)

    res = package_local_data([large_file], tmp_path, out_zip, allow_large_payload=True)
    assert res is not None
    assert res.exists()


def test_bootstrap_large_payload_omits_b64(tmp_path: Path) -> None:
    """Verify bootstrap script omits inline base64 for large files when auto_dataset is enabled."""
    large_zip = tmp_path / "large_pkg.zip"
    large_zip.write_bytes(b"0" * (6 * 1024 * 1024))

    config = RunConfig(command="python train.py", title="Large Run", auto_dataset=True)
    bootstrap_out = tmp_path / "kaggle_bootstrap.py"

    generate_bootstrap_script(
        config=config,
        output_path=bootstrap_out,
        pkg_zip_path=large_zip,
        allow_large_payload=True,
    )

    content = bootstrap_out.read_text(encoding="utf-8")
    assert 'PKG_PAYLOAD_B64 = ""' in content
    assert "pkg_payload.zip" in content


def test_runner_stage_auto_dataset(
    mock_kaggle_api: MagicMock, sample_src_project: Path, tmp_path: Path
) -> None:
    """Verify KaggleRunner stages auto-dataset folder and updates dataset_slugs."""
    staging_dir = tmp_path / "staging"
    runner = KaggleRunner(repo_root=sample_src_project, staging_dir=staging_dir)

    config = RunConfig(
        command="python -m my_ml_project.train",
        title="Auto Dataset Run",
        auto_dataset=True,
        auto_dataset_slug="custom-auto-ds",
    )

    stage_path = runner.stage(config)
    assert stage_path.exists()
    assert (stage_path / "payload_dataset").is_dir()
    assert (stage_path / "payload_dataset" / "dataset-metadata.json").is_file()
    assert (stage_path / "payload_dataset" / "pkg_payload.zip").is_file()
    assert "testuser/custom-auto-ds" in config.dataset_slugs


def test_runner_run_pushes_auto_dataset(
    mock_kaggle_api: MagicMock, sample_src_project: Path, tmp_path: Path
) -> None:
    """Verify KaggleRunner pushes auto-dataset before kernel submission."""
    staging_dir = tmp_path / "staging"
    runner = KaggleRunner(repo_root=sample_src_project, staging_dir=staging_dir)

    config = RunConfig(
        command="python -m my_ml_project.train",
        title="Auto Dataset Run",
        auto_dataset=True,
    )

    with patch("kagglex.api.push_dataset") as mock_push_ds:
        mock_push_ds.return_value = (
            "https://www.kaggle.com/datasets/testuser/kagglex-payload-auto-dataset-run"
        )
        job = runner.run(config=config, wait=False)

        assert mock_push_ds.called
        assert job.kernel_id == "testuser/auto-dataset-run"


def test_cli_parser_auto_dataset() -> None:
    """Verify CLI parser parses --auto-dataset and --auto-dataset-slug."""
    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "--file",
            "train.py",
            "--auto-dataset",
            "--auto-dataset-slug",
            "my-payload-slug",
        ]
    )
    assert args.auto_dataset is True
    assert args.auto_dataset_slug == "my-payload-slug"
