"""Unit tests for remote bootstrap generator."""

import os
from pathlib import Path

import pytest

from kagglex.bootstrap import generate_bootstrap_script
from kagglex.config import RunConfig


def test_generate_bootstrap_script(tmp_path: Path) -> None:
    """Test generating remote bootstrap script with injected parameters."""
    cfg = RunConfig(
        title="Distributed Run",
        command="python -m mypkg.train --epochs 5",
        multi_gpu=True,
        extra_pip_deps=["scikit-learn>=1.4.0"],
        env_vars={"CUSTOM_VAR": "val"},
        kaggle_secrets=["WANDB_API_KEY", "HF_TOKEN"],
    )

    out_file = tmp_path / "kaggle_bootstrap.py"
    generate_bootstrap_script(cfg, output_path=out_file)

    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")

    # Check key components
    assert "mypkg.train" in content
    assert "torchrun --nproc_per_node=" in content
    assert "scikit-learn>=1.4.0" in content
    assert "CUSTOM_VAR" in content
    assert "WANDB_API_KEY" in content
    assert "UserSecretsClient" in content

    # Verify generated bootstrap code compiles without syntax errors
    compile(content, "kaggle_bootstrap.py", "exec")


def test_generate_bootstrap_script_with_payload(tmp_path: Path) -> None:
    """Test that small payloads are inlined into the bootstrap script."""
    cfg = RunConfig(title="Inline Run", command="python train.py")
    pkg_zip = tmp_path / "pkg_payload.zip"
    pkg_zip.write_bytes(b"dummy zip content")

    out_file = tmp_path / "kaggle_bootstrap.py"
    generate_bootstrap_script(cfg, output_path=out_file, pkg_zip_path=pkg_zip)

    content = out_file.read_text(encoding="utf-8")
    assert "PKG_PAYLOAD_B64 = " in content
    assert 'PKG_PAYLOAD_B64 = ""' not in content


def test_generate_bootstrap_script_pkg_size_limit_exceeded(tmp_path: Path) -> None:
    """Test that generate_bootstrap_script raises ValueError when pkg_payload.zip > 5 MB."""
    cfg = RunConfig(title="Oversized Run", command="python train.py")
    pkg_zip = tmp_path / "pkg_payload.zip"
    pkg_zip.write_bytes(os.urandom(6 * 1024 * 1024))

    out_file = tmp_path / "kaggle_bootstrap.py"
    with pytest.raises(
        ValueError, match=r"pkg_payload\.zip size .* exceeds the 5 MB limit"
    ):
        generate_bootstrap_script(cfg, output_path=out_file, pkg_zip_path=pkg_zip)


def test_generate_bootstrap_script_data_size_limit_exceeded(tmp_path: Path) -> None:
    """Test that generate_bootstrap_script raises ValueError when data_payload.zip > 5 MB."""
    cfg = RunConfig(title="Oversized Data Run", command="python train.py")
    data_zip = tmp_path / "data_payload.zip"
    data_zip.write_bytes(os.urandom(6 * 1024 * 1024))

    out_file = tmp_path / "kaggle_bootstrap.py"
    with pytest.raises(
        ValueError, match=r"data_payload\.zip size .* exceeds the 5 MB limit"
    ):
        generate_bootstrap_script(cfg, output_path=out_file, data_zip_path=data_zip)
