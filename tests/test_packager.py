"""Unit tests for packaging and metadata generation."""

import json
import zipfile
from pathlib import Path

from kagglex.config import RunConfig
from kagglex.packager import (
    create_kernel_metadata,
    package_local_data,
    package_project,
)


def test_package_project_src_layout(sample_src_project: Path, tmp_path: Path) -> None:
    """Test packaging project directory and verifying excluded files."""
    output_zip = tmp_path / "pkg_payload.zip"
    package_project(sample_src_project, output_zip)

    assert output_zip.exists()
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        assert "pyproject.toml" in names
        assert "src/my_ml_project/__init__.py" in names
        assert "src/my_ml_project/train.py" in names
        # Caches should be excluded
        assert not any("__pycache__" in n for n in names)


def test_package_project_standalone_script(sample_script: Path, tmp_path: Path) -> None:
    """Test packaging a standalone script."""
    output_zip = tmp_path / "script_payload.zip"
    package_project(sample_script.parent, output_zip, target_file=sample_script)

    assert output_zip.exists()
    with zipfile.ZipFile(output_zip, "r") as zf:
        names = zf.namelist()
        assert "train.py" in names


def test_package_local_data(tmp_path: Path) -> None:
    """Test packaging local data manifests and files."""
    data_dir = tmp_path / "data" / "eval"
    data_dir.mkdir(parents=True)
    (data_dir / "samples.jsonl").write_text('{"label": 1}', encoding="utf-8")

    out_zip = tmp_path / "data_payload.zip"
    res = package_local_data([data_dir], base_dir=tmp_path, output_zip=out_zip)

    assert res is not None
    assert out_zip.exists()
    with zipfile.ZipFile(out_zip, "r") as zf:
        assert any("samples.jsonl" in n for n in zf.namelist())


def test_create_kernel_metadata(tmp_path: Path) -> None:
    """Test generating kernel-metadata.json with accelerator, datasets, and parent kernels."""
    cfg = RunConfig(
        title="BERT Experiment",
        command="python train.py",
        slug="bert-exp",
        gpu_type="t4-2x",
        dataset_slugs=["user/custom-data", "external-dataset"],
        parent_kernels=["user/prep-step-1"],
    )

    meta_file = create_kernel_metadata(
        config=cfg,
        kaggle_username="testuser",
        staging_dir=tmp_path,
        code_file="kaggle_bootstrap.py",
    )

    assert meta_file.exists()
    with open(meta_file, encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["id"] == "testuser/bert-exp"
    assert meta["title"] == "BERT Experiment"
    assert meta["enable_gpu"] == "true"
    assert meta["enable_tpu"] == "false"
    assert meta["dataset_sources"] == ["user/custom-data", "testuser/external-dataset"]
    assert meta["kernel_sources"] == ["user/prep-step-1"]
