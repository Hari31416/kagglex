"""Unit tests for configuration models and validations."""

from pathlib import Path
import pytest

from kagglex.config import DatasetConfig, RunConfig, slugify


def test_slugify() -> None:
    """Test Kaggle slug generation rules."""
    assert slugify("My Experiment #1") == "my-experiment-1"
    assert slugify("Indic_Edge_Pilot_01") == "indic_edge_pilot_01"
    assert slugify("a") == "a-exp"
    assert len(slugify("a" * 100)) <= 50


def test_run_config_validation() -> None:
    """Test RunConfig validation rules and normalization."""
    cfg = RunConfig(title="BERT Fine Tuning", command="python train.py")
    assert cfg.slug == "bert-fine-tuning"
    assert cfg.gpu_type == "t4-2x"
    assert cfg.enable_tpu is False
    assert cfg.enable_internet is True

    # Test TPU configuration
    tpu_cfg = RunConfig(
        title="TPU Experiment",
        command="python train.py",
        gpu_type="v3-8",
    )
    assert tpu_cfg.enable_tpu is True

    # Test invalid accelerator
    with pytest.raises(ValueError, match="Invalid gpu_type"):
        RunConfig(title="Test", command="train", gpu_type="invalid-gpu")

    # Test empty command
    with pytest.raises(ValueError, match="Command cannot be empty"):
        RunConfig(title="Test", command="   ")

    # Test empty title
    with pytest.raises(ValueError, match="Title cannot be empty"):
        RunConfig(title="   ", command="train")


def test_dataset_config_validation(tmp_path: Path) -> None:
    """Test DatasetConfig validation and existence check."""
    data_dir = tmp_path / "sample_data"
    data_dir.mkdir()

    cfg = DatasetConfig(title="Sample Dataset", data_dir=data_dir)
    assert cfg.slug == "sample-dataset"
    assert cfg.license_name == "CC0-1.0"
    assert cfg.is_public is False

    with pytest.raises(FileNotFoundError):
        DatasetConfig(title="Missing", data_dir=tmp_path / "non_existent_folder")
