"""Unit tests for remote bootstrap generator."""

from pathlib import Path

from kagglerun.bootstrap import generate_bootstrap_script
from kagglerun.config import RunConfig


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
