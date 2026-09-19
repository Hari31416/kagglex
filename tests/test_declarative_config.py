"""Tests for declarative project configuration via pyproject.toml and kagglex.toml."""

from pathlib import Path
from unittest.mock import patch

from kagglex.cli import create_parser, handle_run
from kagglex.config import load_project_config, read_toml_file, resolve_jupyter_url


def test_read_toml_file(tmp_path: Path) -> None:
    """Verify read_toml_file successfully parses a valid TOML file."""
    toml_path = tmp_path / "test.toml"
    toml_path.write_text(
        'title = "My Run"\ngpu = "p100"\nmulti_gpu = true\n', encoding="utf-8"
    )
    data = read_toml_file(toml_path)
    assert data == {"title": "My Run", "gpu": "p100", "multi_gpu": True}


def test_load_project_config_pyproject(tmp_path: Path) -> None:
    """Verify load_project_config extracts [tool.kagglex] from pyproject.toml."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "demo"\n\n[tool.kagglex]\ngpu = "p100"\nmulti_gpu = true\nsecrets = ["WANDB_API_KEY"]\n',
        encoding="utf-8",
    )
    cfg = load_project_config(tmp_path)
    assert cfg.get("gpu") == "p100"
    assert cfg.get("multi_gpu") is True
    assert cfg.get("secrets") == ["WANDB_API_KEY"]


def test_load_project_config_kagglex_toml_precedence(tmp_path: Path) -> None:
    """Verify kagglex.toml overrides pyproject.toml."""
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[tool.kagglex]\ngpu = "p100"\nmulti_gpu = false\n',
        encoding="utf-8",
    )
    kagglex_toml = tmp_path / "kagglex.toml"
    kagglex_toml.write_text(
        'gpu = "v3-8"\nmulti_gpu = true\n',
        encoding="utf-8",
    )

    cfg = load_project_config(tmp_path)
    assert cfg.get("gpu") == "v3-8"
    assert cfg.get("multi_gpu") is True


def test_resolve_jupyter_url_from_config(tmp_path: Path) -> None:
    """Verify resolve_jupyter_url reads jupyter URL from kagglex.toml if env is not set."""
    kagglex_toml = tmp_path / "kagglex.toml"
    kagglex_toml.write_text(
        'url = "https://kkb-production.jupyter-proxy.kaggle.net/k/123/token=abc"\n',
        encoding="utf-8",
    )
    with patch.dict("os.environ", {}, clear=True):
        url = resolve_jupyter_url(start_dir=tmp_path)
        assert url == "https://kkb-production.jupyter-proxy.kaggle.net/k/123/token=abc"


def test_cli_handle_run_with_pyproject_config(tmp_path: Path) -> None:
    """Verify handle_run picks up options from pyproject.toml when not supplied on CLI."""
    proj_dir = tmp_path / "app"
    proj_dir.mkdir()
    script = proj_dir / "main.py"
    script.write_text("print('running')\n", encoding="utf-8")

    pyproject = proj_dir / "pyproject.toml"
    pyproject.write_text(
        """
[tool.kagglex]
command = "python main.py"
title = "Configured Experiment"
gpu = "p100"
multi_gpu = true
kaggle_secrets = ["HF_TOKEN"]

[tool.kagglex.env]
MODE = "eval"
""",
        encoding="utf-8",
    )

    parser = create_parser()
    args = parser.parse_args(["run", "--dir", str(proj_dir), "--dry-run"])

    with patch("kagglex.api.KaggleRunner.stage") as mock_stage:
        mock_stage.return_value = tmp_path / "staged"
        ret = handle_run(args)
        assert ret == 0

        # Inspect the staged RunConfig passed
        assert mock_stage.called
        config = mock_stage.call_args[0][0]
        assert config.command == "python main.py"
        assert config.title == "Configured Experiment"
        assert config.gpu_type == "p100"
        assert config.multi_gpu is True
        assert config.kaggle_secrets == ["HF_TOKEN"]
        assert config.env_vars == {"MODE": "eval"}


def test_cli_handle_run_cli_overrides_config(tmp_path: Path) -> None:
    """Verify explicit CLI arguments take precedence over config files."""
    proj_dir = tmp_path / "app"
    proj_dir.mkdir()
    script = proj_dir / "main.py"
    script.write_text("print('running')\n", encoding="utf-8")

    kagglex_toml = proj_dir / "kagglex.toml"
    kagglex_toml.write_text(
        'command = "python main.py"\ngpu = "p100"\ntitle = "TOML Title"\n',
        encoding="utf-8",
    )

    parser = create_parser()
    args = parser.parse_args(
        [
            "run",
            "--dir",
            str(proj_dir),
            "--gpu",
            "v3-8",
            "--title",
            "CLI Override Title",
            "--dry-run",
        ]
    )

    with patch("kagglex.api.KaggleRunner.stage") as mock_stage:
        mock_stage.return_value = tmp_path / "staged"
        ret = handle_run(args)
        assert ret == 0

        config = mock_stage.call_args[0][0]
        assert config.gpu_type == "v3-8"
        assert config.title == "CLI Override Title"
