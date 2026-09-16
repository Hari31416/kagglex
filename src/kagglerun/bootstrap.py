"""Generator for the remote Kaggle bootstrap orchestrator script."""

import base64
import json
import logging
from pathlib import Path
from typing import Optional

from kagglerun.config import RunConfig

logger = logging.getLogger(__name__)

# Remote execution template executed inside the Kaggle environment
BOOTSTRAP_TEMPLATE = '''"""Bootstrap orchestrator executed remotely inside Kaggle kernel."""

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Injected parameters
TARGET_COMMAND = __TARGET_COMMAND_JSON__
MULTI_GPU = __MULTI_GPU_JSON__
EXTRA_PIP_DEPS = __EXTRA_PIP_DEPS_JSON__
ENV_VARS = __ENV_VARS_JSON__
KAGGLE_SECRETS = __KAGGLE_SECRETS_JSON__
PKG_PAYLOAD_B64 = __PKG_PAYLOAD_B64_JSON__
DATA_PAYLOAD_B64 = __DATA_PAYLOAD_B64_JSON__


def log(msg: str) -> None:
    """Print timestamped log message to stdout."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [KAGGLERUN] {msg}", flush=True)


def print_environment_info() -> None:
    """Log system diagnostics including CPU, GPU, TPU, and Python environment."""
    log("=" * 60)
    log("System & Hardware Diagnostics:")
    log(f"Python: {sys.version.split()[0]} ({sys.executable})")
    log(f"Working Directory: {os.getcwd()}")

    # CUDA / GPU check
    try:
        import torch

        log(f"PyTorch Version: {torch.__version__}")
        log(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            log(f"CUDA Device Count: {count}")
            for i in range(count):
                prop = torch.cuda.get_device_properties(i)
                vram_gb = prop.total_memory / (1024**3)
                log(f"  GPU {i}: {prop.name} ({vram_gb:.2f} GB VRAM)")
        else:
            log("Running without CUDA GPU.")
    except Exception as e:
        log(f"PyTorch diagnostic check failed: {e}")

    # TPU check
    if "TPU_NAME" in os.environ or "COLAB_TPU_ADDR" in os.environ:
        log("TPU environment detected.")

    try:
        res = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            log("nvidia-smi output:")
            for line in res.stdout.strip().splitlines():
                log(f"  {line}")
    except Exception:
        pass

    # Log available input datasets / kernel sources
    input_dir = Path("/kaggle/input")
    if input_dir.exists():
        log("Mounted datasets and sources in /kaggle/input:")
        for item in sorted(input_dir.glob("*")):
            log(f"  {item.name}/")
            if item.is_dir():
                for sub in sorted(item.glob("*"))[:10]:
                    log(f"    {sub.name}")

    log("=" * 60)


def load_kaggle_secrets() -> None:
    """Retrieve requested secrets from Kaggle UserSecretsClient."""
    if not KAGGLE_SECRETS:
        return

    try:
        from kaggle_secrets import UserSecretsClient

        user_secrets = UserSecretsClient()
        for secret_key in KAGGLE_SECRETS:
            try:
                secret_val = user_secrets.get_secret(secret_key)
                os.environ[secret_key] = str(secret_val)
                log(f"Injected Kaggle secret: {secret_key}")
            except Exception as e:
                log(f"Warning: could not retrieve Kaggle secret '{secret_key}': {e}")
    except ImportError:
        log("kaggle_secrets module not found. Skipping secret retrieval.")
    except Exception as e:
        log(f"Error accessing Kaggle secrets: {e}")


def extract_package_and_install() -> None:
    """Extract staged project source and install dependencies."""
    extract_dir = Path("/kaggle/working/pkg_src")
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    pkg_zip = Path("/kaggle/working/pkg_payload.zip")

    # If payload zip not on disk, attempt base64 decode fallback
    if not pkg_zip.exists() and PKG_PAYLOAD_B64:
        pkg_zip.write_bytes(base64.b64decode(PKG_PAYLOAD_B64.encode("ascii")))

    if pkg_zip.exists():
        log(f"Extracting package archive from {pkg_zip}...")
        with zipfile.ZipFile(pkg_zip, "r") as zf:
            zf.extractall(extract_dir)

        # Add src and extract_dir to sys.path
        src_dir = extract_dir / "src"
        if src_dir.exists():
            sys.path.insert(0, str(src_dir.resolve()))
            log(f"Added {src_dir} to sys.path.")
        sys.path.insert(0, str(extract_dir.resolve()))

        # Determine dependency installation strategy
        pyproject = extract_dir / "pyproject.toml"
        setup_py = extract_dir / "setup.py"
        req_txt = extract_dir / "requirements.txt"

        if pyproject.exists() or setup_py.exists():
            log(f"Installing package from {extract_dir} in editable mode...")
            res = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", str(extract_dir)],
                capture_output=True,
                text=True,
            )
            if res.returncode != 0:
                log(f"Pip install error: {res.stderr}")
            else:
                log("Successfully installed package.")
        elif req_txt.exists():
            log(f"Installing dependencies from {req_txt}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(req_txt)],
                check=False,
            )
    else:
        log("No package payload archive found.")


def extract_data_payload() -> None:
    """Extract bundled data payload if present."""
    data_zip = Path("/kaggle/working/data_payload.zip")
    if not data_zip.exists() and DATA_PAYLOAD_B64:
        data_zip.write_bytes(base64.b64decode(DATA_PAYLOAD_B64.encode("ascii")))

    if data_zip.exists():
        log("Extracting local data payload to /kaggle/working...")
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall("/kaggle/working")
        log("Extracted data payload.")


def install_extra_deps() -> None:
    """Install extra pip packages specified in config."""
    if not EXTRA_PIP_DEPS:
        return
    log(f"Installing extra pip dependencies: {EXTRA_PIP_DEPS}")
    cmd = [sys.executable, "-m", "pip", "install"] + EXTRA_PIP_DEPS
    subprocess.run(cmd, check=False)


def build_execution_command() -> str:
    """Wrap command with torchrun if multi-GPU execution is requested."""
    cmd = TARGET_COMMAND.strip()

    if MULTI_GPU:
        num_gpus = 1
        try:
            import torch

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
        except Exception:
            num_gpus = 1

        if num_gpus > 1:
            log(f"Multi-GPU enabled: wrapping command with torchrun for {num_gpus} GPUs.")
            if cmd.startswith("python -m "):
                module_name = cmd[len("python -m ") :]
                cmd = f"torchrun --nproc_per_node={num_gpus} -m {module_name}"
            elif cmd.startswith("python "):
                script_path = cmd[len("python ") :]
                cmd = f"torchrun --nproc_per_node={num_gpus} {script_path}"

    return cmd


def main() -> int:
    """Bootstrap entrypoint."""
    start_time = time.time()
    log("Starting remote kagglerun bootstrap...")

    output_dir = Path("/kaggle/working/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set custom environment variables
    for k, v in ENV_VARS.items():
        os.environ[k] = str(v)

    # Redirect caches to /tmp to prevent bloat in downloaded outputs
    os.environ.setdefault("HF_HOME", "/tmp/huggingface")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface")
    os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/huggingface")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    load_kaggle_secrets()
    print_environment_info()
    extract_package_and_install()
    extract_data_payload()
    install_extra_deps()

    exec_cmd = build_execution_command()
    log(f"Executing command: {exec_cmd}")
    log("=" * 60)

    # Stream output live while writing to log file
    proc = subprocess.Popen(
        exec_cmd,
        shell=True,
        cwd="/kaggle/working",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log_file_path = output_dir / "execution.log"
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        if proc.stdout:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)

    proc.wait()
    return_code = proc.returncode
    duration_sec = time.time() - start_time

    log("=" * 60)
    log(f"Execution completed with exit code {return_code} in {duration_sec:.1f}s.")

    # Write execution summary JSON
    summary = {
        "command": TARGET_COMMAND,
        "executed_command": exec_cmd,
        "exit_code": return_code,
        "duration_seconds": round(duration_sec, 2),
        "status": "SUCCESS" if return_code == 0 else "FAILED",
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary, sf, indent=2)

    return return_code


if __name__ == "__main__":
    sys.exit(main())
'''


def generate_bootstrap_script(
    config: RunConfig,
    output_path: Path,
    pkg_zip_path: Optional[Path] = None,
    data_zip_path: Optional[Path] = None,
) -> Path:
    """Generate kaggle_bootstrap.py with injected parameters.

    Args:
        config: Run configuration.
        output_path: Destination path for kaggle_bootstrap.py.
        pkg_zip_path: Optional path to package zip for base64 fallback.
        data_zip_path: Optional path to data zip for base64 fallback.

    Returns:
        Path to generated bootstrap script.
    """
    pkg_b64 = ""
    # Only inline base64 if small (< 5MB) for resilient fallback
    if (
        pkg_zip_path
        and pkg_zip_path.exists()
        and pkg_zip_path.stat().st_size < 5 * 1024 * 1024
    ):
        pkg_b64 = base64.b64encode(pkg_zip_path.read_bytes()).decode("ascii")

    data_b64 = ""
    if (
        data_zip_path
        and data_zip_path.exists()
        and data_zip_path.stat().st_size < 5 * 1024 * 1024
    ):
        data_b64 = base64.b64encode(data_zip_path.read_bytes()).decode("ascii")

    script_content = (
        BOOTSTRAP_TEMPLATE.replace(
            "__TARGET_COMMAND_JSON__", json.dumps(config.command)
        )
        .replace("__MULTI_GPU_JSON__", repr(config.multi_gpu))
        .replace("__EXTRA_PIP_DEPS_JSON__", json.dumps(config.extra_pip_deps))
        .replace("__ENV_VARS_JSON__", json.dumps(config.env_vars))
        .replace("__KAGGLE_SECRETS_JSON__", json.dumps(config.kaggle_secrets))
        .replace("__PKG_PAYLOAD_B64_JSON__", json.dumps(pkg_b64))
        .replace("__DATA_PAYLOAD_B64_JSON__", json.dumps(data_b64))
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(script_content)

    logger.debug("Generated bootstrap script at %s", output_path)
    return output_path
