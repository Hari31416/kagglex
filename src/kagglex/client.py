"""Kaggle API client wrapper for pre-flight checks, kernel dispatch, monitoring, and output sync."""

import fnmatch
import json
import logging
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_EXCLUDES = [
    "cache/**",
    "cache",
    "pkg_src/**",
    "pkg_src",
    "pkg_payload.zip",
    "data_payload.zip",
    "__results__.html",
    "*.pyc",
    "*.pt",  # Unless explicitly included, skip raw PyTorch weights in default pull
]


def get_kaggle_api() -> Any:
    """Instantiate and authenticate KaggleApi client."""
    try:
        from kaggle.api.kaggle_api_extended import (
            KaggleApi,  # type: ignore[import-untyped]
        )

        api = KaggleApi()
        api.authenticate()
        return api
    except Exception as e:
        logger.error("Failed to authenticate Kaggle API: %s", e)
        raise RuntimeError(
            "Kaggle API authentication failed. Ensure ~/.kaggle/kaggle.json exists or "
            "KAGGLE_USERNAME and KAGGLE_KEY environment variables are set."
        ) from e


def check_kaggle_health() -> tuple[bool, str]:
    """Verify Kaggle credentials and network reachability.

    Returns:
        Tuple of (is_healthy, username_or_error).
    """
    try:
        user = get_authenticated_username()
        return True, user
    except Exception as e:
        return False, str(e)


def get_authenticated_username() -> str:
    """Retrieve the current authenticated Kaggle username."""
    api = get_kaggle_api()
    username = api.get_config_value("username")
    if (
        not username
        and hasattr(api, "config_values")
        and "username" in api.config_values
    ):
        username = api.config_values["username"]
    if not username:
        raise ValueError("Could not determine authenticated Kaggle username.")
    return str(username)


def verify_dataset_sources(dataset_slugs: list[str]) -> list[str]:
    """Check existence of requested dataset sources on Kaggle.

    Returns:
        List of missing or inaccessible dataset slugs.
    """
    if not dataset_slugs:
        return []

    api = get_kaggle_api()
    missing = []
    for slug in dataset_slugs:
        slug = slug.strip()
        try:
            status = api.dataset_status(slug)
            if not status:
                missing.append(slug)
        except Exception:
            missing.append(slug)
    return missing


def push_kernel(staging_dir: Path) -> tuple[str, str]:
    """Push staged kernel directory to Kaggle.

    Args:
        staging_dir: Path containing kernel-metadata.json and code.

    Returns:
        Tuple of (actual_kernel_id, kernel_url).
    """
    metadata_file = staging_dir / "kernel-metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(f"kernel-metadata.json not found in {staging_dir}")

    with open(metadata_file, encoding="utf-8") as f:
        meta = json.load(f)
    fallback_id = meta.get("id", "")

    logger.info("Submitting kernel '%s' to Kaggle...", fallback_id)
    api = get_kaggle_api()
    response = api.kernels_push(str(staging_dir))

    actual_id = fallback_id
    url = f"https://www.kaggle.com/code/{fallback_id}"

    if hasattr(response, "url") and response.url:
        url = response.url
        if "/code/" in url:
            actual_id = url.split("/code/")[-1].strip("/")

    logger.info("Kernel successfully submitted: %s (id: %s)", url, actual_id)
    if hasattr(response, "error") and response.error:
        logger.warning("Kernel push note: %s", response.error)
    return actual_id, url


def cancel_kernel(kernel_id: str) -> bool:
    """Cancel a running or queued Kaggle kernel.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').

    Returns:
        True if cancellation was signaled, False otherwise.
    """
    api = get_kaggle_api()
    logger.info("Requesting cancellation for kernel '%s'...", kernel_id)
    try:
        if hasattr(api, "kernels_cancel"):
            api.kernels_cancel(kernel_id)
            logger.info("Cancellation request sent for '%s'.", kernel_id)
            return True

        # Attempt cancellation via kagglesdk if available
        if hasattr(api, "build_kaggle_client"):
            try:
                from kagglesdk.kernels.types.kernels_api_service import (  # type: ignore[import-untyped]
                    ApiCancelKernelSessionRequest,
                )

                with api.build_kaggle_client() as k:
                    req = ApiCancelKernelSessionRequest()
                    k.kernels.kernels_api_client.cancel_kernel_session(req)
                    return True
            except Exception as rpc_err:
                logger.debug("RPC session cancel attempt note: %s", rpc_err)

        logger.warning(
            "Kaggle's public API does not support programmatic session cancellation. "
            "To terminate this kernel, click 'Cancel' / 'Stop' in the Kaggle web interface: "
            "https://www.kaggle.com/code/%s",
            kernel_id,
        )
        return False
    except Exception as e:
        logger.error(
            "Failed to cancel kernel '%s': %s. To cancel manually: https://www.kaggle.com/code/%s",
            kernel_id,
            e,
            kernel_id,
        )
        return False


def get_kernel_status(kernel_id: str) -> dict[str, Any]:
    """Fetch execution status for a Kaggle kernel.

    Args:
        kernel_id: Kernel identifier (e.g. 'username/slug').

    Returns:
        Dictionary containing status and message info.
    """
    api = get_kaggle_api()
    status_obj = api.kernels_status(kernel_id)

    status_str = "unknown"
    failure_msg = None

    if hasattr(status_obj, "status"):
        val = status_obj.status
        status_str = getattr(val, "name", str(val)).lower()
        failure_msg = getattr(status_obj, "failure_message", None)
    elif isinstance(status_obj, dict):
        status_str = str(status_obj.get("status", "unknown")).lower()
        failure_msg = status_obj.get("failureMessage")
    else:
        status_str = str(status_obj).lower()

    if "kernelworkerstatus." in status_str:
        status_str = status_str.split("kernelworkerstatus.")[-1]

    return {
        "id": kernel_id,
        "status": status_str,
        "failure_message": failure_msg,
    }


def poll_kernel(
    kernel_id: str,
    poll_interval_sec: int = 20,
    timeout_sec: int = 43200,
    on_status_change: Callable[[str, float], None] | None = None,
) -> dict[str, Any]:
    """Poll kernel status until completion, failure, or timeout.

    Args:
        kernel_id: Identifier of the kernel to monitor.
        poll_interval_sec: Seconds between status checks.
        timeout_sec: Maximum seconds to wait before timeout.
        on_status_change: Optional callback invoked when status changes.

    Returns:
        Final status dictionary.
    """
    logger.info("Monitoring Kaggle kernel '%s'...", kernel_id)
    start_time = time.time()
    last_status = ""

    terminal_statuses = {"complete", "error", "cancelacknowledged", "failed"}

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            raise TimeoutError(
                f"Kernel {kernel_id} timed out after {elapsed:.0f} seconds."
            )

        try:
            status_info = get_kernel_status(kernel_id)
            current_status = status_info["status"]
        except Exception as e:
            logger.warning(
                "Transient error checking kernel status (%s). Retrying in %ds...",
                e,
                poll_interval_sec,
            )
            time.sleep(poll_interval_sec)
            continue

        if current_status != last_status:
            logger.info(
                "Kernel status changed: [%s] (elapsed: %.0fs)",
                current_status.upper(),
                elapsed,
            )
            if on_status_change:
                on_status_change(current_status, elapsed)
            last_status = current_status

        if current_status in terminal_statuses:
            logger.info("Kernel reached terminal status: %s", current_status.upper())
            status_info["duration_sec"] = elapsed
            return status_info

        time.sleep(poll_interval_sec)


def stream_kernel_logs(
    kernel_id: str, on_line: Callable[[str], None] | None = None
) -> None:
    """Stream live logs from a running Kaggle kernel.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').
        on_line: Optional callback for each streamed log line.
    """
    api = get_kaggle_api()
    logger.info("Connecting to log stream for '%s'...", kernel_id)
    try:
        for chunk in api.kernels_logs_stream(kernel_id):
            data = chunk.get("data", "")
            if on_line:
                on_line(data)
            else:
                sys.stdout.write(data)
                sys.stdout.flush()
    except Exception as e:
        logger.debug("Log streaming completed or unavailable: %s", e)


def _match_pattern(rel_path_str: str, patterns: list[str]) -> bool:
    """Check if path string matches any glob pattern."""
    for pat in patterns:
        if fnmatch.fnmatch(rel_path_str, pat) or fnmatch.fnmatch(
            Path(rel_path_str).name, pat
        ):
            return True
        if pat.endswith("/**") and rel_path_str.startswith(pat[:-3]):
            return True
    return False


def pull_kernel_output(
    kernel_id: str,
    destination_dir: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Download output artifacts from a completed Kaggle kernel with selective filtering.

    Args:
        kernel_id: Identifier of the kernel (e.g. 'username/slug').
        destination_dir: Directory to save downloaded files.
        include_patterns: Optional glob patterns to include.
        exclude_patterns: Optional glob patterns to exclude.

    Returns:
        List of saved file paths.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading outputs for '%s' to %s...", kernel_id, destination_dir)

    effective_excludes = list(exclude_patterns or DEFAULT_OUTPUT_EXCLUDES)
    api = get_kaggle_api()

    # Determine server-side fetch patterns
    patterns_to_fetch: list[str | None] = []
    if include_patterns:
        for pat in include_patterns:
            if pat.endswith("/**"):
                patterns_to_fetch.append(pat[:-3] + "/*")
            else:
                patterns_to_fetch.append(pat)
    else:
        patterns_to_fetch = ["outputs/*"]

    saved_files: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        for s_pat in patterns_to_fetch:
            try:
                api.kernels_output(
                    kernel_id,
                    path=str(tmp_path),
                    file_pattern=s_pat,
                    force=True,
                    quiet=True,
                )
            except Exception as e:
                logger.debug("Server fetch pattern '%s' note: %s", s_pat, e)

        for file_path in tmp_path.rglob("*"):
            if not file_path.is_file():
                continue

            rel_str = str(file_path.relative_to(tmp_path))

            if _match_pattern(rel_str, effective_excludes):
                logger.debug("Excluding output file: %s", rel_str)
                continue

            if include_patterns and not _match_pattern(rel_str, include_patterns):
                logger.debug("Skipping non-included output file: %s", rel_str)
                continue

            clean_rel = rel_str
            if destination_dir.name == "outputs" and rel_str.startswith("outputs/"):
                clean_rel = rel_str[len("outputs/") :]

            dest_file = destination_dir / clean_rel
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, dest_file)
            saved_files.append(dest_file)

    logger.info("Downloaded %d output files to %s.", len(saved_files), destination_dir)
    return saved_files


def append_experiment_record(
    record_file: Path,
    kernel_id: str,
    status_info: dict[str, Any],
    output_dir: Path | None = None,
) -> None:
    """Update or append Kaggle execution details to an experiment record file."""
    if not record_file.exists():
        logger.warning(
            "Experiment record file %s does not exist. Skipping update.", record_file
        )
        return

    kernel_url = f"https://www.kaggle.com/code/{kernel_id}"
    status_str = status_info.get("status", "unknown").upper()
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    log_entry = [
        "",
        "### Kaggle Remote Execution",
        "",
        f"- **Kernel ID:** `{kernel_id}`",
        f"- **Kernel URL:** {kernel_url}",
        f"- **Completed At:** {timestamp}",
        f"- **Status:** `{status_str}`",
    ]

    if status_info.get("failure_message"):
        log_entry.append(f"- **Error:** {status_info['failure_message']}")

    if output_dir:
        try:
            rel_output = output_dir.relative_to(record_file.parent)
        except ValueError:
            rel_output = output_dir
        log_entry.append(f"- **Downloaded Outputs:** `{rel_output}`")

    log_entry.append("")

    with open(record_file, "a", encoding="utf-8") as f:
        f.write("\n".join(log_entry))

    logger.info("Updated experiment record: %s", record_file)
