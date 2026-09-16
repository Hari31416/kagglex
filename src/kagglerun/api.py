"""Programmatic Python API for kagglerun."""

import json
import logging
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from kagglerun.bootstrap import generate_bootstrap_script
from kagglerun.client import (
    cancel_kernel,
    get_authenticated_username,
    get_kernel_status,
    poll_kernel,
    pull_kernel_output,
    push_kernel,
    stream_kernel_logs,
)
from kagglerun.config import RunConfig
from kagglerun.history import RunRecord, get_run, list_runs, record_run, update_run
from kagglerun.packager import (
    create_kernel_metadata,
    package_local_data,
    package_project,
)
from kagglerun.workspace import (
    ProjectType,
    detect_project_type,
    extract_script_path_from_command,
    find_repo_root,
    validate_python_syntax,
)

logger = logging.getLogger(__name__)


class Job:
    """Handle for a submitted Kaggle remote job."""

    def __init__(
        self, kernel_id: str, url: str, config: RunConfig, repo_root: Path
    ) -> None:
        self.kernel_id = kernel_id
        self.url = url
        self.config = config
        self.repo_root = repo_root
        self._last_status_info: Optional[Dict[str, Any]] = None

    @property
    def status(self) -> str:
        """Fetch current remote execution status."""
        status_info = get_kernel_status(self.kernel_id)
        self._last_status_info = status_info
        return str(status_info.get("status", "unknown"))

    @property
    def last_status_info(self) -> Optional[Dict[str, Any]]:
        """Cached or last fetched status dictionary."""
        return self._last_status_info

    def wait(
        self,
        poll_interval_sec: Optional[int] = None,
        timeout_sec: Optional[int] = None,
        on_status_change: Optional[Callable[[str, float], None]] = None,
    ) -> Dict[str, Any]:
        """Block until kernel terminates (complete, error, cancel).

        Args:
            poll_interval_sec: Seconds between polling requests.
            timeout_sec: Maximum seconds before TimeoutError.
            on_status_change: Optional callback for status changes.

        Returns:
            Final status info dictionary.
        """
        interval = poll_interval_sec or self.config.poll_interval
        timeout = timeout_sec or self.config.timeout_sec
        info = poll_kernel(
            self.kernel_id,
            poll_interval_sec=interval,
            timeout_sec=timeout,
            on_status_change=on_status_change,
        )
        self._last_status_info = info
        update_run(
            self.kernel_id,
            {
                "status": info.get("status", "unknown"),
                "duration_sec": info.get("duration_sec"),
                "error_message": info.get("failure_message"),
            },
            repo_root=self.repo_root,
        )
        return info

    def stream_logs(self, on_line: Optional[Callable[[str], None]] = None) -> None:
        """Stream live execution logs to stdout or callback."""
        stream_kernel_logs(self.kernel_id, on_line=on_line)

    def cancel(self) -> bool:
        """Cancel this job on Kaggle."""
        ok = cancel_kernel(self.kernel_id)
        if ok:
            update_run(
                self.kernel_id, {"status": "cancelled"}, repo_root=self.repo_root
            )
        return ok

    def pull_outputs(
        self,
        destination_dir: Optional[Path] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> List[Path]:
        """Download output files from completed job."""
        dest = destination_dir or self.config.output_dir or (self.repo_root / "outputs")
        inc = include_patterns or self.config.include_outputs
        exc = exclude_patterns or self.config.exclude_outputs
        saved = pull_kernel_output(
            self.kernel_id,
            destination_dir=dest,
            include_patterns=inc,
            exclude_patterns=exc,
        )
        update_run(self.kernel_id, {"output_dir": str(dest)}, repo_root=self.repo_root)
        return saved


class KaggleRunner:
    """High-level runner orchestrating local packaging, remote execution, and output sync."""

    def __init__(
        self,
        repo_root: Optional[Path] = None,
        staging_dir: Optional[Path] = None,
    ) -> None:
        self.repo_root = (repo_root or find_repo_root()).resolve()
        self.staging_dir = staging_dir or (
            self.repo_root / "artifacts" / "kaggle_staging"
        )

    def stage(self, config: RunConfig) -> Path:
        """Stage project files, payloads, and bootstrap scripts locally."""
        slug = config.slug or "run"
        staging_base = self.staging_dir / slug
        staging_base.mkdir(parents=True, exist_ok=True)
        user = get_authenticated_username()

        # Target project directory
        proj_dir = config.project_dir or self.repo_root

        # Pre-flight syntax validation if target python script exists
        script_to_check = config.target_file or extract_script_path_from_command(
            config.command, proj_dir
        )
        if script_to_check and script_to_check.exists():
            validate_python_syntax(script_to_check)

        # 1. Package Python project
        pkg_zip = staging_base / "pkg_payload.zip"
        package_project(
            project_dir=proj_dir,
            output_zip=pkg_zip,
            target_file=config.target_file,
        )

        # 2. Package local data if requested
        data_zip_path = None
        if config.include_data:
            data_paths = [Path(p) for p in config.include_data]
            data_zip = staging_base / "data_payload.zip"
            res = package_local_data(data_paths, base_dir=proj_dir, output_zip=data_zip)
            if res:
                data_zip_path = data_zip

        # 3. Generate bootstrap script
        bootstrap_file = staging_base / "kaggle_bootstrap.py"
        generate_bootstrap_script(
            config=config,
            output_path=bootstrap_file,
            pkg_zip_path=pkg_zip,
            data_zip_path=data_zip_path,
        )

        # 4. Generate kernel-metadata.json
        create_kernel_metadata(
            config=config,
            kaggle_username=user,
            staging_dir=staging_base,
            code_file="kaggle_bootstrap.py",
        )

        return staging_base

    def run(
        self,
        config: Optional[RunConfig] = None,
        command: Optional[str] = None,
        title: Optional[str] = None,
        wait: bool = True,
        stream: bool = False,
        pull: bool = True,
        **kwargs: Any,
    ) -> Job:
        """Stage, submit, monitor, and optionally pull outputs for a Kaggle run."""
        if config is None:
            if not command or not title:
                raise ValueError(
                    "Must provide either a RunConfig or both command and title."
                )
            config = RunConfig(command=command, title=title, **kwargs)

        staging_base = self.stage(config)
        actual_kernel_id, kernel_url = push_kernel(staging_base)

        job = Job(
            kernel_id=actual_kernel_id,
            url=kernel_url,
            config=config,
            repo_root=self.repo_root,
        )

        # Record in local history
        rec = RunRecord(
            slug=config.slug or config.title,
            kernel_id=actual_kernel_id,
            title=config.title,
            command=config.command,
            accelerator=config.gpu_type,
            submitted_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            status="queued",
            url=kernel_url,
        )
        record_run(rec, repo_root=self.repo_root)

        if not wait:
            return job

        if stream:
            job.stream_logs()

        status_info = job.wait()
        if pull and status_info.get("status") == "complete":
            job.pull_outputs()

        return job

    def list_runs(self, limit: int = 20) -> List[RunRecord]:
        """List recent runs from local history."""
        return list_runs(limit=limit, repo_root=self.repo_root)

    def cancel(self, kernel_id_or_slug: str) -> bool:
        """Cancel a run by kernel id or slug."""
        rec = get_run(kernel_id_or_slug, repo_root=self.repo_root)
        kernel_id = rec.kernel_id if rec else kernel_id_or_slug
        return cancel_kernel(kernel_id)
