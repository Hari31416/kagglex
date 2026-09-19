"""Local experiment run history repository and state tracking."""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RunRecord:
    """Record representing a submitted remote Kaggle experiment."""

    slug: str
    kernel_id: str
    title: str
    command: str
    accelerator: str
    submitted_at: str
    status: str = "queued"
    duration_sec: float | None = None
    output_dir: str | None = None
    url: str | None = None
    error_message: str | None = None


def get_history_file(repo_root: Path | None = None) -> Path:
    """Get path to local or user-level run history file."""
    if repo_root:
        hist_dir = repo_root / ".kagglex"
        legacy_file = repo_root / ".kagglerun" / "runs.json"
    else:
        hist_dir = Path.home() / ".kagglex"
        legacy_file = Path.home() / ".kagglerun" / "runs.json"
    hist_dir.mkdir(parents=True, exist_ok=True)
    target_file = hist_dir / "runs.json"
    if not target_file.exists() and legacy_file.exists():
        import shutil

        shutil.copy2(legacy_file, target_file)
    return target_file


def load_all_records(repo_root: Path | None = None) -> list[dict[str, Any]]:
    """Read all recorded runs from history file."""
    hist_file = get_history_file(repo_root)
    if not hist_file.exists():
        return []

    try:
        with open(hist_file, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning("Could not read history file %s: %s", hist_file, e)
    return []


def save_all_records(
    records: list[dict[str, Any]], repo_root: Path | None = None
) -> None:
    """Write all records to history file."""
    hist_file = get_history_file(repo_root)
    try:
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        logger.warning("Could not write history file %s: %s", hist_file, e)


def record_run(record: RunRecord, repo_root: Path | None = None) -> None:
    """Append or update a run in local history."""
    records = load_all_records(repo_root)
    record_dict = asdict(record)

    # Update if already exists
    updated = False
    for i, r in enumerate(records):
        if r.get("kernel_id") == record.kernel_id or r.get("slug") == record.slug:
            records[i] = record_dict
            updated = True
            break

    if not updated:
        records.insert(0, record_dict)

    save_all_records(records, repo_root)
    logger.debug(
        "Recorded run '%s' in %s", record.kernel_id, get_history_file(repo_root)
    )


def update_run(
    kernel_id_or_slug: str,
    updates: dict[str, Any],
    repo_root: Path | None = None,
) -> RunRecord | None:
    """Update attributes of an existing run record."""
    records = load_all_records(repo_root)
    target_idx = -1

    for i, r in enumerate(records):
        if (
            r.get("kernel_id") == kernel_id_or_slug
            or r.get("slug") == kernel_id_or_slug
            or r.get("kernel_id", "").endswith(f"/{kernel_id_or_slug}")
        ):
            target_idx = i
            break

    if target_idx == -1:
        return None

    records[target_idx].update(updates)
    save_all_records(records, repo_root)
    rec_dict = records[target_idx]
    return RunRecord(**rec_dict)


def get_run(kernel_id_or_slug: str, repo_root: Path | None = None) -> RunRecord | None:
    """Find a run record by kernel id or slug."""
    records = load_all_records(repo_root)
    for r in records:
        if (
            r.get("kernel_id") == kernel_id_or_slug
            or r.get("slug") == kernel_id_or_slug
            or r.get("kernel_id", "").endswith(f"/{kernel_id_or_slug}")
        ):
            return RunRecord(**r)
    return None


def list_runs(limit: int = 20, repo_root: Path | None = None) -> list[RunRecord]:
    """List recent run records ordered by submission time."""
    records = load_all_records(repo_root)
    results: list[RunRecord] = []
    for r in records[:limit]:
        try:
            results.append(RunRecord(**r))
        except Exception:
            continue
    return results


def parse_timestamp(timestamp_str: str) -> float | None:
    """Parse various timestamp string formats into Unix epoch seconds."""
    import datetime

    if not timestamp_str or not isinstance(timestamp_str, str):
        return None
    cleaned = timestamp_str.strip()
    if cleaned.endswith(" UTC"):
        cleaned = cleaned[:-4].strip()
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            dt = datetime.datetime.strptime(cleaned, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            continue
    return None


def get_quota_usage(
    repo_root: Path | None = None,
    window_days: int = 7,
    gpu_limit_hours: float = 30.0,
    tpu_limit_hours: float = 20.0,
    current_time: float | None = None,
) -> dict[str, Any]:
    """Calculate GPU and TPU accelerator quota usage across a rolling time window.

    Args:
        repo_root: Optional repository root to load history from.
        window_days: Number of days in rolling window (default: 7).
        gpu_limit_hours: Weekly GPU quota limit in hours (default: 30.0).
        tpu_limit_hours: Weekly TPU quota limit in hours (default: 20.0).
        current_time: Optional reference epoch timestamp (default: current time).

    Returns:
        Dictionary containing aggregated hours, limits, percentages, and run items.
    """
    import time

    now = current_time if current_time is not None else time.time()
    cutoff_epoch = now - (window_days * 86400.0)

    records = load_all_records(repo_root)

    gpu_seconds = 0.0
    tpu_seconds = 0.0
    runs_in_window: list[dict[str, Any]] = []

    for r in records:
        sub_time = parse_timestamp(r.get("submitted_at", ""))
        if sub_time is None or sub_time < cutoff_epoch:
            continue

        accel = str(r.get("accelerator", "none")).lower()
        dur = float(r.get("duration_sec") or 0.0)

        if accel in {"t4-2x", "p100", "t4", "p100-1x"}:
            gpu_seconds += dur
            accel_type = "GPU"
        elif accel in {"v3-8", "tpu", "v2-8"}:
            tpu_seconds += dur
            accel_type = "TPU"
        else:
            accel_type = "CPU"

        runs_in_window.append(
            {
                "kernel_id": r.get("kernel_id", ""),
                "title": r.get("title", ""),
                "accelerator": accel,
                "accelerator_type": accel_type,
                "submitted_at": r.get("submitted_at", ""),
                "duration_sec": dur,
                "status": r.get("status", "unknown"),
            }
        )

    gpu_hours = gpu_seconds / 3600.0
    tpu_hours = tpu_seconds / 3600.0

    gpu_remaining = max(0.0, gpu_limit_hours - gpu_hours)
    tpu_remaining = max(0.0, tpu_limit_hours - tpu_hours)

    gpu_pct = (
        min(100.0, (gpu_hours / gpu_limit_hours * 100.0))
        if gpu_limit_hours > 0
        else 0.0
    )
    tpu_pct = (
        min(100.0, (tpu_hours / tpu_limit_hours * 100.0))
        if tpu_limit_hours > 0
        else 0.0
    )

    return {
        "window_days": window_days,
        "gpu": {
            "used_hours": round(gpu_hours, 2),
            "limit_hours": gpu_limit_hours,
            "remaining_hours": round(gpu_remaining, 2),
            "used_pct": round(gpu_pct, 1),
            "run_count": sum(
                1 for item in runs_in_window if item["accelerator_type"] == "GPU"
            ),
        },
        "tpu": {
            "used_hours": round(tpu_hours, 2),
            "limit_hours": tpu_limit_hours,
            "remaining_hours": round(tpu_remaining, 2),
            "used_pct": round(tpu_pct, 1),
            "run_count": sum(
                1 for item in runs_in_window if item["accelerator_type"] == "TPU"
            ),
        },
        "runs": runs_in_window,
    }
