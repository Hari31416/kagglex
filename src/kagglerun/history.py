"""Local experiment run history repository and state tracking."""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

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
    duration_sec: Optional[float] = None
    output_dir: Optional[str] = None
    url: Optional[str] = None
    error_message: Optional[str] = None


def get_history_file(repo_root: Optional[Path] = None) -> Path:
    """Get path to local or user-level run history file."""
    if repo_root:
        hist_dir = repo_root / ".kagglerun"
    else:
        hist_dir = Path.home() / ".kagglerun"
    hist_dir.mkdir(parents=True, exist_ok=True)
    return hist_dir / "runs.json"


def load_all_records(repo_root: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Read all recorded runs from history file."""
    hist_file = get_history_file(repo_root)
    if not hist_file.exists():
        return []

    try:
        with open(hist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as e:
        logger.warning("Could not read history file %s: %s", hist_file, e)
    return []


def save_all_records(
    records: List[Dict[str, Any]], repo_root: Optional[Path] = None
) -> None:
    """Write all records to history file."""
    hist_file = get_history_file(repo_root)
    try:
        with open(hist_file, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
    except Exception as e:
        logger.warning("Could not write history file %s: %s", hist_file, e)


def record_run(record: RunRecord, repo_root: Optional[Path] = None) -> None:
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
    updates: Dict[str, Any],
    repo_root: Optional[Path] = None,
) -> Optional[RunRecord]:
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


def get_run(
    kernel_id_or_slug: str, repo_root: Optional[Path] = None
) -> Optional[RunRecord]:
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


def list_runs(limit: int = 20, repo_root: Optional[Path] = None) -> List[RunRecord]:
    """List recent run records ordered by submission time."""
    records = load_all_records(repo_root)
    results: List[RunRecord] = []
    for r in records[:limit]:
        try:
            results.append(RunRecord(**r))
        except Exception:
            continue
    return results
