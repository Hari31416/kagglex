"""Unit tests for local experiment run history tracking."""

from pathlib import Path

from kagglex.history import (
    RunRecord,
    get_run,
    list_runs,
    record_run,
    update_run,
)


def test_history_crud(tmp_path: Path) -> None:
    """Test recording, querying, listing, and updating run records."""
    rec1 = RunRecord(
        slug="exp-1",
        kernel_id="testuser/exp-1",
        title="Experiment 1",
        command="python train.py",
        accelerator="t4-2x",
        submitted_at="2026-09-16 10:00:00 UTC",
        status="queued",
        project_path=str(tmp_path),
    )

    record_run(rec1, history_path=tmp_path)

    # Query
    found = get_run("exp-1", history_path=tmp_path)
    assert found is not None
    assert found.kernel_id == "testuser/exp-1"
    assert found.status == "queued"
    assert found.project_path == str(tmp_path)

    # Update
    updated = update_run(
        "exp-1",
        {"status": "complete", "duration_sec": 120.5},
        history_path=tmp_path,
    )
    assert updated is not None
    assert updated.status == "complete"
    assert updated.duration_sec == 120.5

    # List
    all_runs = list_runs(limit=10, history_path=tmp_path)
    assert len(all_runs) == 1
    assert all_runs[0].slug == "exp-1"
