from __future__ import annotations

from pathlib import Path

from jhtvs_ft0806.hpc.submission import selected_ids_from_file
from jhtvs_ft0806.schemas import read_csv_rows


ROOT = Path(__file__).resolve().parents[1]


def _xyz_atom_count(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").splitlines()[0])


def test_echo_sp_pilot_uses_one_smallest_ready_job_per_medium() -> None:
    selected = selected_ids_from_file(ROOT / "config" / "echo_sp_pilot_job_ids.txt")
    jobs = {
        row["job_id"]: row
        for row in read_csv_rows(ROOT / "spec" / "sp_job_manifest.csv")
        if row["job_class"] == "smd_energy_sp"
    }
    geometry = {
        row["geometry_key"]: row
        for row in read_csv_rows(ROOT / "data" / "resolved" / "geometry_index.csv")
    }
    selected_rows = [jobs[job_id] for job_id in selected]

    assert len(selected_rows) == 25
    assert {row["solvent_id"] for row in selected_rows} == {
        f"S{index:03d}" for index in range(1, 26)
    }

    for row in selected_rows:
        selected_count = _xyz_atom_count(ROOT / geometry[row["geometry_key"]]["xyz_path"])
        medium_counts = [
            (
                _xyz_atom_count(ROOT / geometry[candidate["geometry_key"]]["xyz_path"]),
                candidate["job_id"],
            )
            for candidate in jobs.values()
            if candidate["solvent_id"] == row["solvent_id"]
        ]
        assert (selected_count, row["job_id"]) == min(medium_counts)
