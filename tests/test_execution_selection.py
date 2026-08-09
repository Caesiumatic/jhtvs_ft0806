from __future__ import annotations

from collections import Counter
from decimal import Decimal
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


def test_echo_sp_retry_is_limited_to_the_two_incomplete_native_smd_jobs() -> None:
    selected = selected_ids_from_file(ROOT / "config" / "echo_sp_retry_job_ids.txt")
    jobs = {
        row["job_id"]: row
        for row in read_csv_rows(ROOT / "spec" / "sp_job_manifest.csv")
    }
    solvents = {
        row["solvent_id"]: row
        for row in read_csv_rows(ROOT / "spec" / "solvent_smd_registry.csv")
    }

    assert selected == {"SP0280", "SP0555"}
    assert {jobs[job_id]["solvent_id"] for job_id in selected} == {"S009", "S019"}
    assert {
        solvents[jobs[job_id]["solvent_id"]]["orca_smd_mode"]
        for job_id in selected
    } == {"native_orca_smd"}


def test_state_class_sp_pilot_covers_small_and_large_examples() -> None:
    selected = selected_ids_from_file(
        ROOT / "config" / "state_class_sp_pilot_job_ids.txt"
    )
    echo = selected_ids_from_file(ROOT / "config" / "echo_sp_pilot_job_ids.txt")
    jobs = {
        row["job_id"]: row
        for row in read_csv_rows(ROOT / "spec" / "sp_job_manifest.csv")
    }
    geometry = {
        row["geometry_key"]: row
        for row in read_csv_rows(ROOT / "data" / "resolved" / "geometry_index.csv")
    }
    categories = {
        ("0", "1"): "neutral",
        ("1", "2"): "radical_cation",
        ("-1", "1"): "anion",
        ("0", "2"): "neutral_radical",
        ("2", "1"): "sigma_dication",
    }
    counts: Counter[str] = Counter()
    sizes: dict[str, list[int]] = {category: [] for category in categories.values()}
    media: set[str] = set()
    for job_id in selected:
        row = jobs[job_id]
        category = categories[(row["formal_charge"], row["multiplicity"])]
        counts[category] += 1
        sizes[category].append(
            _xyz_atom_count(ROOT / geometry[row["geometry_key"]]["xyz_path"])
        )
        media.add(row["solvent_id"])

    assert len(selected) == 10
    assert not selected & echo
    assert counts == Counter({category: 2 for category in categories.values()})
    assert all(min(values) < max(values) for values in sizes.values())
    assert {"S007", "S012"} <= media


def test_optfreq_pilot_is_one_complete_s007_reaction_tuple() -> None:
    selected = selected_ids_from_file(
        ROOT / "config" / "optfreq_reaction_pilot_job_ids.txt"
    )
    jobs = {
        row["job_id"]: row
        for row in read_csv_rows(ROOT / "spec" / "optfreq_job_manifest.csv")
    }
    selected_rows = [jobs[job_id] for job_id in selected]

    assert selected == {"OF040", "OF041"}
    assert {row["reaction_ids"] for row in selected_rows} == {"RXN_AOX_A001"}
    assert {row["state_id"] for row in selected_rows} == {
        "A001_Q0_M2",
        "A001_QM1_M1",
    }
    assert {row["solvent_id"] for row in selected_rows} == {"S007"}
    assert all(
        "complete anion oxidation tuple" in row["purposes"]
        for row in selected_rows
    )
    assert sum(float(row["planning_core_h"]) for row in selected_rows) == 49.64


def test_remaining_production_sp_wave_excludes_all_exact_reuse_jobs() -> None:
    selected = selected_ids_from_file(
        ROOT / "config" / "production_sp_remaining_job_ids.txt"
    )
    echo = selected_ids_from_file(ROOT / "config" / "echo_sp_pilot_job_ids.txt")
    pilot = selected_ids_from_file(
        ROOT / "config" / "state_class_sp_pilot_job_ids.txt"
    )
    rows = read_csv_rows(ROOT / "spec" / "sp_job_manifest.csv")
    all_ids = {row["job_id"] for row in rows}

    assert len(echo) == 25
    assert len(pilot) == 10
    assert not echo & pilot
    assert selected == all_ids - echo - pilot
    assert len(selected) == 700
    assert Counter(
        row["job_class"] for row in rows if row["job_id"] in selected
    ) == Counter({"smd_energy_sp": 670, "diagnostic_gas_sp": 30})
    assert sum(
        (Decimal(row["planning_core_h"]) for row in rows if row["job_id"] in selected),
        Decimal("0"),
    ) == Decimal("1535.44")


def test_remaining_production_optfreq_wave_excludes_pilot_tuple() -> None:
    selected = selected_ids_from_file(
        ROOT / "config" / "production_optfreq_remaining_job_ids.txt"
    )
    pilot = selected_ids_from_file(
        ROOT / "config" / "optfreq_reaction_pilot_job_ids.txt"
    )
    rows = read_csv_rows(ROOT / "spec" / "optfreq_job_manifest.csv")
    all_ids = {row["job_id"] for row in rows}

    assert pilot == {"OF040", "OF041"}
    assert selected == all_ids - pilot
    assert len(selected) == 78
    assert {
        row["method_id"] for row in rows if row["job_id"] in selected
    } == {"T2_wB97X-D3_OptFreq_TZVPD-SP_SMD_v4"}
    assert sum(
        (Decimal(row["planning_core_h"]) for row in rows if row["job_id"] in selected),
        Decimal("0"),
    ) == Decimal("2239.56")
