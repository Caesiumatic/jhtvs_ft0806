from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path
import shutil

from jhtvs_ft0806.geometry.resolution import SIGMA_PREOPT_MANIFEST_FIELDS
from jhtvs_ft0806.hpc.sigma_submission import prepare_sigma_submission
from jhtvs_ft0806.provenance import sha256_file
from jhtvs_ft0806.schemas import write_csv_deterministic


ROOT = Path(__file__).resolve().parents[1]


def test_fullspace_sigma_submission_preserves_2500_exact_tasks(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "spec", repository / "spec")
    run_root = repository / "runs" / "geometry_fullspace"
    source = run_root / "raw_sigma" / "D001_QP2_M1.xyz"
    source.parent.mkdir(parents=True)
    source.write_text("1\nfixture\nH 0 0 0\n", encoding="utf-8")
    source_sha = sha256_file(source)
    manifest_rows = []
    native_rows = []
    for index in range(2500):
        task_id = f"sigma-preopt-{index:04d}"
        output_dir = f"sigma_preopt/D{index:04d}__S001"
        manifest_rows.append(
            {
                "task_id": task_id,
                "geometry_key": f"sigma:{index}",
                "state_id": "D001_QP2_M1",
                "parent_id": "M001",
                "solvent_id": "S001",
                "solvent_name": "fixture",
                "source_xyz": "raw_sigma/D001_QP2_M1.xyz",
                "source_xyz_sha256": source_sha,
                "output_dir": output_dir,
                "formal_charge": "2",
                "multiplicity": "1",
                "uhf": "0",
                "epsilon": "35.688",
                "preopt_method_id": "GFN2-xTB_default_opt_ddCOSMO_v1",
                "monomer_source_sha256": "a" * 64,
                "topology_sha256": "b" * 64,
                "xtb_command": "xtb in.xyz --chrg 2 --uhf 0 --opt --cosmo 35.688",
            }
        )
        native_rows.append(
            "\t".join(
                (
                    task_id,
                    "raw_sigma/D001_QP2_M1.xyz",
                    source_sha,
                    output_dir,
                    "2",
                    "0",
                    "35.688",
                    "b" * 64,
                )
            )
        )
    manifest = run_root / "sigma_preopt_manifest.csv"
    write_csv_deterministic(
        manifest, SIGMA_PREOPT_MANIFEST_FIELDS, manifest_rows, sort_by=("task_id",)
    )
    native_array = run_root / "sigma_preopt_array.tsv"
    native_array.write_text("\n".join(native_rows) + "\n", encoding="utf-8")
    runner = repository / "hpc" / "run_sigma_preopt_budgeted.sh"
    runner.parent.mkdir()
    shutil.copy2(ROOT / "hpc" / "run_sigma_preopt_budgeted.sh", runner)
    preflight = repository / "data" / "resolved" / "preflight.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_text(
        json.dumps(
            {
                "status": "PASS",
                "task_count": 2500,
                "exact_parameter_rows": 2500,
                "exact_command_rows": 2500,
                "source_hash_rows": 2500,
                "array_sha256": sha256_file(native_array),
                "manifest_sha256": sha256_file(manifest),
                "launcher_sha256": sha256_file(runner),
            }
        ),
        encoding="utf-8",
    )
    plan = prepare_sigma_submission(
        submission_id="sigma-fixture",
        spec_dir=repository / "spec",
        run_root=run_root,
        preflight_path=preflight,
        submissions_root=repository / "runs" / "hpc" / "submissions",
        accounting_path=repository / "data" / "resolved" / "accounting.csv",
        ledger_path=repository / "data" / "resolved" / "submission_ledger.csv",
        runner_path=runner,
        planning_core_h=Decimal("192.36"),
    )
    with plan.task_table_path.open(encoding="utf-8", newline="") as handle:
        tasks = list(csv.DictReader(handle, delimiter="\t"))

    assert plan.job_count == plan.array_task_count == 2500
    assert plan.planned_core_h == Decimal("192.36")
    assert len(tasks) == 2500
    assert {task["job_class"] for task in tasks} == {"sigma_preopt"}
    assert {task["nprocs"] for task in tasks} == {"1"}
    assert plan.budget.projected_core_h == Decimal("192.36")
