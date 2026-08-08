from __future__ import annotations

import csv
from decimal import Decimal
import json
from pathlib import Path
import shutil

from jhtvs_ft0806.hpc.model_submission import prepare_model_submission
from jhtvs_ft0806.ml.features import EXPECTED_CHECKPOINT_SHA256
from jhtvs_ft0806.schemas import write_csv_deterministic


ROOT = Path(__file__).resolve().parents[1]


def test_model_submission_pins_five_seed_training_and_all_inputs(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    shutil.copytree(ROOT / "spec", repository / "spec")
    shutil.copytree(ROOT / "config", repository / "config")
    shutil.copy2(ROOT / "AGENTS.md", repository / "AGENTS.md")
    (repository / "hpc").mkdir()
    runner = repository / "hpc" / "run_model_training.sh"
    shutil.copy2(ROOT / "hpc" / "run_model_training.sh", runner)
    resolved = repository / "data" / "resolved"
    resolved.mkdir(parents=True)
    write_csv_deterministic(
        resolved / "geometry_index.csv", ("state_id", "status"), [{"state_id": "S", "status": "resolved"}]
    )
    write_csv_deterministic(
        resolved / "reaction_sp_labels.csv",
        ("reaction_id",),
        [{"reaction_id": f"R{index}"} for index in range(403)],
    )
    write_csv_deterministic(
        resolved / "reaction_final_labels.csv",
        ("reaction_id",),
        [{"reaction_id": f"R{index}"} for index in range(50)],
    )
    write_csv_deterministic(
        resolved / "base_feature_index.csv",
        ("state_id", "checkpoint_sha256"),
        [
            {"state_id": f"S{index}", "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256}
            for index in range(705)
        ],
    )
    (resolved / "base_feature_completion.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "total": 705,
                "missing": 0,
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
            }
        ),
        encoding="utf-8",
    )

    plan = prepare_model_submission(
        submission_id="training-fixture",
        spec_dir=repository / "spec",
        submissions_root=repository / "runs" / "hpc" / "submissions",
        accounting_path=resolved / "accounting.csv",
        ledger_path=resolved / "submission_ledger.csv",
        runner_path=runner,
        planning_core_h=Decimal("160"),
    )
    with plan.task_table_path.open(encoding="utf-8", newline="") as handle:
        task = next(csv.DictReader(handle, delimiter="\t"))
    payload = json.loads((plan.submission_dir / "training_inputs.json").read_text())

    assert plan.nprocs == 8
    assert plan.planned_core_h == Decimal("160")
    assert task["job_id"] == "MACE-ENSEMBLE"
    assert task["nprocs"] == "8"
    assert payload["seeds"] == [17, 29, 43, 71, 101]
    assert payload["head_warmup_epochs"] == 50
    assert payload["max_lora_epochs"] == 300
    assert payload["lora_rank"] == 4
    assert payload["lora_alpha"] == 1.0
    assert len(payload["inputs"]) == 10
