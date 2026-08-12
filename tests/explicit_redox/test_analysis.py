from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from jhtvs_ft0806.explicit_redox.analysis import collect_gap_summaries


def _gap_receipt(raw: Path, logical_id: str, values: np.ndarray) -> None:
    directory = raw / "gaps" / logical_id
    directory.mkdir(parents=True)
    path = directory / "gap-0050.npz"
    np.savez_compressed(path, delta_E_eV=values)
    receipt = {
        "status": "complete",
        "gap_chunks": [
            {
                "path": path.relative_to(raw).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        ],
    }
    (directory / "gaps.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_collect_gap_summaries_preserves_ensembles_and_five_seeds(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    fields = [
        "task_index",
        "logical_trajectory_id",
        "system_id",
        "seed_index",
        "state",
    ]
    tasks = []
    for seed in range(5):
        for state in ("lower", "oxidized"):
            logical_id = f"system-one__seed-{seed}__{state}"
            tasks.append(
                {
                    "task_index": len(tasks) + 1,
                    "logical_trajectory_id": logical_id,
                    "system_id": "system-one",
                    "seed_index": seed,
                    "state": state,
                }
            )
            values = np.full(7500, 6.0 if state == "lower" else 4.0)
            _gap_receipt(raw, logical_id, values)
    with (raw / "production_trajectory_tasks.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(tasks)
    seed_rows, system_rows = collect_gap_summaries(raw_root=raw, mode="production")
    assert len(seed_rows) == 5
    assert seed_rows[0]["delta_F_ox_eV"] == 5.0
    assert seed_rows[0]["lambda_eV"] == 1.0
    assert system_rows == [
        {
            "system_id": "system-one",
            "delta_F_ox_eV": 5.0,
            "raw_voltage_V": 5.0,
            "lambda_eV": 1.0,
            "shell_seed_sd_eV": 0.0,
            "shell_seed_sem_eV": 0.0,
            "within_seed_block_se_eV": 0.0,
        }
    ]
