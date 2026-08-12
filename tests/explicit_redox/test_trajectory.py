from __future__ import annotations

from pathlib import Path

import pytest

from jhtvs_ft0806.explicit_redox.trajectory import prepare_trajectory_tasks


def test_prepare_trajectory_tasks_duplicates_same_cluster_into_two_states(tmp_path: Path) -> None:
    pytest.importorskip("ase")
    raw = tmp_path / "raw"
    cluster = raw / "clusters" / "s" / "seed-0"
    cluster.mkdir(parents=True)
    xyz = cluster / "cluster.xyz"
    xyz.write_text("6\nx\nC 0 0 0\nH 0 0 1\nO 3 0 0\nO 0 3 0\nO 0 0 3\nO -3 0 0\n", encoding="utf-8")
    import hashlib

    manifest = raw / "cluster_manifest.csv"
    manifest.write_text(
        "system_id,seed_index,status,geometry_path,geometry_sha256,target_atoms,solvent_atoms,lower_charge,lower_spin,oxidized_charge,oxidized_spin\n"
        f"s,0,clean,clusters/s/seed-0/cluster.xyz,{hashlib.sha256(xyz.read_bytes()).hexdigest()},1,1,0,1,1,2\n",
        encoding="utf-8",
    )
    tasks = prepare_trajectory_tasks(cluster_manifest=manifest, raw_root=raw, mode="pilot")
    assert len(tasks) == 2
    assert {row["state"] for row in tasks} == {"lower", "oxidized"}
    assert len({row["cluster_geometry_sha256"] for row in tasks}) == 1
    assert len({row["R0_A"] for row in tasks}) == 1
