from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from jhtvs_ft0806.ml.features import (
    CheckpointProvenance,
    EXPECTED_CHECKPOINT_SHA256,
    FeatureRecord,
    assert_rotation_invariant,
    build_invariant_feature_record,
    feature_cache_key,
)
from jhtvs_ft0806.ml.workflow import extract_base_features
from jhtvs_ft0806.provenance import sha256_file


def _outputs() -> dict[str, np.ndarray]:
    return {
        "energy": np.asarray([-12.5]),
        "node_feats": np.asarray(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                [2.0, 1.0, 0.0, 3.0, 4.0, 8.0],
            ]
        ),
        "density_coefficients": np.asarray(
            [[0.4, 1.0, 2.0, 2.0], [-0.4, 2.0, 0.0, 0.0]]
        ),
        "spin_density": np.asarray(
            [[0.2, 0.0, 3.0, 4.0], [-0.2, 1.0, 2.0, 2.0]]
        ),
        "spin_charge_density": np.asarray(
            [
                [[0.3, 1.0, 0.0, 0.0], [0.1, 0.0, 2.0, 0.0]],
                [[-0.1, 0.0, 0.0, 3.0], [-0.3, 2.0, 0.0, 0.0]],
            ]
        ),
        "dipole": np.asarray([[1.0, 2.0, 2.0]]),
        "electrostatic_energy": np.asarray([0.25]),
        "electron_energy": np.asarray([-0.125]),
    }


def _record(outputs: dict[str, np.ndarray] | None = None) -> FeatureRecord:
    return build_invariant_feature_record(
        outputs or _outputs(),
        layer_widths=(4, 2),
        even_scalar_indices=((0, 3), (1,)),
        formal_charge=1,
        multiplicity=2,
    )


def test_feature_cache_key_covers_frozen_identity() -> None:
    assert EXPECTED_CHECKPOINT_SHA256 == (
        "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b"
    )
    key = feature_cache_key(
        checkpoint_sha256="a" * 64,
        geometry_sha256="b" * 64,
        formal_charge=1,
        multiplicity=2,
    )
    assert len(key) == 64
    changed = feature_cache_key(
        checkpoint_sha256="a" * 64,
        geometry_sha256="b" * 64,
        formal_charge=0,
        multiplicity=1,
    )
    assert changed != key


def test_polar_feature_vector_is_rotation_invariant() -> None:
    original = _outputs()
    angle = 0.731
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = {key: value.copy() for key, value in original.items()}
    for name in ("density_coefficients", "spin_density"):
        rotated[name][:, 1:4] = original[name][:, 1:4] @ rotation.T
    for channel in range(2):
        rotated["spin_charge_density"][:, channel, 1:4] = (
            original["spin_charge_density"][:, channel, 1:4] @ rotation.T
        )
    rotated["dipole"] = original["dipole"] @ rotation.T
    first = _record(original)
    second = _record(rotated)
    assert first.invariant_layout == second.invariant_layout
    assert_rotation_invariant(first.feature_vector, second.feature_vector)


def test_feature_record_keeps_required_global_scalars() -> None:
    record = _record()
    by_name = dict(zip(record.invariant_layout, record.feature_vector, strict=True))
    assert record.base_energy_eV == -12.5
    assert by_name["dipole_norm"] == 3.0
    assert by_name["immutable_base_energy_eV"] == -12.5
    assert by_name["atom_count"] == 2.0
    assert by_name["formal_charge"] == 1.0
    assert by_name["multiplicity"] == 2.0


class _FakeBackend:
    calls = 0

    def __init__(self, *, checkpoint: str, device: str) -> None:
        self.provenance = CheckpointProvenance(
            checkpoint_name=checkpoint,
            checkpoint_path="/checkpoint.model",
            checkpoint_sha256="c" * 64,
            mace_version="0.3.16",
            mace_source_commit="4d2da09413ac1407f37cdbb6b81fa28e4c15655e",
            mace_package_sha256="d" * 64,
            graph_electrostatics_version="0.4.0",
            graph_electrostatics_commit="0e21d5546c482d08388a08eb4d948e833227ce47",
            graph_electrostatics_package_sha256="e" * 64,
            default_dtype="float64",
        )

    def extract(
        self, *, xyz_path: Path, formal_charge: int, multiplicity: int
    ) -> FeatureRecord:
        type(self).calls += 1
        return _record()


def test_feature_workflow_writes_and_reuses_content_addressed_cache(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[1]
    geometry = tmp_path / "state.xyz"
    geometry.write_text("2\nstate\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
    index = tmp_path / "geometry_index.csv"
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "state_id",
                "solvent_id",
                "formal_charge",
                "multiplicity",
                "status",
                "xyz_path",
                "xyz_sha256",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "state_id": "M001_QP1_M2",
                "solvent_id": "S001",
                "formal_charge": "1",
                "multiplicity": "2",
                "status": "resolved",
                "xyz_path": str(geometry),
                "xyz_sha256": sha256_file(geometry),
            }
        )
    cache_dir = tmp_path / "cache"
    feature_index = tmp_path / "features.csv"
    baselines = tmp_path / "baselines.csv"
    _FakeBackend.calls = 0
    first = extract_base_features(
        repository_root=repository_root,
        spec_dir=repository_root / "spec",
        geometry_index_path=index,
        cache_dir=cache_dir,
        feature_index_path=feature_index,
        baseline_output_path=baselines,
        backend_factory=_FakeBackend,
    )
    assert first.extracted == 1
    assert first.reused == 0
    first_hash = sha256_file(next(cache_dir.rglob("*.npz")))
    second = extract_base_features(
        repository_root=repository_root,
        spec_dir=repository_root / "spec",
        geometry_index_path=index,
        cache_dir=cache_dir,
        feature_index_path=feature_index,
        baseline_output_path=baselines,
        backend_factory=_FakeBackend,
    )
    assert second.extracted == 0
    assert second.reused == 1
    assert _FakeBackend.calls == 1
    assert sha256_file(next(cache_dir.rglob("*.npz"))) == first_hash
    baseline_rows = list(csv.DictReader(baselines.open()))
    feature_rows = list(csv.DictReader(feature_index.open()))
    assert baseline_rows[0]["E_base_MACE_eV"] == "-12.5"
    assert feature_rows[0]["mace_source_commit"].startswith("4d2da094")
