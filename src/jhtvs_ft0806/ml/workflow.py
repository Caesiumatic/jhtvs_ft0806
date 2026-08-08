"""Content-addressed frozen MACE-POLAR feature workflow."""

from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
from typing import Callable
import zipfile

import numpy as np

from jhtvs_ft0806.ml.features import (
    EXPECTED_CHECKPOINT_NAME,
    FEATURE_SCHEMA_REVISION,
    PolarMACEBackend,
    feature_cache_key,
    serialize_output_shapes,
)
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows, write_csv_deterministic
from jhtvs_ft0806.spec_validation import validate_spec


FEATURE_INDEX_FIELDS = (
    "state_id",
    "solvent_id",
    "geometry_hash",
    "formal_charge",
    "multiplicity",
    "feature_cache_key",
    "feature_path",
    "feature_sha256",
    "feature_dim",
    "E_base_MACE_eV",
    "checkpoint_name",
    "checkpoint_sha256",
    "mace_version",
    "mace_source_commit",
    "mace_package_sha256",
    "graph_electrostatics_version",
    "graph_electrostatics_commit",
    "graph_electrostatics_package_sha256",
    "default_dtype",
    "feature_schema_revision",
    "invariant_layout_sha256",
    "raw_output_shapes",
)
BASELINE_FIELDS = (
    "state_id",
    "solvent_id",
    "geometry_hash",
    "E_base_MACE_eV",
    "feature_cache_key",
    "checkpoint_sha256",
    "feature_schema_revision",
)


class FeatureWorkflowError(RuntimeError):
    """Raised for invalid geometry or cache provenance."""


@dataclass(frozen=True, slots=True)
class FeatureExtractionSummary:
    total: int
    extracted: int
    reused: int
    missing: int
    feature_dimension: int
    checkpoint_sha256: str
    feature_schema_revision: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "status": "PASS" if self.missing == 0 else "INCOMPLETE",
            "total": self.total,
            "extracted": self.extracted,
            "reused": self.reused,
            "missing": self.missing,
            "feature_dimension": self.feature_dimension,
            "checkpoint_sha256": self.checkpoint_sha256,
            "feature_schema_revision": self.feature_schema_revision,
        }


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_geometry_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _write_feature_npz(
    path: Path,
    *,
    feature_vector: np.ndarray,
    invariant_layout: tuple[str, ...],
    base_energy_eV: float,
    raw_output_shapes: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    arrays = {
        "base_energy_eV": np.asarray(base_energy_eV, dtype=np.float64),
        "feature_vector": np.asarray(feature_vector, dtype=np.float64),
        "invariant_layout": np.asarray(invariant_layout, dtype=np.str_),
        "raw_output_shapes": np.asarray(raw_output_shapes, dtype=np.str_),
    }
    with zipfile.ZipFile(
        temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for name, array in sorted(arrays.items()):
            payload = io.BytesIO()
            np.lib.format.write_array(payload, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    temporary.replace(path)


def _read_cached(path: Path) -> tuple[np.ndarray, tuple[str, ...], float, str]:
    with np.load(path, allow_pickle=False) as archive:
        vector = np.asarray(archive["feature_vector"], dtype=np.float64)
        layout = tuple(str(value) for value in archive["invariant_layout"].tolist())
        energy = float(np.asarray(archive["base_energy_eV"]).reshape(()))
        raw_output_shapes = str(np.asarray(archive["raw_output_shapes"]).reshape(()))
    if vector.ndim != 1 or vector.size != len(layout) or not np.all(np.isfinite(vector)):
        raise FeatureWorkflowError(f"invalid frozen feature cache: {path}")
    return vector, layout, energy, raw_output_shapes


def extract_base_features(
    *,
    repository_root: Path,
    spec_dir: Path,
    geometry_index_path: Path,
    cache_dir: Path,
    feature_index_path: Path,
    baseline_output_path: Path,
    checkpoint: str = EXPECTED_CHECKPOINT_NAME,
    device: str = "cpu",
    selected_state_ids: set[str] | None = None,
    backend_factory: Callable[..., PolarMACEBackend] = PolarMACEBackend,
) -> FeatureExtractionSummary:
    validation = validate_spec(spec_dir)
    if not validation.ok:
        raise FeatureWorkflowError("scientific specification validation failed")
    rows = read_csv_rows(geometry_index_path)
    selected = [
        row
        for row in rows
        if row["status"] == "resolved"
        and (selected_state_ids is None or row["state_id"] in selected_state_ids)
    ]
    backend = backend_factory(checkpoint=checkpoint, device=device)
    provenance = backend.provenance
    index_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    extracted = 0
    reused = 0
    missing = 0
    feature_dimension = 0
    expected_layout: tuple[str, ...] | None = None

    for row in selected:
        geometry_path = _resolve_geometry_path(row["xyz_path"], repository_root)
        geometry_hash = row["xyz_sha256"]
        if not geometry_path.is_file() or sha256_file(geometry_path) != geometry_hash:
            missing += 1
            continue
        key = feature_cache_key(
            checkpoint_sha256=provenance.checkpoint_sha256,
            geometry_sha256=geometry_hash,
            formal_charge=int(row["formal_charge"]),
            multiplicity=int(row["multiplicity"]),
        )
        feature_path = cache_dir / key[:2] / f"{key}.npz"
        if feature_path.is_file():
            vector, layout, energy, shapes = _read_cached(feature_path)
            reused += 1
        else:
            record = backend.extract(
                xyz_path=geometry_path,
                formal_charge=int(row["formal_charge"]),
                multiplicity=int(row["multiplicity"]),
            )
            vector = record.feature_vector
            layout = record.invariant_layout
            energy = record.base_energy_eV
            shapes = serialize_output_shapes(record.output_shapes)
            _write_feature_npz(
                feature_path,
                feature_vector=vector,
                invariant_layout=layout,
                base_energy_eV=energy,
                raw_output_shapes=shapes,
            )
            extracted += 1
        if expected_layout is None:
            expected_layout = layout
            feature_dimension = int(vector.size)
        elif layout != expected_layout or vector.size != feature_dimension:
            raise FeatureWorkflowError("feature layout drift across state geometries")
        layout_hash = content_hash(list(layout))
        common = {
            "state_id": row["state_id"],
            "solvent_id": row["solvent_id"],
            "geometry_hash": geometry_hash,
            "E_base_MACE_eV": format(energy, ".15g"),
            "feature_cache_key": key,
            "checkpoint_sha256": provenance.checkpoint_sha256,
            "feature_schema_revision": FEATURE_SCHEMA_REVISION,
        }
        baseline_rows.append(common)
        index_rows.append(
            {
                **common,
                "formal_charge": row["formal_charge"],
                "multiplicity": row["multiplicity"],
                "feature_path": _relative_or_absolute(feature_path, repository_root),
                "feature_sha256": sha256_file(feature_path),
                "feature_dim": int(vector.size),
                "checkpoint_name": provenance.checkpoint_name,
                "mace_version": provenance.mace_version,
                "mace_source_commit": provenance.mace_source_commit,
                "mace_package_sha256": provenance.mace_package_sha256,
                "graph_electrostatics_version": provenance.graph_electrostatics_version,
                "graph_electrostatics_commit": provenance.graph_electrostatics_commit,
                "graph_electrostatics_package_sha256": provenance.graph_electrostatics_package_sha256,
                "default_dtype": provenance.default_dtype,
                "invariant_layout_sha256": layout_hash,
                "raw_output_shapes": shapes,
            }
        )

    write_csv_deterministic(
        feature_index_path,
        FEATURE_INDEX_FIELDS,
        index_rows,
        sort_by=("state_id", "solvent_id", "geometry_hash"),
    )
    write_csv_deterministic(
        baseline_output_path,
        BASELINE_FIELDS,
        baseline_rows,
        sort_by=("state_id", "solvent_id", "geometry_hash"),
    )
    return FeatureExtractionSummary(
        total=len(selected),
        extracted=extracted,
        reused=reused,
        missing=missing,
        feature_dimension=feature_dimension,
        checkpoint_sha256=provenance.checkpoint_sha256,
        feature_schema_revision=FEATURE_SCHEMA_REVISION,
    )
