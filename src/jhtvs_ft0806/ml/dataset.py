"""Leakage-safe reaction-level datasets for the frozen surrogate targets."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from jhtvs_ft0806.labels.assembly import parse_stoichiometry
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows


ROLE_ORDER = ("monomer", "solvent", "anion")
REACTION_CLASS_ORDER = ("redox", "sigma_dimerization")
TARGET_NAMES = ("redox_final", "sigma_final", "sp", "rt")


class DatasetError(ValueError):
    """Raised when labels, features, or frozen splits disagree."""


@dataclass(frozen=True, slots=True)
class StateFeature:
    state_id: str
    solvent_id: str
    coefficient: int
    geometry_hash: str
    feature_cache_key: str
    feature_path: str
    feature_sha256: str
    base_energy_eV: float
    vector: np.ndarray


@dataclass(frozen=True, slots=True)
class ReactionExample:
    reaction_id: str
    reaction_class: str
    role: str
    parent_id: str
    solvent_id: str
    split: str
    states: tuple[StateFeature, ...]
    deltaE_base_MACE_rxn_eV: float
    sp_residual_eV: float | None
    rt_correction_eV: float | None
    final_residual_eV: float | None
    sp_mask: bool
    rt_mask: bool
    final_mask: bool
    row_weight: float
    qc_status_sp: str
    qc_status_final: str


@dataclass(frozen=True, slots=True)
class Normalization:
    mean: float
    std: float
    count: int

    def to_dict(self) -> dict[str, float | int]:
        return {"mean": self.mean, "std": self.std, "count": self.count}


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    examples: tuple[ReactionExample, ...]
    feature_dimension: int
    feature_mean: np.ndarray
    feature_std: np.ndarray
    solvent_mean: np.ndarray
    solvent_std: np.ndarray
    baseline_normalization: Normalization
    target_normalization: Mapping[str, Normalization]
    dataset_sha256: str


def _float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _resolve(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def _load_feature(
    path: Path, expected_sha256: str, expected_base_energy_eV: float
) -> tuple[np.ndarray, float]:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise DatasetError(f"feature cache hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as archive:
        vector = np.asarray(archive["feature_vector"], dtype=np.float64)
        base_energy = float(np.asarray(archive["base_energy_eV"]).reshape(()))
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise DatasetError(f"invalid feature vector: {path}")
    if not math.isclose(
        base_energy, expected_base_energy_eV, rel_tol=0.0, abs_tol=5e-12
    ):
        raise DatasetError(f"feature baseline energy drift: {path}")
    return vector, base_energy


def load_state_feature_index(
    *, repository_root: Path, feature_index_path: Path
) -> tuple[dict[tuple[str, str], StateFeature], int]:
    """Load and hash-verify one content-addressed state-feature index."""

    feature_rows = read_csv_rows(feature_index_path)
    features: dict[tuple[str, str], StateFeature] = {}
    feature_dimension = 0
    for row in feature_rows:
        key = (row["state_id"], row["solvent_id"])
        if key in features:
            raise DatasetError(f"duplicate feature state-medium key: {key}")
        expected_base_energy = _float(row["E_base_MACE_eV"])
        if expected_base_energy is None:
            raise DatasetError(f"{key}: feature index baseline energy missing")
        vector, base_energy = _load_feature(
            _resolve(row["feature_path"], repository_root),
            row["feature_sha256"],
            expected_base_energy,
        )
        if feature_dimension == 0:
            feature_dimension = int(vector.size)
        elif vector.size != feature_dimension:
            raise DatasetError("feature dimension drift")
        features[key] = StateFeature(
            state_id=row["state_id"],
            solvent_id=row["solvent_id"],
            coefficient=0,
            geometry_hash=row["geometry_hash"],
            feature_cache_key=row["feature_cache_key"],
            feature_path=row["feature_path"],
            feature_sha256=row["feature_sha256"],
            base_energy_eV=base_energy,
            vector=vector,
        )
    return features, feature_dimension


def solvent_vectors(spec_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    vectors: dict[str, np.ndarray] = {}
    hashes: dict[str, str] = {}
    for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv"):
        parts = row["resolved_model_vector"].split("|")
        if len(parts) != 8:
            raise DatasetError(f"{row['solvent_id']}: resolved model vector is not 8-D")
        vector = np.asarray([float(value) for value in parts], dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            raise DatasetError(f"{row['solvent_id']}: non-finite model vector")
        expected_log = math.log(float(row["epsilon"]))
        if not math.isclose(vector[0], expected_log, rel_tol=0.0, abs_tol=5e-11):
            raise DatasetError(f"{row['solvent_id']}: log-epsilon drift")
        vectors[row["solvent_id"]] = vector
        hashes[row["solvent_id"]] = content_hash(vector.tolist())
    if len(vectors) != 25:
        raise DatasetError("solvent registry must contain exactly 25 model vectors")
    return vectors, hashes


def _normalization(values: Iterable[float], *, name: str) -> Normalization:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0:
        raise DatasetError(f"no train values for normalization target {name}")
    mean = float(array.mean())
    std = float(array.std(ddof=0))
    if not math.isfinite(std) or std < 1e-12:
        std = 1.0
    return Normalization(mean=mean, std=std, count=int(array.size))


def _vector_stats(vectors: Iterable[np.ndarray], *, name: str) -> tuple[np.ndarray, np.ndarray]:
    materialized = tuple(np.asarray(vector, dtype=np.float64) for vector in vectors)
    if not materialized:
        raise DatasetError(f"no train vectors for {name} normalization")
    matrix = np.stack(materialized)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0, ddof=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def build_reaction_dataset(
    *,
    repository_root: Path,
    spec_dir: Path,
    reaction_sp_path: Path,
    reaction_final_path: Path,
    feature_index_path: Path,
) -> DatasetBundle:
    """Build reaction examples without allowing val/test data into normalization."""

    vectors, _ = solvent_vectors(spec_dir)
    features, feature_dimension = load_state_feature_index(
        repository_root=repository_root,
        feature_index_path=feature_index_path,
    )

    sp_rows = read_csv_rows(reaction_sp_path)
    sp_keys = {(row["reaction_id"], row["solvent_id"]) for row in sp_rows}
    if len(sp_keys) != len(sp_rows):
        raise DatasetError("duplicate SP reaction-medium key")
    final_rows = read_csv_rows(reaction_final_path)
    final_by_key = {
        (row["reaction_id"], row["solvent_id"]): row
        for row in final_rows
    }
    if len(final_by_key) != len(final_rows):
        raise DatasetError("duplicate final reaction-medium key")
    parent_counts = Counter(row["parent_id"] for row in sp_rows)
    examples: list[ReactionExample] = []
    for sp in sp_rows:
        reaction_class = sp["reaction_class"]
        role = sp["role"]
        if reaction_class not in REACTION_CLASS_ORDER:
            raise DatasetError(f"unknown reaction class: {reaction_class}")
        if reaction_class == "redox" and role not in ROLE_ORDER:
            raise DatasetError(f"unknown redox role: {role}")
        terms: list[StateFeature] = []
        for state_id, coefficient in parse_stoichiometry(sp["stoichiometry"]):
            feature = features.get((state_id, sp["solvent_id"]))
            if feature is None:
                raise DatasetError(
                    f"feature missing for {state_id}/{sp['solvent_id']}"
                )
            terms.append(
                StateFeature(
                    state_id=feature.state_id,
                    solvent_id=feature.solvent_id,
                    coefficient=coefficient,
                    geometry_hash=feature.geometry_hash,
                    feature_cache_key=feature.feature_cache_key,
                    feature_path=feature.feature_path,
                    feature_sha256=feature.feature_sha256,
                    base_energy_eV=feature.base_energy_eV,
                    vector=feature.vector,
                )
            )
        baseline = _float(sp["deltaE_base_MACE_rxn_eV"])
        if baseline is None:
            raise DatasetError(f"{sp['reaction_id']}: immutable MACE baseline missing")
        state_baseline = sum(
            state.coefficient * state.base_energy_eV for state in terms
        )
        if not math.isclose(
            baseline, state_baseline, rel_tol=0.0, abs_tol=5e-10
        ):
            raise DatasetError(
                f"{sp['reaction_id']}/{sp['solvent_id']}: reaction baseline differs "
                "from the immutable state-energy aggregation"
            )
        final = final_by_key.get((sp["reaction_id"], sp["solvent_id"]))
        sp_value = _float(sp["sp_residual_eV"])
        sp_mask = sp["qc_status"] == "clean" and sp_value is not None
        final_value = _float(final.get("final_residual_eV")) if final else None
        rt_value = _float(final.get("rt_correction_eV")) if final else None
        final_status = final.get("qc_status", "missing") if final else "missing"
        final_mask = final_status == "clean" and final_value is not None
        rt_mask = final_status == "clean" and rt_value is not None
        examples.append(
            ReactionExample(
                reaction_id=sp["reaction_id"],
                reaction_class=reaction_class,
                role=role,
                parent_id=sp["parent_id"],
                solvent_id=sp["solvent_id"],
                split=sp["split"],
                states=tuple(terms),
                deltaE_base_MACE_rxn_eV=baseline,
                sp_residual_eV=sp_value,
                rt_correction_eV=rt_value,
                final_residual_eV=final_value,
                sp_mask=sp_mask,
                rt_mask=rt_mask,
                final_mask=final_mask,
                row_weight=1.0 / parent_counts[sp["parent_id"]],
                qc_status_sp=sp["qc_status"],
                qc_status_final=final_status,
            )
        )

    assert_no_parent_leakage(examples)
    assert_parent_weights(examples)
    train = tuple(example for example in examples if example.split == "train")
    train_state_vectors = (
        state.vector for example in train for state in example.states
    )
    feature_mean, feature_std = _vector_stats(
        train_state_vectors, name="state feature"
    )
    solvent_mean, solvent_std = _vector_stats(
        (vectors[example.solvent_id] for example in train), name="solvent"
    )
    target_norm = {
        "redox_final": _normalization(
            (
                float(example.final_residual_eV)
                for example in train
                if example.reaction_class == "redox" and example.final_mask
            ),
            name="redox_final",
        ),
        "sigma_final": _normalization(
            (
                float(example.final_residual_eV)
                for example in train
                if example.reaction_class == "sigma_dimerization"
                and example.final_mask
            ),
            name="sigma_final",
        ),
        "sp": _normalization(
            (
                float(example.sp_residual_eV)
                for example in train
                if example.sp_mask
            ),
            name="sp",
        ),
        "rt": _normalization(
            (
                float(example.rt_correction_eV)
                for example in train
                if example.rt_mask
            ),
            name="rt",
        ),
    }
    baseline_normalization = _normalization(
        (example.deltaE_base_MACE_rxn_eV for example in train),
        name="immutable_baseline",
    )
    manifest = [
        {
            "reaction_id": example.reaction_id,
            "solvent_id": example.solvent_id,
            "split": example.split,
            "state_cache_keys": [state.feature_cache_key for state in example.states],
            "stoichiometric_coefficients": [
                state.coefficient for state in example.states
            ],
            "immutable_baseline_eV": example.deltaE_base_MACE_rxn_eV,
            "sp_residual_eV": example.sp_residual_eV,
            "rt_correction_eV": example.rt_correction_eV,
            "final_residual_eV": example.final_residual_eV,
            "sp_mask": example.sp_mask,
            "rt_mask": example.rt_mask,
            "final_mask": example.final_mask,
            "row_weight": example.row_weight,
            "qc_status_sp": example.qc_status_sp,
            "qc_status_final": example.qc_status_final,
        }
        for example in examples
    ]
    return DatasetBundle(
        examples=tuple(examples),
        feature_dimension=feature_dimension,
        feature_mean=feature_mean,
        feature_std=feature_std,
        solvent_mean=solvent_mean,
        solvent_std=solvent_std,
        baseline_normalization=baseline_normalization,
        target_normalization=target_norm,
        dataset_sha256=content_hash(manifest),
    )


def assert_no_parent_leakage(examples: Iterable[ReactionExample]) -> None:
    split_by_parent: dict[str, str] = {}
    for example in examples:
        prior = split_by_parent.setdefault(example.parent_id, example.split)
        if prior != example.split:
            raise DatasetError(
                f"parent split leakage: {example.parent_id} in {prior} and {example.split}"
            )


def assert_parent_weights(examples: Iterable[ReactionExample], *, atol: float = 1e-12) -> None:
    sums: dict[str, float] = {}
    for example in examples:
        sums[example.parent_id] = sums.get(example.parent_id, 0.0) + example.row_weight
    for parent_id, total in sums.items():
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=atol):
            raise DatasetError(f"parent row weights do not sum to one: {parent_id}={total}")
