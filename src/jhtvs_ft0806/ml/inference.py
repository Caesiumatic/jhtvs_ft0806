"""Provenance-complete 5300-row full-space ensemble inference."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jhtvs_ft0806.labels.assembly import (
    PINNED_REFERENCE_CONVERSION_RELATIVE_PATH,
    load_reference_conversion,
    parse_stoichiometry,
)
from jhtvs_ft0806.ml.dataset import (
    DatasetBundle,
    ReactionExample,
    StateFeature,
    build_reaction_dataset,
    load_state_feature_index,
    solvent_vectors,
)
from jhtvs_ft0806.ml.evaluation import (
    _load_member,
    _member_outputs,
    aggregate_member_outputs,
    load_member_references,
)
from jhtvs_ft0806.ml.training import OnlineFeatureProvider, TrainingError
from jhtvs_ft0806.orca.parser import HARTREE_TO_EV, KCAL_PER_HARTREE
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import csv_fieldnames, read_csv_rows, write_csv_deterministic


INFERENCE_REVISION = "jhtvs-ft0806-fullspace-inference-v1"
FULLSPACE_EXPECTED_ROWS = 5300
KCAL_PER_EV = KCAL_PER_HARTREE / HARTREE_TO_EV


class InferenceError(RuntimeError):
    """Raised when full-space identity, features, or frozen policy is incomplete."""


@dataclass(frozen=True, slots=True)
class InferenceCell:
    reaction_id: str
    reaction_class: str
    role: str
    parent_id: str
    solvent_id: str
    states: tuple[StateFeature, ...]
    immutable_baseline_eV: float | None
    geometry_hashes: tuple[str, ...]
    upstream_qc_status: str
    upstream_qc_reasons: tuple[str, ...]

    def reaction_example(self) -> ReactionExample:
        if self.upstream_qc_status != "clean" or self.immutable_baseline_eV is None:
            raise InferenceError(
                f"cannot materialize missing inference cell {self.reaction_id}/{self.solvent_id}"
            )
        return ReactionExample(
            reaction_id=self.reaction_id,
            reaction_class=self.reaction_class,
            role=self.role,
            parent_id=self.parent_id,
            solvent_id=self.solvent_id,
            split="inference",
            states=self.states,
            deltaE_base_MACE_rxn_eV=self.immutable_baseline_eV,
            sp_residual_eV=None,
            rt_correction_eV=None,
            final_residual_eV=None,
            sp_mask=False,
            rt_mask=False,
            final_mask=False,
            row_weight=1.0,
            qc_status_sp="not_computed",
            qc_status_final="not_computed",
        )


@dataclass(frozen=True, slots=True)
class AbstentionPolicy:
    revision: str
    disagreement_threshold_eV: float
    representation_distance_threshold: float
    validation_metrics_sha256: str


@dataclass(frozen=True, slots=True)
class InferenceSummary:
    model_run_id: str
    total_rows: int
    predicted_rows: int
    upstream_missing_rows: int
    abstained_rows: int
    output_path: str
    output_sha256: str
    checkpoint_bundle_sha256: str
    abstention_policy_revision: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS" if self.total_rows == FULLSPACE_EXPECTED_ROWS else "INCOMPLETE",
            "model_run_id": self.model_run_id,
            "total_rows": self.total_rows,
            "predicted_rows": self.predicted_rows,
            "upstream_missing_rows": self.upstream_missing_rows,
            "abstained_rows": self.abstained_rows,
            "output_path": self.output_path,
            "output_sha256": self.output_sha256,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
            "abstention_policy_revision": self.abstention_policy_revision,
        }


def _assigned_media(reaction: Mapping[str, str], solvent_ids: Sequence[str]) -> tuple[str, ...]:
    policy = reaction["solvent_policy"]
    if policy == "all project media at inference":
        return tuple(solvent_ids)
    if policy.startswith("self only (") and policy.endswith(")"):
        return (policy.removeprefix("self only (").removesuffix(")"),)
    raise InferenceError(f"{reaction['reaction_id']}: unsupported solvent policy {policy}")


def build_fullspace_inference_cells(
    *,
    repository_root: Path,
    spec_dir: Path,
    geometry_index_path: Path,
    feature_index_path: Path,
) -> tuple[tuple[InferenceCell, ...], int]:
    features, feature_dimension = load_state_feature_index(
        repository_root=repository_root, feature_index_path=feature_index_path
    )
    geometry_rows = read_csv_rows(geometry_index_path)
    geometry = {
        (row["state_id"], row["solvent_id"]): row
        for row in geometry_rows
    }
    if len(geometry) != len(geometry_rows):
        raise InferenceError("duplicate full-space geometry state-medium key")
    solvent_ids = tuple(
        row["solvent_id"] for row in read_csv_rows(spec_dir / "solvent_smd_registry.csv")
    )
    cells: list[InferenceCell] = []
    for reaction in read_csv_rows(spec_dir / "fullspace_reaction_registry.csv"):
        stoichiometry = parse_stoichiometry(reaction["stoichiometry"])
        for solvent_id in _assigned_media(reaction, solvent_ids):
            states: list[StateFeature] = []
            hashes: list[str] = []
            reasons: list[str] = []
            for state_id, coefficient in stoichiometry:
                geometry_row = geometry.get((state_id, solvent_id))
                if geometry_row is None:
                    reasons.append(f"{state_id}:geometry_index_missing")
                    continue
                if geometry_row["status"] != "resolved":
                    reasons.append(
                        f"{state_id}:geometry_{geometry_row['status']}:{geometry_row['reason']}"
                    )
                    continue
                feature = features.get((state_id, solvent_id))
                if feature is None:
                    reasons.append(f"{state_id}:feature_missing")
                    continue
                if feature.geometry_hash != geometry_row["xyz_sha256"]:
                    reasons.append(f"{state_id}:feature_geometry_hash_mismatch")
                    continue
                hashes.append(feature.geometry_hash)
                states.append(
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
            complete = len(states) == len(stoichiometry) and not reasons
            baseline = (
                sum(state.coefficient * state.base_energy_eV for state in states)
                if complete
                else None
            )
            cells.append(
                InferenceCell(
                    reaction_id=reaction["reaction_id"],
                    reaction_class=reaction["reaction_class"],
                    role=reaction["role"],
                    parent_id=reaction["parent_id"],
                    solvent_id=solvent_id,
                    states=tuple(states),
                    immutable_baseline_eV=baseline,
                    geometry_hashes=tuple(hashes),
                    upstream_qc_status="clean" if complete else "missing",
                    upstream_qc_reasons=tuple(reasons),
                )
            )
    if len(cells) != FULLSPACE_EXPECTED_ROWS:
        raise InferenceError(
            f"full-space inference requires {FULLSPACE_EXPECTED_ROWS} rows, found {len(cells)}"
        )
    keys = {(cell.reaction_id, cell.solvent_id) for cell in cells}
    if len(keys) != len(cells):
        raise InferenceError("duplicate full-space reaction-medium key")
    return tuple(cells), feature_dimension


def reaction_input_descriptor(
    example: ReactionExample,
    *,
    bundle: DatasetBundle,
    medium_vectors: Mapping[str, np.ndarray],
) -> np.ndarray:
    product = np.zeros(bundle.feature_dimension, dtype=np.float64)
    reactant = np.zeros(bundle.feature_dimension, dtype=np.float64)
    for state in example.states:
        vector = (state.vector - bundle.feature_mean) / bundle.feature_std
        if state.coefficient > 0:
            product += state.coefficient * vector
        else:
            reactant += -state.coefficient * vector
    solvent = (
        medium_vectors[example.solvent_id] - bundle.solvent_mean
    ) / bundle.solvent_std
    baseline = np.asarray(
        [
            (
                example.deltaE_base_MACE_rxn_eV
                - bundle.baseline_normalization.mean
            )
            / bundle.baseline_normalization.std
        ]
    )
    return np.concatenate((product, reactant, product - reactant, solvent, baseline))


def representation_distances(
    *,
    train_examples: Sequence[ReactionExample],
    inference_examples: Sequence[ReactionExample],
    bundle: DatasetBundle,
    medium_vectors: Mapping[str, np.ndarray],
) -> dict[tuple[str, str], float]:
    train_by_role: dict[tuple[str, str], list[np.ndarray]] = {}
    for example in train_examples:
        train_by_role.setdefault((example.reaction_class, example.role), []).append(
            reaction_input_descriptor(
                example, bundle=bundle, medium_vectors=medium_vectors
            )
        )
    output: dict[tuple[str, str], float] = {}
    for example in inference_examples:
        references = train_by_role.get((example.reaction_class, example.role))
        if not references:
            raise InferenceError(
                f"no train representation references for {example.reaction_class}/{example.role}"
            )
        descriptor = reaction_input_descriptor(
            example, bundle=bundle, medium_vectors=medium_vectors
        )
        matrix = np.stack(references)
        distances = np.linalg.norm(matrix - descriptor, axis=1) / math.sqrt(
            descriptor.size
        )
        output[(example.reaction_id, example.solvent_id)] = float(distances.min())
    return output


def load_abstention_policy(
    policy_path: Path, *, validation_metrics_path: Path
) -> AbstentionPolicy:
    if not policy_path.is_file():
        raise InferenceError(
            "frozen abstention policy is missing; do not infer one from test/full-space rows"
        )
    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "FROZEN"
        or payload.get("calibration_split") != "val"
        or payload.get("rule") != "any_exceeds_or_upstream_qc"
    ):
        raise InferenceError("abstention policy contract is not frozen on validation")
    metrics_sha = sha256_file(validation_metrics_path)
    if payload.get("validation_metrics_sha256") != metrics_sha:
        raise InferenceError("abstention policy validation-metrics hash mismatch")
    disagreement = float(payload["disagreement_threshold_eV"])
    distance = float(payload["representation_distance_threshold"])
    if not math.isfinite(disagreement) or disagreement <= 0.0:
        raise InferenceError("invalid disagreement threshold")
    if not math.isfinite(distance) or distance <= 0.0:
        raise InferenceError("invalid representation-distance threshold")
    return AbstentionPolicy(
        revision=str(payload["revision"]),
        disagreement_threshold_eV=disagreement,
        representation_distance_threshold=distance,
        validation_metrics_sha256=metrics_sha,
    )


def infer_fullspace(
    *,
    repository_root: Path,
    spec_dir: Path,
    fullspace_geometry_index_path: Path,
    reaction_sp_path: Path,
    reaction_final_path: Path,
    training_feature_index_path: Path,
    fullspace_feature_index_path: Path,
    training_manifest_path: Path,
    validation_metrics_path: Path,
    abstention_policy_path: Path,
    reference_project_path: Path,
    output_path: Path,
    device: str = "cpu",
    batch_size: int = 4,
    provider_factory: Callable[..., OnlineFeatureProvider] = OnlineFeatureProvider,
) -> InferenceSummary:
    if batch_size <= 0:
        raise InferenceError("inference batch size must be positive")
    bundle = build_reaction_dataset(
        repository_root=repository_root,
        spec_dir=spec_dir,
        reaction_sp_path=reaction_sp_path,
        reaction_final_path=reaction_final_path,
        feature_index_path=training_feature_index_path,
    )
    cells, inference_feature_dimension = build_fullspace_inference_cells(
        repository_root=repository_root,
        spec_dir=spec_dir,
        geometry_index_path=fullspace_geometry_index_path,
        feature_index_path=fullspace_feature_index_path,
    )
    if inference_feature_dimension != bundle.feature_dimension:
        raise InferenceError("full-space feature dimension differs from training")
    model_run_id, dataset_sha256, member_references = load_member_references(
        training_manifest_path, repository_root=repository_root
    )
    if dataset_sha256 != bundle.dataset_sha256:
        raise InferenceError("training dataset hash drift before inference")
    policy = load_abstention_policy(
        abstention_policy_path, validation_metrics_path=validation_metrics_path
    )
    available_cells = tuple(cell for cell in cells if cell.upstream_qc_status == "clean")
    examples = tuple(cell.reaction_example() for cell in available_cells)
    by_member: list[dict[tuple[str, str], dict[str, float]]] = []
    for reference in member_references:
        try:
            import torch

            checkpoint = torch.load(
                reference.checkpoint_path, map_location="cpu", weights_only=False
            )
            settings = checkpoint["settings"]
            provider = provider_factory(
                repository_root=repository_root,
                spec_dir=spec_dir,
                geometry_index_path=fullspace_geometry_index_path,
                feature_index_path=fullspace_feature_index_path,
                bundle=bundle,
                device=device,
                rank=int(settings["lora_rank"]),
                alpha=float(settings["lora_alpha"]),
            )
            model = _load_member(
                reference=reference,
                model_run_id=model_run_id,
                bundle=bundle,
                provider=provider,
                device=device,
            )
            by_member.append(
                _member_outputs(
                    model=model,
                    provider=provider,
                    examples=examples,
                    batch_size=batch_size,
                )
            )
        except (ImportError, TrainingError) as exc:
            raise InferenceError(str(exc)) from exc
    media, medium_hashes = solvent_vectors(spec_dir)
    distances = representation_distances(
        train_examples=tuple(
            example for example in bundle.examples if example.split == "train"
        ),
        inference_examples=examples,
        bundle=bundle,
        medium_vectors=media,
    )
    conversion = load_reference_conversion(
        reference_project_path / PINNED_REFERENCE_CONVERSION_RELATIVE_PATH
    )
    checkpoint_bundle_sha256 = content_hash(
        [reference.checkpoint_sha256 for reference in member_references]
    )
    fields = csv_fieldnames(spec_dir / "fullspace_predictions_template.csv")
    rows: list[dict[str, object]] = []
    abstained = 0
    for cell in cells:
        key = (cell.reaction_id, cell.solvent_id)
        common = {
            "prediction_id": f"PRED:{cell.reaction_id}:{cell.solvent_id}",
            "reaction_id": cell.reaction_id,
            "reaction_class": cell.reaction_class,
            "role": cell.role,
            "parent_id": cell.parent_id,
            "solvent_id": cell.solvent_id,
            "internal_unit": "eV",
            "geometry_hashes": ";".join(cell.geometry_hashes),
            "solvent_vector_hash": medium_hashes[cell.solvent_id],
            "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
            "model_run_id": model_run_id,
            "inference_revision": INFERENCE_REVISION,
        }
        if cell.upstream_qc_status != "clean":
            abstained += 1
            rows.append(
                {
                    **common,
                    "prediction_mean_internal": "",
                    "prediction_std_internal": "",
                    "Eox_vs_AgAgCl_mean_V": "",
                    "Eox_vs_AgAgCl_std_V": "",
                    "deltaG_sigma_mean_kcal_mol": "",
                    "deltaG_sigma_std_kcal_mol": "",
                    "ensemble_member_count": 0,
                    "ood_score": "",
                    "abstain": "true",
                    "qc_status": "missing",
                }
            )
            continue
        aggregation = aggregate_member_outputs(
            [member[key] for member in by_member],
            immutable_baseline_eV=float(cell.immutable_baseline_eV),
        )
        mean = float(aggregation["predicted_final_mean_eV"])
        std = float(aggregation["predicted_final_std_eV"])
        distance = distances[key]
        ood_score = max(
            std / policy.disagreement_threshold_eV,
            distance / policy.representation_distance_threshold,
        )
        abstain = ood_score > 1.0
        abstained += int(abstain)
        redox = cell.reaction_class == "redox"
        sigma = cell.reaction_class == "sigma_dimerization"
        rows.append(
            {
                **common,
                "prediction_mean_internal": mean,
                "prediction_std_internal": std,
                "Eox_vs_AgAgCl_mean_V": conversion(mean) if redox else "",
                "Eox_vs_AgAgCl_std_V": std if redox else "",
                "deltaG_sigma_mean_kcal_mol": mean * KCAL_PER_EV if sigma else "",
                "deltaG_sigma_std_kcal_mol": std * KCAL_PER_EV if sigma else "",
                "ensemble_member_count": aggregation["ensemble_member_count"],
                "ood_score": ood_score,
                "abstain": str(abstain).lower(),
                "qc_status": "clean",
            }
        )
    write_csv_deterministic(
        output_path, fields, rows, sort_by=("reaction_id", "solvent_id")
    )
    return InferenceSummary(
        model_run_id=model_run_id,
        total_rows=len(rows),
        predicted_rows=len(available_cells),
        upstream_missing_rows=len(cells) - len(available_cells),
        abstained_rows=abstained,
        output_path=str(output_path.resolve()),
        output_sha256=sha256_file(output_path),
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        abstention_policy_revision=policy.revision,
    )
