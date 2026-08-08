"""Post-freeze ensemble loading, calibration prediction, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from jhtvs_ft0806.ml.dataset import DatasetBundle, ReactionExample, build_reaction_dataset
from jhtvs_ft0806.ml.metrics import summarize_evaluation
from jhtvs_ft0806.ml.model import SolventConditionedReactionModel
from jhtvs_ft0806.ml.training import OnlineFeatureProvider, TrainingError
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import write_csv_deterministic


EVALUATION_PREDICTION_FIELDS = (
    "prediction_id",
    "reaction_id",
    "reaction_class",
    "role",
    "parent_id",
    "solvent_id",
    "split",
    "immutable_baseline_eV",
    "observed_final_eV",
    "predicted_final_mean_eV",
    "predicted_final_std_eV",
    "observed_sp_eV",
    "predicted_sp_mean_eV",
    "predicted_sp_std_eV",
    "observed_rt_eV",
    "predicted_rt_mean_eV",
    "predicted_rt_std_eV",
    "ensemble_member_count",
    "qc_status_final",
    "qc_status_sp",
    "geometry_hashes",
    "checkpoint_bundle_sha256",
    "model_run_id",
)


class EvaluationError(RuntimeError):
    """Raised when a frozen model bundle cannot be evaluated exactly."""


try:
    import torch
except ImportError:  # pragma: no cover - minimal ORCA environments are supported
    torch = None


@dataclass(frozen=True, slots=True)
class MemberReference:
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    model_run_id: str
    prediction_rows: int
    clean_final_rows: int
    prediction_output: str
    metrics_output: str
    checkpoint_bundle_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "PASS",
            "model_run_id": self.model_run_id,
            "prediction_rows": self.prediction_rows,
            "clean_final_rows": self.clean_final_rows,
            "prediction_output": self.prediction_output,
            "metrics_output": self.metrics_output,
            "checkpoint_bundle_sha256": self.checkpoint_bundle_sha256,
        }


def _require_torch() -> Any:
    if torch is None:
        raise EvaluationError("install the project ML dependencies before evaluation")
    return torch


def _resolve(path_text: str, repository_root: Path) -> Path:
    path = Path(path_text)
    return path.resolve() if path.is_absolute() else (repository_root / path).resolve()


def load_member_references(
    manifest_path: Path, *, repository_root: Path
) -> tuple[str, str, tuple[MemberReference, ...]]:
    if not manifest_path.is_file():
        raise EvaluationError(f"training manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("status") != "PASS":
        raise EvaluationError("training manifest is not complete")
    members = tuple(
        MemberReference(
            seed=int(row["seed"]),
            checkpoint_path=_resolve(row["checkpoint_path"], repository_root),
            checkpoint_sha256=str(row["checkpoint_sha256"]),
        )
        for row in payload.get("members", ())
    )
    if tuple(member.seed for member in members) != (17, 29, 43, 71, 101):
        raise EvaluationError("training manifest does not contain the frozen five seeds")
    for member in members:
        if (
            not member.checkpoint_path.is_file()
            or sha256_file(member.checkpoint_path) != member.checkpoint_sha256
        ):
            raise EvaluationError(f"member checkpoint hash mismatch: {member.checkpoint_path}")
    return str(payload["model_run_id"]), str(payload["dataset_sha256"]), members


def _load_member(
    *,
    reference: MemberReference,
    model_run_id: str,
    bundle: DatasetBundle,
    provider: OnlineFeatureProvider,
    device: str,
) -> Any:
    torch_module = _require_torch()
    checkpoint = torch_module.load(
        reference.checkpoint_path, map_location=device, weights_only=False
    )
    if (
        checkpoint.get("model_run_id") != model_run_id
        or checkpoint.get("dataset_sha256") != bundle.dataset_sha256
        or int(checkpoint.get("seed")) != reference.seed
    ):
        raise EvaluationError(f"member checkpoint identity drift: seed {reference.seed}")
    if checkpoint.get("test_split_used_for_training_or_selection") is not False:
        raise EvaluationError(f"seed {reference.seed}: training checkpoint opened test split")
    model = SolventConditionedReactionModel(
        state_feature_dim=bundle.feature_dimension
    ).to(device)
    model.load_state_dict(checkpoint["property_model_state"])
    provider.load_adapter_state_dict(checkpoint["adapter_state"])
    model.eval()
    provider.eval()
    return model


def _member_outputs(
    *,
    model: Any,
    provider: OnlineFeatureProvider,
    examples: Sequence[ReactionExample],
    batch_size: int,
) -> dict[tuple[str, str], dict[str, float]]:
    torch_module = _require_torch()
    output: dict[tuple[str, str], dict[str, float]] = {}
    with torch_module.no_grad():
        for start in range(0, len(examples), batch_size):
            selected = tuple(examples[start : start + batch_size])
            batch = provider.make_batch(selected, training=False)
            predicted = model(batch)
            arrays = {
                name: values.detach().cpu().numpy()
                for name, values in predicted.items()
            }
            for index, example in enumerate(selected):
                key = (example.reaction_id, example.solvent_id)
                if key in output:
                    raise EvaluationError(f"duplicate evaluation reaction-medium key: {key}")
                output[key] = {
                    name: float(values[index]) for name, values in arrays.items()
                }
    return output


def aggregate_member_outputs(
    member_outputs: Sequence[Mapping[str, float]],
    *,
    immutable_baseline_eV: float,
) -> dict[str, float | int]:
    if len(member_outputs) != 5:
        raise EvaluationError("ensemble aggregation requires exactly five members")

    def stats(name: str, *, add_baseline: bool) -> tuple[float, float]:
        values = np.asarray([row[name] for row in member_outputs], dtype=np.float64)
        if add_baseline:
            values = values + float(immutable_baseline_eV)
        if not np.all(np.isfinite(values)):
            raise EvaluationError(f"non-finite ensemble output: {name}")
        return float(values.mean()), float(values.std(ddof=0))

    final_mean, final_std = stats("final_residual_eV", add_baseline=True)
    sp_mean, sp_std = stats("sp_residual_eV", add_baseline=True)
    rt_mean, rt_std = stats("rt_correction_eV", add_baseline=False)
    return {
        "predicted_final_mean_eV": final_mean,
        "predicted_final_std_eV": final_std,
        "predicted_sp_mean_eV": sp_mean,
        "predicted_sp_std_eV": sp_std,
        "predicted_rt_mean_eV": rt_mean,
        "predicted_rt_std_eV": rt_std,
        "ensemble_member_count": len(member_outputs),
    }


def evaluate_ensemble(
    *,
    repository_root: Path,
    spec_dir: Path,
    geometry_index_path: Path,
    reaction_sp_path: Path,
    reaction_final_path: Path,
    feature_index_path: Path,
    training_manifest_path: Path,
    prediction_output_path: Path,
    metrics_output_path: Path,
    device: str = "cpu",
    batch_size: int = 4,
    provider_factory: Callable[..., OnlineFeatureProvider] = OnlineFeatureProvider,
) -> EvaluationSummary:
    if batch_size <= 0:
        raise EvaluationError("evaluation batch size must be positive")
    bundle = build_reaction_dataset(
        repository_root=repository_root,
        spec_dir=spec_dir,
        reaction_sp_path=reaction_sp_path,
        reaction_final_path=reaction_final_path,
        feature_index_path=feature_index_path,
    )
    model_run_id, dataset_sha256, member_references = load_member_references(
        training_manifest_path, repository_root=repository_root
    )
    if dataset_sha256 != bundle.dataset_sha256:
        raise EvaluationError("current label/feature dataset differs from training manifest")
    examples = tuple(
        example for example in bundle.examples if example.split in {"val", "test"}
    )
    by_member: list[dict[tuple[str, str], dict[str, float]]] = []
    for reference in member_references:
        try:
            checkpoint = _require_torch().load(
                reference.checkpoint_path, map_location="cpu", weights_only=False
            )
            settings = checkpoint["settings"]
            provider = provider_factory(
                repository_root=repository_root,
                spec_dir=spec_dir,
                geometry_index_path=geometry_index_path,
                feature_index_path=feature_index_path,
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
        except TrainingError as exc:
            raise EvaluationError(str(exc)) from exc
    checkpoint_bundle_sha256 = content_hash(
        [reference.checkpoint_sha256 for reference in member_references]
    )
    rows: list[dict[str, object]] = []
    for example in examples:
        key = (example.reaction_id, example.solvent_id)
        aggregation = aggregate_member_outputs(
            [member[key] for member in by_member],
            immutable_baseline_eV=example.deltaE_base_MACE_rxn_eV,
        )
        rows.append(
            {
                "prediction_id": f"EVAL:{example.reaction_id}:{example.solvent_id}",
                "reaction_id": example.reaction_id,
                "reaction_class": example.reaction_class,
                "role": example.role,
                "parent_id": example.parent_id,
                "solvent_id": example.solvent_id,
                "split": example.split,
                "immutable_baseline_eV": example.deltaE_base_MACE_rxn_eV,
                "observed_final_eV": (
                    example.deltaE_base_MACE_rxn_eV + float(example.final_residual_eV)
                    if example.final_mask
                    else ""
                ),
                "observed_sp_eV": (
                    example.deltaE_base_MACE_rxn_eV + float(example.sp_residual_eV)
                    if example.sp_mask
                    else ""
                ),
                "observed_rt_eV": (
                    float(example.rt_correction_eV) if example.rt_mask else ""
                ),
                **aggregation,
                "qc_status_final": example.qc_status_final,
                "qc_status_sp": example.qc_status_sp,
                "geometry_hashes": ";".join(state.geometry_hash for state in example.states),
                "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
                "model_run_id": model_run_id,
            }
        )
    write_csv_deterministic(
        prediction_output_path,
        EVALUATION_PREDICTION_FIELDS,
        rows,
        sort_by=("split", "reaction_id", "solvent_id"),
    )
    metrics = summarize_evaluation(rows)
    metrics.update(
        {
            "model_run_id": model_run_id,
            "dataset_sha256": bundle.dataset_sha256,
            "training_manifest_sha256": sha256_file(training_manifest_path),
            "evaluation_predictions_sha256": sha256_file(prediction_output_path),
            "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
            "test_split_opened_after_model_freeze": True,
            "abstention_threshold_frozen": False,
        }
    )
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EvaluationSummary(
        model_run_id=model_run_id,
        prediction_rows=len(rows),
        clean_final_rows=sum(example.final_mask for example in examples),
        prediction_output=str(prediction_output_path.resolve()),
        metrics_output=str(metrics_output_path.resolve()),
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
    )
