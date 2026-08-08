"""Frozen-head warm-up and online PolarMACE LoRA ensemble training."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from jhtvs_ft0806.ml.dataset import (
    DatasetBundle,
    ReactionExample,
    build_reaction_dataset,
)
from jhtvs_ft0806.ml.features import EXPECTED_CHECKPOINT_NAME, PolarMACEBackend
from jhtvs_ft0806.ml.model import (
    SolventConditionedReactionModel,
    architecture_metadata,
    collate_reactions,
    loss_denominators,
    reaction_loss,
)
from jhtvs_ft0806.provenance import content_hash, sha256_file
from jhtvs_ft0806.schemas import read_csv_rows


DEFAULT_MAX_LORA_EPOCHS = 300
DEFAULT_ONLINE_BATCH_SIZE = 4


class TrainingError(RuntimeError):
    """Raised when a frozen training contract or artifact is violated."""


try:
    import torch
except ImportError:  # pragma: no cover - minimal ORCA environments are supported
    torch = None


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    head_warmup_epochs: int
    head_lr: float
    lora_rank: int
    lora_alpha: float
    lora_lr: float
    patience: int
    seeds: tuple[int, ...]
    max_lora_epochs: int = DEFAULT_MAX_LORA_EPOCHS
    online_batch_size: int = DEFAULT_ONLINE_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class MemberTrainingResult:
    seed: int
    warmup_final_loss: float
    best_val_mae_eV: float
    best_lora_epoch: int
    lora_epochs_run: int
    checkpoint_path: str
    checkpoint_sha256: str
    adapter_parameter_count: int


@dataclass(frozen=True, slots=True)
class EnsembleTrainingSummary:
    model_run_id: str
    dataset_sha256: str
    member_count: int
    members: tuple[MemberTrainingResult, ...]
    manifest_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS" if self.member_count == 5 else "INCOMPLETE",
            "model_run_id": self.model_run_id,
            "dataset_sha256": self.dataset_sha256,
            "member_count": self.member_count,
            "members": [asdict(member) for member in self.members],
            "manifest_path": self.manifest_path,
        }


def _require_torch() -> Any:
    if torch is None:
        raise TrainingError("install the project ML dependencies before training")
    return torch


def load_training_settings(
    spec_dir: Path,
    *,
    max_lora_epochs: int = DEFAULT_MAX_LORA_EPOCHS,
    online_batch_size: int = DEFAULT_ONLINE_BATCH_SIZE,
) -> TrainingSettings:
    rows = read_csv_rows(spec_dir / "training_config.csv")
    values = {(row["category"], row["key"]): row["value"] for row in rows}

    def value(category: str, key: str) -> str:
        try:
            return values[(category, key)]
        except KeyError as exc:
            raise TrainingError(f"training config missing {category}/{key}") from exc

    early_stopping = value("training", "early_stopping")
    if "patience 30" not in early_stopping:
        raise TrainingError("early-stopping patience drift")
    settings = TrainingSettings(
        head_warmup_epochs=int(value("training", "head_warmup_epochs")),
        head_lr=float(value("training", "head_lr")),
        lora_rank=int(value("training", "lora_rank")),
        lora_alpha=float(value("training", "lora_alpha")),
        lora_lr=float(value("training", "lora_lr")),
        patience=30,
        seeds=tuple(int(item) for item in value("training", "seeds").split(";")),
        max_lora_epochs=int(max_lora_epochs),
        online_batch_size=int(online_batch_size),
    )
    if settings.seeds != (17, 29, 43, 71, 101):
        raise TrainingError("ensemble seed drift")
    if settings.max_lora_epochs <= settings.patience:
        raise TrainingError("LoRA epoch cap must exceed early-stopping patience")
    if settings.online_batch_size <= 0:
        raise TrainingError("online batch size must be positive")
    return settings


def seed_everything(seed: int) -> None:
    torch_module = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    torch_module.use_deterministic_algorithms(True, warn_only=True)


def _chunks(values: Sequence[ReactionExample], size: int) -> Iterable[tuple[ReactionExample, ...]]:
    for start in range(0, len(values), size):
        yield tuple(values[start : start + size])


def warmup_frozen_head(
    *,
    bundle: DatasetBundle,
    spec_dir: Path,
    settings: TrainingSettings,
    seed: int,
    device: str,
) -> tuple[Any, tuple[float, ...]]:
    """Train only the solvent/reaction network from content-addressed frozen features."""

    torch_module = _require_torch()
    seed_everything(seed)
    train_examples = tuple(example for example in bundle.examples if example.split == "train")
    if not train_examples:
        raise TrainingError("training split is empty")
    batch = collate_reactions(
        train_examples, bundle=bundle, spec_dir=spec_dir, device=device
    )
    denominators = loss_denominators(train_examples)
    model = SolventConditionedReactionModel(
        state_feature_dim=bundle.feature_dimension
    ).to(device)
    optimizer = torch_module.optim.AdamW(model.parameters(), lr=settings.head_lr)
    history: list[float] = []
    model.train()
    for _ in range(settings.head_warmup_epochs):
        optimizer.zero_grad(set_to_none=True)
        losses = reaction_loss(
            model(batch),
            batch,
            target_normalization=bundle.target_normalization,
            denominators=denominators,
        )
        losses["total"].backward()
        optimizer.step()
        observed = float(losses["total"].detach().cpu())
        if not math.isfinite(observed):
            raise TrainingError("non-finite frozen-head loss")
        history.append(observed)
    return model, tuple(history)


class OnlineFeatureProvider:
    """Rebuild state graphs and run every LoRA-stage feature forward online."""

    def __init__(
        self,
        *,
        repository_root: Path,
        spec_dir: Path,
        geometry_index_path: Path,
        feature_index_path: Path,
        bundle: DatasetBundle,
        device: str,
        rank: int,
        alpha: float,
        backend_factory: Callable[..., Any] = PolarMACEBackend,
    ) -> None:
        self.repository_root = repository_root.resolve()
        self.spec_dir = spec_dir
        self.bundle = bundle
        self.device = device
        rows = read_csv_rows(geometry_index_path)
        self.geometry_by_key: dict[tuple[str, str], Mapping[str, str]] = {}
        for row in rows:
            key = (row["state_id"], row["solvent_id"])
            if key in self.geometry_by_key:
                raise TrainingError(f"duplicate geometry state-medium key: {key}")
            self.geometry_by_key[key] = row
        feature_rows = read_csv_rows(feature_index_path)
        checkpoint_hashes = {row["checkpoint_sha256"] for row in feature_rows}
        if len(checkpoint_hashes) != 1:
            raise TrainingError("feature index checkpoint hash drift")
        self.backend = backend_factory(
            checkpoint=EXPECTED_CHECKPOINT_NAME, device=device
        )
        if self.backend.provenance.checkpoint_sha256 not in checkpoint_hashes:
            raise TrainingError("online checkpoint differs from frozen feature checkpoint")
        self.backend.enable_lora(rank=rank, alpha=alpha)
        trainable = [
            name for name, parameter in self.backend.model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable or any(
            "lora_A" not in name and "lora_B" not in name for name in trainable
        ):
            raise TrainingError("official LoRA injection left unexpected trainable parameters")
        self._graph_cache: dict[tuple[str, str, str], Any] = {}

    @property
    def adapter_parameters(self) -> tuple[Any, ...]:
        return tuple(
            parameter for parameter in self.backend.model.parameters()
            if parameter.requires_grad
        )

    @property
    def adapter_parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.adapter_parameters)

    def adapter_state_dict(self) -> dict[str, Any]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.backend.model.named_parameters()
            if parameter.requires_grad
        }

    def load_adapter_state_dict(self, state: Mapping[str, Any]) -> None:
        parameters = dict(self.backend.model.named_parameters())
        if set(state) != {
            name for name, parameter in parameters.items() if parameter.requires_grad
        }:
            raise TrainingError("adapter checkpoint keys differ from injected model")
        with _require_torch().no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(device=self.device, dtype=parameters[name].dtype))

    def _geometry_path(self, raw: str) -> Path:
        path = Path(raw)
        return path.resolve() if path.is_absolute() else (self.repository_root / path).resolve()

    def _graph(self, state: Any) -> Any:
        key = (state.state_id, state.solvent_id)
        row = self.geometry_by_key.get(key)
        if row is None or row["status"] != "resolved":
            raise TrainingError(f"online geometry unavailable: {key}")
        if row["xyz_sha256"] != state.geometry_hash:
            raise TrainingError(f"online geometry hash differs from feature cache: {key}")
        path = self._geometry_path(row["xyz_path"])
        if not path.is_file() or sha256_file(path) != state.geometry_hash:
            raise TrainingError(f"online geometry content hash mismatch: {path}")
        graph_key = (state.state_id, state.solvent_id, state.geometry_hash)
        graph = self._graph_cache.get(graph_key)
        if graph is None:
            graph = self.backend.build_graph(
                xyz_path=path,
                formal_charge=int(row["formal_charge"]),
                multiplicity=int(row["multiplicity"]),
            )
            self._graph_cache[graph_key] = graph
        return graph

    def make_batch(
        self, examples: Sequence[ReactionExample], *, training: bool
    ) -> Any:
        torch_module = _require_torch()
        batch = collate_reactions(
            examples,
            bundle=self.bundle,
            spec_dir=self.spec_dir,
            device=self.device,
        )
        mean = torch_module.as_tensor(
            self.bundle.feature_mean, dtype=torch_module.float64, device=self.device
        )
        std = torch_module.as_tensor(
            self.bundle.feature_std, dtype=torch_module.float64, device=self.device
        )
        online: list[Any] = []
        for example in examples:
            for state in example.states:
                row = self.geometry_by_key[(state.state_id, state.solvent_id)]
                vector = self.backend.extract_tensor(
                    batch=self._graph(state),
                    formal_charge=int(row["formal_charge"]),
                    multiplicity=int(row["multiplicity"]),
                    training=training,
                    immutable_base_energy_eV=state.base_energy_eV,
                )
                if int(vector.numel()) != self.bundle.feature_dimension:
                    raise TrainingError("online invariant feature dimension drift")
                online.append((vector - mean) / std)
        return replace(batch, state_features=torch_module.stack(online))

    def train(self) -> None:
        self.backend.model.train()

    def eval(self) -> None:
        self.backend.model.eval()


def grouped_validation_mae(
    *,
    model: Any,
    examples: Sequence[ReactionExample],
    batch_builder: Callable[[Sequence[ReactionExample]], Any],
    batch_size: int,
) -> float:
    """Average available final-property MAE within parent, then across parents."""

    torch_module = _require_torch()
    eligible = tuple(example for example in examples if example.final_mask)
    if not eligible:
        raise TrainingError("validation split has no clean final labels")
    errors: dict[str, list[float]] = {}
    with torch_module.no_grad():
        for group in _chunks(eligible, batch_size):
            batch = batch_builder(group)
            prediction = model(batch)["final_residual_eV"]
            observed = torch_module.abs(prediction - batch.final_target_eV).detach().cpu()
            for example, error in zip(group, observed.tolist(), strict=True):
                errors.setdefault(example.parent_id, []).append(float(error))
    parent_mae = [sum(values) / len(values) for values in errors.values()]
    value = float(sum(parent_mae) / len(parent_mae))
    if not math.isfinite(value):
        raise TrainingError("non-finite grouped validation MAE")
    return value


def fine_tune_lora(
    *,
    model: Any,
    provider: OnlineFeatureProvider,
    bundle: DatasetBundle,
    settings: TrainingSettings,
    seed: int,
    device: str,
) -> tuple[tuple[dict[str, float | int], ...], float, int]:
    """Fine-tune online adapters and heads without ever reading test rows."""

    torch_module = _require_torch()
    train_examples = tuple(example for example in bundle.examples if example.split == "train")
    val_examples = tuple(example for example in bundle.examples if example.split == "val")
    if not train_examples or not val_examples:
        raise TrainingError("train/validation split is empty")
    denominators = loss_denominators(train_examples)
    optimizer = torch_module.optim.AdamW(
        (
            {"params": tuple(model.parameters()), "lr": settings.head_lr},
            {"params": provider.adapter_parameters, "lr": settings.lora_lr},
        )
    )
    rng = random.Random(seed)
    best_mae = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_model_state: Mapping[str, Any] | None = None
    best_adapter_state: Mapping[str, Any] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, settings.max_lora_epochs + 1):
        order = list(train_examples)
        rng.shuffle(order)
        model.train()
        provider.train()
        total_loss = 0.0
        for group in _chunks(order, settings.online_batch_size):
            optimizer.zero_grad(set_to_none=True)
            batch = provider.make_batch(group, training=True)
            losses = reaction_loss(
                model(batch),
                batch,
                target_normalization=bundle.target_normalization,
                denominators=denominators,
            )
            losses["total"].backward()
            optimizer.step()
            total_loss += float(losses["total"].detach().cpu())
        model.eval()
        provider.eval()
        val_mae = grouped_validation_mae(
            model=model,
            examples=val_examples,
            batch_builder=lambda rows: provider.make_batch(rows, training=False),
            batch_size=settings.online_batch_size,
        )
        history.append(
            {"epoch": epoch, "train_loss": total_loss, "val_mae_eV": val_mae}
        )
        if val_mae < best_mae - 1e-12:
            best_mae = val_mae
            best_epoch = epoch
            stale_epochs = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_adapter_state = provider.adapter_state_dict()
        else:
            stale_epochs += 1
        if stale_epochs >= settings.patience:
            break
    if best_model_state is None or best_adapter_state is None:
        raise TrainingError("LoRA early stopping did not capture a finite checkpoint")
    model.load_state_dict(best_model_state)
    provider.load_adapter_state_dict(best_adapter_state)
    return tuple(history), best_mae, best_epoch


def _normalization_payload(bundle: DatasetBundle) -> dict[str, Any]:
    return {
        "feature_mean": bundle.feature_mean.tolist(),
        "feature_std": bundle.feature_std.tolist(),
        "solvent_mean": bundle.solvent_mean.tolist(),
        "solvent_std": bundle.solvent_std.tolist(),
        "baseline": bundle.baseline_normalization.to_dict(),
        "targets": {
            name: normalization.to_dict()
            for name, normalization in bundle.target_normalization.items()
        },
    }


def _portable_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def train_ensemble(
    *,
    repository_root: Path,
    spec_dir: Path,
    geometry_index_path: Path,
    reaction_sp_path: Path,
    reaction_final_path: Path,
    feature_index_path: Path,
    artifact_dir: Path,
    manifest_path: Path,
    device: str = "cpu",
    max_lora_epochs: int = DEFAULT_MAX_LORA_EPOCHS,
    online_batch_size: int = DEFAULT_ONLINE_BATCH_SIZE,
    backend_factory: Callable[..., Any] = PolarMACEBackend,
) -> EnsembleTrainingSummary:
    """Train the frozen five-seed ensemble and persist adapter-only checkpoints."""

    torch_module = _require_torch()
    settings = load_training_settings(
        spec_dir,
        max_lora_epochs=max_lora_epochs,
        online_batch_size=online_batch_size,
    )
    bundle = build_reaction_dataset(
        repository_root=repository_root,
        spec_dir=spec_dir,
        reaction_sp_path=reaction_sp_path,
        reaction_final_path=reaction_final_path,
        feature_index_path=feature_index_path,
    )
    model_run_id = content_hash(
        {
            "dataset_sha256": bundle.dataset_sha256,
            "settings": asdict(settings),
            "architecture": architecture_metadata(),
        }
    )
    run_dir = artifact_dir / model_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    members: list[MemberTrainingResult] = []
    for seed in settings.seeds:
        model, warmup_history = warmup_frozen_head(
            bundle=bundle,
            spec_dir=spec_dir,
            settings=settings,
            seed=seed,
            device=device,
        )
        provider = OnlineFeatureProvider(
            repository_root=repository_root,
            spec_dir=spec_dir,
            geometry_index_path=geometry_index_path,
            feature_index_path=feature_index_path,
            bundle=bundle,
            device=device,
            rank=settings.lora_rank,
            alpha=settings.lora_alpha,
            backend_factory=backend_factory,
        )
        lora_history, best_mae, best_epoch = fine_tune_lora(
            model=model,
            provider=provider,
            bundle=bundle,
            settings=settings,
            seed=seed,
            device=device,
        )
        checkpoint = run_dir / f"member_seed_{seed}.pt"
        torch_module.save(
            {
                "model_run_id": model_run_id,
                "seed": seed,
                "dataset_sha256": bundle.dataset_sha256,
                "settings": asdict(settings),
                "architecture": architecture_metadata(),
                "normalization": _normalization_payload(bundle),
                "property_model_state": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "adapter_state": provider.adapter_state_dict(),
                "foundation_provenance": provider.backend.provenance.to_dict(),
                "warmup_history": warmup_history,
                "lora_history": lora_history,
                "best_lora_epoch": best_epoch,
                "best_val_mae_eV": best_mae,
                "test_split_used_for_training_or_selection": False,
            },
            checkpoint,
        )
        members.append(
            MemberTrainingResult(
                seed=seed,
                warmup_final_loss=warmup_history[-1],
                best_val_mae_eV=best_mae,
                best_lora_epoch=best_epoch,
                lora_epochs_run=len(lora_history),
                checkpoint_path=_portable_path(checkpoint, repository_root),
                checkpoint_sha256=sha256_file(checkpoint),
                adapter_parameter_count=provider.adapter_parameter_count,
            )
        )
    payload = {
        "status": "PASS",
        "model_run_id": model_run_id,
        "dataset_sha256": bundle.dataset_sha256,
        "settings": asdict(settings),
        "architecture": architecture_metadata(),
        "normalization_sha256": content_hash(_normalization_payload(bundle)),
        "members": [asdict(member) for member in members],
        "test_split_used_for_training_or_selection": False,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return EnsembleTrainingSummary(
        model_run_id=model_run_id,
        dataset_sha256=bundle.dataset_sha256,
        member_count=len(members),
        members=tuple(members),
        manifest_path=str(manifest_path.resolve()),
    )
