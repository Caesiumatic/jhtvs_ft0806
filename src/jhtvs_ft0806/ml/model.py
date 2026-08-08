"""Frozen solvent-conditioned reaction model and loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from jhtvs_ft0806.ml.dataset import (
    DatasetBundle,
    REACTION_CLASS_ORDER,
    ROLE_ORDER,
    ReactionExample,
    solvent_vectors,
)


STATE_HIDDEN_DIM = 256
SOLVENT_HIDDEN_DIM = 64
SOLVENT_EMBED_DIM = 128
ROLE_EMBED_DIM = 16
OUTPUT_ORDER = ("final_residual_eV", "sp_residual_eV", "rt_correction_eV")
LOSS_WEIGHTS = {
    "redox_final": 1.0,
    "sigma_final": 1.0,
    "sp": 0.5,
    "rt": 0.25,
    "consistency": 0.25,
}


class ModelError(RuntimeError):
    """Raised when a batch or runtime dependency violates the model contract."""


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by minimal ORCA-only environments
    torch = None
    nn = None


@dataclass(frozen=True, slots=True)
class ReactionBatch:
    state_features: Any
    state_reaction_index: Any
    stoichiometric_coefficients: Any
    solvent_vectors: Any
    role_indices: Any
    sigma_mask: Any
    baseline_eV: Any
    baseline_feature: Any
    final_target_eV: Any
    sp_target_eV: Any
    rt_target_eV: Any
    final_mask: Any
    sp_mask: Any
    rt_mask: Any
    row_weight: Any
    reaction_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    splits: tuple[str, ...]


def _require_torch() -> Any:
    if torch is None:
        raise ModelError("install the project ML dependencies before training")
    return torch


def collate_reactions(
    examples: Iterable[ReactionExample],
    *,
    bundle: DatasetBundle,
    spec_dir: Any,
    device: str,
) -> ReactionBatch:
    """Flatten variable-stoichiometry reactions into one deterministic tensor batch."""

    torch_module = _require_torch()
    selected = tuple(examples)
    if not selected:
        raise ModelError("cannot collate an empty reaction batch")
    medium_vectors, _ = solvent_vectors(spec_dir)
    state_features: list[np.ndarray] = []
    reaction_indices: list[int] = []
    coefficients: list[float] = []
    for reaction_index, example in enumerate(selected):
        for state in example.states:
            state_features.append((state.vector - bundle.feature_mean) / bundle.feature_std)
            reaction_indices.append(reaction_index)
            coefficients.append(float(state.coefficient))
    state_matrix = np.stack(state_features).astype(np.float64)
    solvent_matrix = np.stack(
        [
            (medium_vectors[example.solvent_id] - bundle.solvent_mean)
            / bundle.solvent_std
            for example in selected
        ]
    ).astype(np.float64)

    def tensor(values: Any, *, dtype: Any = None) -> Any:
        return torch_module.as_tensor(
            values,
            dtype=dtype or torch_module.float64,
            device=device,
        )

    final_values = [
        0.0 if example.final_residual_eV is None else example.final_residual_eV
        for example in selected
    ]
    sp_values = [
        0.0 if example.sp_residual_eV is None else example.sp_residual_eV
        for example in selected
    ]
    rt_values = [
        0.0 if example.rt_correction_eV is None else example.rt_correction_eV
        for example in selected
    ]
    return ReactionBatch(
        state_features=tensor(state_matrix),
        state_reaction_index=tensor(reaction_indices, dtype=torch_module.long),
        stoichiometric_coefficients=tensor(coefficients),
        solvent_vectors=tensor(solvent_matrix),
        role_indices=tensor(
            [ROLE_ORDER.index(example.role) if example.role in ROLE_ORDER else 0 for example in selected],
            dtype=torch_module.long,
        ),
        sigma_mask=tensor(
            [example.reaction_class == "sigma_dimerization" for example in selected],
            dtype=torch_module.bool,
        ),
        baseline_eV=tensor([example.deltaE_base_MACE_rxn_eV for example in selected]),
        baseline_feature=tensor(
            [
                (
                    example.deltaE_base_MACE_rxn_eV
                    - bundle.baseline_normalization.mean
                )
                / bundle.baseline_normalization.std
                for example in selected
            ]
        ),
        final_target_eV=tensor(final_values),
        sp_target_eV=tensor(sp_values),
        rt_target_eV=tensor(rt_values),
        final_mask=tensor([example.final_mask for example in selected], dtype=torch_module.bool),
        sp_mask=tensor([example.sp_mask for example in selected], dtype=torch_module.bool),
        rt_mask=tensor([example.rt_mask for example in selected], dtype=torch_module.bool),
        row_weight=tensor([example.row_weight for example in selected]),
        reaction_ids=tuple(example.reaction_id for example in selected),
        parent_ids=tuple(example.parent_id for example in selected),
        splits=tuple(example.split for example in selected),
    )


if nn is not None:

    class SolventConditionedReactionModel(nn.Module):
        """Shared state/medium encoder with role-aware redox and sigma heads."""

        def __init__(self, *, state_feature_dim: int) -> None:
            super().__init__()
            if state_feature_dim <= 0:
                raise ModelError("state feature dimension must be positive")
            self.state_feature_dim = int(state_feature_dim)
            self.state_encoder = nn.Sequential(
                nn.Linear(state_feature_dim, STATE_HIDDEN_DIM, dtype=torch.float64),
                nn.SiLU(),
                nn.LayerNorm(STATE_HIDDEN_DIM, dtype=torch.float64),
                nn.Linear(STATE_HIDDEN_DIM, STATE_HIDDEN_DIM, dtype=torch.float64),
                nn.SiLU(),
                nn.LayerNorm(STATE_HIDDEN_DIM, dtype=torch.float64),
            )
            self.solvent_encoder = nn.Sequential(
                nn.Linear(8, SOLVENT_HIDDEN_DIM, dtype=torch.float64),
                nn.SiLU(),
                nn.LayerNorm(SOLVENT_HIDDEN_DIM, dtype=torch.float64),
                nn.Linear(SOLVENT_HIDDEN_DIM, SOLVENT_EMBED_DIM, dtype=torch.float64),
                nn.SiLU(),
                nn.LayerNorm(SOLVENT_EMBED_DIM, dtype=torch.float64),
            )
            self.film = nn.Linear(
                SOLVENT_EMBED_DIM, 2 * STATE_HIDDEN_DIM, dtype=torch.float64
            )
            self.gate = nn.Linear(
                SOLVENT_EMBED_DIM, STATE_HIDDEN_DIM, dtype=torch.float64
            )
            reaction_dim = 3 * STATE_HIDDEN_DIM + 1
            self.role_embedding = nn.Embedding(
                len(ROLE_ORDER), ROLE_EMBED_DIM, dtype=torch.float64
            )
            self.redox_head = self._head(reaction_dim + ROLE_EMBED_DIM)
            self.sigma_head = self._head(reaction_dim)

        @staticmethod
        def _head(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, STATE_HIDDEN_DIM, dtype=torch.float64),
                nn.SiLU(),
                nn.LayerNorm(STATE_HIDDEN_DIM, dtype=torch.float64),
                nn.Linear(STATE_HIDDEN_DIM, len(OUTPUT_ORDER), dtype=torch.float64),
            )

        def forward(self, batch: ReactionBatch) -> Mapping[str, Any]:
            state = self.state_encoder(batch.state_features)
            medium = self.solvent_encoder(batch.solvent_vectors)
            state_medium = medium[batch.state_reaction_index]
            gamma, beta = self.film(state_medium).chunk(2, dim=-1)
            film = (1.0 + gamma) * state + beta
            gate = torch.sigmoid(self.gate(state_medium))
            fused = gate * film + (1.0 - gate) * state

            n_reactions = int(batch.solvent_vectors.shape[0])
            product = fused.new_zeros((n_reactions, STATE_HIDDEN_DIM))
            reactant = fused.new_zeros((n_reactions, STATE_HIDDEN_DIM))
            coefficients = batch.stoichiometric_coefficients
            positive = torch.clamp(coefficients, min=0.0).unsqueeze(-1)
            negative = torch.clamp(-coefficients, min=0.0).unsqueeze(-1)
            product.index_add_(
                0, batch.state_reaction_index, fused * positive
            )
            reactant.index_add_(
                0, batch.state_reaction_index, fused * negative
            )
            signed = product - reactant
            reaction = torch.cat(
                (product, reactant, signed, batch.baseline_feature.unsqueeze(-1)), dim=-1
            )
            redox = self.redox_head(
                torch.cat((reaction, self.role_embedding(batch.role_indices)), dim=-1)
            )
            sigma = self.sigma_head(reaction)
            output = torch.where(batch.sigma_mask.unsqueeze(-1), sigma, redox)
            return {name: output[:, index] for index, name in enumerate(OUTPUT_ORDER)}

else:  # pragma: no cover - dependency error is covered through _require_torch

    class SolventConditionedReactionModel:
        def __init__(self, **_: Any) -> None:
            _require_torch()


def _weighted_huber(
    prediction: Any,
    target: Any,
    mask: Any,
    weight: Any,
    *,
    scale: Any,
    denominator_override: float | None = None,
) -> Any:
    torch_module = _require_torch()
    selected_weight = weight * mask.to(weight.dtype)
    denominator = (
        selected_weight.sum()
        if denominator_override is None
        else prediction.new_tensor(float(denominator_override))
    )
    if float(denominator.detach().cpu()) == 0.0:
        return prediction.sum() * 0.0
    residual = (prediction - target) / scale
    loss = torch_module.nn.functional.smooth_l1_loss(
        residual, torch_module.zeros_like(residual), reduction="none", beta=1.0
    )
    return (loss * selected_weight).sum() / denominator


def reaction_loss(
    outputs: Mapping[str, Any],
    batch: ReactionBatch,
    *,
    target_normalization: Mapping[str, Any],
    denominators: Mapping[str, float] | None = None,
) -> Mapping[str, Any]:
    """Apply the frozen masked reaction loss in normalized target units."""

    torch_module = _require_torch()
    final_prediction = outputs["final_residual_eV"]
    sp_prediction = outputs["sp_residual_eV"]
    rt_prediction = outputs["rt_correction_eV"]
    redox_mask = ~batch.sigma_mask
    sigma_mask = batch.sigma_mask
    dtype = final_prediction.dtype
    device = final_prediction.device

    def scale(name: str) -> Any:
        return torch_module.as_tensor(
            float(target_normalization[name].std), dtype=dtype, device=device
        )

    def denominator(name: str) -> float | None:
        return None if denominators is None else float(denominators[name])

    redox_final = _weighted_huber(
        final_prediction,
        batch.final_target_eV,
        batch.final_mask & redox_mask,
        batch.row_weight,
        scale=scale("redox_final"),
        denominator_override=denominator("redox_final"),
    )
    sigma_final = _weighted_huber(
        final_prediction,
        batch.final_target_eV,
        batch.final_mask & sigma_mask,
        batch.row_weight,
        scale=scale("sigma_final"),
        denominator_override=denominator("sigma_final"),
    )
    sp = _weighted_huber(
        sp_prediction,
        batch.sp_target_eV,
        batch.sp_mask,
        batch.row_weight,
        scale=scale("sp"),
        denominator_override=denominator("sp"),
    )
    rt = _weighted_huber(
        rt_prediction,
        batch.rt_target_eV,
        batch.rt_mask,
        batch.row_weight,
        scale=scale("rt"),
        denominator_override=denominator("rt"),
    )
    consistency_scale = torch_module.where(
        sigma_mask,
        scale("sigma_final").expand_as(final_prediction),
        scale("redox_final").expand_as(final_prediction),
    )
    consistency = _weighted_huber(
        final_prediction,
        sp_prediction + rt_prediction,
        batch.final_mask & batch.sp_mask & batch.rt_mask,
        batch.row_weight,
        scale=consistency_scale,
        denominator_override=denominator("consistency"),
    )
    total = (
        LOSS_WEIGHTS["redox_final"] * redox_final
        + LOSS_WEIGHTS["sigma_final"] * sigma_final
        + LOSS_WEIGHTS["sp"] * sp
        + LOSS_WEIGHTS["rt"] * rt
        + LOSS_WEIGHTS["consistency"] * consistency
    )
    return {
        "total": total,
        "redox_final": redox_final,
        "sigma_final": sigma_final,
        "sp": sp,
        "rt": rt,
        "consistency": consistency,
    }


def loss_denominators(examples: Iterable[ReactionExample]) -> dict[str, float]:
    """Return fixed split-level denominators so minibatching preserves parent weights."""

    totals = {
        "redox_final": 0.0,
        "sigma_final": 0.0,
        "sp": 0.0,
        "rt": 0.0,
        "consistency": 0.0,
    }
    for example in examples:
        if example.final_mask:
            target = (
                "sigma_final"
                if example.reaction_class == "sigma_dimerization"
                else "redox_final"
            )
            totals[target] += example.row_weight
        if example.sp_mask:
            totals["sp"] += example.row_weight
        if example.rt_mask:
            totals["rt"] += example.row_weight
        if example.final_mask and example.sp_mask and example.rt_mask:
            totals["consistency"] += example.row_weight
    missing = [name for name, value in totals.items() if value <= 0.0]
    if missing:
        raise ModelError(f"loss targets have zero effective weight: {missing}")
    return totals


def architecture_metadata() -> dict[str, Any]:
    return {
        "state_encoder": f"two-layer-{STATE_HIDDEN_DIM}-SiLU-LayerNorm",
        "solvent_encoder": f"8->{SOLVENT_HIDDEN_DIM}->{SOLVENT_EMBED_DIM}-SiLU-LayerNorm",
        "fusion": "FiLM-plus-sigmoid-gating",
        "reaction_aggregation": "product_sum|reactant_sum|signed_difference|immutable_baseline",
        "redox_head": f"shared-plus-{ROLE_EMBED_DIM}D-role-embedding",
        "sigma_head": "separate",
        "outputs": list(OUTPUT_ORDER),
        "loss_weights": dict(LOSS_WEIGHTS),
        "roles": list(ROLE_ORDER),
        "reaction_classes": list(REACTION_CLASS_ORDER),
    }
