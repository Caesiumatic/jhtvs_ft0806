from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jhtvs_ft0806.ml.dataset import (  # noqa: E402
    DatasetBundle,
    Normalization,
    ReactionExample,
    StateFeature,
)
from jhtvs_ft0806.ml.model import (  # noqa: E402
    SolventConditionedReactionModel,
    collate_reactions,
    loss_denominators,
    reaction_loss,
)


def _state(state_id: str, coefficient: int, vector: list[float]) -> StateFeature:
    return StateFeature(
        state_id=state_id,
        solvent_id="S001",
        coefficient=coefficient,
        geometry_hash="a" * 64,
        feature_cache_key="b" * 64,
        feature_path="feature.npz",
        feature_sha256="c" * 64,
        base_energy_eV=-1.0,
        vector=np.asarray(vector, dtype=np.float64),
    )


def _examples() -> tuple[ReactionExample, ...]:
    return (
        ReactionExample(
            reaction_id="RXN_RED",
            reaction_class="redox",
            role="monomer",
            parent_id="P1",
            solvent_id="S001",
            split="train",
            states=(
                _state("R0", -1, [1.0, 2.0, 3.0]),
                _state("R1", 1, [2.0, 3.0, 4.0]),
            ),
            deltaE_base_MACE_rxn_eV=2.0,
            sp_residual_eV=1.0,
            rt_correction_eV=0.5,
            final_residual_eV=1.5,
            sp_mask=True,
            rt_mask=True,
            final_mask=True,
            row_weight=1.0,
            qc_status_sp="clean",
            qc_status_final="clean",
        ),
        ReactionExample(
            reaction_id="RXN_SIG",
            reaction_class="sigma_dimerization",
            role="sigma",
            parent_id="P2",
            solvent_id="S001",
            split="train",
            states=(
                _state("MPLUS", -2, [3.0, 4.0, 5.0]),
                _state("DPLUS2", 1, [4.0, 5.0, 6.0]),
            ),
            deltaE_base_MACE_rxn_eV=-1.0,
            sp_residual_eV=2.0,
            rt_correction_eV=-0.25,
            final_residual_eV=1.75,
            sp_mask=True,
            rt_mask=True,
            final_mask=True,
            row_weight=1.0,
            qc_status_sp="clean",
            qc_status_final="clean",
        ),
    )


def _bundle(examples: tuple[ReactionExample, ...]) -> DatasetBundle:
    norms = {
        name: Normalization(mean=0.0, std=1.0, count=2)
        for name in ("redox_final", "sigma_final", "sp", "rt")
    }
    return DatasetBundle(
        examples=examples,
        feature_dimension=3,
        feature_mean=np.zeros(3),
        feature_std=np.ones(3),
        solvent_mean=np.zeros(8),
        solvent_std=np.ones(8),
        baseline_normalization=Normalization(mean=0.5, std=1.5, count=2),
        target_normalization=norms,
        dataset_sha256="d" * 64,
    )


def test_frozen_head_forward_and_backward() -> None:
    examples = _examples()
    bundle = _bundle(examples)
    batch = collate_reactions(
        examples,
        bundle=bundle,
        spec_dir=Path(__file__).resolve().parents[1] / "spec",
        device="cpu",
    )
    assert batch.baseline_eV.tolist() == pytest.approx([2.0, -1.0])
    assert batch.baseline_feature.tolist() == pytest.approx([1.0, -1.0])
    model = SolventConditionedReactionModel(state_feature_dim=3)
    outputs = model(batch)
    assert set(outputs) == {
        "final_residual_eV",
        "sp_residual_eV",
        "rt_correction_eV",
    }
    assert all(value.shape == (2,) for value in outputs.values())
    losses = reaction_loss(
        outputs, batch, target_normalization=bundle.target_normalization
    )
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_official_lora_gradients_and_immutable_baseline() -> None:
    pytest.importorskip("mace")
    from mace.modules.lora import inject_lora

    torch.manual_seed(7)
    baseline = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.SiLU())
    adapted = copy.deepcopy(baseline)
    inputs = torch.randn(3, 4)
    immutable_output = baseline(inputs).detach().clone()
    inject_lora(adapted, rank=4, alpha=1.0)
    assert torch.allclose(adapted(inputs), immutable_output, atol=1e-12, rtol=0.0)
    loss = adapted(inputs).square().sum()
    loss.backward()
    adapter_parameters = [
        parameter
        for name, parameter in adapted.named_parameters()
        if "lora_A" in name or "lora_B" in name
    ]
    assert adapter_parameters
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for parameter in adapter_parameters
    )
    assert torch.allclose(baseline(inputs), immutable_output, atol=0.0, rtol=0.0)


def test_polar_lora_bias_instruction_inference_compatibility() -> None:
    pytest.importorskip("mace")
    from e3nn import o3
    from mace.modules.lora import inject_lora

    from jhtvs_ft0806.ml.features import patch_incompatible_mace_lora_inference

    torch.manual_seed(11)
    baseline = o3.Linear("2x0e", "2x0e", biases=True).double()
    adapted = copy.deepcopy(baseline)
    inputs = torch.randn(5, 2, dtype=torch.float64)
    immutable_output = baseline(inputs).detach().clone()
    inject_lora(adapted, rank=4, alpha=1.0)
    assert patch_incompatible_mace_lora_inference(adapted) == 1
    with torch.no_grad():
        observed = adapted(inputs)
    assert torch.allclose(observed, immutable_output, atol=1e-12, rtol=0.0)
    adapted(inputs).square().sum().backward()
    assert any(
        parameter.grad is not None and torch.any(parameter.grad != 0)
        for name, parameter in adapted.named_parameters()
        if "lora_A" in name or "lora_B" in name
    )


def test_fixed_denominators_make_minibatch_loss_additive() -> None:
    examples = _examples()
    bundle = _bundle(examples)
    model = SolventConditionedReactionModel(state_feature_dim=3)
    denominators = loss_denominators(examples)
    full_batch = collate_reactions(
        examples,
        bundle=bundle,
        spec_dir=Path(__file__).resolve().parents[1] / "spec",
        device="cpu",
    )
    full = reaction_loss(
        model(full_batch),
        full_batch,
        target_normalization=bundle.target_normalization,
        denominators=denominators,
    )
    partials = []
    for example in examples:
        batch = collate_reactions(
            (example,),
            bundle=bundle,
            spec_dir=Path(__file__).resolve().parents[1] / "spec",
            device="cpu",
        )
        partials.append(
            reaction_loss(
                model(batch),
                batch,
                target_normalization=bundle.target_normalization,
                denominators=denominators,
            )
        )
    for name in ("total", "redox_final", "sigma_final", "sp", "rt", "consistency"):
        observed = sum(part[name] for part in partials)
        assert torch.allclose(observed, full[name], atol=1e-12, rtol=0.0)
