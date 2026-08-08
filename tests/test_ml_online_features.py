from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jhtvs_ft0806.ml.features import (  # noqa: E402
    build_invariant_feature_record,
    build_torch_invariant_feature_matrix,
    build_torch_invariant_feature_vector,
)


def _arrays() -> dict[str, np.ndarray]:
    return {
        "energy": np.asarray([-12.5]),
        "node_feats": np.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2.0, 1.0, 0.0, 3.0, 4.0, 8.0]]
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


def test_torch_online_features_match_frozen_features_and_keep_gradients() -> None:
    arrays = _arrays()
    expected = build_invariant_feature_record(
        arrays,
        layer_widths=(4, 2),
        even_scalar_indices=((0, 3), (1,)),
        formal_charge=1,
        multiplicity=2,
    ).feature_vector
    outputs = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in arrays.items()
    }
    observed = build_torch_invariant_feature_vector(
        outputs,
        layer_widths=(4, 2),
        even_scalar_indices=((0, 3), (1,)),
        formal_charge=1,
        multiplicity=2,
    )
    assert observed.detach().numpy() == pytest.approx(expected, abs=1e-12, rel=0.0)
    observed.square().sum().backward()
    assert outputs["node_feats"].grad is not None
    assert outputs["density_coefficients"].grad is not None
    assert outputs["energy"].grad is not None


def test_online_feature_can_pin_the_original_checkpoint_energy() -> None:
    arrays = _arrays()
    arrays["energy"] = np.asarray([-9.0])
    outputs = {
        name: torch.tensor(value, dtype=torch.float64, requires_grad=True)
        for name, value in arrays.items()
    }
    observed = build_torch_invariant_feature_vector(
        outputs,
        layer_widths=(4, 2),
        even_scalar_indices=((0, 3), (1,)),
        formal_charge=1,
        multiplicity=2,
        immutable_base_energy_eV=-12.5,
    )
    assert observed[-6].item() == -12.5
    observed.square().sum().backward()
    assert outputs["energy"].grad is None
    assert outputs["node_feats"].grad is not None


def test_batched_online_features_equal_independent_single_graph_features() -> None:
    first = _arrays()
    second = {name: value * 1.3 for name, value in _arrays().items()}
    second["energy"] = np.asarray([-7.25])
    atom_fields = {
        "node_feats",
        "density_coefficients",
        "spin_density",
        "spin_charge_density",
    }
    outputs = {
        name: torch.tensor(
            np.concatenate((first[name], second[name]), axis=0),
            dtype=torch.float64,
            requires_grad=True,
        )
        for name in first
        if name in atom_fields
    }
    for name in set(first) - atom_fields:
        outputs[name] = torch.tensor(
            np.concatenate((first[name], second[name]), axis=0),
            dtype=torch.float64,
            requires_grad=True,
        )
    observed = build_torch_invariant_feature_matrix(
        outputs,
        atom_graph_index=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        graph_count=2,
        layer_widths=(4, 2),
        even_scalar_indices=((0, 3), (1,)),
        formal_charges=(1, -1),
        multiplicities=(2, 1),
        immutable_base_energies_eV=(-12.5, -8.0),
    )
    expected = torch.stack(
        (
            build_torch_invariant_feature_vector(
                {name: torch.tensor(value, dtype=torch.float64) for name, value in first.items()},
                layer_widths=(4, 2),
                even_scalar_indices=((0, 3), (1,)),
                formal_charge=1,
                multiplicity=2,
                immutable_base_energy_eV=-12.5,
            ),
            build_torch_invariant_feature_vector(
                {name: torch.tensor(value, dtype=torch.float64) for name, value in second.items()},
                layer_widths=(4, 2),
                even_scalar_indices=((0, 3), (1,)),
                formal_charge=-1,
                multiplicity=1,
                immutable_base_energy_eV=-8.0,
            ),
        )
    )
    assert observed.detach().numpy() == pytest.approx(
        expected.detach().numpy(), abs=1e-12, rel=0.0
    )
    observed.square().sum().backward()
    assert outputs["node_feats"].grad is not None
    assert outputs["density_coefficients"].grad is not None
    assert outputs["energy"].grad is None
