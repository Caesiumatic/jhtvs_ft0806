from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jhtvs_ft0806.ml.features import (  # noqa: E402
    build_invariant_feature_record,
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
