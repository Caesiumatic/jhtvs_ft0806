from __future__ import annotations

import numpy as np
import pytest

from jhtvs_ft0806.explicit_redox.alignment import assert_disjoint_keys, fit_global_intercept
from jhtvs_ft0806.explicit_redox.marcus import (
    assemble_seed,
    assemble_system,
    block_statistics,
    ev_per_electron_to_volts,
)
from jhtvs_ft0806.explicit_redox.restraint import FlatBottomShell
from jhtvs_ft0806.explicit_redox.vertical_gap import (
    evaluate_gap_batch,
    evaluate_gap_batch_chunked,
    write_gap_chunk,
)


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Atoms:
    def __init__(self, shift=0.0):
        self.positions = np.zeros((7, 3))
        self.positions[2:, 0] = np.arange(5) + shift
        self.info = {}

    def copy(self):
        result = _Atoms()
        result.positions = self.positions.copy()
        result.info = self.info.copy()
        return result

    def get_chemical_symbols(self):
        return ["C", "C", "O", "O", "O", "O", "O"]


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return False


class _Backend:
    class Torch:
        @staticmethod
        def no_grad():
            return _NoGrad()

    _torch = Torch()

    def build_graph_from_atoms(self, *, atoms, formal_charge, multiplicity):
        return (atoms, formal_charge, multiplicity)

    def batch_graphs(self, graphs):
        class Batch:
            def __init__(self, items):
                self.items = items

            def to_dict(self):
                return {"items": self.items}

        return Batch(graphs)

    def model(self, payload, **_kwargs):
        energies = [10.0 + charge + float(atoms.positions.sum()) * 0.01 for atoms, charge, _ in payload["items"]]
        atom_rows = sum(len(atoms.positions) for atoms, _, _ in payload["items"])
        return {
            "energy": _Tensor(energies),
            "density_coefficients": _Tensor(np.ones((atom_rows, 4))),
            "spin_density": _Tensor(np.ones((atom_rows, 4))),
            "spin_charge_density": _Tensor(np.ones((atom_rows, 2, 4))),
        }


def test_vertical_gap_sign_and_same_coordinate_definition() -> None:
    lower = np.asarray([-10.0, -9.5])
    oxidized = np.asarray([-4.0, -4.25])
    np.testing.assert_allclose(oxidized - lower, [6.0, 5.25])


def test_batched_two_state_gap_and_raw_result_reproducibility(tmp_path) -> None:
    restraint = FlatBottomShell(
        target_heavy_indices=[0, 1],
        solvent_groups=[[2], [3], [4], [5], [6]],
        masses=[12, 12, 16, 16, 16, 16, 16],
        R0_A=2.0,
    )
    batch = evaluate_gap_batch(
        backend=_Backend(),
        atoms_batch=[_Atoms(0.0), _Atoms(0.2)],
        lower_charge=-1,
        lower_spin=1,
        oxidized_charge=0,
        oxidized_spin=2,
        restraint=restraint,
    )
    np.testing.assert_allclose(batch.delta_E_eV, [1.0, 1.0])
    assert batch.lower_diagnostics["density_coefficients"].shape == (2, 7, 4)
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    assert write_gap_chunk(first, batch) == write_gap_chunk(second, batch)


def test_gap_microbatching_preserves_order_and_diagnostics() -> None:
    restraint = FlatBottomShell(
        target_heavy_indices=[0, 1],
        solvent_groups=[[2], [3], [4], [5], [6]],
        masses=[12, 12, 16, 16, 16, 16, 16],
        R0_A=2.0,
    )
    frames = [_Atoms(float(index) / 10.0) for index in range(7)]
    direct = evaluate_gap_batch(
        backend=_Backend(),
        atoms_batch=frames,
        lower_charge=-1,
        lower_spin=1,
        oxidized_charge=0,
        oxidized_spin=2,
        restraint=restraint,
    )
    chunked = evaluate_gap_batch_chunked(
        backend=_Backend(),
        atoms_batch=frames,
        lower_charge=-1,
        lower_spin=1,
        oxidized_charge=0,
        oxidized_spin=2,
        restraint=restraint,
        batch_size=3,
    )
    assert chunked.coordinate_sha256 == direct.coordinate_sha256
    np.testing.assert_allclose(chunked.delta_E_eV, direct.delta_E_eV)
    np.testing.assert_allclose(
        chunked.oxidized_diagnostics["spin_density"],
        direct.oxidized_diagnostics["spin_density"],
    )


def test_marcus_formula_on_synthetic_harmonic_data() -> None:
    lower = np.full(7500, 6.0)
    oxidized = np.full(7500, 4.0)
    seed = assemble_seed(lower, oxidized)
    assert seed.mu_lower_eV == 6.0
    assert seed.mu_oxidized_eV == 4.0
    assert seed.delta_F_ox_eV == 5.0
    assert seed.lambda_eV == 1.0
    system = assemble_system([seed] * 5)
    assert system.delta_F_ox_eV == 5.0
    assert system.raw_voltage_V == 5.0


def test_ev_to_v_one_electron() -> None:
    assert ev_per_electron_to_volts(3.25, electrons=1) == 3.25


def test_contiguous_block_statistics_do_not_count_frames_as_independent() -> None:
    block_values = np.repeat(np.arange(30, dtype=float), 250)
    stats = block_statistics(block_values)
    expected = np.arange(30, dtype=float).std(ddof=1) / np.sqrt(30)
    assert stats.block_count == 30
    assert stats.block_standard_error_eV == pytest.approx(expected)


def test_calibration_validation_disjointness() -> None:
    assert_disjoint_keys(["a", "b"], ["c"])
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_keys(["a", "b"], ["b", "c"])


def test_intercept_only_alignment_slope_is_fixed_to_one() -> None:
    raw = np.asarray([1.0, 2.0, 3.0])
    experimental = raw - 4.5 + np.asarray([-0.1, 0.0, 0.1])
    fit = fit_global_intercept(experimental, raw)
    assert fit.C_model_V == pytest.approx(-4.5)
    np.testing.assert_allclose(raw + fit.C_model_V, [-3.5, -2.5, -1.5])
    assert fit.mae_after_V < fit.mae_before_V
