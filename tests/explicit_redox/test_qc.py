from __future__ import annotations

import numpy as np
import pytest

from jhtvs_ft0806.explicit_redox.qc import (
    connectivity_changes,
    localization_diagnostics,
    localization_flag,
    solvent_shell_diagnostics,
)


def _shell_distances(solvent_zero: list[float], *, other_distance: float = 8.0) -> np.ndarray:
    values = np.full((len(solvent_zero), 5), other_distance, dtype=float)
    values[:, 0] = solvent_zero
    return values


def test_transient_crossing_of_R0_is_activation_not_escape() -> None:
    diagnostics = solvent_shell_diagnostics(
        [_shell_distances([9.0, 10.01, 9.5])], R0_A=10.0
    )
    assert diagnostics[0]["restraint_activation_fraction"] == 1.0 / 3.0
    assert diagnostics[0]["max_excess_over_R0_A"] == pytest.approx(0.01)
    assert diagnostics[0]["shell_escape"] is False


def test_final_frame_only_crossing_is_not_escape() -> None:
    diagnostics = solvent_shell_diagnostics(
        [_shell_distances([9.0] * 49 + [12.01])], R0_A=10.0
    )
    assert diagnostics[0]["longest_continuous_exceedance_over_R0_plus_2A_ps"] == 0.02
    assert diagnostics[0]["shell_escape"] is False


def test_escape_boundary_crossing_for_less_than_one_ps_is_not_escape() -> None:
    diagnostics = solvent_shell_diagnostics(
        [_shell_distances([12.01] * 49)], R0_A=10.0
    )
    assert diagnostics[0]["longest_continuous_exceedance_over_R0_plus_2A_ps"] == 0.98
    assert diagnostics[0]["shell_escape"] is False


def test_escape_boundary_crossing_for_exactly_one_ps_is_escape() -> None:
    diagnostics = solvent_shell_diagnostics(
        [_shell_distances([12.01] * 50)], R0_A=10.0
    )
    assert diagnostics[0]["longest_continuous_exceedance_over_R0_plus_2A_ps"] == 1.0
    assert diagnostics[0]["shell_escape"] is True


def test_escape_duration_is_continuous_across_restart_boundary() -> None:
    diagnostics = solvent_shell_diagnostics(
        [
            _shell_distances([9.0] * 5 + [12.01] * 25),
            _shell_distances([12.01] * 25 + [9.0] * 5),
        ],
        R0_A=10.0,
    )
    assert diagnostics[0]["longest_continuous_exceedance_over_R0_plus_2A_ps"] == 1.0
    assert diagnostics[0]["shell_escape"] is True


def test_connectivity_qc_is_fragment_scoped() -> None:
    symbols = ["C", "H"] + ["O"] * 5
    initial = np.asarray([[0, 0, 0], [1, 0, 0], [3, 0, 0], [0, 3, 0], [0, 0, 3], [-3, 0, 0], [0, -3, 0]], dtype=float)
    clean = connectivity_changes(
        initial_symbols=symbols,
        initial_positions=initial,
        final_symbols=symbols,
        final_positions=initial + 10.0,
        target_atoms=2,
        solvent_atoms=1,
    )
    assert clean == {
        "target_bonds_broken": 0,
        "target_bonds_formed": 0,
        "solvent_bonds_broken": 0,
        "solvent_bonds_formed": 0,
    }
    broken = initial.copy()
    broken[1, 0] = 5.0
    changed = connectivity_changes(
        initial_symbols=symbols,
        initial_positions=initial,
        final_symbols=symbols,
        final_positions=broken,
        target_atoms=2,
        solvent_atoms=1,
    )
    assert changed["target_bonds_broken"] == 1


def test_localization_uses_density_change_and_spin_regions(tmp_path) -> None:
    path = tmp_path / "gap.npz"
    lower_density = np.zeros((2, 4, 1))
    oxidized_density = np.zeros((2, 4, 1))
    oxidized_density[:, 0, 0] = 2.0
    oxidized_spin = np.zeros((2, 4, 1))
    oxidized_spin[:, 0, 0] = 3.0
    np.savez_compressed(
        path,
        lower_density_coefficients=lower_density,
        oxidized_density_coefficients=oxidized_density,
        oxidized_spin_density=oxidized_spin,
    )
    diagnostics = localization_diagnostics([path], target_atoms=1)
    assert diagnostics == {
        "target_density_change_fraction": 1.0,
        "target_spin_density_fraction": 1.0,
    }
    assert localization_flag(diagnostics, connectivity_clean=True) == "oxidation_localization_target"
    assert localization_flag(diagnostics, connectivity_clean=False) == "localization_ambiguous"
