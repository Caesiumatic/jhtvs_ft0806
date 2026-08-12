from __future__ import annotations

import numpy as np

from jhtvs_ft0806.explicit_redox.qc import (
    connectivity_changes,
    localization_diagnostics,
    localization_flag,
)


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
