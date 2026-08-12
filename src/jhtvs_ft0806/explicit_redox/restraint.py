from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RestraintResult:
    energy_eV: float
    forces_eV_A: NDArray[np.float64]
    active_count: int
    maximum_excursion_A: float


class FlatBottomShell:
    def __init__(
        self,
        *,
        target_heavy_indices: Sequence[int],
        solvent_groups: Sequence[Sequence[int]],
        masses: Sequence[float],
        R0_A: float,
        k_eV_A2: float = 0.5,
    ) -> None:
        if R0_A <= 0 or k_eV_A2 <= 0:
            raise ValueError("restraint radius and force constant must be positive")
        self.target = np.asarray(target_heavy_indices, dtype=int)
        self.solvents = tuple(np.asarray(group, dtype=int) for group in solvent_groups)
        self.masses = np.asarray(masses, dtype=float)
        self.R0_A = float(R0_A)
        self.k_eV_A2 = float(k_eV_A2)
        if not len(self.target) or len(self.solvents) != 5 or any(not len(group) for group in self.solvents):
            raise ValueError("restraint requires target heavy atoms and exactly five solvent groups")

    def evaluate(self, positions_A: NDArray[np.float64]) -> RestraintResult:
        positions = np.asarray(positions_A, dtype=float)
        forces = np.zeros_like(positions)
        target_centroid = positions[self.target].mean(axis=0)
        energy = 0.0
        active_count = 0
        maximum_excursion = 0.0
        for group in self.solvents:
            group_masses = self.masses[group]
            com = np.average(positions[group], axis=0, weights=group_masses)
            displacement = com - target_centroid
            distance = float(np.linalg.norm(displacement))
            excursion = max(0.0, distance - self.R0_A)
            maximum_excursion = max(maximum_excursion, excursion)
            if excursion == 0.0:
                continue
            active_count += 1
            energy += 0.5 * self.k_eV_A2 * excursion * excursion
            total_solvent_force = -self.k_eV_A2 * excursion * displacement / distance
            forces[group] += total_solvent_force * (group_masses / group_masses.sum())[:, None]
            forces[self.target] -= total_solvent_force / len(self.target)
        return RestraintResult(
            energy_eV=energy,
            forces_eV_A=forces,
            active_count=active_count,
            maximum_excursion_A=maximum_excursion,
        )
