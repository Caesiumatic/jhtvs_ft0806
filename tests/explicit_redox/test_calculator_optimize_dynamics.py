from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from jhtvs_ft0806.explicit_redox.calculator import (
    PolarMACEStateCalculator,
    apply_state_metadata,
    ensure_torch_compiler_compat,
    model_parameter_sha256,
)
from jhtvs_ft0806.explicit_redox.dynamics import chunk_plan, pending_chunks


class FakeAtoms:
    def __init__(self) -> None:
        self.info: dict[str, object] = {}
        self.pbc = True
        self.positions = np.zeros((2, 3))

    def copy(self):
        result = FakeAtoms()
        result.positions = self.positions.copy()
        return result

    def __len__(self) -> int:
        return len(self.positions)


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def numpy(self):
        return self.values


def test_charge_spin_and_external_field_reach_calculator_metadata() -> None:
    atoms = FakeAtoms()
    apply_state_metadata(atoms, charge=-1, spin=2)
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 2
    np.testing.assert_array_equal(atoms.info["external_field"], np.zeros(3))
    assert atoms.pbc is False


def test_torch_compiler_compat_uses_dynamo_probe_without_overwriting_native_api() -> None:
    probe = lambda: False
    old_torch = SimpleNamespace(_dynamo=SimpleNamespace(is_compiling=probe))
    assert ensure_torch_compiler_compat(old_torch) is True
    assert old_torch.compiler.is_compiling is probe

    native_probe = lambda: True
    new_torch = SimpleNamespace(
        _dynamo=SimpleNamespace(is_compiling=probe),
        compiler=SimpleNamespace(is_compiling=native_probe),
    )
    assert ensure_torch_compiler_compat(new_torch) is False
    assert new_torch.compiler.is_compiling is native_probe


def test_charge_and_spin_reach_polar_backend_api() -> None:
    calls = []

    class Batch:
        def to_dict(self):
            return {}

    class Model:
        def __call__(self, *_args, **_kwargs):
            return {"energy": FakeTensor([1.25]), "forces": FakeTensor(np.zeros((2, 3)))}

    class Backend:
        model = Model()

        def build_graph_from_atoms(self, *, atoms, formal_charge, multiplicity):
            calls.append((atoms.info["charge"], atoms.info["spin"], formal_charge, multiplicity))
            return Batch()

    calculator = PolarMACEStateCalculator.__new__(PolarMACEStateCalculator)
    calculator.charge = -1
    calculator.spin = 2
    calculator.backend = Backend()
    calculator.results = {}
    calculator.raw_diagnostics = {}
    calculator.atoms = None
    calculator._last_geometry_key = None
    atoms = FakeAtoms()
    calculator.calculate(atoms)
    assert calls == [(-1, 2, -1, 2)]
    assert calculator.results["energy"] == 1.25


def test_model_parameter_hash_detects_changes() -> None:
    class Model:
        values = np.asarray([1.0, 2.0])

        def state_dict(self):
            return {"weight": FakeTensor(self.values)}

    model = Model()
    before = model_parameter_sha256(model)
    assert model_parameter_sha256(model) == before
    model.values[0] = 3.0
    assert model_parameter_sha256(model) != before


def test_md_chunk_plan_and_restart_idempotency(tmp_path: Path) -> None:
    plan = chunk_plan("val-test-lower-seed1")
    assert len(plan) == 200
    assert sum(chunk.phase == "equilibration" for chunk in plan) == 50
    assert sum(chunk.phase == "production" for chunk in plan) == 150
    assert len({chunk.random_seed for chunk in plan}) == 200
    (tmp_path / "chunk-0000.json").write_text(
        json.dumps({"status": "complete", "chunk_index": 0}), encoding="utf-8"
    )
    assert [chunk.index for chunk in pending_chunks(plan, tmp_path)][:2] == [1, 2]
    assert len(pending_chunks(plan, tmp_path)) == 199


def test_optimizer_smoke_with_harmonic_calculator(tmp_path: Path) -> None:
    ase = pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    from jhtvs_ft0806.explicit_redox.optimize import optimize_state
    from jhtvs_ft0806.explicit_redox.restraint import FlatBottomShell

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.positions
            self.results = {"energy": float(0.5 * (positions**2).sum()), "forces": -positions}

    atoms = Atoms("C6", positions=np.asarray([[0.2, 0, 0], [0, 0.2, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]))
    restraint = FlatBottomShell(
        target_heavy_indices=[0], solvent_groups=[[1], [2], [3], [4], [5]], masses=atoms.get_masses(), R0_A=5.0
    )
    result = optimize_state(
        atoms=atoms,
        model_calculator=Harmonic(),
        restraint=restraint,
        charge=0,
        spin=1,
        output_dir=tmp_path,
        fmax_eV_A=0.05,
        max_steps=100,
    )
    assert result["converged"] is True
    assert result["observed_max_force_eV_A"] <= 0.05


def test_md_smoke_writes_exact_sample_count_and_restart_receipt(tmp_path: Path) -> None:
    pytest.importorskip("ase")
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    from jhtvs_ft0806.explicit_redox.dynamics import run_md
    from jhtvs_ft0806.explicit_redox.restraint import FlatBottomShell

    class Harmonic(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = atoms.positions
            self.results = {"energy": float(0.01 * (positions**2).sum()), "forces": -0.02 * positions}

    atoms = Atoms(
        "C6",
        positions=np.asarray(
            [[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]],
            dtype=float,
        ),
    )
    restraint = FlatBottomShell(
        target_heavy_indices=[0],
        solvent_groups=[[1], [2], [3], [4], [5]],
        masses=atoms.get_masses(),
        R0_A=5.0,
    )
    result = run_md(
        logical_id="smoke",
        atoms=atoms,
        model_calculator=Harmonic(),
        restraint=restraint,
        charge=0,
        spin=1,
        velocity_seed=12345,
        output_dir=tmp_path,
        equilibration_ps=0.001,
        production_ps=0.001,
        checkpoint_ps=0.001,
        sample_interval_fs=0.5,
    )
    assert result["status"] == "complete"
    assert result["completed_production_samples"] == 2
    assert result["temperature_mean_K"] > 0.0
    restarted = run_md(
        logical_id="smoke",
        atoms=atoms,
        model_calculator=Harmonic(),
        restraint=restraint,
        charge=0,
        spin=1,
        velocity_seed=12345,
        output_dir=tmp_path,
        equilibration_ps=0.001,
        production_ps=0.001,
        checkpoint_ps=0.001,
        sample_interval_fs=0.5,
    )
    assert restarted["new_production_samples"] == 0
