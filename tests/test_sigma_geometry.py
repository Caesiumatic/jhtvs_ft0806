from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from rdkit import Chem

from jhtvs_ft0806.geometry.sigma import (
    SigmaTopology,
    build_sigma_complex,
    load_sigma_topologies,
    validate_sigma_complex,
)
from jhtvs_ft0806.geometry.topology import molecule_from_smiles


SPEC_DIR = Path(__file__).resolve().parents[1] / "spec"
REPRESENTATIVE_PARENTS = ("M001", "M060", "M069", "M084", "M091", "M100")


@pytest.fixture(scope="module")
def topology_by_parent() -> dict[str, SigmaTopology]:
    rows = load_sigma_topologies(SPEC_DIR / "sigma_coupling_topology.csv")
    return {row.parent_id: row for row in rows}


@pytest.mark.parametrize("parent_id", REPRESENTATIVE_PARENTS)
def test_representative_sigma_complexes_pass_construction_qc(
    parent_id: str, topology_by_parent: dict[str, SigmaTopology]
) -> None:
    topology = topology_by_parent[parent_id]
    sigma = build_sigma_complex(topology, n_conformers=4)

    validate_sigma_complex(sigma)
    assert sigma.charge == 2
    assert sigma.multiplicity == 1
    assert sigma.formula == topology.expected_formula
    assert sigma.junction_atom_indices == (
        topology.junction_copy1_atom_index_0based,
        topology.junction_copy2_atom_index_0based,
    )
    assert len(Chem.GetMolFrags(sigma.neutral_dimer)) == 1
    assert all(sigma.symbols[index] == "H" for index in sigma.restored_hydrogen_indices)

    parent_symbols = Counter(
        atom.GetSymbol()
        for atom in Chem.AddHs(molecule_from_smiles(topology.monomer_smiles)).GetAtoms()
    )
    assert Counter(sigma.symbols) == Counter(
        {element: 2 * count for element, count in parent_symbols.items()}
    )


def test_c_c_and_c_n_use_the_same_indexed_constructor(
    topology_by_parent: dict[str, SigmaTopology]
) -> None:
    carbon_carbon = build_sigma_complex(topology_by_parent["M001"], n_conformers=2)
    carbon_nitrogen = build_sigma_complex(topology_by_parent["M060"], n_conformers=2)

    assert carbon_carbon.topology.link_atom_pair == "C-C"
    assert carbon_nitrogen.topology.link_atom_pair == "C-N"
    assert carbon_carbon.force_field == carbon_nitrogen.force_field == "MMFF94"


def test_sigma_embedding_is_byte_deterministic(
    topology_by_parent: dict[str, SigmaTopology]
) -> None:
    topology = topology_by_parent["M001"]
    first = build_sigma_complex(topology, n_conformers=4)
    second = build_sigma_complex(topology, n_conformers=4)

    assert first.xyz_text() == second.xyz_text()
