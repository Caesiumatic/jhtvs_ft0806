"""Deterministic graph construction from the frozen sigma topology table."""

from __future__ import annotations

import re
from collections import Counter

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdchem import Mol


class TopologyError(ValueError):
    """Raised when a frozen coupling topology cannot be constructed exactly."""


def molecule_from_smiles(smiles: str) -> Mol:
    """Parse and sanitize a source SMILES without changing its atom order."""

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise TopologyError(f"RDKit could not parse SMILES: {smiles}")
    return molecule


def build_repeat_chain(
    monomer_smiles: str,
    site_a_atom_index_0based: int,
    site_b_atom_index_0based: int,
    copies: int,
) -> Mol:
    """Join exact monomer copies using copy_i.site_b--copy_i+1.site_a."""

    if copies < 1:
        raise TopologyError(f"copies must be positive, found {copies}")
    monomer = molecule_from_smiles(monomer_smiles)
    atoms_per_copy = monomer.GetNumAtoms()
    site_a = int(site_a_atom_index_0based)
    site_b = int(site_b_atom_index_0based)
    if site_a == site_b:
        raise TopologyError("site_a and site_b must be different atoms")
    if not 0 <= site_a < atoms_per_copy or not 0 <= site_b < atoms_per_copy:
        raise TopologyError(
            f"coupling indices {(site_a, site_b)} are outside a {atoms_per_copy}-atom monomer"
        )

    combined = Chem.RWMol()
    for _ in range(copies):
        combined.InsertMol(monomer)
    for copy_index in range(copies - 1):
        combined.AddBond(
            copy_index * atoms_per_copy + site_b,
            (copy_index + 1) * atoms_per_copy + site_a,
            Chem.BondType.SINGLE,
        )
    molecule = combined.GetMol()
    try:
        Chem.SanitizeMol(molecule)
    except Exception as exc:  # pragma: no cover - RDKit exception types vary by release
        raise TopologyError(f"constructed repeat chain does not sanitize: {exc}") from exc
    return molecule


def canonical_smiles(molecule: Mol) -> str:
    """Return an isomeric canonical SMILES for exact graph comparison."""

    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def molecular_formula(molecule: Mol) -> str:
    """Return RDKit's molecular formula for a sanitized graph."""

    return rdMolDescriptors.CalcMolFormula(molecule)


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_CHARGE_SUFFIX = re.compile(r"([+-])(\d*)$")


def formula_composition(formula: str) -> tuple[Counter[str], int]:
    """Parse a simple molecular formula into element counts and net charge."""

    charge = 0
    charge_match = _CHARGE_SUFFIX.search(formula)
    body = formula
    if charge_match is not None:
        magnitude = int(charge_match.group(2) or "1")
        charge = magnitude if charge_match.group(1) == "+" else -magnitude
        body = formula[: charge_match.start()]
    counts: Counter[str] = Counter()
    cursor = 0
    for match in _FORMULA_TOKEN.finditer(body):
        if match.start() != cursor:
            raise TopologyError(f"unsupported molecular formula: {formula}")
        counts[match.group(1)] += int(match.group(2) or "1")
        cursor = match.end()
    if cursor != len(body) or not counts:
        raise TopologyError(f"unsupported molecular formula: {formula}")
    return counts, charge
