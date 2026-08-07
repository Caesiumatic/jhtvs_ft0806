# Sigma coupling topology decision

## Decision

Approve **Option 2**. Freeze all 100 sigma topologies from the supplied explicit n=6 hexamer
registry.

`spec/sigma_coupling_topology.csv` is authoritative. It also contains the canonical neutral n=2 backbone SMILES, explicit combined-molecule junction indices, and expected sigma formula. Runtime inference from spin density, family
rules, atom names, or SMARTS heuristics is forbidden.

## Frozen derivation

For every `M001-M100`:

1. Parse the exact source monomer SMILES and its row-aligned explicit n=6 hexamer.
2. Find the unique exact cover of the hexamer by six monomer graphs.
3. The two monomer atoms incident to the five inter-repeat single bonds are the termini.
4. `site_a` is the lower 0-based RDKit index in the exact source SMILES; `site_b` is the higher.
5. The repeat connection is `copy_i.site_b -- copy_i+1.site_a`.

This rule reproduces the canonical SMILES of all 100 frozen hexamers exactly.

## Sigma-complex construction

Use the 20260707 geometry-level semantics, with explicit table junction indices:

```text
two exact monomer graphs
→ single bond copy_1.site_b -- copy_2.site_a
→ neutral dehydrogenated n=2 backbone
→ deterministic embedding
→ add one H to each junction atom on opposite faces
→ charge +2, multiplicity 1
→ target-medium GFN2-xTB/ddCOSMO preoptimization
→ connectivity and formula QC
```

The same implementation handles:

- 91 C-C topologies;
- 9 C-N topologies: `M060-M068`.

For C-N, restore one H on N and one on C. The final sigma-complex atom multiset must equal two
parent monomers exactly.

The supplied `coupling_smiles` uses two iodine markers. Iodine is uniform across the table because
the library contains a native brominated monomer; using Br markers would create three functional
groups for that row. The atom-index columns remain the primary authority.

## Required QC

Every `D001-D100` must have:

- one connected component;
- exactly one inter-copy bond at the table-specified pair;
- heavy-atom multiset equal to 2 × parent monomer;
- total atom multiset after two restored H equal to 2 × parent monomer;
- charge `+2`, multiplicity `1`;
- no unexpected connectivity change after preoptimization;
- recorded monomer SHA, topology SHA, XYZ SHA, medium, and preoptimization settings.

Do not enumerate alternative couplings in this task.
