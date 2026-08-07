# Data dictionary

## ID system

| Prefix | Meaning |
|---|---|
| `M001–M100` | monomer parents |
| `S001–S025` | media |
| `A001–A011` | canonical-SMILES-unique anions |
| `D001–D100` | monomer-specific sigma-complex dications |
| `RXN_MOX_*` | monomer oxidation |
| `RXN_SOX_*` | solvent-target self oxidation |
| `RXN_AOX_*` | anion oxidation |
| `RXN_SIG_*` | sigma coupling |

State IDs encode charge and multiplicity, for example `M022_QP1_M2` and `A003_QM1_M1`.

## Production authorities

- `source_fullspace_*.csv`: identity/provenance snapshots.
- `solvent_smd_registry.csv`: sole source of ORCA SMD payloads and model medium vectors.
- `fullspace_state_registry.csv`: electronic-state definition and geometry lookup key.
- `fullspace_reaction_registry.csv`: exact stoichiometry and inference medium policy.
- `calibration_tuple_design.csv`: split and sparse/full-25 medium assignment.
- job manifests: exact planned scientific jobs.

Historical SMD status text in `source_fullspace_solvents.csv` is not executable configuration.

## SMD vector

Order:

```text
ln(epsilon), soln_293K, soln25_298K, sola, solb, solg, solc, solh
```

The registry values are frozen model inputs. ORCA echo parsing verifies execution but never overwrites the registry or model vector.

Native rows use the exact native ORCA keyword. Custom rows use `orca_parameter_payload_resolved`. Water always uses native `SMD(Water)`.

## Reaction stoichiometry

`stoichiometry` is a semicolon-delimited list of `state_id:coefficient`. Every energy, Gibbs value and reaction embedding uses the same signed coefficients.

## Scientific job key

```text
job_class | state_id | solvent_id | geometry_hash | workflow_revision | method_id
```

A scheduler bundle may contain several logical jobs, but it cannot change this identity.

## Exact reuse key

A prior output is reusable only when state, medium, geometry hash, method ID, workflow revision and thermochemistry convention match. Store output SHA-256 and QC status.

## Output units

| Quantity | Unit |
|---|---|
| raw ORCA energy | Hartree |
| ML state/reaction energy | eV |
| redox report | V vs project Ag/AgCl convention |
| sigma report | kcal/mol; train in eV |
| geometry | Å |
| planning/actual CPU use | core-hour |

## Split

`parent_id` is the split unit. All charge states, media, sigma product and conformers of a parent remain in one split. Medium identity is a shared condition across splits.

## Result tables

- `state_sp_labels.csv`: raw state-medium SP and fixed base-model quantities.
- `reaction_sp_labels.csv`: complete reaction SP residuals and SMD decomposition.
- `reaction_final_labels.csv`: complete adiabatic reaction labels and final residuals.
- `fullspace_predictions_template.csv`: final prediction schema.
