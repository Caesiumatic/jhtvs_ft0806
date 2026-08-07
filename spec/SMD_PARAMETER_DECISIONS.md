# JHTVS 25-medium SMD parameter decisions

## Authority

`solvent_smd_registry.csv` is the production SMD registry for `jhtvs_ft0806`.
It contains **25/25 complete numeric vectors**. The provenance-only source table is `source_fullspace_solvents.csv`; it must not generate SMD decks or model features.

The project uses these media to predict solvent-conditioned monomer, solvent-target and anion oxidation properties and sigma-coupling thermodynamics within the joint monomer–solvent–electrolyte screen.

## Frozen vector

```text
log_epsilon | soln | soln25 | sola | solb | solg | solc | solh
```

- `log_epsilon`: natural logarithm of the static dielectric constant.
- `soln`: refractive index at 293 K.
- `soln25`: refractive index at 298 K.
- `sola`, `solb`: Abraham hydrogen-bond acidity and basicity used by SMD.
- `solg`: macroscopic surface-tension descriptor in cal mol^-1 A^-2.
- `solc`: aromatic-carbon fraction among non-hydrogen atoms.
- `solh`: F/Cl/Br fraction among non-hydrogen atoms.

For a measured surface tension `gamma` in mN/m or dyn/cm:

```text
solg = 1.43932 * gamma
```

These definitions follow the ORCA 6.1 manual, the original SMD paper, and the Minnesota Solvent Descriptor Database.

## Execution rule

| Registry class | ORCA deck | Model input |
|---|---|---|
| `native_orca_smd` | Use the exact native `SMD(name)`/`SMDsolvent` keyword in the CSV. | Use the resolved numeric vector. |
| `custom_smd` | Write the exact `orca_parameter_payload_resolved` fields. S007/S012 use their own ORCA SMD name only as an initialization seed, then override all nine registry fields with `smd18 false` and `draco false`. | Use the same resolved numeric vector. |
| Water | Always use native `SMD(Water)` so ORCA retains its special internal CDS treatment. | Use the resolved water vector. |

The pilot must parse ORCA's echoed fields and require:

```text
abs(observed - expected) <= max(5e-4, 5e-5 * abs(expected))
```

A mismatch is a scientific stop before production submission. For S007/S012,
ORCA 6.1's echoed solvent label must be `CUSTOM`: this identifies execution of a
user-defined descriptor set and is not the scientific medium identity. Their medium
identity is the bound combination of `solvent_id`, approved self seed, exact registry-row
SHA-256, nine-field echo, input/output hashes, method and workflow revision. Exact reuse
uses those bound fields plus state and geometry identity; it never uses the display-only
solvent label as a key.

## Resolved non-native and completed rows

| ID | Medium | Decision class | Frozen decision |
|---|---|---|---|
| S003 | Propylene carbonate (PC) | `project_complete_custom` | Retain the project-reviewed complete propylene-carbonate custom SMD set without modification. |
| S007 | Dimethyl sulfoxide (DMSO) | `custom_self_seed_registry_exact` | Keep the frozen MNSol vector unchanged; initialize with `SMDsolvent "DMSO"`, then explicitly override all nine fields because the native ORCA 6.1 echo differs from the registry. |
| S011 | γ-Butyrolactone (GBL) | `project_complete_custom` | Retain the project-reviewed complete gamma-butyrolactone custom SMD set without modification. |
| S012 | Sulfolane | `custom_self_seed_registry_exact` | Keep the frozen MNSol/n25-completed vector unchanged; initialize with `SMDsolvent "Sulfolane"`, then explicitly override all nine fields because the native ORCA 6.1 echo is incomplete or differs. |
| S013 | N-Methyl-2-pyrrolidone (NMP) | `physical_properties_plus_amide_analogue` | Use project epsilon/n and measured surface tension (41.3 mN/m); convert solg with 1.43932. Set alpha/phi/psi to zero from composition and use the MNSol tertiary-amide DMAc beta=0.78 analogue. |
| S014 | Boron trifluoride diethyl etherate (BFEE) | `project_effective_medium_plus_ether_analogue` | Preserve the project BFEE effective-medium epsilon/n. Use the MNSol diethyl-ether CDS analogue (alpha=0, beta=0.41, gamma=23.96), with BFEE composition-specific halogenicity 3/9. |
| S015 | 1-Butyl-3-methylimidazolium hexafluorophosphate ([BMIM][PF6]) | `project_epsilon_plus_smd_gil` | Use the project-specific dielectric constant and SMD-GIL n/alpha/beta/gamma; compute aromaticity and halogenicity exactly from the 1:1 ion-pair composition. |
| S016 | 1-Butyl-3-methylimidazolium tetrafluoroborate ([BMIM][BF4]) | `project_epsilon_plus_smd_gil` | Use the project-specific dielectric constant and SMD-GIL n/alpha/beta/gamma; compute aromaticity and halogenicity exactly from the 1:1 ion-pair composition. |
| S017 | 1-Ethyl-3-methylimidazolium bis(trifluoromethanesulfonyl)imide ([EMIM][TFSI]) | `project_epsilon_plus_smd_gil` | Use the project-specific dielectric constant and SMD-GIL n/alpha/beta/gamma; compute aromaticity and halogenicity exactly from the 1:1 ion-pair composition. |
| S018 | Choline chloride/urea (1:2) deep eutectic solvent | `reline_effective_ionic_medium` | Freeze one 1:2 reline effective-medium vector: project epsilon, measured n at 298.15 K, measured surface tension 52 mN/m converted to solg, SMD-GIL alpha/beta completion, and exact Cl heavy-atom fraction 1/16. |
| S022 | Ethylene carbonate | `project_epsilon_plus_published_ec_cds` | Use the project pure-EC dielectric at 313.15 K and the published pure-EC SMD non-electrostatic descriptors. Do not use the EC:EMC blend dielectric from that publication. |
| S023 | Dimethyl carbonate | `physical_properties_plus_carbonate_basicity` | Use project epsilon/n, measured surface tension 28.08 mN/m converted to solg, zero alpha/phi/psi, and the measured carbonate hydrogen-bond-acceptance value 0.40 as the SMD basicity completion. |
| S025 | Diethylene glycol dimethyl ether (diglyme) | `published_complete_cds_plus_project_epsilon` | Use the project dielectric and the published diglyme SMD CDS set (n=1.4097, beta=0.859, gamma=36.83; alpha/phi/psi=0). |

## Composition-derived fractions

| Medium | `solc` | `solh` |
|---|---:|---:|
| BFEE | 0 | 3/9 = 0.3333333333 |
| [BMIM][PF6] | 3/17 = 0.1764705882 | 6/17 = 0.3529411765 |
| [BMIM][BF4] | 3/15 = 0.2 | 4/15 = 0.2666666667 |
| [EMIM][TFSI] | 3/23 = 0.1304347826 | 6/23 = 0.2608695652 |
| ChCl/urea 1:2 | 0 | 1/16 = 0.0625 |

Fractions are computed on the exact neutral ion-pair or fixed 1:2 cluster representation in the project registry.

## Validation result

- Rows: 25 unique `solvent_id` values.
- Native execution rows: 12.
- Explicit custom rows: 13.
- All eight model fields are finite.
- `epsilon > 1`, `1 < soln, soln25 < 2`, `sola, solb, solg >= 0`, and `0 <= solc, solh <= 1` for all rows.
- `resolved_model_vector` follows the declared order and uses natural-log epsilon.
- Every custom row contains a complete explicit ORCA payload.
- No `missing`, `incomplete`, `TODO`, `CPCM until validated`, or `import approved project vector` status remains in the resolved registry.

## Source map

All URLs, source quality, supported fields and exact usage are recorded in `source_registry.csv`. Row-level source IDs and URLs are also embedded directly in `solvent_smd_registry.csv`.
