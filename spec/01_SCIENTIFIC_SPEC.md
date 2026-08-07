# Scientific and model specification

## 1. Targets

For oxidation in medium `s`:

\[
\Delta G_{ox}^{T2,SMD}(s)=G_{oxidized}^{T2,SMD}(s)-G_{initial}^{T2,SMD}(s)
\]

Monomer, solvent target and anion use one shared redox model with a role embedding. The project deterministic conversion produces `Eox_vs_AgAgCl`; the model does not learn the reference-electrode constant.

For coupling:

\[
\Delta G_{sigma}^{T2,SMD}(s)=G_{D^{2+}}^{T2,SMD}(s)-2G_{M^{+\bullet}}^{T2,SMD}(s)
\]

This is the current `deltaG_sigma-only` definition. It has no proton term.

## 2. Reference geometries

For monomer, solvent target and anion states:

```text
same state and target medium
→ Tier-1 GFN2-xTB/ddCOSMO optimized geometry
→ immutable geometry SHA-256
```

For sigma dications:

```text
deterministic project sigma constructor
→ charge +2, multiplicity 1
→ target-medium GFN2-xTB/ddCOSMO preoptimization
→ connectivity/topology QC
→ immutable geometry SHA-256
```

These geometries are inputs to fixed-geometry Tier-2 SP labels and to full-space inference. The ML model does not optimize them.

## 3. Fixed-geometry Tier-2 SMD SP

Every `smd_energy_sp` row uses:

```text
ORCA 6.1
wB97X-D3 / def2-TZVPD
RIJCOSX + def2/J
TightSCF
DEFGRID3
SMD
8 MPI ranks
%maxcore 3000 MB/rank
energy only
```

Capture final electronic energy, `CPCM Dielectric`, `SMD CDS`, echoed solvent fields, geometry hash, output hash and normal termination.

The 30 gas jobs are parser/decomposition diagnostics only.

## 4. Complete Tier-2 free energy

Every Opt/Freq row uses the current project protocol:

```text
functional: wB97X-D3
q < 0 Opt/Freq basis: ma-def2-TZVP
q >= 0 Opt/Freq basis: def2-TZVP
final SP basis: def2-TZVPD
RIJCOSX + def2/J
TightSCF; DEFGRID3
tight geometry; MaxIter 200
SMD
298.15 K; 1 atm; Quasi-RRHO
project 1 M composite-Gibbs convention
```

\[
G_{T2,SMD}^{min}=E_{SP,TZVPD}+[G_{Freq}-E_{Freq}]
\]

Only complete, clean reaction tuples form final scalar labels.

## 5. Reaction-level labels

Let the original, unmodified `polar-1-l` checkpoint provide the fixed baseline energy for every state geometry. For stoichiometric coefficients `nu_i`:

\[
\Delta E_{base}^{rxn}=\sum_i \nu_i E_{polar-1-l,i}
\]

\[
y_{SP}=\Delta E_{T2,SMD,SP}^{rxn}-\Delta E_{base}^{rxn}
\]

\[
y_{RT}=\Delta G_{T2,SMD,min}^{rxn}-\Delta E_{T2,SMD,SP}^{rxn}
\]

\[
y_{final}=\Delta G_{T2,SMD,min}^{rxn}-\Delta E_{base}^{rxn}
\]

The production prediction is:

\[
\widehat{\Delta G}_{T2,SMD}^{rxn}=\Delta E_{base}^{rxn}+\widehat y_{final}
\]

Do not train a state-level absolute-energy target. Do not let LoRA change the stored baseline or target definitions.

## 6. Model

```text
PolarMACE state graph
  → raw model forward
  → invariant state features

8-D resolved SMD vector
  → 8→64→128 MLP, SiLU, LayerNorm

state representation + solvent representation
  → FiLM/gating
  → exact reaction aggregation
  → shared redox head + role embedding
  → separate sigma head
  → final, SP and RT outputs
```

Use MACE `v0.3.16`, `polar-1-l`, float64 and the raw PolarMACE model forward. Required official outputs are `energy`, `node_feats`, `density_coefficients`, `spin_density`, `spin_charge_density`, `dipole`, `electrostatic_energy` and `electron_energy`.

Invariant features:

- split `node_feats` using the loaded checkpoint irreps; retain `0e` channels from each interaction; sum and mean pool by molecule;
- density and spin-density monopole summaries and `l=1` norms;
- total dipole norm;
- electrostatic energy, electron energy, atom count, charge and multiplicity.

Do not feed raw orientation-dependent vector components.

## 7. Training

1. Frozen-backbone head warm-up for 50 epochs. Frozen features may be cached.
2. LoRA stage with online graph forward; rank 4, alpha 1.0. Frozen feature caches are invalid for this stage.
3. Five seeds: `17,29,43,71,101`.
4. Parent-grouped validation early stopping; patience 30.
5. Test split and blind full-25 anchors are untouched until architecture and hyperparameters are frozen.

Loss:

\[
L=L_{redox,final}+L_{sigma,final}+0.5L_{SP}+0.25L_{RT}+0.25L_{consistency}
\]

Normalize each target with train-only statistics. The total row weight within each parent is one.

## 8. QC and uncertainty

- `clean`: enter the applicable scalar losses;
- `flagged`: retain value and reason, exclude from primary scalar loss;
- `missing`: retain failure record only.

Each ensemble member produces the reaction property and screening margins. Compute ensemble mean/std after the reaction arithmetic. Report OOD/abstention from ensemble disagreement, representation distance and upstream QC.

## 9. Full-space output

Expected rows:

```text
100 monomers × 25 media = 2500
25 solvent targets × self-medium = 25
11 anions × 25 media = 275
100 sigma reactions × 25 media = 2500
total = 5300
```

All predictions carry geometry, solvent-vector, checkpoint and run provenance.
