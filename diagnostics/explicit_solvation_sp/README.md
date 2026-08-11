# Explicit-solvation DFT vs MACE-POLAR-1 diagnostic

This directory is an isolated, fixed-coordinate electronic-energy benchmark. It computes

`delta E(R) = E(q=+1, R) - E(q=0, R)`

for thiophene in acetonitrile and DMSO in dichloromethane. It does not perform geometry optimization, frequencies, thermochemistry, continuum solvation, reference-electrode conversion, Marcus averaging, training, or fine-tuning.

## Fixed inputs

The four molecule geometries are byte-identical snapshots of the neutral Tier-2 optimized XYZ files in the read-only `20260707` project. `source_provenance.csv` records the source XYZ, ORCA input/output paths, SHA-256 values, and convergence/termination evidence. DMSO uses the available neutral `CS04_DMSO_CS04_chg0` intramolecular geometry; no DMSO/DCM optimized geometry exists in that source set.

Packmol fixes one solute at the origin and packs every solvent molecule into a centered all-atom containment cube with a 2.000 Å intermolecular tolerance:

| system | cluster | solvent count | box (Å) | seed |
|---|---:|---:|---:|---:|
| thiophene/acetonitrile | R5 | 5 | -5.650 to +5.650 | 814105 |
| thiophene/acetonitrile | R50 | 50 | -9.750 to +9.750 | 814150 |
| DMSO/dichloromethane | R5 | 5 | -5.800 to +5.800 | 814205 |
| DMSO/dichloromethane | R50 | 50 | -10.250 to +10.250 | 814250 |

The box rule in `protocol.json` starts from liquid molecular volumes (86.73 Å³ per acetonitrile and 106.36 Å³ per dichloromethane), adds an effective solute volume, takes the cube root, and adds one solvent molecular span so that the all-atom box leaves the intended center-accessible volume.

## Electronic-structure matrix

- ORCA 6.1, R5 only: `wB97M-V def2-TZVPD def2/J RIJCOSX TightSCF DEFGRID3`, gas/no SMD/CPCM, SPE, 8 MPI ranks, `%maxcore 3000`.
- MACE-POLAR-1-L, R5 and R50: checkpoint SHA-256 `9f65f8dc...ef114b`, float64, CPU.
- States for every evaluated cluster: charge 0/multiplicity 1 and charge +1/multiplicity 2.
- Each charge-state pair reads the same cluster XYZ. ORCA coordinate blocks are byte-hash checked across the pair.

## Reproduction

From the repository root, with Packmol installed:

```bash
PYTHONPATH=src python diagnostics/explicit_solvation_sp/run_diagnostic.py prepare \
  --source-root /path/containing/20260707 --packmol /path/to/packmol
```

On Lop, the prepared ORCA task table is submitted with the repository `hpc/run_orca.sh` runner. MACE is run through `run_mace.sh`, which activates the locked `jhtvs-ft0806` environment and checks the checkpoint and cluster-manifest hashes. After the four ORCA outputs and MACE result are present:

```bash
PYTHONPATH=src python diagnostics/explicit_solvation_sp/run_diagnostic.py collect
```

`cluster_manifest.csv`, `orca/job_manifest.csv`, `comparison.csv`, `summary.csv`, `qc.json`, `execution_status.json`, and `REPORT.md` are the acceptance artifacts. Large ORCA outputs and scratch files remain untracked; `orca/raw_results.json` retains their stable paths, SHA-256 values, raw Hartree energies, and QC. MACE provenance records the checkpoint filename and SHA-256 without embedding a host-specific cache path.
