# Chauhan explicit-cation oxidation experiment

## Purpose

This isolated diagnostic tests how one explicit ionic-liquid cation changes the GFN2-xTB/ddCOSMO vertical oxidation IP of an explicit solvent-anion complex. Experimental values are the 50 mol% ionic liquid + 50 mol% solvent CV anodic limits reported by Rohit Chauhan et al., *Journal of Power Sources* 669 (2026) 239438, all versus Ag/AgCl. The values are joined only for comparison; this workflow performs no fitting, calibration, regression, offset fitting, or IP-to-Eox conversion.

## Matrix and controlled topologies

The solvents are PC (`epsilon=65.0`), EG (`37.0`), and THF (`7.6`). The cation sets are NTF2 × {EMIM, BMIM, HMIM}, OTF × {EMIM, HMIM, BMPYRR}, and PF6 × {BMIM, HMIM}. This gives 24 experimental compositions and 72 triads.

Each composition has three deterministic, approximately linear initial arrangements: `CAS` (anion in the middle), `CSA` (solvent in the middle), and `ACS` (cation in the middle). They are controlled electrostatic starting topologies, not equilibrium linear-trimer claims. RDKit ETKDGv3 uses seed `20260817`; MMFF94 is applied when parameters exist, and PF6 is constructed as an ideal octahedron. Deterministic axial orientation and incremental separation remove clashes. Two weak xTB harmonic anchor-distance restraints (`force constant=0.005 Eh/Bohr^2`, reference distance `auto`) preserve the intended middle fragment without fixing fragments or imposing exact collinearity. The restraint is active only during triad optimization.

## Electronic states

All calculations use GFN2-xTB >= 6.6 and ddCOSMO via `--cosmo <static epsilon>`, with the explicit and bulk solvent matched. The constrained optimization generates only the reduced-state geometry `R0`. Every vertical IP uses an unrestrained reduced-state SP and an unrestrained oxidized-state SP on byte-identical copies of `R0`; the optimization energy is never used in the IP.

Triads use `0/0 -> +1/1`; A-S pairs and isolated anions use `-1/0 -> 0/1`; isolated solvents use `0/0 -> +1/1`. The 12 unique benchmark-required isolated cation/solvent references use `+1/0 -> +2/1` and are reused across matching anions and topologies. The production manifest contains 105 optimized geometries: 72 triads, 9 A-S pairs, 3 solvents, 9 anions, and 12 cations. Each task has one optimization plus two SP calculations, for 315 xTB invocations. The smoke reruns are not additional production chemistry.

## Generate, submit, parse

From the repository root:

```bash
python workflows/chauhan_cation_eox/build_structures.py
python workflows/chauhan_cation_eox/make_manifest.py
qsub hpc/run_chauhan_cation_eox.sh
python workflows/chauhan_cation_eox/parse_results.py
python workflows/chauhan_cation_eox/aggregate_results.py
```

The array runner is resumable across `reduced_opt`, `reduced_sp`, and `oxidized_sp`. Raw calculations and provenance stay under `runs/chauhan_cation_eox/`; both SP input hashes must equal the optimized-geometry hash. `triad_results.csv` keeps the constrained optimization energy separate from the two SP energies and records fragment charge changes, isolated-cation IP shifts, topology preservation, minimum heavy-atom distances, and anchor distances. `cation_reference_results.csv` contains the 12 isolated-cation references. `as_reference_results.csv` records the direct A-S IP and the diagnostic `min(IP_anion, IP_solvent - 2.8 eV)` descriptor; the 2.8 eV term is not recalibrated here.

Expected tracked table sizes are 72 rows in `triad_results.csv`, 9 in `as_reference_results.csv`, 12 in `cation_reference_results.csv`, and 24 in `composition_summary.csv`. Before calculations they are emitted with `not_run_or_incomplete` status rather than fabricated values. This workflow performs no IP-to-Eox calibration.
