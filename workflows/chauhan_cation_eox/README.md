# Chauhan explicit-cation oxidation experiment

## Purpose

This isolated diagnostic tests how one explicit ionic-liquid cation changes the GFN2-xTB/ddCOSMO vertical oxidation IP of an explicit solvent-anion complex. Experimental values are the 50 mol% ionic liquid + 50 mol% solvent CV anodic limits reported by Rohit Chauhan et al., *Journal of Power Sources* 669 (2026) 239438, all versus Ag/AgCl. The values are joined only for comparison; this workflow performs no fitting, calibration, regression, offset fitting, or IP-to-Eox conversion.

## Matrix and controlled topologies

The solvents are PC (`epsilon=65.0`), EG (`37.0`), and THF (`7.6`). The cation sets are NTF2 × {EMIM, BMIM, HMIM}, OTF × {EMIM, HMIM, BMPYRR}, and PF6 × {BMIM, HMIM}. This gives 24 experimental compositions and 72 triads.

Each composition has three deterministic, approximately linear initial arrangements: `CAS` (anion in the middle), `CSA` (solvent in the middle), and `ACS` (cation in the middle). They are controlled electrostatic starting topologies, not equilibrium linear-trimer claims. RDKit ETKDGv3 uses seed `20260817`; MMFF94 is applied when parameters exist, and PF6 is constructed as an ideal octahedron. Deterministic axial orientation and incremental separation remove clashes. Two weak xTB harmonic anchor-distance restraints (`force constant=0.02 Eh/Bohr^2`, reference distance `auto`) preserve the intended middle fragment without fixing fragments or imposing exact collinearity.

## Electronic states

All calculations use GFN2-xTB >= 6.6 and `--cosmo <epsilon>` with the explicit and bulk solvent matched. Triads optimize at charge/UHF `0/0`, then run the oxidized single point at the identical geometry with `+1/1`. The nine A-S references and nine solvent-conditioned isolated anions optimize at `-1/0`, then run `0/1`. The three isolated solvents optimize at `0/0`, then run `+1/1`. The full manifest contains 93 two-state tasks, or 186 xTB calculations.

## Generate, submit, parse

From the repository root:

```bash
python workflows/chauhan_cation_eox/build_structures.py
python workflows/chauhan_cation_eox/make_manifest.py
qsub hpc/run_chauhan_cation_eox.sh
python workflows/chauhan_cation_eox/parse_results.py
python workflows/chauhan_cation_eox/aggregate_results.py
```

The array runner is resumable: a state is reused only when its energy, atom-resolved charges, and required geometry are parseable. Raw calculations and provenance stay under `runs/chauhan_cation_eox/`. The oxidized input hash must equal the optimized reduced-geometry hash. `triad_results.csv` records fragment charge changes, oxidation localization, topology preservation, minimum heavy-atom distances, and anchor distances. `as_reference_results.csv` records the direct A-S IP and the diagnostic `min(IP_anion, IP_solvent - 2.8 eV)` descriptor; the 2.8 eV term is not revalidated for ddCOSMO.

Expected tracked table sizes are 72 rows in `triad_results.csv`, 9 in `as_reference_results.csv`, and 24 in `composition_summary.csv`. Before calculations they are emitted with `not_run_or_incomplete` status rather than fabricated values.
