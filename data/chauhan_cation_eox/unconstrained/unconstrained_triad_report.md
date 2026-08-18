# Fully unconstrained Chauhan triad report

- SGE job IDs: `424251`; complete unconstrained triads: 72/72.
- Initial-topology preservation: CAS 24/24, CSA 24/24, ACS 21/24.
- Final-minimum classes at heavy-atom RMSD ≤ 0.25 Å: all_same=0, two_same_one_distinct=0, three_distinct=24.
- Median RMSD (Å): CAS–CSA=2.8219, CAS–ACS=3.4593, CSA–ACS=3.4114.
- Free-minus-restrained IP change (eV): mean=0.020143, median=0.009757, median absolute=0.025343, range=-0.135594 to 0.201729.
- Oxidation localization: C=41, A=0, S=31 (restrained: C=41, A=0, S=31).

## Raw absolute metrics

| descriptor | MAE / V | RMSE / V | signed error / V | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| AS_direct | 5.060356 | 5.077952 | 5.060356 | 0.463069 | 0.605086 |
| restrained_triad_min | 4.692510 | 4.709151 | 4.692510 | -0.085654 | -0.053101 |
| unconstrained_free_min | 4.719703 | 4.736644 | 4.719703 | -0.154659 | -0.131012 |
| unconstrained_lowest_energy_geometry | 4.774135 | 4.790266 | 4.774135 | -0.181682 | -0.079652 |

## Global offset-only metrics (slope fixed at 1)

| descriptor | offset / V | MAE / V | RMSE / V | R² | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|---:|
| AS_direct | -5.060356 | 0.337267 | 0.422369 | -0.649713 | 0.463069 | 0.605086 |
| restrained_triad_min | -4.692510 | 0.278591 | 0.395545 | -0.446829 | -0.085654 | -0.053101 |
| unconstrained_free_min | -4.719703 | 0.299205 | 0.400250 | -0.481448 | -0.154659 | -0.131012 |
| unconstrained_lowest_energy_geometry | -4.774135 | 0.279692 | 0.392794 | -0.426771 | -0.181682 | -0.079652 |

## Direct answers

Unconstrained free-min versus restrained triad min: raw MAE change=+0.027193 V; offset-fitted MAE change=+0.020614 V; R² change=-0.034619; Pearson change=-0.069005; Spearman change=-0.077911.
Lowest-neutral-energy and minimum-IP choices are the same for 10/24 compositions. Their raw MAE difference (lowest-energy minus free-min) is +0.054432 V; offset-fitted difference is -0.019513 V.
Unconstrained free-min versus A-S pair-only: raw MAE change=-0.340653 V; offset-fitted MAE change=-0.038062 V.
Topology changes and charge-localization maxima are reported as numerical outcomes only; no mechanistic assignment is inferred.
