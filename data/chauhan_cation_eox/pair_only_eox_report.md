# Cation-free solvent-anion pair Eox test

All nine existing A-S calculations passed the frozen protocol and were reused; no A-S, triad, or cation calculation was rerun.

Calculated pair-only Eox range: 6.787794 to 8.133676 V vs Ag/AgCl. Experimental range: 1.540000 to 2.840000 V.
Oxidation localization across the 9 unique pairs: A=0, S=9.

## Absolute 24-row comparison

| Anion | Solvent | Cation | Experimental Eox vs Ag/AgCl / V | Pair IP / eV | Pair calculated Eox vs Ag/AgCl / V | Error / V | Oxidized fragment |
| ----- | ------- | ------ | ------------------------------: | -----------: | ---------------------------------: | --------: | ----------------- |
| NTF2 | PC | EMIM | 2.800000 | 12.419319 | 7.942319 | 5.142319 | S |
| NTF2 | EG | EMIM | 1.950000 | 11.750515 | 7.273515 | 5.323515 | S |
| NTF2 | THF | EMIM | 2.100000 | 11.351040 | 6.874040 | 4.774040 | S |
| NTF2 | PC | BMIM | 2.800000 | 12.419319 | 7.942319 | 5.142319 | S |
| NTF2 | EG | BMIM | 2.500000 | 11.750515 | 7.273515 | 4.773515 | S |
| NTF2 | THF | BMIM | 2.200000 | 11.351040 | 6.874040 | 4.674040 | S |
| NTF2 | PC | HMIM | 2.840000 | 12.419319 | 7.942319 | 5.102319 | S |
| NTF2 | EG | HMIM | 2.580000 | 11.750515 | 7.273515 | 4.693515 | S |
| NTF2 | THF | HMIM | 2.060000 | 11.351040 | 6.874040 | 4.814040 | S |
| OTF | PC | EMIM | 1.600000 | 12.367127 | 7.890127 | 6.290127 | S |
| OTF | EG | EMIM | 1.980000 | 11.722122 | 7.245122 | 5.265122 | S |
| OTF | THF | EMIM | 1.540000 | 11.264794 | 6.787794 | 5.247794 | S |
| OTF | PC | HMIM | 2.550000 | 12.367127 | 7.890127 | 5.340127 | S |
| OTF | EG | HMIM | 2.410000 | 11.722122 | 7.245122 | 4.835122 | S |
| OTF | THF | HMIM | 2.300000 | 11.264794 | 6.787794 | 4.487794 | S |
| OTF | PC | BMPYRR | 2.550000 | 12.367127 | 7.890127 | 5.340127 | S |
| OTF | EG | BMPYRR | 2.350000 | 11.722122 | 7.245122 | 4.895122 | S |
| OTF | THF | BMPYRR | 2.270000 | 11.264794 | 6.787794 | 4.517794 | S |
| PF6 | PC | BMIM | 2.400000 | 12.610676 | 8.133676 | 5.733676 | S |
| PF6 | EG | BMIM | 2.400000 | 11.832753 | 7.355753 | 4.955753 | S |
| PF6 | THF | BMIM | 2.200000 | 11.497465 | 7.020465 | 4.820465 | S |
| PF6 | PC | HMIM | 2.350000 | 12.610676 | 8.133676 | 5.783676 | S |
| PF6 | EG | HMIM | 2.640000 | 11.832753 | 7.355753 | 4.715753 | S |
| PF6 | THF | HMIM | 2.240000 | 11.497465 | 7.020465 | 4.780465 | S |

## Pair-only metrics

| weighting | n | MAE / V | RMSE / V | mean signed error / V | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| 24-row composition-weighted | 24 | 5.060356 | 5.077952 | 5.060356 | 0.463069 | 0.605086 |
| 9-row unique-AS | 9 | 5.068275 | 5.081147 | 5.068275 | 0.642093 | 0.833333 |

## Pair-only versus existing triad descriptors

| descriptor | n | MAE / V | RMSE / V | mean signed error / V | Pearson r | Spearman rho |
|---|---:|---:|---:|---:|---:|---:|
| AS_direct | 24 | 5.060356 | 5.077952 | 5.060356 | 0.463069 | 0.605086 |
| CAS | 24 | 4.773655 | 4.789306 | 4.773655 | -0.051081 | 0.164092 |
| CSA | 24 | 4.734367 | 4.751072 | 4.734367 | -0.124921 | -0.108814 |
| ACS | 24 | 4.768031 | 4.784625 | 4.768031 | -0.188425 | -0.220675 |
| triad_min | 24 | 4.692510 | 4.709151 | 4.692510 | -0.085654 | -0.053101 |
| triad_mean | 24 | 4.758684 | 4.774523 | 4.758684 | -0.128448 | -0.168444 |

At the fixed 4.477 V reference conversion, pair-only MAE is 5.060356 V. Relative absolute agreement: CAS: triad better by 0.286701 V MAE; CSA: triad better by 0.325989 V MAE; ACS: triad better by 0.292325 V MAE; triad_min: triad better by 0.367846 V MAE; triad_mean: triad better by 0.301672 V MAE.
No offset was fitted, no Chauhan calibration was performed, and no data were centered.
