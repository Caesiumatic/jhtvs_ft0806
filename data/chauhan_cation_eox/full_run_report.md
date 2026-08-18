# Chauhan cation-effect full-run report

- SGE production job IDs: `424249`.
- Complete evidence: 72/72 triads, 24/24 compositions; same-geometry validation passed during parsing for every retained task.
- Requested topology preserved: 70/72 (97.2%).
- Oxidation localization overall: C=41, A=0, S=31.

## Does cation identity measurably affect oxidation?

Yes in part: 8/21 experimental cation-pair contrasts exceed their reported RSS standard deviation. Group experimental spreads range from 0.0400 to 0.9500 V. These are direct group contrasts; no fitted calibration or mechanistic assignment was introduced.

| anion | solvent | n | experimental spread (V) | cation-only IP spread (eV) | CAS | CSA | ACS | min | mean | mean topology span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NTF2 | PC | 3 | 0.0400 | 0.7609 | 0.5544 | 0.4615 | 0.4877 | 0.4989 | 0.5012 | 0.0469 |
| NTF2 | EG | 3 | 0.6300 | 0.7766 | 0.2390 | 0.2846 | 0.1615 | 0.2061 | 0.2283 | 0.2511 |
| NTF2 | THF | 3 | 0.1400 | 0.9245 | 0.1370 | 0.0901 | 0.0695 | 0.1370 | 0.0849 | 0.1096 |
| OTF | PC | 3 | 0.9500 | 0.7609 | 0.4935 | 0.5354 | 0.5365 | 0.5365 | 0.5218 | 0.0451 |
| OTF | EG | 3 | 0.4300 | 0.7766 | 0.2382 | 0.0906 | 0.2082 | 0.1674 | 0.1741 | 0.1850 |
| OTF | THF | 3 | 0.7600 | 0.9245 | 0.1231 | 0.2053 | 0.0466 | 0.1178 | 0.1102 | 0.1806 |
| PF6 | PC | 2 | 0.0500 | 0.2114 | 0.2272 | 0.1752 | 0.1934 | 0.1862 | 0.1986 | 0.0294 |
| PF6 | EG | 2 | 0.2400 | 0.2190 | 0.1219 | 0.0582 | 0.0444 | 0.0582 | 0.0748 | 0.1953 |
| PF6 | THF | 2 | 0.0400 | 0.2913 | 0.0681 | 0.0348 | 0.0630 | 0.0681 | 0.0553 | 0.2228 |

## Descriptor comparison on 21 within-group cation contrasts

| descriptor | MAE (V) | RMSE (V) | Pearson r | Spearman rho | sign agreement | centered Pearson r | centered Spearman rho |
|---|---:|---:|---:|---:|---:|---:|---:|
| fadel_as_zero | 0.296666666667 | 0.438813334867 | NA | NA | 0.0952380952381 | NA | NA |
| isolated_cation | 0.778068528893 | 0.937116847155 | -0.552161552164 | -0.565514993481 | 0.142857142857 | -0.744944801208 | -0.711768696269 |
| CAS | 0.47289143914 | 0.617080542019 | -0.38015599269 | -0.376057970733 | 0.142857142857 | -0.631368035092 | -0.6299762055 |
| CSA | 0.445123032858 | 0.601383261089 | -0.411100906035 | -0.396877789182 | 0.238095238095 | -0.598987585948 | -0.602567019763 |
| ACS | 0.436793159381 | 0.591384111754 | -0.368067412744 | -0.34287638508 | 0.190476190476 | -0.582497560821 | -0.53034599068 |
| triad_min | 0.456334323854 | 0.607173800763 | -0.428115862467 | -0.413143272345 | 0.142857142857 | -0.639902279071 | -0.61387874721 |
| triad_mean | 0.451207868164 | 0.601641762862 | -0.397747495461 | -0.385817260631 | 0.190476190476 | -0.61742894444 | -0.584729295713 |

Lowest pairwise MAE among calculated descriptors: **ACS**, 0.436793159381 V; improvement over the Fadel A-S zero-contrast baseline: -0.140126 V.

The best calculated descriptor is still worse than the zero-contrast baseline by 0.140126 V MAE; the full-triad calculation therefore does not improve these 21 cation contrasts.
All three topology-specific Pearson correlations are negative, so none reproduces the aggregate experimental cation ranking direction across the 21 contrasts.

## Topology diagnostics

Composition topology-span distribution (eV): min=0.007356, Q1=0.047800, median=0.157401, mean=0.139579, Q3=0.217940, max=0.308245.

Non-preserved requested topologies:

| cation | anion | solvent | requested | inferred |
|---|---|---|---|---|
| HMIM | NTF2 | PC | ACS | CAS |
| EMIM | OTF | THF | CSA | CAS |

Ten largest topology spans:

| rank | cation | anion | solvent | span (eV) | min topology |
|---:|---|---|---|---:|---|
| 1 | BMIM | NTF2 | EG | 0.308245 | CSA |
| 2 | EMIM | NTF2 | EG | 0.238938 | ACS |
| 3 | EMIM | OTF | EG | 0.228770 | CSA |
| 4 | BMIM | PF6 | EG | 0.227191 | CSA |
| 5 | HMIM | PF6 | THF | 0.225308 | CAS |
| 6 | BMIM | PF6 | THF | 0.220248 | CAS |
| 7 | HMIM | OTF | THF | 0.217171 | CAS |
| 8 | HMIM | NTF2 | EG | 0.206091 | CSA |
| 9 | BMPYRR | OTF | THF | 0.178705 | CAS |
| 10 | BMPYRR | OTF | EG | 0.168149 | ACS |
