# Fadel vacuum vertical-IP benchmark

- SGE job IDs: `424252,424253,424254,424255,424256,424257,424258,424259,424260,424261,424262`; complete A-S: 16/16; complete Li-A-S topologies: 48/48.
- Official Figshare data were accessed. The workbook contains snapshot IDs, IPs and oxidation branches, but no atomic coordinates; the exact-geometry xTB snapshot diagnostic was therefore not run.
- Raw MAE: A-S=3.297728 eV; triad-min=6.705527 eV.
- Raw trends: A-S Pearson=0.793251, Spearman=0.732353; triad-min Pearson=0.794023, Spearman=0.650000.
- Offset-only MAE: A-S=0.778670 eV; triad-min=0.735815 eV.
- Paired comparison at |improvement| <= 0.05 eV tie tolerance: triad-min better=0, A-S better=16, tied=0; mean improvement=-3.407798 eV; median=-3.355025 eV.
- Fadel dominant A/S branch match: A-S=13/16; triad-min=15/16.
- All Li-A-S topology calculations localization: Li(C)=0, anion=16, solvent=32.
- Chemistry-resolved errors are in fadel_metrics_by_chemistry.csv; no electrochemical reference conversion or ddCOSMO term is present.
- This benchmark is intentionally independent of Chauhan CV and reference-electrode effects.
