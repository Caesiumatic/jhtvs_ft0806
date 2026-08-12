# Explicit-R5 DFT Eox benchmark

This isolated diagnostic evaluates adiabatic oxidation potentials for one target plus five explicit solvent molecules, with the same solvent also represented by the frozen project SMD row.

Phase A contains six unique species-medium keys and twelve independent state-specific `wB97X-D3` Opt/Freq calculations. Each Compound deck evaluates its final `def2-TZVPD` SMD single point on the state-specific optimized `Compound_1.xyz`; only the oxidized final step requests Hirshfeld analysis.

The seven input CSVs and all upstream ORCA/XYZ sources are SHA-256 bound. `/Users/shichen/GitHub/20260707` is a read-only source checkout. Packmol uses a deterministic key-derived seed, 2.000 Å tolerance, one fixed central target during packing, and five shell molecules. No atoms are constrained during DFT optimization.

Commands:

```bash
PYTHONPATH=src python diagnostics/explicit_solvation_eox_r5/run_diagnostic.py prepare \
  --benchmark-dir /path/to/jhtvs_8_validation_plots_csv \
  --detail-dir /path/to/detail_csvs \
  --source-checkout /path/to/20260707 \
  --packmol packmol
PYTHONPATH=src python diagnostics/explicit_solvation_eox_r5/run_diagnostic.py validate-prepared
PYTHONPATH=src python diagnostics/explicit_solvation_eox_r5/run_diagnostic.py collect \
  --source-checkout /path/to/20260707
```
