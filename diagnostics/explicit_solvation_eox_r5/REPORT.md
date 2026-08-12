# Explicit-R5 DFT Eox benchmark report

Workflow: `jhtvs-ft0806-explicit-r5-eox-v1`

Completed clean unique systems: 0 / 0

| calculation key | explicit R5 Eox / V | QC | reasons |
|---|---:|---|---|
| acetonitrile__self | — | missing | reduced:output_missing;oxidized:output_missing |
| propylene_carbonate__self | — | missing | reduced:output_missing;oxidized:output_missing |
| gamma_butyrolactone__self | — | missing | reduced:output_missing;oxidized:output_missing |
| clo4__acetonitrile | — | missing | reduced:output_missing;oxidized:output_missing |
| tfsi__acetonitrile | — | missing | reduced:output_missing;oxidized:output_missing |
| clo4__propylene_carbonate | — | missing | reduced:output_missing;oxidized:output_missing |

All accuracy metrics use the same completed-clean subset for implicit calibrated xTB, implicit DFT, and explicit-R5 DFT. Anion pairs with solvent-dominant oxidized Hirshfeld spin are retained as flagged raw values and excluded from primary metrics.

See `metrics_summary.json`, `record_comparison.csv`, `fragment_spin_qc.csv`, and `qc.json` for machine-readable results.
