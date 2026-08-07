# jhtvs_ft0806 implementation handoff v3

## Use this bundle only

Copy every file in this directory into the Codex workspace. Start with `CODEX_PROMPT.md`.
Do not mix the older starter bundle or the older unresolved SMD registry into the implementation.

## Frozen task

Build one solvent-conditioned property surrogate for:

- monomer oxidation in 25 media;
- each solvent target's self-medium oxidation;
- 11 unique anion oxidations in 25 media;
- `deltaG_sigma = G(D2+) - 2 G(M+ radical)` in 25 media.

The model does not perform geometry optimization and does not provide validated forces.

## Authoritative inputs

| File | Rows | Role |
|---|---:|---|
| `solvent_smd_registry.csv` | 25 | Sole production SMD registry; 25/25 complete vectors |
| `fullspace_state_registry.csv` | 372 | Unique electronic states |
| `fullspace_reaction_registry.csv` | 236 | Full-space reaction definitions |
| `calibration_tuple_design.csv` | 88 | 60 train / 14 val / 14 test calibration reactions |
| `sp_job_manifest.csv` | 735 | 705 SMD SP + 30 gas diagnostics |
| `optfreq_job_manifest.csv` | 80 | First complete Opt/Freq state-medium jobs |
| `training_config.csv` | 57 | Model and training settings |
| `compute_budget.csv` | 5 | Runtime guards and planned cost |

The six full-25 SP anchors are exactly:

- train: `RXN_MOX_M022`, `RXN_AOX_A005`, `RXN_SIG_M022`;
- blind test: `RXN_MOX_M084`, `RXN_AOX_A002`, `RXN_SIG_M027`.

`RXN_SIG_M084` is a four-medium sparse tuple (`S004;S011;S016;S024`), not a seventh full-25 anchor.

## Initial calculation package

- fixed-geometry jobs: 735; planned `1601.30` core-h;
- Opt/Freq jobs: 80; planned `2289.20` core-h;
- initial total: `3890.50` core-h before retries;
- first-round hard stop: `8000` core-h;
- whole-project cap: `12000` core-h.

## Scientific precedence

1. `01_SCIENTIFIC_SPEC.md` and the registries/manifests in this bundle;
2. `solvent_smd_registry.csv` for all medium parameters;
3. pinned current Tier-1/Tier-2 workflow sources in `source_registry.csv`;
4. `20260707/` only for implementation patterns.

The supplied full-space solvent CSV is provenance only. Its historical incomplete-SMD notes do not override the resolved registry.
