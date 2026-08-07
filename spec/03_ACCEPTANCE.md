# Implementation acceptance

## Required repository layout

```text
README.md
AGENTS.md
pyproject.toml
spec/                  # immutable supplied bundle
config/
src/jhtvs_ft0806/
  cli.py
  schemas.py
  provenance.py
  geometry/
  orca/
  hpc/
  labels/
  ml/
hpc/
tests/
docs/
data/resolved/
runs/                  # gitignored
artifacts/             # large files gitignored; small indexes allowed
```

## Required CLI

```text
validate-spec
resolve-geometries
scan-reuse
build-decks
submit
status
collect-accounting
parse-results
assemble-labels
extract-base-features
train
evaluate
infer-fullspace
```

## Dataset acceptance

- 25 complete SMD rows: 12 native, 13 custom;
- 372 states and 236 reactions;
- 88 calibration reactions: 60 train, 14 val, 14 test;
- exactly six full-25 anchors;
- 735 SP rows: 705 SMD + 30 gas diagnostics;
- 80 Opt/Freq rows, yielding 50 complete reaction-medium final labels;
- every Opt/Freq state-medium cell has a matching fixed-geometry SMD SP cell;
- zero parent split leakage.

## Workflow acceptance

- inputs and manifests validate before any deck is written;
- resolved geometry hash is present before submission;
- native/custom SMD payloads are generated from the authoritative registry;
- 25-medium echo pilot, state-class pilot and one complete Opt/Freq tuple pass before production waves;
- submission is idempotent and resumable;
- qacct actual cost and queued planned cost enforce 8000/12000 core-hour guards;
- parser tests include real 20260707-derived golden fixtures and synthetic edge cases;
- sign, stoichiometry and Hartree/eV/kcal mol-1 conversions have unit tests;
- no test anchor contributes to fitting, normalization, early stopping or model selection.

## Model acceptance

- exact MACE source/version and checkpoint checksum are recorded;
- frozen-stage feature cache is content-addressed;
- LoRA stage uses online forward and reaches adapter parameters;
- original checkpoint energy remains the immutable reaction baseline;
- only rotation-invariant features enter heads;
- five-member ensemble produces property and margin distributions;
- evaluation reports grouped final-property error, solvent trends, screening-boundary errors, interval coverage and abstention curves;
- full-space inference writes 5300 provenance-complete rows.

## Scientific-stop format

```text
SCIENTIFIC DECISION REQUIRED
Decision:
Evidence:
Affected items:
Options:
Safe work already pushed:
Last pushed commit:
```

Stop only when the answer changes a chemical object, charge/multiplicity, sigma topology/stoichiometry, medium/SMD vector, Tier-2 method, standard-state/reference conversion, target, split, QC inclusion or model scientific architecture. Solve paths, packaging, API compatibility, scheduler bundling, logging, parser formatting, fixtures and dependency wiring without stopping.
