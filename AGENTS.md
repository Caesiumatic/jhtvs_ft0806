# AGENTS.md

Implement the supplied `jhtvs_ft0806` specification. Do not respond with a plan before coding.

## Working style

- For each small stage: implement, test, inspect the diff/results, fix, commit and push.
- Keep stage updates to completed work, tests and commit SHA. Review concrete diffs/results only; no speculative risk essay.
- Never force-push, rewrite history or combine unrelated stages.
- Solve engineering issues autonomously. Stop only for a scientific decision listed below.

## Source precedence

1. `spec/01_SCIENTIFIC_SPEC.md` and authoritative CSVs.
2. `spec/solvent_smd_registry.csv` for every SMD field and model vector.
3. Pinned current project workflows in `spec/source_registry.csv`.
4. `20260707/` for implementation patterns only.

Do not copy the 20260707 scientific task matrix, target definitions, split or workflow boundary.

## Scientific stop

Stop immediately before making a choice that changes chemical identity, charge/multiplicity, sigma topology/stoichiometry, SMD vector, Tier-2 method, standard-state/reference conversion, target, split, QC inclusion or model scientific architecture.

```text
SCIENTIFIC DECISION REQUIRED
Decision:
Evidence:
Affected items:
Options:
Safe work already pushed:
Last pushed commit:
```

Paths, package APIs, SGE bundling, logging, parsing details, tests and dependency wiring are engineering work; resolve them and continue.
