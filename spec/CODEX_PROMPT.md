# Implement `jhtvs_ft0806`

Work in the empty repository `Caesiumatic/jhtvs_ft0806`. You have Lop access, GitHub access, this supplied bundle and access to `20260707/`.

**Do not answer with a plan. Read the bundle, validate it and start implementation. Do not produce a defensive risk register or a long review.**

## Non-negotiable working rule

For every small, reversible stage:

```text
implement → test → inspect results/diff → fix → commit → push
```

Use focused commits and push each one to `main`. Never force-push, rebase published history, squash unrelated work or leave a large uncommitted batch. Review only concrete code, tests, scientific invariants and outputs; do not write speculative risk lists. A stage update is at most a few lines: completed work, tests, commit SHA.

Solve engineering problems yourself: paths, environment discovery, package APIs, scheduler bundling, logging, parsers, fixtures, retries and dependency wiring are not reasons to stop.

For a scientific ambiguity, stop immediately after pushing all safe work and report only:

```text
SCIENTIFIC DECISION REQUIRED
Decision:
Evidence:
Affected items:
Options:
Safe work already pushed:
Last pushed commit:
```

Scientific stops are limited to chemical object identity, charge/multiplicity, sigma topology/stoichiometry, SMD vector/medium definition, Tier-2 method, standard state/reference conversion, target, split, QC inclusion or model scientific architecture. Do not silently choose among conflicting scientific definitions.

## Authority

Verify `package_manifest.csv` against the supplied bundle first. Put the byte-identical `AGENTS.md` at repository root; copy every other supplied CSV/MD into `spec/` without editing it.

Precedence:

1. `01_SCIENTIFIC_SPEC.md`, `03_ACCEPTANCE.md` and authoritative CSVs;
2. `solvent_smd_registry.csv` for all 25 medium parameters and model vectors;
3. pinned Tier-1/Tier-2 sources in `source_registry.csv`;
4. `20260707/` only for implementation patterns.

The source solvent CSV is provenance only. Ignore its historical incomplete-SMD notes. Never replace, refit, average or web-search a value in the resolved registry. Native rows use their native ORCA keyword; custom rows use the exact resolved payload; water uses native `SMD(Water)`. Echo comparison is an execution audit and never updates the model vector.

Use `20260707/` selectively for ORCA Compound/deck construction, Lop modules and SGE launcher, parser/QC, qacct, provenance, job idempotency and sigma-constructor code. Do not inherit its task matrix, scientific purpose, split, labels or Tier-1/Tier-2 isolation rule. When a reused implementation assumption affects science, stop.

## Repository structure

Create and keep this structure:

```text
README.md
AGENTS.md
pyproject.toml
spec/
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
runs/          # gitignored
artifacts/     # large files gitignored; small indexes allowed
```

Use a typed Python package, deterministic CSV writing, content hashes, structured logging and pytest. Raw ORCA outputs, checkpoints and large caches stay outside Git; commit manifests, compact parsed tables, checksums and reports.

## Implement in these stages

### 1. Scaffold and specification validation

Implement CLI commands:

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

`validate-spec` must enforce the counts and invariants in `03_ACCEPTANCE.md`, source-ID closure, unique scientific keys, exactly six full-25 anchors, parent split isolation, complete SMD vectors and budget totals. Add tests. Commit and push.

### 2. Geometry and exact reuse

Resolve every manifest `geometry_key` to a same-run XYZ and SHA-256.

- monomer/solvent-target/anion states: target-medium Tier-1 GFN2/ddCOSMO state geometry;
- sigma dication: deterministic constructor → charge +2/multiplicity 1 → target-medium GFN2/ddCOSMO preopt → connectivity QC.

Reuse an old Tier-2 output only when state, medium, geometry hash, method ID, workflow revision and thermochemistry convention match exactly. Populate `existing_tier2_reuse_inventory.csv`; do not alter supplied manifests. Commit and push.

### 3. ORCA and Lop pipeline

Generate decks directly from the manifests and resolved registry.

SP method:

```text
wB97X-D3/def2-TZVPD; RIJCOSX + def2/J; TightSCF; DEFGRID3; SMD; energy only
```

Opt/Freq method:

```text
q<0: ma-def2-TZVP Opt/Freq
q>=0: def2-TZVP Opt/Freq
final SP: def2-TZVPD
wB97X-D3; RIJCOSX + def2/J; TightSCF; DEFGRID3
tight geometry; MaxIter 200; SMD; 298.15 K; 1 atm; Quasi-RRHO
project 1 M composite Gibbs
```

Use ORCA 6.1/openMPI and Lop conventions verified from the pinned project/20260707 implementation. Build scheduler bundles without changing logical job identity. Submission must be idempotent. Account consumed and queued planned core-hours; block new submissions at 8000 first-round or 12000 project total. Add golden deck tests. Commit and push.

### 4. Parser, QC and label assembly

Parse normal termination, energies, CPCM dielectric term, SMD CDS term, echoed medium fields, optimized geometry, frequencies, thermal correction and scheduler accounting. Require the registry echo tolerance. An echo mismatch blocks production and triggers the scientific-stop report.

Implement:

```text
G = E_final_TZVPD_SP + (G_freq - E_freq)
sp_residual = deltaE_T2_SMD_SP_rxn - deltaE_base_MACE_rxn
rt_correction = deltaG_T2_SMD_min_rxn - deltaE_T2_SMD_SP_rxn
final_residual = deltaG_T2_SMD_min_rxn - deltaE_base_MACE_rxn
```

Use exact stoichiometry from the reaction registry. Call the pinned project deterministic Ag/AgCl conversion; do not recreate a constant. `clean` enters scalar loss, `flagged` retains value/reason outside primary loss, `missing` retains failure only. Use a small set of real 20260707 outputs as golden fixtures plus synthetic failure cases. Test signs and units. Commit and push.

### 5. Execute calculations

Run in this order:

1. one echo SP per medium;
2. state-class SP pilot covering neutral, radical cation, anion, neutral radical and sigma dication, including small and large structures;
3. one complete Opt/Freq reaction tuple;
4. remaining SP jobs after parser/QC/accounting pass;
5. Opt/Freq jobs in small waves after SP label assembly passes.

Do not add, replace or rebalance calibration rows. Commit compact pilot/completion reports and accounting indexes after each wave; do not commit raw outputs.

### 6. MACE-POLAR model

Pin MACE `v0.3.16` at commit `4d2da09413ac1407f37cdbb6b81fa28e4c15655e`, `polar-1-l`, float64, and record the actual checkpoint and dependency hashes.

Use the raw PolarMACE forward. Required outputs are `energy`, `node_feats`, `density_coefficients`, `spin_density`, `spin_charge_density`, `dipole`, `electrostatic_energy` and `electron_energy`.

Build only rotation-invariant features:

- checkpoint-irrep-aware `0e` node channels, sum and mean pooled per interaction;
- charge/spin monopole summaries and `l=1` norms;
- dipole norm, electrostatic/electron energy, atom count, charge and multiplicity.

Architecture and targets must follow `training_config.csv` and `01_SCIENTIFIC_SPEC.md`:

```text
state encoder + 8-D solvent encoder
→ FiLM/gating
→ exact reaction aggregation
→ shared redox head with role embedding / separate sigma head
→ final residual + SP residual + RT correction
```

The baseline is always the original unmodified `polar-1-l` reaction energy. In the frozen-head stage, cache base features. In the LoRA stage, run graphs online so gradients reach adapters; never train LoRA from the frozen cache and never redefine baseline labels with the adapted energy. Warm up heads for 50 epochs, then LoRA rank 4/alpha 1.0. Train five seeds. Add feature-rotation tests, frozen-head smoke test, adapter-gradient test, baseline-invariance test and leakage tests. Commit and push before full training.

### 7. Evaluation and full-space inference

Report grouped val/test MAE, Spearman, per-parent solvent trend, near-boundary false pass/false fail, top-K recall, interval coverage and abstention coverage-accuracy. Blind full-25 test anchors are opened only after the model and thresholds are frozen.

Each ensemble member first produces final reaction properties and screening margins; then calculate mean/std. Write 5300 rows to the schema in `fullspace_predictions_template.csv` with geometry, medium-vector, checkpoint and run provenance.

Write a concise `docs/IMPLEMENTATION_REPORT.md` containing commands, versions, commit/checkpoint hashes, calculation counts, exact reuse, QC, qacct core-hours, model metrics and remaining scientific blockers. Run the complete tests and final scientific-drift/leakage/unit/idempotency review, fix issues, commit and push.

## Completion condition

The task is complete when the resolved CSVs produce reproducible decks, parsed labels, the five-member model bundle, evaluation tables and 5300 full-space predictions; tests pass; manifests and hashes reproduce; no duplicate submission occurred; and the budget guard is active.

Start now. Do not send a preliminary plan.
