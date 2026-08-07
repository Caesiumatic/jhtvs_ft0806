# Continue jhtvs_ft0806 from scientific stop

Continue from pushed commit `db2a80f`. Do not answer with a plan, risk list, or broad review.
Read the supplied files and directly implement, test, inspect concrete outputs, fix, commit, and
push.

## Approved scientific decision

Option 2 is approved. Use:

- `sigma_coupling_topology.csv`
- `source_fullspace_monomers_with_coupling.csv`
- `source_fullspace_hexamers.csv`
- `SIGMA_TOPOLOGY_DECISION.md`
- `source_registry_sigma_addendum.csv`

The explicit n=6 hexamer is the topology authority. The 0-based atom indices refer to RDKit atom
order from the exact source monomer SMILES. Build the repeat bond as
`copy_i.site_b -- copy_i+1.site_a`. Do not infer or change sites from spin density or family rules.

## Implement

1. Add these files under `spec/`, merge provenance, and update `package_manifest.csv`.
2. Extend `validate-spec` for 100 unique topology rows and exact parent/state/reaction coverage.
3. Add a regression that reconstructs n=6 from the supplied indices and requires exact canonical
   equality with all 100 frozen hexamers.
4. Build the neutral n=2 backbone directly from the two monomer graphs and explicit indices; require equality with the supplied `neutral_dimer_smiles` and formula fields.
5. Generalize the 20260707 sigma constructor:
   neutral n=2 → add one H at each junction on opposite faces → q=+2, multiplicity=1.
6. Carry explicit junction indices through construction. Remove the old requirement that the
   junction be a unique non-ring aromatic C-C bond.
7. Use one code path for 91 C-C and 9 C-N rows. The C-N rows are M060-M068.
8. Add QC for exact junction, connectivity, 2× heavy-atom composition, restored 2× total atom
   composition, charge/multiplicity, and post-preoptimization graph preservation.
9. Resolve D001-D100 geometry keys and continue Stage 2.

The iodine-marker `coupling_smiles` is compatibility data; atom indices are primary.

## Tests before production continuation

- 100/100 row alignment;
- 100/100 unique six-copy exact covers;
- 100/100 exact reconstructed-hexamer matches;
- 91 C-C and 9 C-N links;
- each site has at least one restorable H;
- all 100 neutral n=2 backbones sanitize;
- representative sigma builds pass for M001, M060, M069, M084, M091, and M100;
- supplied scientific manifests retain their membership and counts.

## Commits

Push small rollback-safe commits after:

1. topology/spec integration and validation;
2. sigma constructor and tests;
3. geometry resolution and QC integration.

Do not force-push or rewrite pushed history.

Stop only if a supplied row cannot reproduce its frozen hexamer, cannot preserve the required
two-monomer atom composition/connectivity, or conflicts with q=+2/multiplicity=1. Solve engineering
issues directly.
