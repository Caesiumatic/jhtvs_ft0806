# MACE-POLAR-1-L five-solvent redox

This isolated workflow predicts zero-shot explicit-solvent oxidation free energies with
MACE-POLAR-1-L. It does not read or modify accepted artifacts under
`diagnostics/explicit_solvation_sp/`.

The frozen validation scope combines the nine included monomer observations from the
validation-plot audit with the primary-audited v2 solvent/anion core. The latter contains
seven solvent identities and five anion/environment identities; repeated anion observations
map to one system prediction.

Generate the frozen manifests:

```bash
python -m jhtvs_ft0806.explicit_redox.manifest \
  --legacy-audit /path/to/validation_points_audit.csv \
  --primary-core /path/to/core_primary_eox.csv \
  --calibration-registry /path/to/jhtvs_fullspace/calib/calib_data.csv \
  --monomer-registry spec/source_fullspace_monomers.csv \
  --solvent-registry spec/source_fullspace_solvents.csv \
  --output workflows/mace_polar_5solv_redox \
  --base-commit a6374267297c6c65e01934863e3e29651fe4ab4d
```
