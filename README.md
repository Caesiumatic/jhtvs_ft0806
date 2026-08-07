# jhtvs_ft0806

Typed, provenance-first implementation of the supplied solvent-conditioned
Tier-2 property-surrogate specification. The immutable scientific authorities
and manifests are stored in `spec/`.

## Development setup

```bash
python -m pip install -e '.[test]'
jhtvs-ft0806 validate-spec
pytest
```

The CLI defaults to the repository-local `spec/` directory. Calculation output
belongs under `runs/` and large model/cache files under `artifacts/`; both are
excluded from Git apart from compact indexes and reports explicitly promoted
to tracked locations.
