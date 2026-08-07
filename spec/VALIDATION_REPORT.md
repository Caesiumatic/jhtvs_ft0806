# Sigma topology validation report

Status: **PASS**

| Check | Result |
|---|---:|
| Monomer rows | 100 |
| Hexamer rows | 100 |
| Exact row-name alignment | 100/100 |
| RDKit parse | 200/200 |
| Unique six-copy exact cover | 100/100 |
| Exactly two termini | 100/100 |
| Five inter-repeat bonds | 100/100 |
| Inter-repeat bond type | 500/500 single |
| Exact n=6 canonical reconstruction | 100/100 |
| Neutral n=2 graph sanitization | 100/100 |
| Neutral n=2 formula = 2M - H2 | 100/100 |
| Expected sigma formula = 2M, charge +2 | 100/100 |
| C-C rows | 91 |
| C-N rows | 9 |
| Coupling-site atoms | C=191, N=9 |
| Site H availability | one H=191, two H=9 |
| Symmetry-equivalent site pairs | 60 |
| Non-equivalent site pairs | 40 |
| Approved topology rows | 100 |

Source SHA-256:

```text
monomer CSV: a7781bfb8860257e1817131466ffa686c910c9c9ef7a428975924fcda5fdb3ed
hexamer CSV: 591811f592d032728632b85ae35a270f4cb82308dd5d1631d05c44d4709052e2
```

The reconstruction test creates six exact monomer copies, joins
`site_b(i)` to `site_a(i+1)` with a single bond, sanitizes, and compares canonical SMILES with the
frozen explicit hexamer.
