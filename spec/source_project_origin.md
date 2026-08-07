# Computational Methods Outline

**High-Throughput Screening for Joint Optimization of Electropolymerization Chemistries**

*Monomer - Solvent - Electrolyte Compatibility*

## 1. Objective

Design and execute a high-throughput computational screen that jointly evaluates the compatibility of conjugated monomer, solvent, and supporting electrolyte chemistries for electropolymerization. The screen ranks candidate *monomer-solvent-electrolyte* triads by enforcing three simultaneous constraints: (i) the monomer oxidation potential must fall within the solvent's electrochemical stability window; (ii) the electrolyte anion must be stable at the applied potential; and (iii) the monomer must be sufficiently soluble in the chosen solvent. A secondary objective is to predict the resulting polymer's doping-onset potential and band gap to assess material quality.

## 2. Chemical Space Definition

### 2.1 Monomer Library

The monomer library should span the electrochemically relevant classes of conjugated heterocycles and their functionalized derivatives. A recommended starting pool of ~80-150 monomers is organized below.

| Class | Representative Monomers |
| --- | --- |
| Thiophenes | Thiophene, 3-methylthiophene, 3-hexylthiophene (3HT), 3,4-ethylenedioxythiophene (EDOT), 3,4-propylenedioxythiophene (ProDOT), 3-fluorothiophene, 3,4-difluorothiophene, 3-methoxythiophene, bithiophene, terthiophene |
| Pyrroles | Pyrrole, N-methylpyrrole, N-octylpyrrole, 3,4-ethylenedioxypyrrole (EDOP), 3,4-dimethylpyrrole |
| Furans | Furan, 3-methylfuran, 3-hexylfuran, bifuran, terfuran, 3,4-disubstituted furans |
| Selenophenes | Selenophene, 3,4-ethylenedioxyselenophene (EDOS), 3-alkylselenophenes |
| Anilines | Aniline, o-methoxyaniline, o-aminophenol, o-toluidine, diphenylamine |
| Carbazoles | Carbazole, N-vinylcarbazole, 3,6-disubstituted carbazoles |
| Fluorenes | 9,9-dialkylfluorene, 9,9-dioctyl-fluorene |
| Donor-Acceptor units | Benzothiadiazole-thiophene, thienopyrazine, diketopyrrolopyrrole (DPP), isoindigo, thiadiazoloquinoxaline |
| Fused-ring systems | Cyclopentadithiophene (CPDT), dithienopyrrole, indacenodithiophene, acenaphtho[1,2-c]pyrrole hybrids |

### 2.2 Solvent Library

Include solvents spanning a range of dielectric constants and electrochemical windows. ~25-35 solvents are recommended.

| Solvent | Notes | εᵣ | ESW (V, approx) |
| --- | --- | ---: | ---: |
| Acetonitrile (MeCN) | High ε, wide window, most common | 37.5 | ~6.0 |
| Dichloromethane (DCM) | Low ε, good for non-polar monomers | 8.9 | ~4.5 |
| Propylene carbonate (PC) | High ε, wide window | 64.9 | ~6.5 |
| Nitrobenzene | Moderate ε, stabilizes radicals | 34.8 | ~4.0 |
| Benzonitrile | Moderate ε | 25.2 | ~5.0 |
| DMF | High ε, nucleophilic | 36.7 | ~4.5 |
| DMSO | Very high ε | 46.7 | ~4.0 |
| THF | Low ε, limited window | 7.6 | ~5.3 |
| Nitromethane | Moderate ε, wide window | 35.9 | ~5.0 |
| Water (pH 1-7) | Cheap, biocompatible | 80.1 | ~1.2-2.0 |
| γ-Butyrolactone (GBL) | High ε | 39.0 | ~5.5 |
| Sulfolane | High ε, thermally stable | 43.3 | ~5.5 |
| NMP | High ε, good solvation | 32.0 | ~4.5 |
| BFEE | Lowers Eᵒˣ dramatically | - | Lewis acid |
| [BMIM][PF₆] | Ionic liquid, dual role | - | ~4.5-5.5 |
| [BMIM][BF₄] | Ionic liquid | - | ~4.0-5.0 |
| [EMIM][TFSI] | Ionic liquid, wide window | - | ~5.0-6.0 |
| Choline Cl/urea DES | Deep eutectic solvent | - | ~2.5-3.0 |

### 2.3 Electrolyte Salt Library

The electrolyte anion becomes the dopant ion in the polymer film. Include ~20-30 salts.

| Salt | Class | Notes |
| --- | --- | --- |
| TBAPF₆ | Tetraalkylammonium | Standard for organic solvents |
| TBABF₄ | Tetraalkylammonium | Common alternative |
| TBAClO₄ | Tetraalkylammonium | Wide availability |
| TBAOTf | Tetraalkylammonium | Triflate dopant |
| TBATFSI | Tetraalkylammonium | Large, delocalized anion |
| TEAPF₆ | Tetraalkylammonium | Smaller cation |
| LiClO₄ | Lithium | Compact cation, aqueous-compatible |
| LiBF₄ | Lithium | Battery-relevant |
| LiTFSI | Lithium | Wide ESW, high solubility |
| NaClO₄ | Sodium | Aqueous systems |
| NaPSS | Polymeric | Large dopant, immobilized anion |
| NaDBSA | Surfactant | Amphiphilic dopant |
| H₂SO₄ (0.5-1 M) | Acid | Aniline polymerization |
| HClO₄ (0.1-1 M) | Acid | Wide window acid |
| pTSA | Organic acid | Tosylate dopant |
| CSA | Organic acid | Chiral dopant |
| KCl | Inorganic | Simple aqueous |
| AgClO₄ | Silver | Reference electrode use |

## 3. Properties to Calculate

### 3.1 Monomer Properties

| Property | Method | Target / Rationale |
| --- | --- | --- |
| Eᵒˣ(monomer) | Adiabatic ionization potential (AIP) via ΔSCF in implicit solvent, converted to E vs. Ag/AgCl | Target: 0.5-2.0 V vs. Ag/AgCl for facile polymerization. Must be < solvent anodic limit. |
| HOMO / LUMO | Kohn-Sham orbital energies at B3LYP/6-31G(d,p) + COSMO or SMD | Use as initial filter before ΔSCF. HOMO correlates with Eᵒˣ. |
| Spin density (radical cation) | Mulliken or Hirshfeld spin density on the monomer radical cation | High spin at α-carbon positions predicts α-α′ coupling selectivity. |
| Dimerization energy | ΔG for radical-radical coupling of two monomer cation radicals | Exothermic coupling (ΔG < 0) is required; more negative = more favorable. |
| Solvation free energy | SMD or COSMO-RS ΔGₛₒₗᵥ in each solvent | Proxy for solubility; ΔGₛₒₗᵥ < -5 kcal/mol suggests adequate solubility. |
| Oligomer Eᵒˣ (n = 2-6) | AIP of dimer through hexamer | Should decrease with n; polymer Eᵒˣ < monomer Eᵒˣ ensures film is not over-oxidized during growth. |
| Optical gap (polymer) | TD-DFT or sTDA-xTB on hexamer | Target: 1.0-3.0 eV depending on application. |

### 3.2 Solvent Properties

| Property | Method | Target / Rationale |
| --- | --- | --- |
| Anodic limit (Eᵒˣ solvent) | Adiabatic ΔSCF oxidation potential of solvent molecule in implicit self-solvent | Must exceed monomer Eᵒˣ by ≥ 0.3 V margin. |
| Cathodic limit (Eʳᵉᵈ solvent) | Adiabatic ΔSCF reduction potential | Defines lower bound of working window. |
| Dielectric constant (εᵣ) | Computed or experimental | Higher ε -> better ionic conductivity; use as filter. |
| Reorganization energy (λ) | λ = E(vertical) - E(adiabatic) for both oxidation and reduction | Large λ provides kinetic overpotential protection beyond thermodynamic window. |

### 3.3 Electrolyte Properties

| Property | Method | Target / Rationale |
| --- | --- | --- |
| Anion oxidation potential | Adiabatic ΔSCF of anion -> neutral radical in implicit solvent | Must exceed monomer Eᵒˣ. Anion should not be oxidized before monomer. |
| Cation reduction potential | Adiabatic ΔSCF of cation in implicit solvent | Must be below cathodic limit of solvent. |
| Ion-pair dissociation energy | ΔG for salt -> cation + anion in solvent (COSMO-RS or SMD) | Lower = better dissociation = higher conductivity. |
| Anion size / volume | Van der Waals volume from optimized geometry | Larger anions produce more open, porous polymer morphologies. |

## 4. Multi-Tier Computational Workflow

The workflow proceeds in three tiers of increasing accuracy, with each tier filtering the candidate space before promotion to the next.

### 4.1 Tier 1 - Rapid Pre-Screen (xTB / Semiempirical)

**Cost:** ~1-10 CPU-seconds per species. **Scope:** All monomers × all solvents × all electrolytes (full combinatorial space).

- **Structure generation.** Convert monomer SMILES to 3D using RDKit. For oligomers (n = 2-6), assemble with the Supramolecular Toolkit (stk). Run MMFF94 conformer search (50-200 conformers), retain lowest-energy structure.
- **GFN2-xTB geometry optimization** of neutral monomer, cation radical, anion, and solvent molecules in vacuum.
- **IPEA-xTB** vertical and adiabatic IP/EA calculations for monomers and solvents. Apply previously calibrated linear model (Zwijnenburg et al.) to map to B3LYP-quality values. Use GBSA implicit solvation with the dielectric constant of each candidate solvent.
- **COSMO-RS solvation free energies** for each monomer in each solvent (use COSMOtherm or open-source openCOSMO-RS). Filter out monomer-solvent pairs where ΔGₛₒₗᵥ > -3 kcal/mol.
- **sTDA-xTB optical gaps** for hexamer models of each monomer class. Calibrate against TD-DFT reference set.
- **Filtering criteria.** Retain triads where: monomer AIP < solvent anodic limit - 0.3 V; anion oxidation potential > monomer AIP + 0.2 V; monomer ΔGₛₒₗᵥ < -3 kcal/mol. Expected retention: ~10-20% of input space.

### 4.2 Tier 2 - DFT Refinement

**Cost:** ~0.5-2 CPU-hours per species. **Scope:** Tier 1 survivors only (~500-2,000 triads).

- **DFT optimization.** B3LYP/6-31G(d,p) with SMD implicit solvation (solvent-specific parameters). Optimize neutral, cation radical, and anion states for monomers; neutral and cation/anion states for solvents and electrolyte ions.
- **Adiabatic ΔSCF redox potentials.** Eᵒˣ = [G(cation) - G(neutral)] / F - E°(ref). Convert to vs. Ag/AgCl using the absolute potential of the reference electrode (~4.28 V for SHE, then -0.197 V shift). Include thermal and zero-point corrections at 298 K.
- **Spin density analysis** on monomer radical cations (Hirshfeld partitioning). Map reactive sites for α-α′ vs. α-β coupling.
- **Dimerization thermodynamics.** Compute ΔG for 2 M⁺˙ -> [M-M]²⁺ + 2H⁺ at the B3LYP/6-31G(d,p)/SMD level. Exothermic required; ΔG < -10 kcal/mol is strongly favorable.
- **Oligomer band-gap convergence.** TD-B3LYP or CAM-B3LYP on n = 1-6 oligomers. Verify convergence at n ≈ 4-6 for donor-acceptor and n ≈ 6 for homopolymers.
- **Reorganization energies.** Compute λ for solvent oxidation [λ = E(vertical IP) - E(adiabatic IP)]. Solvents with large λ have kinetic protection beyond the thermodynamic window.
- **Refined filtering.** Tighten margins to monomer AIP < solvent anodic limit - 0.5 V. Rank by a composite score (Section 5).

### 4.3 Tier 3 - High-Accuracy Validation (Optional)

**Cost:** ~10-100 CPU-hours per triad. **Scope:** Top 20-50 candidates.

- **Range-separated hybrid DFT.** CAM-B3LYP or ωB97X-D/6-311+G(d,p) with SMD for more accurate redox potentials. Particularly important for D-A monomers where B3LYP systematically underestimates gaps.
- **Explicit solvation shell.** Cluster-continuum approach: place 2-4 explicit solvent molecules around monomer and re-optimize in implicit solvent. Assess shift in Eᵒˣ due to specific solvent-monomer interactions (H-bonding, Lewis acid coordination).
- **Ab initio molecular dynamics (AIMD).** Short (5-10 ps) Born-Oppenheimer MD trajectories at the GFN2-xTB level for monomer cation radical in explicit solvent to probe radical stability and lifetime.
- **Electrode interface effects.** For the highest-priority triads, DFT slab calculations of monomer adsorption on model Au(111) or ITO surfaces to assess nucleation energetics.

## 5. Composite Scoring Function

Rank each monomer-solvent-electrolyte triad using a weighted composite score. The score penalizes triads that violate constraints and rewards favorable properties.

**S = w₁·f₁(window margin) + w₂·f₂(anion stability) + w₃·f₃(solubility) + w₄·f₄(dimerization) + w₅·f₅(band gap)**

| Weight | Component | Optimal | Notes |
| --- | --- | --- | --- |
| w₁ = 0.30 | f₁ = ESW margin = Eᵒˣ(solvent) - Eᵒˣ(monomer) | ≥ 0.5 V | Hard constraint; reject if < 0.3 V |
| w₂ = 0.20 | f₂ = Eᵒˣ(anion) - Eᵒˣ(monomer) | ≥ 0.3 V | Anion must survive at polymerization potential |
| w₃ = 0.20 | f₃ = -ΔGₛₒₗᵥ (normalized) | ≥ 5 kcal/mol | More negative = better solubility |
| w₄ = 0.15 | f₄ = -ΔG(dimerization) (normalized) | < -10 kcal/mol | Thermodynamic driving force for coupling |
| w₅ = 0.15 | f₅ = \|E<sub>g</sub> - E<sub>g,target</sub>\| penalty | Application-dependent | Penalize deviation from target band gap |

Weights are adjustable depending on the target application (e.g., bioelectronics may increase w₃ for aqueous solubility; photovoltaics may increase w₅ for band-gap control). All component functions should be min-max normalized to [0, 1] before summation.

## 6. Recommended Software Stack

| Task | Software | Purpose |
| --- | --- | --- |
| Structure generation | RDKit, stk (Python) | SMILES -> 3D, conformer search, oligomer assembly |
| Semiempirical (Tier 1) | xtb (Grimme group) | GFN2-xTB, IPEA-xTB, sTDA-xTB, GBSA solvation |
| DFT (Tier 2-3) | Gaussian 16, ORCA 5, or Turbomole | B3LYP, CAM-B3LYP, ωB97X-D; SMD solvation |
| Solvation models | COSMOtherm or openCOSMO-RS | COSMO-RS solvation free energies, solubility prediction |
| Workflow orchestration | AiiDA, Fireworks, or Snakemake | Job submission, provenance tracking, database storage |
| Data storage / analysis | ASE database, pandas, SQLite | Store geometries, energies; query and filter |
| Visualization | py3Dmol, matplotlib, seaborn | Chemical space maps, Pareto fronts |

## 7. Validation Against Experiment

Before deploying the screen, validate the computational pipeline against known experimental data.

- **Benchmark set.** Assemble ≥30 monomer-solvent-electrolyte combinations with experimentally measured oxidation potentials from cyclic voltammetry (literature values for EDOT, pyrrole, thiophene, aniline, etc., in ACN/TBAPF₆).
- **Accuracy targets.** Mean absolute error (MAE) for monomer Eᵒˣ: < 0.15 V at Tier 2 DFT, < 0.3 V at Tier 1 xTB after calibration. Solvent ESW: MAE < 0.3 V. Qualitative rank-ordering of polymerization feasibility (yes/no) should achieve > 85% accuracy.
- **Calibration protocol.** Following Zwijnenburg et al., fit linear models (slope, intercept) mapping xTB properties to DFT values using the benchmark set. Apply these corrections to all Tier 1 results before filtering. Re-fit if new functional groups outside the training domain are introduced.

## 8. Expected Outputs

- A ranked database of monomer-solvent-electrolyte triads with computed properties and composite scores.
- Pareto-optimal frontiers trading off electrochemical window margin vs. solubility vs. band gap.
- Identification of non-obvious solvent-electrolyte pairings for under-explored monomers (e.g., selenophenes or furans in ionic liquids).
- Chemical-space maps (e.g., t-SNE or UMAP on molecular fingerprints) colored by computed polymerization feasibility score.
- A shortlist of 20-50 triads recommended for experimental validation by cyclic voltammetry and electropolymerization trials.

## 9. Estimated Computational Resources

| Tier | # Calculations | Cost per Calc | Total |
| --- | ---: | ---: | ---: |
| Tier 1 (xTB) | ~10,000-50,000 | 1-10 s each | ~50-500 CPU-hours |
| Tier 2 (DFT) | ~500-2,000 | 0.5-2 h each | ~500-4,000 CPU-hours |
| Tier 3 (high-accuracy) | ~20-50 | 10-100 h each | ~500-5,000 CPU-hours |
| Total | - | - | ~1,500-10,000 CPU-hours |

These estimates assume a moderately sized monomer library (~100 monomers), ~25 solvents, and ~20 electrolytes. The full combinatorial space is ~50,000 triads, but Tier 1 filtering reduces Tier 2 scope dramatically. The study is feasible on a university-scale HPC cluster within 1-2 weeks of wall-clock time.

## 10. Key Literature

1. Wilbraham, L.; Berardo, E.; Tuber, L.; Sherwood, P.; Sherwood, P.; Sherwood, P.; Zwijnenburg, M. A. *J. Chem. Inf. Model.* **2018**, *58*, 2450-2459. (High-throughput xTB screening of conjugated polymers.)
2. Heath-Apostolopoulos, I.; Wilbraham, L.; Zwijnenburg, M. A. *Faraday Discuss.* **2019**, *215*, 98-110. (Photocatalyst screening, effect of sequence isomerism.)
3. McCormick, T. M. et al. *Macromolecules* **2013**, *46*, 3879-3886. (DFT benchmarking for polymer orbital energies.)
4. Ong, S. P. et al. *Chem. Mater.* **2011**, *23*, 2979-2986. (MD+DFT electrochemical windows of ionic liquids.)
5. Bhatt, M. D. et al. *Phys. Chem. Chem. Phys.* **2015**, *17*, 4799. (DFT screening of battery electrolyte redox stability.)
6. Holubowitch et al. *J. Comput. Chem.* **2026**. (DFT thermochemistry of thiophene electropolymerization.)
7. Hutchison, G. R.; Ratner, M. A.; Marks, T. J. *J. Phys. Chem. A* **2002**, *106*, 10596-10605. (Semiempirical band-gap prediction.)
8. *Chemical Reviews* **2025** (Electropolymerization of OMIECs: fundamentals and bioelectronics applications.)
