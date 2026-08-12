# MACE-POLAR-1-L five-solvent pilot

- Status: **FAIL**
- Scope: 4 systems, 8 state trajectories, one shell seed each
- Scheduler jobs: trajectory `423770, 423773`, gap `423774`
- Summed wallclock: trajectory 19.30 h; gap 0.20 h
- Raw pilot storage: 60.52 MiB
- Critical QC flags: shell_escape

## Integrity checks

- PASS: finite_energy_force_temperature
- PASS: charge_spin_propagation
- PASS: three_restart_chunks_per_trajectory
- PASS: same_coordinate_two_state_gap_batches
- PASS: exact_five_solvent_clusters
- PASS: molecule_count_and_atom_order_qc
- FAIL: no_immediate_fragmentation_or_shell_loss
- PASS: restraint_force_conservation_and_state_cancellation_tests
