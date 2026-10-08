# HfO2 SQD

`sqd.py` now uses the HfO2 extended Hubbard input in `hubbard.py` instead of
molecular H2 integrals. It prepares RHF and CCSD in the lattice basis, rotates
the Hamiltonian into the RHF orbital basis, and builds a single-layer UCJ
state. SQD tensors and circuit amplitudes therefore use the same basis.
The neutral model has 18 spatial orbitals, 36 qubits, and (12,12) electrons.
Energies are in eV and the nuclear energy offset is zero.

See `data/hfo2/SOURCE.md` for provenance and the explicit onsite tensor
convention. The default matches the paper repository's preprocessing.

Run from this folder with an environment containing numpy, pyscf, ffsim,
matplotlib, qiskit, qiskit-ibm-runtime, and qiskit-addon-sqd:

```bash
python sqd.py
```

The script submits 1000 shots to the configured IBM backend using saved
Runtime credentials. The retained 350-sample batches are a small experiment;
they are not the paper's converged SQD/Ext-SQD settings. No full FCI reference
is computed for this large model. The script computes the neutral ground
state only; reproducing the band gap requires separate 23- and 25-electron
calculations and E(23)+E(25)-2E(24), including an open-shell circuit workflow.

For a local smoke run (requires qiskit-aer):

```bash
OMP_NUM_THREADS=2 MPLBACKEND=Agg python sqd.py --local
```

Local sampling uses an approximate MPS with bond dimension capped at 64,
50 samples per batch, and three recovery iterations.
Truncation can cause substantial particle-number leakage; use the result only
as a workflow check, not as a converged material energy. Outputs include
`hfo2_local_counts.json`, `hfo2_run_result.json`, and the convergence plot.
