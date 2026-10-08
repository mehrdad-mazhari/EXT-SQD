# Local SQD + CIPSI

`test.py` loads the saved IBM bitstrings and matching Hamiltonian in `n2test/`,
runs SQD locally, then performs external-space CIPSI refinement. It needs no
IBM account, cloud machine, VPN, or new sampling job.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -u test.py --output local_results
```

Output directories must be new. SQD uses 30 samples per batch, three batches,
up to ten recovery iterations, and four PySCF threads by default. Adjust
`--threads` for your CPU. `--cipsi-iterations 0` runs just SQD.

`cipsi.py` computes Epstein--Nesbet scores. `cipsi_solver.py` generates all
same-spin single/double substitutions of the variational strings, evaluates
external residuals with the full SQD state, selects 25 external determinant
pairs by default, and rediagonalizes. The alpha/beta Cartesian candidate pool
covers all connected single/double excitations; disconnected entries contribute
zero. Beta blocks bound the contraction array size without pruning candidates.

Selection adds alpha and beta strings to SQD's Cartesian representation,
including cross-pairs and spin-exchanged strings. Thus 25 selected pairs may
add more than 25 variational determinants. This is Cartesian-space CIPSI
refinement, not a minimal arbitrary-determinant CIPSI implementation. Refinement
runs after SQD recovery, not inside each recovery batch. The PT2 diagnostic is
for the state before each expansion, uses regularized denominators, and is not
added to the final variational energy.

Full external scoring can be expensive on N2. The code stops rather than
silently pruning if the configured working or variational dimension cap is
exceeded. A dimension cap bounds array entries, not total process RAM. Reduce
`--beta-block-size` if needed. Defaults perform one CIPSI growth iteration;
use `--cipsi-iterations` and `--cipsi-keep` to adjust it.

Results include SQD energy history and state, CIPSI history and state (when an
expansion occurs), and a final JSON summary. SQD is checkpointed before CIPSI.
The saved Hamiltonian must match the original sampled orbital basis.

The original hardware example `sqd.py` and molecule modules remain available;
`sqd.py` still submits a new IBM job if executed. Cloud transfer scripts and
archive have been removed. Saved data in `n2test/` and `run_n2/` are retained.

The saved metadata states that historical orbital identity is unverified. Validate
the orbital basis before treating saved-data energies as physical benchmarks.

Molecular model modules now expose `prepare()` rather than running on import.
Use `python sqd.py --molecule H2 --backend ibm_marrakesh` for hardware sampling.

Tests:
```bash
.venv/bin/python -m unittest test_cipsi test_cipsi_solver test_benchmark -v
```

## Compare ordinary SQD with SQD+CIPSI

```bash
.venv/bin/python -u benchmark.py --output benchmark_results
```

The default compares seeds 23, 24, and 25, with the same saved samples and
identical SQD settings for each pair. Each method runs sequentially in a fresh
process. SQD+CIPSI includes the cost of its starting SQD calculation. Method
order alternates across seeds. This measures the benefit/cost of refinement;
it does not claim equal runtime or determinant budgets.

Options include `--seeds 23 24 25`, `--threads 4`, `--samples-per-batch 30`,
`--num-batches 3`, `--sqd-iterations 10`, `--cipsi-iterations 1`,
`--cipsi-keep 25`, `--beta-block-size 32`, and `--target-error 1.0` (mEh).

Outputs:
- `report.md`: per-run comparison, paired differences, definitions and settings.
- `results.csv`: energies, signed/absolute CASCI error, wall/CPU times, peak RSS,
  final Cartesian dimension, effective determinants, spin squared, projected
  residual norm, iteration counts and first recorded target time.
- `summary.json`: per-method mean/sample standard deviation and paired changes.
- `comparison.png`, `convergence.png`: runtime/accuracy/memory/space comparisons
  and recorded energy-error histories.
- Per-method/seed JSON and logs: stage times, CIPSI selection/diagonalization
  times, curves and PT2 diagnostics. Failures are listed rather than hidden.
- `metadata.json`: package versions, input/code hashes and machine/settings.

Wall/CPU times exclude imports, loading, final diagnostics and output writes;
they include lightweight progress callbacks. RSS is the process-lifetime peak
through solver completion (including imports/loading), not incremental memory.
A projected residual is not a full-space variance. Seeds describe variation
within this hardware dataset, not independent hardware acquisitions. The
stored CASCI energy is the reference; its computation is not timed.

The projected residual uses the physical Hamiltonian. Since these runs also use
PySCF's singlet spin penalty, a nonzero physical-Hamiltonian residual can reflect
that penalty in an incomplete selected space, as well as eigensolver error.
Interpret it together with spin squared; it is not a standalone convergence flag.

For an initial three-seed comparison with a smaller recovery budget:
```bash
.venv/bin/python -u benchmark.py --sqd-iterations 1 --output benchmark_results_quick
```
This caps recovery at one iteration for both methods, then adds the configured
CIPSI refinement only to SQD+CIPSI. It is a limited-budget comparison, not a
claim that either method has converged. The full default remains ten SQD
iterations. Completed rows survive interruptions; incomplete runs are labeled.
