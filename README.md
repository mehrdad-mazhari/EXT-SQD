# Sample-based quantum diagonalization with CIPSI refinement

Research code exploring hybrid quantum–classical ground-state calculations:

- **Molecules:** recover a selected configuration-interaction space from sampled bitstrings, then expand it using Epstein–Nesbet CIPSI importance scores.
- **HfO₂:** run SQD on an 18-orbital extended Hubbard model, using a CCSD-derived UCJ circuit and either IBM hardware or approximate local MPS sampling.

Read [PRESENTATION.md](PRESENTATION.md) for the step-by-step program walkthrough and a suggested presentation sequence.

## Installation

Use Python 3.11 or newer with a version supported by the pinned dependencies. The existing environment was used for validation; a fresh installation has not been verified.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

For molecular circuit preparation, IBM sampling, and the local HfO₂ simulator:

```bash
.venv/bin/python -m pip install -r requirements-quantum.txt
```

## Run locally with saved N₂ samples

From the repository root:

```bash
.venv/bin/python sqd+cipsi/test.py --output local_results
```

This performs SQD and one CIPSI expansion. Outputs must go into a new directory. For a smaller SQD-only workflow check:

```bash
.venv/bin/python sqd+cipsi/test.py --sqd-iterations 1 --cipsi-iterations 0 --threads 2 --output local_results_smoke
```

**Dataset limitation:** `sqd+cipsi/n2test/job.json` states that the Hamiltonian was rebuilt and historical orbital identity is not verified. The saved-data workflow is usable for experimentation, but physical claims require verifying that samples and tensors share the same orbital basis.

## Compare methods

```bash
.venv/bin/python sqd+cipsi/benchmark.py --output benchmark_results
```

The benchmark compares ordinary SQD and SQD+CIPSI for seeds 23, 24, and 25 in separate processes. Reports include accuracy against the stored CASCI reference, runtime, memory and selected-space size. Refinement time includes its initial SQD cost. See [the molecular README](sqd+cipsi/README.md) for metric definitions and options.

## Quantum sampling

Molecular hardware execution prepares H₂ by default; N₂ and CO are also available:

```bash
.venv/bin/python sqd+cipsi/sqd.py --molecule H2 --backend ibm_marrakesh
```

This command submits a new 1,000-shot IBM job. Configure IBM Runtime credentials outside the repository and select a backend accessible to your account. Importing the entry-point modules does not submit a job.

Local HfO₂ workflow check:

```bash
cd duriez
OMP_NUM_THREADS=2 MPLBACKEND=Agg ../.venv/bin/python sqd.py --local
```

Without `--local`, the HfO₂ script submits a new IBM job. Local MPS sampling caps bond dimension at 64; the saved result is not a converged material prediction. See [the HfO₂ README](duriez/README.md).

## Validation

```bash
.venv/bin/python -m unittest discover -s sqd+cipsi -p 'test_*.py' -v
```

Tests cover CIPSI scoring, blocked external-space contraction against full-CI residuals, expanded-space recovery of full CI on a small model, and benchmark aggregation.

## Files and publication

| Path | Purpose |
| --- | --- |
| `sqd+cipsi/test.py` | Local saved-sample SQD+CIPSI entry point |
| `sqd+cipsi/cipsi.py` | Representation-independent importance scoring |
| `sqd+cipsi/cipsi_solver.py` | External selection and rediagonalization |
| `sqd+cipsi/benchmark.py` | Paired comparison and reports |
| `sqd+cipsi/H2.py`, `N2.py`, `CO.py` | On-demand molecular preparation via `prepare()` |
| `sqd+cipsi/n2test/` | Saved samples, tensors and historical provenance |
| `sqd+cipsi/run_n2/` | Additional archived Hamiltonian and source snapshot |
| `duriez/hubbard.py` | HfO₂ parameters, tensor conventions and preparation |
| `duriez/sqd.py` | HfO₂ sampling and SQD |
| `duriez/data/hfo2/` | Upstream hopping data, provenance and license |

Historical `molecule_source.py` snapshots are retained for provenance and are not entry points. The saved HfO₂ result and convergence plot are retained; generated logs, virtual environments and default result directories are ignored.

The HfO₂ data includes its upstream Apache-2.0 license. No project-wide license has been selected for the original code. Choose one before offering the code for reuse. This folder is the repository root for upload. No GitHub upload has been made.
