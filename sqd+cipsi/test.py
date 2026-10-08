"""Run local SQD, then external-space CIPSI, using saved IBM samples."""
import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'n2test')
    parser.add_argument('--output', type=Path, default=Path('local_results'))
    parser.add_argument('--samples-per-batch', type=int, default=30)
    parser.add_argument('--num-batches', type=int, default=3)
    parser.add_argument('--sqd-iterations', type=int, default=10)
    parser.add_argument('--cipsi-iterations', type=int, default=1)
    parser.add_argument('--cipsi-keep', type=int, default=25)
    parser.add_argument('--beta-block-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.samples_per_batch, args.num_batches, args.sqd_iterations,
           args.cipsi_keep, args.beta_block_size, args.threads) <= 0 or args.cipsi_iterations < 0:
        parser.error('counts must be positive; cipsi-iterations may be zero')
    from pyscf import lib
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
    from cipsi_solver import refine_cipsi
    lib.num_threads(args.threads)
    with np.load(args.data / 'samples.npz', allow_pickle=False) as data:
        samples = BitArray(data['array'], num_bits=int(data['num_bits']))
    with np.load(args.data / 'hamiltonian.npz', allow_pickle=False) as data:
        hcore, eri = data['hcore'], data['eri']
        norb, nelec = int(data['norb']), tuple(map(int, data['nelec']))
        core, reference = float(data['core_energy']), float(data['reference_energy'])
    if samples.num_bits != 2 * norb:
        raise ValueError('Saved sample width does not match Hamiltonian')
    args.output.mkdir(parents=True, exist_ok=False)
    history = []

    def save_state(result, filename):
        state = result.sci_state
        np.savez_compressed(args.output / filename, energy=result.energy,
            amplitudes=state.amplitudes, ci_strs_a=state.ci_strs_a, ci_strs_b=state.ci_strs_b)

    def record_sqd(results):
        history.append([float(r.energy) + core for r in results])
        (args.output / 'sqd_history.json').write_text(json.dumps(history, indent=2))
        print(f'SQD iteration {len(history)}: best total energy {min(history[-1]):.12f} Eh', flush=True)

    print(f'Loaded {samples.num_shots} saved shots; running local SQD.', flush=True)
    result = diagonalize_fermionic_hamiltonian(hcore, eri, samples,
        args.samples_per_batch, norb, nelec, num_batches=args.num_batches,
        max_iterations=args.sqd_iterations, energy_tol=1e-4, occupancies_tol=1e-4,
        sci_solver=partial(solve_sci_batch, spin_sq=0.0, max_cycle=200),
        symmetrize_spin=True, carryover_threshold=1e-5, seed=23, callback=record_sqd)
    save_state(result, 'sqd_state.npz')
    sqd_energy = float(result.energy) + core
    cipsi_history = []

    def record_cipsi(state, record):
        cipsi_history.append(record)
        save_state(state, 'cipsi_state.npz')
        (args.output / 'cipsi_history.json').write_text(json.dumps(cipsi_history, indent=2))
        print(f'CIPSI iteration {record["iteration"]}: total energy {state.energy + core:.12f} Eh', flush=True)

    def progress(block, total):
        if block == 1 or block == total or block % 10 == 0:
            print(f'CIPSI candidate block {block}/{total}', flush=True)

    result, _ = refine_cipsi(result, hcore, eri, norb, nelec,
        iterations=args.cipsi_iterations, keep=args.cipsi_keep,
        beta_block_size=args.beta_block_size, callback=record_cipsi, progress=progress)
    summary = {'sqd_total_energy': sqd_energy, 'final_total_energy': float(result.energy) + core,
               'casci_total_energy': reference, 'cipsi_iterations_completed': len(cipsi_history),
               'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
