"""Paired, process-isolated benchmark of ordinary SQD and SQD+CIPSI."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

METHODS = ('sqd', 'sqd_cipsi')
METRICS = ('wall_seconds', 'cpu_seconds', 'peak_rss_mib', 'absolute_error_mEh',
           'signed_error_mEh', 'dimension', 'effective_determinants',
           'spin_squared', 'projected_residual_norm_Eh')


def summarize(rows):
    """Aggregate successful runs and compare only matching seed pairs."""
    summary = {}
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        summary[method] = {'runs': len(group)}
        for key in METRICS:
            values = [r[key] for r in group]
            summary[method][key] = {
                'mean': statistics.mean(values) if values else None,
                'sample_std': statistics.stdev(values) if len(values) > 1 else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
            }
    pairs = []
    for seed in sorted({r['seed'] for r in rows}):
        by_method = {r['method']: r for r in rows if r['seed'] == seed}
        if len(by_method) != 2:
            continue
        normal, enhanced = (by_method[m] for m in METHODS)
        pairs.append({
            'seed': seed,
            'sqd_start_energy_difference_Eh': enhanced['sqd_total_energy_Eh'] - normal['sqd_total_energy_Eh'],
            'absolute_error_reduction_mEh': normal['absolute_error_mEh'] - enhanced['absolute_error_mEh'],
            'energy_lowering_mEh': (normal['total_energy_Eh'] - enhanced['total_energy_Eh']) * 1000,
            'runtime_ratio_cipsi_over_sqd': enhanced['wall_seconds'] / normal['wall_seconds'],
            'extra_wall_seconds': enhanced['wall_seconds'] - normal['wall_seconds'],
            'dimension_ratio': enhanced['dimension'] / normal['dimension'],
        })
    return {'by_method': summary, 'paired_comparisons': pairs}


def worker(args):
    # These variables are set in the parent before any numerical imports.
    import resource
    import numpy as np
    from pyscf import fci, lib
    from qiskit.primitives import BitArray
    from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
    from functools import partial
    from cipsi_solver import refine_cipsi

    lib.num_threads(args.threads)
    with np.load(args.data / 'samples.npz', allow_pickle=False) as saved:
        samples = BitArray(saved['array'], num_bits=int(saved['num_bits']))
    with np.load(args.data / 'hamiltonian.npz', allow_pickle=False) as saved:
        hcore, eri = saved['hcore'], saved['eri']
        norb, nelec = int(saved['norb']), tuple(map(int, saved['nelec']))
        core, reference = float(saved['core_energy']), float(saved['reference_energy'])
    if samples.num_bits != 2 * norb or not math.isfinite(reference):
        raise ValueError('Invalid sample width or CASCI reference')
    curve = []
    best_energy = float('inf')
    sqd_iterations = 0
    max_batch_dimension = 0
    wall_start, cpu_start = time.perf_counter(), time.process_time()

    def sqd_callback(results):
        nonlocal best_energy, sqd_iterations, max_batch_dimension
        sqd_iterations += 1
        best_energy = min(best_energy, min(float(r.energy) + core for r in results))
        max_batch_dimension = max(max_batch_dimension, max(r.sci_state.amplitudes.size for r in results))
        curve.append({'stage': 'sqd', 'iteration': sqd_iterations,
                      'wall_seconds': time.perf_counter() - wall_start,
                      'total_energy_Eh': best_energy,
                      'absolute_error_mEh': abs(best_energy - reference) * 1000})
        print(f'{args.method} seed {args.seed}: SQD {sqd_iterations}, E={best_energy:.12f}', flush=True)

    result = diagonalize_fermionic_hamiltonian(hcore, eri, samples,
        args.samples_per_batch, norb, nelec, num_batches=args.num_batches,
        max_iterations=args.sqd_iterations, energy_tol=1e-4, occupancies_tol=1e-4,
        sci_solver=partial(solve_sci_batch, spin_sq=0.0, max_cycle=200),
        symmetrize_spin=True, carryover_threshold=1e-5, seed=args.seed, callback=sqd_callback)
    sqd_seconds = time.perf_counter() - wall_start
    sqd_cpu_seconds = time.process_time() - cpu_start
    sqd_energy = float(result.energy) + core
    sqd_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    cipsi_history = []
    cipsi_seconds = cipsi_cpu_seconds = 0.0
    if args.method == 'sqd_cipsi':
        def cipsi_callback(updated, record):
            energy = float(updated.energy) + core
            curve.append({'stage': 'cipsi', 'iteration': record['iteration'],
                          'wall_seconds': time.perf_counter() - wall_start,
                          'total_energy_Eh': energy,
                          'absolute_error_mEh': abs(energy - reference) * 1000})
            print(f'CIPSI {record["iteration"]}, E={energy:.12f}', flush=True)

        def progress(block, total):
            if block == 1 or block % 20 == 0 or block == total:
                print(f'CIPSI candidate block {block}/{total}', flush=True)

        cipsi_wall_start, cipsi_cpu_start = time.perf_counter(), time.process_time()
        result, cipsi_history = refine_cipsi(result, hcore, eri, norb, nelec,
            iterations=args.cipsi_iterations, keep=args.cipsi_keep,
            beta_block_size=args.beta_block_size, callback=cipsi_callback, progress=progress)
        cipsi_seconds = time.perf_counter() - cipsi_wall_start
        cipsi_cpu_seconds = time.process_time() - cipsi_cpu_start
    wall_seconds = sqd_seconds + cipsi_seconds
    cpu_seconds = sqd_cpu_seconds + cipsi_cpu_seconds
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is process-lifetime peak RSS, not a phase-local allocation count.
    rss_divisor = 1024 * 1024 if sys.platform == 'darwin' else 1024
    state = result.sci_state
    coeff = np.asarray(state.amplitudes)
    norm = float(np.linalg.norm(coeff))
    probabilities = np.abs(coeff.ravel()) ** 2 / norm**2
    # Diagnostic operations below are outside measured solver runtime and peak RSS.
    h2 = fci.direct_spin1.absorb_h1e(hcore, eri, norb, nelec, 0.5)
    ci = fci.selected_ci._as_SCIvector(coeff, (state.ci_strs_a, state.ci_strs_b))
    projected_hc = np.asarray(fci.selected_ci.contract_2e(h2, ci, norb, nelec))
    residual_norm = float(np.linalg.norm(projected_hc - result.energy * coeff) / norm)
    energy = float(result.energy) + core
    reaches_target = [p['wall_seconds'] for p in curve if p['absolute_error_mEh'] <= args.target_error]
    row = {
        'method': args.method, 'seed': args.seed, 'shots': samples.num_shots,
        'norb': norb, 'nelec': list(nelec), 'reference_energy_Eh': reference,
        'sqd_total_energy_Eh': sqd_energy, 'total_energy_Eh': energy,
        'signed_error_mEh': (energy - reference) * 1000,
        'absolute_error_mEh': abs(energy - reference) * 1000,
        'wall_seconds': wall_seconds, 'cpu_seconds': cpu_seconds,
        'sqd_wall_seconds': sqd_seconds, 'cipsi_wall_seconds': cipsi_seconds,
        'sqd_cpu_seconds': sqd_cpu_seconds, 'cipsi_cpu_seconds': cipsi_cpu_seconds,
        'peak_rss_mib': peak / rss_divisor, 'peak_rss_after_sqd_mib': sqd_peak / rss_divisor,
        'dimension': int(coeff.size), 'alpha_strings': len(state.ci_strs_a),
        'beta_strings': len(state.ci_strs_b), 'max_sqd_batch_dimension': int(max_batch_dimension),
        'effective_determinants': float(1 / np.sum(probabilities**2)),
        'coefficient_norm': norm, 'spin_squared': float(state.spin_square()),
        'projected_residual_norm_Eh': residual_norm,
        'sqd_iterations_completed': sqd_iterations,
        'cipsi_iterations_completed': len(cipsi_history),
        'target_error_mEh': args.target_error,
        'first_recorded_target_seconds': min(reaches_target) if reaches_target else None,
        'curve': curve, 'cipsi_history': cipsi_history,
    }
    args.worker_output.write_text(json.dumps(row, indent=2, allow_nan=False))


def make_report(output, rows, failures, metadata):
    results = summarize(rows)
    (output / 'summary.json').write_text(json.dumps(results, indent=2, allow_nan=False))
    scalar_rows = [{k: v for k, v in r.items() if k not in ('curve', 'cipsi_history', 'nelec')} for r in rows]
    if scalar_rows:
        with (output / 'results.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scalar_rows[0]))
            writer.writeheader()
            writer.writerows(scalar_rows)
    lines = ['# Ordinary SQD versus SQD+CIPSI', '',
        'Status: ' + metadata.get('status', 'unknown'), '',
        'Same saved samples, Hamiltonian, seed and SQD settings per pair. Each method runs in a fresh process.',
        'SQD+CIPSI includes the initial SQD cost. This is an added-refinement comparison, not equal time or equal determinant budget.', '',
        '| Method | Seed | Total energy (Eh) | Absolute error (mEh) | Wall (s) | CPU (s) | Peak RSS (MiB) | Determinants |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f'| {r["method"]} | {r["seed"]} | {r["total_energy_Eh"]:.10f} | '
                     f'{r["absolute_error_mEh"]:.4f} | {r["wall_seconds"]:.3f} | '
                     f'{r["cpu_seconds"]:.3f} | {r["peak_rss_mib"]:.1f} | {r["dimension"]} |')
    lines += ['', '## Paired differences', '', '| Seed | Error reduction (mEh) | Runtime ratio (CIPSI / SQD) | SQD starting-energy mismatch (Eh) |',
              '|---|---:|---:|---:|']
    for pair in results['paired_comparisons']:
        lines.append(f'| {pair["seed"]} | {pair["absolute_error_reduction_mEh"]:.4f} | '
                     f'{pair["runtime_ratio_cipsi_over_sqd"]:.3f} | {pair["sqd_start_energy_difference_Eh"]:.3e} |')
    lines += ['', '## Interpretation', '',
        '- Runtime measures the solver stages, including lightweight progress callbacks. Imports, input loading, final diagnostics and result writing are excluded.',
        '- Peak RSS is process-lifetime high-water memory through solver completion, including imports and input loading; it is not incremental CIPSI memory.',
        '- CPU time sums CPU use by process threads; it can exceed wall time.',
        '- Accuracy is against the supplied CASCI reference in the same saved active space. No CASCI runtime is measured.',
        '- Seeds vary configuration recovery and subsampling of one fixed hardware dataset, not independent hardware acquisitions.',
        '- Projected residual is computed with the physical Hamiltonian within the selected space. It can reflect the singlet penalty as well as solver error; it is not a full-space residual or energy variance.',
        '- Finite iteration caps do not establish convergence. This report compares the configured runs only.',
        '- Effective determinants = 1 / sum(normalized CI probabilities squared). Spin squared should be near zero for the targeted singlet.',
        '- First recorded target time is measured only at completed iterations. A null value means the requested error was not reached.',
        '- CIPSI expands Cartesian spin-string spaces; the number of added determinants may exceed the number of selected pairs.',
        '- Per-method means and sample standard deviations are in summary.json. One run cannot establish variability.',
        '- PT2 is a pre-expansion diagnostic and is not added to the reported variational energy.', '',
        '## Run settings', '', '```json', json.dumps(metadata, indent=2), '```']
    if failures:
        lines += ['', '## Incomplete runs', '', *[f'- {f}' for f in failures]]
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    if rows:
        plot_results(output, rows)


def plot_results(output, rows):
    os.environ.setdefault('MPLCONFIGDIR', str(output.resolve() / '.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    colors = {'sqd': '#2864b4', 'sqd_cipsi': '#c55c18'}
    for method in METHODS:
        group = [r for r in rows if r['method'] == method]
        if not group:
            continue
        color = colors[method]
        axes[0, 0].scatter([r['wall_seconds'] for r in group], [r['absolute_error_mEh'] for r in group], label=method, color=color)
        for index, r in enumerate(group):
            offset = [(0, -18), (-22, 16), (-30, 2)][index % 3]
            axes[0, 0].annotate(str(r['seed']), (r['wall_seconds'], r['absolute_error_mEh']),
                                xytext=offset, textcoords='offset points', fontsize=8,
                                arrowprops={'arrowstyle': '-', 'lw': .5}, annotation_clip=False)
        for ax, key in [(axes[0, 1], 'wall_seconds'), (axes[1, 0], 'peak_rss_mib'), (axes[1, 1], 'dimension')]:
            x = METHODS.index(method)
            values = [r[key] for r in group]
            ax.bar(x, statistics.mean(values), width=.55, color=color, alpha=.5)
            ax.scatter([x] * len(values), values, color=color, s=25)
            ax.set_xticks(range(2), ['SQD', 'SQD+CIPSI'])
    axes[0, 0].set(xlabel='Solver wall time (s)', ylabel='Absolute CASCI error (mEh)', title='Accuracy versus runtime (labels are seeds)')
    axes[0, 0].set_xscale('log')
    axes[0, 0].set_xlabel('Solver wall time (s, log scale)')
    axes[0, 0].margins(x=.18, y=.15)
    axes[0, 0].legend()
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_ylim(min(r['wall_seconds'] for r in rows) * .5, max(r['wall_seconds'] for r in rows) * 2)
    axes[0, 1].set(ylabel='Seconds (log scale)', title='Solver runtime: means and individual seeds')
    axes[1, 0].set(ylabel='MiB', title='Process peak RSS through solve')
    axes[1, 1].set(ylabel='Cartesian determinants', title='Final variational space')
    for ax in axes.ravel():
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in rows:
        ax.plot([p['wall_seconds'] for p in r['curve']], [p['absolute_error_mEh'] for p in r['curve']],
                'o-', drawstyle='steps-post', color=colors[r['method']], alpha=.65, label=f'{r["method"]}, seed {r["seed"]}')
    ax.set(xlabel='Elapsed solver wall time (s)', ylabel='Absolute CASCI error (mEh)', title='Best recorded energy error (updates at completed iterations)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'convergence.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'n2test')
    parser.add_argument('--output', type=Path, default=Path('benchmark_results'))
    parser.add_argument('--seeds', type=int, nargs='+', default=[23, 24, 25])
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--samples-per-batch', type=int, default=30)
    parser.add_argument('--num-batches', type=int, default=3)
    parser.add_argument('--sqd-iterations', type=int, default=10)
    parser.add_argument('--cipsi-iterations', type=int, default=1)
    parser.add_argument('--cipsi-keep', type=int, default=25)
    parser.add_argument('--beta-block-size', type=int, default=32)
    parser.add_argument('--target-error', type=float, default=1.0, help='Target absolute CASCI error in mEh')
    parser.add_argument('--method', choices=METHODS, help=argparse.SUPPRESS)
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    parser.add_argument('--worker-output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.threads, args.samples_per_batch, args.num_batches, args.sqd_iterations,
           args.cipsi_iterations, args.cipsi_keep, args.beta_block_size) <= 0:
        parser.error('counts must be positive')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('seeds must be unique and non-negative')
    if not math.isfinite(args.target_error) or args.target_error <= 0:
        parser.error('target-error must be finite and positive')
    if args.method:
        worker(args)
        return
    from importlib.metadata import version
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'status': 'running',
        'settings': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k not in ('method', 'seed', 'worker_output')},
        'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'python': sys.version, 'platform': platform.platform(), 'logical_cpus': os.cpu_count(),
        'packages': {p: version(p) for p in ('numpy', 'pyscf', 'qiskit', 'qiskit-addon-sqd', 'matplotlib')},
        'input_sha256': {name: hashlib.sha256((args.data / name).read_bytes()).hexdigest() for name in ('samples.npz', 'hamiltonian.npz')},
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('benchmark.py', 'cipsi_solver.py', 'cipsi.py')},
    }
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = str(args.threads)
    rows, failures = [], []
    for index, seed in enumerate(args.seeds):
        # Alternate order to reduce systematic order effects across seeds.
        methods = METHODS if index % 2 == 0 else METHODS[::-1]
        for method in methods:
            print(f'Starting {method}, seed {seed}', flush=True)
            stem = f'{method}_seed{seed}'
            destination = args.output / f'{stem}.json'
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--method', method,
                       '--seed', str(seed), '--worker-output', str(destination), '--data', str(args.data)]
            for key in ('threads', 'samples_per_batch', 'num_batches', 'sqd_iterations', 'cipsi_iterations', 'cipsi_keep', 'beta_block_size', 'target_error'):
                command += ['--' + key.replace('_', '-'), str(getattr(args, key))]
            launched = time.perf_counter()
            with (args.output / f'{stem}.log').open('w') as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                except KeyboardInterrupt:
                    metadata['status'] = 'interrupted'
                    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
                    failures.append(f'{stem}: interrupted; see {stem}.log')
                    make_report(args.output, rows, failures, metadata)
                    raise SystemExit(130)
            if process.returncode:
                failures.append(f'{stem}: exit {process.returncode}; see {stem}.log')
                print(f'FAILED: {failures[-1]}', flush=True)
            else:
                row = json.loads(destination.read_text())
                row['process_elapsed_seconds'] = time.perf_counter() - launched
                destination.write_text(json.dumps(row, indent=2, allow_nan=False))
                rows.append(row)
                print(f'Finished: {row["absolute_error_mEh"]:.4f} mEh error, {row["wall_seconds"]:.2f} s solver time', flush=True)
            make_report(args.output, rows, failures, metadata)
    metadata['status'] = 'failed' if failures else 'completed'
    (args.output / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    make_report(args.output, rows, failures, metadata)
    print(f'Report: {args.output / "report.md"}', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
