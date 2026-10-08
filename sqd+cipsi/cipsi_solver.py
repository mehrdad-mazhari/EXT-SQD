"""External-space Epstein--Nesbet selection followed by SQD rediagonalization.

Uses Cartesian alpha/beta spaces, as required by SQD's PySCF solver. Adding a
selected pair also includes Cartesian cross-pairs. All single/double connected
external determinants are scored, with beta blocks limiting working memory.
"""
from itertools import combinations
from time import perf_counter

import numpy as np
from pyscf import fci
from qiskit_addon_sqd.fermion import solve_sci_batch

from cipsi import cipsi_scores_from_residuals


def excitation_strings(strings, norb):
    """Union of same-spin zero, single and double substitutions."""
    result = set(map(int, strings))
    for raw in strings:
        value = int(raw)
        occupied = [i for i in range(norb) if value >> i & 1]
        virtual = [i for i in range(norb) if not value >> i & 1]
        for degree in (1, 2):
            for holes in combinations(occupied, degree):
                removed = value ^ sum(1 << i for i in holes)
                for particles in combinations(virtual, degree):
                    result.add(removed | sum(1 << i for i in particles))
    return np.array(sorted(result), dtype=np.int64)


def select_external(result, hcore, eri, norb, nelec, *, keep=25,
                    beta_block_size=32, max_working_dim=2_000_000,
                    level_shift=1e-6, score_threshold=1e-12, progress=None):
    """Return top external pairs and full regularized external-space PT2.

    Each block embeds the entire variational state, so residuals include every
    variational coefficient. Existing determinants are masked out. Candidate
    rectangles may contain disconnected determinants, which have zero residual.
    No coefficients or connected candidates are pruned.
    """
    if keep <= 0 or beta_block_size <= 0 or max_working_dim <= 0:
        raise ValueError('keep, beta_block_size and max_working_dim must be positive')
    if not np.isfinite(level_shift) or level_shift <= 0:
        raise ValueError('level_shift must be finite and positive')
    if not np.isfinite(score_threshold) or score_threshold < 0:
        raise ValueError('score_threshold must be finite and non-negative')
    state = result.sci_state
    old_a = np.asarray(state.ci_strs_a, dtype=np.int64)
    old_b = np.asarray(state.ci_strs_b, dtype=np.int64)
    pool_a = excitation_strings(old_a, norb)
    pool_b = excitation_strings(old_b, norb)
    new_b = np.setdiff1d(pool_b, old_b)
    if len(pool_a) * (len(old_b) + min(beta_block_size, len(new_b))) > max_working_dim:
        raise ValueError('CIPSI working space exceeds max_working_dim; reduce beta_block_size or increase the explicit memory limit')
    h2 = fci.direct_spin1.absorb_h1e(hcore, eri, norb, nelec, 0.5)
    ia = np.searchsorted(pool_a, old_a)
    # First score external alpha strings with original beta strings; subsequent
    # blocks score new beta strings only. Thus every external pair appears once.
    blocks = [old_b] + [new_b[i:i + beta_block_size] for i in range(0, len(new_b), beta_block_size)]
    best = []
    pt2 = 0.0
    for block_index, targets in enumerate(blocks):
        work_b = np.union1d(old_b, targets)
        ib = np.searchsorted(work_b, old_b)
        ci = np.zeros((len(pool_a), len(work_b)))
        ci[np.ix_(ia, ib)] = state.amplitudes
        sci_ci = fci.selected_ci._as_SCIvector(ci, (pool_a, work_b))
        residual = np.asarray(fci.selected_ci.contract_2e(h2, sci_ci, norb, nelec))
        diagonal = fci.selected_ci.make_hdiag(hcore, eri, (pool_a, work_b), norb, nelec).reshape(ci.shape)
        target_columns = np.searchsorted(work_b, targets)
        r = residual[:, target_columns]
        d = diagonal[:, target_columns]
        scores = cipsi_scores_from_residuals(result.energy, d.ravel(), r.ravel(), level_shift=level_shift)
        mask = np.ones(r.shape, dtype=bool)
        if block_index == 0:
            mask[ia, :] = False
        eligible = np.flatnonzero(mask.ravel())
        pt2 += float(scores.pt2_contributions[eligible].sum())
        eligible = eligible[scores.scores[eligible] > score_threshold]
        order = eligible[np.argsort(-scores.scores[eligible], kind='stable')[:keep]]
        for index in order:
            a, b = divmod(int(index), len(targets))
            best.append((float(scores.scores[index]), int(pool_a[a]), int(targets[b])))
        best = sorted(best, key=lambda x: (-x[0], x[1], x[2]))[:keep]
        if progress:
            progress(block_index + 1, len(blocks))
    return best, pt2


def refine_cipsi(result, hcore, eri, norb, nelec, *, iterations=1, keep=25,
                 beta_block_size=32, max_working_dim=2_000_000,
                 max_variational_dim=250_000, spin_sq=0.0,
                 callback=None, progress=None):
    """Grow an existing SQD result while retaining every old determinant."""
    if iterations < 0:
        raise ValueError('iterations must be non-negative')
    history = []
    for iteration in range(iterations):
        selection_started = perf_counter()
        selected, pt2 = select_external(result, hcore, eri, norb, nelec,
            keep=keep, beta_block_size=beta_block_size,
            max_working_dim=max_working_dim, progress=progress)
        selection_seconds = perf_counter() - selection_started
        if not selected:
            break
        a = np.union1d(result.sci_state.ci_strs_a, [x[1] for x in selected])
        b = np.union1d(result.sci_state.ci_strs_b, [x[2] for x in selected])
        if spin_sq == 0.0 and nelec[0] == nelec[1]:
            a = b = np.union1d(a, b)
        if len(a) * len(b) > max_variational_dim:
            raise ValueError('Expanded Cartesian space exceeds max_variational_dim')
        diagonalization_started = perf_counter()
        updated = solve_sci_batch([(a, b)], hcore, eri, norb, nelec,
                                  spin_sq=spin_sq, max_cycle=200)[0]
        diagonalization_seconds = perf_counter() - diagonalization_started
        if updated.energy > result.energy + 1e-7:
            raise RuntimeError('Expanded-space energy increased; diagonalization may not have converged')
        record = {'iteration': iteration + 1, 'energy_before': float(result.energy),
                  'energy_after': float(updated.energy), 'pt2_before': pt2,
                  'selected_pairs': len(selected), 'dimension': len(a) * len(b),
                  'selection_seconds': selection_seconds,
                  'diagonalization_seconds': diagonalization_seconds}
        history.append(record)
        result = updated
        if callback:
            callback(result, record)
    return result, history
