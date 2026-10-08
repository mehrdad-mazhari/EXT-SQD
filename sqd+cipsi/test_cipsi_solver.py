import unittest

import numpy as np
from pyscf import fci, lib
from qiskit_addon_sqd.fermion import solve_sci_batch

from cipsi_solver import excitation_strings, refine_cipsi, select_external


class ExternalCIPSITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lib.num_threads(1)
        cls.norb, cls.nelec = 4, (2, 2)
        rng = np.random.default_rng(8)
        raw = rng.normal(size=(4, 4))
        cls.h = np.diag([-3.0, -2.0, 1.0, 3.0]) + 0.1 * (raw + raw.T) / 2
        factors = rng.normal(size=(3, 4, 4)) * 0.15
        factors += factors.transpose(0, 2, 1)
        cls.eri = np.einsum('Lij,Lkl->ijkl', factors, factors)
        cls.strings = fci.cistring.make_strings(range(4), 2)
        cls.result = solve_sci_batch([(np.array([3, 5]), np.array([3, 5]))],
            cls.h, cls.eri, 4, (2, 2), max_cycle=200)[0]

    def test_blocked_scores_equal_independent_full_fci_residual(self):
        result = self.result
        state = result.sci_state
        ci = np.zeros((6, 6))
        ia = np.searchsorted(self.strings, state.ci_strs_a)
        ib = np.searchsorted(self.strings, state.ci_strs_b)
        ci[np.ix_(ia, ib)] = state.amplitudes
        h2 = fci.direct_spin1.absorb_h1e(self.h, self.eri, 4, (2, 2), 0.5)
        residual = fci.direct_spin1.contract_2e(h2, ci, 4, (2, 2)).ravel()
        diag = fci.direct_spin1.make_hdiag(self.h, self.eri, 4, (2, 2))
        mask = np.ones((6, 6), dtype=bool)
        mask[np.ix_(ia, ib)] = False
        contributions = residual[mask.ravel()] ** 2 / (result.energy - diag[mask.ravel()])
        for block in (1, 3, 20):
            pairs, pt2 = select_external(result, self.h, self.eri, 4, (2, 2),
                keep=100, beta_block_size=block)
            self.assertAlmostEqual(pt2, float(contributions.sum()), places=10)
            expected = {}
            for index in np.flatnonzero(mask.ravel()):
                a, b = divmod(int(index), 6)
                score = abs(residual[index] ** 2 / (result.energy - diag[index]))
                if score > 1e-12:
                    expected[(int(self.strings[a]), int(self.strings[b]))] = score
            self.assertEqual({(a, b) for _, a, b in pairs}, set(expected))
            for score, a, b in pairs:
                self.assertAlmostEqual(score, expected[a, b], places=10)

    def test_expansion_reaches_full_ci_and_preserves_old_space(self):
        result, history = refine_cipsi(self.result, self.h, self.eri, 4, (2, 2),
            iterations=1, keep=100, spin_sq=None)
        exact, _ = fci.direct_spin1.kernel(self.h, self.eri, 4, (2, 2))
        self.assertAlmostEqual(result.energy, exact, places=8)
        self.assertLessEqual(result.energy, self.result.energy + 1e-9)
        self.assertEqual(len(history), 1)
        self.assertTrue(set(self.result.sci_state.ci_strs_a) <= set(result.sci_state.ci_strs_a))

    def test_working_limit_rejects_before_contraction(self):
        with self.assertRaisesRegex(ValueError, 'working space'):
            select_external(self.result, self.h, self.eri, 4, (2, 2), max_working_dim=1)

    def test_excitation_strings_preserve_electrons_and_cover_doubles(self):
        strings = excitation_strings([3], 4)
        self.assertEqual(set(map(int, strings)), set(map(int, self.strings)))
        self.assertTrue(all(int(s).bit_count() == 2 for s in strings))


if __name__ == '__main__':
    unittest.main()
