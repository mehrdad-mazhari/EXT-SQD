import unittest

import numpy as np

from cipsi import cipsi_scores, cipsi_scores_from_residuals, select_top_cipsi


class CIPSIScoreTest(unittest.TestCase):
    def test_epstein_nesbet_scores_and_pt2(self):
        result = cipsi_scores(
            energy=-1.0,
            coefficients=np.array([0.8, 0.6]),
            external_diagonal=np.array([-0.5, 0.0]),
            couplings=np.array([[0.1, 0.2], [0.3, -0.1]]),
        )

        np.testing.assert_allclose(result.residuals, [0.2, 0.18])
        np.testing.assert_allclose(result.denominators, [-0.5, -1.0])
        np.testing.assert_allclose(result.pt2_contributions, [-0.08, -0.0324])
        np.testing.assert_allclose(result.scores, [0.08, 0.0324])
        self.assertAlmostEqual(result.pt2_correction, -0.1124)

    def test_complex_coefficients_use_squared_magnitude(self):
        result = cipsi_scores(
            -1.0,
            np.array([1j]),
            np.array([0.0]),
            np.array([[0.5]]),
        )
        np.testing.assert_allclose(result.pt2_contributions, [-0.25])

    def test_level_shift_regularizes_zero_denominator(self):
        result = cipsi_scores(
            -1.0,
            np.array([1.0]),
            np.array([-1.0]),
            np.array([[0.1]]),
            level_shift=0.01,
        )
        np.testing.assert_allclose(result.denominators, [0.01])
        np.testing.assert_allclose(result.scores, [1.0])

    def test_precomputed_residuals_match_coupling_interface(self):
        direct = cipsi_scores_from_residuals(-1.0, [0.0], [0.5])
        via_couplings = cipsi_scores(-1.0, [1.0], [0.0], [[0.5]])
        np.testing.assert_allclose(direct.scores, via_couplings.scores)
        np.testing.assert_allclose(
            direct.pt2_contributions, via_couplings.pt2_contributions
        )

    def test_selection_is_descending_stable_and_thresholded(self):
        selected = select_top_cipsi([0.2, 0.7, 0.7, 0.1], 2, score_threshold=0.2)
        np.testing.assert_array_equal(selected, [1, 2])

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected"):
            cipsi_scores(-1.0, [1.0, 0.0], [0.0], [[0.2]])


if __name__ == "__main__":
        unittest.main()
