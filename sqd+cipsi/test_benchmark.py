import unittest
from benchmark import METRICS, summarize


class BenchmarkSummaryTest(unittest.TestCase):
    def row(self, method, seed, wall, error, dim=10):
        row = {key: 0.0 for key in METRICS}
        row.update(method=method, seed=seed, wall_seconds=wall,
                   absolute_error_mEh=error, dimension=dim,
                   sqd_total_energy_Eh=-1.0, total_energy_Eh=-1.0 + error / 1000)
        return row

    def test_paired_runtime_uses_total_enhanced_cost(self):
        report = summarize([self.row('sqd', 23, 2, 10), self.row('sqd_cipsi', 23, 8, 3, 30)])
        pair = report['paired_comparisons'][0]
        self.assertEqual(pair['runtime_ratio_cipsi_over_sqd'], 4)
        self.assertEqual(pair['extra_wall_seconds'], 6)
        self.assertEqual(pair['absolute_error_reduction_mEh'], 7)
        self.assertEqual(pair['dimension_ratio'], 3)
        self.assertIsNone(report['by_method']['sqd']['wall_seconds']['sample_std'])

    def test_missing_seed_is_not_paired_with_another_seed(self):
        report = summarize([self.row('sqd', 23, 2, 10), self.row('sqd_cipsi', 24, 8, 3)])
        self.assertEqual(report['paired_comparisons'], [])

    def test_statistics_report_sample_variability(self):
        report = summarize([self.row('sqd', 23, 2, 10), self.row('sqd', 24, 4, 20)])
        wall = report['by_method']['sqd']['wall_seconds']
        self.assertEqual(wall['mean'], 3)
        self.assertAlmostEqual(wall['sample_std'], 2**0.5)


if __name__ == '__main__':
    unittest.main()
