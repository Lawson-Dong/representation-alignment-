"""Offline numeric checks against direct feature-row resampling."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

p = Path(__file__).resolve().parents[1] / 'scripts/stratified_bootstrap.py'
spec = importlib.util.spec_from_file_location('bootstrap', p)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)

class BootstrapTests(unittest.TestCase):
    def test_preserves_class_counts_and_reproducibility(self):
        labels = np.array([3, 5, 3, 5, 3, 5, 5])
        idx = b.stratified_indices(labels, 30, 7)
        np.testing.assert_array_equal(idx, b.stratified_indices(labels, 30, 7))
        self.assertTrue(np.all((labels[idx] == 3).sum(1) == 3))
        self.assertTrue(np.all((labels[idx] == 5).sum(1) == 4))
        self.assertTrue(any(len(np.unique(i)) < len(labels) for i in idx))

    def test_gram_resampling_matches_direct_features_with_duplicates(self):
        rng = np.random.default_rng(8)
        features = [rng.normal(size=(12, d)) for d in [5, 9, 3]]
        idx = np.array([0, 0, 0, 2, 3, 7, 7, 9, 10, 11, 11, 11])
        got = b.cka_curve([x @ x.T for x in features], idx)
        expected = [np.nan]
        for x, y in zip(features, features[1:]):
            x = x[idx] - x[idx].mean(0)
            y = y[idx] - y[idx].mean(0)
            expected.append(np.linalg.norm(x.T @ y)**2 /
                            (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))
        np.testing.assert_allclose(got, expected, equal_nan=True, atol=1e-12)

    def test_contrasts_exclude_downsample_and_stem(self):
        self.assertEqual(b.stage_contrasts(['stem', 's1.b1', 's1.b2', 'down1',
                                           's2.b1', 's2.b2', 's2.b3']),
                         [(1, 1, [2]), (2, 4, [5, 6])])

    def test_degenerate_cka_is_not_silently_zero(self):
        with self.assertRaises(ValueError):
            b.cka_curve([np.ones((5, 5)), np.eye(5)], np.arange(5))

if __name__ == '__main__':
    unittest.main()
