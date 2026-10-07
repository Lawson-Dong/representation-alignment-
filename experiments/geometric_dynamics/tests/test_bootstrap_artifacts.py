"""Verify exported image pairing, replicate integrity and run provenance."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/block_geometry_metrics'

class BootstrapArtifactTests(unittest.TestCase):
    def test_completed_run_and_hashes(self):
        record = json.loads((OUT/'run_record.json').read_text())
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['bootstrap_repeats'], 1000)
        self.assertEqual(record['classes'], {'3': 100, '5': 100})
        for filename, expected in record['csv_sha256'].items():
            self.assertEqual(hashlib.sha256((OUT/filename).read_bytes()).hexdigest(), expected)

    def test_each_draw_preserves_balance(self):
        images = pd.read_csv(OUT/'sample_manifest.csv')
        draws = pd.read_csv(OUT/'bootstrap_indices.csv').to_numpy()
        self.assertEqual(draws.shape, (1000, 200))
        self.assertEqual(images.archive_path.nunique(), 200)
        self.assertTrue(np.issubdtype(draws.dtype, np.integer))
        self.assertTrue(np.all((draws >= 0) & (draws < 200)))
        labels = images.label.to_numpy()[draws]
        self.assertTrue(np.all((labels == 3).sum(1) == 100))
        self.assertTrue(np.all((labels == 5).sum(1) == 100))

    def test_complete_curves_and_pointwise_percentiles(self):
        samples = pd.read_csv(OUT/'CKA_prev_bootstrap_replicates.csv')
        summary = pd.read_csv(OUT/'CKA_prev_bootstrap.csv')
        self.assertEqual(len(samples), 122000)
        self.assertEqual(len(summary), 122)
        self.assertFalse(samples.duplicated(['model', 'depth', 'replicate']).any())
        for row in summary.itertuples():
            values = samples.loc[(samples.model == row.model) & (samples.depth == row.depth), 'CKA_prev'].to_numpy()
            self.assertEqual(len(values), 1000)
            if row.depth == 0:
                self.assertTrue(np.isnan(values).all())
            else:
                self.assertTrue(np.all((values >= 0) & (values <= 1 + 1e-10)))
                np.testing.assert_allclose(np.percentile(values, [10, 50, 90]), [row.p10, row.median, row.p90])

    def test_stage_dip_replicates_match_full_curves(self):
        curves = pd.read_csv(OUT/'CKA_prev_bootstrap_replicates.csv')
        samples = pd.read_csv(OUT/'boundary_dip_replicates.csv')
        summary = pd.read_csv(OUT/'boundary_dip.csv')
        self.assertEqual(len(samples), 16000)
        self.assertEqual(len(summary), 16)
        for row in summary.itertuples():
            wide = curves.loc[curves.model == row.model].pivot(index='replicate', columns='stage', values='CKA_prev')
            expected = wide[row.comparator_blocks.split(';')].median(axis=1) - wide[row.boundary_block]
            got = samples.loc[(samples.model == row.model) & (samples.stage == row.stage)].sort_values('replicate').delta
            np.testing.assert_allclose(got, expected)
            self.assertAlmostEqual(row.fraction_positive, np.mean(expected > 0))
            np.testing.assert_allclose(np.percentile(got, [10, 50, 90]), [row.p10, row.median, row.p90])

if __name__ == '__main__':
    unittest.main()
