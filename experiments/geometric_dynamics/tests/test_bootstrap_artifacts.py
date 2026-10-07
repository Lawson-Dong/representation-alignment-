"""Verify exported image pairing, replicate integrity and run provenance."""
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/geometry/cross-model/block_geometry_metrics'

class BootstrapArtifactTests(unittest.TestCase):
    def test_completed_run_and_hashes(self):
        record = json.loads((OUT/'run_record.json').read_text())
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['bootstrap_repeats'], 1000)
        self.assertEqual(record['classes'], {'3': 100, '5': 100})
        for filename, expected in record['csv_sha256'].items():
            self.assertEqual(hashlib.sha256((OUT/filename).read_bytes()).hexdigest(), expected)




    def test_final_bootstrap_summaries(self):
        summary = pd.read_csv(OUT/'CKA_prev_bootstrap.csv')
        self.assertEqual(len(summary), 122)
        self.assertFalse(summary.duplicated(['model', 'depth']).any())
        for row in summary.itertuples():
            if row.depth == 0:
                self.assertTrue(np.isnan([row.p10, row.median, row.p90]).all())
            else:
                self.assertTrue(0 <= row.p10 <= row.median <= row.p90 <= 1 + 1e-10)
        dips = pd.read_csv(OUT/'boundary_dip.csv')
        self.assertEqual(len(dips), 16)
        for row in dips.itertuples():
            self.assertTrue(row.p10 <= row.median <= row.p90)
            self.assertTrue(0 <= row.fraction_positive <= 1)
        for name in ('CKA_prev_bootstrap_replicates.csv', 'boundary_dip_replicates.csv', 'bootstrap_indices.csv'):
            self.assertFalse((OUT/name).exists())

if __name__ == '__main__':
    unittest.main()
