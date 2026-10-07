"""Whole-curve alignment definitions, identity guards and published summary consistency."""
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('bootstrap_curves',ROOT/'scripts/analyze_bootstrap_curves.py')
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class CurveAlignmentTests(unittest.TestCase):
    def test_shape_agreement_does_not_remove_numerical_shift(self):
        base=np.array([.3,.5,.4,.8])
        x=np.array([base,base+.02])
        summary,*_=m.curve_statistics(x,base)
        self.assertAlmostEqual(summary['pearson_to_original_p50'],1)
        self.assertAlmostEqual(summary['spearman_to_original_p50'],1)
        self.assertAlmostEqual(summary['adjacent_change_pearson_p50'],1)
        self.assertAlmostEqual(summary['rmse_to_original_p50'],.01)
        self.assertAlmostEqual(summary['mean_signed_bias'],.01)

    def test_matches_labels_and_rejects_incomplete_or_duplicate_draws(self):
        ref=pd.DataFrame({'depth':[0,1,2,3], 'stage':['stem','a','b','c'], 'CKA_prev':[np.nan,.3,.5,.4]})
        draws=pd.concat([ref.assign(replicate=i) for i in range(2)],ignore_index=True)
        aligned,x,base=m.aligned_curves(draws.sample(frac=1,random_state=42),ref,2)
        self.assertEqual(list(aligned.depth),[1,2,3])
        np.testing.assert_array_equal(x,np.tile(base,(2,1)))
        for bad in (draws.drop(index=3),pd.concat([draws,draws.iloc[[1]]]),draws.assign(stage='wrong')):
            with self.assertRaises(ValueError):m.aligned_curves(bad,ref,2)

    def test_published_layer_tables_match_original_intervals(self):
        for model,slug in m.MODELS.items():
            folder=ROOT/'results/geometry'/slug
            layer=pd.read_csv(folder/'bootstrap_curve_alignment_by_layer.csv')
            original=pd.read_csv(folder/'CKA_prev.csv')
            interval=pd.read_csv(folder/'CKA_prev_bootstrap.csv')
            original=original[original.depth>0]
            interval=interval[interval.depth>0]
            self.assertEqual(list(layer.depth),list(original.depth))
            self.assertEqual(list(layer.stage),list(original.stage))
            np.testing.assert_allclose(layer.estimate,original.CKA_prev)
            np.testing.assert_allclose(layer[['p10','median','p90']],interval[['p10','median','p90']])
            np.testing.assert_allclose(layer.interval_width,layer.p90-layer.p10)
            np.testing.assert_allclose(layer.median_minus_original,layer['median']-layer.estimate)
            summary=pd.read_csv(folder/'bootstrap_curve_alignment_summary.csv').iloc[0]
            self.assertEqual(summary.model,model)
            self.assertEqual(summary.n_bootstrap,1000)
            self.assertEqual(summary.n_valid_depths,len(layer))
            self.assertAlmostEqual(summary.pointwise_80_width_median,layer.interval_width.median())
            self.assertAlmostEqual(summary.pointwise_80_width_max,layer.interval_width.max())
        record=json.loads((ROOT/'results/metadata/cnn_bootstrap/curve_alignment_record.json').read_text())
        self.assertEqual(record['n_bootstrap'],1000)
        self.assertEqual(record['status'],'completed')

if __name__=='__main__':unittest.main()
