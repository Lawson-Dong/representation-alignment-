"""Offline scientific and provenance checks; no model downloads."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('metrics', ROOT/'scripts/metrics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class GeometryTests(unittest.TestCase):
    def test_pairwise_definitions(self):
        x=np.random.default_rng(7).normal(size=(12,5)); y=np.array([3]*6+[5]*6)
        z=x/np.linalg.norm(x,axis=1,keepdims=True)
        cosine=1-z@z.T; euclid=np.sqrt(np.maximum(0,2*cosine))
        mask=np.triu_indices(6,1)
        r=m.representation_stats(x,y)
        w=(cosine[:6,:6][mask].mean()+cosine[6:,6:][mask].mean())/2
        b=cosine[:6,6:].mean()
        self.assertAlmostEqual(r['d_within_cos'],w)
        self.assertAlmostEqual(r['d_between_cos'],b)
        self.assertAlmostEqual(r['S'],b/w)
        self.assertAlmostEqual(r['d_within_euclid_unit'],(euclid[:6,:6][mask].mean()+euclid[6:,6:][mask].mean())/2)
        self.assertAlmostEqual(r['d_between_euclid_unit'],euclid[:6,6:].mean())
        self.assertGreaterEqual(r['PR'],1-1e-10)
        self.assertLessEqual(r['PR'],5+1e-10)
    def test_cka_width_and_invariances(self):
        x=np.random.default_rng(8).normal(size=(20,4))
        self.assertAlmostEqual(m.linear_cka(x,x),1)
        wider=np.column_stack([2*x+3,np.zeros((20,3))])
        self.assertAlmostEqual(m.linear_cka(x,wider),1)
        changed=np.random.default_rng(9).normal(size=(20,7))
        self.assertTrue(0<=m.linear_cka(x,changed)<=1+1e-10)

class ArtifactTests(unittest.TestCase):
    def test_source_hashes(self):
        manifest=json.loads((ROOT/'results/source_manifest.json').read_text())
        for name,info in manifest.items():
            p=ROOT/info['path']
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),info['sha256'])
    def test_python_syntax(self):
        for p in (ROOT/'scripts').glob('*.py'):
            compile(p.read_text(),str(p),'exec')
        for p in (ROOT/'notebooks').glob('*.ipynb'):
            n=json.loads(p.read_text());self.assertEqual(n['nbformat'],4)
            for i,c in enumerate(n['cells']):
                if c['cell_type']=='code':compile(''.join(c['source']),f'{p}:{i}','exec')
    def test_cnn_csv(self):
        import pandas as pd
        folder = ROOT/'results/block_geometry_metrics'
        frames = [pd.read_csv(p) for p in folder.glob('*.csv')
                  if p.stem in ['CKA_prev', 'S', 'd_within_cos', 'd_between_cos']]
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=['model', 'depth', 'stage', 'boundary'], validate='one_to_one')
        rows = merged.fillna('').to_dict('records')
        self.assertEqual(len(rows),122)
        for model,count in [('ResNet-18',9),('ResNet-152',51),('ConvNeXt-Tiny',22),('ConvNeXt-Base',40)]:
            r=[q for q in rows if q['model']==model]
            self.assertEqual(len(r),count)
            self.assertEqual([int(q['depth']) for q in r],list(range(count)))
            self.assertEqual(len({q['stage'] for q in r}),count)
            self.assertEqual(r[0]['CKA_prev'],'')
            for q in r:
                self.assertAlmostEqual(float(q['S']),float(q['d_between_cos'])/float(q['d_within_cos']))
                if q['CKA_prev'] != '':self.assertTrue(0<=float(q['CKA_prev'])<=1+1e-10)

if __name__=='__main__':unittest.main()
