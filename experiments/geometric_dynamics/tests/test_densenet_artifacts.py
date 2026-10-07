"""Offline checks of the measured DenseNet Colab artifacts (no torch download)."""
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/metadata/densenet_20260930'
COUNTS={'DenseNet-121':63,'DenseNet-169':87,'DenseNet-201':103}


def rows(name):
    if name in ('densenet_geometry_metrics.csv','densenet_lle_summary.csv'):
        suffix='geometry' if name=='densenet_geometry_metrics.csv' else 'lle_summary'
        return [r for slug in ('densenet121','densenet169','densenet201') for r in rows(slug+'_'+suffix+'.csv')]
    category='lle' if '_lle' in name else 'geometry'
    slug=name.split('_')[0]
    with (ROOT/'results'/category/slug/name).open() as f:
        return list(csv.DictReader(f))


class DenseNetArtifacts(unittest.TestCase):
    def test_provenance_and_executed_sources(self):
        protocol=json.loads((RUN/'protocol.json').read_text())
        self.assertEqual(protocol['status'],'completed')
        self.assertEqual(protocol['gpu'],'Tesla T4')
        for model in protocol['models']:
            self.assertTrue(model['weights'].endswith('IMAGENET1K_V1'))
            self.assertEqual(model['observations'],COUNTS[model['model']])
            self.assertEqual(model['growth_rate'],32)
        manifest=json.loads((ROOT/'results/metadata/final_outputs_manifest.json').read_text())
        for name,info in manifest.items():
            self.assertEqual(hashlib.sha256((ROOT/'results'/name).read_bytes()).hexdigest(),info['sha256'])
        note=json.loads((ROOT/'notebooks/Cat_Dog_DenseNet_121_169_201_Matched_Geometry_LLE.ipynb').read_text())
        code=next(c for c in note['cells'] if c['cell_type']=='code')
        self.assertEqual(code['execution_count'],1)
        embedded=[]
        for node in ast.walk(ast.parse(''.join(code['source']))):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='write_text':
                embedded.append(ast.literal_eval(node.args[0]))
        for name in ('metrics.py','run_densenet_geometry.py'):
            self.assertIn((ROOT/'scripts'/name).read_text(),embedded)
        for cell in note['cells']:
            self.assertFalse(any(o['output_type']=='error' for o in cell.get('outputs',[])))


    def test_geometry_and_transitions(self):
        data=rows('densenet_geometry_metrics.csv')
        self.assertEqual(len(data),253)
        for model,count in COUNTS.items():
            q=[r for r in data if r['model']==model]
            self.assertEqual(len(q),count)
            self.assertEqual([int(r['depth']) for r in q],list(range(count)))
            self.assertEqual(len(set(r['stage'] for r in q)),count)
            self.assertEqual(q[-1]['stage'],'final_norm_relu')
            for i,r in enumerate(q):
                self.assertAlmostEqual(float(r['S']),float(r['d_between_cos'])/float(r['d_within_cos']))
                self.assertTrue(1-1e-8<=float(r['PR'])<=199+1e-8)
                self.assertTrue(0<=float(r['probe_mean'])<=1)
                for field in ('S','Fisher_raw','PR','probe_mean'):
                    if i: self.assertAlmostEqual(float(r['delta_'+field]),float(r[field])-float(q[i-1][field]))
                    else: self.assertEqual(r['delta_'+field],'')
                if i: self.assertTrue(0<=float(r['CKA_prev'])<=1+1e-8)
                else: self.assertEqual(r['CKA_prev'],'')


    def test_final_entropy_summaries(self):
        summaries=rows('densenet_lle_summary.csv')
        self.assertEqual(len(summaries),1265)
        for model,count in COUNTS.items():
            slug=model.lower().replace('-','')
            subset=rows(slug+'_lle_summary.csv')
            self.assertEqual(subset,[r for r in summaries if r['model']==model])
            self.assertEqual(len(subset),count*5)
            self.assertEqual({int(r['k']) for r in subset},{4,8,16,32,64})
        for r in summaries:
            self.assertTrue(0<=float(r['LLE'])<=1)
            self.assertTrue(0<=float(r['fraction_H_ge_0_8'])<=1)
            self.assertAlmostEqual(float(r['order_vs_shuffle']),1-float(r['LLE'])/float(r['shuffle_mean']))

if __name__=='__main__': unittest.main()
