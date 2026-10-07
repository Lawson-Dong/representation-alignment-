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
RUN=ROOT/'results/geometry/shared/densenet_20260930'
COUNTS={'DenseNet-121':63,'DenseNet-169':87,'DenseNet-201':103}


def artifact_path(name):
    category = 'figures' if name.endswith('.png') else 'lle' if '_lle' in name else 'geometry'
    model = next((slug for slug in ('densenet121', 'densenet169', 'densenet201') if name.startswith(slug + '_')), None)
    return ROOT/'results'/category/(model if model else 'shared/densenet_20260930')/name


def rows(name):
    with artifact_path(name).open() as f: return list(csv.DictReader(f))


class DenseNetArtifacts(unittest.TestCase):
    def test_provenance_and_executed_sources(self):
        protocol=json.loads((RUN/'protocol.json').read_text())
        self.assertEqual(protocol['status'],'completed')
        self.assertEqual(protocol['gpu'],'Tesla T4')
        for model in protocol['models']:
            self.assertTrue(model['weights'].endswith('IMAGENET1K_V1'))
            self.assertEqual(model['observations'],COUNTS[model['model']])
            self.assertEqual(model['growth_rate'],32)
        manifest=json.loads((RUN/'artifact_manifest.json').read_text())
        for name,info in manifest.items():
            if name.endswith('.npz'): continue  # runtime-only activations, explicitly omitted
            self.assertEqual(hashlib.sha256(artifact_path(name).read_bytes()).hexdigest(),info['sha256'])
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

    def test_cohort_and_paired_splits(self):
        sample=rows('sample_manifest.csv')
        self.assertEqual([int(r['sample_index']) for r in sample],list(range(200)))
        self.assertEqual(len(set(r['image_path'] for r in sample)),200)
        for label in ('3','5'): self.assertEqual(sum(r['label']==label for r in sample),100)
        grouped=defaultdict(dict)
        for r in rows('probe_splits.csv'): grouped[int(r['split'])].setdefault(r['role'],[]).append(int(r['sample_index']))
        self.assertEqual(set(grouped),set(range(5)))
        for roles in grouped.values():
            self.assertEqual(len(roles['train']),140); self.assertEqual(len(roles['test']),60)
            self.assertFalse(set(roles['train'])&set(roles['test']))
            self.assertEqual(set(roles['train'])|set(roles['test']),set(range(200)))
            for role,n in [('train',70),('test',30)]:
                for label in ('3','5'): self.assertEqual(sum(sample[i]['label']==label for i in roles[role]),n)

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

    def test_pointwise_entropy_matches_summaries(self):
        sample=rows('sample_manifest.csv')
        summaries={(r['model'],r['depth'],r['stage'],r['k']):r for r in rows('densenet_lle_summary.csv')}
        self.assertEqual(len(summaries),1265)
        grouped=defaultdict(list)
        for model,count in COUNTS.items():
            slug=model.lower().replace('-','')
            point=rows(slug+'_lle_per_image.csv')
            self.assertEqual(len(point),count*5*200)
            for r in point:
                i=int(r['sample_index']); self.assertEqual(r['label'],sample[i]['label'])
                self.assertEqual(r['image_path'],sample[i]['image_path'])
                self.assertEqual(r['model'],model)
                h=float(r['H']); self.assertTrue(0<=h<=1+1e-12)
                grouped[(model,r['depth'],r['stage'],r['k'])].append((i,h))
        self.assertEqual(set(grouped),set(summaries))
        for key,values in grouped.items():
            self.assertEqual([i for i,h in values],list(range(200)))
            r=summaries[key]
            self.assertAlmostEqual(sum(h for i,h in values)/200,float(r['LLE']))
            self.assertAlmostEqual(sum(h>=.8 for i,h in values)/200,float(r['fraction_H_ge_0_8']))
            self.assertAlmostEqual(float(r['order_vs_shuffle']),1-float(r['LLE'])/float(r['shuffle_mean']))

if __name__=='__main__': unittest.main()
