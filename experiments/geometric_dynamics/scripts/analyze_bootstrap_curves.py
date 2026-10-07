"""Within-model alignment of paired stratified-bootstrap adjacent CKA curves.

Uses saved bootstrap draws; never reruns models or matches depths across models.
Default input is the original completed run preserved in Git history. Supply
--replicates for a new run's CKA_prev_bootstrap_replicates.csv.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import subprocess
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODELS = {'ResNet-18': 'resnet18', 'ResNet-152': 'resnet152',
          'ConvNeXt-Tiny': 'convnext_tiny', 'ConvNeXt-Base': 'convnext_base'}
SOURCE_COMMIT = 'ce2945f'
SOURCE_PATH = 'experiments/geometric_dynamics/results/block_geometry_metrics/CKA_prev_bootstrap_replicates.csv'


def aligned_curves(replicates, reference, expected_repeats=1000):
    """Align by explicit depth/stage identity, rejecting missing or duplicate data."""
    if replicates.duplicated(['replicate', 'depth']).any() or reference.depth.duplicated().any():
        raise ValueError('Duplicate replicate/depth or reference depth.')
    ref = reference.loc[reference.depth > 0].sort_values('depth')
    draws = replicates.loc[replicates.depth > 0]
    wide = draws.pivot(index='replicate', columns='depth', values='CKA_prev').sort_index(axis=1)
    if len(wide) != expected_repeats or list(wide.columns) != list(ref.depth):
        raise ValueError('Incomplete replicate count or mismatched layer identities.')
    stages = draws[['depth', 'stage']].drop_duplicates().sort_values('depth')
    if list(stages.itertuples(index=False, name=None)) != list(ref[['depth', 'stage']].itertuples(index=False, name=None)):
        raise ValueError('Stage identity differs from reference.')
    x, base = wide.to_numpy(), ref.CKA_prev.to_numpy()
    if not np.isfinite(x).all() or not np.isfinite(base).all():
        raise ValueError('Missing or nonfinite CKA outside the undefined first observation.')
    if np.any((x < -1e-10) | (x > 1 + 1e-10)) or np.any((base < -1e-10) | (base > 1 + 1e-10)):
        raise ValueError('CKA outside [0, 1].')
    return ref, x, base


def curve_statistics(x, base):
    if len(base) < 3 or np.std(base) == 0 or np.any(np.std(x, axis=1) == 0):
        raise ValueError('Curve correlations require nonconstant curves with at least three points.')
    values = {
        'pearson_to_original': np.array([np.corrcoef(row, base)[0, 1] for row in x]),
        'spearman_to_original': np.array([spearmanr(row, base).statistic for row in x]),
        'adjacent_change_pearson': np.array([np.corrcoef(np.diff(row), np.diff(base))[0, 1] for row in x]),
        'rmse_to_original': np.sqrt(np.mean((x - base) ** 2, axis=1)),
    }
    pairwise = np.corrcoef(x)
    values['pairwise_curve_pearson'] = pairwise[np.triu_indices(len(x), 1)]
    summary = {}
    for name, samples in values.items():
        for percentile, value in zip((10, 50, 90), np.percentile(samples, (10, 50, 90))):
            summary[f'{name}_p{percentile}'] = value
    lo, med, hi = np.percentile(x, (10, 50, 90), axis=0)
    summary.update(pointwise_80_width_median=np.median(hi-lo),
                   pointwise_80_width_max=np.max(hi-lo), mean_signed_bias=np.mean(x-base),
                   all_curve_pearson_min=values['pearson_to_original'].min())
    return summary, lo, med, hi


def plot_model(model, ref, x, base, stats, lo, med, hi, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    destination.mkdir(parents=True, exist_ok=True)
    depth = ref.depth.to_numpy()
    for suffix in ('alignment', 'deviations'):
        fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
        if suffix == 'alignment':
            ax.plot(depth, x.T, color='#8ba6bd', alpha=.012, lw=.65)
            ax.fill_between(depth, lo, hi, color='#6ea6d6', alpha=.35)
            ax.plot(depth, med, color='#246ba0', lw=1.8)
            ax.plot(depth, base, color='#c64b32', ls='--', lw=1.8)
            ax.set(ylabel='Adjacent linear CKA', ylim=(0, 1.025),
                   title=f'{model}: {len(x):,} bootstrap curves | median r = {stats["pearson_to_original_p50"]:.4f}')
            handles=[Line2D([0],[0],color='#8ba6bd',label='All bootstrap curves'),
                     Patch(color='#6ea6d6',alpha=.35,label='Pointwise p10-p90'),
                     Line2D([0],[0],color='#246ba0',label='Bootstrap median'),
                     Line2D([0],[0],color='#c64b32',ls='--',label='Original curve')]
        else:
            ax.fill_between(depth, lo-base, hi-base, color='#6ea6d6', alpha=.35)
            ax.plot(depth, med-base, color='#246ba0', lw=1.8)
            ax.axhline(0, color='#c64b32', ls='--', lw=1.5)
            ax.set(ylabel='Bootstrap CKA minus original CKA', ylim=(-.1,.1),
                   title=f'{model}: numerical deviations | median RMSE = {stats["rmse_to_original_p50"]:.4f}')
            handles=[Patch(color='#6ea6d6',alpha=.35,label='Pointwise p10-p90'),
                     Line2D([0],[0],color='#246ba0',label='Bootstrap median'),
                     Line2D([0],[0],color='#c64b32',ls='--',label='Zero difference')]
        ax.set_xlabel('Original observation index')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
        fig.legend(handles=handles, loc='outside lower center', fontsize=9, ncol=2, frameon=False)
        fig.savefig(destination/f'bootstrap_curve_{suffix}.png', dpi=140)
        plt.close(fig)


def analyze(results=ROOT/'results', replicates=None, expected_repeats=1000):
    results = Path(results)
    if replicates is None:
        repo = ROOT.parents[1]
        payload = subprocess.check_output(['git', 'show', f'{SOURCE_COMMIT}:{SOURCE_PATH}'], cwd=repo)
        commit = subprocess.check_output(['git', 'rev-parse', SOURCE_COMMIT], cwd=repo, text=True).strip()
        source = {'git_commit': commit, 'git_path': SOURCE_PATH}
    else:
        payload = Path(replicates).read_bytes()
        source = {'file_name': Path(replicates).name}
    draws = pd.read_csv(io.BytesIO(payload))
    if set(draws.model) != set(MODELS):
        raise ValueError('Input must contain exactly the four configured CNN models.')
    source['sha256'] = hashlib.sha256(payload).hexdigest()
    reference_hashes = {}
    for model, slug in MODELS.items():
        folder = results/'geometry'/slug
        ref_path = folder/'CKA_prev.csv'
        reference_hashes[str(ref_path.relative_to(results))] = hashlib.sha256(ref_path.read_bytes()).hexdigest()
        ref, x, base = aligned_curves(draws.loc[draws.model == model], pd.read_csv(ref_path), expected_repeats)
        stats, lo, med, hi = curve_statistics(x, base)
        published = pd.read_csv(folder/'CKA_prev_bootstrap.csv').set_index('depth').loc[ref.depth]
        np.testing.assert_allclose(np.column_stack((lo,med,hi)), published[['p10','median','p90']].to_numpy())
        row = dict(model=model, n_bootstrap=len(x), n_valid_depths=len(base), **stats)
        pd.DataFrame([row]).to_csv(folder/'bootstrap_curve_alignment_summary.csv', index=False)
        layer = ref[['model','depth','stage']].copy()
        layer['estimate'] = base
        for name, value in [('p10',lo),('median',med),('p90',hi),('interval_width',hi-lo),
                            ('median_minus_original',med-base)]:
            layer[name] = value
        layer.to_csv(folder/'bootstrap_curve_alignment_by_layer.csv', index=False)
        plot_model(model, ref, x, base, stats, lo, med, hi, results/'figures'/slug)
        print(f'{model}: median r={stats["pearson_to_original_p50"]:.6f}, median RMSE={stats["rmse_to_original_p50"]:.6f}')
    metadata = results/'metadata/cnn_bootstrap'
    metadata.mkdir(parents=True, exist_ok=True)
    record = dict(status='completed', analysis='within-model complete CKA curve alignment',
                  input=source, reference_sha256=reference_hashes, n_bootstrap=expected_repeats,
                  alignment='Identical original depth and stage labels; no interpolation or warping.',
                  excluded='Depth 0: adjacent CKA is undefined.',
                  interval='Pointwise p10-p90; not simultaneous confidence bands.',
                  interpretation='Descriptive resampling stability conditional on the current image cohort and fixed weights; draws are not independent model-training experiments.',
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (metadata/'curve_alignment_record.json').write_text(json.dumps(record, indent=2)+'\n')
    manifest = {str(p.relative_to(results)): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                                             'bytes': p.stat().st_size}
                for p in sorted(results.rglob('*')) if p.suffix in ('.csv', '.png')}
    (results/'metadata/final_outputs_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--replicates', type=Path, help='New run replicate CSV; defaults to saved historical run in Git.')
    parser.add_argument('--expected-repeats', type=int, default=1000)
    args = parser.parse_args()
    analyze(args.results, args.replicates, args.expected_repeats)
