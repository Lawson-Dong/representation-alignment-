"""Paired, class-stratified image bootstrap for adjacent biased linear CKA.

Reuses raw pooled feature caches. Intervals are pointwise 10–90 percentiles
(central 80%), not simultaneous bands or causal tests. Probe split SD is
reported separately and is never called image-bootstrap uncertainty.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

CACHE_FILES = {
    'ResNet-18': 'resnet18_per_image_vectors.npz',
    'ResNet-152': 'resnet152_per_image_vectors.npz',
    'ConvNeXt-Tiny': 'convnext_tiny_per_image_vectors.npz',
    'ConvNeXt-Base': 'convnext_base_per_image_vectors.npz',
}
METRICS = ['CKA_prev', 'd_within_cos', 'd_between_cos', 'S',
           'd_within_euclid_unit', 'd_between_euclid_unit', 'Fisher_raw',
           'PR', 'mean_raw_norm', 'probe_mean', 'probe_sd']


def stratified_indices(labels, repeats=1000, seed=20261006):
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(np.unique(labels)) < 2 or repeats < 1:
        raise ValueError('Need a label vector, two classes, and positive repeats.')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    return np.concatenate([rng.choice(g, (repeats, len(g)), replace=True)
                           for g in groups], axis=1)


def center_gram(gram):
    return gram - gram.mean(0, keepdims=True) - gram.mean(1, keepdims=True) + gram.mean()


def cka_curve(grams, indices):
    """Recenter AFTER row/column resampling, including duplicated observations."""
    result = np.full(len(grams), np.nan)
    previous = None
    for i, gram in enumerate(grams):
        k = center_gram(gram[np.ix_(indices, indices)])
        norm = np.sqrt(np.sum(k * k))
        if norm <= 1e-20:
            raise ValueError('Degenerate representation: CKA is undefined.')
        k = k / norm
        if previous is not None:
            result[i] = np.sum(previous * k)
        previous = k
    return result


def stage_contrasts(stages):
    """First-block adjacent CKA vs later block adjacencies in the same stage.

    ConvNeXt's downsample adjacency is separately visible in the full curve;
    sK.b1 measures downsample output -> first block, unlike ResNet sK.b1.
    This contrast reduces but does not eliminate depth/architecture confounds.
    """
    groups = {}
    for i, name in enumerate(stages):
        match = re.fullmatch(r's(\d+)\.b(\d+)', str(name))
        if match:
            stage, block = map(int, match.groups())
            groups.setdefault(stage, {})[block] = i
    return [(s, blocks[1], [blocks[b] for b in sorted(blocks) if b > 1])
            for s, blocks in sorted(groups.items()) if 1 in blocks and len(blocks) > 1]


def bootstrap_caches(cache_dir='.', repeats=1000, seed=20261006, progress=print):
    reference = None
    summaries, samples, dips, dip_samples = [], [], [], []
    draw_indices = None
    for model, filename in CACHE_FILES.items():
        path = Path(cache_dir) / filename
        with np.load(path, allow_pickle=False) as data:
            labels = data['labels'].copy()
            paths = data['archive_paths'].copy()
            stages = data['stages'].copy()
            if reference is None:
                reference = (labels, paths)
                if len(paths) != len(set(paths.tolist())):
                    raise ValueError('Sample cache contains duplicate source images.')
                draw_indices = stratified_indices(labels, repeats, seed)
            elif not (np.array_equal(labels, reference[0]) and np.array_equal(paths, reference[1])):
                raise ValueError(f'Mismatched image identities/order: {filename}')
            grams = []
            for i in range(len(stages)):
                x = np.asarray(data[f'features_{i}'], dtype=np.float64)
                if x.ndim != 2 or len(x) != len(labels) or not np.isfinite(x).all():
                    raise ValueError(f'Invalid features: {filename}, layer {i}')
                # Translation invariance allows this precentering for precision.
                x -= x.mean(0, keepdims=True)
                grams.append(x @ x.T)
        original = cka_curve(grams, np.arange(len(labels)))
        boot = np.empty((repeats, len(stages)))
        for b, idx in enumerate(draw_indices):
            boot[b] = cka_curve(grams, idx)
            if progress and ((b + 1) % 100 == 0 or b + 1 == repeats):
                progress(f'{model}: bootstrap {b + 1}/{repeats}', flush=True)
        for depth, stage in enumerate(stages):
            values = boot[:, depth]
            q = [np.nan] * 3 if depth == 0 else np.percentile(values, [10, 50, 90])
            summaries.append(dict(model=model, depth=depth, stage=stage,
                                  estimate=original[depth], p10=q[0], median=q[1], p90=q[2],
                                  n_images=len(labels), n_bootstrap=repeats, seed=seed))
            samples.extend(dict(model=model, replicate=b, depth=depth, stage=stage,
                                CKA_prev=value) for b, value in enumerate(values))
        for stage, boundary, later in stage_contrasts(stages):
            delta = np.median(boot[:, later], axis=1) - boot[:, boundary]
            q = np.percentile(delta, [10, 50, 90])
            dips.append(dict(model=model, stage=f's{stage}', boundary_block=stages[boundary],
                             comparator_blocks=';'.join(stages[later]), n_comparators=len(later),
                             estimate=np.median(original[later]) - original[boundary],
                             p10=q[0], median=q[1], p90=q[2],
                             fraction_positive=float(np.mean(delta > 0)),
                             n_bootstrap=repeats, seed=seed))
            dip_samples.extend(dict(model=model, stage=f's{stage}', replicate=b, delta=value)
                               for b, value in enumerate(delta))
    return { 'CKA_prev_bootstrap': pd.DataFrame(summaries),
             'CKA_prev_bootstrap_replicates': pd.DataFrame(samples),
             'boundary_dip': pd.DataFrame(dips),
             'boundary_dip_replicates': pd.DataFrame(dip_samples),
             'bootstrap_indices': pd.DataFrame(draw_indices),
             'sample_manifest': pd.DataFrame({'row': np.arange(len(reference[0])),
                                             'label': reference[0], 'archive_path': reference[1]}) }


def export_metric_csvs(results, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    keys = ['model', 'depth', 'stage', 'boundary']
    for metric in METRICS:
        results[keys + [metric]].to_csv(output / f'{metric}.csv', index=False)
