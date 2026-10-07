"""Reproduce the per-metric notebook and stratified bootstrap (GPU recommended)."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from IPython.display import display

def main():
    output = Path(os.environ.get("GEOMETRY_OUTPUT_DIR", "outputs")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    import torch, torchvision, numpy as np, pandas as pd, matplotlib.pyplot as plt
    from torchvision import models
    from torch.utils.data import Dataset, DataLoader, Subset
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.decomposition import PCA
    from PIL import Image
    from io import BytesIO
    import zipfile, urllib.request, hashlib, gc
    
    SEED=42; N_PER_CLASS=100; BATCH_SIZE=16; PROBE_SPLITS=5
    np.random.seed(SEED); torch.manual_seed(SEED)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Run in order, releasing each network before loading the next.
    MODEL_SPECS=[
     ('ResNet-18',models.resnet18,models.ResNet18_Weights.IMAGENET1K_V1),
     ('ResNet-152',models.resnet152,models.ResNet152_Weights.IMAGENET1K_V2),
     ('ConvNeXt-Tiny',models.convnext_tiny,models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
     ('ConvNeXt-Base',models.convnext_base,models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    ]
    print('torch',torch.__version__,'torchvision',torchvision.__version__,'device',device)
    
    BOOTSTRAP_REPEATS = 1000
    BOOTSTRAP_SEED = 20261006
    RESULT_DIR = Path("block_geometry_metrics")
    RESULT_DIR.mkdir(exist_ok=True)

    # 1. Same 100 cats and 100 dogs for every model and every stage.
    # Zenodo Cats and Dogs sample: https://zenodo.org/records/5226945
    # Fixed ResNet-18 V1 preprocessing for all models; ResNet-152 V2 recommends a different recipe.
    transform = MODEL_SPECS[0][2].transforms()
    # Use this single transform for every architecture and weight condition.
    archive = Path('cats_dogs_light.zip')
    if not archive.exists():
        urllib.request.urlretrieve('https://zenodo.org/records/5226945/files/cats_dogs_light.zip?download=1', archive)
    expected_md5 = '5e014163374c3bf7069c923de2d619c8'
    actual_md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    assert actual_md5 == expected_md5, f'Download checksum mismatch: {actual_md5}'
    
    class ZipCatsDogs(Dataset):
        def __init__(self, path, transform):
            self.path, self.transform = path, transform
            with zipfile.ZipFile(path) as zf:
                self.names = [name for name in zf.namelist()
                              if '/test/' in name.lower() and name.lower().endswith(('.jpg', '.jpeg', '.png'))
                              and Path(name).name.lower().startswith(('cat.', 'dog.'))]
            self.targets = [3 if Path(name).name.lower().startswith('cat.') else 5 for name in self.names]
            assert self.names, 'No labeled cat/dog test images found; inspect archive paths.'
        def __len__(self): return len(self.names)
        def __getitem__(self, index):
            with zipfile.ZipFile(self.path) as zf:
                image = Image.open(BytesIO(zf.read(self.names[index]))).convert('RGB')
            return self.transform(image), self.targets[index]
    
    raw = ZipCatsDogs(archive, transform)
    y_all = np.asarray(raw.targets)
    rng = np.random.default_rng(SEED)
    assert all((y_all == cls).sum() >= N_PER_CLASS for cls in (3, 5)), 'Reduce N_PER_CLASS.'
    indices = np.concatenate([rng.choice(np.flatnonzero(y_all == cls), N_PER_CLASS, replace=False) for cls in (3, 5)])
    indices = rng.permutation(indices).tolist()
    selected = Subset(raw, indices)
    loader = DataLoader(selected, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=(device.type == 'cuda'))
    labels = np.asarray([y_all[i] for i in indices])
    print(f'{len(selected)} images: {(labels == 3).sum()} cats, {(labels == 5).sum()} dogs')

    # The hook list follows actual module execution order. Stage boundaries are explicit.
    def observed_modules(name,model):
        pairs=[]; boundaries=[]
        if name.startswith('ResNet'):
            pairs.append(('stem',model.maxpool)); boundaries.append('stem')
            for i in range(1,5):
                for j,block in enumerate(getattr(model,f'layer{i}')):
                    label=f's{i}.b{j+1}'
                    pairs.append((label,block))
                    if j==0: boundaries.append(label)
        else:
            for idx,module in enumerate(model.features):
                if idx==0:
                    pairs.append(('stem',module)); boundaries.append('stem')
                elif idx%2==0:
                    label=f'down{idx//2}'
                    pairs.append((label,module)); boundaries.append(label)
                else:
                    stage=(idx+1)//2
                    for j,block in enumerate(module):
                        label=f's{stage}.b{j+1}'
                        pairs.append((label,block))
                        if j==0: boundaries.append(label)
        return pairs,boundaries
    
    def extract_blocks(model,loader,pairs):
        model=model.to(device).eval()
        record={key:[] for key,_ in pairs}; cache={}; hooks=[]
        for key,module in pairs:
            def grab(m,inp,out,key=key):
                h=out.detach()
                if h.ndim==4: h=h.mean((-2,-1))
                cache[key]=h.flatten(1).cpu()
            hooks.append(module.register_forward_hook(grab))
        try:
            with torch.inference_mode():
                for images,_ in loader:
                    _=model(images.to(device))
                    for key in record: record[key].append(cache.pop(key))
        finally:
            for h in hooks: h.remove()
            model.cpu()
            if device.type=='cuda': torch.cuda.empty_cache()
        return {key:torch.cat(parts).numpy() for key,parts in record.items()}

    # Run after the notebook's setup, dataset, and hook-function cells.
    # The saved rows preserve the selected image order across all nine stages.
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    pairs, _ = observed_modules('ResNet-18', model)
    feats = extract_blocks(model, loader, pairs)
    del model
    stage_names = np.array(list(feats))
    names = np.array([raw.names[i] for i in indices])
    assert len(stage_names) == 9 and len(names) == len(labels) == 200
    payload = {'stages': stage_names, 'labels': labels, 'archive_paths': names,
               'selected_indices': np.array(indices)}
    for j, stage in enumerate(stage_names):
        payload[f'features_{j}'] = feats[stage].astype(np.float32)
    out = 'resnet18_per_image_vectors.npz'
    np.savez_compressed(out, **payload)
    print('Saved', out, 'stages:', list(stage_names),
          'shapes:', [feats[s].shape for s in stage_names],
          'label counts:', np.unique(labels, return_counts=True))

    # Run after setup, dataset, and hook-function cells in this notebook.
    # The sample order and preprocessing are identical to the ResNet-18 export.
    model152 = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V2)
    pairs152, _ = observed_modules('ResNet-152', model152)
    feats152 = extract_blocks(model152, loader, pairs152)
    del model152
    stages152 = np.array(list(feats152))
    assert len(stages152) == 51 and len(labels) == 200
    payload152 = {
        'stages': stages152,
        'labels': labels,
        'archive_paths': np.array([raw.names[i] for i in indices]),
        'selected_indices': np.array(indices),
    }
    for j, stage in enumerate(stages152):
        payload152[f'features_{j}'] = feats152[stage].astype(np.float32)
    out152 = 'resnet152_per_image_vectors.npz'
    np.savez_compressed(out152, **payload152)
    print('Saved', out152, 'points:', len(stages152),
          'first/last:', stages152[0], stages152[-1],
          'feature widths:', sorted(set(x.shape[1] for x in feats152.values())),
          'label counts:', np.unique(labels, return_counts=True))
    del feats152, payload152
    gc.collect()

    # Same selected 200 images and transform as the ResNet experiments.
    model_tiny = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    pairs_tiny, _ = observed_modules('ConvNeXt-Tiny', model_tiny)
    feats_tiny = extract_blocks(model_tiny, loader, pairs_tiny)
    del model_tiny
    stages_tiny = np.array(list(feats_tiny))
    assert len(stages_tiny) == 22 and len(labels) == 200
    payload_tiny = {
        'stages': stages_tiny,
        'labels': labels,
        'archive_paths': np.array([raw.names[i] for i in indices]),
        'selected_indices': np.array(indices),
    }
    for j, stage in enumerate(stages_tiny):
        payload_tiny[f'features_{j}'] = feats_tiny[stage].astype(np.float32)
    out_tiny = 'convnext_tiny_per_image_vectors.npz'
    np.savez_compressed(out_tiny, **payload_tiny)
    print('Saved', out_tiny, 'points:', len(stages_tiny),
          'stages:', list(stages_tiny),
          'feature widths:', sorted(set(x.shape[1] for x in feats_tiny.values())),
          'label counts:', np.unique(labels, return_counts=True))
    del feats_tiny, payload_tiny
    gc.collect()

    # Same selected 200 images and transform as the ResNet experiments.
    model_base = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    pairs_base, _ = observed_modules('ConvNeXt-Base', model_base)
    feats_base = extract_blocks(model_base, loader, pairs_base)
    del model_base
    stages_base = np.array(list(feats_base))
    assert len(stages_base) == 40 and len(labels) == 200
    payload_base = {
        'stages': stages_base,
        'labels': labels,
        'archive_paths': np.array([raw.names[i] for i in indices]),
        'selected_indices': np.array(indices),
    }
    for j, stage in enumerate(stages_base):
        payload_base[f'features_{j}'] = feats_base[stage].astype(np.float32)
    out_base = 'convnext_base_per_image_vectors.npz'
    np.savez_compressed(out_base, **payload_base)
    print('Saved', out_base, 'points:', len(stages_base),
          'stages:', list(stages_base),
          'feature widths:', sorted(set(x.shape[1] for x in feats_base.values())),
          'label counts:', np.unique(labels, return_counts=True))
    del feats_base, payload_base
    gc.collect()

    # Metrics: cosine pairwise geometry on per-image unit vectors; raw Fisher ratio.
    # Squared Euclidean distance on unit vectors is 2 * cosine distance, so these are not independent measurements.
    def representation_stats(x,y):
        x=np.asarray(x,dtype=np.float64)
        z=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        cat,dog=z[y==3],z[y==5]
        def within_cos(q):
            n=len(q)
            return 1-(np.sum(q.sum(axis=0)**2)-n)/(n*(n-1))
        dc=(within_cos(cat)+within_cos(dog))/2
        db=1-cat.mean(axis=0)@dog.mean(axis=0)
        # For mean *Euclidean* pairwise distances, use a 200x200 distance matrix.
        sim=np.clip(z@z.T,-1,1)
        dist=np.sqrt(np.maximum(0,2-2*sim))
        ci=np.flatnonzero(y==3); di=np.flatnonzero(y==5)
        wc=dist[np.ix_(ci,ci)][np.triu_indices(len(ci),1)].mean()
        wd=dist[np.ix_(di,di)][np.triu_indices(len(di),1)].mean()
        edw=(wc+wd)/2; edb=dist[np.ix_(ci,di)].mean()
        # Fisher on raw, globally pooled features, invariant to a global scalar only.
        a,b=x[y==3],x[y==5]; ma,mb=a.mean(0),b.mean(0)
        var=np.mean(np.sum((a-ma)**2,axis=1))+np.mean(np.sum((b-mb)**2,axis=1))
        fisher=np.sum((ma-mb)**2)/max(var,1e-20)
        centered=z-z.mean(0)
        eigen=np.maximum(np.linalg.eigvalsh(centered@centered.T),0)
        pr=eigen.sum()**2/max(np.square(eigen).sum(),1e-20)
        return {'d_within_cos':dc,'d_between_cos':db,'S':db/dc,
                'd_within_euclid_unit':edw,'d_between_euclid_unit':edb,
                'Fisher_raw':fisher,'PR':pr,'mean_raw_norm':np.linalg.norm(x,axis=1).mean()}
    
    def centered_gram(x):
        x=np.asarray(x,dtype=np.float64)
        x=x-x.mean(0,keepdims=True)
        gram=x@x.T
        return gram/np.maximum(np.linalg.norm(gram,'fro'),1e-20)
    
    def linear_cka(x1,x2):
        return float(np.sum(centered_gram(x1)*centered_gram(x2)))
    
    def probe_scores(x,y,splits):
        acc=[]
        for train,test in splits:
            p=make_pipeline(Normalizer(),StandardScaler(),LogisticRegression(C=1,max_iter=2000,solver='liblinear',random_state=SEED))
            p.fit(x[train],y[train]); acc.append(accuracy_score(y[test],p.predict(x[test])))
        return float(np.mean(acc)),float(np.std(acc,ddof=1))
    
    def relative_geometry_trajectory(feats,y):
        # Shared coordinates: each sample's similarities to the SAME ordered set of images.
        # Center and unit-normalize each sample Gram matrix before pooling all class centroids in a single PCA.
        points=[]
        for x in feats.values():
            z=x.astype(np.float64); z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            k=centered_gram(z)
            points.extend([k[y==3].mean(0),k[y==5].mean(0)])
        return PCA(n_components=2).fit_transform(np.asarray(points))

    from stratified_bootstrap import CACHE_FILES, bootstrap_caches, export_metric_csvs

    splits=list(StratifiedShuffleSplit(n_splits=PROBE_SPLITS,test_size=.30,random_state=SEED).split(np.zeros(len(labels)),labels))
    all_rows=[]; trajectories={}; model_boundaries={}; model_stages={}
    for name,constructor,weights in MODEL_SPECS:
        print('Starting',name,flush=True)
        with np.load(CACHE_FILES[name], allow_pickle=False) as data:
            assert np.array_equal(data['labels'], labels)
            assert np.array_equal(data['archive_paths'], np.array([raw.names[i] for i in indices]))
            feats = {str(stage): data[f'features_{j}'].copy() for j, stage in enumerate(data['stages'])}
        boundaries = [stage for stage in feats if stage == 'stem' or stage.startswith('down') or stage.endswith('.b1')]
        pairs = list(feats.items())
        model_boundaries[name]=boundaries
        model_stages[name]=list(feats)
        trajectories[name]=relative_geometry_trajectory(feats,labels)
        previous=None
        for depth,(stage,x) in enumerate(feats.items()):
            stats=representation_stats(x,labels)
            mean,sd=probe_scores(x,labels,splits)
            all_rows.append({'model':name,'depth':depth,'stage':stage,'boundary':stage in boundaries,
                             'CKA_prev':np.nan if previous is None else linear_cka(previous,x),
                             'probe_mean':mean,'probe_sd':sd,**stats})
            previous=x
        del feats,previous
        print('Finished',name,'measurement points',len(pairs),flush=True)
    results=pd.DataFrame(all_rows)
    display(results.groupby('model',sort=False).agg(points=('stage','size'),
        min_adjacent_CKA=('CKA_prev','min'),S_first=('S','first'),S_last=('S','last'),
        probe_first=('probe_mean','first'),probe_last=('probe_mean','last')).round(3))

    bootstrap = bootstrap_caches('.', BOOTSTRAP_REPEATS, BOOTSTRAP_SEED)
    for filename, frame in bootstrap.items():
        frame.to_csv(RESULT_DIR / f'{filename}.csv', index=False)
    display(bootstrap['CKA_prev_bootstrap'].head())

    # CKA prev: one independent figure per architecture.
    results[['model','depth','stage','boundary','CKA_prev']].to_csv(RESULT_DIR / 'CKA_prev.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['CKA_prev'], '.-', label='Original sample')
        band = bootstrap['CKA_prev_bootstrap'].query('model == @name').sort_values('depth')
        assert np.array_equal(g.depth.to_numpy(), band.depth.to_numpy())
        ax.fill_between(band.depth.to_numpy(), band.p10.to_numpy(), band.p90.to_numpy(), alpha=.2,
                        label='Pointwise bootstrap p10–p90 (80%)')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — CKA prev', xlabel='Observed layer / transition index', ylabel='CKA_prev')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_CKA_prev.png'), dpi=160)
        plt.show()

    # d within cos: one independent figure per architecture.
    results[['model','depth','stage','boundary','d_within_cos']].to_csv(RESULT_DIR / 'd_within_cos.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['d_within_cos'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — d within cos', xlabel='Observed layer / transition index', ylabel='d_within_cos')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_d_within_cos.png'), dpi=160)
        plt.show()

    # d between cos: one independent figure per architecture.
    results[['model','depth','stage','boundary','d_between_cos']].to_csv(RESULT_DIR / 'd_between_cos.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['d_between_cos'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — d between cos', xlabel='Observed layer / transition index', ylabel='d_between_cos')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_d_between_cos.png'), dpi=160)
        plt.show()

    # S: one independent figure per architecture.
    results[['model','depth','stage','boundary','S']].to_csv(RESULT_DIR / 'S.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['S'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — S', xlabel='Observed layer / transition index', ylabel='S')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_S.png'), dpi=160)
        plt.show()

    # d within euclid unit: one independent figure per architecture.
    results[['model','depth','stage','boundary','d_within_euclid_unit']].to_csv(RESULT_DIR / 'd_within_euclid_unit.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['d_within_euclid_unit'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — d within euclid unit', xlabel='Observed layer / transition index', ylabel='d_within_euclid_unit')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_d_within_euclid_unit.png'), dpi=160)
        plt.show()

    # d between euclid unit: one independent figure per architecture.
    results[['model','depth','stage','boundary','d_between_euclid_unit']].to_csv(RESULT_DIR / 'd_between_euclid_unit.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['d_between_euclid_unit'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — d between euclid unit', xlabel='Observed layer / transition index', ylabel='d_between_euclid_unit')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_d_between_euclid_unit.png'), dpi=160)
        plt.show()

    # Fisher raw: one independent figure per architecture.
    results[['model','depth','stage','boundary','Fisher_raw']].to_csv(RESULT_DIR / 'Fisher_raw.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['Fisher_raw'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — Fisher raw', xlabel='Observed layer / transition index', ylabel='Fisher_raw')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_Fisher_raw.png'), dpi=160)
        plt.show()

    # PR: one independent figure per architecture.
    results[['model','depth','stage','boundary','PR']].to_csv(RESULT_DIR / 'PR.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['PR'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — PR', xlabel='Observed layer / transition index', ylabel='PR')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_PR.png'), dpi=160)
        plt.show()

    # mean raw norm: one independent figure per architecture.
    results[['model','depth','stage','boundary','mean_raw_norm']].to_csv(RESULT_DIR / 'mean_raw_norm.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['mean_raw_norm'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — mean raw norm', xlabel='Observed layer / transition index', ylabel='mean_raw_norm')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_mean_raw_norm.png'), dpi=160)
        plt.show()

    # probe mean: one independent figure per architecture.
    results[['model','depth','stage','boundary','probe_mean']].to_csv(RESULT_DIR / 'probe_mean.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['probe_mean'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — probe mean', xlabel='Observed layer / transition index', ylabel='probe_mean')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_probe_mean.png'), dpi=160)
        plt.show()

    # probe sd: one independent figure per architecture.
    results[['model','depth','stage','boundary','probe_sd']].to_csv(RESULT_DIR / 'probe_sd.csv', index=False)
    for name, g in results.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        ax.plot(g.depth, g['probe_sd'], '.-', label='Original sample')
        for depth in g.loc[g.boundary & (g.depth > 0), 'depth']:
            ax.axvline(depth, color='gray', linestyle=':', alpha=.35)
        ax.set(title=name + ' — probe sd', xlabel='Observed layer / transition index', ylabel='probe_sd')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_probe_sd.png'), dpi=160)
        plt.show()

    trajectory_rows = []
    for name, coords in trajectories.items():
        g = results.query('model == @name').sort_values('depth')
        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for offset, category in [(0, 'cat'), (1, 'dog')]:
            xy = coords[offset::2]
            ax.plot(xy[:, 0], xy[:, 1], '-o', markersize=3, label=category + ' centroid')
            for j, row in enumerate(g.itertuples()):
                trajectory_rows.append(dict(model=name, depth=row.depth, stage=row.stage, category=category,
                                            pca1=xy[j, 0], pca2=xy[j, 1]))
        ax.set(title=name + ' — similarity-fingerprint trajectory', xlabel='PCA 1', ylabel='PCA 2')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_similarity_fingerprint_trajectory.png'), dpi=160)
        plt.show()
    pd.DataFrame(trajectory_rows).to_csv(RESULT_DIR / 'similarity_fingerprint_trajectory.csv', index=False)

    dip = bootstrap['boundary_dip']
    display(dip)
    for name, g in dip.groupby('model', sort=False):
        fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
        y = np.arange(len(g))
        ax.hlines(y, g.p10, g.p90, linewidth=3, label='Bootstrap p10–p90 (80%)')
        ax.scatter(g.estimate, y, label='Original Δ', zorder=3)
        ax.axvline(0, color='gray', linestyle=':')
        ax.set_yticks(y, g.stage)
        ax.set(title=name + ' — within-stage dip', xlabel='Median later-block CKA − first-block CKA')
        ax.legend()
        fig.savefig(RESULT_DIR / (name + '_boundary_dip.png'), dpi=160)
        plt.show()

    import json, platform, shutil
    export_metric_csvs(results, RESULT_DIR)
    assert len(results) == 122
    assert all(np.isfinite(results.loc[results.depth > 0, 'CKA_prev']))
    for name, g in results.groupby('model', sort=False):
        check = bootstrap['CKA_prev_bootstrap'].query('model == @name')
        assert np.allclose(g.CKA_prev, check.estimate, equal_nan=True, atol=1e-10)
    record = dict(status='completed', bootstrap_repeats=BOOTSTRAP_REPEATS,
                  bootstrap_seed=BOOTSTRAP_SEED, sample_seed=SEED,
                  interval='pointwise p10-p90 (central 80%)',
                  classes={str(c): int((labels == c).sum()) for c in np.unique(labels)},
                  models={name: str(weights) for name, _, weights in MODEL_SPECS},
                  preprocessing=str(transform), python=platform.python_version(),
                  torch=torch.__version__, torchvision=torchvision.__version__,
                  numpy=np.__version__, pandas=pd.__version__)
    record['csv_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(RESULT_DIR.glob('*.csv'))}
    (RESULT_DIR / 'run_record.json').write_text(json.dumps(record, indent=2))
    shutil.make_archive(str(RESULT_DIR), 'zip', root_dir=RESULT_DIR.parent, base_dir=RESULT_DIR.name)
    print('Verified and packed', RESULT_DIR.with_suffix('.zip'))

    from datetime import datetime, timezone
    record = json.loads((RESULT_DIR / 'run_record.json').read_text())
    record.update(executed_at_utc=datetime.now(timezone.utc).isoformat(),
                  execution_backend='Python CLI',
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, dataset_md5=actual_md5,
                  feature_cache_sha256={filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                                        for filename in CACHE_FILES.values()})
    record['csv_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(RESULT_DIR.glob('*.csv'))}
    (RESULT_DIR / 'run_record.json').write_text(json.dumps(record, indent=2))
    shutil.make_archive(str(RESULT_DIR), 'zip', root_dir=RESULT_DIR.parent, base_dir=RESULT_DIR.name)
    print('Completed run:', len(results), 'measurement points;', len(list(RESULT_DIR.glob('*.csv'))), 'CSV files')

if __name__ == "__main__":
    main()
