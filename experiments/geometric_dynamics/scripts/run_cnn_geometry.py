"""Maintained command-line reproduction of the supplied experiment."""
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from IPython.display import display
OUTPUT = Path(os.environ.get("GEOMETRY_OUTPUT_DIR", "outputs")).resolve()
OUTPUT.mkdir(parents=True, exist_ok=True)
os.chdir(OUTPUT)

def main():
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
    from pathlib import Path
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
    
    
    # 1. Same 100 cats and 100 dogs for every model and every stage.
    # Zenodo Cats and Dogs sample: https://zenodo.org/records/5226945
    # Fixed ResNet-18 V1 preprocessing controls inputs; ResNet-152 V2 has a different preferred recipe.
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
    
    
    splits=list(StratifiedShuffleSplit(n_splits=PROBE_SPLITS,test_size=.30,random_state=SEED).split(np.zeros(len(labels)),labels))
    all_rows=[]; trajectories={}; model_boundaries={}; model_stages={}
    for name,constructor,weights in MODEL_SPECS:
        print('Starting',name,flush=True)
        model=constructor(weights=weights)
        pairs,boundaries=observed_modules(name,model)
        feats=extract_blocks(model,loader,pairs)
        del model; gc.collect()
        model_boundaries[name]=boundaries
        model_stages[name]=list(feats)
        payload = {'stages': np.array(list(feats)), 'labels': labels,
                   'archive_paths': np.array([raw.names[i] for i in indices]),
                   'selected_indices': np.array(indices), 'weights': np.array(str(weights))}
        for j, x in enumerate(feats.values()): payload[f'features_{j}'] = x.astype(np.float32)
        np.savez_compressed(name.lower().replace('-', '').replace('convnext', 'convnext_') + '_per_image_vectors.npz', **payload)
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
    
    
    fig,axs=plt.subplots(3,2,figsize=(16,12),constrained_layout=True)
    measures=[('CKA_prev','Adjacent linear CKA (raw pooled vectors)'),
              ('d_within_cos','Within-class cosine distance'),('d_between_cos','Between-class cosine distance'),
              ('S','Relative separation S'),('Fisher_raw','Raw-vector Fisher ratio'),('PR','Effective dimension PR')]
    for ax,(field,title) in zip(axs.flat,measures):
        for name,g in results.groupby('model',sort=False):
            ax.plot(g.depth,g[field],marker='.',markersize=3,label=name)
        ax.set(title=title,xlabel='Observed block / transition index')
        ax.legend(fontsize=8)
    plt.savefig('cnn_figure_1.png', dpi=160); plt.close()
    
    fig,axs=plt.subplots(1,2,figsize=(16,4),constrained_layout=True)
    for name,g in results.groupby('model',sort=False):
        axs[0].plot(g.depth,g.d_within_euclid_unit,label=f'{name} within')
        axs[0].plot(g.depth,g.d_between_euclid_unit,'--',label=f'{name} between')
        axs[1].plot(g.depth,g.probe_mean,label=name)
        axs[1].fill_between(g.depth,g.probe_mean-g.probe_sd,g.probe_mean+g.probe_sd,alpha=.12)
    axs[0].set(title='Euclidean distances of unit vectors',xlabel='Observed index',ylabel='Mean distance')
    axs[1].axhline(.5,color='gray',linestyle=':')
    axs[1].set(title='Held-out linear probe',xlabel='Observed index',ylabel='Accuracy',ylim=(.4,1.05))
    for ax in axs: ax.legend(fontsize=7,ncol=2)
    plt.savefig('cnn_figure_2.png', dpi=160); plt.close()
    
    
    fig,axs=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
    for ax,(name,coords) in zip(axs.flat,trajectories.items()):
        cat,dog=coords[::2],coords[1::2]
        ax.plot(cat[:,0],cat[:,1],'-o',markersize=2,label='cat centroid')
        ax.plot(dog[:,0],dog[:,1],'-o',markersize=2,label='dog centroid')
        for i in [0,len(cat)-1]:
            ax.annotate(str(i),cat[i],fontsize=8)
            ax.annotate(str(i),dog[i],fontsize=8)
        for i in range(len(cat)):
            ax.plot([cat[i,0],dog[i,0]],[cat[i,1],dog[i,1]],color='gray',alpha=.13)
        ax.set(title=name,xlabel='PCA 1 of similarity fingerprints',ylabel='PCA 2')
        ax.legend(fontsize=8)
    plt.savefig('cnn_figure_3.png', dpi=160); plt.close()
    
    # See transitions, not merely endpoint effects.
    for name,g in results.groupby('model',sort=False):
        print('\n',name)
        display(g.nsmallest(6,'CKA_prev')[['stage','boundary','CKA_prev','S','Fisher_raw','PR','probe_mean']].round(3))
    
    
    # Boundary changes and category-separation increments
    changes = results.copy()
    for field in ['S', 'Fisher_raw', 'PR', 'probe_mean']:
        changes['delta_' + field] = changes.groupby('model', sort=False)[field].diff()
    valid = changes.dropna(subset=['CKA_prev'])
    boundary_summary = (valid.groupby(['model', 'boundary'], sort=False)
                        .agg(n=('CKA_prev','size'), median_CKA=('CKA_prev','median'),
                             median_delta_S=('delta_S','median'), mean_delta_S=('delta_S','mean'))
                        .reset_index())
    display(boundary_summary.round(3))
    for name, g in valid.groupby('model', sort=False):
        print(name, 'largest positive S increments')
        display(g.nlargest(5, 'delta_S')[['stage','boundary','CKA_prev','delta_S','S','delta_Fisher_raw','delta_PR','probe_mean']].round(3))
    print('Boundary is the first block of a ResNet stage, or a ConvNeXt downsample / first stage block. Consecutive measurements are not equal-depth steps.')
    
    
    # Full per-block measurements for later analysis
    results.to_csv('block_geometry_metrics.csv', index=False)
    print('Saved', len(results), 'rows to block_geometry_metrics.csv')

if __name__ == "__main__": main()
