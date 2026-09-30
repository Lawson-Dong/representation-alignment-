"""Matched DenseNet-121/169/201 geometry and LLE; run in Colab or Python.

Dense-layer observations pool the full cumulative concatenated state, not
only the 32 newly produced channels. Transitions and final norm+ReLU are
explicit observations. Checkpoints differ, so this is not a depth-only test.
"""
import os
from pathlib import Path
import gc, hashlib, json, time, zipfile, urllib.request, platform
from io import BytesIO
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import sklearn, scipy
from metrics import representation_stats, centered_gram

SEED=42
KS=(4,8,16,32,64)
ROOT=Path(os.environ.get('GEOMETRY_OUTPUT_DIR','outputs/densenet')).resolve()
SPECS=(('DenseNet-121',models.densenet121,models.DenseNet121_Weights.IMAGENET1K_V1,(6,12,24,16)),
       ('DenseNet-169',models.densenet169,models.DenseNet169_Weights.IMAGENET1K_V1,(6,12,32,32)),
       ('DenseNet-201',models.densenet201,models.DenseNet201_Weights.IMAGENET1K_V1,(6,12,48,32)))


def cohort():
    ROOT.mkdir(parents=True,exist_ok=True)
    archive=ROOT/'cats_dogs_light.zip'
    if not archive.exists():
        urllib.request.urlretrieve('https://zenodo.org/records/5226945/files/cats_dogs_light.zip?download=1',archive)
    assert hashlib.md5(archive.read_bytes()).hexdigest()=='5e014163374c3bf7069c923de2d619c8'
    with zipfile.ZipFile(archive) as z:
        names=[n for n in z.namelist() if '/test/' in n.lower()
               and n.lower().endswith(('.jpg','.jpeg','.png'))
               and Path(n).name.lower().startswith(('cat.','dog.'))]
    targets=np.array([3 if Path(n).name.lower().startswith('cat.') else 5 for n in names])
    rng=np.random.default_rng(SEED)
    indices=np.concatenate([rng.choice(np.flatnonzero(targets==c),100,replace=False) for c in (3,5)])
    indices=rng.permutation(indices)
    labels=targets[indices]; paths=[names[i] for i in indices]
    assert len(set(paths))==200 and np.array_equal(np.unique(labels,return_counts=True)[1],[100,100])
    transform=models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    class Images(Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self,i):
            with zipfile.ZipFile(archive) as z:
                image=Image.open(BytesIO(z.read(paths[i]))).convert('RGB')
            return transform(image), int(labels[i])
    manifest=pd.DataFrame(dict(sample_index=np.arange(200),selected_index=indices,image_path=paths,label=labels))
    manifest.to_csv(ROOT/'sample_manifest.csv',index=False)
    return Images(),labels,paths,indices,str(transform)


def extract(net,loader,device,config):
    """Read cumulative states without modifying model activations.

    GAP commutes with channel concatenation. At each dense block entry capture
    its input GAP; append GAP of each dense layer's newly produced channels.
    Verify the last cumulative state against actual dense-block output GAP.
    """
    stages=['stem']; boundaries={'stem'}
    for s,count in enumerate(config,1):
        for j in range(count):
            key=f's{s}.d{j+1}'; stages.append(key)
            if j==0: boundaries.add(key)
        if s<4: stages.append(f'down{s}'); boundaries.add(f'down{s}')
    stages.append('final_norm_relu')
    record={k:[] for k in stages}; cache={}; cumulative={}; handles=[]
    def pooled(out): return out.detach().float().mean((-2,-1)).cpu().numpy()
    def simple(key,relu=False):
        def capture(_m,_inp,out): cache[key]=pooled(torch.relu(out) if relu else out)
        return capture
    handles.append(net.features.pool0.register_forward_hook(simple('stem')))
    for s,count in enumerate(config,1):
        block=getattr(net.features,f'denseblock{s}')
        assert len(block)==count
        def begin(_m,inp,s=s): cumulative[s]=pooled(inp[0])
        handles.append(block.register_forward_pre_hook(begin))
        for j,layer in enumerate(block):
            key=f's{s}.d{j+1}'
            def capture(_m,_inp,out,s=s,key=key):
                cumulative[s]=np.concatenate([cumulative[s],pooled(out)],axis=1)
                cache[key]=cumulative[s]
            handles.append(block[layer].register_forward_hook(capture))
        def check(_m,_inp,out,s=s,count=count):
            np.testing.assert_allclose(cache[f's{s}.d{count}'],pooled(out),rtol=1e-5,atol=1e-6)
        handles.append(block.register_forward_hook(check))
        if s<4: handles.append(getattr(net.features,f'transition{s}').register_forward_hook(simple(f'down{s}')))
    handles.append(net.features.norm5.register_forward_hook(simple('final_norm_relu',True)))
    start=time.time(); seen=[]
    try:
        net.to(device).eval()
        for p in net.parameters(): p.requires_grad_(False)
        with torch.inference_mode():
            for i,(images,y) in enumerate(loader):
                cache.clear(); cumulative.clear(); net(images.to(device))
                assert list(cache)==stages
                for key in stages: record[key].append(cache[key])
                seen.extend(y.numpy().tolist())
                if (i+1)%5==0 or i+1==len(loader): print(f'  batch {i+1}/{len(loader)} {time.time()-start:.1f}s',flush=True)
    finally:
        for h in handles: h.remove()
        net.cpu(); torch.cuda.empty_cache()
    feats={k:np.concatenate(v) for k,v in record.items()}
    return feats,boundaries,np.array(seen)


def point_entropy(nn,y,k):
    p=(y[nn[:,:k]]==5).mean(1)
    return entropy(np.stack([1-p,p],axis=1),base=2,axis=1)


def analyze(feats,name,labels,paths,boundaries,splits,shuffled):
    rows=[]; lle=[]; points=[]; previous=None
    for depth,(stage,x) in enumerate(feats.items()):
        assert x.shape[0]==200 and x.ndim==2 and np.isfinite(x).all()
        assert np.all(np.linalg.norm(x,axis=1)>1e-12)
        gram=centered_gram(x)
        scores=[]
        for train,test in splits:
            p=make_pipeline(Normalizer(),StandardScaler(),LogisticRegression(C=1,max_iter=2000,solver='liblinear',random_state=SEED))
            p.fit(x[train],labels[train]); scores.append(accuracy_score(labels[test],p.predict(x[test])))
        rows.append(dict(model=name,depth=depth,stage=stage,boundary=stage in boundaries,
                         readout_kind='final_norm_relu' if stage=='final_norm_relu' else ('transition' if stage.startswith('down') else 'cumulative_state'),
                         feature_width=x.shape[1],CKA_prev=np.nan if previous is None else float(np.sum(previous*gram)),
                         probe_mean=float(np.mean(scores)),probe_sd=float(np.std(scores,ddof=1)),**representation_stats(x,labels)))
        previous=gram
        d=cosine_distances(x); np.fill_diagonal(d,np.inf)
        nn=np.argsort(d,axis=1,kind='stable')[:,:max(KS)]
        assert not np.any(nn==np.arange(200)[:,None])
        for k in KS:
            h=point_entropy(nn,labels,k)
            null=np.array([point_entropy(nn,y,k).mean() for y in shuffled])
            lle.append(dict(model=name,depth=depth,stage=stage,k=k,LLE=float(h.mean()),shuffle_mean=float(null.mean()),shuffle_sd=float(null.std(ddof=1)),order_vs_shuffle=float(1-h.mean()/null.mean()),fraction_H_ge_0_8=float((h>=.8).mean())))
            points.extend(dict(model=name,depth=depth,stage=stage,k=k,sample_index=i,image_path=paths[i],label=int(labels[i]),H=float(h[i])) for i in range(200))
        if depth%10==0: print(f'  measured {depth+1}/{len(feats)} observations',flush=True)
    df=pd.DataFrame(rows)
    for field in ('S','Fisher_raw','PR','probe_mean'): df['delta_'+field]=df[field].diff()
    return df,pd.DataFrame(lle),pd.DataFrame(points)


def main():
    torch.manual_seed(SEED); np.random.seed(SEED); torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data,labels,paths,indices,transform=cohort()
    loader=DataLoader(data,batch_size=8,shuffle=False,num_workers=0)
    splits=list(StratifiedShuffleSplit(n_splits=5,test_size=.30,random_state=SEED).split(np.zeros(200),labels))
    null_rng=np.random.default_rng(SEED+1000)
    shuffled=np.stack([null_rng.permutation(labels) for _ in range(100)])
    split_rows=[dict(split=s,sample_index=int(i),role=role) for s,(tr,te) in enumerate(splits) for role,idx in [('train',tr),('test',te)] for i in idx]
    pd.DataFrame(split_rows).to_csv(ROOT/'probe_splits.csv',index=False)
    metadata=dict(status='running',seed=SEED,batch_size=8,cohort_size=200,archive_md5='5e014163374c3bf7069c923de2d619c8',
                  preprocessing=transform,k_values=KS,n_shuffles=100,shuffle_seed=SEED+1000,probe='5 paired stratified 70/30 splits; Normalizer+train-only StandardScaler+LogisticRegression(C=1,liblinear)',
                  precision='float32 inference; float64 geometry; sklearn cosine_distances float32 LLE',
                  device=str(device),gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                  python=platform.python_version(),torch=torch.__version__,torchvision=torchvision.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                  pooling='spatial mean of cumulative concatenated dense state; final norm+ReLU separately observed',
                  cautions='Model depth, width and independently trained checkpoints vary; not a causal depth-only comparison. No boundary diagnostic in this run.',models=[])
    (ROOT/'protocol.json').write_text(json.dumps(metadata,indent=2))
    all_geo=[]; all_lle=[]; endpoints=[]
    print('Matched 100 cats + 100 dogs;',device,metadata['gpu'],flush=True)
    for name,constructor,weights,config in SPECS:
        started=time.time(); print('Starting',name,str(weights),flush=True)
        net=constructor(weights=weights)
        count=sum(p.numel() for p in net.parameters())
        feats,boundaries,seen=extract(net,loader,device,config)
        assert np.array_equal(seen,labels)
        assert len(feats)==sum(config)+5
        del net; gc.collect()
        slug=name.lower().replace('-','')
        payload=dict(stages=np.array(list(feats)),labels=labels,archive_paths=np.array(paths),selected_indices=indices,weights=np.array(str(weights)))
        payload.update({f'features_{i}':x for i,x in enumerate(feats.values())})
        np.savez_compressed(ROOT/f'{slug}_per_image_vectors.npz',**payload)
        df,l,p=analyze(feats,name,labels,paths,boundaries,splits,shuffled)
        df.to_csv(ROOT/f'{slug}_geometry.csv',index=False); l.to_csv(ROOT/f'{slug}_lle_summary.csv',index=False)
        p.to_csv(ROOT/f'{slug}_lle_per_image.csv',index=False)
        all_geo.append(df); all_lle.append(l)
        q=l[l.k==16]; dip=df.loc[df.CKA_prev.idxmin()]
        endpoints.append(dict(model=name,observations=len(df),S_first=df.S.iloc[0],S_last=df.S.iloc[-1],within_first=df.d_within_cos.iloc[0],within_last=df.d_within_cos.iloc[-1],LLE_k16_first=q.LLE.iloc[0],LLE_k16_last=q.LLE.iloc[-1],LLE_k16_increasing_transitions=int((q.LLE.diff()>0).sum()),probe_first=df.probe_mean.iloc[0],probe_last=df.probe_mean.iloc[-1],min_CKA=dip.CKA_prev,min_CKA_at=dip.stage))
        metadata['models'].append(dict(model=name,weights=str(weights),checkpoint_url=weights.url,parameters=count,block_config=config,growth_rate=32,bn_size=4,observations=len(df),seconds=time.time()-started,stages=list(feats),widths=[x.shape[1] for x in feats.values()]))
        (ROOT/'protocol.json').write_text(json.dumps(metadata,indent=2))
        print('Finished',name,len(df),'observations',flush=True)
        del feats,p,payload; gc.collect()
    g=pd.concat(all_geo,ignore_index=True); l=pd.concat(all_lle,ignore_index=True)
    g.to_csv(ROOT/'densenet_geometry_metrics.csv',index=False); l.to_csv(ROOT/'densenet_lle_summary.csv',index=False)
    pd.DataFrame(endpoints).to_csv(ROOT/'densenet_endpoint_summary.csv',index=False)
    assert len(g)==253 and len(l)==1265
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(3,2,figsize=(14,11),constrained_layout=True)
    for ax,(field,title) in zip(axs.flat,[('S','Relative separation S'),('CKA_prev','Adjacent raw linear CKA'),('d_within_cos','Within cosine distance'),('d_between_cos','Between cosine distance'),('PR','Participation ratio'),('probe_mean','Held-out probe accuracy')]):
        for name,q in g.groupby('model',sort=False): ax.plot(q.depth,q[field],label=name)
        ax.set(title=title,xlabel='Observation index'); ax.legend(fontsize=8)
    fig.savefig(ROOT/'densenet_geometry.png',dpi=140); plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
    for ax,(name,q) in zip(axs,l.groupby('model',sort=False)):
        for k,v in q.groupby('k'): ax.plot(v.depth,v.LLE,label=f'k={k}')
        ax.set(title=name,xlabel='Observation index',ylabel='Mean local label entropy (bits)',ylim=(0,1));ax.legend(fontsize=8)
    fig.savefig(ROOT/'densenet_lle.png',dpi=140);plt.close(fig)
    metadata['status']='completed'; metadata['completed_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()
    (ROOT/'protocol.json').write_text(json.dumps(metadata,indent=2))
    files={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in ROOT.iterdir() if p.suffix in ('.csv','.json','.png','.npz') and p.name!='artifact_manifest.json'}
    (ROOT/'artifact_manifest.json').write_text(json.dumps(files,indent=2))
    print(pd.DataFrame(endpoints).to_string(index=False),flush=True)
    print('COMPLETED: 253 geometry rows, 1265 LLE summary rows, 253000 per-image LLE rows.',flush=True)

if __name__=='__main__': main()
