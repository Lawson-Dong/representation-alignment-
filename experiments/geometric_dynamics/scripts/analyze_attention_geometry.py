"""Geometry metrics and descriptive plots for matched-image attention models."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__import__("os").environ.get("GEOMETRY_OUTPUT_DIR", "outputs")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
SLUGS=('vit_b16_cls','vit_b16_patchmean','swin_t')

def stats(x,y):
    x=np.asarray(x,dtype=np.float64)
    z=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    a,b=z[y==3],z[y==5]
    def wc(q):
        n=len(q)
        return 1-(np.square(q.sum(0)).sum()-n)/(n*(n-1))
    within=(wc(a)+wc(b))/2
    between=1-a.mean(0)@b.mean(0)
    ac,bc=x[y==3],x[y==5]
    ma,mb=ac.mean(0),bc.mean(0)
    spread=np.square(ac-ma).sum(1).mean()+np.square(bc-mb).sum(1).mean()
    centered=z-z.mean(0)
    ev=np.maximum(np.linalg.eigvalsh(centered@centered.T),0)
    return dict(d_within_cos=within,d_between_cos=between,S=between/within,
                Fisher_raw=np.square(ma-mb).sum()/max(spread,1e-20),
                PR=ev.sum()**2/max(np.square(ev).sum(),1e-20))

def gram(x):
    x=x.astype(np.float64);x=x-x.mean(0)
    k=x@x.T
    return k/max(np.linalg.norm(k),1e-20)

def main():
    rows=[]
    ref=np.load(ROOT/f'{SLUGS[0]}_per_image_vectors.npz')
    for slug in SLUGS:
        a=np.load(ROOT/f'{slug}_per_image_vectors.npz')
        assert np.array_equal(a['archive_paths'],ref['archive_paths'])
        assert np.array_equal(a['labels'],ref['labels'])
        prev=None
        for i,stage in enumerate(a['stages']):
            x=a[f'features_{i}'];g=gram(x)
            rows.append(dict(slug=slug,model=str(a['model']),readout=str(a['readout']),
                             depth=i,stage=str(stage),feature_width=x.shape[1],
                             CKA_prev=np.nan if prev is None else np.sum(g*prev),
                             **stats(x,a['labels'])))
            prev=g
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'attention_geometry_metrics.csv',index=False)
    summary=[]
    for slug,g in df.groupby('slug',sort=False):
        i=g.CKA_prev.idxmin()
        summary.append(dict(slug=slug,observations=len(g),S_first=float(g.S.iloc[0]),
                            S_last=float(g.S.iloc[-1]),min_CKA=float(g.loc[i,'CKA_prev']),
                            min_CKA_at=g.loc[i,'stage'],PR_last=float(g.PR.iloc[-1]),
                            Fisher_last=float(g.Fisher_raw.iloc[-1])))
    (ROOT/'attention_geometry_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    fig,axs=plt.subplots(3,2,figsize=(13,10),constrained_layout=True)
    fields=(('CKA_prev','Adjacent linear CKA (raw vectors)'),('S','Relative separation S'),
            ('d_within_cos','Within-class cosine distance'),('d_between_cos','Between-class cosine distance'),
            ('PR','Participation ratio (unit vectors)'),('Fisher_raw','Fisher ratio (raw vectors)'))
    for ax,(f,title) in zip(axs.flat,fields):
        for slug,g in df.groupby('slug',sort=False):
            ax.plot(g.depth,g[f],'-o',ms=3,label=f'{g.model.iloc[0]} / {g.readout.iloc[0]}')
        ax.set(title=title,xlabel='Observed output index (not equal architectural depth)')
        ax.grid(alpha=.22)
        ax.legend(fontsize=8)
    fig.suptitle('Same 200 cat/dog images; fixed preprocessing; descriptive cross-architecture comparison',fontsize=13)
    fig.savefig(ROOT/'attention_geometry_metrics.png',dpi=140)

if __name__=='__main__':main()
