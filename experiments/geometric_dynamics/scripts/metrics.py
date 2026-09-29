"""Geometry definitions from the CNN experiment; labels 3=cat, 5=dog."""
import numpy as np
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

