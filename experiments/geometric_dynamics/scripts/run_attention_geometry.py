"""Matched-image ViT-B/16 and Swin-T geometry. Run in Python or Colab.

Dependencies: torch, torchvision, numpy. Uses the same archive, seed, and
ResNet-18 V1 preprocessing as the preceding CNN experiment.
"""
import os
import sys
from pathlib import Path
ROOT = Path(__import__("os").environ.get("GEOMETRY_OUTPUT_DIR", "outputs")).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.append(str(ROOT / 'vision_deps'))
os.environ.setdefault('TORCH_HOME', str(ROOT / 'model_cache'))
import gc
import hashlib
import json
import time
import urllib.request
import zipfile
from io import BytesIO
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import models
from torch.utils.data import Dataset, DataLoader

def dataset():
    archive = ROOT / 'cats_dogs_light.zip'
    if not archive.exists():
        print('Downloading matched dataset', flush=True)
        urllib.request.urlretrieve('https://zenodo.org/records/5226945/files/cats_dogs_light.zip?download=1', archive)
    assert hashlib.md5(archive.read_bytes()).hexdigest() == '5e014163374c3bf7069c923de2d619c8'
    transform = models.ResNet18_Weights.IMAGENET1K_V1.transforms()
    class Images(Dataset):
        def __init__(self):
            with zipfile.ZipFile(archive) as z:
                self.names = [n for n in z.namelist() if '/test/' in n.lower()
                              and n.lower().endswith(('.jpg', '.jpeg', '.png'))
                              and Path(n).name.lower().startswith(('cat.', 'dog.'))]
            targets = np.array([3 if Path(n).name.lower().startswith('cat.') else 5 for n in self.names])
            rng = np.random.default_rng(42)
            self.indices = np.concatenate([rng.choice(np.flatnonzero(targets == c), 100, replace=False) for c in (3, 5)])
            self.indices = rng.permutation(self.indices)
            self.labels = targets[self.indices]
            self.paths = np.array([self.names[i] for i in self.indices])
            reference = ROOT / 'resnet18_per_image_vectors.npz'
            if reference.exists():
                with np.load(reference) as a:
                    assert np.array_equal(a['archive_paths'], self.paths)
                    assert np.array_equal(a['labels'], self.labels)
        def __len__(self): return 200
        def __getitem__(self, j):
            with zipfile.ZipFile(archive) as z:
                image = Image.open(BytesIO(z.read(self.paths[j]))).convert('RGB')
            return transform(image)
    return Images(), str(transform)

def extract(model, observations, loader, device, readers):
    record = {r: {name: [] for name, _ in observations} for r in readers}
    hooks = []
    for name, module in observations:
        def capture(m, inp, out, name=name):
            for r, pool in readers.items():
                record[r][name].append(pool(out.detach()).cpu().numpy())
        hooks.append(module.register_forward_hook(capture))
    start = time.time()
    try:
        model.eval().to(device)
        with torch.inference_mode():
            for j, batch in enumerate(loader):
                model(batch.to(device))
                print(f'  batch {j+1}/{len(loader)}, {time.time()-start:.1f}s', flush=True)
    finally:
        for hook in hooks: hook.remove()
        model.cpu()
        if device.type == 'cuda': torch.cuda.empty_cache()
    return {r: {k: np.concatenate(v) for k, v in feats.items()} for r, feats in record.items()}

def save(slug, model_name, readout, feats, data, transform, weight):
    # Before attention the CLS token is identical across images, so cosine
    # geometry/CKA is degenerate. Exclude it from this readout, explicitly.
    if readout == 'CLS': feats.pop('tokens+pos', None)
    payload = dict(stages=np.array(list(feats)), labels=data.labels,
                   archive_paths=data.paths, selected_indices=data.indices,
                   model=np.array(model_name), readout=np.array(readout),
                   weights=np.array(weight), preprocessing=np.array(transform))
    for j, (name, x) in enumerate(feats.items()):
        assert x.ndim == 2 and x.shape[0] == 200 and np.isfinite(x).all(), (name, x.shape)
        payload[f'features_{j}'] = x.astype(np.float32)
    np.savez_compressed(ROOT / f'{slug}_per_image_vectors.npz', **payload)
    print('Saved', slug, len(feats), 'observations', flush=True)

def main():
    torch.manual_seed(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data, transform = dataset()
    loader = DataLoader(data, batch_size=8, shuffle=False, num_workers=0)
    print('Matched images:', len(data), 'device:', device, flush=True)
    print('Starting ViT-B/16', flush=True)
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    obs = [('tokens+pos', model.encoder.dropout)]
    obs += [(f'b{i+1}', b) for i, b in enumerate(model.encoder.layers)]
    obs += [('final_norm', model.encoder.ln)]
    feats = extract(model, obs, loader, device,
                    {'CLS': lambda h: h[:, 0, :], 'patch_mean': lambda h: h[:, 1:, :].mean(1)})
    save('vit_b16_cls', 'ViT-B/16', 'CLS', feats['CLS'], data, transform, 'IMAGENET1K_V1')
    save('vit_b16_patchmean', 'ViT-B/16', 'patch_mean', feats['patch_mean'], data, transform, 'IMAGENET1K_V1')
    del model, feats, obs
    gc.collect()
    print('Starting Swin-T', flush=True)
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    obs = []
    for i, m in enumerate(model.features):
        if i == 0: obs.append(('stem', m))
        elif i % 2 == 0: obs.append((f'merge{i//2}', m))
        else: obs.extend((f's{(i+1)//2}.b{j+1}', b) for j, b in enumerate(m))
    obs.append(('final_norm', model.norm))
    feats = extract(model, obs, loader, device, {'spatial_mean': lambda h: h.mean((1, 2))})
    save('swin_t', 'Swin-T', 'spatial_mean', feats['spatial_mean'], data, transform, 'IMAGENET1K_V1')
    manifest = dict(torch=torch.__version__, torchvision=torchvision.__version__, device=str(device),
                    seed=42, cohort_size=200, class_counts={'cat':100,'dog':100},
                    transform=transform, output_basis='raw pooled block outputs; final_norm is explicit',
                    cls_start='b1; constant input CLS excluded',
                    caveat='Fixed ResNet-18 transform across architectures; checkpoint recipes differ.')
    (ROOT / 'attention_geometry_provenance.json').write_text(json.dumps(manifest, indent=2))

if __name__ == '__main__': main()
