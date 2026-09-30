# Representation-vector geometric dynamics

Exploratory, frozen-network experiments on the same 100 cat and 100 dog images. The question is how class geometry and local label mixing change across successive block outputs. This is forward-pass geometry, not training-time dynamics or a test of alignment interventions.

## Experiments

| Entry point | Models / readouts | Observations | Checkpoints |
|---|---|---|---|
| `scripts/run_cnn_geometry.py` | ResNet-18, ResNet-152, ConvNeXt-Tiny, ConvNeXt-Base | 9, 51, 22, 40 | ResNet-152 V2; others V1 |
| `scripts/run_lle.py` | Same four CNNs; cosine kNN local label entropy | 9, 51, 22, 40 | ResNet-152 V2; others V1 |
| `scripts/run_attention_geometry.py` | ViT-B/16 CLS, ViT-B/16 patch mean, Swin-T | 13, 14, 17 | All V1 |
| `scripts/run_densenet_geometry.py` | DenseNet-121, DenseNet-169, DenseNet-201; cumulative states + final norm/ReLU | 63, 87, 103 | All V1 |

Checkpoints are explicit torchvision ImageNet-1K enums. **ResNet-152 LLE and geometry now both use V2 weights.** The completed Colab V2 rerun replaces the previous V1 LLE notebook; other model configurations are unchanged.

## Reproduce

From the repository root, use Python 3.11 or 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r experiments/geometric_dynamics/requirements.txt
export GEOMETRY_OUTPUT_DIR="$PWD/outputs/geometric_dynamics"
python experiments/geometric_dynamics/scripts/run_cnn_geometry.py
python experiments/geometric_dynamics/scripts/run_lle.py
python experiments/geometric_dynamics/scripts/run_densenet_geometry.py
python experiments/geometric_dynamics/scripts/run_attention_geometry.py
python experiments/geometric_dynamics/scripts/analyze_attention_geometry.py
python experiments/geometric_dynamics/scripts/make_attention_video.py vit_b16_cls
python experiments/geometric_dynamics/scripts/make_attention_video.py vit_b16_patchmean
python experiments/geometric_dynamics/scripts/make_attention_video.py swin_t
```

Install an appropriate matching torch/torchvision build for your CPU or CUDA environment. GPU is recommended; CPU is supported but slow for the largest models. Downloads require internet. MP4 generation additionally requires `ffmpeg` with libx264 on PATH. Scripts save outputs in the selected directory (default: `outputs` relative to the invocation directory). CNN scripts export all four per-image NPZ files; the attention analysis checks matching sample paths and labels across its three readouts, without requiring a prior CNN run. Attention extraction additionally checks a ResNet-18 reference if present.

The [new DenseNet notebook](notebooks/Cat_Dog_DenseNet_121_169_201_Matched_Geometry_LLE.ipynb) was executed on the specified Colab T4 on 2026-09-30. Its [CSV exports and provenance](results/densenet_20260930/) are preserved with nine passing offline checks.

The earlier geometry and attention `notebooks/` files remain historical uploads. The LLE notebook has been replaced in place with the completed Colab ResNet-152 V2 rerun; only its execution-status markdown was updated after upload, preserving every code cell and output. The source manifest records the current notebook hashes. For local/headless runs use `scripts/`. See [protocol and audit](docs/protocol.md) for execution evidence, corrected stale notebook text, and interpretation limits.

## Measurements

- **S:** mean between-class cosine distance / balanced mean within-class cosine distance. Normalize each image vector; exclude self-pairs within classes. Inspect numerator and denominator separately: higher S need not mean absolute contraction.
- **Adjacent linear CKA:** Frobenius inner product of normalized centered raw sample Gram matrices. Feature widths can differ.
- **LLE:** local **label** entropy in bits, not locally linear embedding. Exclude the query from its cosine neighbors; stable sorting breaks ties by sample order. Use k = 4, 8, 16, 32, 64. Compare with 100 shuffled-label baselines; cross-model runs share permutations.
- **Boundary diagnostic:** ResNet-18 only; five-fold held-out logistic hyperplanes after normalization and fold-fitted standardization. High entropy is H >= 0.8; near-boundary means the lowest quartile of absolute margins. k sensitivity includes exploratory hypergeometric enrichment tests.
- **Distance and scale measurements:** within- and between-class cosine distances, unit-vector Euclidean distances, raw Fisher ratio and mean raw-vector norm. Unit-vector Euclidean and cosine distances describe related pairwise geometry, not independent evidence.
- **PR:** participation ratio of the centered unit-vector sample Gram spectrum; a spectral effective-dimension estimate. Both CNN and attention geometry analyses compute PR.
- **Linear-probe accuracy:** CNNs only; logistic regression after unit normalization and training-split standardization, evaluated on five paired stratified 70/30 shuffle splits. Report mean accuracy and sample standard deviation. This differs from the five-fold boundary diagnostic.
- **Transition measurements:** adjacent changes in S, Fisher ratio, PR and probe accuracy, computed in CNN transition analysis.
- **Entropy-derived measurements:** relative order against shuffled-label entropy, high-entropy fraction, endpoint entropy decrease and number of entropy-increasing transitions. Boundary analysis includes absolute margins, entropy–margin Spearman correlation, proximity AUC, near/far high-entropy rates, median margins, enrichment checks and high-entropy misclassification.

See the [full metric definitions](../../README.md#main-measurements). Attention analysis computes cosine distances, S, CKA, Fisher ratio and PR; LLE and linear probes are implemented for the CNN experiments, not the attention readouts.

Animations use one PCA of same-image cosine fingerprints across all observations within a video. Endpoints are measured vectors; intermediate motion is interpolated. The two smallest adjacent CKA values receive more screen time. Video axes from separate PCA fits are not directly comparable.

## Included results and validation

`results/block_geometry_metrics.csv` is the supplied 122-row CNN export, preserved exactly. Its original filename and SHA-256 appear in `results/source_manifest.json`. LLE result tables/plots remain available in executed notebook outputs; separate historical CNN LLE CSVs and activations are not included. DenseNet LLE CSVs, including all per-image entropies, are included in `results/densenet_20260930/`. The completed attention geometry metrics are preserved in `results/attention_geometry_metrics.csv`. Seven supplied MP4 videos are now available in the branch-root [visualization directory](../../visualization/), covering four CNNs and three attention readouts. Their presence does not constitute an independent model rerun. The attention notebook is preserved as the reproducibility record for the ViT-B/16 and Swin-T geometry experiment; its numerical export is preserved in `results/attention_geometry_metrics.csv`.

```bash
python -m unittest discover -s experiments/geometric_dynamics/tests -v
```

Offline checks cover formulas against explicit pairwise distances, CKA invariance and changing feature width, notebook Python syntax, historical source hashes and CNN CSV integrity. CI repeats these checks without downloading datasets or model weights. A complete pretrained rerun is a separate, resource-intensive verification step.

Dataset: [Zenodo record 5226945](https://zenodo.org/records/5226945); MD5 `5e014163374c3bf7069c923de2d619c8`. The archive and pretrained weights are not redistributed. See the source record/providers for their terms.

## License

Repository code is released under the [MIT License](../../LICENSE). External datasets, pretrained checkpoints and referenced publications retain their own terms.
