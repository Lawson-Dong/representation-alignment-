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

The [new DenseNet notebook](notebooks/Cat_Dog_DenseNet_121_169_201_Matched_Geometry_LLE.ipynb) was executed on the specified Colab T4 on 2026-09-30. Its [CSV exports and provenance](results/metadata/densenet_20260930/) are preserved with offline final-output checks.

The CNN geometry notebook has been replaced with the completed Colab stratified-bootstrap run; its 52 original figures are published in the four CNN model folders under `results/figures/`. The attention notebook remains a historical upload. The LLE notebook has been replaced in place with the completed Colab ResNet-152 V2 rerun; only its execution-status markdown was updated after upload, preserving every code cell and output. For local/headless runs use `scripts/`. See [protocol and audit](docs/protocol.md) for execution evidence, corrected stale notebook text, and interpretation limits.

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

Final CNN metric and bootstrap summary CSVs are stored separately in `results/geometry/resnet18/`, `resnet152/`, `convnext_tiny/` and `convnext_base/`. They cover all 122 measurement points and 1,000 paired class-stratified resamples; settings and execution provenance are in `results/metadata/cnn_bootstrap/`. Separate within-stage first-block contrasts are model-specific diagnostics. The original combined tables remain recoverable in Git history.

DenseNet final geometry, endpoint summaries and LLE layer/k summaries reside in the three model folders under `results/geometry/` and `results/lle/`. Attention geometry is stored separately under `results/geometry/vit_b16_cls/`, `vit_b16_patchmean/` and `swin_t/`. CNN LLE tables/plots remain in executed notebook outputs; standalone historical CNN LLE CSVs and activations are not included. Figures are in per-model folders under `results/figures/`. Seven supplied MP4s remain in [visualization](../../visualization/); their presence is not an independent rerun. Execution records and a complete retained-output hash manifest are under `results/metadata/`.

```bash
python -m unittest discover -s experiments/geometric_dynamics/tests -v
```

Offline checks cover formulas against explicit pairwise distances, CKA invariance and changing feature width, notebook Python syntax, historical source hashes and CNN CSV integrity. CI repeats these checks without downloading datasets or model weights. A complete pretrained rerun is a separate, resource-intensive verification step.

Dataset: [Zenodo record 5226945](https://zenodo.org/records/5226945); MD5 `5e014163374c3bf7069c923de2d619c8`. The archive and pretrained weights are not redistributed. See the source record/providers for their terms.

## License

Repository code is released under the [MIT License](../../LICENSE). External datasets, pretrained checkpoints and referenced publications retain their own terms.

## Final-results publication policy

`results/` stores final layer-wise metrics, model-specific endpoint summaries, statistical intervals and figures once per model. `metadata/` stores common settings and provenance. Combined duplicates and side-by-side plots without a separate cross-model measurement are omitted. No `cross-model/` directory is currently needed. Intermediate bootstrap draws, sample/split indices and per-image entropy are excluded. Original executed notebook code and outputs are preserved; historical files can be recovered from Git history.

## Whole-curve bootstrap stability

The primary bootstrap analysis assesses the complete CKA trajectory within each CNN at identical layer identities, rather than assuming a universal first-block drop. Read the [methods, numerical results and limitations](docs/bootstrap_curve_stability.md), then use the [results index](results/README.md) to locate per-model CSVs and figures. Run `python experiments/geometric_dynamics/scripts/analyze_bootstrap_curves.py` from the repository root; this reuses the original saved draws in Git history without rerunning models.
