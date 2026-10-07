# DenseNet matched-image Colab run — 2026-09-30

Completed in the [requested Colab notebook](https://colab.research.google.com/drive/18ROQJhjCUETMp6Ih4Y6XwmPe6l0E76o_) on a Tesla T4. Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128; full environment and explicit checkpoint URLs are in `protocol.json`. The [executed notebook](../../../../notebooks/Cat_Dog_DenseNet_121_169_201_Matched_Geometry_LLE.ipynb) embeds the exact maintained extraction/measurement source and contains completed numerical tables and plots. No error outputs are saved.

## Controls and observations

This run adds one new architecture family: DenseNet-121, DenseNet-169 and DenseNet-201, all ImageNet-1K V1 weights, growth rate 32 and bottleneck multiplier 4. It retains the checksum-verified Zenodo archive, original ZIP order, default_rng(42) sample selection, 100 cats + 100 dogs, fixed ResNet-18 V1 transform, evaluation mode, spatial mean pooling and original metric definitions. Every new model receives exactly the same images in the same order, with batch size 8 and float32 inference. No fine-tuning, augmentation or mixed precision is used.

Dense-layer observations are complete **cumulative concatenated states**, rather than only newly produced channels. Spatial pooling commutes with channel concatenation: pool the block input and append each new layer's pooled channels. Runtime assertions compare each block's reconstructed cumulative state with its real pooled output in every batch. Stem, three transition modules and final norm+ReLU are explicit observations, giving 63, 87 and 103 points. Final norm+ReLU is a readout transformation, not another dense layer.

Five paired stratified 70/30 probe splits and 100 label permutations are shared by all observations and models. Probe preprocessing is fit only on training samples. LLE uses cosine distance, self exclusion, stable sorting and k=4,8,16,32,64; shared shuffle seed is 1042. All geometry and LLE values use the same extracted activation pass per model.

## Measured endpoints

The final endpoint below includes norm+ReLU.

| Model | Observations | S: stem → final | Mean LLE, k=16: stem → final | LLE-increasing transitions | Final probe accuracy |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 | 63 | 1.0440 → 1.3045 | 0.9375 → 0.1137 | 12 | 99.67% |
| DenseNet-169 | 87 | 1.0713 → 1.2598 | 0.9386 → 0.0797 | 19 | 100.00% |
| DenseNet-201 | 103 | 1.0264 → 1.2797 | 0.9444 → 0.0776 | 26 | 100.00% |

Mean within-class cosine distances rise from 0.0456→0.3608, 0.0386→0.4024 and 0.0311→0.4226. These results indicate increased **relative** class separation, not absolute within-class contraction. Entropy falls overall but has local reversals. Increasing transitions count strictly positive adjacent mean-LLE differences, not a significance test.

The readout change matters: at the last dense state, before norm+ReLU, S is 1.2340 / 1.2540 / 1.2547 and k=16 mean LLE is 0.2242 / 0.1902 / 0.1562. Norm+ReLU further changes geometry and local mixing. This distinguishes depth effects from the final readout.

## Files

Per-model geometry tables are in `../../densenet121/`, `../../densenet169/` and `../../densenet201/`; final LLE layer/k summaries are under `../../../lle/<model>/`. Combined final tables are retained here and under `../../../lle/cross-model/densenet_20260930/`. Combined plots are in `../../../figures/cross-model/densenet_20260930/`.

- `densenet_geometry_metrics.csv`: all 253 layer observations and final metrics.
- `densenet_endpoint_summary.csv`: final endpoint comparisons.
- `protocol.json` and `run_record.json`: settings, environment and execution origin.
- `artifact_manifest.json`: hashes of retained runtime exports.

Per-image entropy, image manifests, probe split indices and activation archives are omitted from the published results. Code and original executed notebook outputs remain available for reproduction.

## Limits

Depth, block allocation, resulting widths and independently trained checkpoints vary. This within-family comparison cannot isolate depth causally. Retaining old channels itself promotes similar adjacent geometry, so large CKA does not mean an individual newly added feature is unimportant. Pooling discards spatial structure. Observation indices are not equal computational depth. The finite cohort and overlapping probe splits limit generalization; split SD and shuffle SD are not confidence intervals. No held-out boundary diagnostic was run for these models. CKA minima are descriptive changes, not proof of phase transitions.

## Reproduce and validate

From the repository root, with compatible dependencies installed:

```bash
GEOMETRY_OUTPUT_DIR=outputs/densenet python experiments/geometric_dynamics/scripts/run_densenet_geometry.py
python -m unittest discover -s experiments/geometric_dynamics/tests -v
```

Offline checks validate retained hashes, executed source identity, complete layer/k summaries, geometry identities and adjacent deltas. They do not recompute summaries from omitted intermediate data or replace a pretrained GPU rerun.
