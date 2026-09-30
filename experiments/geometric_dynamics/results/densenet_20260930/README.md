# DenseNet matched-image Colab run — 2026-09-30

Completed in the [requested Colab notebook](https://colab.research.google.com/drive/18ROQJhjCUETMp6Ih4Y6XwmPe6l0E76o_) on a Tesla T4. Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128; full environment and explicit checkpoint URLs are in `protocol.json`. The [executed notebook](../../notebooks/Cat_Dog_DenseNet_121_169_201_Matched_Geometry_LLE.ipynb) embeds the exact maintained extraction/measurement source and contains completed numerical tables and plots. No error outputs are saved.

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

- `densenet_geometry_metrics.csv`: 253 observations, including S, distances, Fisher, PR, raw norms, CKA, probes and adjacent deltas.
- `densenet_lle_summary.csv`: 1,265 layer/k summaries, null baselines and high-entropy fractions.
- `densenet121/169/201_lle_per_image.csv`: 63,000 / 87,000 / 103,000 rows (253,000 total), retaining sample indices, paths, labels and entropy.
- Per-model geometry/LLE summaries and `densenet_endpoint_summary.csv` are direct exports from the run.
- `sample_manifest.csv`, `probe_splits.csv`, `protocol.json`, `run_record.json`: sample identities, paired splits, environment, checkpoints and execution origin.
- `artifact_manifest.json`: original runtime SHA-256 hashes. Three listed NPZs are deliberately omitted from Git; all remaining listed files are preserved byte-for-byte.

The cohort is matched to the historical protocol by the verified archive and identical selection algorithm. Historical CNN activation NPZs are absent from this checkout, so this run does not claim a direct comparison against their stored identities. New-model passes are explicitly checked against the saved sample manifest.

## Limits

Depth, block allocation, resulting widths and independently trained checkpoints vary. This within-family comparison cannot isolate depth causally. Retaining old channels itself promotes similar adjacent geometry, so large CKA does not mean an individual newly added feature is unimportant. Pooling discards spatial structure. Observation indices are not equal computational depth. The finite cohort and overlapping probe splits limit generalization; split SD and shuffle SD are not confidence intervals. No held-out boundary diagnostic was run for these models. CKA minima are descriptive changes, not proof of phase transitions.

## Reproduce and validate

From the repository root, with compatible dependencies installed:

```bash
GEOMETRY_OUTPUT_DIR=outputs/densenet python experiments/geometric_dynamics/scripts/run_densenet_geometry.py
python -m unittest discover -s experiments/geometric_dynamics/tests -v
```

The nine offline checks include reaggregation of all per-image entropies to the summaries, image-order consistency, train/test disjointness and class balance, source identity against the executed notebook, retained runtime hashes, geometry identities and adjacent deltas. They do not replace a pretrained GPU rerun.
