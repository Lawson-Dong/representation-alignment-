# Representation Vector Dynamics

Exploratory experiments on how representation-vector geometry changes as the same images propagate through successive layers and blocks of pretrained visual networks.

The central question is:

> How do class structure, local label mixing, and representational geometry evolve across network depth?

This branch focuses on **layerwise forward-pass dynamics in frozen networks**. Depth is the progression variable; these experiments do not track changes during training.

## Research Questions

- Do representations become more separated by class as depth increases?
- Does increased class separation reflect within-class contraction, between-class expansion, or both?
- Does geometry change gradually, or are there blocks with unusually large transformations?
- How does local label entropy evolve, and where are high-entropy samples located?
- Which patterns recur across architectures, depths, and representation readouts?

## Experimental Setup

The experiments follow the same **100 cat and 100 dog images** through pretrained ImageNet-1K models. Each observation provides one representation vector per image.

| Architecture | Models / readouts | Observations per trajectory |
|---|---|---|
| ResNet | ResNet-18; ResNet-152 | 9; 51 |
| ConvNeXt | ConvNeXt-Tiny; ConvNeXt-Base | 22; 40 |
| Vision Transformer | ViT-B/16 CLS token; patch-token mean | 13; 14 |
| Swin Transformer | Swin-T | 17 |
| DenseNet | DenseNet-121; DenseNet-169; DenseNet-201 | 63; 87; 103 |

The [2026-09-30 DenseNet Colab run](experiments/geometric_dynamics/results/geometry/densenet_20260930/) adds three same-family models with matched geometry, LLE and linear probes. Dense-layer observations are cumulative concatenated states; final norm+ReLU is a separate readout. All three checkpoints use explicit V1 weights.

Observation counts follow the extraction protocol and are not counts of every computational layer. ViT CLS and patch-mean representations are analyzed as separate readouts.

**Checkpoint alignment:** ResNet-152 geometry and the completed local label entropy rerun both use explicit torchvision ImageNet-1K V2 weights. Other listed experiments retain V1 weights. Shared ResNet-18 V1 preprocessing is unchanged.

## Main Measurements

The measurements below describe complementary aspects of representation dynamics. The CNN geometry pipeline exports the geometry and scale measurements; the LLE pipeline computes neighborhood entropy and its diagnostics. Metric availability differs by experiment and readout.

### Class Geometry and Vector Scale

Let $x_i$ be a raw pooled representation and $z_i=x_i/\|x_i\|_2$ its unit-normalized vector.

| Measurement | Code field | Interpretation |
|---|---|---|
| Within-class cosine distance | `d_within_cos` | Balanced mean of cat–cat and dog–dog distances, excluding self-pairs; measures angular dispersion within classes. |
| Between-class cosine distance | `d_between_cos` | Mean distance over all cat–dog pairs; measures angular separation across classes. |
| Relative separation S | `S` | Between-class cosine distance divided by within-class cosine distance. |
| Within-class unit-vector Euclidean distance | `d_within_euclid_unit` | Balanced mean Euclidean distance within the two classes, excluding self-pairs. |
| Between-class unit-vector Euclidean distance | `d_between_euclid_unit` | Mean Euclidean distance over all cat–dog pairs. |
| Raw-vector Fisher ratio | `Fisher_raw` | Squared distance between raw class centroids divided by the sum of the two classes' mean squared deviations from their centroids. |
| Mean raw-vector norm | `mean_raw_norm` | Mean representation magnitude; tracks changes in vector scale before normalization. |

$$
S_\ell = \frac{d_{\mathrm{between},\ell}}{d_{\mathrm{within},\ell}}
$$

A larger **S** indicates greater between-class distance relative to within-class distance. It does **not** establish absolute within-class contraction: inspect both distances separately.

For raw class centroids $\mu_{\mathrm{cat}}$ and $\mu_{\mathrm{dog}}$, the Fisher ratio is:

$$
F_\ell =
\frac{\|\mu_{\mathrm{cat}}-\mu_{\mathrm{dog}}\|_2^2}
{\mathbb{E}_{i\in\mathrm{cat}}\|x_i-\mu_{\mathrm{cat}}\|_2^2+
 \mathbb{E}_{i\in\mathrm{dog}}\|x_i-\mu_{\mathrm{dog}}\|_2^2}
$$

Fisher ratio is invariant to a common scalar rescaling, but can change under feature-specific scaling. Raw norms are scale-sensitive and should be interpreted in the context of feature width and pooling.

For unit vectors, $d_E^2=2d_{\cos}$. The Euclidean and cosine measurements therefore do not provide independent evidence; their pairwise means differ because the square root is applied before averaging Euclidean distances.

### Geometry Across Adjacent Blocks

| Measurement | Code field | Interpretation |
|---|---|---|
| Adjacent linear CKA | `CKA_prev` | Similarity between centered raw sample Gram matrices at consecutive observations; supports different feature widths. |
| Change in relative separation | `delta_S` | $S_\ell-S_{\ell-1}$; locates steps that increase or decrease relative class separation. |
| Change in Fisher ratio | `delta_Fisher_raw` | $F_\ell-F_{\ell-1}$; tracks changes in raw-vector class separation relative to scatter. |

The delta fields are calculated in the CNN transition analysis, rather than exported as columns in the base geometry CSV. Stage-boundary summaries compare adjacent CKA and changes in S at boundaries versus other observations.

Lower adjacent CKA marks a larger change in sample geometry under this measure. Combined with distance trajectories and separation increments, it identifies candidate abrupt transitions; it does not by itself establish a physical phase transition. Consecutive observations are not necessarily equal steps in computational depth.

### Local Label Entropy and Local Order

**LLE means local label entropy**, measured in bits over cosine nearest neighbors, excluding the query image:

$$
H_k(i) = -\sum_{c \in \{\mathrm{cat},\mathrm{dog}\}} p_c(i;k)\log_2 p_c(i;k)
$$

Here $p_c(i;k)$ is the fraction of the $k$ neighbors belonging to class $c$, with $0\log_2 0=0$. Low entropy indicates a locally label-consistent neighborhood; high entropy indicates class mixing. The analysis uses **k = 4, 8, 16, 32, 64**.

| Measurement | Code field | Interpretation |
|---|---|---|
| Per-image entropy | `H` | Local class mixing around each individual image vector. |
| Mean local label entropy | `LLE` | Mean $H_k(i)$ over the sampled images at one observation. |
| Shuffled-label reference | `shuffle_mean`, `shuffle_sd` | Mean and standard deviation of mean entropy over 100 label permutations. |
| Relative local label order | `order_vs_shuffle` | $1-\overline H/\overline H_{\mathrm{shuffle}}$; positive values indicate less mixing than the shuffled baseline. |
| High-entropy fraction | `fraction_H_ge_0_8` | Fraction of samples with $H_k(i)\geq0.8$. |
| Endpoint entropy decrease | `net_drop` | Initial mean LLE minus final mean LLE, by model and k. |
| Number of entropy-increasing transitions | `n_increasing_transitions` | Counts adjacent increases in mean LLE, allowing a net decrease to be distinguished from monotonic decrease. |

Per-image entropy distributions and comparisons across k reveal heterogeneity and neighborhood-scale dependence that a single mean can hide. Relative local order is a baseline-relative statistic, not a general thermodynamic order parameter.

### High-Entropy Boundary Diagnostics

The ResNet-18 diagnostic examines whether high-entropy samples are concentrated near **five-fold held-out logistic decision boundaries**. Each image is scored by a model fitted without that image. Features are unit-normalized and standardized using training-fold statistics.

| Measurement | Code field | Interpretation |
|---|---|---|
| Absolute boundary margin | `abs_margin` | Absolute decision score divided by coefficient norm; distance to the fitted hyperplane in fold-standardized feature coordinates. |
| Margin percentile | `margin_percentile` | Within-observation ranking of absolute margins. |
| Entropy–margin correlation | `spearman_H_abs_margin` | Spearman correlation between entropy and absolute margin; negative values associate greater mixing with smaller margins. |
| Boundary proximity AUC | `auc_near_predicts_high` | ROC AUC for predicting high entropy using negative absolute margin. |
| Near- and far-boundary high-entropy rates | `near_high_rate`, `far_high_rate` | High-entropy frequency in the closest margin quartile versus the remaining samples. |
| Median margins by entropy group | `median_abs_margin_high`, `median_abs_margin_low` | Compares typical boundary distance for high- and low-entropy samples. |
| Fraction of high-entropy samples near the boundary | `fraction_high_near` | Share of high-entropy samples in the closest margin quartile, evaluated across k. |
| Boundary enrichment test | `hypergeom_p` | Exploratory hypergeometric test of high-entropy sample enrichment in the closest quartile. |
| Misclassification among high-entropy samples | `high_misclassified` | Helps distinguish boundary mixing from confidently misclassified neighborhoods. |

High entropy uses $H\geq0.8$; “near boundary” uses the lowest quartile of absolute margins. These are operational thresholds. Margin magnitudes from different fold-standardized spaces are not a common original-space distance. The enrichment tests are exploratory and should be interpreted alongside k sensitivity.

### Effective Dimension: Participation Ratio (PR)

PR measures how broadly variance is distributed across representation directions. In the CNN geometry pipeline, unit-normalized vectors are centered across samples, and $\lambda_j$ are the eigenvalues of their sample Gram matrix:

$$
\mathrm{PR}_\ell =
\frac{\left(\sum_j \lambda_j\right)^2}{\sum_j \lambda_j^2}
$$

| Measurement | Code field | Interpretation |
|---|---|---|
| Participation ratio | `PR` | Effective dimension of the centered unit-vector cloud. |
| Adjacent change in PR | `delta_PR` | $\mathrm{PR}_\ell-\mathrm{PR}_{\ell-1}$; calculated in the CNN transition analysis. |

Lower PR indicates variance concentrated in fewer directions; higher PR indicates a more distributed variance spectrum. PR is a spectral effective-dimension estimator, not a direct estimate of manifold intrinsic dimension. A decrease in PR does not by itself establish class separation or reduced pairwise distances.

### Linear-Probe Accuracy

The CNN geometry pipeline evaluates how linearly decodable the cat/dog labels are at each observation. A logistic-regression probe is fitted to frozen representations after per-image unit normalization and training-split feature standardization.

The evaluation uses **five stratified shuffle splits**, each with **70% training and 30% held-out test images**, with seed 42. The same splits are used across observations and CNN models.

| Measurement | Code field | Interpretation |
|---|---|---|
| Mean held-out probe accuracy | `probe_mean` | Mean classification accuracy over the five held-out splits. |
| Probe accuracy standard deviation | `probe_sd` | Sample standard deviation across splits; describes split variability, not a confidence interval. |
| Adjacent change in probe accuracy | `delta_probe_mean` | Difference in mean accuracy between consecutive observations; calculated in the CNN transition analysis. |

The probe uses logistic regression with C = 1. Normalization, feature standardization, and fitting are performed within each training split. Higher accuracy indicates better linear label decodability under this protocol, rather than a complete measure of representation quality.

This accuracy experiment uses stratified shuffle splits and is separate from the **five-fold out-of-fold boundary diagnostic** above. The boundary analysis additionally records held-out correctness and accuracy (`oof_accuracy`) to help interpret entropy and margins.

PR and probe accuracy complement the distance, CKA, and entropy measurements. Together they describe effective dimension, label decodability, class geometry, local mixing, and transformations across depth.

## Visualizations

The [visualization directory](visualization/) contains ten representation-dynamics videos:

| Model / readout | Video |
|---|---|
| ResNet-18 | [10-second projection](visualization/resnet18_real_projection_10s.mp4) |
| ResNet-152 | [10-second projection](visualization/resnet152_real_projection_10s.mp4) |
| ConvNeXt-Tiny | [14-second projection](visualization/convnext_tiny_real_projection_slow_14s.mp4) |
| ConvNeXt-Base | [16-second projection](visualization/convnext_base_real_projection_slow_16s.mp4) |
| ViT-B/16 CLS | [12-second geometry animation](visualization/vit_b16_cls_geometry_slow_12s.mp4) |
| ViT-B/16 patch mean | [12-second geometry animation](visualization/vit_b16_patchmean_geometry_slow_12s.mp4) |
| Swin-T | [12-second geometry animation](visualization/swin_t_geometry_slow_12s.mp4) |
| DenseNet-121 | [16-second geometry animation](visualization/densenet121_geometry_slow_16s.mp4) |
| DenseNet-169 | [16-second geometry animation](visualization/densenet169_geometry_slow_16s.mp4) |
| DenseNet-201 | [16-second geometry animation](visualization/densenet201_geometry_slow_16s.mp4) |

The documented animation method projects same-image cosine fingerprints using one shared PCA fit across the observations in each video. Measured observations form the endpoints; motion between them is interpolated. Transitions with the two lowest adjacent CKA values receive more screen time.

Animations help inspect trajectories, but interpretation should rely on the original-space metrics. Axes from independently fitted videos are not directly comparable.

## Reproduce and Explore

See the [experiment README](experiments/geometric_dynamics/README.md) for installation, execution commands, checkpoint details, and validation.

- [Scripts](experiments/geometric_dynamics/scripts/): reproducible extraction, analysis, and attention-video generation.
- [Notebooks](experiments/geometric_dynamics/notebooks/): historical exploratory experiments and existing outputs.
- [CNN geometry results](experiments/geometric_dynamics/results/geometry/block_geometry_metrics/): per-metric CSVs for 122 measurement points, paired stratified CKA bootstrap, stage-dip summaries, and run provenance.
- [DenseNet measured results](experiments/geometric_dynamics/results/geometry/densenet_20260930/): 253 geometry observations, 1,265 LLE summaries and 253,000 per-image entropy rows, with the executed notebook and provenance.
- [Protocol and audit](experiments/geometric_dynamics/docs/protocol.md): methods, source provenance, and interpretation limits.

## Interpretation

The research focus is **relative class organization and geometric transformation across depth**, rather than an assumption that all vectors contract monotonically.

Decreasing local label entropy can indicate increasing class organization. Calling this “self-organization” remains an interpretation: frozen pretrained networks implement transformations learned during training, and these measurements alone do not establish a general mechanism or universal law.

The evidence is exploratory and limited to the sampled images, checkpoints, architectures, and readouts. Historical notebook outputs and the supplied CNN table should be distinguished from independently reproduced results.

## License

Repository code is released under the [MIT License](LICENSE). External datasets and pretrained checkpoints retain their own terms.
