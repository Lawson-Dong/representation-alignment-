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

Observation counts follow the extraction protocol and are not counts of every computational layer. ViT CLS and patch-mean representations are analyzed as separate readouts.

**Checkpoint distinction:** ResNet-152 geometry uses torchvision V2 weights, while its local label entropy experiment uses V1. These conditions must be interpreted separately. Other listed experiments use V1 weights.

## Main Measurements

### Class Geometry

For unit-normalized representation vectors, measure mean cosine distances within each class and between classes. Define:

$$
S = \frac{d_{\mathrm{between}}}{d_{\mathrm{within}}}
$$

The denominator is the balanced mean of the two within-class distances, excluding self-pairs.

A larger **S** indicates greater between-class distance relative to within-class distance. It does **not** establish absolute within-class contraction: both distances must be inspected separately.

### Geometry Across Adjacent Blocks

**Adjacent linear CKA** compares centered sample Gram matrices at consecutive observations, including observations with different feature widths. Lower similarity marks a larger change in sample geometry under this measure.

These transitions identify candidate abrupt changes for closer inspection; they do not by themselves establish a physical phase transition.

### Local Label Entropy

**LLE means local label entropy**, measured in bits over cosine nearest neighbors, excluding the query image:

$$
H_k(i) = -\sum_{c \in \{\mathrm{cat},\mathrm{dog}\}} p_c(i;k)\log_2 p_c(i;k)
$$

Low entropy indicates a locally label-consistent neighborhood; high entropy indicates class mixing. The analysis uses **k = 4, 8, 16, 32, 64** and compares against shuffled-label baselines.

A ResNet-18 boundary diagnostic examines whether high-entropy samples are enriched near five-fold held-out logistic decision boundaries. This diagnostic is distinct from the cross-model entropy measurements.

## Visualizations

The [visualization directory](visualization/) is the destination for representation-dynamics videos.

The documented animation method projects same-image cosine fingerprints using one shared PCA fit across the observations in each video. Measured observations form the endpoints; motion between them is interpolated. Transitions with the two lowest adjacent CKA values receive more screen time.

Animations help inspect trajectories, but interpretation should rely on the original-space metrics. Axes from independently fitted videos are not directly comparable.

## Reproduce and Explore

See the [experiment README](experiments/geometric_dynamics/README.md) for installation, execution commands, checkpoint details, and validation.

- [Scripts](experiments/geometric_dynamics/scripts/): reproducible extraction, analysis, and attention-video generation.
- [Notebooks](experiments/geometric_dynamics/notebooks/): historical exploratory experiments and existing outputs.
- [CNN geometry results](experiments/geometric_dynamics/results/block_geometry_metrics.csv): the preserved 122-row export.
- [Protocol and audit](experiments/geometric_dynamics/docs/protocol.md): methods, source provenance, and interpretation limits.

## Interpretation

The research focus is **relative class organization and geometric transformation across depth**, rather than an assumption that all vectors contract monotonically.

Decreasing local label entropy can indicate increasing class organization. Calling this “self-organization” remains an interpretation: frozen pretrained networks implement transformations learned during training, and these measurements alone do not establish a general mechanism or universal law.

The evidence is exploratory and limited to the sampled images, checkpoints, architectures, and readouts. Historical notebook outputs and the supplied CNN table should be distinguished from independently reproduced results.

## License

Repository code is released under the [MIT License](LICENSE). External datasets and pretrained checkpoints retain their own terms.
