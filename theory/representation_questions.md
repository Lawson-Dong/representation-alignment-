# Representation Questions

This repository explores three related questions about learned representations: correspondence with biological vision, semantic composition, and geometric dynamics across network depth.

## 1. Model–brain correspondence

How does representational geometry change across the layers of a visual neural network, and to what extent does that geometry correspond to different stages of biological visual processing?

The ResNet project approaches this question using Representational Similarity Analysis (RSA), comparing model-layer representations with Early Visual Cortex (EVC) and Inferotemporal Cortex (IT).

Project branch: [ResNet and human visual cortex](https://github.com/Lawson-Dong/representation-alignment-/tree/resnet-human-visual-cortex-rsa).

## 2. Semantic composition

How are multiple concepts combined inside a learned representation space?

The CLIP project approaches this question using hybrid visual stimuli and tests whether a hybrid representation can be described in terms of the representations of its component concepts.

Project branch: [CLIP semantic composition](https://github.com/Lawson-Dong/representation-alignment-/tree/clip-semantic-composition).

## 3. Representation vector dynamics

How does the geometry of the same samples' representation vectors evolve as they propagate through successive layers and blocks of a neural network?

This project follows a matched set of 100 cat and 100 dog images through frozen pretrained ResNet, ConvNeXt, ViT, and Swin models. It studies **layerwise forward-pass transformations**, with depth as the progression variable, rather than changes in model parameters during training.

Project branch: [Representation vector geometric dynamics](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics).

### Research questions

- **Class organization:** Does depth increase class separation, and does this result from within-class contraction, between-class expansion, or changes in both?
- **Transition structure:** Are transformations gradual, or concentrated at particular blocks and stage boundaries?
- **Local mixing:** Do neighborhoods become more label-consistent across depth? Does an overall entropy decrease include local reversals?
- **Boundary structure:** Are high-entropy samples concentrated near held-out class decision boundaries, or do they also occur in misclassified neighborhoods?
- **Effective dimension:** Does representational variance become concentrated in fewer directions, and how does this relate to class geometry?
- **Linear decodability:** How does the accessibility of category information to a linear classifier change across depth?
- **Cross-architecture patterns:** Which observations recur across model families, depths, checkpoints, and readouts?

### Measurement framework

| Aspect | Measurements | Question addressed |
|---|---|---|
| Angular class geometry | Within-class and between-class cosine distances | Are same-class vectors less dispersed, and are different classes farther apart? |
| Relative separation | \(S=d_{\mathrm{between}}/d_{\mathrm{within}}\); adjacent \(\Delta S\) | How does class separation change relative to within-class dispersion? |
| Unit-vector distance geometry | Within-class and between-class Euclidean distances | How does pairwise distance evolve after normalization? |
| Raw-vector class geometry | Fisher ratio; adjacent changes in Fisher ratio | How does class-centroid separation compare with within-class scatter? |
| Vector scale | Mean raw-vector norm | How does representation magnitude change before normalization? |
| Adjacent representational similarity | Linear CKA between consecutive observations | Where does sample geometry change most strongly? |
| Local label mixing | Per-image and mean local label entropy; high-entropy fraction | How mixed are labels within cosine nearest-neighbor neighborhoods? |
| Local order relative to chance | Shuffled-label entropy baseline; \(1-\overline H/\overline H_{\mathrm{shuffle}}\) | Is local label organization stronger than under random label assignments? |
| Entropy trajectory | Endpoint entropy decrease; number of entropy-increasing transitions; comparisons across k | Is increasing local organization consistent across depth and neighborhood scales? |
| Boundary association | Absolute held-out margins, margin percentiles, entropy–margin Spearman correlation, proximity AUC, near/far high-entropy rates, median margins by entropy group | Does high local mixing coincide with proximity to a fitted category boundary? |
| Boundary enrichment and errors | Fraction of high-entropy samples near the boundary, exploratory hypergeometric enrichment tests, misclassification among high-entropy samples | Are high-entropy neighborhoods boundary-associated or related to classification errors? |
| Effective dimension | Participation ratio (PR); adjacent \(\Delta\mathrm{PR}\) | How many variance directions contribute substantially to the representation cloud? |
| Linear decodability | Held-out linear-probe accuracy, split standard deviation, adjacent accuracy changes | How accessible are category labels to a linear readout? |

These measurements are complementary, but not all are independent. For unit vectors, squared Euclidean distance equals twice cosine distance. Availability also differs by experiment: the full framework should not be read as a claim that every measurement has been evaluated for every model.

### Key definitions

**Relative class separation**

$$
S_\ell=\frac{d_{\mathrm{between},\ell}}{d_{\mathrm{within},\ell}}
$$

Within-class distance is the balanced mean of cat–cat and dog–dog cosine distances, excluding self-pairs. An increase in S can occur without an absolute decrease in within-class distance.

**Local label entropy**

$$
H_k(i)=-\sum_{c\in\{\mathrm{cat},\mathrm{dog}\}}p_c(i;k)\log_2 p_c(i;k)
$$

Here \(p_c(i;k)\) is the class fraction among the k cosine nearest neighbors, excluding the query sample, and \(0\log_2 0=0\). LLE denotes **local label entropy**, not locally linear embedding. The experiments use k = 4, 8, 16, 32, 64 and 100 shuffled-label baselines. This is entropy of neighborhood labels, not thermodynamic entropy or the full information content of a representation.

**Participation ratio**

$$
\mathrm{PR}_\ell=
\frac{\left(\sum_j\lambda_j\right)^2}{\sum_j\lambda_j^2}
$$

The implemented CNN measurement uses eigenvalues of the centered unit-vector sample Gram matrix. PR estimates spectral effective dimension; it is not a direct manifold intrinsic-dimension estimate.

**Linear-probe accuracy**

The CNN protocol uses logistic regression on unit-normalized features, with feature standardization fitted on each training split. Five stratified 70/30 shuffle splits provide mean held-out accuracy and its sample standard deviation. This is separate from the five-fold out-of-fold logistic boundary diagnostic, currently performed for ResNet-18.

### Interpretation and working hypotheses

A working hypothesis is that learned forward transformations can produce progressively stronger class organization, visible in relative separation, local label consistency, and linear decodability. The trajectories may nevertheless be nonmonotonic and differ across architectures.

Describing an entropy decrease as “self-organization” or “self-structuring” is an interpretation to investigate, not an established general mechanism. The networks are frozen at measurement time, and their transformations were learned during prior training.

Low adjacent CKA identifies candidate abrupt geometric changes, but does not establish a physical phase transition. Likewise, lower PR does not necessarily imply class separation, and higher probe accuracy does not fully characterize representation quality.

Projected animations support inspection of trajectories. Conclusions about contraction, separation, and local organization should be checked using original-space measurements because projection and interpolation can alter the visual impression.

The evidence is exploratory and conditional on the sampled images, preprocessing, checkpoints, and representation readouts. In particular, ResNet-152 geometry uses V2 weights while its LLE experiment uses V1; these are separate conditions. Observed block indices are not equal computational-depth steps across architectures.

See the [project README](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/README.md) and [experimental protocol](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/docs/protocol.md) for implementation details, provenance, and validation limits.

## Shared perspective

All three projects examine measurable internal representational structure alongside model outputs.

Model–brain correspondence asks how artificial and biological geometries relate. Semantic composition asks how concepts combine within an embedding space. Representation vector dynamics asks how sample geometry, local organization, effective dimension, and label decodability evolve through a network.

Together, they motivate a broader question:

> What structure exists inside learned representations, how is that structure transformed, and what can those observations tell us about intelligence and cognition?

These experiments do not assume that artificial and biological systems implement identical computations. Links to cognition remain hypotheses requiring additional evidence.

## Scope

This document provides the repository-wide conceptual overview. Each project branch contains its own methods and evidence. Measurement definitions, exploratory interpretations, and verified empirical findings should remain distinct.
