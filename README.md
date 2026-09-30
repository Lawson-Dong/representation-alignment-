# Representation Alignment

This repository collects a series of self-directed experiments exploring the nature and properties of neural representations at the intersection of artificial intelligence and cognitive science.

These projects were developed during my late high school and early undergraduate years, driven by my interest in understanding what neural networks learn internally, how information is represented, and how artificial representations relate to human cognition.

## Research Themes

The repository currently focuses on three related directions:

1. **Model–brain correspondence** — comparing the geometry of artificial visual representations with biological visual representations.
2. **Semantic composition** — studying how multiple concepts are combined inside a learned embedding space.
3. **Representation-vector geometric dynamics** — tracking how class geometry and local label mixing change across successive neural-network blocks.

A short conceptual overview is available in [`theory/representation_questions.md`](theory/representation_questions.md).

## Projects

### 1. ResNet and Human Visual Cortex

[Project branch](https://github.com/Lawson-Dong/representation-alignment-/tree/resnet-human-visual-cortex-rsa)

**Motivation**

Deep neural networks and the human visual system both appear to process visual information hierarchically. I was interested in whether the internal representations of a neural network across distinct layers would show different degrees of correspondence with different levels of the human visual hierarchy.

**Setup**

I used the 92-image dataset from the **Algonauts 2019 benchmark** and applied **Representational Similarity Analysis (RSA)** to compare representations from different layers of ResNet-18, ResNet-50, and ResNet-152 with two regions of the human visual cortex:

- **EVC (Early Visual Cortex)**
- **IT (Inferotemporal Cortex)**

The analysis examines how the representational geometry of the network changes across layers and how it corresponds to the representational geometry of these brain regions.

**Conclusion**

Across the ResNet architectures, the correspondence between model representations and human visual representations was layer-dependent.

In general, shallower network representations showed stronger correspondence with **EVC**, while deeper representations showed stronger correspondence with **IT**. This pattern is consistent with the idea that the hierarchical organization of visual representations in deep neural networks can partially correspond to the hierarchical organization of the human visual system.

The result does not imply that the networks implement the same computations as the brain. Rather, it provides evidence that different stages of a neural network can exhibit different forms of representational correspondence with different stages of biological visual processing.

---

### 2. CLIP and Semantic Composition

[Project branch](https://github.com/Lawson-Dong/representation-alignment-/tree/clip-semantic-composition)

**Motivation**

The first project led me to a more fundamental question: if neural networks develop structured internal representations, how are multiple concepts represented when they appear together?

I became particularly interested in whether a neural network's representation of a novel or semantically conflicting concept could be understood in terms of the representations of its constituent concepts.

**Setup**

I constructed hybrid visual stimuli combining different semantic components, such as a **dog body with a cat head**, and examined how these stimuli were represented in **CLIP's embedding space**.

I compared the representation of the hybrid concept with the representations of its component concepts and investigated whether the hybrid representation could be explained as a combination of these components.

I then explored an **adaptive linear-combination model** in which the contribution of each semantic component changes according to the visual composition of the stimulus.

**Conclusion**

The experiments suggested that the representation of a hybrid concept could be substantially explained through combinations of the representations of its constituent concepts.

Rather than behaving as an entirely independent representation, the hybrid representation appeared to occupy a position in the embedding space that reflected the composition of its underlying semantic components. Allowing the contribution of each component to vary with the visual composition provided a more flexible description of this relationship.

This experiment motivated a broader question that continues to interest me:

> **How are multiple concepts combined and transformed within the representation space of a neural network?**

---

### 3. Representation-Vector Geometric Dynamics

**Research question**

How does the geometry of image representations change as the same inputs propagate through a frozen visual network?

**Setup**

The experiments follow the same 200 images (100 cats and 100 dogs) through ImageNet-pretrained ResNet-18, ResNet-152, ConvNeXt-Tiny, ConvNeXt-Base, ViT-B/16 and Swin-T. Measurements are taken at block outputs and explicit architectural transitions, using fixed preprocessing and documented pooling rules.

The measurements cover within-class and between-class cosine and unit-vector Euclidean distances, relative separation **S**, raw-vector **Fisher ratio**, mean raw-vector norm, adjacent linear **CKA**, **participation ratio (PR)** as an effective-dimension estimate, and held-out **linear-probe accuracy**. Adjacent changes in S, Fisher ratio, PR and probe accuracy describe the CNN transitions. The CNN LLE experiments additionally measure cosine-neighborhood **local label entropy**, local order relative to shuffled labels, high-entropy fractions and entropy trajectory summaries across neighborhood sizes. A held-out linear-boundary diagnostic examines high-LLE samples in ResNet-18 using margins, correlations, AUC and enrichment checks. Measurement availability differs by model and pipeline; see the [full definitions](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/README.md#main-measurements).

Seven [representation-dynamics videos](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/visualization) cover ResNet-18, ResNet-152, ConvNeXt-Tiny, ConvNeXt-Base, ViT-B/16 CLS, ViT-B/16 patch mean and Swin-T. Motion between measured observations is interpolated.

**Findings and interpretation**

The supplied CNN result table shows higher final S than initial S in all four CNNs. This is relative class separation: within-class distances do not uniformly decrease. LLE tracks local label mixing rather than entropy of the activation vectors themselves. Trajectories can be nonmonotonic, and a CKA dip is a descriptive geometry change rather than proof of a phase transition.

Checkpoint differences are explicit: ResNet-152 uses V2 weights in the CNN geometry experiment and V1 in the LLE experiment. Architecture, pooling and pretraining differences limit depth-only comparisons. The experiment documentation distinguishes executed notebook evidence from results that have not been independently rerun.

**Code and reproducibility**

The implementation is maintained on the [`representation-vector-geometric-dynamics` branch](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics):

- [Experiment README and reproduction commands](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/experiments/geometric_dynamics)
- [Original notebooks](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/experiments/geometric_dynamics/notebooks)
- [Command-line scripts](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/experiments/geometric_dynamics/scripts)
- [CNN result tables and source hashes](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results)
- [Protocol and research audit](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/experiments/geometric_dynamics/docs/protocol.md)

The branch includes an MIT license, contribution guidance and GitHub Actions checks for metric definitions, Python syntax, source integrity and the supplied CNN CSV. These offline checks do not replace a full pretrained-model rerun.

---

## Repository Structure

```text
representation-alignment-/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
└── theory/
    └── representation_questions.md
```

The structure above describes the default branch. The geometric dynamics experiment lives under `experiments/geometric_dynamics/` on its [dedicated branch](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics), organized into `notebooks/`, `scripts/`, `results/`, `docs/` and `tests/`, with videos in the branch-root `visualization/` directory.

## Reproducibility

For geometric dynamics, first check out the experiment branch and follow its experiment-specific instructions:

```bash
git clone https://github.com/Lawson-Dong/representation-alignment-.git
cd representation-alignment-
git switch representation-vector-geometric-dynamics
pip install -r experiments/geometric_dynamics/requirements.txt
```

The [experiment README](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics/experiments/geometric_dynamics) documents dataset checksums, output directories, model execution and optional video generation.

Install the base environment with:

```bash
pip install -r requirements.txt
```

Large datasets, extracted activations, model checkpoints, and generated analysis artifacts should not be committed directly to the repository. Dataset-specific instructions and expected file layouts should be documented in [`data/README.md`](data/README.md) and in the corresponding experiment directory.

## Research Motivation

Although these experiments study different models and questions, they share a common theme:

> **What structure exists inside neural representations, and what can that structure tell us about intelligence and cognition?**

The ResNet experiment approached this question from the perspective of **model–brain correspondence**, asking whether different stages of artificial visual processing resemble different stages of biological visual processing.

The CLIP experiment approached it from the perspective of **model-internal representation**, asking how multiple semantic concepts are organized and combined within a learned representation space.

The geometric dynamics project studies how category structure changes across a forward pass, connecting global class separation with local neighborhood organization.

Together, these projects reflect my early interest in studying neural networks not only through their outputs, but through the internal representations that give rise to those outputs.

## Research Process

These projects were independently developed out of personal interest during my late high school and early undergraduate years. I learned the relevant concepts largely through reading papers, discussing ideas, formulating research questions, writing and debugging code, and iteratively designing and analyzing experiments by discussing my questions and understandings hundreds of hours with AI.

At the time, I did not have formal training in research methodology, cognitive neuroscience, or representation learning. As a result, these projects were exploratory, and some of the questions and interpretations evolved substantially during the process.

I keep them here as an early record of my attempt to understand neural representations through computational experiments.
