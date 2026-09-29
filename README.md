# Representation Alignment

This repository collects a series of self-directed experiments exploring the nature and properties of neural representations at the intersection of artificial intelligence and cognitive science.

These projects were developed during my late high school and early undergraduate years, driven by my interest in understanding what neural networks learn internally, how information is represented, and how artificial representations relate to human cognition.

## Representation-vector geometric dynamics

The `representation-vector-geometric-dynamics` branch now includes [reproducible cat/dog experiments](experiments/geometric_dynamics/README.md) across ResNet, ConvNeXt, ViT and Swin: blockwise geometry, cosine local label entropy, held-out boundary diagnostics and attention animations. Original notebooks and the supplied CNN result table are preserved with source hashes. See the experiment README for methods, checkpoint differences, commands and limitations.

## Research Themes

The repository currently focuses on two related directions:

1. **Model–brain correspondence** — comparing the geometry of artificial visual representations with biological visual representations.
2. **Semantic composition** — studying how multiple concepts are combined inside a learned embedding space.

A short conceptual overview is available in [`theory/representation_questions.md`](theory/representation_questions.md).

## Projects

### 1. ResNet and Human Visual Cortex

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

## Repository Structure

```text
representation-alignment-/
├── README.md
├── requirements.txt
├── experiments/
│   └── geometric_dynamics/
├── .gitignore
├── data/
│   └── README.md
└── theory/
    └── representation_questions.md
```

The geometric dynamics experiments contain notebooks, scripts, small result tables, protocol documentation and offline checks.

## Reproducibility

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

Together, these projects reflect my early interest in studying neural networks not only through their outputs, but through the internal representations that give rise to those outputs.

## Research Process

These projects were independently developed out of personal interest during my late high school and early undergraduate years. I learned the relevant concepts largely through reading papers, discussing ideas, formulating research questions, writing and debugging code, and iteratively designing and analyzing experiments by discussing my questions and understandings hundreds of hours with AI.

At the time, I did not have formal training in research methodology, cognitive neuroscience, or representation learning. As a result, these projects were exploratory, and some of the questions and interpretations evolved substantially during the process.

I keep them here as an early record of my attempt to understand neural representations through computational experiments.
