# Data

This repository studies neural representations using external datasets and generated intermediate representations.

Large datasets, extracted activations, representational dissimilarity matrices, model checkpoints, and generated result archives should generally not be committed directly to Git.

## Project-specific data

### ResNet and Human Visual Cortex

The experiment described in the root README uses the 92-image dataset from the Algonauts 2019 benchmark together with biological visual-representation data for EVC and IT.

### CLIP and Semantic Composition

The semantic-composition experiment uses constructed hybrid visual stimuli and model embeddings generated from those stimuli.

Experiment notebooks are available on the [RSA](https://github.com/Lawson-Dong/representation-alignment-/tree/resnet-human-visual-cortex-rsa) and [CLIP](https://github.com/Lawson-Dong/representation-alignment-/tree/clip-semantic-composition) branches. Consult the notebook cells for data acquisition and preprocessing.

### Cat/dog geometric dynamics

[Zenodo record 5226945](https://zenodo.org/records/5226945), archive `cats_dogs_light.zip`, MD5 `5e014163374c3bf7069c923de2d619c8`. Scripts select the same 100 cats and 100 dogs using seed 42, preserving archive order before sampling and matching image order across models. Dataset images and model weights are not redistributed.

The [geometric dynamics branch](https://github.com/Lawson-Dong/representation-alignment-/tree/representation-vector-geometric-dynamics) contains the experiment-specific scripts and requirements. See its [protocol](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/docs/protocol.md) for cohort selection and pooling, and its [experiment README](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/README.md) for output layouts and reproduction commands.
