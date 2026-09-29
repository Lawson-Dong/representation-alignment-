# Data

This repository studies neural representations using external datasets and generated intermediate representations.

Large datasets, extracted activations, representational dissimilarity matrices, model checkpoints, and generated result archives should generally not be committed directly to Git.

## Project-specific data

### ResNet and Human Visual Cortex

The experiment described in the root README uses the 92-image dataset from the Algonauts 2019 benchmark together with biological visual-representation data for EVC and IT.

### CLIP and Semantic Composition

The semantic-composition experiment uses constructed hybrid visual stimuli and model embeddings generated from those stimuli.

When experiment code is added, this directory should document the exact expected file names, directory layout, preprocessing steps, and acquisition instructions needed for reproducibility.

### Cat/dog geometric dynamics

[Zenodo record 5226945](https://zenodo.org/records/5226945), archive `cats_dogs_light.zip`, MD5 `5e014163374c3bf7069c923de2d619c8`. Scripts download and verify it automatically, then select 100 cats and 100 dogs in a seeded, shared order. Dataset images and model weights are not redistributed. See [protocol](../experiments/geometric_dynamics/docs/protocol.md).
