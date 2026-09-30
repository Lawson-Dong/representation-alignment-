# Data

The geometric dynamics experiments use external images and pretrained model weights, and generate intermediate representations during execution.

Large datasets, extracted activations, representational dissimilarity matrices, model checkpoints, and generated result archives should generally not be committed directly to Git.

## Cat/dog geometric dynamics

[Zenodo record 5226945](https://zenodo.org/records/5226945), archive `cats_dogs_light.zip`, MD5 `5e014163374c3bf7069c923de2d619c8`. Scripts download and verify it automatically, then select 100 cats and 100 dogs using seed 42 in a shared order. Do not sort archive paths before sampling, as this changes the selected cohort.

Dataset images and model weights are not redistributed. See the [protocol](../experiments/geometric_dynamics/docs/protocol.md) for cohort selection, preprocessing and pooling, and the [experiment README](../experiments/geometric_dynamics/README.md) for output files and execution commands.

The preserved CNN metrics table is in `experiments/geometric_dynamics/results/`; supplied videos are in [visualization](../visualization/). Generated per-image activation archives remain outside Git.

## Other research directions

The [RSA](https://github.com/Lawson-Dong/representation-alignment-/tree/resnet-human-visual-cortex-rsa) and [CLIP semantic composition](https://github.com/Lawson-Dong/representation-alignment-/tree/clip-semantic-composition) experiments have separate branches. The [main-branch data overview](https://github.com/Lawson-Dong/representation-alignment-/blob/main/data/README.md) describes the repository's three data contexts.
