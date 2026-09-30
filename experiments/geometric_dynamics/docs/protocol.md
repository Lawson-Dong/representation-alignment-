# Protocol and research audit

## Cohort and preprocessing

Read archive entries in ZIP order; retain cat./dog. image files under `/test/`. Labels are 3 (cat), 5 (dog). With NumPy `default_rng(42)`, sample 100 indices without replacement per class, concatenate cat then dog, then permute. Do not sort archive paths before sampling: that would change identities. All architectures use the explicit ResNet-18 V1 transform (resize 256, center crop 224, ImageNet normalization). This controls inputs but is not every checkpoint's preferred recipe. In particular the original CNN comment that all checkpoints share the same preferred recipe is inaccurate for ResNet-152 V2.

CNN outputs use spatial mean pooling of NCHW activations; stems are after maxpool for ResNet and the first feature module for ConvNeXt. ResNet stage first blocks and ConvNeXt downsample/first blocks are marked boundaries. ConvNeXt stage blocks retain their execution order. ViT CLS excludes the initially constant token; patch mean excludes CLS and includes positional input. Swin averages NHWC spatial dimensions. Final normalization is an explicit readout observation, not an attention block.

## Evidence and changes

The supplied CNN notebook contains execution outputs and exports 122 observations. The supplied LLE notebook contains executed ResNet-18/boundary analyses and outputs for all three added CNNs; the markdown saying the added cells were not run is stale. Those outputs document a prior run, not execution in this repository setup. Original notebooks are unchanged to preserve provenance. Their historical descriptions of PR/probes as auxiliary do not override the current [measurement framework](../../../README.md#main-measurements).

Maintained scripts remove browser downloads and save headless plots, export CNN vectors during the measurement pass, and extract the actual Python programs embedded in the attention notebook. Attention analysis uses the first attention readout as its cohort reference, resolving its otherwise mandatory, undocumented CNN NPZ prerequisite. The animation docstring incorrectly named ConvNeXt-Tiny; it now names attention models. No new model results are fabricated.

Seven supplied MP4 videos are included in the branch-root [visualization directory](../../../visualization/). The original notebook/CSV source manifest covers the historical sources listed there; it does not establish activation-level provenance for these later video uploads. Separate activation archives and attention metrics CSVs are not included.

## Statistical limits

LLE reduction indicates less local label mixing on this finite labeled cohort. It does not establish decreasing differential entropy of activation vectors, self-organization in a physical sense, intelligence, or alignment. Endpoint decreases can coexist with local increases. Boundary evidence concerns fitted held-out linear hyperplanes, not the full nonlinear class boundary; it was tested only for ResNet-18 here. Margins live in fold-standardized coordinates. Hypergeometric p-values are exploratory, uncorrected for the layer/k searches, and omit dependence from overlapping neighbor sets.

Permutation SD is null dispersion, not a confidence interval for unseen images. Repeated 70/30 probe splits overlap. Pooling discards spatial information. Relative depth is normalized observation order, not equal computation or semantically aligned stages. Different architectures and pretraining recipes confound depth-only interpretations. A CKA dip can reflect downsampling or normalization; it is not proof of a phase transition.

## Future work

Repeat across independent image cohorts and seeds; bootstrap image identities with attention to pair dependence; compare checkpoints within each architecture; preserve manifests and software/device metadata with every run. PR and linear-probe accuracy are included in the current measurement framework alongside geometry and LLE; future runs should state which readouts were evaluated for each model. Investigate stage-boundary effects before proposing change-point claims.
