# Protocol and research audit

## Cohort and preprocessing

Read archive entries in ZIP order; retain cat./dog. image files under `/test/`. Labels are 3 (cat), 5 (dog). With NumPy `default_rng(42)`, sample 100 indices without replacement per class, concatenate cat then dog, then permute. Do not sort archive paths before sampling: that would change identities. All architectures use the explicit ResNet-18 V1 transform (resize 256, center crop 224, ImageNet normalization). This controls inputs but is not every checkpoint's preferred recipe. In particular the original CNN comment that all checkpoints share the same preferred recipe is inaccurate for ResNet-152 V2.

CNN outputs use spatial mean pooling of NCHW activations; stems are after maxpool for ResNet and the first feature module for ConvNeXt. ResNet stage first blocks and ConvNeXt downsample/first blocks are marked boundaries. ConvNeXt stage blocks retain their execution order. ViT CLS excludes the initially constant token; patch mean excludes CLS and includes positional input. Swin averages NHWC spatial dimensions. Final normalization is an explicit readout observation, not an attention block.

## Evidence and changes

The supplied CNN notebook contains execution outputs and exports 122 observations. The LLE notebook was replaced in place with the completed Colab ResNet-152 V2 rerun on 2026-09-30, aligning its checkpoint with the geometry experiment. ResNet-18 and ConvNeXt-Tiny/Base remain V1, and sampling, preprocessing and analysis code are unchanged. Its execution-status markdown now describes the completed run; all uploaded code and outputs are preserved. The maintained LLE script uses the same V2 enum. These outputs document the uploaded Colab run, not a new full model execution in this repository setup. Other notebooks are unchanged. The source manifest records the current LLE artifact. Their historical descriptions of PR/probes as auxiliary do not override the current [measurement framework](../../../README.md#main-measurements).

Maintained scripts remove browser downloads and save headless plots, export CNN vectors during the measurement pass, and extract the actual Python programs embedded in the attention notebook. Attention analysis uses the first attention readout as its cohort reference, resolving its otherwise mandatory, undocumented CNN NPZ prerequisite. The animation docstring incorrectly named ConvNeXt-Tiny; it now names attention models. No new model results are fabricated.

Seven supplied MP4 videos are included in the branch-root [visualization directory](../../../visualization/). Separate activation archives are not included. The completed 44-row attention metrics CSV is preserved in `results/geometry/shared/attention_geometry_metrics.csv`.

## Statistical limits

LLE reduction indicates less local label mixing on this finite labeled cohort. It does not establish decreasing differential entropy of activation vectors, self-organization in a physical sense, intelligence, or alignment. Endpoint decreases can coexist with local increases. Boundary evidence concerns fitted held-out linear hyperplanes, not the full nonlinear class boundary; it was tested only for ResNet-18 here. Margins live in fold-standardized coordinates. Hypergeometric p-values are exploratory, uncorrected for the layer/k searches, and omit dependence from overlapping neighbor sets.

Permutation SD is null dispersion, not a confidence interval for unseen images. Repeated 70/30 probe splits overlap. Pooling discards spatial information. Relative depth is normalized observation order, not equal computation or semantically aligned stages. Different architectures and pretraining recipes confound depth-only interpretations. A CKA dip can reflect downsampling or normalization; it is not proof of a phase transition.

## Future work

Repeat across independent image cohorts and seeds; bootstrap image identities with attention to pair dependence; compare checkpoints within each architecture; preserve manifests and software/device metadata with every run. PR and linear-probe accuracy are included in the current measurement framework alongside geometry and LLE; future runs should state which readouts were evaluated for each model. Investigate stage-boundary effects before proposing change-point claims.

## DenseNet extension, 2026-09-30

The [completed requested Colab run](../results/geometry/shared/densenet_20260930/) adds DenseNet-121/169/201 V1 with the same archive and original selection algorithm, preprocessing, metric formulas, paired probes and shared LLE permutations. Dense layers read cumulative concatenated channels, verified against actual dense-block output pooling in every batch. Transitions and final norm+ReLU are separate observations. Source, identities, versions and runtime hashes accompany CSVs; activation NPZs remain outside Git. Widths, block allocation and independently trained checkpoints vary, so this is a controlled input/measurement comparison rather than a causal depth-only experiment. Historical NPZ identity checks cannot be claimed because those archives are not in the checkout. See the run README for observed endpoints and final-readout effects.


## Stratified CKA bootstrap (October 2026)

The primary readout is adjacent biased linear CKA on raw spatially pooled representations. S and held-out probe are secondary; Fisher, PR, pairwise distances and norms are descriptive. Use one plot per architecture at its own observed layer index.

Draw 100 cats with replacement and 100 dogs with replacement from the observed image identities, repeating 1,000 times (bootstrap seed 20261006). Reuse the same index draw for every layer and all four models; validate both labels and archive paths before pairing. Resample raw Gram matrices on both axes and recenter for each draw. Use pointwise 10th–90th percentile intervals (central 80%). These quantify image-sampling uncertainty conditional on the observed empirical cohort, fixed weights and preprocessing; they do not cover new training seeds or distribution shift.

For each stage, Δ is the median adjacent CKA of its later blocks minus the adjacent CKA of its first block. Export original Δ, p10, median, p90 and the fraction of bootstrap draws with Δ > 0; the fraction is not a p-value or a posterior probability. Exclude stem and standalone downsample observations from this contrast. ConvNeXt first-block adjacencies start at the downsample output, whereas ResNet first-block adjacencies start at the previous stage output; inspect downsample points separately. Within-stage contrasts reduce but do not eliminate depth confounding, and are not a causal downsampling test.

Do not fit probe train/test splits on duplicated bootstrap rows: duplicate image identities could leak across splits. Preserve the existing five paired held-out splits and report their descriptive SD separately.
