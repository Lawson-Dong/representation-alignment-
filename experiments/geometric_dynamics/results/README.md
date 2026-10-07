# Research results index

Final outputs are stored once per model. Metadata records settings and provenance; it is not another result set. No duplicate combined result tables are published.

| Model / readout | Final geometry | Final LLE CSVs | Figures |
|---|---|---|---|
| ResNet-18 | [geometry](geometry/resnet18/) | Executed CNN LLE notebook only | [figures](figures/resnet18/) |
| ResNet-152 V2 | [geometry](geometry/resnet152/) | Executed CNN LLE notebook only | [figures](figures/resnet152/) |
| ConvNeXt-Tiny | [geometry](geometry/convnext_tiny/) | Executed CNN LLE notebook only | [figures](figures/convnext_tiny/) |
| ConvNeXt-Base | [geometry](geometry/convnext_base/) | Executed CNN LLE notebook only | [figures](figures/convnext_base/) |
| DenseNet-121 | [geometry](geometry/densenet121/) | [LLE](lle/densenet121/) | [figures](figures/densenet121/) |
| DenseNet-169 | [geometry](geometry/densenet169/) | [LLE](lle/densenet169/) | [figures](figures/densenet169/) |
| DenseNet-201 | [geometry](geometry/densenet201/) | [LLE](lle/densenet201/) | [figures](figures/densenet201/) |
| ViT-B/16 CLS | [geometry](geometry/vit_b16_cls/) | Not measured | See historical attention notebook / videos |
| ViT-B/16 patch mean | [geometry](geometry/vit_b16_patchmean/) | Not measured | See historical attention notebook / videos |
| Swin-T | [geometry](geometry/swin_t/) | Not measured | See historical attention notebook / videos |

## Bootstrap analysis

Start with [whole-curve stability: methods, results and limitations](../docs/bootstrap_curve_stability.md). Each of the four CNN folders has curve-alignment summary and layer-wise CSVs, plus alignment and deviation figures. Existing `boundary_dip.csv` is a secondary within-stage diagnostic, not the primary research hypothesis. Only CKA was image-bootstrapped.

## Provenance and reproduction

- [Run metadata and output hashes](metadata/)
- [Experiment setup and reproduction commands](../README.md)
- [Protocol and historical execution audit](../docs/protocol.md)
- [Executed CNN LLE notebook](../notebooks/Cat_Dog_LLE_Matched_Samples_resnet_convnext.ipynb)
- [Videos](../../../visualization/)

Per-replicate draws, sampling/split indices, activations and per-image entropy are excluded from published final outputs. Removed artifacts can be recovered from Git history. Executed notebook code and original outputs are preserved.
