# Figures by model

- [resnet18](resnet18/)
- [resnet152](resnet152/)
- [convnext_tiny](convnext_tiny/)
- [convnext_base](convnext_base/)
- [densenet121](densenet121/)
- [densenet169](densenet169/)
- [densenet201](densenet201/)

Each CNN folder contains 13 existing figures. Each DenseNet folder contains a geometry figure and an LLE figure, rendered from the existing model-specific CSVs. No model extraction or entropy measurement was rerun.

The two original combined DenseNet figures remain in [shared/densenet_20260930/](shared/densenet_20260930/) as historical runtime artifacts. Per-model videos are in the repository's [visualization directory](../../../../visualization/).

Regenerate the per-model DenseNet figures from the repository root:

```bash
python experiments/geometric_dynamics/scripts/plot_densenet_results.py
```
