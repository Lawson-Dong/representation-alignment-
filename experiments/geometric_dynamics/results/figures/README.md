# Figures by model

- [resnet18](resnet18/)
- [resnet152](resnet152/)
- [convnext_tiny](convnext_tiny/)
- [convnext_base](convnext_base/)
- [densenet121](densenet121/)
- [densenet169](densenet169/)
- [densenet201](densenet201/)

Each CNN has 13 original figures plus two whole-curve bootstrap analysis figures. Each DenseNet has a geometry and an LLE figure, rendered from its canonical CSVs. Combined duplicate panels are omitted. Per-model videos are in [visualization](../../../../visualization/).

Regenerate the DenseNet figures with:

```bash
python experiments/geometric_dynamics/scripts/plot_densenet_results.py
```
