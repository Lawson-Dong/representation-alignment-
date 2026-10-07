# Within-model bootstrap CKA curve stability

## Research purpose

Evaluate whether the observed **complete adjacent-CKA trajectory within each model** is sensitive to image resampling. This supplements the measured curve with uncertainty evidence; it does not assume that every stage's first block must show a CKA decrease. First-block contrasts in `boundary_dip.csv` remain a secondary diagnostic.

The fixed-weight experiment uses the same 100 cats and 100 dogs. Each of 1,000 paired class-stratified bootstrap draws samples 100 cats and 100 dogs with replacement. The same image draw is used at every observation, preserving the joint cross-layer trajectory. These are image resamples, not independent training runs or new datasets.

## Read the results

Each model has two final CSVs under `results/geometry/<model>/` and two figures under `results/figures/<model>/`:

- `bootstrap_curve_alignment_summary.csv`: distribution summaries of whole-curve Pearson correlation, Spearman correlation, adjacent-change correlation, RMSE, and bootstrap-to-bootstrap curve correlation.
- `bootstrap_curve_alignment_by_layer.csv`: original CKA, bootstrap p10/median/p90, interval width and median-minus-original difference at each defined observation.
- `bootstrap_curve_alignment.png`: all 1,000 trajectories, original curve, bootstrap median and pointwise central 80% band.
- `bootstrap_curve_deviations.png`: layer-wise numerical differences from the original curve. This makes shifts visible even when the overall shape is similar.

All alignment uses the **same original depth and stage labels within one model**. There is no curve warping, smoothing, interpolation, normalization of depth or cross-model layer matching. Depth 0 is excluded because adjacent CKA is undefined. The valid observation counts are 8, 50, 21 and 39 respectively.

| Model | Pearson r median [p10, p90] | Spearman rho median | Adjacent-change r median | RMSE median |
|---|---:|---:|---:|---:|
| ResNet-18 | 0.9720 [0.9374, 0.9902] | 0.9762 | 0.9806 | 0.0184 |
| ResNet-152 | 0.9993 [0.9987, 0.9996] | 0.9759 | 0.9991 | 0.0052 |
| ConvNeXt-Tiny | 0.9975 [0.9950, 0.9987] | 0.9935 | 0.9978 | 0.0178 |
| ConvNeXt-Base | 0.9985 [0.9968, 0.9993] | 0.9915 | 0.9987 | 0.0118 |

The table reports each model's own descriptive results; it is not a test of superiority between architectures.

## Definitions and interpretation

For a reference curve `c` and bootstrap curve `b`, evaluated at identical observations:

- **Pearson correlation:** agreement in centered whole-curve shape; 1 means perfect positive linear association, not identical numerical values or a percentage agreement.
- **Spearman correlation:** agreement in ranking of layer values. Small differences among near-equal CKA values can change ranks.
- **Adjacent-change Pearson correlation:** correlation between successive differences of `b` and `c`. It checks local increases/decreases beyond a single whole-curve correlation.
- **RMSE:** `sqrt(mean((b - c)^2))`, in CKA units. Lower values mean closer numerical agreement without removing shifts or rescaling.
- **Pairwise curve correlation:** correlation for all 499,500 distinct pairs of bootstrap trajectories. These pairs overlap and are dependent; their percentiles are descriptive, not evidence from 499,500 independent experiments.
- **Pointwise band:** each observation's p10-p90 interval across draws. It does not provide 80% simultaneous coverage for the entire curve.

The results support high overall shape agreement under resampling of the current cohort, with more visible variability in ResNet-18. Absolute numerical deviations remain: median curve RMSE ranges from approximately 0.0052 to 0.0184. Bootstrap medians are often above original estimates; high shape correlations must not conceal these shifts. The difference plots explicitly show them.

Large troughs and a wide dynamic range can dominate Pearson correlation. High whole-curve r does not establish that every small peak or trough is stable. Report rank agreement, local-change agreement and numerical error together. These descriptive metrics were selected after the completed bootstrap run and are exploratory; no significance threshold was prespecified.

This analysis strengthens **conditional image-resampling stability evidence for CKA trajectories**. It does not validate S, LLE or probe uncertainty, generalization to new datasets, training-seed variability, causal explanations, or phase transitions.

## Reproduce without rerunning a model

From the repository root, install numpy, pandas, scipy and matplotlib, then:

```bash
python experiments/geometric_dynamics/scripts/analyze_bootstrap_curves.py
python -m unittest discover -s experiments/geometric_dynamics/tests -v
```

The default input is the original replicate CSV recovered directly from Git commit `ce2945f`, without restoring intermediate files into the working tree. A shallow checkout must fetch that commit first. The analysis checks explicit depth/stage identity, all 1,000 complete trajectories, and equality of recomputed percentiles to the existing final bootstrap summaries.

For a new completed bootstrap run, place its matching original `CKA_prev.csv` and final `CKA_prev_bootstrap.csv` in each model folder of the selected results directory and run:

```bash
python experiments/geometric_dynamics/scripts/analyze_bootstrap_curves.py \
  --replicates /path/to/CKA_prev_bootstrap_replicates.csv \
  --results /path/to/matching/results
```

The source commit/path and SHA-256, reference hashes, script hash, alignment rules and exclusions are recorded in [curve_alignment_record.json](../results/metadata/cnn_bootstrap/curve_alignment_record.json). The script updates the retained-output hash manifest. Raw draws remain available in Git history and need not be republished. Original executed notebook code and outputs are unchanged; the notebook contains a linked analysis appendix.
