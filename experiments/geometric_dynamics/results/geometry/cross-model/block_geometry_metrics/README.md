# Completed stratified-bootstrap results

This folder replaces `block_geometry_metrics.csv`. The executed notebook contains separate metric cells and GitHub CSV links. Matched 100 cats + 100 dogs; four pretrained CNNs; 1,000 paired class-stratified bootstrap draws. See `run_record.json` for run settings and CSV hashes. Figures are available under `../../../figures/<model>/`.

| Export | CSV |
| --- | --- |
| CKA_prev | [CKA_prev.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/CKA_prev.csv) |
| CKA_prev_bootstrap | [CKA_prev_bootstrap.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/CKA_prev_bootstrap.csv) |
| Fisher_raw | [Fisher_raw.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/Fisher_raw.csv) |
| PR | [PR.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/PR.csv) |
| S | [S.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/S.csv) |
| boundary_dip | [boundary_dip.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/boundary_dip.csv) |
| d_between_cos | [d_between_cos.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/d_between_cos.csv) |
| d_between_euclid_unit | [d_between_euclid_unit.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/d_between_euclid_unit.csv) |
| d_within_cos | [d_within_cos.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/d_within_cos.csv) |
| d_within_euclid_unit | [d_within_euclid_unit.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/d_within_euclid_unit.csv) |
| mean_raw_norm | [mean_raw_norm.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/mean_raw_norm.csv) |
| probe_mean | [probe_mean.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/probe_mean.csv) |
| probe_sd | [probe_sd.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/probe_sd.csv) |
| similarity_fingerprint_trajectory | [similarity_fingerprint_trajectory.csv](https://github.com/Lawson-Dong/representation-alignment-/blob/representation-vector-geometric-dynamics/experiments/geometric_dynamics/results/geometry/cross-model/block_geometry_metrics/similarity_fingerprint_trajectory.csv) |

Intervals are pointwise p10–p90 (central 80%). Metrics and centroid projections are interpreted within each model; independent PCA axes are not comparable across models.

Per-model CSVs are in `../../resnet18/`, `../../resnet152/`, `../../convnext_tiny/` and `../../convnext_base/`. Figures are in `../../../figures/<model>/`. Combined final metric tables and run settings remain here. Per-replicate values and sampling indices are omitted.
