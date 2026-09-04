# Untracked artifacts

Model weights and prediction parquets are excluded from git (~4.3 GB total).
The reports in this directory are computed from them.

| Artifact | Regenerate with |
|---|---|
| `official_baseline_models/*/model.safetensors` | `scripts/ca_train_official_baseline.py` |
| `official_baseline_dev_predictions.parquet` | `scripts/ca_eval_official_baseline.py` |
| `../ranking_v2/cross_encoder_task1/*/checkpoints/` | cross-encoder training scripts |

The US-slice bootstrap in `us_slice_bootstrap/` reads the frozen prediction
parquets; regenerate those first if rerunning it.
