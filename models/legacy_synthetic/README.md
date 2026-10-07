# Legacy Artifacts Quarantine & Historical Provenance Audit

These model artifacts and metrics were generated prior to Phase 1/Phase 2 data migrations and preprocessing repairs. In accordance with Phase 3 audit standards, each artifact is explicitly classified based on direct evidence:

| Artifact | Classification | Provenance & Evidence |
| :--- | :--- | :--- |
| `unet_satellite.pt` | **SYNTHETIC TRAINING** | Weights trained on 120 procedural geometric PIL-drawn synthetic satellite tiles (`data/synthetic_backup/spacenet/`). Invalid for real-world remote sensing inference. |
| `resnet_condition.pt` | **SYNTHETIC TRAINING** | Weights trained on 180 procedural PIL-drawn synthetic cartoon drawings (`data/synthetic_backup/property_conditions/`). Invalid for structural condition classification. |
| `xgboost_price.pkl` | **LEAKAGE-COMPROMISED TRAINING** | Trained on real Ames housing data, but compromised by dataset-wide median imputation and scaling performed prior to train/test partitioning (CRIT-01 data leakage). |
| `lightgbm_price.pkl` | **LEAKAGE-COMPROMISED TRAINING** | Trained on real Ames housing data, but compromised by dataset-wide median imputation performed prior to train/test partitioning. |
| `unet_metrics.json` | **SYNTHETIC TRAINING EVALUATION** | Evaluation metrics generated from testing `unet_satellite.pt` on procedural synthetic tiles. |
| `condition_metrics.json` | **SYNTHETIC TRAINING EVALUATION** | Evaluation metrics generated from testing `resnet_condition.pt` on synthetic cartoon drawings. |
| `price_metrics.json` | **LEAKAGE-COMPROMISED EVALUATION** | Evaluation metrics generated from leaked Ames housing preprocessing split. |
| `forecaster_summary.json` | **REAL DATA / OLD EXPERIMENT** | Baseline Prophet forecast summary generated from authentic Zillow ZHVI data, but using legacy top-50 filtering by row count rather than market importance. |

### Quarantine Status
**STRICTLY QUARANTINED & DEPRECATED.**  
No artifact in this directory may be loaded, executed, or cited for active production inference, validation, or research claims.
