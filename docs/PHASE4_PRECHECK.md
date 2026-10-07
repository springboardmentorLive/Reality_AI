# Phase 4A — Research-Hardening Precheck & System Audit

**Date:** 2026-10-02  
**Status:** Complete  
**Repository:** RealtyAI2  
**Purpose:** Independent verification of Phase 3 model artifacts, dataset manifests, split populations, feature contracts, time-series caches, and uncertainty quantification methodology prior to dashboard integration.

---

## 1. Canonical Active Checkpoints & Duplicate Analysis

Each production model pipeline in Phase 3 produced an active checkpoint as well as duplicate/legacy alias files. Every file was inspected directly via SHA-256 hashing and PyTorch/pickle state dictionary comparisons:

| Model Architecture | Canonical Active Checkpoint | Size (Bytes) | SHA-256 (First 16 chars) | Duplicate / Alias Artifact | Duplicate Status & Lineage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SpaceNet 2 U-Net** | `models/saved/unet_spacenet_v1.pt` | 7,821,487 | `28bedc6a4750f99b` | `models/saved/unet_satellite.pt` (7,821,239 B) | **Verified Byte-Equivalent Weights:** PyTorch state dictionary has 0 parameter differences across all 46 layers. Created as a legacy alias during Phase 3 saving. Canonical designated: `unet_spacenet_v1.pt`. |
| **PEER ResNet-18** | `models/saved/resnet_peer_collapse_v1.pt` | 45,056,045 | `b5c5d063f1dfa2a0` | `models/saved/resnet_condition.pt` (45,055,036 B) | **Verified Byte-Equivalent Weights:** PyTorch state dictionary has 0 parameter differences across all 122 layers. Created as a legacy alias during Phase 3 saving. Canonical designated: `resnet_peer_collapse_v1.pt`. |
| **Ames XGBoost Regressor** | `models/saved/xgboost_ames_v1.pkl` | 727,686 | `e9700a39ecf17ef5` | `models/saved/xgboost_price.pkl` (727,686 B) | **100% Byte-Identical:** Identical SHA-256 checksum. Canonical designated: `xgboost_ames_v1.pkl`. |
| **Ames LightGBM Regressor** | `models/saved/lightgbm_ames_v1.pkl` | 351,648 | `383d9a841af5ba69` | `models/saved/lightgbm_price.pkl` (351,648 B) | **100% Byte-Identical:** Identical SHA-256 checksum. Canonical designated: `lightgbm_ames_v1.pkl`. |
| **Zillow Prophet Forecaster** | `models/saved/forecaster_summary.json` + `data/processed/forecasts_cache.csv` | 13,052 / 34,714 | `b214df47775a...` | None | Canonical serialized parameters and projected forecasts cache for Top 10 MSAs. |

**Quarantine & Selection Rule:**
All active runtime loaders (`app/views/buyer_view.py`, `app/views/urban_planner_view.py`, `app/views/model_metrics_view.py`) must point strictly to the canonical versioned artifacts:
- `models/saved/unet_spacenet_v1.pt`
- `models/saved/resnet_peer_collapse_v1.pt`
- `models/saved/xgboost_ames_v1.pkl`
- `models/saved/forecaster_summary.json`

---

## 2. Dataset Manifests Used by Each Model

| Model | Official Dataset Manifest File(s) | Verification Check |
| :--- | :--- | :--- |
| **SpaceNet U-Net** | `data/manifests/spacenet_train_manifest.csv`<br>`data/manifests/spacenet_val_manifest.csv` | Verified present. Relative paths to authentic Las Vegas AOI RGB GeoTIFFs and vector-rasterized binary building masks. |
| **PEER ResNet-18** | `data/manifests/peer_train_manifest.csv`<br>`data/manifests/peer_val_manifest.csv`<br>`data/manifests/peer_test_manifest.csv` | Verified present. Relative image paths with ground-truth integer labels (`0: global_collapse`, `1: non_collapse`, `2: partial_collapse`). |
| **Ames XGBoost** | Processed feature matrices and split CSVs:<br>`data/processed/housing_train_df.csv`<br>`data/processed/housing_val_df.csv`<br>`data/processed/housing_test_df.csv` | Verified present. Generated from authentic `train.csv` (1,460 raw properties minus 2 commercial outliers = 1,458 valid residential rows). |
| **Zillow Prophet** | Raw source: `data/raw/zillow/zillow_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv`<br>Processed summary: `data/processed/zillow_metro_summary.csv`<br>Time series: `data/processed/zillow_zhvi_processed.csv` | Verified present. Filtered strictly to `RegionType == 'msa'` and sorted by ascending `SizeRank` (1 to 10). |

---

## 3. Exact Training, Validation, and Test Populations

| Model | Split Partition | Exact Sample Count (N) | Evaluation Scope / Notes |
| :--- | :--- | :--- | :--- |
| **SpaceNet U-Net** | Train | 24 satellite chips | 256×256 pixels, 3-band RGB, Las Vegas AOI. |
| | Validation | 6 satellite chips | Held-out development validation chips (`vegas_1041`, `1042`, `1047`, `1048`, `1049`, `1051`). |
| | Test | 0 chips | **No official public test ground-truth labels available locally.** Evaluated strictly on the 6 held-out validation chips. Never presented as a broad benchmark. |
| **PEER ResNet-18** | Train | 1,042 images | Training set for transfer learning with train-only class weights (`0: 0.7788`, `1: 1.2676`, `2: 1.0787`). |
| | Validation | 184 images | Used strictly for early checkpoint selection (Epoch 3 selected: Macro F1 = 0.7294). |
| | Benchmark Test | 146 images | Untouched benchmark test partition evaluated strictly once (Acc = 0.7055, Macro F1 = 0.6721). |
| **Ames XGBoost** | Train | 1,020 properties (70%) | Used for 5-fold cross-validation and primary model fitting. |
| | Validation | 219 properties (15%) | Used strictly for model selection between XGBoost (RMSE $25,667.70) and LightGBM (RMSE $25,669.62). |
| | Held-out Test | 219 properties (15%) | Evaluated strictly once after model selection (XGBoost MAE $15,092.41, RMSE $22,203.78, R² 0.9103, MAPE 9.17%). |
| **Zillow Prophet** | Train | 264 monthly observations per MSA | Historical window: 2000-01-31 to 2021-12-31 across 10 MSAs (2,640 total points). |
| | Validation | 24 monthly observations per MSA | Intermediate window: 2022-01-31 to 2023-12-31 across 10 MSAs (240 total points). |
| | Held-out Test | 32 monthly observations per MSA | Out-of-sample test window: 2024-01-31 to 2026-08-31 across 10 MSAs (320 total points). |
| | Future Horizon | 36 monthly projected intervals per MSA | Post-observed projections: 2026-09-30 to 2029-08-31. |

---

## 4. Exact Authoritative Metric Source Files

All dashboard views must load authoritative metrics directly from these JSON files, with zero hard-coded fallback numbers in Python UI code:

1. SpaceNet U-Net: `models/saved/unet_metrics.json`
   - Aggregate metrics: `aggregate_metrics.mean_iou` (`0.3593`), `aggregate_metrics.mean_dice` (`0.4985`), `aggregate_metrics.mean_precision` (`0.7142`), `aggregate_metrics.mean_recall` (`0.4334`).
   - Per-chip distribution: 6 chips with individual IoU/Dice.
2. PEER ResNet-18: `models/saved/condition_metrics.json`
   - Benchmark test evaluation: `accuracy` (`0.7055`), `macro_f1` (`0.6721`), `weighted_f1` (`0.6984`), `confusion_matrix` (3×3 matrix), per-class breakdowns.
3. Ames XGBoost: `models/saved/price_metrics.json`
   - Held-out test evaluation: `mae` (`15092.41`), `rmse` (`22203.78`), `r2_score` (`0.9103`), `mape_pct` (`9.17`).
   - 5-fold CV results for XGBoost and LightGBM.
   - Conformal uncertainty quantification specification.
4. Zillow Prophet: `models/saved/forecaster_summary.json`
   - Held-out test aggregate metrics: `mean_mae` (`39580.75`), `mean_rmse` (`45888.30`), `mean_mape_pct` (`8.71%`), `mean_95pct_interval_coverage` (`40.31%`).
   - Per-metro evaluations for all 10 MSAs.

---

## 5. Exact Inference Entry Points

- **SpaceNet U-Net**: `models.segmentation_unet.UNet` (in `models/segmentation_unet.py`).
  - Preprocessing: Resize to 256×256, normalize with ImageNet mean/std.
  - Forward: `torch.sigmoid(model(tensor)) > threshold`.
- **PEER ResNet-18**: `models.condition_resnet.PropertyConditionClassifier` (in `models/condition_resnet.py`).
  - Preprocessing: `get_condition_transforms(is_train=False)` (Resize 224×224, ImageNet normalization).
  - Forward: `torch.softmax(model(tensor), dim=1)`.
- **Ames Price Regressor**: `RealEstatePricePredictor.predict_property(input_dict)` (in `models/price_regressor.py`).
  - Preprocessing: `housing_preprocessor.pkl` (ColumnTransformer fitted strictly on `X_train`).
  - Prediction: Exponentiates log-scale model output: $\hat{y} = \exp(\hat{z}) - 1$.
  - Interval: Multiplicative relative conformal margin $[ \hat{y}(1 - q), \hat{y}(1 + q) ]$.
- **Zillow Forecaster**: Cached precomputed trajectory reader via `data/processed/forecasts_cache.csv`.

---

## 6. Exact Feature Ordering for the Pricing Model

Verified 100% agreement between `models/pricing_feature_contract.py` (`FEATURE_CONTRACT`) and `data/processed/housing_features.json` (`all_features`):

1. `LotArea` (Numeric)
2. `OverallQual` (Numeric)
3. `OverallCond` (Numeric)
4. `YearBuilt` (Numeric)
5. `YearRemodAdd` (Numeric)
6. `TotalBsmtSF` (Numeric)
7. `1stFlrSF` (Numeric)
8. `2ndFlrSF` (Numeric)
9. `GrLivArea` (Numeric)
10. `FullBath` (Numeric)
11. `HalfBath` (Numeric)
12. `BedroomAbvGr` (Numeric)
13. `KitchenAbvGr` (Numeric)
14. `TotRmsAbvGrd` (Numeric)
15. `Fireplaces` (Numeric)
16. `GarageCars` (Numeric)
17. `GarageArea` (Numeric)
18. `WoodDeckSF` (Numeric)
19. `OpenPorchSF` (Numeric)
20. `Neighborhood` (Categorical)
21. `MSZoning` (Categorical)
22. `BldgType` (Categorical)
23. `HouseStyle` (Categorical)
24. `ExterQual` (Categorical)
25. `Foundation` (Categorical)
26. `BsmtQual` (Categorical)
27. `KitchenQual` (Categorical)

Breakdown: 19 numeric features, 8 categorical features = 27 total features.

---

## 7. Exact Forecast-Cache Provenance

- File: `data/processed/forecasts_cache.csv`
- Generated by: `pipelines/06_train_forecaster.py` using Facebook Prophet with `interval_width=0.95`.
- Metropolitan Selection Protocol: Filtered strictly on `RegionType == 'msa'`, ordered by ascending `SizeRank` (ranks 1 through 10).
- Selected Metros:
  1. New York, NY
  2. Los Angeles, CA
  3. Chicago, IL
  4. Dallas, TX
  5. Houston, TX
  6. Washington, DC
  7. Philadelphia, PA
  8. Miami, FL
  9. Atlanta, GA
  10. Boston, MA
- Columns: `Date`, `RegionName`, `SizeRank`, `Forecast_ZHVI`, `Forecast_Lower`, `Forecast_Upper`.
- Temporal Boundaries:
  - Historical Observed: 2000-01-31 to 2026-08-31.
  - Evaluation Window (Held-Out Test): 2024-01-31 to 2026-08-31 (32 monthly steps per MSA).
  - Out-of-Sample Projection Window: 2026-09-30 to 2029-08-31 (36 monthly steps per MSA).

---

## 8. Audit of Uncertainty Calibration & Methodological Assumptions

### 8.1 Ames Housing Conformal Interval (CRITICAL FINDING)
- **Phase 3 Implementation**: In `pipelines/05_train_price_regressor.py`, model selection between XGBoost and LightGBM was determined by computing the lowest RMSE on the validation set `X_val` ($N=219$). Immediately following this selection, conformal residuals were calibrated on the **same** validation set `X_val` ($N=219$), arriving at a 90% relative margin of $\pm 22.23\%$.
- **Methodological Flaw**: Split Conformal Prediction requires exchangeability between the calibration scores and the unobserved test score conditional on the fitted model. When the candidate model $\hat{f} = \arg\min_{f \in \mathcal{M}} L(f, D_{val})$ is chosen via empirical risk minimization on $D_{val}$, conditioning on the selected model induces post-selection dependence on $D_{val}$. The empirical distribution of residuals on $D_{val}$ is systematically deflated (post-selection bias / winner's curse), destroying exchangeability.
- **Remediation in Phase 4B**:
  - Implement 5-Fold Cross-Conformal Prediction (Vovk 2015; Barber et al. 2021) utilizing out-of-fold (OOF) relative absolute residuals across all $N=1,020$ training properties generated during the 5-fold CV loop.
  - Because each out-of-fold prediction is evaluated strictly on an unseen CV fold, and because `X_val` ($N=219$) was completely withheld from CV residual calculation and reserved solely for model selection, the calibration residuals are independent of model selection.
  - Alternatively, a dedicated train-fit ($N=816$) vs calibration ($N=204$) split achieves valid split-conformal calibration. We adopt 5-fold cross-conformal out-of-fold calibration on $N=1,020$ because it preserves the full training set for model fitting, yielding a mathematically defensible 90% relative margin of **$\pm 19.59\%$** (achieving **92.24%** empirical coverage on the held-out test set $N=219$).

### 8.2 Zillow Prophet Forecast Interval
- **Methodology**: Facebook Prophet generates Bayesian posterior predictive intervals via Monte Carlo simulation under specified Gaussian observation noise and Laplace trend change-point priors (`interval_width=0.95`).
- **Empirical Validity**: The held-out test evaluation across all 10 MSAs yielded an empirical coverage of **40.31%**, far below the 95% nominal specification.
- **Required UI Presentation**:
  - The interval must be labeled strictly as `95% forecast interval`, NOT `95% calibrated interval` or `95% confidence interval`.
  - The empirical 40.31% test coverage must be displayed prominently and honestly.
  - No unsupported causal claims (e.g. attributing the gap to interest rate shifts) may be presented without an econometric causal analysis.
  - The target metric is Zillow's Home Value Index (ZHVI), which represents typical single-family and condo values across entire metropolitan regions and is distinct from single-property transaction prices.
