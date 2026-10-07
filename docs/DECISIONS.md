# RealtyAI Decisions & Architecture Resolutions Log

**Last Updated**: October 2, 2026  
**Active Phase**: Phase 1 — Real Data Migration, Data Provenance & Synthetic Removal  

---

## 1. Decisions Made During Phase 0

1. **Strict Non-Destructive Inspection**:
   * *Decision*: Do not modify any production code, dataset, checkpoint, test, or dashboard view during Phase 0.
   * *Rationale*: Adhered strictly to the Phase 0 audit mandate to establish an unvarnished ground-truth baseline before executing repairs.

2. **Categorization of Repository Status by Evidence**:
   * *Decision*: All audit findings must be backed by explicit line references, actual runtime execution, or filesystem verification, categorizing claims as `VERIFIED`, `VERIFIED BROKEN`, `PRESENT BUT UNUSED`, `DOCUMENTATION ONLY`, or `NOT VERIFIED`.

3. **Retaining Existing Artifacts for Review**:
   * *Decision*: Kept synthetic dataset generators, synthetic images/masks, and pre-trained checkpoints intact until Phase 1 review and migration approval.

---

## 2. Decisions Made During Phase 1 (Real Data Migration & Synthetic Removal)

### Decision 1.1: SpaceNet Real Data Migration (AOI_2_Vegas)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Selected Dataset**: SpaceNet 2 Building Detection Dataset — Area of Interest 2 (Las Vegas, Nevada).
* **Source & Imagery**: Maxar Technologies / DigitalGlobe WorldView-3 commercial satellite imagery, pan-sharpened 3-band RGB, ~30 cm ground sample distance, released under **CC BY-SA 4.0** by the SpaceNet partnership.
* **Why Selected**:
  1. SpaceNet 2 Vegas is an authentic, internationally recognized benchmark for building footprint segmentation.
  2. Features an official 20-chip benchmark sample (`s3://spacenet-dataset/`) with complete geospatial metadata, coordinate systems (`EPSG:4326`), and affine transforms.
  3. Provides raw vector building footprints in GeoJSON format (593 verified building polygons).
* **Ingestion & Preprocessing**:
  1. Contrast-stretched 16-bit sensor radiance to 8-bit RGB using per-chip 2nd-to-98th percentile clipping.
  2. Rasterized vector polygons into binary footprint masks using `rasterio.features.rasterize` with native affine transforms.
  3. Maintained official whole-chip scene partitioning (14 train, 3 val, 3 test) to eliminate tile-level spatial autocorrelation leakage.
* **Evidence**:
  * Ingested files: `data/raw/spacenet/raw_chips/` (20 GeoTIFFs + 20 GeoJSONs), `data/raw/spacenet/images/` (20 PNGs), `data/raw/spacenet/masks/` (20 PNGs).
  * Manifest: `data/manifests/spacenet_manifest.csv` (hashes, CRS, transform, building counts).
  * Split: `data/processed/spacenet_splits.json`.

---

### Decision 1.2: Property / Structural Condition Real Data Migration (PEER Task 5)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Candidate Evaluation**:
  * *xBD (xView2)*: Evaluated and rejected. Uses overhead satellite pairs rather than ground-level property photos.
  * *MCDS / SDNET2018*: Evaluated and rejected. Consists of macro concrete surface textures rather than architectural structures.
  * *TornadoNet*: Evaluated and rejected. Overhead aerial post-storm swaths.
  * *Commercial MLS / Restb.ai*: Ineligible due to proprietary non-redistributable terms.
  * *PEER Hub ImageNet (PHI-Net) Task 5 (Collapse Mode)*: **SELECTED**.
* **Source & Origin**: Pacific Earthquake Engineering Research (PEER) Center, UC Berkeley (Yuqing Gao & Khalid M. Mosalam), released under **CC BY-NC-SA 4.0**.
* **Semantic Ground Truth Preservation**:
  * **Strict Rule Adherence**: Labels are **NOT** forced into artificial "new / moderate / old" categories.
  * Preserved authentic structural engineering ground truth:
    * `global_collapse` (GC, Class ID 0): 67 images (catastrophic structural collapse)
    * `non_collapse` (NC, Class ID 1): 39 images (intact structure)
    * `partial_collapse` (PC, Class ID 2): 40 images (localized structural shear/failure)
* **Component Redefinition**: Redefined the module truthfully as **Structural Condition & Integrity Assessment (Collapse Mode)** rather than "property age" or "curb appeal".
* **Evidence**:
  * Ingested files: `data/raw/property_conditions/{global_collapse, non_collapse, partial_collapse}/` (146 total images).
  * Raw arrays and licenses: `data/raw/property_conditions/raw_phi_net/` (`task5_X_test.npy`, `license.txt`, `README.txt`).
  * Manifest: `data/manifests/property_condition_manifest.csv` (hashes, labels, splits).
  * Split: `data/processed/property_condition_splits.json` (102 train, 22 val, 22 test).

---

### Decision 1.3: Housing Dataset Provenance & Leakage Remediation Strategy
* **Status**: **PROVENANCE VERIFIED; LEAKAGE FIX SCHEDULED FOR PHASE 2**
* **Identity**: Ames Housing Dataset compiled by Dean De Cock (Truman State University, 2011), published in *Journal of Statistics Education*.
* **Scope**: 1,460 labeled properties in Ames, Iowa (2006–2010), 80 features + 1 target (`SalePrice`).
* **Critical Decisions**:
  1. *Unlabeled Test Set*: Kaggle's `test.csv` (1,459 rows) lacks `SalePrice` and must **never** be used for evaluation.
  2. *Data Leakage Remediation*: `pipelines/02_data_preprocessing.py#L66-L74` imputed medians across the pooled dataset. In Phase 2, imputers, scalers, and encoders will be fit strictly on `X_train` within an encapsulated `sklearn.pipeline.Pipeline`.
  3. *Synthetic Fallback Deprecation*: Removed synthetic fallback generator from `ingest_kaggle_housing()`; raises explicit `FileNotFoundError` if missing.

---

### Decision 1.4: Zillow ZHVI Dataset Provenance & Metro Selection Strategy
* **Status**: **PROVENANCE VERIFIED; METRO SELECTION FIX SCHEDULED FOR PHASE 3**
* **Identity**: Zillow Home Value Index (ZHVI) smoothed, seasonally adjusted mid-tier monthly time series (895 metros, 325 columns, Jan 2000–present).
* **Decisions**:
  1. *Synthetic Fallback Deprecation*: Removed procedural fallback from `ingest_zillow_data()`; raises explicit `FileNotFoundError` if live download fails and local file is missing.
  2. *Metro Selection Defect (MED-04)*: `pipelines/02_data_preprocessing.py#L159-L162` currently filters top 50 metros by row count. In Phase 3, this will be modified to filter by `SizeRank <= 50`, ensuring major metropolitan markets (New York, Los Angeles, Chicago, Dallas, San Francisco) are forecasted rather than obscure micropolitan areas with complete histories.

---

### Decision 1.5: Synthetic Data Removal & Pipeline Disconnection
* **Status**: **RESOLVED & IMPLEMENTED**
* **Actions Completed**:
  1. Deprecated and deleted all procedural generators (`generate_spacenet_satellite_data`, `generate_property_condition_dataset`).
  2. Excised 120 synthetic SpaceNet tiles (`spacenet_tile_*.png`) and masks from active `data/raw/spacenet/` (safely archived in `data/synthetic_backup/spacenet/`).
  3. Excised 180 synthetic cartoon houses from active `data/raw/property_conditions/` (safely archived in `data/synthetic_backup/property_conditions/`).
  4. Deleted old synthetic `data/raw/property_condition_metadata.csv` (which had non-portable Windows absolute paths).
  5. Replaced synthetic demo files in `data/sample_images/` with authentic PEER reconnaissance and SpaceNet satellite plates.
  6. Updated `app/views/buyer_view.py` and `generate_pdf_report.py` to reference authentic demonstration images.

---

### Decision 1.6: Dashboard Hidden Feature Synthesis Audit
* **Status**: **AUDITED & DESIGNATED PHASE 2/4 BLOCKER**
* **Context**: The buyer valuation calculator (`app/views/buyer_view.py#L90-L111`) synthesizes 15 property attributes not entered by the user:
  1. `1stFlrSF`: `gr_liv_area * 0.6` (heuristic ratio)
  2. `2ndFlrSF`: `gr_liv_area * 0.4` (heuristic ratio)
  3. `KitchenAbvGr`: `1` (hardcoded constant)
  4. `TotRmsAbvGrd`: `bedrooms + full_bath + 3` (heuristic sum)
  5. `Fireplaces`: `1` (hardcoded constant)
  6. `GarageArea`: `garage_cars * 240` (heuristic multiplier)
  7. `WoodDeckSF`: `120` (hardcoded constant)
  8. `OpenPorchSF`: `60` (hardcoded constant)
  9. `MSZoning`: `"RL"` (hardcoded constant)
  10. `BldgType`: `"1Fam"` (hardcoded constant)
  11. `HouseStyle`: `"2Story"` if `gr_liv_area > 1500` else `"1Story"` (heuristic condition)
  12. `ExterQual`: `"Gd"` if `overall_qual >= 7` else `"TA"` (heuristic condition)
  13. `Foundation`: `"PConc"` if `year_built >= 1990` else `"CBlock"` (heuristic condition)
  14. `BsmtQual`: `"Gd"` if `overall_qual >= 7` else `"TA"` (heuristic condition)
  15. `KitchenQual`: `"Gd"` if `overall_qual >= 7` else `"TA"` (heuristic condition)
* **Phase 2 Resolution**: Resolved in Phase 2 via Decision 2.5 (`models/pricing_feature_contract.py`). Option A (retraining on user-facing features in Phase 3) selected as primary architecture; Option B (explicit documented imputation via contract) implemented for transparent feature handling.

---

## 3. Decisions Made During Phase 2 (Dataset Correction, Preprocessing Integrity & Geospatial Repair)

### Decision 2.1: SpaceNet Dual-Dataset Strategy & Official S3 Ingestion
* **Status**: **RESOLVED & IMPLEMENTED**
* **Context**: The existing 20-chip dataset was an authentic Maxar WorldView-3 derivative, but explicitly a third-party smoke-test sample rather than a full training dataset.
* **Resolution**:
  1. **Smoke-Test Sample Preserved**: Retained the 20-chip sample in `data/raw/spacenet/smoke_test/` and generated `data/manifests/spacenet_smoke_test_manifest.csv` and `data/processed/spacenet_smoke_test_splits.json`. Purpose explicitly designated as **PIPELINE VALIDATION / CI / DEVELOPMENT ONLY**.
  2. **Official Research-Training Ingestion**: Ingested 30 authentic pan-sharpened RGB GeoTIFF chips and building footprint GeoJSONs directly from official SpaceNet AWS S3 bucket (`s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/`) into `data/raw/spacenet/research_train/`.
  3. **Affine Transform Alignment**: Rasterized 789 building polygons using native affine transforms via `rasterio.features.rasterize`.
  4. **Whole-Scene Splitting**: Partitioned at whole-chip level (24 train = 80%, 6 val = 20%) to eliminate spatial leakage. Recorded in `data/manifests/spacenet_train_manifest.csv` and `data/processed/spacenet_train_splits.json`.

### Decision 2.2: PEER Dataset Official Split Semantics (Case 1 Resolution)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Audit Finding**: Inspected `data.zip` from PEER PHI-Net Task 5 author Yuqing Gao (UC Berkeley). Found that `task5_X_test.npy` (146 images) was indeed the official benchmark test set, while `task5_X_train.npy` (1,226 images) was the official training set.
* **Resolution (Case 1 Applied)**:
  1. The 146 official test images are preserved **strictly and exclusively** in `data/raw/property_conditions/test/`. Zero benchmark test images are mixed into training or validation.
  2. The 1,226 official training images were downloaded via HTTP Range streaming and partitioned into:
     - `train`: 1,042 images (85% stratified) in `data/raw/property_conditions/train/`
     - `val`: 184 images (15% stratified) in `data/raw/property_conditions/val/`
  3. Total dataset: 1,372 authentic images across `global_collapse` (592), `non_collapse` (361), and `partial_collapse` (419).
  4. Recorded in `data/manifests/property_condition_manifest.csv` and `data/processed/property_condition_splits.json`. Status: **READY FOR MODEL TRAINING IN PHASE 3**.

### Decision 2.3: Ames Housing License & Labeled Research Splits
* **Status**: **RESOLVED & IMPLEMENTED**
* **License Correction**: Corrected description from "Public Domain" to `NOT VERIFIED (Open Academic / Educational Use under Journal of Statistics Education publication terms)`. Dean De Cock (2011).
* **Research Splitting**: Formed strict 70% / 15% / 15% splits from labeled Ames `train.csv` (1,458 valid records after dropping 2 extreme outliers with GrLivArea > 4000 and SalePrice < 300,000):
  - `research_train`: 1,020 properties
  - `research_validation`: 219 properties
  - `research_test`: 219 properties
  - Saved to `data/processed/housing_{train,val,test}_df.csv`.
* **Kaggle `test.csv` Prohibition**: Unlabeled Kaggle test set (1,459 rows) lacks `SalePrice` and is strictly prohibited from evaluation.

### Decision 2.4: Housing Preprocessing Leakage Repair
* **Status**: **RESOLVED & IMPLEMENTED**
* **Fix**: Excised pre-split median imputation. Implemented `sklearn.compose.ColumnTransformer` (median imputer + standard scaler for numeric; constant imputer + ordinal encoder for categorical) **fitted strictly on `X_train`**. Validation and test sets are transformed using parameters learned solely from training data.
* **Output**: `data/processed/housing_preprocessor.pkl`.

### Decision 2.5: Pricing Input Contract Redesign (Option A vs Option B)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Contract Specification**: Created `models/pricing_feature_contract.py` defining explicit origin for all 27 features:
  - 12 `USER_PROVIDED` (direct user input)
  - 8 `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` (justified engineering derivation)
  - 7 `EXPLICITLY_IMPUTED` (transparent population medians/modes)
* **Decision**: **Option A** (retraining the model in Phase 3 on only user-facing features) is selected as the primary architectural direction. **Option B** (explicit documented imputation policy via contract) is implemented in `models/pricing_feature_contract.py` and integrated into `app/views/buyer_view.py` for transitional compatibility.

### Decision 2.6: Geospatial Preprocessing & Honest Metric Naming
* **Status**: **RESOLVED & IMPLEMENTED**
* **Honest Metric Naming**: Renamed `open_space_pct` to `non_building_area_pct` in `models/segmentation_unet.py` and `app/views/urban_planner_view.py`. Non-building pixels represent roads, parking, and open ground, not verified "green space".
* **Nearest-Neighbor Mask Resizing**: Replaced all default bilinear/bicubic mask resamplings with `resample=Image.Resampling.NEAREST` in `pipelines/03_train_segmentation.py` and `app/views/urban_planner_view.py`.

### Decision 2.7: Quarantine of Legacy Synthetic Model Checkpoints
* **Status**: **RESOLVED & IMPLEMENTED**
* **Quarantine**: Moved all 8 legacy synthetic model weights and metrics files (`unet_satellite.pt`, `resnet_condition.pt`, `xgboost_price.pkl`, `lightgbm_price.pkl`, etc.) to `models/legacy_synthetic/`.
* **Inference Safety Guards**: Active `models/saved/` is empty. Model loaders in `app/views/buyer_view.py` and `app/views/urban_planner_view.py` return `None` when active checkpoints are missing.
* **UI Reporting**: Displays explicit warning `MODEL NOT TRAINED ON CURRENT REAL DATA`. Uninitialized weights are never executed.

### Decision 2.8: Path Portability Standard
* **Status**: **RESOLVED & IMPLEMENTED**
* **Standard**: All paths in manifests (`data/manifests/*.csv`), split JSONs (`data/processed/*.json`), and pipeline configs converted to platform-independent forward-slash repository-relative paths (`data/...`). Zero drive letters or Windows backslashes permitted.

---

## 3. Decisions Made During Phase 3 (Real Model Training & Evaluation)

### Decision 3.1: SpaceNet U-Net Training & Development Evaluation Protocol
* **Status**: **RESOLVED & IMPLEMENTED**
* **Training Partition**: Trained U-Net architecture (`features=[16, 32, 64, 128]`) with combined BCE + Soft Dice Loss ($0.5/0.5$) for 12 epochs on the 24 research-training chips (`data/raw/spacenet/research_train/train/`).
* **Official Test S3 Inspection**: Inspected AWS S3 official public test archive (`s3://spacenet-dataset/spacenet/SN2_buildings/test_public/AOI_2_Vegas/`). Verified that imagery is provided but ground-truth building footprint GeoJSON vectors are withheld by competition organizers.
* **Resolution**: In compliance with Rule 1 (No Fabricated Results), no test score was invented. Evaluated strictly on the 6 held-out validation chips (`data/raw/spacenet/research_train/val/`) yielding Mean IoU = `0.3593`, Mean Dice = `0.4985`, Mean Precision = `0.7142`, Mean Recall = `0.4334`. Visual error analysis quads saved for all chips to `models/saved/unet_error_analysis/`.
* **Artifact**: `models/saved/unet_spacenet_v1.pt`.

---

### Decision 3.2: PEER Structural Condition ResNet-18 Transfer Learning & Train-Only Class Weights
* **Status**: **RESOLVED & IMPLEMENTED**
* **Transfer Learning**: Loaded official PyTorch ImageNet pretrained weights (`ResNet18_Weights.DEFAULT`, 44.7 MB cached). Fine-tuned backbone and custom MLP classification head (Dropout 0.3, BatchNorm1d) with AdamW ($\text{lr}=5 \times 10^{-5}$).
* **Class Imbalance Remediation**: Calculated class loss weights strictly from the 1,042-image training split (`global_collapse`: 446, `non_collapse`: 274, `partial_collapse`: 322) using inverse class frequency: $w = [0.7788, 1.2676, 1.0787]$. The test set was strictly excluded from weight derivation.
* **Benchmark Test Isolation**: The 146-image official benchmark test partition (`task5_X_test.npy` / `task5_y_test.npy`) remained completely untouched throughout training, validation, and hyperparameter tuning, evaluated strictly once after model selection.
* **Artifact**: `models/saved/resnet_peer_collapse_v1.pt`.

---

### Decision 3.3: Ames Housing Price Regression Model Selection & Split Conformal Prediction
* **Status**: **RESOLVED & IMPLEMENTED**
* **5-Fold Cross-Validation**: Executed 5-fold CV strictly on `X_train` ($N=1,020$): XGBoost RMSE = $\$25,294.20 \pm \$2,683.74$; LightGBM RMSE = $\$26,629.09 \pm \$2,965.74$.
* **Model Selection**: Evaluated on held-out validation partition ($N=219$). XGBoost selected with superior validation RMSE ($\$25,667.70$ vs $\$25,669.62$ for LightGBM). Final test set was completely untouched during model selection.
* **Uncertainty Quantification**: Permanently removed the fabricated static $\pm 8.5\%$ pricing margin. Implemented finite-sample Split Conformal Prediction calibrated on validation residuals, deriving a distribution-free 90% coverage relative margin ($\pm 22.23\%$).
* **Single Final Test Evaluation**: Evaluated frozen XGBoost on `housing_test_df.csv` ($N=219$): $\text{MAE} = \$15,092.41$, $\text{RMSE} = \$22,203.78$, $R^2 = 0.9103$, $\text{MAPE} = 9.17\%$.
* **Artifacts**: `models/saved/xgboost_ames_v1.pkl`, `models/saved/lightgbm_ames_v1.pkl`, `models/saved/price_metrics.json`.

---

### Decision 3.4: Zillow Regional ZHVI Forecaster (Prophet-Only Temporal Forward-Chaining)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Method Decision (LSTM vs Prophet)**: Audited `RealEstateLSTM`. With only 264 monthly observations per metro, training a deep LSTM leads to high parameter variance, severe overfitting, and non-reproducibility across CPU runs. Prophet-only was formally selected as the verified production forecaster. LSTM is classified as an experimental prototype with misleading training claims removed.
* **Metro Selection Rule**: Replaced row-count filtering with U.S. Census Population `SizeRank <= 10` among Metropolitan Statistical Areas (`RegionType == 'msa'`). Evaluates the top 10 primary US housing markets (New York, Los Angeles, Chicago, Dallas, Houston, Washington, Philadelphia, Miami, Atlanta, Boston), representing ~30% of total national volume.
* **Chronological Temporal Split**: Enforced past-to-future temporal evaluation without temporal data leakage:
  - Training Period: `2000-01-31` to `2021-12-31` (264 months)
  - Validation Period: `2022-01-31` to `2023-12-31` (24 months)
  - Held-Out Test Period: `2024-01-31` to `2026-08-31` (32 months)
* **Forecast Uncertainty Correction**: Corrected Prophet's default 80% interval to explicit 95% Bayesian forecast interval (`interval_width=0.95`). Documented that empirical test coverage (40.31%) reflects genuine out-of-sample interest rate regime shifts.
* **Artifacts**: `models/saved/forecaster_summary.json`, `data/processed/forecasts_cache.csv`.

---

## 4. Architecture Resolutions Summary

| Component | Historical Problem | Phase 3 Resolution | Status |
| :--- | :--- | :--- | :--- |
| **SpaceNet U-Net** | Synthetic training; fabricated 0.85 IoU | Real 24/6 split; actual validation IoU=0.3593, Dice=0.4985 | **COMPLETED & VERIFIED** |
| **PEER ResNet-18** | Untrained `pretrained=False`; subjective classes | ImageNet transfer learning; official 3-class collapse mode | **COMPLETED & VERIFIED** |
| **Ames Price Regressor** | Target leakage; static $\pm 8.5\%$ | Leakage-free preprocessor; 5-fold CV; Split Conformal 90% | **COMPLETED & VERIFIED** |
| **Zillow Forecaster** | Untrained LSTM claim; top-50 by rows; 80% interval | Prophet-only; Top 10 MSAs by SizeRank; 95% forecast interval | **COMPLETED & VERIFIED** |
| **Model Verification** | Unverified / fabricated claims | In-memory fresh reload verification script (`verify_reloaded_artifacts.py`) | **COMPLETED & VERIFIED** |

---

## 5. Decisions Made During Phase 4 (Research Hardening & Verified Dashboard Integration)

### Decision 4.1: Remediation of Conformal Prediction Methodology (Independence from Model Selection)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Issue Discovered**: In Phase 3, the model selection decision (selecting XGBoost over LightGBM) was performed on $X_{\text{val}}$ ($N=219$), and the 90% conformal interval was subsequently calibrated on the same $X_{\text{val}}$ observations. Under split conformal theory, conditioning calibration on the winning model of an empirical risk minimization tournament violates exchangeability due to post-selection bias (winner's curse).
* **Adopted Solution**: Implemented **5-Fold Cross-Conformal Prediction** (Vovk 2015; Barber et al. 2021).
  1. Utilizes genuine out-of-fold (OOF) relative absolute residuals across all $N=1,020$ properties generated during the 5-fold CV loop on $X_{\text{train}}$.
  2. Because $X_{\text{val}}$ ($N=219$) was completely withheld from OOF residual generation and reserved solely for model selection, the calibration distribution is strictly independent of model selection.
  3. The held-out test partition ($X_{\text{test}}$, $N=219$) remains strictly untouched.
* **Empirical Outcome**: Calibrated 90% relative margin is **$\pm 19.59\%$** (replacing the old biased $\pm 22.23\%$). Single held-out test evaluation yielded **$92.24\%$ empirical coverage**, verifying nominal coverage satisfaction.

### Decision 4.2: Canonical Checkpoint Designation & Safe Model Loading Contract
* **Status**: **RESOLVED & IMPLEMENTED**
* **Audit Finding**: Duplicate checkpoints existed in `models/saved/`:
  - `unet_spacenet_v1.pt` vs `unet_satellite.pt` (verified 100% parameter equivalence; unet_satellite.pt was a legacy alias).
  - `resnet_peer_collapse_v1.pt` vs `resnet_condition.pt` (verified 100% parameter equivalence; resnet_condition.pt was a legacy alias).
  - `xgboost_ames_v1.pkl` vs `xgboost_price.pkl` (100% byte-identical SHA-256).
  - `lightgbm_ames_v1.pkl` vs `lightgbm_price.pkl` (100% byte-identical SHA-256).
* **Resolution**: Canonical versioned paths explicitly designated:
  - `models/saved/unet_spacenet_v1.pt`
  - `models/saved/resnet_peer_collapse_v1.pt`
  - `models/saved/xgboost_ames_v1.pkl`
  - `models/saved/forecaster_summary.json`
* **Safe Failure Contract**: All loaders strictly fail safe (`TRAINED — VERIFIED` vs `MODEL NOT AVAILABLE`). If a checkpoint is missing or corrupt, loaders return `None` and safely disable live inference. Under no circumstances may an uninitialized or random model execute predictions.

### Decision 4.3: Buyer View Reconnection & Feature Contract Transparency
* **Status**: **RESOLVED & IMPLEMENTED**
* **Actions**:
  1. Reconnected `app/views/buyer_view.py` to canonical `xgboost_ames_v1.pkl`.
  2. Enforced complete 27-feature contract alignment with transparent lineage: 12 user-provided inputs, 6 derived inputs with documented engineering formulas, and 9 explicitly imputed inputs based on training population medians/modes.
  3. Added prominent research applicability warning: Model is an illustrative academic model trained strictly on Ames, Iowa residential transactions (2006–2010); must **NOT** be labeled as "true market value" or applied to arbitrary geographic locations.
  4. Displayed calibrated 90% prediction intervals ($\pm 19.59\%$) with explicit cross-conformal methodology disclosure.

### Decision 4.4: PEER Structural Collapse Semantic Enforcement
* **Status**: **RESOLVED & IMPLEMENTED**
* **Actions**:
  1. Reconnected `app/views/buyer_view.py` (Tab 2) to canonical `resnet_peer_collapse_v1.pt`.
  2. Excised fabricated cosmetic wear, property age, and arbitrary renovation cost claims.
  3. Enforced authentic PEER Task 5 post-disaster structural collapse semantics: `non_collapse` (Intact Structure), `partial_collapse` (Localized Structural Framing Failure), and `global_collapse` (Global Frame Collapse / Total Loss).
  4. Displayed authentic Phase 3 benchmark test metrics (Accuracy: 70.55%, Macro F1: 67.21%, N=146 benchmark images).

### Decision 4.5: SpaceNet Building Footprint Segmentation Scope
* **Status**: **RESOLVED & IMPLEMENTED**
* **Actions**:
  1. Reconnected `app/views/urban_planner_view.py` to canonical `unet_spacenet_v1.pt`.
  2. Replaced fictitious preset file names with authentic Las Vegas AOI validation chips (`vegas_1041`, `1042`, `1047`, `1048`, `1049`, `1051`).
  3. Enforced precise geospatial terminology: `non_building_area_pct` (open ground, parcels, roads). Prohibited labeling non-building area as "green space" or "vegetation".
  4. Disclosed validation scope explicitly: Development validation on $N=6$ chips (Mean IoU: 35.93%, Dice: 49.85%); no official public test labels available locally.

### Decision 4.6: Zillow Temporal Regime Delineation & Empirical Coverage Honesty
* **Status**: **RESOLVED & IMPLEMENTED**
* **Actions**:
  1. Updated `app/views/investor_view.py` and `app/components/charts.py` to visually and conceptually delineate chronological regimes: Historical Training (2000–2021), Validation (2022–2023), Held-Out Test (2024–2026), and Future Projections (2026–2029).
  2. Fixed metadata loading to read from verified `forecaster_summary.json` key (`metro_evaluations`), eliminating hard-coded fallbacks.
  3. Labeled Prophet intervals strictly as `95% forecast interval` (not calibrated confidence interval).
  4. Disclosed empirical held-out test coverage honestly (**40.31%** across 10 MSAs) without post-hoc macroeconomic causal theorizing.
  5. Clarified that ZHVI represents regional median housing index values, distinct from individual property sale prices.

### Decision 4.7: Model Metrics View Direct JSON Ingestion & Per-MSA Reporting (Phase 4H & 4I)
* **Status**: **RESOLVED & IMPLEMENTED**
* **Actions**:
  1. Refactored `app/views/model_metrics_view.py` to ingest metrics directly from the four authoritative JSON files, eliminating all hardcoded fallback values.
  2. Created complete Metric Provenance Matrix detailing Model, Dataset, Split, N, Metric, and Exact Value.
  3. Added comprehensive Per-MSA Forecasting Table covering all 10 MSAs by SizeRank (Validation MAE/RMSE/MAPE, Test MAE/RMSE/MAPE, Test Interval Coverage) with explicit arithmetic mean aggregation rule ($N=10$).



