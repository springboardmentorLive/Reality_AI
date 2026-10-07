# RealtyAI Phase 0: Complete Repository Audit, Baseline Verification & Defect Catalog

**Project Repository**: `RealtyAI2` / `realityai`  
**Execution Phase**: PHASE 0 ONLY (Inspection & Baseline Audit)  
**Audit Date**: October 2, 2026  
**Auditor**: Antigravity AI Pair Programmer (DeepMind Advanced Agentic Coding)  

---

## Table of Contents
1. [Executive Audit Summary](#1-executive-audit-summary)
2. [Repository Baseline & Git Status](#2-repository-baseline--git-status)
3. [Environment & Package Baseline](#3-environment--package-baseline)
4. [Mentor Specification Requirements Matrix](#4-mentor-specification-requirements-matrix)
5. [Complete Dataset Inventory & Provenance Audit](#5-complete-dataset-inventory--provenance-audit)
6. [Synthetic-Data Audit](#6-synthetic-data-audit)
7. [SpaceNet Satellite Segmentation Audit](#7-spacenet-satellite-segmentation-audit)
8. [Property-Condition Classification Audit](#8-property-condition-classification-audit)
9. [Housing Price Prediction Pipeline Audit](#9-housing-price-prediction-pipeline-audit)
10. [Price Uncertainty & Confidence Interval Audit](#10-price-uncertainty--confidence-interval-audit)
11. [Zillow Market Trend Forecasting Audit](#11-zillow-market-trend-forecasting-audit)
12. [Green-Space, Open-Space & Zoning Semantics Audit](#12-green-space-open-space--zoning-semantics-audit)
13. [Geospatial Architecture Audit](#13-geospatial-architecture-audit)
14. [Data Leakage Audit](#14-data-leakage-audit)
15. [Hardcoded Results & Dashboard Fallbacks Audit](#15-hardcoded-results--dashboard-fallbacks-audit)
16. [Model Checkpoints & Persistence Audit](#16-model-checkpoints--persistence-audit)
17. [Interactive Dashboard (Streamlit) Audit](#17-interactive-dashboard-streamlit-audit)
18. [Documentation vs. Implementation Contradiction Matrix](#18-documentation-vs-implementation-contradiction-matrix)
19. [Automated Test Suite Audit](#19-automated-test-suite-audit)
20. [Dependency Audit](#20-dependency-audit)
21. [Portability & Operating System Audit](#21-portability--operating-system-audit)
22. [Dead & Unused Code Audit](#22-dead--unused-code-audit)
23. [Comprehensive Defect Catalog (Critical, Medium, Minor)](#23-comprehensive-defect-catalog)
24. [Unverified Items](#24-unverified-items)

---

## 1. Executive Audit Summary

The **RealtyAI** repository presents an end-to-end multi-modal machine learning platform combining satellite imagery segmentation, property condition classification, tabular price regression, and time-series appreciation forecasting. 

A rigorous, evidence-based audit of all source code, datasets, serialized checkpoints, tests, and documentation reveals:
1. **Real Datasets**: The tabular housing dataset (`data/raw/housing/train.csv`, 1,460 rows, 81 columns) and the regional Zillow housing time series (`data/raw/zillow/Metro_zhvi_month.csv`, 895 metros, 320 date snapshots) are genuine datasets.
2. **Synthetic Datasets**: The SpaceNet satellite imagery (`data/raw/spacenet/images`, 120 files) and paired ground truth masks (`data/raw/spacenet/masks`, 120 files), as well as the property condition photographs (`data/raw/property_conditions`, 180 files), are **100% synthetic geometric drawings** procedurally generated using the Python `PIL.ImageDraw` module. No actual satellite or real house inspection photographs exist in the repository.
3. **Data Leakage**: Preprocessing leakage occurs in `pipelines/02_data_preprocessing.py` where median imputation is fitted over the entire dataset prior to splitting. Test-set model selection leakage occurs in `pipelines/05_train_price_regressor.py` where the holdout test set is evaluated to select between XGBoost and LightGBM.
4. **Methodological Deficiencies**: Price prediction confidence intervals are hardcoded to a static $\pm 8.5\%$ multiplier ($[0.915 \times \hat{y}, 1.085 \times \hat{y}]$). Transfer learning claims are contradicted by code explicitly passing `pretrained=False`. The claimed LSTM neural sequence forecaster is an unreferenced 15-line class that was never trained or evaluated.
5. **Portability Issues**: Multiple manifests and metadata files hardcode user-specific Windows absolute paths (`C:\Users\LENOVO\Desktop\realityai\...`), rendering the codebase non-portable to other machines.
6. **Execution State**: All existing automated unit tests (11/11) pass, and the Streamlit dashboard executes without crashing; however, unit tests evaluate dummy tensors rather than verifying real weights, and the dashboard calculation button does not actually control execution.

---

## 2. Repository Baseline & Git Status

* **Repository Path**: `c:\Users\LENOVO\Desktop\realityai`
* **Git Repository Initialized**: **NO (`.git` directory is absent)**
* **Git Status Command**: `git status`
* **Exit Code**: 1
* **Console Output**:
  ```text
  fatal: not a git repository (or any of the parent directories): .git
  ```
* **Git Log Command**: `git log -n 5 --oneline`
* **Exit Code**: 1
* **Console Output**:
  ```text
  fatal: not a git repository (or any of the parent directories): .git
  ```
* **Current Branch**: `UNKNOWN` (Git not initialized)
* **Working Tree Status**: `NOT TRACKED`
* **Uncommitted Changes**: Cannot be determined via Git; all files are untracked local filesystem entries.
* **Evidence**: Filesystem scan of repository root contains no `.git` subdirectory.

---

## 3. Environment & Package Baseline

The active execution environment was inspected directly via Python runtime introspection without modifying or reinstalling packages.

| Package / Tool | Version / Status | Verified Source |
| :--- | :--- | :--- |
| **Operating System** | `Windows 11 (10.0.26300-SP0, 64-bit AMD64)` | `platform.platform()` |
| **Python** | `3.13.14 (tags/v3.13.14:fd17997)` | `sys.version` |
| **PyTorch (`torch`)** | `2.9.1+cpu` | `torch.__version__` |
| **TorchVision (`torchvision`)** | `0.24.1+cpu` | `torchvision.__version__` |
| **NumPy (`numpy`)** | `2.2.6` | `numpy.__version__` |
| **Pandas (`pandas`)** | `2.3.3` | `pandas.__version__` |
| **Scikit-Learn (`sklearn`)** | `1.8.0` | `sklearn.__version__` |
| **XGBoost (`xgboost`)** | `3.1.3` | `xgboost.__version__` |
| **LightGBM (`lightgbm`)** | `4.7.0` | `lightgbm.__version__` |
| **Prophet (`prophet`)** | `1.4.0` | `prophet.__version__` |
| **Streamlit (`streamlit`)** | `1.52.2` | `streamlit.__version__` |
| **ReportLab (`reportlab`)** | `5.0.1` | `reportlab.__version__` |
| **OpenCV (`cv2`)** | `4.13.0` | `cv2.__version__` |
| **Ultralytics (`ultralytics`)** | **`NOT INSTALLED`** | `ImportError` |
| **Pytest (`pytest`)** | `9.0.2` | `pytest.__version__` |
| **Plotly (`plotly`)** | `6.5.0` | `plotly.__version__` |
| **Folium (`folium`)** | `0.20.0` | `folium.__version__` |
| **GeoPandas (`geopandas`)** | `1.1.4` | `geopandas.__version__` |
| **Shapely (`shapely`)** | `2.1.2` | `shapely.__version__` |
| **Rasterio (`rasterio`)** | `1.5.0` | `rasterio.__version__` |
| **Pillow (`PIL`)** | `12.0.0` | `PIL.__version__` |

---

## 4. Mentor Specification Requirements Matrix

Derived from the formal project requirements tracked across `docs/MILESTONES.md`, `docs/ARCHITECTURE.md`, and `docs/REPORT.md`:

| Mentor Requirement | Current Implementation | Evidence (File & Line) | Status | Identified Gap / Defect |
| :--- | :--- | :--- | :--- | :--- |
| **1. Kaggle Housing Ingestion & Cleaning** | Ingested Ames Iowa dataset; median/modal imputation; living area outlier removal. | `pipelines/01_data_ingestion.py:35-109`, `pipelines/02_data_preprocessing.py:31-120` | **IMPLEMENTED BUT LEAKED** | Imputation occurs prior to splitting; synthetic fallback data generator exists in code. |
| **2. Zillow ZHVI Time-Series Acquisition** | Ingested official Zillow monthly CSV (4.4 MB); melted to long format; computed historical appreciation. | `pipelines/01_data_ingestion.py:111-181`, `pipelines/02_data_preprocessing.py:122-176` | **IMPLEMENTED** | Synthetic data generator fallback exists in code; metro filtering heuristic ignores `SizeRank`. |
| **3. SpaceNet Satellite Imagery Tiles & Footprint Masks** | Generated 120 256×256 RGB tiles and binary footprint masks. | `pipelines/01_data_ingestion.py:183-266` | **IMPLEMENTED BUT INCORRECT (SYNTHETIC)** | 100% synthetic PIL box drawings. No genuine SpaceNet satellite imagery or geospatial annotations. |
| **4. Multi-Class Property Condition Photos** | Generated 180 224×224 photos across `New`, `Moderate`, `Old`. | `pipelines/01_data_ingestion.py:269-362` | **IMPLEMENTED BUT INCORRECT (SYNTHETIC)** | 100% synthetic PIL cartoon drawings. No real property photos. |
| **5. U-Net Satellite Footprint Segmentation** | PyTorch U-Net with contracting/expansive paths and `BCEDiceLoss`. | `models/segmentation_unet.py:34-103`, `pipelines/03_train_segmentation.py:71-180` | **IMPLEMENTED** | Model trains exclusively on synthetic PIL boxes; reports artificial 99.13% IoU. |
| **6. Urban Zoning & Green Space Analytics** | Computes footprint coverage %, non-building area, and density tier. | `models/segmentation_unet.py:124-150`, `app/views/urban_planner_view.py:99-148` | **IMPLEMENTED BUT INCORRECT** | Equates `100 - building_density` to green/pervious space (includes roads); zoning tiers are arbitrary heuristics. |
| **7. ResNet-18 Property Condition Classifier** | ResNet-18 with custom MLP classifier head and dropout. | `models/condition_resnet.py:22-44`, `pipelines/04_train_condition_cnn.py:52-155` | **IMPLEMENTED BUT INCORRECT** | Explicitly trained with `pretrained=False`. Transfer learning claim is false. |
| **8. Condition Scoring & Renovation Multiplier** | Formula mapping class probabilities into 0–100 score and renovation cost brackets. | `models/condition_resnet.py:65-97` | **IMPLEMENTED** | Softmax maximum probability is falsely labeled "calibrated confidence" in dashboard. |
| **9. XGBoost & LightGBM Regressors** | Trained on $\log(1 + \text{SalePrice})$; feature importance extracted. | `models/price_regressor.py:20-90`, `pipelines/05_train_price_regressor.py:28-88` | **IMPLEMENTED BUT LEAKED** | Best model selection is decided on the holdout test set RMSE. |
| **10. Valuation Confidence Intervals** | 90% confidence interval around point prediction. | `models/price_regressor.py:110-118`, `app/views/buyer_view.py:111-138` | **IMPLEMENTED BUT INCORRECT** | Hardcoded static multiplier $[0.915 \times \hat{y}, 1.085 \times \hat{y}]$ ($\pm 8.5\%$). Not statistically derived. |
| **11. Facebook Prophet Time-Series Forecaster** | Additive model with yearly seasonality and changepoint detection; 24-mo holdout evaluation. | `models/trend_forecaster.py:20-91`, `pipelines/06_train_forecaster.py:28-97` | **IMPLEMENTED** | Default Prophet credible interval is 80%, contradicted by 95% claims in documentation. |
| **12. PyTorch LSTM Sequence Forecaster** | Neural sequence architecture for multi-step projection. | `models/trend_forecaster.py:93-108` | **NOT IMPLEMENTED (DEAD CODE)** | `RealEstateLSTM` class is defined but never trained, evaluated, saved, or imported. |
| **13. Investor Intelligence Metrics** | 3-year projected appreciation, CAGR, volatility rating, opportunity tiers. | `models/trend_forecaster.py:110-160` | **IMPLEMENTED** | Fully integrated in `app/views/investor_view.py`. |
| **14. Unified Evaluation Pipeline** | Aggregates all model metrics into consolidated JSON for dashboard. | `pipelines/07_evaluate_all.py:25-82` | **IMPLEMENTED** | Generates `data/processed/master_evaluation_metrics.json`. |
| **15. Multi-Persona Streamlit Dashboard** | Home Buyer, Investor, Urban Planner, and Model Diagnostics personas. | `app/main.py`, `app/views/*.py` | **IMPLEMENTED (WITH DEFECTS)** | Calculate button does not gate execution; 15 hidden features fabricated; hardcoded version text. |
| **16. Executive PDF Report Generator** | Multi-page report with tables, flowables, charts, and metrics. | `generate_pdf_report.py:1-710` | **IMPLEMENTED** | Generates two 326 KB PDF reports; contains hardcoded feature importances. |
| **17. Automated Unit Test Suite** | Pytest test suite covering data pipeline, deep learning models, and UI imports. | `tests/test_*.py` | **PARTIALLY IMPLEMENTED** | 11 unit tests pass, but models are tested with random dummy tensors; no tests for LightGBM or forecaster. |

---

## 5. Complete Dataset Inventory & Provenance Audit

| Dataset Path | File Type | Records / Count | Schema / Shape | Date Range | Ground Truth Labels | Real or Synthetic | Verified Provenance / Source | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `data/raw/housing/train.csv` | CSV | 1,460 rows | 81 columns | 2006–2010 | `SalePrice` | **REAL** | Kaggle Ames Housing Dataset (De Cock, 2011) | `VERIFIED` |
| `data/raw/housing/test.csv` | CSV | 1,459 rows | 80 columns | 2006–2010 | None (unlabeled) | **REAL** | Kaggle Ames Housing Test Split | `VERIFIED` |
| `data/raw/housing/data_description.txt` | TXT | 518 lines | Text description | N/A | Feature dictionary | **REAL** | Ames Assessor's Office reference | `VERIFIED` |
| `data/raw/zillow/Metro_zhvi_month.csv` | CSV | 895 rows | 325 columns | 2000-01 to 2026-08 | Monthly median ZHVI dollar values | **REAL** | Zillow Research Public Data Portal | `VERIFIED` |
| `data/raw/spacenet/images/` | PNG | 120 images | $256 \times 256 \times 3$ RGB | N/A | Synthetic visual shapes | **SYNTHETIC** | Procedurally drawn via `PIL.ImageDraw` in `pipelines/01_data_ingestion.py` | `VERIFIED BROKEN` |
| `data/raw/spacenet/masks/` | PNG | 120 masks | $256 \times 256 \times 1$ Grayscale | N/A | Binary {0, 255} box footprints | **SYNTHETIC** | Procedurally drawn via `PIL.ImageDraw` in `pipelines/01_data_ingestion.py` | `VERIFIED BROKEN` |
| `data/raw/property_conditions/new/` | JPG | 60 images | $224 \times 224 \times 3$ RGB | N/A | Label 0: `new` | **SYNTHETIC** | Procedurally drawn cartoon houses in `pipelines/01_data_ingestion.py` | `VERIFIED BROKEN` |
| `data/raw/property_conditions/moderate/` | JPG | 60 images | $224 \times 224 \times 3$ RGB | N/A | Label 1: `moderate` | **SYNTHETIC** | Procedurally drawn cartoon houses in `pipelines/01_data_ingestion.py` | `VERIFIED BROKEN` |
| `data/raw/property_conditions/old/` | JPG | 60 images | $224 \times 224 \times 3$ RGB | N/A | Label 2: `old` | **SYNTHETIC** | Procedurally drawn cartoon houses in `pipelines/01_data_ingestion.py` | `VERIFIED BROKEN` |
| `data/raw/property_condition_metadata.csv` | CSV | 180 rows | `filename`, `filepath`, `condition_class`, `label` | N/A | 3 classes (60 each) | **METADATA** | Generated in `pipelines/01_data_ingestion.py`; contains absolute Windows paths | `VERIFIED BROKEN` |
| `data/sample_images/` | JPG | 12 images | $224 \times 224 \times 3$ RGB | N/A | 4 new, 4 moderate, 4 old | **SYNTHETIC** | Cloned copies of synthetic condition images for UI demo | `VERIFIED BROKEN` |
| `data/processed/zillow_zhvi_processed.csv` | CSV | 11,156 rows | 7 columns | 2000-01 to 2026-08 | `ZHVI` numeric index | **DERIVED** | Derived from real Zillow CSV via unpivot melt | `VERIFIED` |
| `data/processed/zillow_metro_summary.csv` | CSV | 35 rows | 8 columns | 2000-01 to 2026-08 | Aggregate growth metrics | **DERIVED** | Computed historical CAGR/YoY per metro | `VERIFIED` |
| `data/processed/forecasts_cache.csv` | CSV | 288 rows | 5 columns | 2026-09 to 2029-08 | `Forecast_ZHVI`, lower, upper | **GENERATED** | Generated by Prophet forecaster in `pipelines/06_train_forecaster.py` | `VERIFIED` |
| `data/processed/housing_X_train.npy` | NPY | 1,029 rows | 27 features | N/A | Transformed numerical/categorical | **DERIVED** | Output of `ColumnTransformer` (contains leakage) | `VERIFIED` |
| `data/processed/housing_X_val.npy` | NPY | 220 rows | 27 features | N/A | Transformed numerical/categorical | **DERIVED** | Output of `ColumnTransformer` (contains leakage) | `VERIFIED` |
| `data/processed/housing_X_test.npy` | NPY | 219 rows | 27 features | N/A | Transformed numerical/categorical | **DERIVED** | Output of `ColumnTransformer` (contains leakage) | `VERIFIED` |
| `data/processed/spacenet_splits.json` | JSON | 120 items | `train` (90), `val` (18), `test` (12) | N/A | Relative paths to synthetic PNGs | **DERIVED** | Split manifest from `pipelines/02_data_preprocessing.py` | `VERIFIED` |
| `data/processed/property_condition_splits.json` | JSON | 180 items | `train` (126), `val` (27), `test` (27) | N/A | Dicts with absolute Windows paths | **DERIVED** | Split manifest from `pipelines/02_data_preprocessing.py` | `VERIFIED BROKEN` |

---

## 6. Synthetic-Data Audit

Every procedural synthetic generator in the codebase was located, isolated, and analyzed:

### 1. Synthetic SpaceNet Imagery & Mask Generator
* **File**: `pipelines/01_data_ingestion.py`
* **Function**: `generate_spacenet_satellite_data(num_tiles=120)`
* **Lines**: 183–266
* **What it generates**:
  * Creates an RGB image with green grass background (`Image.new("RGB", (256, 256), color=(45, 90, 45))`).
  * Draws asphalt road ribbons (`draw_img.rectangle([0, y_road, 256, y_road + road_width])`) with yellow dashed center lines.
  * Draws 6–16 rectangular building footprints with roof palette colors (terracotta, light gray, slate, beige) and offset shadow rectangles.
  * Draws binary masks (`draw_mask.rectangle([bx, by, bx + bw, by + bh], fill=255)`).
  * Adds Gaussian blur (`radius=0.6`) and normal noise.
* **Where it is called**: `pipelines/01_data_ingestion.py` line 368 inside `run_ingestion()`.
* **Enters Production/Research Pipeline?**: **YES**. This synthetic data directly feeds `pipelines/02_data_preprocessing.py`, `pipelines/03_train_segmentation.py`, `models/saved/unet_satellite.pt`, and `app/views/urban_planner_view.py`.

### 2. Synthetic Property Condition Image Generator
* **File**: `pipelines/01_data_ingestion.py`
* **Function**: `generate_property_condition_dataset(samples_per_class=60)`
* **Lines**: 269–362
* **What it generates**:
  * 180 $224 \times 224$ images across `new`, `moderate`, and `old`.
  * Draws sky rectangle (`140, 180, 230`), lawn rectangle, house body rectangle (`40, 75, 144, 90`), triangular gable roof polygon, doors, and windows.
  * Adds condition features: old homes get drawn crack lines and green ellipse moss patches; new homes get white trim lines and walkways.
* **Where it is called**: `pipelines/01_data_ingestion.py` line 369 inside `run_ingestion()`.
* **Enters Production/Research Pipeline?**: **YES**. Directly feeds `pipelines/02_data_preprocessing.py`, `pipelines/04_train_condition_cnn.py`, `models/saved/resnet_condition.pt`, and `app/views/buyer_view.py`.

### 3. Fallback Synthetic Kaggle Housing Generator
* **File**: `pipelines/01_data_ingestion.py`
* **Function**: `ingest_kaggle_housing()`
* **Lines**: 57–105
* **What it generates**: Procedural DataFrame of 1,460 rows using `np.random` with synthetic features (`MSSubClass`, `LotArea`, `YearBuilt`, `GrLivArea`, `SalePrice`) computed from a hardcoded linear formula with Gaussian noise.
* **Where it is called**: `pipelines/01_data_ingestion.py` line 366 inside `run_ingestion()`.
* **Enters Production/Research Pipeline?**: **NO (STANDBY FALLBACK)**. Because `data/raw/housing/train.csv` already exists with real Kaggle data, this fallback block was bypassed during the latest ingestion run. However, it remains live in code.

### 4. Fallback Synthetic Zillow Time-Series Generator
* **File**: `pipelines/01_data_ingestion.py`
* **Function**: `ingest_zillow_data()`
* **Lines**: 130–180
* **What it generates**: Procedural time series for 15 metros across monthly dates from 2000-01 to 2026-06 with hardcoded annual growth, volatility, simulated 2008 crash cycle (-0.004), and 2020 pandemic boom (+0.012).
* **Where it is called**: `pipelines/01_data_ingestion.py` line 367 inside `run_ingestion()`.
* **Enters Production/Research Pipeline?**: **NO (STANDBY FALLBACK)**. The live download succeeded, saving the authentic 4.47 MB Zillow dataset to `data/raw/zillow/Metro_zhvi_month.csv`. The generator remains present as fallback.

### 5. Fabricated Hidden Feature Generator in Dashboard
* **File**: `app/views/buyer_view.py`
* **Function**: `render_buyer_view`
* **Lines**: 90–110
* **What it generates**: Synthesizes 15 unentered model features via ad-hoc formulas (`1stFlrSF = gr_liv_area * 0.6`, `TotRmsAbvGrd = bedrooms + full_bath + 3`, `WoodDeckSF = 120`, `OpenPorchSF = 60`, `MSZoning = "RL"`, `HouseStyle = "2Story" if gr_liv_area > 1500 else "1Story"`, etc.).
* **Enters Production/Research Pipeline?**: **YES**. Used for live inference whenever a user tests property valuation in the Streamlit UI.

---

## 7. SpaceNet Satellite Segmentation Audit

1. **Existence of Real SpaceNet Files**: **NONE**. The directory `data/raw/spacenet/` contains only procedural PNG drawings generated by `generate_spacenet_satellite_data()`.
2. **Polygons and Rasterization**: **NONE**. SpaceNet provides vector GeoJSON building footprint polygons associated with WorldView-2/3 8-band or RGB satellite imagery. In this repository, there is no GeoJSON parsing, no shapely polygon geometry, and no rasterization via `rasterio.features.rasterize`. Rectangles are drawn directly as pixel grids using PIL.
3. **Coordinate Reference System (CRS) & Georeferencing**: **NONE**. Images are 8-bit standard RGB PNGs lacking affine geotransform matrices, ground control points (GCPs), or EPSG projections.
4. **Spatial Leakage Across Splits**: The dataset split in `pipelines/02_data_preprocessing.py` (lines 195–205) shuffles the 120 image pairs randomly via `np.random.shuffle(pairs)`. Because the tiles are procedural and non-geographic, geographic distance splitting was not performed.
5. **Segmentation Target Semantics**: The model segments rectangular roof drawings. However, the application claims in `app/views/urban_planner_view.py` (lines 128–134) that any pixel not classified as building footprint is "Open / Green Space" and "Pervious Area". Roads (drawn with grey pixel ribbons) are systematically counted as green open space.
6. **False Human Annotation Claim**: In `app/views/urban_planner_view.py` line 194, the ground-truth mask is labeled:
   `caption="Ground Truth SpaceNet Mask (Human Annotated)"`
   This statement is **completely fabricated**. The mask was generated via `draw_mask.rectangle(...)`.

---

## 8. Property-Condition Classification Audit

1. **Dataset Authenticity**: **100% SYNTHETIC**. Consists of 180 cartoon drawings created via PIL (`new_000.jpg` to `old_059.jpg`).
2. **Transfer Learning Claim Verification**:
   * **Claim in Documentation**: `README.md` (line 18), `docs/ARCHITECTURE.md` (line 27), `docs/REPORT.md` (line 61) claim "ResNet-18 Transfer Learning" and "transfer-learned ResNet-18 model".
   * **Actual Code**: In `pipelines/04_train_condition_cnn.py` line 68:
     ```python
     model = PropertyConditionClassifier(num_classes=3, pretrained=False).to(device)
     ```
   * **Verdict**: **Documentation claim not supported by implementation.** Pretrained ImageNet weights were never downloaded or loaded. The model was trained entirely from scratch on synthetic color blocks.
3. **Test Accuracy**:
   * Evaluated on 27 synthetic test images in `models/saved/condition_metrics.json`:
     `test_accuracy: 1.0 (100.0%)`, `test_f1_score: 1.0 (100.0%)`.
   * Contradicted in `docs/REPORT.md` (line 64) and `docs/ARCHITECTURE.md` (line 34), which claim `Accuracy: 92.6%` and `F1: 92.4%`.
4. **Softmax "Confidence" Mislabeling**:
   * In `app/views/buyer_view.py` line 217:
     `Confidence: <b>{probs[pred_idx]*100:.1f}%</b>`
   * Softmax maximum probability is presented directly to the user as "Confidence" without Platt scaling, temperature scaling, or isotonic calibration.
5. **Renovation Cost Heuristics**: Condition scores and renovation multipliers in `models/condition_resnet.py` (lines 65–97) are fixed hardcoded formulas ($95 \cdot P_{\text{new}} + 72 \cdot P_{\text{mod}} + 35 \cdot P_{\text{old}}$) without empirical cost data backing.

---

## 9. Housing Price Prediction Pipeline Audit

1. **Data Ingestion**: Real Ames Iowa dataset ingested with 1,460 transactions and 81 raw attributes. Outliers with $\text{GrLivArea} > 4000$ and $\text{SalePrice} < \$300,000$ are appropriately filtered.
2. **Preprocessing Leakage Before Split**:
   * In `pipelines/02_data_preprocessing.py` lines 66–74:
     ```python
     # Impute missing values
     for col in num_features:
         X[col] = X[col].fillna(X[col].median())
     for col in cat_features:
         X[col] = X[col].fillna("Missing").astype(str)

     # Train / Val / Test Split (70 / 15 / 15)
     X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
     ```
   * Computing `X[col].median()` on the entire dataset `X` leaks test and validation statistics directly into training inputs.
3. **Test-Set Model Selection Leakage**:
   * In `pipelines/05_train_price_regressor.py` lines 46, 53, and 73:
     ```python
     xgb_metrics = xgb_predictor.evaluate(X_test, y_test)
     lgb_metrics = lgb_predictor.evaluate(X_test, y_test)
     best_model_name = "XGBoost" if xgb_metrics["rmse"] <= lgb_metrics["rmse"] else "LightGBM"
     ```
   * The test set was used to select the production model, invalidating its role as an unbiased holdout.
4. **Target Transformation**: Correctly applies $\log(1 + y)$ transformation in `models/price_regressor.py` lines 28–29 and inverts via $\exp(\hat{y}) - 1$ at line 74.
5. **Feature Importance Discrepancy**:
   * Actual XGBoost artifact (`models/saved/price_metrics.json`): OverallQual (30.88%), ExterQual (11.57%), GarageCars (9.69%), GrLivArea (6.06%), BsmtQual (5.41%).
   * Documentation claims (`docs/REPORT.md` line 79): Overall Quality (28.4%), Living Area Sq Ft (19.2%), Total Basement Area (11.8%), Garage Capacity (9.4%), Year Built (7.1%).
   * The documentation feature importance values are fabricated/stale and do not match the serialized model.

---

## 10. Price Uncertainty & Confidence Interval Audit

| Feature | Audit Finding | Evidence (File & Line) | Status |
| :--- | :--- | :--- | :--- |
| **Claimed Confidence Interval** | "90% confidence interval estimation based on average model error" | `models/price_regressor.py:110`, `README.md:17`, `docs/ARCHITECTURE.md:101` | `DOCUMENTATION ONLY` |
| **Actual Implementation** | Fixed multiplication: $[0.915 \times \hat{y}, 1.085 \times \hat{y}]$ ($\pm 8.5\%$) | `models/price_regressor.py:111-112` | `IMPLEMENTED BUT INCORRECT` |
| **Hardcoded UI Confidence** | `<div class="metric-delta-pos">Model Confidence: 91.5%</div>` | `app/views/buyer_view.py:127` | `HARDCODED STRING` |
| **Statistical Derivation** | None. No residual standard error, quantile loss, conformal prediction, or bootstrap. | `models/price_regressor.py:91-120` | `NOT IMPLEMENTED` |

* **Analysis**: Multiplying any predicted price by 0.915 and 1.085 has no statistical validity. A property with extreme variance receives the exact same $\pm 8.5\%$ window as a standard suburban house.

---

## 11. Zillow Market Trend Forecasting Audit

1. **Dataset Integrity**: Ingested real Zillow ZHVI data (`data/raw/zillow/Metro_zhvi_month.csv`) covering 895 metros and 320 monthly snapshots (January 2000 through August 2026).
2. **Metro Filtering Heuristic**:
   * `pipelines/06_train_forecaster.py` lines 39–40:
     ```python
     metro_counts = df.groupby("RegionName")["Date"].count()
     top_metros = metro_counts.nlargest(top_n_metros).index.tolist()
     ```
   * Metros are filtered by row count rather than `SizeRank` or population.
3. **Temporal Validation**: Properly implemented in `models/trend_forecaster.py` lines 63–64 by holding out the final 24 months chronologically (`train_df = prophet_df.iloc[:-test_months]`, `test_df = prophet_df.iloc[-test_months:]`).
4. **Prophet Credible Interval Discrepancy**:
   * Documentation claims: "95% confidence intervals" (`docs/ARCHITECTURE.md` line 109, `docs/MILESTONES.md` line 54, `docs/REPORT.md` line 97).
   * Actual Code: `models/trend_forecaster.py` line 37 instantiates `Prophet(...)` without setting `interval_width`. Verified runtime default for Prophet is `0.8` (80% credible interval).
5. **Abandoned LSTM Sequence Forecaster**:
   * `models/trend_forecaster.py` defines `class RealEstateLSTM(nn.Module)` (lines 93–108).
   * The class is never imported, trained, saved, or called.
   * `docs/REPORT.md` line 17 and `docs/MILESTONES.md` line 52 falsely claim LSTM sequence modeling was executed and evaluated.

---

## 12. Green-Space, Open-Space & Zoning Semantics Audit

* **Calculation Formula**:
  * In `models/segmentation_unet.py` lines 131–146:
    ```python
    density_pct = float((building_pixels / total_pixels) * 100)
    open_space_pct = round(100.0 - density_pct, 2)
    ```
* **Dashboard Presentation**:
  * In `app/views/urban_planner_view.py` line 128:
    `<div class="metric-title">Open / Green Space</div>`
    `<div class="metric-value">{zone_info['open_space_pct']}%</div>`
    `<div class="metric-subtitle">Pervious Area</div>`
* **Audit Finding**: The calculation `100.0 - density_pct` assumes that **all non-building surfaces are pervious green space**. Asphalt roadways, parking lots, concrete sidewalks, and bare dirt are all categorized as "Green Space".
* **Zoning Claim**: Legal zoning categories ("Low Density Residential / Rural", "Medium Density Suburban Residential", "High Density Urban Commercial / Mixed") are assigned based purely on hardcoded thresholds ($<10\%$, $10-28\%$, $>28\%$) without genuine municipal zoning shapefiles or land use records.

---

## 13. Geospatial Architecture Audit

1. **Coordinate Handling**:
   * Metros are located using a hardcoded dictionary `METRO_COORDINATES` in `app/components/map_view.py` lines 10–29 containing 18 city coordinates.
   * Metros outside these 18 cities cannot be displayed on the map.
2. **GIS Libraries**:
   * `geopandas>=0.14.0` and `shapely>=2.0.0` are declared in `requirements.txt`.
   * Neither library is imported in any `.py` script in the entire repository.
3. **Coordinate Reference Systems & Projections**:
   * No GeoTIFF raster parsing (`rasterio` is never imported).
   * No GeoJSON, Shapefile, or GeoPackage vector layers exist.
   * SpaceNet images are plain pixel PNGs without geospatial headers.

---

## 14. Data Leakage Audit

| Leakage Category | Affected Files & Functions | Exact Line Reference | Observed Behavior | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **Preprocessing Leakage** | `pipelines/02_data_preprocessing.py`<br>`preprocess_housing_data` | Lines 66–74 | Computes `.fillna(X[col].median())` across the full dataset `X` before invoking `train_test_split`. | **CRITICAL** |
| **Test-Set Model Selection Leakage** | `pipelines/05_train_price_regressor.py`<br>`train_price_models` | Lines 46, 53, 73 | Evaluates both XGBoost and LightGBM on holdout test set `(X_test, y_test)` to choose `best_model_name`. | **CRITICAL** |
| **Spatial Tile Leakage** | `pipelines/02_data_preprocessing.py`<br>`prepare_vision_dataset_splits` | Lines 195–205 | Randomly shuffles SpaceNet image pairs across train, val, and test splits without spatial blocking. | **MEDIUM** |
| **Property ID Group Leakage** | `pipelines/02_data_preprocessing.py`<br>`prepare_vision_dataset_splits` | Lines 215–217 | Stratifies condition images strictly by class label without checking property identity. | **LOW (Synthetic Only)** |

---

## 15. Hardcoded Results & Dashboard Fallbacks Audit

1. **Hardcoded Feature Importances in PDF Generator**:
   * `generate_pdf_report.py` lines 80–81 hardcodes:
     `importances = [0.3088, 0.1157, 0.0969, 0.0606, 0.0541, 0.0536, 0.0380, 0.0321, 0.0291, 0.0237]`
2. **Hardcoded Confidence Percentage**:
   * `app/views/buyer_view.py` line 127: `<div class="metric-delta-pos">Model Confidence: 91.5%</div>`.
3. **Hardcoded Benchmark Fallbacks in Model Diagnostics Hub**:
   * `app/views/model_metrics_view.py`:
     * Line 59: `unet_m.get("test_mean_iou", 0.952)`
     * Line 74: `cond_m.get("test_accuracy", 0.926)`
     * Line 90: `xgb_m.get("mae", 15092)`
     * Line 105: `fcst_m.get("mean_mape_pct", 11.64)`
     * Line 132: `price_m.get("xgboost", {"mae": 15092.41, "rmse": 22203.78, "r2_score": 0.9103, "mape_pct": 9.17})`
4. **Hardcoded Package Versions in System Diagnostics**:
   * `app/views/model_metrics_view.py` lines 212–217 hardcodes string literals for Python, Streamlit, XGBoost, LightGBM, and Prophet versions.

---

## 16. Model Checkpoints & Persistence Audit

| Model Role | Architecture | Checkpoint File Path | File Size | Checkpoint Exists? | Matches Architecture? | Missing-Checkpoint Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Satellite Segmentation** | PyTorch U-Net (`features=[16, 32, 64, 128]`) | `models/saved/unet_satellite.pt` | 7.82 MB | **YES** | Matches | **CRITICAL DEFECT**: Instantiates random uninitialized weights and runs forward pass silently. |
| **Property Condition** | ResNet-18 + Custom MLP Head | `models/saved/resnet_condition.pt` | 45.05 MB | **YES** | Matches | **CRITICAL DEFECT**: Instantiates random uninitialized weights and runs forward pass silently. |
| **Price Regression (Best)** | XGBoost Regressor | `models/saved/xgboost_price.pkl` | 727.6 KB | **YES** | Matches | Returns `None` and displays `st.info` placeholder. |
| **Price Regression (Bench)** | LightGBM Regressor | `models/saved/lightgbm_price.pkl` | 351.5 KB | **YES** | Matches | Not loaded by dashboard buyer view (used only for diagnostics table). |
| **Trend Forecaster** | Facebook Prophet | `data/processed/forecasts_cache.csv` | 16.3 KB | **YES (CSV)** | Matches | Precomputed forecasts cached in CSV; fits Prophet dynamically per metro. |
| **Sequence Forecaster** | PyTorch LSTM | None | N/A | **NO** | N/A | Abandoned; no checkpoint exists. |

---

## 17. Interactive Dashboard (Streamlit) Audit

1. **Calculate Button Logic Flaw**:
   * In `app/views/buyer_view.py` line 78:
     `calc_btn = st.button("⚡ Calculate Predictive Valuation", use_container_width=True, type="primary")`
   * Line 81 proceeds unconditionally: `predictor = load_pricing_model()`.
   * The calculation and results display are not indented inside an `if calc_btn:` block. The button triggers an ordinary Streamlit rerun but does not gate computation.
2. **Fabricated Model Features**:
   * As documented in Section 6, 15 features are dynamically invented using hardcoded multipliers.
3. **Multi-Persona Navigation**:
   * Navigation bar rendered via `streamlit_option_menu` in `app/components/header.py`. Correctly routes between 4 views: Buyer, Investor, Urban Planner, Model Hub.

---

## 18. Documentation vs. Implementation Contradiction Matrix

| Component | Documentation Claim | Verified Code Reality | Exact Discrepancy Evidence |
| :--- | :--- | :--- | :--- |
| **ResNet-18 Weights** | "ResNet-18 Transfer Learning" | `pretrained=False` | `pipelines/04_train_condition_cnn.py:68` |
| **Condition Classification Accuracy** | 92.6% Accuracy, 92.4% F1 | 100.0% Accuracy, 100.0% F1 | `docs/REPORT.md:64` vs `models/saved/condition_metrics.json:8` |
| **U-Net Test IoU** | 98.7% IoU, 99.3% Dice | 99.13% IoU, 99.56% Dice | `docs/REPORT.md:57` vs `models/saved/unet_metrics.json:9-10` |
| **Prophet Forecast Uncertainty** | "95% confidence intervals" | Default 80% credible interval | `docs/REPORT.md:97` vs `Prophet().interval_width == 0.8` |
| **Valuation Uncertainty** | "90% confidence prediction interval" | Hardcoded static $\pm 8.5\%$ multiplier | `docs/ARCHITECTURE.md:101` vs `models/price_regressor.py:111` |
| **LSTM Model** | "Implemented LSTM neural sequence architecture" | Dead code: class defined, never trained or evaluated | `docs/MILESTONES.md:52` vs `models/trend_forecaster.py:93` |
| **XGBoost Feature Importance** | Overall Quality (28.4%), Living Area (19.2%) | OverallQual (30.88%), ExterQual (11.57%) | `docs/REPORT.md:79` vs `models/saved/price_metrics.json:17-33` |
| **SpaceNet Mask Provenance** | "Ground Truth SpaceNet Mask (Human Annotated)" | Drawn procedurally by `PIL.ImageDraw.rectangle` | `app/views/urban_planner_view.py:194` vs `pipelines/01_data_ingestion.py:244` |
| **Green Space Semantics** | "Green / Open Space: Pervious Area" | Non-building area ($100 - \text{density}$), includes asphalt roads | `app/views/urban_planner_view.py:128` vs `models/segmentation_unet.py:146` |

---

## 19. Automated Test Suite Audit

* **Test Execution Command**: `python -m pytest -v`
* **Test Outcome**: 11 passed in 17.81s
* **Test Analysis**:
  * `test_app_imports`: Smoke tests importing UI modules and verifying string lengths.
  * `test_housing_processed_arrays`: Checks non-null status and shapes of preprocessed `.npy` arrays.
  * `test_housing_feature_metadata`: Verifies `housing_features.json` exists.
  * `test_zillow_processed_time_series`: Verifies row count and columns of processed CSV.
  * `test_spacenet_splits`: Verifies length of split JSON.
  * `test_unet_architecture_forward`: Evaluates dummy tensor `torch.randn(2, 3, 256, 256)` on newly initialized U-Net.
  * `test_segmentation_metrics`: Asserts IoU/Dice calculation on synthetic tensors.
  * `test_satellite_zone_analysis`: Asserts zone analytics on a dummy numpy box.
  * `test_resnet_condition_classifier`: Evaluates dummy tensor `torch.randn(2, 3, 224, 224)` on newly initialized ResNet.
  * `test_inspection_report_logic`: Checks condition scoring on static dictionary.
  * `test_price_predictor_inference`: Runs live inference on `xgboost_price.pkl`.
* **Testing Gaps**:
  * No unit tests load `unet_satellite.pt` or `resnet_condition.pt`.
  * No unit tests evaluate `lightgbm_price.pkl`.
  * No unit tests evaluate `RegionalProphetForecaster` or `RealEstateLSTM`.
  * No tests check for preprocessing or temporal data leakage.

---

## 20. Dependency Audit

* **Declared in `requirements.txt` but Never Imported in Code**:
  * `geopandas>=0.14.0`
  * `shapely>=2.0.0`
  * `opencv-python>=4.8.0`
  * `seaborn>=0.13.0`
* **Imported in Code but Missing from `requirements.txt`**:
  * `reportlab` (imported and used in `generate_pdf_report.py`)
  * `joblib` (imported across 5 modules, though typically bundled with scikit-learn)
* **Required in Mentor Prompt but Not Installed**:
  * `ultralytics` (`NOT INSTALLED`)

---

## 21. Portability & Operating System Audit

* **Hardcoded Machine Absolute Paths**:
  * `data/raw/property_condition_metadata.csv` lines 2–181: hardcodes `C:\Users\LENOVO\Desktop\realityai\...`.
  * `data/processed/property_condition_splits.json` lines 4–1085: hardcodes `C:\Users\LENOVO\Desktop\realityai\...`.
  * `pipelines/01_data_ingestion.py` line 31: hardcodes `~\Downloads\house-prices-advanced-regression-techniques`.
  * `PROJECT_SUMMARY_AND_USER_GUIDE.md` line 274: hardcodes `cd c:\Users\LENOVO\Desktop\realityai`.
* **Windows Backslash Paths in Manifests**:
  * `data/processed/spacenet_splits.json` lines 4–487: uses Windows backslashes (`data\\raw\\spacenet\\images\\...`). On Linux/macOS, this fails to resolve file paths.

---

## 22. Dead & Unused Code Audit

1. **`RealEstateLSTM`**: `models/trend_forecaster.py` lines 93–108. Defined but never called anywhere.
2. **`DOWNLOADS_DIR` check**: `pipelines/01_data_ingestion.py` line 31.
3. **Unused Imports in `requirements.txt`**: `geopandas`, `shapely`, `cv2`, `seaborn`.

---

## 23. Comprehensive Defect Catalog

### CRITICAL DEFECTS

#### Defect CRIT-01: 100% Synthetic SpaceNet Satellite Dataset
* **Severity**: `CRITICAL`
* **File**: `pipelines/01_data_ingestion.py`
* **Function**: `generate_spacenet_satellite_data`
* **Lines**: 183–266
* **Observed Behavior**: Generates 120 synthetic PIL drawings (colored rectangles for roofs, grey lines for roads) and saves them as SpaceNet benchmark tiles.
* **Why it is a problem**: No real satellite imagery or genuine SpaceNet annotations exist. The model evaluates trivial PIL geometric shapes, yielding an artificial 99.13% IoU that cannot generalize to real satellite imagery.
* **Evidence**: Lines 197–266 in `pipelines/01_data_ingestion.py`.
* **Recommended Fix for Later Phase**: In Phase 1, replace with real SpaceNet building footprint imagery and human-annotated polygon/raster masks.

#### Defect CRIT-02: 100% Synthetic Property Exterior Images
* **Severity**: `CRITICAL`
* **File**: `pipelines/01_data_ingestion.py`
* **Function**: `generate_property_condition_dataset`
* **Lines**: 269–362
* **Observed Behavior**: Generates 180 PIL drawings of cartoon houses (sky, lawn, gable roofs, drawn crack lines/moss).
* **Why it is a problem**: ResNet-18 achieves a meaningless 100% test accuracy on trivial color blocks. The model cannot evaluate real real-estate exterior photographs.
* **Evidence**: Lines 269–362 in `pipelines/01_data_ingestion.py`.
* **Recommended Fix for Later Phase**: In Phase 1, ingest genuine multi-class property inspection photos.

#### Defect CRIT-03: False Transfer Learning Claim (Trained from Scratch)
* **Severity**: `CRITICAL`
* **File**: `pipelines/04_train_condition_cnn.py`
* **Function**: `train_condition_model`
* **Line**: 68
* **Observed Behavior**: `model = PropertyConditionClassifier(num_classes=3, pretrained=False).to(device)`.
* **Why it is a problem**: Pretrained weights are never loaded, contradicting transfer learning claims across `README.md`, `ARCHITECTURE.md`, `REPORT.md`, and `MILESTONES.md`.
* **Evidence**: `pipelines/04_train_condition_cnn.py` line 68.
* **Recommended Fix for Later Phase**: In Phase 3, instantiate with `weights=models.ResNet18_Weights.DEFAULT` and train with appropriate layer freezing.

#### Defect CRIT-04: Missing Checkpoints Cause Silent Forward Pass on Random Weights
* **Severity**: `CRITICAL`
* **Files**: `app/views/buyer_view.py` (lines 33–38), `app/views/urban_planner_view.py` (lines 22–28)
* **Functions**: `load_condition_model`, `load_unet_model`
* **Observed Behavior**: If checkpoint files do not exist, the model loaders instantiate random uninitialized neural networks and execute forward passes without error or warning.
* **Why it is a problem**: Produces silent garbage predictions to end users when model checkpoints are missing.
* **Evidence**: `app/views/buyer_view.py` lines 33–38; `app/views/urban_planner_view.py` lines 22–28.
* **Recommended Fix for Later Phase**: In Phase 4, raise an explicit `FileNotFoundError` or display an error message and block inference when checkpoints are missing.

#### Defect CRIT-05: Hardcoded Price Confidence Intervals
* **Severity**: `CRITICAL`
* **Files**: `models/price_regressor.py` (lines 110–118), `app/views/buyer_view.py` (line 127)
* **Function**: `predict_property`
* **Observed Behavior**: Calculates price range by multiplying point prediction by 0.915 and 1.085 ($\pm 8.5\%$).
* **Why it is a problem**: Lacks statistical derivation (not based on residual error variance, quantiles, or conformal prediction), yet labeled "90% confidence interval".
* **Evidence**: `models/price_regressor.py` lines 110–118.
* **Recommended Fix for Later Phase**: In Phase 4, implement formal conformal prediction or quantile regression.

#### Defect CRIT-06: Preprocessing Data Leakage Across Splits
* **Severity**: `CRITICAL`
* **File**: `pipelines/02_data_preprocessing.py`
* **Function**: `preprocess_housing_data`
* **Lines**: 66–74
* **Observed Behavior**: Median imputation `X[col].median()` is computed across the full dataset `X` prior to calling `train_test_split`.
* **Why it is a problem**: Test and validation set statistics leak into training features.
* **Evidence**: `pipelines/02_data_preprocessing.py` lines 66–74.
* **Recommended Fix for Later Phase**: In Phase 2, remove the premature imputation loop and fit imputers strictly on `X_train`.

#### Defect CRIT-07: Test-Set Model Selection Leakage
* **Severity**: `CRITICAL`
* **File**: `pipelines/05_train_price_regressor.py`
* **Function**: `train_price_models`
* **Lines**: 46, 53, 73
* **Observed Behavior**: Evaluates both XGBoost and LightGBM on `(X_test, y_test)` to choose `best_model_name`.
* **Why it is a problem**: Uses the final holdout test set for hyperparameter/model selection, causing test set selection bias.
* **Evidence**: `pipelines/05_train_price_regressor.py` lines 46, 53, 73.
* **Recommended Fix for Later Phase**: In Phase 3, select best models using validation set metrics (`X_val, y_val`) or k-fold CV.

#### Defect CRIT-08: Absolute Windows Paths in Portable Manifests
* **Severity**: `CRITICAL`
* **Files**: `data/raw/property_condition_metadata.csv` (lines 2–181), `data/processed/property_condition_splits.json` (lines 4–1085)
* **Observed Behavior**: Manifests store paths as `C:\Users\LENOVO\Desktop\realityai\data\raw\property_conditions\...`.
* **Why it is a problem**: Repository cannot run on any other user account, drive letter, or operating system without `FileNotFoundError`.
* **Evidence**: `data/raw/property_condition_metadata.csv` line 2.
* **Recommended Fix for Later Phase**: In Phase 2, convert all manifest paths to portable relative paths.

---

### MEDIUM DEFECTS

#### Defect MED-01: Abandoned LSTM Sequence Forecaster
* **Severity**: `MEDIUM`
* **File**: `models/trend_forecaster.py`
* **Function**: `class RealEstateLSTM`
* **Lines**: 93–108
* **Observed Behavior**: Class defined but never trained, saved, or imported, despite documentation claims.
* **Why it is a problem**: Unverified capability and false documentation claim.
* **Evidence**: `models/trend_forecaster.py` line 93; `pipelines/06_train_forecaster.py` imports only Prophet.
* **Recommended Fix for Later Phase**: In Phase 3, either complete LSTM training pipeline or remove claims from documentation.

#### Defect MED-02: Valuation Calculate Button Does Not Gate Calculation
* **Severity**: `MEDIUM`
* **File**: `app/views/buyer_view.py`
* **Lines**: 78–118
* **Observed Behavior**: Calculation and metric display run unconditionally outside of an `if calc_btn:` block.
* **Why it is a problem**: The button does nothing; changing any input recalculates immediately on rerun.
* **Evidence**: `app/views/buyer_view.py` lines 78–82.
* **Recommended Fix for Later Phase**: In Phase 4, guard inference with `if calc_btn:`.

#### Defect MED-03: Fabricated Hidden Features in Dashboard
* **Severity**: `MEDIUM`
* **File**: `app/views/buyer_view.py`
* **Lines**: 90–110
* **Observed Behavior**: 15 features are synthesized via hardcoded multipliers (`1stFlrSF = gr_liv_area * 0.6`, `TotRmsAbvGrd = bedrooms + full_bath + 3`, etc.).
* **Why it is a problem**: Injects uninspected heuristics into user valuation predictions.
* **Evidence**: `app/views/buyer_view.py` lines 90–110.
* **Recommended Fix for Later Phase**: In Phase 4, expose key attributes or use training set median defaults.

#### Defect MED-04: Non-Building Surface Mislabeled as Green Space
* **Severity**: `MEDIUM`
* **Files**: `models/segmentation_unet.py` (lines 145–147), `app/views/urban_planner_view.py` (lines 128–134)
* **Function**: `analyze_satellite_zone`
* **Observed Behavior**: Calculates `open_space_pct = 100.0 - density_pct` and labels it "Open / Green Space" and "Pervious Area".
* **Why it is a problem**: Asphalt roads and parking lots are counted as green pervious space.
* **Evidence**: `models/segmentation_unet.py` line 146.
* **Recommended Fix for Later Phase**: In Phase 4, rename to "Non-Building Coverage %" or integrate NDVI bands.

#### Defect MED-05: Zillow Metro Selection Ignores SizeRank Metadata
* **Severity**: `MEDIUM`
* **File**: `pipelines/06_train_forecaster.py`
* **Lines**: 39–40
* **Observed Behavior**: Selects metros via row count grouping rather than `SizeRank`.
* **Why it is a problem**: Arbitrary selection heuristic ignores economic market rank.
* **Evidence**: `pipelines/06_train_forecaster.py` lines 39–40.
* **Recommended Fix for Later Phase**: In Phase 2, sort by `SizeRank` ascending.

#### Defect MED-06: Prophet Credible Interval Discrepancy (80% vs 95%)
* **Severity**: `MEDIUM`
* **Files**: `models/trend_forecaster.py` (line 37), `app/components/charts.py` (line 48), `docs/REPORT.md` (line 97)
* **Observed Behavior**: Code uses Prophet default `interval_width=0.8` (80%), while documentation and chart legends state 95%.
* **Why it is a problem**: False statistical documentation.
* **Evidence**: `Prophet().interval_width == 0.8`.
* **Recommended Fix for Later Phase**: In Phase 3, explicitly set `interval_width=0.95`.

---

### MINOR DEFECTS

#### Defect MIN-01: Hardcoded Package Versions in System Diagnostics
* **Severity**: `MINOR`
* **File**: `app/views/model_metrics_view.py` (lines 212–217)
* **Observed Behavior**: String literals in YAML block instead of dynamic inspection.
* **Recommended Fix for Later Phase**: Query packages dynamically via `pkg.__version__`.

#### Defect MIN-02: Missing Dependencies in requirements.txt
* **Severity**: `MINOR`
* **File**: `requirements.txt`
* **Observed Behavior**: `reportlab` is imported by `generate_pdf_report.py`, but missing from `requirements.txt`.
* **Recommended Fix for Later Phase**: Add `reportlab>=4.0.0` to `requirements.txt`.

#### Defect MIN-03: Unused Dependencies Declared in requirements.txt
* **Severity**: `MINOR`
* **File**: `requirements.txt`
* **Observed Behavior**: `geopandas`, `shapely`, `opencv-python`, and `seaborn` are declared but never imported.
* **Recommended Fix for Later Phase**: Prune unused packages from `requirements.txt`.

#### Defect MIN-04: Windows Backslashes in SpaceNet Splits Manifest
* **Severity**: `MINOR`
* **File**: `data/processed/spacenet_splits.json` (lines 4–487)
* **Observed Behavior**: Backslashes break path joining on Linux/macOS.
* **Recommended Fix for Later Phase**: Convert to forward slashes.

#### Defect MIN-05: Hardcoded Desktop Path in User Guide
* **Severity**: `MINOR`
* **File**: `PROJECT_SUMMARY_AND_USER_GUIDE.md` (line 274)
* **Observed Behavior**: `cd c:\Users\LENOVO\Desktop\realityai`.
* **Recommended Fix for Later Phase**: Change to `cd realityai`.

---

## 24. Unverified Items

1. **Real-world Generalization of U-Net**: Because no real satellite tiles exist in the repository, the U-Net's performance on genuine WorldView/SpaceNet satellite imagery is **`NOT VERIFIED`**.
2. **Real-world Generalization of ResNet**: Because no real property inspection photos exist in the repository, the ResNet classifier's performance on real photos is **`NOT VERIFIED`**.
3. **Cross-Platform Compatibility on Linux/macOS**: Execution has only been verified on Windows 11. Due to Windows backslashes in manifests, behavior on POSIX systems is **`VERIFIED BROKEN`**.
