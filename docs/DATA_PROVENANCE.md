# DATA PROVENANCE & SPECIFICATION AUDIT (PHASE 2)

This document establishes the official data provenance, legal licensing, physical storage paths, preprocessing transformations, verification hashes, and methodological limitations for all authentic datasets utilized in **RealtyAI2**.

In strict accordance with the **Absolute Rules**, every claim below is verified from downloaded files, authoritative upstream repositories, or primary scientific publications. Unverified claims are explicitly designated `NOT VERIFIED`. Procedural synthetic data generation has been fully deprecated and excised from the active ingestion pipeline.

---

## 1. SpaceNet 2: Building Footprint Extraction (AOI_2_Vegas)

### Official Origin vs. Smoke-Test Derivative
* **Original Official Dataset**: SpaceNet 2 Building Detection Dataset — Area of Interest 2 (Las Vegas, Nevada, USA) hosted directly on public AWS S3 bucket `s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/`.
* **Third-Party 20-Chip Derivative**: Hugging Face repository `khalilurrahmanridoykhan/spacenet-buildings-vegas-smoketest-sample`. This is an authentic 20-chip subset explicitly released by its author as a smoke-test sample. It is NOT the full challenge training dataset.
* **Phase 2 Dual-Dataset Strategy**:
  1. **Smoke-Test / CI Sample**: The 20-chip sample is preserved in `data/raw/spacenet/smoke_test/` and recorded in `data/manifests/spacenet_smoke_test_manifest.csv`. Its explicit, documented role is **PIPELINE VALIDATION / CI / DEVELOPMENT ONLY**. It must never be reported as a final research training result.
  2. **Research-Training Subset**: A 30-chip subset ingested directly from the official SpaceNet AWS S3 bucket (`s3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/`) into `data/raw/spacenet/research_train/` and recorded in `data/manifests/spacenet_train_manifest.csv`.

### Official Upstream Organization
* **Consortium**: SpaceNet LLC (Maxar Technologies / DigitalGlobe, CosmiQ Works, In-Q-Tel, AWS, Radiant Earth Foundation).

### License & Attribution
* **License**: Creative Commons Attribution-ShareAlike 4.0 International (**CC BY-SA 4.0**).
* **Commercial Use Permitted**: Yes, with attribution and share-alike derivative distribution.
* **Citation**:
  ```text
  Van Etten, A., Lindenbaum, D., & Bacastow, T. M. (2018).
  SpaceNet: A Remote Sensing Dataset and Challenge Series.
  arXiv preprint arXiv:1807.01238.
  Imagery provided by Maxar Technologies (DigitalGlobe WorldView-3).
  ```

### Contents & Geography
* **Sensor / Platform**: DigitalGlobe WorldView-3 Commercial Satellite.
* **Native Resolution**: ~0.3m ground sample distance (pan-sharpened 3-band RGB).
* **Native Dimensions**: 650 × 650 pixels per chip, 3 spectral bands (16-bit uint16).
* **Coordinate Reference System (CRS)**: `WGS 84` (`EPSG:4326`), Units: Decimal Degrees.
* **Affine Transform**: Pixel resolution `~2.7e-6` degrees per pixel (~30 cm GSD).
* **Geographic Coverage**: Las Vegas, Nevada, USA (Bounding Box: Latitude ~36.13° N, Longitude ~-115.31° W).
* **Ground Truth Labels**: Vector building footprints in GeoJSON format (`EPSG:4326` polygon geometries).

### Research-Training Dataset Specifications (Phase 2 S3 Ingestion)
* **Ingestion Script**: `scripts/ingest_spacenet_research_train.py`
* **Raw GeoTIFF Chips**: `data/raw/spacenet/research_train/raw_chips/img*.tif` (30 files, 72.6 MB total)
* **Raw Vector Polygons**: `data/raw/spacenet/research_train/raw_chips/img*.geojson` (30 files, ~1.2 MB total)
* **Processed 8-bit RGB Images**: `data/raw/spacenet/research_train/images/spacenet_vegas_chip_*.png` (30 files, 650 × 650 × 3, uint8)
* **Processed Ground-Truth Masks**: `data/raw/spacenet/research_train/masks/spacenet_vegas_chip_*_mask.png` (30 files, 650 × 650, uint8 binary: 0 = background, 255 = building)
* **Total Building Polygons**: 789 distinct vector building footprints.
* **Mean Building Pixel Coverage**: 16.95% of chip area.
* **Active Manifest**: `data/manifests/spacenet_train_manifest.csv` (100% relative, portable forward-slash paths).
* **Split Definition**: `data/processed/spacenet_train_splits.json`
  * **Train**: 24 whole chips (80%) — 650 buildings
  * **Validation**: 6 whole chips (20%) — 139 buildings (`img1041`, `img1042`, `img1047`, `img1048`, `img1049`, `img1051`)
  * **Geographic Leakage Safeguard**: Strict whole-scene chip separation. Neighboring sub-tiles do not cross partitions.

### Smoke-Test Dataset Specifications
* **Location**: `data/raw/spacenet/smoke_test/` (20 chips, 593 buildings, 17.14% coverage).
* **Manifest**: `data/manifests/spacenet_smoke_test_manifest.csv`.
* **Splits**: `data/processed/spacenet_smoke_test_splits.json` (14 train, 3 val, 3 test for CI smoke testing).

### Processing & Rasterization Integrity
1. **Radiometric Preservation**: 16-bit GeoTIFFs normalized via 2nd-98th percentile contrast stretching for 8-bit visual derivatives while keeping raw GeoTIFF as authoritative source.
2. **Affine-Preserved Spatial Rasterization**: Vector building polygons rasterized via `rasterio.features.rasterize` parameterized by the exact native affine transform of each GeoTIFF.
3. **Nearest-Neighbor Interpolation**: All mask resizing in pipelines (`pipelines/03_train_segmentation.py`) and UI (`app/views/urban_planner_view.py`) strictly enforces `resample=Image.Resampling.NEAREST` to prevent label blurring.
4. **Honest Geospatial Metrics**: Renamed non-building surface area from `open_space_pct` to `non_building_area_pct`. No claims of "green space" are made without multi-spectral vegetation indices (e.g. NDVI).

### Limitations
* **Geographic Scope**: Exclusively Las Vegas, Nevada (arid desert biome, low tree canopy, regular suburban grid). Does not generalize to dense urban canyons, high-canopy tropical environments, or historical European street layouts.
* **Scale**: 30 authentic chips selected for local disk feasibility (~75 MB total); scaled production models will require downloading further AOIs from SpaceNet S3.

---

## 2. Structural Condition Assessment: PEER Hub ImageNet (PHI-Net) Task 5

### Upstream Identity & Provenance
* **Name**: PEER Hub ImageNet (PHI-Net) — Task 5: Collapse Mode Classification
* **Version / Release**: PHI-Net Benchmark Dataset (Release 2020)
* **Source Organization**: Pacific Earthquake Engineering Research (PEER) Center, University of California, Berkeley
* **Principal Investigators / Authors**: Yuqing Gao (`gaoyuqing@berkeley.edu`), Khalid M. Mosalam
* **Primary Reference**:
  ```text
  Gao, Y., & Mosalam, K. M. (2020).
  PEER Hub ImageNet: A large-scale multi-attribute benchmark dataset of structural images.
  PEER Report No. 2020/02, Pacific Earthquake Engineering Research Center, UC Berkeley.
  ```
* **Distribution Endpoint**: Hugging Face dataset repository `kks32/building-damage` (`data.zip`).

### Official Source Split Semantics & Ingestion (Phase 2 Case 1 Resolution)
In accordance with **Phase 2 Decision Rule B (Case 1)**, the source archive was audited to determine the true semantic meaning of its filenames:
1. `task5_X_test.npy` & `task5_y_test.npy` (146 images): The official benchmark test partition designated by Yuqing Gao (UC Berkeley).
2. `task5_X_train.npy` & `task5_y_train.npy` (1,226 images): The official training population.
3. **Methodological Resolution**:
   - The 146 official test images are preserved **strictly and exclusively** in `data/raw/property_conditions/test/`. Under no circumstances are benchmark test images used for training or validation.
   - The 1,226 official training images were downloaded via streamed HTTP Range requests and partitioned into:
     - **Train**: 1,042 images (85% stratified) in `data/raw/property_conditions/train/`
     - **Validation**: 184 images (15% stratified) in `data/raw/property_conditions/val/`
   - **Zero Benchmark Leakage**: Test images have 0% overlap with training or validation partitions.

### License & Attribution
* **License**: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (**CC BY-NC-SA 4.0**).
* **Commercial Restrictions**: Strictly non-commercial academic research.

### Class Mapping & Counts Audit
The official collapse-mode classes defined by the PEER Center are preserved verbatim:
* Class 0 (`gc`): `global_collapse` (Total: 592, 43.15%)
* Class 1 (`nc`): `non_collapse` (Total: 361, 26.31%)
* Class 2 (`pc`): `partial_collapse` (Total: 419, 30.54%)

#### Partition Breakdown Table:
| Partition | Global Collapse | Non-Collapse | Partial Collapse | Total |
| :--- | :--- | :--- | :--- | :--- |
| **Train** (Official Train Subsplit) | 446 (42.80%) | 274 (26.30%) | 322 (30.90%) | **1,042** |
| **Validation** (Official Train Subsplit) | 79 (42.93%) | 48 (26.09%) | 57 (30.98%) | **184** |
| **Test** (Official Benchmark Test) | 67 (45.89%) | 39 (26.71%) | 40 (27.40%) | **146** |
| **Total Population** | **592** | **361** | **419** | **1,372** |

### Local Storage & Manifest
* **Directory Layout**: `data/raw/property_conditions/{train,val,test}/{global_collapse,non_collapse,partial_collapse}/`
* **Manifest**: `data/manifests/property_condition_manifest.csv` (1,372 records, 100% relative, portable paths).
* **Splits JSON**: `data/processed/property_condition_splits.json`.

### Limitations
* **Structural Context**: Images originate from post-earthquake structural reconnaissance (e.g. 2011 Christchurch, 2014 South Napa, 2015 Nepal earthquakes). They evaluate structural collapse modes and life-safety integrity, NOT cosmetic residential curb appeal, paint wear, or property age.
* **Class Imbalance**: Non-collapse (26.3%) has fewer samples than global collapse (43.2%).
* **Image Geometry**: Standardized $224 \times 224$ crops centered on structural elements.

---

## 3. Housing Price Regression: Ames Housing Dataset

### Identity & Provenance
* **Name**: Ames Housing Dataset
* **Author**: Dean De Cock (Truman State University)
* **Primary Publication**: *Journal of Statistics Education*, Volume 19, Number 3 (2011).
* **Citation**:
  ```text
  De Cock, D. (2011).
  Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Project.
  Journal of Statistics Education, 19(3), 1-14.
  ```

### License Statement Correction
* **Verified License Status**: `NOT VERIFIED (Open Academic / Educational Use under Journal of Statistics Education publication terms)`.
* **Correction Note**: The dataset was previously described informally as "Public Domain". Authoritative review shows it was published openly in the *Journal of Statistics Education* for educational and academic research. It is NOT formally dedicated under CC0 or public domain dedication.

### Labeled vs. Unlabeled Population Rules
1. **Labeled Population (`train.csv`)**: 1,460 single-family home transactions (2006–2010) in Ames, Iowa, with verified ground-truth `SalePrice`.
   - Outliers: In accordance with Dean De Cock's documented recommendation, 2 extreme outliers with `GrLivArea > 4000` and `SalePrice < $300,000` were dropped, leaving 1,458 valid records.
2. **Kaggle Unlabeled Set (`test.csv`)**: 1,459 records. Contains NO `SalePrice` targets.
   - **Absolute Rule**: `test.csv` is strictly forbidden from being used for model evaluation, validation, or benchmarking.
3. **Phase 2 Research Splits**: Formed strictly from the 1,458 labeled properties using fixed seed 42:
   - **research_train**: 1,020 properties (70%)
   - **research_validation**: 219 properties (15%)
   - **research_test**: 219 properties (15%)
   - Stored in: `data/processed/housing_{train,val,test}_df.csv`.

### Preprocessing Leakage Repair (Phase 2 Fix)
* **Previous Defect**: Medians and standard scalers were computed across the entire dataset prior to splitting.
* **Phase 2 Architecture**: Leakage-safe `sklearn.compose.ColumnTransformer`:
  - Numerical Pipeline: `SimpleImputer(strategy="median")` + `StandardScaler()`
  - Categorical Pipeline: `SimpleImputer(strategy="constant", fill_value="Missing")` + `OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`
  - **Fitted strictly on `X_train`**. Validation and test partitions are transformed without influencing fitted parameters.
  - Serialized preprocessor: `data/processed/housing_preprocessor.pkl`.

### Limitations
* **Geographic Specificity**: Single municipality (Ames, Iowa). Reflects Midwestern suburban and college-town housing patterns.
* **Temporal Period**: 2006–2010 real estate market, capturing the 2008 financial crisis. Does not reflect contemporary post-2020 home price dynamics.

---

## 4. Metro Market Time Series: Zillow Home Value Index (ZHVI)

### Identity & Source
* **Name**: Zillow Home Value Index (ZHVI) — Mid-Tier Single-Family & Condos (Smoothed, Seasonally Adjusted)
* **Organization**: Zillow Group, Inc. (Zillow Research)
* **Endpoint**: `https://files.zillowstatic.com/research/public_csvs/zhvi/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv`
* **Local Ingestion Path**: `data/raw/zillow/Metro_zhvi_month.csv` (895 metros, 325 columns).
* **Access Date**: Verified October 2, 2026.

### License & Attribution Terms
* **License**: Zillow Terms of Use for Public Research Data.
* **Attribution Requirement**: Must cite "Zillow Home Value Index (ZHVI)" and reference Zillow Real Estate Data (`https://www.zillow.com/research/data/`).
* **Redistribution Terms**: Free for research, public analysis, and educational modeling with source attribution. Not permitted for direct commercial data reselling.

### Contents & Schema
* **Coverage**: 895 US Metropolitan Statistical Areas (MSAs) plus national aggregate.
* **Time Range**: Monthly snapshots from January 31, 2000 through current release (320 months).
* **Processed Dataset**: `data/processed/zillow_zhvi_processed.csv` (unpivoted long format).

### Limitations
* **Aggregate Indicator**: ZHVI measures median valuation of mid-tier housing across entire metropolitan regions; it does not evaluate micro-neighborhood or parcel-specific variations.
* **Smoothing**: Values are seasonally adjusted and smoothed, dampening extreme short-term shocks.

---

## 5. Pricing Input Interface Contract (Phase 2 Contract Redesign)

To eliminate hidden feature fabrication in buyer and investor workflows, all 27 tabular model features are governed by the formal interface contract (`models/pricing_feature_contract.py`):

| Model Feature | Feature State | Documentation / Justification Rule |
| :--- | :--- | :--- |
| `GrLivArea` | `USER_PROVIDED` | Above-grade finished living area in sq ft |
| `OverallQual` | `USER_PROVIDED` | Overall material & finish quality (1–10) |
| `OverallCond` | `USER_PROVIDED` | Overall condition rating (1–10) |
| `YearBuilt` | `USER_PROVIDED` | Original construction year |
| `YearRemodAdd` | `USER_PROVIDED` | Remodel date (defaults to construction year) |
| `TotalBsmtSF` | `USER_PROVIDED` | Total basement area in sq ft |
| `BedroomAbvGr` | `USER_PROVIDED` | Bedrooms above grade |
| `FullBath` | `USER_PROVIDED` | Full bathrooms |
| `HalfBath` | `USER_PROVIDED` | Half bathrooms |
| `GarageCars` | `USER_PROVIDED` | Garage capacity (cars) |
| `LotArea` | `USER_PROVIDED` | Lot size in sq ft |
| `Neighborhood` | `USER_PROVIDED` | Ames neighborhood physical location |
| `1stFlrSF` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | Floor area partition based on building story height |
| `2ndFlrSF` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | Upper story area derived from total living area |
| `GarageArea` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | Standard 240 sq ft allocation per garage car stall |
| `ExterQual` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | 'Gd' for luxury (OverallQual >= 7), 'TA' for average |
| `BsmtQual` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | 'Gd' for luxury (OverallQual >= 7), 'TA' for average |
| `KitchenQual` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | 'Gd' for luxury (OverallQual >= 7), 'TA' for average |
| `HouseStyle` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | '2Story' if multi-story living area, else '1Story' |
| `Foundation` | `DERIVED_FROM_USER_INPUT_WITH_JUSTIFIED_RULE` | Poured concrete ('PConc') post-1990, cinder block pre-1990 |
| `TotRmsAbvGrd` | `EXPLICITLY_IMPUTED` | Population median (6 rooms) |
| `KitchenAbvGr` | `EXPLICITLY_IMPUTED` | Population median (1 kitchen) |
| `Fireplaces` | `EXPLICITLY_IMPUTED` | Population median (1 fireplace) |
| `WoodDeckSF` | `EXPLICITLY_IMPUTED` | Population median (0 sq ft) |
| `OpenPorchSF` | `EXPLICITLY_IMPUTED` | Population median (25 sq ft) |
| `MSZoning` | `EXPLICITLY_IMPUTED` | Population mode ('RL' - Residential Low Density) |
| `BldgType` | `EXPLICITLY_IMPUTED` | Population mode ('1Fam' - Single-family detached) |

* **Phase 2 Strategic Decision**: Documented in `docs/DECISIONS.md`. Option A (retrain in Phase 3 on user-facing features) is selected as the primary architectural direction. Option B (explicit documented imputation) is implemented in `models/pricing_feature_contract.py` for transitional compatibility.

---

## 6. Legacy Synthetic Checkpoint Quarantine Audit

All legacy models trained on synthetic imagery or flawed preprocessing have been quarantined to `models/legacy_synthetic/`. Active inference directory `models/saved/` contains 0 synthetic checkpoints.

| Quarantined Artifact | Original Role | Training Data Status | Quarantine Reason |
| :--- | :--- | :--- | :--- |
| `unet_satellite.pt` | Satellite Segmentation U-Net | **Synthetic Imagery** | Trained on procedurally drawn geometric rectangles. Unfit for scientific use. |
| `resnet_condition.pt` | Structural Condition ResNet | **Synthetic Imagery** | Trained on procedural PIL drawings. Unfit for scientific use. |
| `xgboost_price.pkl` | Tabular Pricing XGBoost | **Leaked Preprocessing** | Trained on pre-split median imputed data without leakage protection. |
| `lightgbm_price.pkl` | Tabular Pricing LightGBM | **Leaked Preprocessing** | Trained on pre-split median imputed data without leakage protection. |
| `unet_metrics.json` | U-Net Evaluation Metrics | **Synthetic Metrics** | Fabricated metrics on synthetic test split. |
| `condition_metrics.json` | ResNet Evaluation Metrics | **Synthetic Metrics** | Fabricated metrics on synthetic test split. |
| `price_metrics.json` | Pricing Metrics | **Leaked Metrics** | Flawed benchmark metrics. |
| `forecaster_summary.json` | Forecast Summary | **Legacy Baseline** | Baseline metrics pending Phase 3 expansion. |

### Safe Failure Behavior
When an active model checkpoint is missing from `models/saved/`, loaders return `None` rather than falling back to uninitialized random weights. The UI displays an explicit status: `MODEL NOT AVAILABLE` (or `TRAINED — VERIFIED` when verified real weights are active). No random initialization or synthetic data fallback is ever executed.
