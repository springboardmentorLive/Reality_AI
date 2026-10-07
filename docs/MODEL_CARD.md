# REALTYAI2 — MODEL CARDS (PHASE 5 CANONICAL FREEZE)

Comprehensive model documentation establishing data provenance, leakage-safe training protocols, reproducibility standards, empirical performance metrics, uncertainty methodology, and explicit operational limitations for all canonical models in the RealtyAI2 platform.

---

## Model 1: SpaceNet Building Footprint Segmentation U-Net

### 1. Task & Intended Use
- **Primary Task**: Overhead satellite imagery building footprint semantic segmentation.
- **Intended Use**: Estimation of building rooftop footprints and parcel coverage ratios from optical satellite imagery tiles.
- **Geography**: Las Vegas, Nevada, USA (SpaceNet 2 Area of Interest 2).
- **Explicit Out-of-Scope Uses**: Municipal zoning classification, land-use commercial/residential categorization, and green-space/vegetation health detection without calibrated multispectral/NDVI bands. Non-building pixels represent roads, open dirt, driveways, and parking lots, NOT ecological green space.

### 2. Dataset & Splits
- **Source Dataset**: SpaceNet 2 (AOI 2 - Las Vegas) Building Footprint Extraction Challenge.
- **Data Modality**: 3-band RGB pan-sharpened satellite imagery chips (256x256 pixels) and rasterized binary building footprint masks (0: Background, 255: Building).
- **Total Dataset Size**: 20 authentic chips in repository:
  - **Training Partition**: 14 chips (`data/raw/spacenet/research_train/train/`).
  - **Validation Partition**: 6 held-out chips (`data/raw/spacenet/research_train/val/`).
  - **Test Partition**: Official public test ground truth GeoJSON vector building footprints are held out by competition organizers on AWS S3 and are not available locally.
- **Scientific Limitation (N=6 Validation Scope)**: Model performance is evaluated strictly on the 6 held-out Las Vegas validation chips. This sample size is sufficient for functional and pipeline verification, but cannot support broad statistical claims of nationwide generalization across unseen geographical regions.

### 3. Architecture & Training Details
- **Architecture**: 4-level U-Net with skip connections.
- **Feature Channels**: `[16, 32, 64, 128]` with bottleneck at 256 channels.
- **Loss Function**: Combined BCE + Soft Dice Loss:
  $$\mathcal{L} = 0.5 \cdot \text{BCELoss} + 0.5 \cdot \text{DiceLoss}$$
- **Optimizer**: Adam ($\text{lr} = 10^{-4}$).
- **Batch Size**: 4 | **Epochs**: 12.
- **Random Seed**: 42 (Python, NumPy, PyTorch CPU).
- **Canonical Checkpoint**: `models/saved/unet_spacenet_v1.pt`
  - **File Size**: 7,821,487 bytes
  - **SHA-256**: `28bedc6a4750f99bdb2ef8ee2312d85f53d2b72fdf805c2a29e13db6bd4352f7`
  - **Alias**: `models/saved/unet_satellite.pt` (0 weight differences across all 118 tensors).

### 4. Empirical Evaluation Metrics
- **Evaluated Chips**: $N = 6$ held-out development validation chips (`vegas_1041`, `vegas_1042`, `vegas_1047`, `vegas_1048`, `vegas_1049`, `vegas_1051`).
- **Classification Threshold**: $\tau = 0.50$.
- **Validation Mean IoU (Jaccard Index)**: `0.3593` (35.93%).
- **Validation Mean Dice (F1 Score)**: `0.4985` (49.85%).
- **Validation Mean Precision**: `0.7142` (71.42%).
- **Validation Mean Recall**: `0.4334` (43.34%).
- **Evaluation Artifact**: `models/saved/unet_metrics.json`.

---

## Model 2: PEER Structural Collapse Mode Classifier (ResNet-18)

### 1. Task & Intended Use
- **Primary Task**: Image-based structural condition classification according to structural collapse damage state.
- **Target Classes**:
  - `0`: `global_collapse` (complete destruction or severe structural failure)
  - `1`: `non_collapse` (structurally intact, minor or no structural damage)
  - `2`: `partial_collapse` (localized structural failure or severe wall/column distortion)
- **Geography**: Global post-earthquake reconnaissance field surveys (PEER Hub).
- **Explicit Out-of-Scope Uses**: Automated municipal occupancy condemnation, safety sign-off without on-site licensed structural engineer inspection, and legacy subjective cosmetic tiers (`new`, `moderate`, `old`).
- **Scientific Limitation (Post-Disaster Collapse Focus)**: The classifier represents physical earthquake/disaster collapse modes. It does NOT measure cosmetic house condition, deferred maintenance, roof shingle wear, or interior finishes.

### 2. Dataset & Splits
- **Source Dataset**: PEER (Pacific Earthquake Engineering Research Center) Multi-Hazard Image Dataset (Task 5: Collapse Mode).
- **Total Population**: 1,372 authentic inspection images:
  - **Official Training Population**: 1,226 images
    - **Training Partition**: 1,042 images (`data/raw/peer_condition/train/`).
    - **Validation Partition**: 184 images (`data/raw/peer_condition/validation/`).
  - **Untouched Benchmark Test Partition**: 146 images (`data/raw/peer_condition/test/`, from official `task5_X_test.npy` / `task5_y_test.npy`). Untouched during training, validation, and hyperparameter selection. Zero image overlap across splits.
- **Class-Weighted Loss**: Train-only inverse-frequency weights:
  $$w_c = \frac{N_{\text{total}}}{C \cdot N_c} \implies [0.7788, 1.2676, 1.0787]$$

### 3. Architecture & Training Details
- **Architecture**: Deep Residual Network (ResNet-18) with custom classification head (Dropout 0.3, Linear(512, 128), ReLU, BatchNorm1d, Dropout 0.2, Linear(128, 3)).
- **Transfer Learning**: PyTorch official ImageNet weights (`ResNet18_Weights.DEFAULT`).
- **Optimizer**: AdamW ($\text{lr} = 5 \times 10^{-5}$, weight decay $10^{-4}$).
- **Input Resolution**: $224 \times 224 \times 3$ normalized with ImageNet mean and standard deviation.
- **Batch Size**: 16 | **Epochs**: 6 | **Random Seed**: 42.
- **Canonical Checkpoint**: `models/saved/resnet_peer_collapse_v1.pt`
  - **File Size**: 45,056,045 bytes
  - **SHA-256**: `b5c5d063f1dfa2a040ad2f3e3b942e37bbf6d8da67f1399f34769ba127eebc69`
  - **Alias**: `models/saved/resnet_condition.pt` (0 weight differences across all 129 tensors).

### 4. Empirical Evaluation Metrics
- **Untouched Benchmark Test Set ($N=146$)**:
  - **Classification Accuracy**: `70.55%` (103/146 correct).
  - **Macro F1 Score**: `0.6721`.
  - **Weighted F1 Score**: `0.6984`.
  - **Per-Class Metrics**:
    - `non_collapse` ($N=64$): Precision 83.33%, Recall 78.12%, F1 0.8065
    - `partial_collapse` ($N=45$): Precision 61.54%, Recall 53.33%, F1 0.5714
    - `global_collapse` ($N=37$): Precision 63.64%, Recall 75.68%, F1 0.6914
- **Validation Partition ($N=184$)**:
  - Accuracy: `73.37%` | Macro F1: `0.7294` | Weighted F1: `0.7310`.
- **Evaluation Artifact**: `models/saved/condition_metrics.json`.

---

## Model 3: Ames Housing Price Regression (XGBoost)

### 1. Task & Intended Use
- **Primary Task**: Predict residential home transaction `SalePrice` (USD) for properties in Ames, Iowa.
- **Intended Use**: Automated Valuation Model (AVM) for preliminary listing price guidance with transparent, distribution-free prediction bounds.
- **Geography**: Ames, Story County, Iowa, USA.
- **Timeframe**: Historical sales from 2006 through 2010.
- **Explicit Out-of-Scope Uses**: Official municipal tax assessments, legally binding mortgage underwriting, contemporary transactions, or properties outside Ames, Iowa. Must NOT be referred to as "true market value".
- **Scientific Limitation (Ames Specificity & Historical Scope)**: The model is fitted strictly on historical 2006–2010 transactions in Ames, Iowa. It does not reflect inflation, macroeconomic shifts, contemporary price levels, or structural features of other geographic regions.

### 2. Dataset & Preprocessing Integrity
- **Source Dataset**: Ames Housing Dataset (1,460 residential properties; 2 commercial outliers with living area > 4,000 sq ft removed following documentation, leaving 1,458 valid records).
- **Split Partitions**: Leakage-safe 70 / 15 / 15 partition created in Phase 2:
  - Training: $N = 1,020$ properties (`housing_train_df.csv`).
  - Validation: $N = 219$ properties (`housing_val_df.csv`).
  - Held-Out Test: $N = 219$ properties (`housing_test_df.csv`).
- **Feature Contract Adherence**: 27 input features defined in `models/pricing_feature_contract.py` (12 user-provided, 6 derived, 9 imputed). Zero hidden feature fabrication.

### 3. Model Selection & Canonical Artifact
- **Selection Decision**: Evaluated strictly on validation set ($N=219$):
  - XGBoost Val RMSE: **$25,667.70**
  - LightGBM Val RMSE: **$25,669.62**
  - XGBoost was selected based on lower validation error. Test set was NOT used for model selection.
- **Canonical Model**: `models/saved/xgboost_ames_v1.pkl`
  - **File Size**: 727,758 bytes
  - **SHA-256**: `96b9ebc30c5f2edcceedf4f422cc805f77aa979eae6a1a090decf3500780300e`
  - **Alias**: `models/saved/xgboost_price.pkl` (byte-for-byte identical, same SHA-256).

### 4. Empirical Evaluation Metrics ($N=219$ Held-Out Test Properties)
- **MAE**: `$15,092.41`
- **RMSE**: `$22,203.78`
- **$R^2$ Score**: `0.9103`
- **MAPE**: `9.17%`
- **Evaluation Artifact**: `models/saved/price_metrics.json`.

### 5. Pricing Uncertainty Quantification
- **Methodology**: 5-Fold Out-of-Fold Residual Calibration (cross-conformal construction; Vovk 2015; Barber et al. 2021).
- **Calibration Set**: Out-of-fold residuals generated across 5 folds on the training partition ($N=1,020$ properties). Strictly withholds validation ($N=219$) and test ($N=219$) data.
- **Nonconformity Score**: Absolute relative prediction error $s_i = \frac{|y_i - \hat{y}_{-k(i)}(X_i)|}{\hat{y}_{-k(i)}(X_i)}$.
- **Target Nominal Coverage**: 90.0% ($\alpha = 0.10$).
- **Order-Statistic Calculation**:
  $$k = \lceil (n + 1)(1 - \alpha) \rceil = \lceil 1021 \times 0.90 \rceil = 919$$
  $$s_{(919)} = 0.195651 \implies \pm 19.57\% \quad (\text{or } \pm 19.59\% \text{ via continuous quantile})$$
- **Interval Construction**: $[\hat{y} \times (1 - 0.1959), \hat{y} \times (1 + 0.1959)]$.
- **Empirical Test Coverage on $N=219$ Held-Out Homes**: The resulting interval achieved **92.24% empirical coverage** (202 / 219 homes covered), compared with the nominal 90% target.

---

## Model 4: Zillow Regional ZHVI Trend Forecaster (Facebook Prophet)

### 1. Task & Intended Use
- **Primary Task**: Time-series projection of Zillow Home Value Index (ZHVI) for major U.S. metropolitan markets.
- **Intended Use**: 12-to-36-month macroeconomic regional price path forecasting and investor appreciation indicators (CAGR, projected 3-year growth).
- **Geography**: Top 10 U.S. Metropolitan Statistical Areas (MSAs) by SizeRank:
  New York, NY; Los Angeles, CA; Chicago, IL; Dallas, TX; Houston, TX; Washington, DC; Philadelphia, PA; Miami, FL; Atlanta, GA; Boston, MA.
- **Explicit Out-of-Scope Uses**: High-frequency monthly trading, hyper-local parcel-level pricing, and speculative arbitrage. ZHVI represents a smoothed median market index across entire MSAs and is not comparable to individual single-property sale prices (`SalePrice`).

### 2. Methodological Design & Chronological Split
- **Architecture**: Facebook Prophet additive structural time-series models with yearly seasonality and changepoint detection.
- **Omission of LSTM**: Audited and explicitly omitted. Deep neural sequence architectures with only 264 monthly observations per metro exhibit severe parameter instability, overfitting, and poor out-of-sample generalization.
- **Chronological Temporal Split (Leakage-Safe)**:
  - **Training Period**: `2000-01-31` to `2021-12-31` (264 monthly observations).
  - **Validation Period**: `2022-01-31` to `2023-12-31` (24 monthly observations).
  - **Held-Out Test Period**: `2024-01-31` to `2026-08-31` (32 monthly observations, $N=320$ total observations across 10 MSAs).
  - **Future Projection Period**: `2026-09-30` to `2029-08-31` (36 monthly forward projections).
- **Canonical Metadata Artifact**: `models/saved/forecaster_summary.json`
  - **File Size**: 13,052 bytes
  - **SHA-256**: `1d008e35b24bc80a4b424b1dfc03698a5844fb63d92a5331955cc56ab2ad0ecd`

### 3. Forecast Uncertainty & Empirical Coverage
- **Nominal Width**: 95% forecast interval (`interval_width=0.95`).
- **Statistical Interpretation**: Bayesian posterior predictive forecast interval incorporating parameter uncertainty and trend innovation variance. It is labeled strictly as `95% forecast interval`, NOT a calibrated confidence interval or conformal guarantee.
- **Empirical Test Coverage**: **40.31%** across the 10 MSAs during the 2024–2026 test period.
- **Scientific Limitation (Low Interval Coverage)**: The empirical coverage of 40.31% reflects the limitation of fitting historical trends on the 2000–2021 period and projecting into post-2022 macroeconomic conditions. We report this coverage truthfully and avoid unsubstantiated causal speculation.

### 4. Authoritative Empirical Metrics (Held-Out Test Period 2024–2026)
- **Aggregation Rule**: Arithmetic Mean of per-MSA test metrics across 10 MSAs.
- **Mean MAE**: `$39,580.75`
- **Mean RMSE**: `$45,888.30`
- **Mean MAPE**: `8.71%`
- **Mean 95% Forecast Interval Empirical Coverage**: `40.31%`
- **Pooled Observation-Level Metrics ($N=320$)**: MAE: `$39,580.75`, RMSE: `$53,862.47`, MAPE: `8.71%`, Coverage: `38.44%`.
