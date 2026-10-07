# RealtyAI: Smart Real Estate Insight Platform
## Comprehensive Final Academic & Technical Research Report

**Author**: RealtyAI Engineering Team  
**Date**: October 2026  
**Status**: Completed & Research-Frozen (Phase 5 Final Validation)  

---

## 1. Executive Summary
Real estate acquisition, investment allocation, and municipal urban planning have historically suffered from siloed information: pricing relies on historical transactions, property structural integrity assessment requires manual engineering surveys, and zoning analysis relies on outdated land surveys.

**RealtyAI** is an integrated AI-driven real estate platform that combines:
1. **SpaceNet 2 Satellite Imagery**: Automated building footprint segmentation and urban density profiling using PyTorch U-Net on high-resolution aerial tiles.
2. **Visual Structural Integrity Assessment**: Computer vision classification of post-disaster structural collapse modes (`non_collapse`, `partial_collapse`, `global_collapse`) using a fine-tuned ResNet-18 classifier on authentic PEER Hub ImageNet Task 5 imagery.
3. **Micro-Market Valuation**: Hedonic property valuation using gradient-boosted ensembles (XGBoost & LightGBM) with 90% cross-conformal prediction intervals.
4. **Macro Trend Forecasting**: 36-month regional appreciation and risk modeling using Facebook Prophet across the Top 10 US Metropolitan Statistical Areas.
5. **Interactive Multi-Persona Interface**: An end-to-end Streamlit application tailored for Home Buyers, Real Estate Investors, and Urban Planners, backed by an automated 46-test regression and safety verification suite.

---

## 2. Project Statement & Objectives
The primary objective of RealtyAI is to build a scientifically rigorous, leakage-safe machine learning platform that evaluates structural collapse risks, predicts price trends with valid uncertainty bounds, and segments satellite images of real estate parcels.

### Key Target Outcomes:
- Robust multimodal data pipelines handling both unstructured imagery and structured tabular data without data leakage.
- Validated computer vision architectures trained and tested on authentic, non-synthetic datasets.
- Statistically sound uncertainty quantification (finite-sample conformal prediction bounds and empirical forecast intervals).
- Actionable market intelligence delivered via an intuitive, accessible dashboard that fails safely when models or inputs are missing.

---

## 3. Data Ingestion, Cleaning & Provenance

### 3.1 Datasets Utilized
- **Ames Housing Dataset (Kaggle)**: 1,460 residential transaction records in Ames, Iowa, containing 81 raw attributes. After removing 2 non-typical commercial/partial sales with living area > 4,000 sq ft (as documented by Prof. Dean De Cock), 1,458 valid residential properties remain:
  - Training: 1,020 properties (70%)
  - Validation: 219 properties (15%)
  - Held-out Test: 219 properties (15%)
- **Zillow Home Value Index (ZHVI)**: Official Zillow Research monthly time series dataset tracking smoothed, seasonally adjusted typical home values across the Top 10 US MSAs by SizeRank from January 2000 through August 2026 (264 monthly observations per metro).
  - Training Period: 2000-01-31 to 2021-12-31 (264 months)
  - Validation Period: 2022-01-31 to 2023-12-31 (24 months)
  - Held-out Test Period: 2024-01-31 to 2026-08-31 (32 months)
  - Future Projections: 36 months forward (September 2026 through August 2029)
- **SpaceNet 2 Las Vegas Building Footprints**: Authentic 20-chip sample (256x256 RGB aerial tiles paired with human-annotated binary building footprint masks) from the SpaceNet 2 Challenge (AOI 2 Las Vegas):
  - Training: 14 chips
  - Held-out Validation: 6 chips (`vegas_1041`, `vegas_1042`, `vegas_1047`, `vegas_1048`, `vegas_1049`, `vegas_1051`)
- **PEER Hub ImageNet Task 5 Structural Imagery**: Authentic 1,372 exterior disaster reconnaissance photographs from the Pacific Earthquake Engineering Research (PEER) Center:
  - Official Training Population: 1,226 images (1,042 train, 184 validation)
  - Untouched Benchmark Test Population: 146 images (isolated benchmark test set with zero overlap)

### 3.2 Preprocessing & Feature Engineering
- **Outlier Mitigation**: Filtered 2 extreme living area transactions (>4,000 sq ft) to stabilize regression gradients.
- **Missing Value Strategy**: Imputation pipelines fitted strictly on training data ($X_{\text{train}}$) to prevent data leakage:
  - Numerical features: Imputed using median statistics.
  - Categorical features: Constant imputation (`"Missing"`) followed by ordinal encoding.
- **Target Transformation**: Applied logarithmic transformation $y = \log(1 + \text{SalePrice})$ to normalize target distributions and stabilize variance across disparate price brackets.
- **Vision Preprocessing**: Resizing to canonical sizes (256x256 for U-Net, 224x224 for ResNet), random horizontal/vertical flips for augmentation, and ImageNet standardization ($\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$).

---

## 4. Modeling Methodologies & Canonical Architectures

### 4.1 Satellite Building Footprint Segmentation: PyTorch U-Net
- **Canonical Artifact**: `models/saved/unet_spacenet_v1.pt` (SHA-256: `28bedc6a4750f99bdb2ef8ee2312d85f53d2b72fdf805c2a29e13db6bd4352f7`)
- **Architecture**: Contracting encoder path with 4 downsampling stages (DoubleConv + MaxPool2d), an intermediate bottleneck, and 4 expansive decoding stages (ConvTranspose2d + skip-connection concatenation).
- **Loss Function**: `BCEDiceLoss` combining Binary Cross-Entropy (for pixel-level classification) and Dice Loss (for spatial boundary overlap):
  $$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot (1 - \text{Dice})$$
- **Held-Out Validation Performance ($N=6$ chips)**:
  - **Mean IoU (Jaccard Index)**: **35.93%**
  - **Mean Dice Coefficient**: **49.85%**
  - **Mean Precision**: **71.42%**
  - **Mean Recall**: **43.34%**
- **Urban Planning Integration**: Automatically extracts building coverage % and non-building ground surface % (unpaved dirt, roads, driveways). Non-building ground area is strictly not classified as vegetation or green space without multi-spectral indices (e.g. NDVI).

### 4.2 Structural Collapse Classification: ResNet-18
- **Canonical Artifact**: `models/saved/resnet_peer_collapse_v1.pt` (SHA-256: `b5c5d063f1dfa2a040ad2f3e3b942e37bbf6d8da67f1399f34769ba127eebc69`)
- **Architecture**: Residual network (ResNet-18) backbone adapted with a custom multi-layer classification head with batch normalization and dropout regularization ($p=0.3, 0.2$).
- **Classes**: 3 authentic PEER Hub ImageNet Task 5 post-disaster collapse modes (`non_collapse`, `partial_collapse`, `global_collapse`).
- **Untouched Benchmark Test Performance ($N=146$ images)**:
  - **Classification Accuracy**: **70.55%**
  - **Macro F1 Score**: **0.6721**
  - **Weighted F1 Score**: **0.6984**
  - **Per-Class Metrics**:
    - `non_collapse` ($N=64$): Precision 83.33%, Recall 78.12%, F1 0.8065
    - `partial_collapse` ($N=45$): Precision 61.54%, Recall 53.33%, F1 0.5714
    - `global_collapse` ($N=37$): Precision 63.64%, Recall 75.68%, F1 0.6914
- **Inspection Safety Integration**: Flags immediate life-safety structural hazards and recommends professional structural engineering evaluation. Cosmetic wear and arbitrary renovation cost claims are explicitly omitted.

### 4.3 Property Valuation Regressors: XGBoost & LightGBM
- **Canonical Artifact**: `models/saved/xgboost_ames_v1.pkl` (SHA-256: `96b9ebc30c5f2edcceedf4f422cc805f77aa979eae6a1a090decf3500780300e`)
- **Comparison Artifact**: `models/saved/lightgbm_price.pkl`
- **Model Selection**: Conducted strictly on validation set ($N=219$):
  - XGBoost Val RMSE: **$25,667.70**
  - LightGBM Val RMSE: **$25,669.62**
  - XGBoost was selected based on lower validation error. Test data remained completely untouched.
- **Held-Out Test Performance ($N=219$ properties)**:
  | Metric | XGBoost (Canonical) | LightGBM | Delta / Best |
  | :--- | :--- | :--- | :--- |
  | **MAE** | **$15,092.41** | $15,227.48 | XGBoost (-$135.07) |
  | **RMSE** | **$22,203.78** | $22,498.19 | XGBoost (-$294.41) |
  | **$R^2$ Score** | **0.9103** | 0.9079 | XGBoost (+0.0024) |
  | **MAPE** | **9.17%** | 9.33% | XGBoost (-0.16%) |
- **Pricing Uncertainty Quantification**:
  - Implemented 5-fold out-of-fold residual calibration on $N=1,020$ training residuals (cross-conformal construction), strictly withholding validation and test data.
  - Mathematical finite-sample order statistic index: $k = \lceil(n+1)(1-\alpha)\rceil = \lceil 1,021 \times 0.90 \rceil = 919$.
  - Calibrated relative margin: $\pm 19.59\%$ (discrete order statistic $s_{(919)} = 0.195651 \implies \pm 19.57\%$).
  - **Empirical Test Coverage**: The resulting interval achieved **92.24% empirical coverage** (202/219) on the held-out test set, compared with the nominal 90% target.

### 4.4 Regional Trend Forecasting: Facebook Prophet
- **Canonical Artifact**: `models/saved/forecaster_summary.json` (SHA-256: `1d008e35b24bc80a4b424b1dfc03698a5844fb63d92a5331955cc56ab2ad0ecd`)
- **Methodology**: Additive time-series model decomposing regional Zillow ZHVI home prices into piecewise linear trend trajectories and annual seasonality cycles:
  $$y(t) = g(t) + s(t) + \epsilon_t$$
- **Evaluation Methodology**: Fitted strictly on historical data (2000–2021). Evaluated across the 10 largest US MSAs on 32 held-out test months (2024-01-31 to 2026-08-31; $N=320$ total observations).
- **Authoritative Headline Metrics (Arithmetic Mean of 10 MSAs)**:
  - **Mean MAE**: **$39,580.75**
  - **Mean RMSE**: **$45,888.30**
  - **Mean MAPE**: **8.71%**
  - **Mean 95% Forecast Interval Coverage**: **40.31%**
- **Per-MSA Evaluation Breakdown**:
  | Metro Area | Test N | MAE ($) | RMSE ($) | MAPE (%) | 95% Forecast Interval Coverage |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | New York, NY | 32 | $48,348.65 | $51,643.08 | 7.95% | 15.62% |
  | Los Angeles, CA | 32 | $88,095.34 | $91,915.22 | 10.02% | 0.00% |
  | Chicago, IL | 32 | $20,381.16 | $22,467.43 | 6.75% | 84.38% |
  | Dallas, TX | 32 | $30,958.87 | $37,845.24 | 8.35% | 40.62% |
  | Houston, TX | 32 | $21,123.69 | $26,380.08 | 7.90% | 56.25% |
  | Washington, DC | 32 | $39,266.36 | $42,668.78 | 7.39% | 21.88% |
  | Philadelphia, PA | 32 | $29,088.13 | $32,689.70 | 8.57% | 34.38% |
  | Miami, FL | 32 | $51,805.81 | $55,420.91 | 11.08% | 18.75% |
  | Atlanta, GA | 32 | $35,283.47 | $44,978.11 | 9.42% | 43.75% |
  | Boston, MA | 32 | $31,455.97 | $52,874.45 | 9.68% | 87.50% |

---

## 5. System Usability & Multi-Persona Application
The frontend application was developed using Streamlit with custom CSS glassmorphism, responsive metrics cards, Plotly interactive graphics, and Folium geospatial mapping:

1. **Buyer & Property Inspector Persona**:
   - Dynamic valuation estimator with strict 27-feature contract transparency (12 user inputs, 6 derived, 9 imputed).
   - Prominent research applicability warning (illustrative Ames, Iowa model; never "true market value").
   - 90% cross-conformal prediction interval display ($\pm 19.59\%$).
   - Interactive property photo inspector classifying authentic PEER structural collapse modes (`non_collapse`, `partial_collapse`, `global_collapse`) with safe failure handling.
2. **Real Estate Investor Persona**:
   - 36-month regional price trend forecasting with 95% forecast intervals.
   - Clear delineation between observed historical data, held-out test data, and future projection periods.
   - Transparent disclosure of 40.31% empirical interval coverage without unsubstantiated macroeconomic claims.
   - Interactive Folium national map and capital allocation ROI simulator.
3. **Urban Planner & Satellite AI Persona**:
   - Interactive aerial tile selector with real-time U-Net building footprint segmentation.
   - Authentic Las Vegas validation chip evaluation ($N=6$).
   - Land use zoning analytics: building footprint coverage % vs non-building ground area % (never "green space").
4. **Model Performance Hub**:
   - Unified diagnostic suite ingesting live JSON artifacts with zero hardcoded numbers.
   - Complete Metric Provenance Matrix with SHA-256 checksums and exact split counts.

---

## 6. Verification, Testing & Reproducibility
All components have undergone automated testing and independent verification:
- **PyTest Suite (`python -m pytest -v`)**: 46 passed, 0 failed in 26.23s.
  - 16 data pipeline and manifest integrity tests (`test_data_pipeline.py`)
  - 14 model architecture, tensor flow, and conformal order statistic tests (`test_models.py`)
  - 15 dashboard view integration and safety tests (`test_dashboard_integration.py`)
  - 1 application smoke test (`test_app.py`)
- **Independent Verification Script (`scripts/final_independent_verification.py`)**:
  - Reloads all 4 canonical artifacts independently from raw/processed data.
  - Recomputes SpaceNet 6-chip segmentation metrics, PEER 146-image test classification metrics, Ames 219-property test regression metrics, and Zillow 10-MSA forecast metrics.
  - Outputs machine-readable `models/saved/final_verification.json` confirming 100% agreement with saved artifacts.

---

## 7. Known Scientific Limitations & Ethical Scope
1. **SpaceNet Validation Scope**: Validation is conducted on 6 Las Vegas aerial chips due to local unavailability of official challenge test labels.
2. **PEER Task 5 Scope**: The classifier detects post-disaster earthquake structural collapse modes, not cosmetic wear, roof aging, or deferred maintenance.
3. **Ames Pricing Scope**: The regression model is fitted on historical 2006–2010 transactions in Ames, Iowa, and must not be used as an appraisal tool for contemporary or nationwide transactions.
4. **Zillow Forecast Coverage**: 95% forecast interval coverage is 40.31% due to macroeconomic interest rate shifts post-2022 that deviate from the historical 2000–2021 training trend.
