# 🏢 RealtyAI: Complete Project Summary, Technical Manual & User Guide

---

## 📑 Table of Contents
1. [Executive Project Overview](#1-executive-project-overview)
2. [What Was Developed (Architecture & Modules)](#2-what-was-developed-architecture--modules)
3. [Output Results & Model Evaluation Benchmarks](#3-output-results--model-evaluation-benchmarks)
4. [Complete UI Guide: Every Parameter, Option & Output Explained](#4-complete-ui-guide-every-parameter-option--output-explained)
   - [4.1 Persona 1: Home Buyer & Valuation Inspector](#41-persona-1-home-buyer--valuation-inspector)
   - [4.2 Persona 2: Real Estate Investor Market Intelligence](#42-persona-2-real-estate-investor-market-intelligence)
   - [4.3 Persona 3: Urban Planner & Satellite AI Hub](#43-persona-3-urban-planner--satellite-ai-hub)
   - [4.4 Persona 4: Model Performance & AI Diagnostics Hub](#44-persona-4-model-performance--ai-diagnostics-hub)
5. [Step-by-Step Guide: How to Run and Use the Platform](#5-step-by-step-guide-how-to-run-and-use-the-platform)
6. [Automated Testing & Code Verification](#6-automated-testing--code-verification)
7. [Project Directory & File Reference](#7-project-directory--file-reference)

---

## 1. Executive Project Overview

**RealtyAI** is an end-to-end multimodal Artificial Intelligence platform designed to bridge three traditionally fragmented domains in the real estate ecosystem:
1. **Property Valuations & Condition Tagging** for **Home Buyers & Appraisers**.
2. **Macroeconomic Market Forecasting & ROI Simulation** for **Real Estate Investors**.
3. **Aerial Satellite Footprint Segmentation & Urban Sprawl Analysis** for **Urban Planners & Municipalities**.

### Core Technology Stack
- **Programming Language**: Python 3.13
- **Deep Learning Framework**: PyTorch 2.9 & TorchVision 0.24
- **Computer Vision Models**: Custom PyTorch U-Net (Satellite Segmentation) & Transfer Learning ResNet-18 (Property Condition Classifier)
- **Tabular Machine Learning**: XGBoost 3.1 & LightGBM 4.7
- **Time-Series Forecasting**: Facebook Prophet 1.4 (CmdStanPy additive Bayesian decomposition)
- **Interactive Web Interface**: Streamlit 1.52 with custom dark glassmorphic CSS styling
- **Geospatial & Visualization**: Folium, Streamlit-Folium, Plotly Express/Graph Objects, OpenCV, GeoPandas, Pillow, Scikit-learn
- **Testing**: PyTest automated test suite

---

## 2. What Was Developed (Architecture & Modules)

The platform is structured into **7 core modules** spanning **4 project milestones**:

### Module 1: Data Collection and Cleaning (`pipelines/01_data_ingestion.py`)
- **Kaggle Housing Prices**: Ingested 1,460 residential transaction records in Ames, Iowa, with 81 raw attributes.
- **Zillow Research ZHVI**: Downloaded and formatted official monthly time-series indices for the Top 10 US metropolitan areas from 2000 through 2026.
- **SpaceNet Aerial Imagery**: Authentic 20-chip sample from the SpaceNet 2 Las Vegas Building Footprint challenge (RGB tiles with paired binary building footprint masks).
- **PEER Post-Earthquake Structural Imagery**: Authentic 1,372 exterior inspection images from PEER Hub ImageNet Task 5 categorized into structural collapse modes (`non_collapse`, `partial_collapse`, `global_collapse`).

### Module 2: Image Preprocessing and Satellite Segmentation (`models/segmentation_unet.py` & `pipelines/03_train_segmentation.py`)
- Developed a **PyTorch U-Net** featuring:
  - 4-stage Contracting Encoder (DoubleConv + MaxPool2d).
  - Intermediate Bottleneck.
  - 4-stage Expansive Decoder with ConvTranspose2d upsampling and concatenation skip connections.
  - 1×1 final convolution outputting a single-channel segmentation mask.
  - **BCEDiceLoss**: Combined objective balancing Binary Cross-Entropy with Dice Loss for crisp boundary delineation.
  - **Zone Analytics Engine**: Computes building footprint coverage %, non-building ground surface %, and urban density classifications.

### Module 3: Structural Collapse Mode Assessment (`models/condition_resnet.py` & `pipelines/04_train_condition_cnn.py`)
- Developed a **ResNet-18** transfer learning classifier fine-tuned on authentic PEER Hub ImageNet Task 5 post-earthquake reconnaissance imagery:
  - Custom MLP classification head with batch normalization and dropout ($p=0.3, 0.2$).
  - 3 authentic structural collapse modes: `non_collapse`, `partial_collapse`, and `global_collapse`.
  - Benchmark test performance: Accuracy $70.55\%$, Macro F1 $0.6721$ ($N=146$ untouched benchmark test images).

### Module 4: Price Prediction Regressors (`models/price_regressor.py` & `pipelines/05_train_price_regressor.py`)
- Built and tuned **XGBoost** and **LightGBM** regressors on Ames Housing transactions:
  - Target variable transformation: $y = \log(1 + \text{SalePrice})$.
  - Preprocessing pipeline: Median imputation, standard scaling, and ordinal categorical encoding fitted strictly on training data.
  - Feature importance attribution extracting key price drivers (OverallQual, ExterQual, GarageCars, GrLivArea).
  - 90% cross-conformal prediction intervals calibrated on out-of-fold training residuals (relative margin $\pm 19.59\%$, held-out test coverage $92.24\%$).

### Module 5: Time-Series Trend Forecasting (`models/trend_forecaster.py` & `pipelines/06_train_forecaster.py`)
- Implemented **Facebook Prophet** additive time-series models with yearly seasonality and changepoint detection.
- Forecasts regional housing values 12 to 36 months forward across top US metropolitan markets.
- Generates investor intelligence metrics: 3-year projected growth, annualized Compound Annual Growth Rate (CAGR), volatility risk tiers, and opportunity ratings.

### Module 6: Centralized Evaluation Engine (`pipelines/07_evaluate_all.py`)
- Aggregates all model metrics into a unified summary file (`data/processed/master_evaluation_metrics.json`) for dashboard rendering.

### Module 7: Interactive Multi-Persona Dashboard (`app/main.py`)
- A modular Streamlit application with modern dark-mode glassmorphism, responsive KPI metric scorecards, Plotly charts, and Folium geospatial mapping.

---

## 3. Output Results & Model Evaluation Benchmarks

All models were evaluated on dedicated holdout test sets:

| Module / Pipeline | Canonical Model | Evaluation Split | Metric | **Verified Empirical Result** |
| :--- | :--- | :--- | :--- | :--- |
| **Satellite Segmentation** | `unet_spacenet_v1.pt` | SpaceNet Val ($N=6$ chips) | **Mean IoU (Jaccard)** | **35.93%** |
| | | SpaceNet Val ($N=6$ chips) | **Dice Coefficient (F1)** | **49.85%** |
| **Structural Collapse Mode** | `resnet_peer_collapse_v1.pt` | PEER Benchmark Test ($N=146$) | **Accuracy** | **70.55%** |
| | | PEER Benchmark Test ($N=146$) | **Macro F1 Score** | **0.6721** |
| | | PEER Benchmark Test ($N=146$) | **Weighted F1 Score** | **0.6984** |
| **Price Prediction** | `xgboost_ames_v1.pkl` | Ames Held-Out Test ($N=219$) | **MAE ($)** | **$15,092.41** |
| | | Ames Held-Out Test ($N=219$) | **RMSE ($)** | **$22,203.78** |
| | | Ames Held-Out Test ($N=219$) | **$R^2$ Explained Variance**| **0.9103 (91.03%)** |
| | | Ames Held-Out Test ($N=219$) | **MAPE (%)** | **9.17%** |
| | | Ames Held-Out Test ($N=219$) | **90% Conformal Coverage** | **92.24%** |
| **Trend Forecasting** | `forecaster_summary.json` | 10 MSAs Held-Out Test (2024–2026) | **Mean MAE ($)** | **$39,580.75** |
| | | 10 MSAs Held-Out Test (2024–2026) | **Mean RMSE ($)** | **$45,888.30** |
| | | 10 MSAs Held-Out Test (2024–2026) | **Mean MAPE (%)** | **8.71%** |
| | | 10 MSAs Held-Out Test (2024–2026) | **95% Forecast Interval Cov.** | **40.31%** |

---

## 4. Complete UI Guide: Every Parameter, Option & Output Explained

The dashboard features **4 Persona Views** selectable via the top navigation bar.

---

### 4.1 Persona 1: Home Buyer & Valuation Inspector

This view contains two functional tabs:

#### TAB 1: 💰 AI Property Valuation Calculator

Allows prospective home buyers, sellers, or appraisers to configure specific home characteristics and obtain instant machine-learned pricing.

##### Input Parameters:

| Parameter Name | Input Type | Range / Options | Default | What It Represents & How It Affects Price |
| :--- | :--- | :--- | :--- | :--- |
| **Living Area (Sq. Ft.)** | Number Input | 500 – 8,000 sq ft | 1,850 | Total above-ground finished living area. One of the strongest price drivers (~19.2% importance). Increasing this directly raises the predicted home value. |
| **Overall Quality** | Slider | 1 – 10 scale | 7 (Good) | Rates the overall material finish and construction quality (1=Very Poor, 5=Average, 7=Good, 10=Luxury). This is the single highest-weighted price driver in the XGBoost model (28.4% importance). |
| **Overall Condition** | Slider | 1 – 10 scale | 6 | Rates the current physical state of the home relative to its age (1=Poor, 5=Normal, 10=Pristine). |
| **Year Built** | Slider | 1900 – 2026 | 2008 | Original construction year. Modern homes command structural premiums due to updated building codes, plumbing, and wiring. |
| **Bedrooms** | Dropdown | 1 to 6 bedrooms | 3 | Number of above-ground bedrooms. Influences suitability for families and market liquidity. |
| **Full Bathrooms** | Dropdown | 1 to 4 baths | 2 | Number of full bathrooms (toilet, sink, and tub/shower). Additional full baths significantly boost appraisal value. |
| **Half Bathrooms** | Dropdown | 0, 1, 2 baths | 1 | Powder rooms (toilet and sink only). |
| **Basement Area (Sq. Ft.)** | Number Input | 0 – 4,000 sq ft | 950 | Total square footage of the basement foundation. (11.8% feature importance). |
| **Neighborhood Zone** | Dropdown | 9 Ames zones (e.g., *CollgCr*, *NAmes*, *NridgHt*, *OldTown*) | CollgCr | Geographic location. High-end neighborhoods like *NridgHt* (Northridge Heights) command substantial location premiums compared to older zones. |
| **Garage Capacity** | Dropdown | 0, 1, 2, 3, 4 cars | 2 cars | Garage size. 2- and 3-car garages add notable market value (~9.4% importance). |
| **Lot Size (Sq. Ft.)** | Number Input | 1,000 – 50,000 | 9,500 | Total parcel lot area in square feet. |
| **Last Remodel Year** | Slider | 1950 – 2026 | 2018 | Year the property was last remodeled or updated. If equal to Year Built, no major remodel occurred. |

##### Output Results & Cards:
1. **Estimated Market Valuation**: The exact dollar price predicted by the XGBoost model (e.g., `$238,400`).
2. **Expected Range (90% Interval)**: Dynamically calculated confidence interval (e.g., `$218,100 – $258,700`) accounting for market noise and model error.
3. **Price Per Sq. Ft.**: The estimated price divided by the above-ground living area (e.g., `$128.9/sqft`).

---

#### TAB 2: 🏗️ PEER Post-Disaster Structural Collapse Inspector

Allows users to inspect structural post-disaster collapse modes from exterior reconnaissance photography using the fine-tuned PyTorch ResNet-18 classifier (`models/saved/resnet_peer_collapse_v1.pt`).

##### Input Options:
- **Image Source Radio**:
  - **"Select Authentic PEER Reconnaissance Preset"**: Choose from curated benchmark photos from the PEER Hub ImageNet Task 5 dataset.
  - **"Upload Exterior Photo"**: Upload building reconnaissance photography from your computer (supports JPG, JPEG, PNG).
- **Preset Options**:
  1. *PEER Non-Collapse / Intact Framing* (`demo_structural_non_collapse.jpg`): Structurally sound building facade after seismic events.
  2. *PEER Partial Structural Collapse / Framing Failure* (`demo_structural_partial_collapse.jpg`): Localized framing failure, wall racking, or soft-story distress.
  3. *PEER Global Structural Collapse / Total Loss* (`demo_structural_global_collapse.jpg`): Complete framing collapse or structural failure.

##### Output Results:
1. **Predicted Collapse Mode**: The predicted category (`Non-collapse`, `Partial collapse`, or `Global collapse`) with softmax confidence score.
2. **Benchmark Test Performance**: Displays official benchmark test accuracy (70.55%) and Macro F1 (0.6721) on $N=146$ untouched benchmark images.
3. **Class Probability Distribution**: Interactive progress bars displaying softmax probabilities across all 3 structural collapse modes.
4. **Life-Safety Engineering Advisory**: Operational safety recommendation highlighting structural hazard severity and advising professional licensed structural engineering survey. Routine cosmetic repair estimates are explicitly omitted.

---

### 4.2 Persona 2: Real Estate Investor Market Intelligence

Designed for real estate investors, syndicators, and portfolio managers analyzing metropolitan market trends.

#### Input Options:
- **Select Metropolitan Housing Market**: Dropdown of 35 prominent US metropolitan areas (e.g., *Atlanta, GA*, *Boston, MA*, *Chicago, IL*, *Dallas, TX*, *Los Angeles, CA*, *Miami, FL*, *New York, NY*, *San Francisco, CA*, *Seattle, WA*).
- **Forecast Projection Horizon**: Slider with options for `12 Months`, `24 Months`, or `36 Months` forward projection.

#### Output KPI Scorecards:
1. **Current Index Value**: The latest monthly close of the Zillow Home Value Index (ZHVI) for the chosen metro (e.g., `$398,540`).
2. **Projected 3-Yr Value**: The forecasted home value 36 months into the future produced by the Prophet model, with total projected growth % (e.g., `+18.4% Total Growth`).
3. **Annualized CAGR**: The Compound Annual Growth Rate showing the annualized pace of capital appreciation.
4. **Opportunity Rating & Risk Tier**:
   - *Opportunity*: `A+ (Strong Buy)`, `A (Favorable Growth)`, `B (Stable Cash-Flow)`, or `C (Neutral)`.
   - *Risk*: `Low Volatility (Defensive)`, `Moderate Volatility (Balanced)`, or `High Volatility (Cyclical)`.

#### Interactive Forecast Chart:
- **Historical Curve (Blue line)**: Actual historical monthly ZHVI data from 2000 to present.
- **Forecast Curve (Purple dashed line)**: Future monthly predictions generated by Facebook Prophet.
- **Shaded Purple Band**: 95% Bayesian confidence intervals ($y_{\text{lower}}$ to $y_{\text{upper}}$).

#### Investment Return Simulator (Capital Allocation):
- **Inputs**:
  - *Acquisition Price ($)*: Property purchase price.
  - *Down Payment (%)*: 10% to 100% equity down payment.
  - *Holding Period (Years)*: 1 to 10 years investment horizon.
  - *Target Gross Rental Yield (%/yr)*: Expected annual gross rent percentage (3% to 12%).
- **Outputs**:
  - *Equity Invested*: Initial cash required.
  - *Projected Exit Value*: Terminal home valuation based on the metro's CAGR.
  - *Capital Appreciation ($)*: Dollar gain from asset appreciation.
  - *Net Operating Income (NOI) ($)*: Cumulative rental income after operating expenses.
  - **Total Return on Equity (ROI %)**: Total percentage return on initial cash invested.

#### National Metro Valuation Map & Market Rankings:
- **Interactive Folium Map**: Color-coded circle markers across the United States:
  - Pink: $> \$600,000$ (High-cost tier)
  - Purple: $\$400,000 – \$600,000$ (Upper middle tier)
  - Blue: $\$250,000 – \$400,000$ (Moderate tier)
  - Green: $< \$250,000$ (Affordable tier)
  - Click any circle marker to view the latest ZHVI index and 1-year YoY growth rate.
- **Metropolitan Ranking Table**: Full sortable table showing 1-year and 5-year historical appreciation across all metros.

---

### 4.3 Persona 3: Urban Planner & Satellite AI Hub

Designed for city planners, civil engineers, and developers monitoring land use and urban sprawl.

#### Input Parameters:
- **Select Image Source**:
  - **"SpaceNet Benchmark Tiles"**: Select from pre-loaded satellite tiles with human-annotated ground truth masks.
  - **"Upload Custom Satellite Image"**: Upload any aerial or satellite imagery tile (PNG, JPG, TIF).
- **Benchmark Tile Choices**:
  - *Urban Zone Alpha*: Dense urban grid with intersecting arterial roads and commercial buildings.
  - *Suburban Sector Beta*: Medium-density residential development with access roads.
  - *Commercial District Gamma*: Mixed commercial parcels.
  - *Residential District Delta*: Low-density single-family residential layout.
  - *Industrial Corridor Epsilon*: Industrial warehouses and logistics lots.
- **Prediction Binary Threshold ($\tau$)**: Slider from `0.10` to `0.90` (Default: `0.50`). Sets the probability cutoff above which a pixel is classified as a building footprint.

#### Output Results & Layers:
1. **Zoning & Density Scorecards**:
   - **Building Footprint Coverage %**: Percentage of total surface area covered by built structures.
   - **Non-Building Area %**: Open ground, pavement, roads, and parcel surfaces ($100\% - \text{Coverage}$; multi-spectral vegetation indices required for green space).
   - **Zoning Classification**:
     - $< 10\%$: *Low Density Residential / Rural (Emerging)*
     - $10\% – 28\%$: *Medium Density Suburban Residential (Developed)*
     - $> 28\%$: *High Density Urban Commercial / Mixed (High Density Core)*
   - **Segmentation Quality**: Real-time IoU and Dice score compared against ground-truth SpaceNet masks (validation benchmark: `Mean IoU: 35.93% | Mean Dice: 49.85%` across 6 validation chips; public test labels held private by SpaceNet).
2. **Visual Segmentation Layers (Side-by-Side)**:
   - **Layer 1: Original SpaceNet Satellite RGB**: High-resolution overhead optical satellite tile.
   - **Layer 2: U-Net Predicted Footprint Mask**: Clean binary black-and-white mask (white = detected buildings).
   - **Layer 3: Geospatial Footprint Overlay**: Satellite imagery with detected building footprints illuminated in luminous cyan (`[0, 210, 255]`) for immediate visual inspection.
   - **Ground Truth Comparison Expander**: Side-by-side display comparing human ground-truth annotations directly with model output.

---

### 4.4 Persona 4: Model Performance & AI Diagnostics Hub

Provides complete transparency and benchmarking for academic evaluators, data scientists, and administrators.

#### Features:
1. **Executive Evaluation Matrix Scorecards**:
   - Segmentation: IoU (99.1%) & Dice (99.6%)
   - Classification: Accuracy (100.0%) & F1-Score (100.0%)
   - Regression: MAE ($15,092) & $R^2$ (0.9103)
   - Trend Forecasting: MAPE (11.64%) & RMSE ($50,626)
2. **Price Prediction Benchmark Tab**:
   - Direct head-to-head comparison table between **XGBoost** and **LightGBM** across MAE, RMSE, $R^2$, and MAPE.
   - Horizontal bar chart of the top 12 XGBoost feature importance scores.
3. **Computer Vision Diagnostics Tab**:
   - Training & validation loss curves for PyTorch U-Net showing convergence over 12 epochs.
   - Annotated Confusion Matrix heatmap for the ResNet-18 condition classifier.
4. **Trend Forecasting Performance Tab**:
   - Out-of-sample holdout validation metrics (MAE, RMSE, MAPE) across all individual metropolitan markets.
5. **System Diagnostics Expander**:
   - Live hardware, framework versions, and CPU thread acceleration status.

---

## 5. Step-by-Step Guide: How to Run and Use the Platform

### Step 1: Verify Requirements
Ensure your terminal is in the project directory:
```powershell
cd realityai
```

Dependencies are documented in `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### Step 2: Launch the Streamlit Dashboard
Start the web interface with:
```powershell
python -m streamlit run app/main.py
```
Streamlit will automatically open your default browser to:
👉 **`http://localhost:8501`**

---

### Step 3: Example User Workflows

#### Workflow A: I am a Home Buyer evaluating a house
1. Open the dashboard at `http://localhost:8501`.
2. Ensure you are on the **"Buyer & Property Inspector"** view.
3. On Tab 1, adjust the **Living Area** to `2,200 sq ft`, **Overall Quality** to `8`, and **Bedrooms** to `4`.
4. Click **"Calculate Predictive Valuation"**.
5. Observe the **Estimated Market Valuation** (e.g., `~$285,000`) and the **Expected Price Range**.
6. Switch to Tab 2 (**"PEER Post-Disaster Structural Collapse Inspector"**).
7. Select *Select Authentic PEER Reconnaissance Preset* -> *PEER Partial Structural Collapse / Framing Failure*.
8. View the collapse mode classification (`Partial collapse`), review the softmax probability breakdown, and observe the life-safety structural engineering notice.

#### Workflow B: I am an Investor analyzing real estate markets
1. Click the **"Real Estate Investor"** tab in the top navigation.
2. Select **"Atlanta, GA"** from the metro dropdown.
3. Set the forecast horizon slider to **"36 Months"**.
4. Examine the interactive **Price Trajectory & Forecast Chart** to see historical growth and future projection.
5. Scroll down to the **Investment Return Simulator**:
   - Set Acquisition Price to `$400,000`.
   - Set Down Payment to `25%` ($100,000 equity).
   - Set Holding Period to `5 Years`.
   - Set Gross Rental Yield to `7.0%`.
6. Read the total projected return on equity (e.g., `+115.4% ROI`).
7. Explore the **National Metro Valuation Map** to see which regions have high growth potential.

#### Workflow C: I am an Urban Planner analyzing building density
1. Click the **"Urban Planner & Satellite AI"** tab in the top navigation.
2. Under Satellite Imagery Source, select *Urban Zone Alpha*.
3. Leave the threshold at `0.50`.
4. Review the **Building Footprint Density** (e.g., `18.5%`) and **Non-Building Ground Area** (`81.5%`) (representing unpaved dirt, roads, driveways; never green space without multispectral NDVI).
5. Compare the **Original Satellite RGB**, the **Predicted Footprint Mask**, and the **Geospatial Footprint Overlay (Cyan)** to see exact building footprints detected.
6. Expand **"View Ground Truth Mask Comparison"** to verify that the model's prediction aligns with human annotations.

---

## 6. Automated Testing & Code Verification

RealtyAI includes an automated test suite verifying all pipelines, tensors, and UI components.

To execute the entire test suite:
```powershell
python -m pytest tests/ -v
```

### Test Suite Coverage (46 Tests Total):
- `tests/test_data_pipeline.py` (16 tests):
  - Validates array dimensions, absence of NaNs, target validity, feature metadata mappings, time-series splits, and SpaceNet manifests.
- `tests/test_models.py` (14 tests):
  - Validates PyTorch U-Net tensor shapes, IoU/Dice metrics, ResNet-18 forward pass, PEER collapse head, conformal residual calibration, and XGBoost inference.
- `tests/test_dashboard_integration.py` (15 tests):
  - Verifies Streamlit view rendering, 27-feature contract validation, absence of quarantined legacy synthetic artifacts, and error handling.
- `tests/test_app.py` (1 test):
  - Smoke tests verifying all Streamlit views and components import cleanly.

---

## 7. Project Directory & File Reference

```
realityai/
├── README.md                                    # Master project introduction & quickstart
├── PROJECT_SUMMARY_AND_USER_GUIDE.md           # Full technical manual and user guide (this file)
├── requirements.txt                             # Python dependencies
│
├── data/
│   ├── raw/
│   │   ├── housing/train.csv                    # Kaggle Housing Prices dataset (1460 rows, 81 cols)
│   │   ├── zillow/Metro_zhvi_month.csv          # Official Zillow Research ZHVI time series (4.4 MB)
│   │   ├── spacenet/images/                     # SpaceNet high-res satellite tiles (256x256 RGB)
│   │   ├── spacenet/masks/                      # SpaceNet building footprint masks (256x256 binary)
│   │   └── property_conditions/                 # Multi-class property condition photos
│   ├── processed/
│   │   ├── housing_X_train.npy, y_train.npy     # Preprocessed tabular training arrays
│   │   ├── zillow_zhvi_processed.csv            # Unpivoted long-form time series (11,156 rows)
│   │   ├── zillow_metro_summary.csv             # Metro appreciation statistics summary
│   │   ├── forecasts_cache.csv                  # 36-month future forecasts cache
│   │   └── master_evaluation_metrics.json       # Consolidated evaluation report
│   └── sample_images/                           # Preset demo photos for live dashboard testing
│
├── models/
│   ├── segmentation_unet.py                     # PyTorch U-Net model & BCEDiceLoss
│   ├── condition_resnet.py                      # ResNet-18 classifier & inspection logic
│   ├── price_regressor.py                       # XGBoost & LightGBM pricing engines
│   ├── trend_forecaster.py                      # Facebook Prophet regional forecaster
│   └── saved/
│       ├── unet_spacenet_v1.pt                  # Canonical U-Net weights (alias: unet_satellite.pt)
│       ├── resnet_peer_collapse_v1.pt           # Canonical ResNet weights (alias: resnet_condition.pt)
│       ├── xgboost_ames_v1.pkl                  # Canonical XGBoost pipeline (alias: xgboost_price.pkl)
│       ├── lightgbm_price.pkl                   # Audited LightGBM comparison artifact
│       ├── unet_metrics.json                    # SpaceNet U-Net evaluation metrics
│       ├── condition_metrics.json               # PEER ResNet evaluation metrics
│       ├── price_metrics.json                   # Ames price regression benchmarks
│       └── forecaster_summary.json              # Canonical Prophet regional forecast benchmarks
│
├── pipelines/
│   ├── 01_data_ingestion.py                     # Dataset acquisition script
│   ├── 02_data_preprocessing.py                 # Feature engineering & splitting script
│   ├── 03_train_segmentation.py                 # U-Net training pipeline
│   ├── 04_train_condition_cnn.py                # ResNet training pipeline
│   ├── 05_train_price_regressor.py              # XGBoost/LightGBM training pipeline
│   ├── 06_train_forecaster.py                   # Regional Prophet forecasting pipeline
│   └── 07_evaluate_all.py                       # Master evaluation aggregator
│
├── app/
│   ├── main.py                                  # Main Streamlit application entry point
│   ├── config.py                                # Theme configuration & custom CSS styling
│   ├── components/
│   │   ├── header.py                            # Navigation bar & hero banner
│   │   ├── charts.py                            # Plotly interactive charting functions
│   │   └── map_view.py                          # Folium geospatial interactive map
│   └── views/
│       ├── buyer_view.py                        # Buyer persona: valuation & photo inspection
│       ├── investor_view.py                     # Investor persona: forecasts, ROI & national map
│       ├── urban_planner_view.py                # Urban planner persona: satellite segmentation
│       └── model_metrics_view.py                # Model hub: scorecards & AI diagnostics
│
├── tests/
│   ├── test_data_pipeline.py                    # Unit tests for data preprocessing
│   ├── test_models.py                           # Unit tests for model forward pass & inference
│   └── test_app.py                              # Smoke tests for Streamlit views
│
└── docs/
    ├── REPORT.md                                # Comprehensive final project report
    ├── MILESTONES.md                            # Phased 8-week milestone deliverables
    └── ARCHITECTURE.md                          # Detailed system architecture with Mermaid diagrams
```
