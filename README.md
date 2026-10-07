# 🏢 RealtyAI: Smart Real Estate Insight Platform

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-red.svg)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B.svg)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange.svg)](https://xgboost.ai)
[![Prophet](https://img.shields.io/badge/Prophet-1.4-blueviolet.svg)](https://facebook.github.io/prophet/)
[![License](https://img.shields.io/badge/License-MIT%20%26%20Academic%20Research-green.svg)](docs/DATA_PROVENANCE.md)

An end-to-end AI platform that evaluates property conditions, predicts price trends, and segments satellite images of real estate regions for property buyers, investors, and urban planners.

---

## 🌟 Key Capabilities

### 1. 🏠 Home Buyer & Valuation Inspector
- **Instant Valuation Engine**: Estimate hedonic transaction prices using an XGBoost model trained on Ames, Iowa historical residential transaction data (held-out test MAE = $15,092.41, RMSE = $22,203.78, R² = 0.9103). Pricing uncertainty uses 5-fold out-of-fold residual calibration with a nominal 90% prediction interval (margin ±19.57% discrete / ±19.59% continuous quantile). The resulting interval achieved 92.24% empirical coverage (202/219) on the held-out test set, compared with the nominal 90% target.
- **Visual Structural Integrity Assessment**: ResNet-18 classification of PEER Task 5 post-disaster structural collapse modes (`non_collapse`, `partial_collapse`, `global_collapse`). Official benchmark test accuracy = 70.55% and macro F1 = 0.6721 on 146 images.

### 2. 💼 Real Estate Investor Market Intelligence
- **36-Month Price Forecasting**: Prophet-based monthly ZHVI forecasting for 10 major MSAs. Arithmetic-mean held-out test MAE = $39,580.75, RMSE = $45,888.30, MAPE = 8.71%. The nominal 95% forecast interval achieved 40.31% empirical coverage during the 2024-01 through 2026-08 test period.
- **Investment Return Simulator**: Interactive ROI calculator simulating cash-on-cash return, terminal sales value, and net rental income.
- **National Valuation Map**: Folium interactive map showing price tiers and appreciation rates across major US metros.

### 3. 🌆 Urban Planner & Geospatial AI Hub
- **Satellite Footprint Segmentation**: U-Net building-footprint segmentation evaluated on six held-out validation chips from a limited Las Vegas research subset. Mean IoU = 0.3593 and mean Dice = 0.4985 (official public test labels unavailable locally).
- **Zoning & Density Analytics**: Real-time extraction of building coverage % and non-building ground surface % (unpaved dirt, roads, driveways; multi-spectral vegetation indices required for green space).

### 4. ⚙️ Model Hub & AI Diagnostics
- Live evaluation matrix tracking **IoU, Dice Score, Accuracy, Precision, Recall, MAE, RMSE, and MAPE** directly ingested from saved JSON artifacts with zero hardcoded values.
- Interactive confusion matrix, training loss curves, and feature importance bar charts.

---

## 📁 Repository Structure

```
realityai/
├── README.md                           # Master project documentation
├── requirements.txt                    # Python dependencies
├── data/
│   ├── raw/                            # Ingested datasets (Kaggle Housing, Zillow, SpaceNet)
│   ├── processed/                      # Preprocessed arrays, feature metadata, splits
│   └── sample_images/                  # Demo property photos and test satellite tiles
├── models/
│   ├── saved/                          # Saved model checkpoints (.pt, .pkl, .json)
│   ├── segmentation_unet.py            # PyTorch U-Net architecture for satellite segmentation
│   ├── condition_resnet.py             # ResNet property condition classifier
│   ├── price_regressor.py              # XGBoost & LightGBM pricing regressors
│   └── trend_forecaster.py             # Prophet regional time-series forecaster
├── pipelines/
│   ├── 01_data_ingestion.py            # Downloads/prepares Kaggle, Zillow, and SpaceNet data
│   ├── 02_data_preprocessing.py        # Tabular imputation, scaling, and train/val/test splits
│   ├── 03_train_segmentation.py        # U-Net satellite segmentation training
│   ├── 04_train_condition_cnn.py       # ResNet property condition training
│   ├── 05_train_price_regressor.py     # XGBoost and LightGBM model training
│   ├── 06_train_forecaster.py          # Regional Prophet trend model training
│   └── 07_evaluate_all.py              # Unified evaluation aggregator
├── app/
│   ├── main.py                         # Streamlit application entry point
│   ├── config.py                       # Glassmorphic CSS styling & theme settings
│   ├── components/                     # Header, navigation, Plotly charts, Folium map
│   └── views/                          # Buyer, Investor, Urban Planner, and Model views
├── tests/                              # Automated pytest unit test suite
└── docs/
    ├── REPORT.md                       # Comprehensive final project report
    ├── MILESTONES.md                   # 8-week milestone deliverables tracking
    └── ARCHITECTURE.md                 # System architecture and data flow diagrams
```

---

## 🚀 Quick Start Guide

### 1. Installation
Ensure Python 3.10+ is installed, then install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Pipelines (Optional — Models Pre-trained)
To re-run any pipeline from scratch:
```bash
# Data Acquisition & Cleaning (Milestone I)
python pipelines/01_data_ingestion.py
python pipelines/02_data_preprocessing.py

# Model Training (Milestones II & III)
python pipelines/03_train_segmentation.py
python pipelines/04_train_condition_cnn.py
python pipelines/05_train_price_regressor.py
python pipelines/06_train_forecaster.py

# Unified Evaluation (Milestone IV)
python pipelines/07_evaluate_all.py
```

### 3. Launch the Interactive Dashboard
Launch the multi-persona Streamlit platform:
```bash
streamlit run app/main.py
```
Open your browser to `http://localhost:8501`.

---

## 🧪 Automated Testing
Run the comprehensive unit test suite:
```bash
python -m pytest tests/ -v
```

---

## 📊 Evaluation Summary

| Module | Canonical Checkpoint | Evaluation Split | Target Metric | Verified Empirical Result |
| :--- | :--- | :--- | :--- | :--- |
| **Satellite Building Footprint** | `unet_spacenet_v1.pt` | SpaceNet Val ($N=6$ chips) | Mean IoU / Dice | **35.93% IoU / 49.85% Dice** |
| **Structural Collapse Mode** | `resnet_peer_collapse_v1.pt` | PEER Benchmark Test ($N=146$) | Accuracy / Macro F1 | **70.55% Acc / 0.6721 Macro F1** |
| **Ames Price Regression** | `xgboost_ames_v1.pkl` | Ames Held-Out Test ($N=219$) | MAE / $R^2$ / Coverage | **$15,092.41 MAE / 0.9103 $R^2$ / 92.24% Test Cov. (±19.57%)** |
| **Regional ZHVI Forecasting** | `forecaster_summary.json` | 10 MSAs Held-Out Test (2024–2026) | Mean MAE / MAPE / Cov. | **$39,580.75 MAE / 8.71% MAPE / 40.31% 95% Forecast Cov.** |

---

## 📜 Documentation
- 📄 [Final Project Report](docs/REPORT.md)
- 🗺️ [System Architecture & Diagrams](docs/ARCHITECTURE.md)
- 📅 [8-Week Milestone Deliverables Matrix](docs/MILESTONES.md)
