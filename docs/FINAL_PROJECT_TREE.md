# REALTYAI2 — FINAL PROJECT TREE & COMPONENT DIRECTORY

**Status**: Research-Frozen Academic Architecture  
**Date**: October 2, 2026  
**Auditor**: RealtyAI Research Validation Committee  

---

```
realityai/
├── README.md                                    # Master project README & research summary
├── PROJECT_SUMMARY_AND_USER_GUIDE.md            # Comprehensive user guide & operational manual
├── RealtyAI_Executive_Project_Report.pdf        # Compiled professional executive PDF report
├── RealtyAI_Project_Summary_Report.pdf          # Summary PDF release artifact
├── generate_pdf_report.py                       # Automated ReportLab PDF generator
├── requirements.txt                             # Pinned Python package dependencies
├── .gitignore                                   # Git ignore rules (caches, venvs, raw npy arrays)
├── LICENSE                                      # MIT codebase license & third-party dataset notices
│
├── app/                                         # Streamlit multi-persona web application
│   ├── main.py                                  # Application entry point & persona navigation
│   ├── config.py                                # UI theme constants, color palettes, paths
│   ├── components/                              # Reusable UI presentation widgets
│   │   ├── charts.py                            # Interactive Plotly chart builders
│   │   ├── header.py                            # Header & system diagnostic status banner
│   │   └── map_view.py                          # Folium geospatial national map builder
│   └── views/                                   # Domain-specific persona interfaces
│       ├── buyer_view.py                        # Buyer valuation & PEER collapse inspector
│       ├── investor_view.py                     # Regional 36-month ZHVI forecaster & ROI simulator
│       ├── urban_planner_view.py                # SpaceNet building segmentation & density analytics
│       └── model_metrics_view.py                # Authoritative metrics scorecard & provenance matrix
│
├── data/                                        # Data assets & split manifests
│   ├── manifests/                               # Manifest CSVs with filepaths & metadata
│   │   ├── property_condition_manifest.csv      # Complete 1,372-image PEER dataset manifest
│   │   ├── spacenet_manifest.csv                # Complete SpaceNet dataset manifest
│   │   ├── spacenet_smoke_test_manifest.csv     # Isolated 20-chip smoke test manifest
│   │   └── spacenet_train_manifest.csv          # 20-chip research development manifest
│   ├── processed/                               # Preprocessed tabular arrays & split JSONs
│   │   ├── property_condition_splits.json       # PEER splits (1042 train / 184 val / 146 test)
│   │   ├── spacenet_splits.json                 # SpaceNet research splits (14 train / 6 val)
│   │   ├── spacenet_smoke_test_splits.json      # Isolated smoke test splits (10 train / 5 val / 5 test)
│   │   ├── spacenet_train_splits.json           # SpaceNet training splits
│   │   ├── housing_train_df.csv                 # Ames train split (N=1,020 properties)
│   │   ├── housing_val_df.csv                   # Ames validation split (N=219 properties)
│   │   ├── housing_test_df.csv                  # Ames held-out test split (N=219 properties)
│   │   ├── housing_preprocessor.pkl             # Fitted tabular preprocessor pipeline
│   │   ├── housing_features.json                # 27-feature contract metadata
│   │   ├── zillow_zhvi_processed.csv            # Unpivoted historical monthly ZHVI time series
│   │   ├── zillow_metro_summary.csv             # Summary statistics across top metropolitan areas
│   │   ├── forecasts_cache.csv                  # Precomputed 36-month Prophet forecasts
│   │   ├── master_evaluation_metrics.json       # Consolidated benchmark evaluation metrics
│   │   ├── eda_summary.json                     # Initial exploratory data analysis summary
│   │   └── charts/                              # Pre-rendered training curves & error charts
│   ├── raw/                                     # Raw authentic benchmark datasets
│   │   ├── housing/                             # Ames Housing raw dataset (train.csv, test.csv)
│   │   ├── property_conditions/                 # PEER Task 5 structural collapse RGB images
│   │   ├── spacenet/                            # SpaceNet 2 Las Vegas aerial tiles & masks
│   │   └── zillow/                              # Raw Zillow Research monthly ZHVI CSV
│   ├── sample_images/                           # Preset demonstration imagery for live dashboard
│   │   ├── demo_satellite_chip.png              # SpaceNet sample aerial tile
│   │   ├── demo_satellite_mask.png              # SpaceNet sample ground-truth mask
│   │   ├── demo_structural_non_collapse.jpg     # Authentic PEER non_collapse sample
│   │   ├── demo_structural_partial_collapse.jpg # Authentic PEER partial_collapse sample
│   │   └── demo_structural_global_collapse.jpg  # Authentic PEER global_collapse sample
│   └── synthetic_backup/                        # Quarantined legacy synthetic imagery (Archival)
│
├── docs/                                        # Comprehensive research & audit documentation
│   ├── ARCHITECTURE.md                          # Technical system architecture & data contracts
│   ├── DATA_PROVENANCE.md                       # Comprehensive dataset provenance & licensing
│   ├── DECISIONS.md                             # Architectural Decision Records (ADRs)
│   ├── EXPERIMENT_LOG.md                        # Master experiment ledger (Exps 1–7)
│   ├── IMPLEMENTATION_STATUS.md                 # Staged milestone implementation tracker
│   ├── MILESTONES.md                            # 8-week milestone deliverables matrix
│   ├── MODEL_CARD.md                            # Detailed model cards for all canonical models
│   ├── PHASE0_AUDIT.md                          # Baseline project inspection & defect audit
│   ├── PHASE4_PRECHECK.md                       # Pre-integration audit & verification checklist
│   ├── PHASE4_FINAL_AUDIT.md                    # Dashboard integration audit report
│   ├── PHASE5_METRIC_RECONCILIATION.md          # Exhaustive reconciliation of all discrepancies
│   ├── PHASE5_FINAL_AUDIT.md                    # Master Phase 5 research readiness audit
│   ├── FINAL_PROJECT_TREE.md                    # Final project structure & inventory (this file)
│   ├── GITHUB_PREP_AUDIT.md                     # GitHub pre-push audit & verification report
│   ├── PROJECT_HYGIENE_REPORT.md                # Hygiene audit & file cleanup documentation
│   ├── REPORT.md                                # Comprehensive academic & technical report
│   └── VALIDATION_LOG.md                        # Incremental validation & verification log
│
├── models/                                      # Model architecture code & canonical weights
│   ├── segmentation_unet.py                     # PyTorch U-Net & BCEDiceLoss implementation
│   ├── condition_resnet.py                      # ResNet-18 PEER structural collapse classifier
│   ├── price_regressor.py                       # XGBoost/LightGBM regressors & uncertainty calibration
│   ├── pricing_feature_contract.py              # 27-feature schema contract & validation guards
│   ├── trend_forecaster.py                      # Facebook Prophet regional time-series forecaster
│   ├── saved/                                   # Canonical model artifacts & authoritative metrics
│   │   ├── unet_spacenet_v1.pt                  # Canonical U-Net checkpoint (SHA: 28bedc6a...)
│   │   ├── unet_satellite.pt                    # Historical alias (0 weight differences)
│   │   ├── resnet_peer_collapse_v1.pt           # Canonical ResNet-18 checkpoint (SHA: b5c5d063...)
│   │   ├── resnet_condition.pt                  # Historical alias (0 weight differences)
│   │   ├── xgboost_ames_v1.pkl                  # Canonical XGBoost model (SHA: 96b9ebc3...)
│   │   ├── xgboost_price.pkl                    # Byte-identical alias (same SHA-256)
│   │   ├── lightgbm_ames_v1.pkl                 # LightGBM validation comparison artifact
│   │   ├── lightgbm_price.pkl                   # LightGBM alias
│   │   ├── forecaster_summary.json              # Canonical Prophet 10-MSA metadata (SHA: 1d008e35...)
│   │   ├── unet_metrics.json                    # SpaceNet 6-chip validation metrics
│   │   ├── condition_metrics.json               # PEER 146-image benchmark test metrics
│   │   ├── price_metrics.json                   # Ames 219-property test regression & uncertainty
│   │   ├── final_verification.json              # Machine-readable independent reload verification
│   │   └── unet_error_analysis/                 # 4-panel diagnostic quad images for 6 validation chips
│   └── legacy_synthetic/                        # Quarantined legacy synthetic checkpoints (Archival)
│
├── pipelines/                                   # Leakage-safe data & training pipelines
│   ├── 01_data_ingestion.py                     # Raw dataset download & extraction pipeline
│   ├── 02_data_preprocessing.py                 # Feature scaling, imputation, and splitting
│   ├── 03_train_segmentation.py                 # SpaceNet U-Net training pipeline
│   ├── 04_train_condition_cnn.py                # PEER ResNet-18 fine-tuning pipeline
│   ├── 05_train_price_regressor.py              # XGBoost/LightGBM tuning & residual calibration
│   ├── 06_train_forecaster.py                   # Zillow Prophet multi-MSA forecasting pipeline
│   └── 07_evaluate_all.py                       # Master evaluation compilation script
│
├── scripts/                                     # Data setup & verification utilities
│   ├── final_independent_verification.py        # Independent end-to-end reload verification
│   ├── ingest_property_condition_real.py        # PEER dataset ingestion script
│   ├── ingest_spacenet_real.py                  # SpaceNet dataset ingestion script
│   ├── ingest_spacenet_research_train.py        # SpaceNet research dataset ingestion script
│   ├── reorganize_peer_official_splits.py       # PEER official split reorganization script
│   ├── setup_spacenet_smoke_test.py             # SpaceNet smoke test split setup
│   └── verify_reloaded_artifacts.py             # Phase 3 reload verification script
│
└── tests/                                       # Comprehensive automated test suite (46 tests)
    ├── test_app.py                              # Application smoke & import tests
    ├── test_dashboard_integration.py            # Dashboard view integration, safety, & contract tests
    ├── test_data_pipeline.py                    # Data integrity, manifest, and split tests
    └── test_models.py                           # Deep learning tensor, architecture, & math tests
```
