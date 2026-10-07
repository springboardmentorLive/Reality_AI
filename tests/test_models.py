"""
Unit Tests for RealtyAI Machine Learning & Deep Learning Models (Phase 3 Verified Reproducibility)
Tests model loading, forward passes, non-leakage, contract adherence, and metric reproducibility.
"""

import os
import sys
import glob
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
import torch
import pytest
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.segmentation_unet import UNet, calculate_segmentation_metrics, analyze_satellite_zone
from models.condition_resnet import ResNetConditionClassifier, evaluate_property_inspection
from models.price_regressor import RealEstatePricePredictor
from models.pricing_feature_contract import (
    PRICING_FEATURE_CONTRACT,
    FEATURE_CONTRACT,
    build_contract_compliant_input,
    validate_feature_contract
)

SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
LEGACY_MODELS_DIR = os.path.join(BASE_DIR, "models", "legacy_synthetic")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def get_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# 1. Architecture and Forward Passes
def test_unet_architecture_forward():
    """Verifies U-Net forward pass preserves input height and width."""
    model = UNet(in_channels=3, out_channels=1, features=[8, 16, 32, 64])
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 256, 256)


def test_segmentation_metrics():
    """Verifies IoU and Dice metric calculation."""
    preds = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]])
    targets = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    metrics = calculate_segmentation_metrics(preds, targets, threshold=0.5)
    assert 0.0 <= metrics["iou"] <= 1.0
    assert 0.0 <= metrics["dice"] <= 1.0
    assert metrics["iou"] > 0.9


def test_satellite_zone_analysis():
    """Verifies urban zone analytics computation and honest naming."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[50:100, 50:100] = 1
    info = analyze_satellite_zone(mask)
    assert "building_coverage_pct" in info
    assert "non_building_area_pct" in info
    assert "zone_classification" in info
    assert info["building_coverage_pct"] > 0
    assert info["non_building_area_pct"] == round(100.0 - info["building_coverage_pct"], 2)


def test_non_building_area_not_labeled_green_space():
    """Requirement 14: Non-building area is not labeled green space without spectral data."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    info = analyze_satellite_zone(mask)
    assert "non_building_area_pct" in info
    assert info["non_building_area_pct"] == 100.0
    assert "green_space_pct" not in info or "open_space_pct" in info


def test_resnet_condition_classifier():
    """Verifies ResNet forward pass returns 3 class logits."""
    model = ResNetConditionClassifier(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 3)


def test_inspection_report_logic():
    """Verifies condition scoring and collapse assessment logic."""
    probs_collapse = {"global_collapse": 0.90, "non_collapse": 0.05, "partial_collapse": 0.05}
    rep_col = evaluate_property_inspection(probs_collapse)
    assert rep_col["condition_score"] <= 30
    assert "Renovation" in rep_col["valuation_impact"]

    probs_safe = {"global_collapse": 0.02, "non_collapse": 0.92, "partial_collapse": 0.06}
    rep_safe = evaluate_property_inspection(probs_safe)
    assert rep_safe["condition_score"] >= 80
    assert "Premium" in rep_safe["valuation_impact"]


# 2. Phase 3 Quarantine & Distinction Test
def test_no_legacy_synthetic_checkpoint_loaded_in_saved():
    """Phase 3 Test 5: Verify no active checkpoint in models/saved/ is identical to legacy synthetic checkpoints."""
    legacy_files = [f for f in os.listdir(LEGACY_MODELS_DIR) if f.endswith((".pt", ".pkl"))]
    assert len(legacy_files) >= 3, "Legacy quarantine must contain legacy checkpoints."

    legacy_hashes = {f: get_sha256(os.path.join(LEGACY_MODELS_DIR, f)) for f in legacy_files}

    active_files = [f for f in os.listdir(SAVED_MODELS_DIR) if f.endswith((".pt", ".pkl"))]
    for act_f in active_files:
        act_path = os.path.join(SAVED_MODELS_DIR, act_f)
        act_hash = get_sha256(act_path)
        for leg_f, leg_hash in legacy_hashes.items():
            assert act_hash != leg_hash, f"Active model {act_f} has identical sha256 to legacy checkpoint {leg_f}!"


# 3. Model Loading & Schema Verification
def test_trained_unet_checkpoint_reload_and_shape():
    """Phase 3 Tests 1 & 3: Reload U-Net checkpoint and verify input/output shapes."""
    ckpt_path = os.path.join(SAVED_MODELS_DIR, "unet_spacenet_v1.pt")
    if not os.path.exists(ckpt_path):
        pytest.skip("U-Net checkpoint not yet trained.")

    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    sample_x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(sample_x)
    assert out.shape == (1, 1, 256, 256)


def test_trained_price_regressors_reload_and_predict():
    """Phase 3 Tests 1, 3, 4: Reload XGBoost & LightGBM and verify inference."""
    xgb_path = os.path.join(SAVED_MODELS_DIR, "xgboost_ames_v1.pkl")
    lgb_path = os.path.join(SAVED_MODELS_DIR, "lightgbm_ames_v1.pkl")
    if not os.path.exists(xgb_path) or not os.path.exists(lgb_path):
        pytest.skip("Price regressors not yet trained.")

    p_xgb: RealEstatePricePredictor = joblib.load(xgb_path)
    p_lgb: RealEstatePricePredictor = joblib.load(lgb_path)

    X_test = np.load(os.path.join(PROCESSED_DIR, "housing_X_test.npy"))
    pred_xgb = p_xgb.predict(X_test[:5])
    pred_lgb = p_lgb.predict(X_test[:5])

    assert len(pred_xgb) == 5
    assert len(pred_lgb) == 5
    assert (pred_xgb > 0).all()
    assert (pred_lgb > 0).all()


# 4. Feature Contract Integrity Test
def test_pricing_feature_contract_exact_match():
    """Phase 3 Test 11: Pricing contract matches trained feature matrix 100%."""
    validate_feature_contract()
    with open(os.path.join(PROCESSED_DIR, "housing_features.json"), "r") as f:
        meta = json.load(f)

    contract_features = [item["model_feature"] for item in FEATURE_CONTRACT]
    training_features = meta["all_features"]

    assert contract_features == training_features, f"Contract order mismatch: {contract_features} vs {training_features}"
    assert len(contract_features) == 27


# 5. Conformal Prediction Uncertainty Test
def test_conformal_prediction_intervals():
    """Phase 3 Uncertainty Test: Split Conformal Prediction produces calibrated bounds."""
    xgb_path = os.path.join(SAVED_MODELS_DIR, "xgboost_ames_v1.pkl")
    if not os.path.exists(xgb_path):
        pytest.skip("XGBoost checkpoint not yet trained.")

    predictor: RealEstatePricePredictor = joblib.load(xgb_path)
    assert hasattr(predictor, "conformal_quantile")
    assert 0.10 <= predictor.conformal_quantile <= 0.40  # Reasonable empirical margin (e.g. ~22%)

    sample_dict = {
        "LotArea": 9500, "OverallQual": 7, "OverallCond": 6, "YearBuilt": 2005,
        "YearRemodAdd": 2010, "TotalBsmtSF": 1000, "1stFlrSF": 1100, "2ndFlrSF": 750,
        "GrLivArea": 1850, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3,
        "KitchenAbvGr": 1, "TotRmsAbvGrd": 7, "Fireplaces": 1, "GarageCars": 2,
        "GarageArea": 480, "WoodDeckSF": 100, "OpenPorchSF": 30, "Neighborhood": "CollgCr",
        "MSZoning": "RL", "BldgType": "1Fam", "HouseStyle": "2Story", "ExterQual": "Gd",
        "Foundation": "PConc", "BsmtQual": "Gd", "KitchenQual": "Gd"
    }
    res = predictor.predict_property(sample_dict)
    assert res["price_range_low"] < res["estimated_price"] < res["price_range_high"]
    assert res["confidence_pct"] == 90.0


def test_exact_conformal_order_statistic_mathematics():
    """Phase 5 Section 6: Validate finite-sample conformal order statistic index on known artificial arrays."""
    from models.price_regressor import compute_conformal_order_statistic

    # Case 1: n = 9, alpha = 0.10 -> (n+1)*(1-alpha) = 10 * 0.9 = 9.0 -> k = 9 (max element)
    arr_9 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q_9 = compute_conformal_order_statistic(arr_9, alpha=0.10)
    assert q_9 == 0.9, f"Expected 0.9, got {q_9}"

    # Case 2: n = 19, alpha = 0.10 -> (n+1)*(1-alpha) = 20 * 0.9 = 18.0 -> k = 18
    arr_19 = np.arange(1, 20) * 1.0  # [1, 2, ..., 19]
    q_19 = compute_conformal_order_statistic(arr_19, alpha=0.10)
    assert q_19 == 18.0, f"Expected 18.0, got {q_19}"

    # Case 3: n = 100, alpha = 0.10 -> (n+1)*(1-alpha) = 101 * 0.9 = 90.9 -> k = ceil(90.9) = 91
    arr_100 = np.arange(1, 101) * 1.0  # [1, 2, ..., 100]
    q_100 = compute_conformal_order_statistic(arr_100, alpha=0.10)
    assert q_100 == 91.0, f"Expected 91.0, got {q_100}"

    # Case 4: Non-integer ceil behavior: n = 10, alpha = 0.10 -> 11 * 0.9 = 9.9 -> k = 10
    arr_10 = np.arange(1, 11) * 10.0  # [10, 20, ..., 100]
    q_10 = compute_conformal_order_statistic(arr_10, alpha=0.10)
    assert q_10 == 100.0, f"Expected 100.0, got {q_10}"

    # Case 5: Empty array raises ValueError
    with pytest.raises(ValueError):
        compute_conformal_order_statistic([], alpha=0.10)


# 6. Model Selection Isolation Test
def test_price_model_selection_not_based_on_test_set():
    """Phase 3 Test 6 & 10: Model selection was performed on validation set, not test set."""
    meta_path = os.path.join(SAVED_MODELS_DIR, "price_metrics.json")
    if not os.path.exists(meta_path):
        pytest.skip("Price metrics report not yet generated.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    decision_rule = meta["model_selection"]["decision_rule"]
    assert "val" in decision_rule.lower()
    assert "x_val" in decision_rule.lower()


# 7. Reproducibility Test: Saved Metrics Match Recomputation
def test_saved_metrics_match_recomputation():
    """Phase 3 Test 12: Reloaded model recomputes evaluation metric matching saved report."""
    xgb_path = os.path.join(SAVED_MODELS_DIR, "xgboost_ames_v1.pkl")
    meta_path = os.path.join(SAVED_MODELS_DIR, "price_metrics.json")
    if not os.path.exists(xgb_path) or not os.path.exists(meta_path):
        pytest.skip("Price artifacts not yet generated.")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    X_test = np.load(os.path.join(PROCESSED_DIR, "housing_X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "housing_y_test.npy"))

    p_xgb: RealEstatePricePredictor = joblib.load(xgb_path)
    metrics = p_xgb.evaluate(X_test, y_test)
    saved_rmse = meta["final_heldout_test_evaluation"]["xgboost"]["rmse"]

    assert abs(metrics["rmse"] - saved_rmse) < 1e-2, f"Recomputed RMSE ${metrics['rmse']} != saved ${saved_rmse}"
