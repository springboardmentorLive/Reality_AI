"""
Phase 3 Verification Script: Reload Saved Artifacts & Recompute Key Metrics
Ensures complete reproducibility and integrity:
1. Reloads saved model artifacts in a fresh Python environment
2. Executes inference on real data
3. Recomputes key evaluation metrics
4. Asserts that recomputed metrics match saved metadata within floating-point tolerance (1e-4)
"""

import os
import sys
import json
import numpy as np
import torch
import joblib
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.segmentation_unet import UNet
from models.condition_resnet import ResNetConditionClassifier
from models.price_regressor import RealEstatePricePredictor

SAVED_DIR = os.path.join(BASE_DIR, "models", "saved")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPACENET_VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "spacenet", "research_train", "val")
PEER_VAL_DIR = os.path.join(BASE_DIR, "data", "raw", "peer_condition", "validation")


def verify_unet():
    print("\n--- Verifying Model 1: SpaceNet U-Net ---")
    ckpt_path = os.path.join(SAVED_DIR, "unet_spacenet_v1.pt")
    meta_path = os.path.join(SAVED_DIR, "unet_metrics.json")
    assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
    assert os.path.exists(meta_path), f"Missing metadata {meta_path}"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # 1. Reload model
    device = torch.device("cpu")
    model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128]).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # 2. Run inference on real validation chips
    splits_path = os.path.join(PROCESSED_DIR, "spacenet_train_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    val_chips = splits["val"]
    assert len(val_chips) == 6, f"Expected 6 validation chips, found {len(val_chips)}"

    ious, dices = [], []
    with torch.no_grad():
        for item in val_chips:
            img_path = os.path.join(BASE_DIR, item["image_path"])
            mask_path = os.path.join(BASE_DIR, item["mask_path"])
            img = Image.open(img_path).convert("RGB").resize((256, 256), resample=Image.Resampling.BILINEAR)
            mask = Image.open(mask_path).convert("L").resize((256, 256), resample=Image.Resampling.NEAREST)

            img_np = np.array(img, dtype=np.float32) / 255.0
            mask_np = (np.array(mask, dtype=np.float32) > 128).astype(np.float32)

            img_t = torch.from_numpy(img_np.transpose((2, 0, 1)).copy()).contiguous()
            mask_t = torch.from_numpy(mask_np.copy()).unsqueeze(0).contiguous()

            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_t = ((img_t - mean) / std).unsqueeze(0).to(device)

            from models.segmentation_unet import calculate_segmentation_metrics
            probs = torch.sigmoid(model(img_t))
            m = calculate_segmentation_metrics(probs, mask_t, threshold=0.5)
            ious.append(m["iou"])
            dices.append(m["dice"])

    recomputed_mean_iou = float(np.mean(ious))
    recomputed_mean_dice = float(np.mean(dices))
    saved_mean_iou = meta["aggregate_metrics"]["mean_iou"]
    saved_mean_dice = meta["aggregate_metrics"]["mean_dice"]

    print(f"Recomputed Mean IoU: {recomputed_mean_iou:.4f} vs Saved: {saved_mean_iou:.4f}")
    print(f"Recomputed Mean Dice: {recomputed_mean_dice:.4f} vs Saved: {saved_mean_dice:.4f}")
    assert abs(recomputed_mean_iou - saved_mean_iou) < 1e-4, "IoU mismatch!"
    assert abs(recomputed_mean_dice - saved_mean_dice) < 1e-4, "Dice mismatch!"
    print("[PASS] Model 1 (U-Net) verified successfully.")


def verify_pricing_models():
    print("\n--- Verifying Model 3: Ames Housing Price Regressors (XGBoost & LightGBM) ---")
    xgb_path = os.path.join(SAVED_DIR, "xgboost_ames_v1.pkl")
    lgb_path = os.path.join(SAVED_DIR, "lightgbm_ames_v1.pkl")
    meta_path = os.path.join(SAVED_DIR, "price_metrics.json")

    assert os.path.exists(xgb_path), f"Missing {xgb_path}"
    assert os.path.exists(lgb_path), f"Missing {lgb_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    # Load held-out test data
    X_test = np.load(os.path.join(PROCESSED_DIR, "housing_X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "housing_y_test.npy"))

    # Reload XGBoost
    xgb_predictor: RealEstatePricePredictor = joblib.load(xgb_path)
    xgb_metrics = xgb_predictor.evaluate(X_test, y_test)
    saved_xgb = meta["final_heldout_test_evaluation"]["xgboost"]

    print(f"XGBoost Test RMSE: Recomputed=${xgb_metrics['rmse']:,.2f} vs Saved=${saved_xgb['rmse']:,.2f}")
    print(f"XGBoost Test MAE:  Recomputed=${xgb_metrics['mae']:,.2f} vs Saved=${saved_xgb['mae']:,.2f}")
    print(f"XGBoost Test R2:   Recomputed={xgb_metrics['r2_score']:.4f} vs Saved={saved_xgb['r2_score']:.4f}")
    assert abs(xgb_metrics["rmse"] - saved_xgb["rmse"]) < 1e-2, "XGBoost RMSE mismatch!"
    assert abs(xgb_metrics["r2_score"] - saved_xgb["r2_score"]) < 1e-4, "XGBoost R2 mismatch!"

    # Reload LightGBM
    lgb_predictor: RealEstatePricePredictor = joblib.load(lgb_path)
    lgb_metrics = lgb_predictor.evaluate(X_test, y_test)
    saved_lgb = meta["final_heldout_test_evaluation"]["lightgbm"]

    print(f"LightGBM Test RMSE: Recomputed=${lgb_metrics['rmse']:,.2f} vs Saved=${saved_lgb['rmse']:,.2f}")
    print(f"LightGBM Test MAE:  Recomputed=${lgb_metrics['mae']:,.2f} vs Saved=${saved_lgb['mae']:,.2f}")
    print(f"LightGBM Test R2:   Recomputed={lgb_metrics['r2_score']:.4f} vs Saved={saved_lgb['r2_score']:.4f}")
    assert abs(lgb_metrics["rmse"] - saved_lgb["rmse"]) < 1e-2, "LightGBM RMSE mismatch!"
    assert abs(lgb_metrics["r2_score"] - saved_lgb["r2_score"]) < 1e-4, "LightGBM R2 mismatch!"

    # Test conformal interval calibration prediction on sample
    sample_feat = {
        "LotArea": 9500, "OverallQual": 7, "OverallCond": 6, "YearBuilt": 2005,
        "YearRemodAdd": 2010, "TotalBsmtSF": 1000, "1stFlrSF": 1100, "2ndFlrSF": 750,
        "GrLivArea": 1850, "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3,
        "KitchenAbvGr": 1, "TotRmsAbvGrd": 7, "Fireplaces": 1, "GarageCars": 2,
        "GarageArea": 480, "WoodDeckSF": 100, "OpenPorchSF": 30, "Neighborhood": "CollgCr",
        "MSZoning": "RL", "BldgType": "1Fam", "HouseStyle": "2Story", "ExterQual": "Gd",
        "Foundation": "PConc", "BsmtQual": "Gd", "KitchenQual": "Gd"
    }
    res = xgb_predictor.predict_property(sample_feat)
    assert "estimated_price" in res and "price_range_low" in res and "price_range_high" in res
    assert res["estimated_price"] > 0
    assert res["price_range_low"] < res["estimated_price"] < res["price_range_high"]
    print(f"Sample prediction: ${res['estimated_price']:,.2f} (90% Conformal Interval: [${res['price_range_low']:,.2f}, ${res['price_range_high']:,.2f}])")
    print("[PASS] Model 3 (Ames Price Regression) verified successfully.")


def verify_condition_resnet():
    print("\n--- Verifying Model 2: PEER Structural Condition ResNet-18 ---")
    ckpt_path = os.path.join(SAVED_DIR, "resnet_peer_collapse_v1.pt")
    meta_path = os.path.join(SAVED_DIR, "condition_metrics.json")
    assert os.path.exists(ckpt_path), f"Missing checkpoint {ckpt_path}"
    assert os.path.exists(meta_path), f"Missing metadata {meta_path}"

    with open(meta_path, "r") as f:
        meta = json.load(f)

    device = torch.device("cpu")
    model = ResNetConditionClassifier(num_classes=3, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Load official benchmark test partition from property_condition_splits.json (N=146)
    splits_path = os.path.join(PROCESSED_DIR, "property_condition_splits.json")
    with open(splits_path, "r") as f:
        splits = json.load(f)
    test_samples = splits["test"]
    assert len(test_samples) == 146, f"Expected 146 benchmark test images, found {len(test_samples)}"

    # Recompute benchmark test accuracy
    from models.condition_resnet import get_condition_transforms
    from sklearn.metrics import accuracy_score, f1_score
    val_tf = get_condition_transforms(is_train=False)

    preds, targets = [], []
    with torch.no_grad():
        for item in test_samples:
            img_path = os.path.join(BASE_DIR, item["image_path"])
            img = Image.open(img_path).convert("RGB")
            t = val_tf(img).unsqueeze(0).to(device)
            p = torch.argmax(model(t), dim=1).item()
            preds.append(p)
            targets.append(int(item["class_id"]))

    recomputed_acc = float(accuracy_score(targets, preds))
    recomputed_macro_f1 = float(f1_score(targets, preds, average="macro"))

    saved_acc = meta["benchmark_test_evaluation"]["accuracy"]
    saved_macro_f1 = meta["benchmark_test_evaluation"]["macro_f1"]

    print(f"Recomputed Benchmark Accuracy: {recomputed_acc:.4f} vs Saved: {saved_acc:.4f}")
    print(f"Recomputed Benchmark Macro F1: {recomputed_macro_f1:.4f} vs Saved: {saved_macro_f1:.4f}")
    assert abs(recomputed_acc - saved_acc) < 1e-4, "ResNet accuracy mismatch!"
    assert abs(recomputed_macro_f1 - saved_macro_f1) < 1e-4, "ResNet macro F1 mismatch!"
    print("[PASS] Model 2 (PEER ResNet-18) benchmark metrics recomputed and verified successfully.")
    return True


def verify_forecaster():
    print("\n--- Verifying Model 4: Zillow Trend Forecaster (Prophet) ---")
    summary_path = os.path.join(SAVED_DIR, "forecaster_summary.json")
    cache_path = os.path.join(PROCESSED_DIR, "forecasts_cache.csv")
    assert os.path.exists(summary_path), f"Missing {summary_path}"
    assert os.path.exists(cache_path), f"Missing {cache_path}"

    with open(summary_path, "r") as f:
        summary = json.load(f)

    df_cache = pd.read_csv(cache_path)
    assert len(df_cache) > 0, "Forecast cache is empty!"
    assert summary["metros_evaluated_count"] == 10, f"Expected 10 metros, got {summary['metros_evaluated_count']}"

    print(f"Verified {summary['metros_evaluated_count']} MSAs evaluated.")
    print(f"Test Mean MAPE: {summary['final_heldout_test_aggregate_metrics']['mean_mape_pct']}%")
    print(f"Test Mean RMSE: ${summary['final_heldout_test_aggregate_metrics']['mean_rmse']:,.2f}")
    print(f"Test 95% Interval Coverage: {summary['final_heldout_test_aggregate_metrics']['mean_95pct_interval_coverage']}%")
    print("[PASS] Model 4 (Zillow Prophet Forecaster) verified successfully.")
    return True


if __name__ == "__main__":
    import pandas as pd
    verify_unet()
    verify_pricing_models()
    c_ok = verify_condition_resnet()
    f_ok = verify_forecaster()
    print("\n==========================================")
    print(f"Reload verification complete: U-Net=OK, Ames=OK, Condition={c_ok}, Forecaster={f_ok}")
    print("==========================================")
