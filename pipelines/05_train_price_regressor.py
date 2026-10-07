"""
Pipeline 05: Train Real Estate Price Prediction Models (XGBoost & LightGBM)
Phase 3: Real Model Training + Leakage-Safe Evaluation + Reproducible Metrics
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import logging
import platform
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.price_regressor import RealEstatePricePredictor
from models.pricing_feature_contract import FEATURE_CONTRACT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

SEED = 42


def evaluate_predictions(y_true, y_pred, model_name="MODEL"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100)
    return {
        "model": model_name,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2_score": round(float(r2), 4),
        "mape_pct": round(float(mape), 2)
    }


def train_price_models(seed=SEED):
    logger.info(f"=== Starting Price Prediction Training Pipeline (Seed: {seed}) ===")

    # 1. Load Processed Arrays and Check Alignment
    X_train = np.load(os.path.join(PROCESSED_DIR, "housing_X_train.npy"))
    y_train = np.load(os.path.join(PROCESSED_DIR, "housing_y_train.npy"))
    X_val = np.load(os.path.join(PROCESSED_DIR, "housing_X_val.npy"))
    y_val = np.load(os.path.join(PROCESSED_DIR, "housing_y_val.npy"))
    X_test = np.load(os.path.join(PROCESSED_DIR, "housing_X_test.npy"))
    y_test = np.load(os.path.join(PROCESSED_DIR, "housing_y_test.npy"))

    preprocessor = joblib.load(os.path.join(PROCESSED_DIR, "housing_preprocessor.pkl"))
    with open(os.path.join(PROCESSED_DIR, "housing_features.json"), "r") as f:
        feature_meta = json.load(f)

    # Verify Feature Contract Alignment
    contract_features = [item["model_feature"] for item in FEATURE_CONTRACT]
    training_features = feature_meta["all_features"]
    assert contract_features == training_features, f"Contract mismatch: {contract_features} vs {training_features}"
    logger.info(f"Feature Contract verified: All {len(training_features)} features align 100% with contract.")

    # 2. 5-Fold Cross-Validation on Training Data Only
    logger.info("Executing 5-Fold Cross-Validation on training partition (X_train)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_train_log = np.log1p(y_train)

    xgb_cv_rmses, xgb_cv_maes = [], []
    lgb_cv_rmses, lgb_cv_maes = [], []
    oof_preds_xgb = np.zeros(len(y_train))
    oof_preds_lgb = np.zeros(len(y_train))

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X_train), 1):
        X_tr_f, y_tr_f = X_train[trn_idx], y_train_log[trn_idx]
        X_va_f, y_va_f = X_train[val_idx], y_train[val_idx]

        # XGBoost Fold
        m_xgb = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1
        )
        m_xgb.fit(X_tr_f, y_tr_f)
        p_xgb = np.expm1(m_xgb.predict(X_va_f))
        oof_preds_xgb[val_idx] = p_xgb
        xgb_cv_rmses.append(np.sqrt(mean_squared_error(y_va_f, p_xgb)))
        xgb_cv_maes.append(mean_absolute_error(y_va_f, p_xgb))

        # LightGBM Fold
        m_lgb = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.04, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=seed, verbose=-1, n_jobs=-1
        )
        m_lgb.fit(X_tr_f, y_tr_f)
        p_lgb = np.expm1(m_lgb.predict(X_va_f))
        oof_preds_lgb[val_idx] = p_lgb
        lgb_cv_rmses.append(np.sqrt(mean_squared_error(y_va_f, p_lgb)))
        lgb_cv_maes.append(mean_absolute_error(y_va_f, p_lgb))

    cv_results = {
        "xgboost_cv": {
            "mean_rmse": round(float(np.mean(xgb_cv_rmses)), 2),
            "std_rmse": round(float(np.std(xgb_cv_rmses)), 2),
            "mean_mae": round(float(np.mean(xgb_cv_maes)), 2),
            "std_mae": round(float(np.std(xgb_cv_maes)), 2)
        },
        "lightgbm_cv": {
            "mean_rmse": round(float(np.mean(lgb_cv_rmses)), 2),
            "std_rmse": round(float(np.std(lgb_cv_rmses)), 2),
            "mean_mae": round(float(np.mean(lgb_cv_maes)), 2),
            "std_mae": round(float(np.std(lgb_cv_maes)), 2)
        }
    }
    logger.info(f"5-Fold CV: XGBoost RMSE=${cv_results['xgboost_cv']['mean_rmse']:,.2f}, LightGBM RMSE=${cv_results['lightgbm_cv']['mean_rmse']:,.2f}")

    # Compute genuine Out-of-Fold relative residuals on training partition (strictly independent of X_val)
    oof_rel_res_xgb = np.abs(y_train - oof_preds_xgb) / np.maximum(oof_preds_xgb, 1.0)
    oof_rel_res_lgb = np.abs(y_train - oof_preds_lgb) / np.maximum(oof_preds_lgb, 1.0)

    # 3. Train Full Models on X_train and Evaluate on X_val for Model Selection
    logger.info("Training full XGBoost Regressor on X_train...")
    xgb_predictor = RealEstatePricePredictor(preprocessor, feature_meta, model_type="xgboost")
    xgb_predictor.fit_xgboost(X_train, y_train, X_val, y_val)
    xgb_val_metrics = xgb_predictor.evaluate(X_val, y_val)

    logger.info("Training full LightGBM Regressor on X_train...")
    lgb_predictor = RealEstatePricePredictor(preprocessor, feature_meta, model_type="lightgbm")
    lgb_predictor.fit_lightgbm(X_train, y_train, X_val, y_val)
    lgb_val_metrics = lgb_predictor.evaluate(X_val, y_val)

    logger.info(f"Validation Results: XGBoost RMSE=${xgb_val_metrics['rmse']:,.2f} (MAE=${xgb_val_metrics['mae']:,.2f}, R2={xgb_val_metrics['r2_score']:.4f})")
    logger.info(f"Validation Results: LightGBM RMSE=${lgb_val_metrics['rmse']:,.2f} (MAE=${lgb_val_metrics['mae']:,.2f}, R2={lgb_val_metrics['r2_score']:.4f})")

    # Model Selection strictly based on Validation RMSE (NOT on test data)
    if xgb_val_metrics["rmse"] <= lgb_val_metrics["rmse"]:
        winning_model_name = "XGBoost"
        winning_predictor = xgb_predictor
        selection_rationale = f"XGBoost selected based on superior validation RMSE (${xgb_val_metrics['rmse']:,.2f} vs ${lgb_val_metrics['rmse']:,.2f} for LightGBM)."
    else:
        winning_model_name = "LightGBM"
        winning_predictor = lgb_predictor
        selection_rationale = f"LightGBM selected based on superior validation RMSE (${lgb_val_metrics['rmse']:,.2f} vs ${xgb_val_metrics['rmse']:,.2f} for XGBoost)."

    logger.info(f"MODEL SELECTION DECISION: {winning_model_name} selected. ({selection_rationale})")

    # 4. Calibrate Conformal Prediction Intervals using Independent Out-Of-Fold Residuals
    # Methodological fix: Calibration uses N=1,020 out-of-fold residuals from X_train,
    # strictly independent of X_val (used for model selection) and X_test (held-out test).
    logger.info("Calibrating 90% Conformal Prediction intervals using independent out-of-fold residuals (N=1,020)...")
    q_xgb = xgb_predictor.calibrate_conformal_intervals(
        oof_residuals=oof_rel_res_xgb,
        coverage=0.90,
        method="5-Fold Cross-Conformal (Independent OOF Residuals)"
    )
    q_lgb = lgb_predictor.calibrate_conformal_intervals(
        oof_residuals=oof_rel_res_lgb,
        coverage=0.90,
        method="5-Fold Cross-Conformal (Independent OOF Residuals)"
    )
    logger.info(f"Calibrated 90% conformal relative margin: XGBoost={q_xgb*100:.2f}%, LightGBM={q_lgb*100:.2f}%")

    # 5. Single Final Evaluation on Held-Out Test Set (Frozen Models)
    logger.info("Executing final single evaluation on held-out test partition (housing_test_df.csv, N=219)...")
    xgb_test_metrics = xgb_predictor.evaluate(X_test, y_test)
    lgb_test_metrics = lgb_predictor.evaluate(X_test, y_test)

    # Test empirical coverage calculation
    p_xgb_test = xgb_predictor.predict(X_test)
    cov_xgb = float(np.mean((y_test >= p_xgb_test * (1.0 - q_xgb)) & (y_test <= p_xgb_test * (1.0 + q_xgb))) * 100.0)
    p_lgb_test = lgb_predictor.predict(X_test)
    cov_lgb = float(np.mean((y_test >= p_lgb_test * (1.0 - q_lgb)) & (y_test <= p_lgb_test * (1.0 + q_lgb))) * 100.0)

    logger.info(f"FINAL TEST EVALUATION — XGBoost: MAE=${xgb_test_metrics['mae']:,.2f}, RMSE=${xgb_test_metrics['rmse']:,.2f}, R2={xgb_test_metrics['r2_score']:.4f}, MAPE={xgb_test_metrics['mape_pct']:.2f}%, 90% Coverage={cov_xgb:.2f}%")
    logger.info(f"FINAL TEST EVALUATION — LightGBM: MAE=${lgb_test_metrics['mae']:,.2f}, RMSE=${lgb_test_metrics['rmse']:,.2f}, R2={lgb_test_metrics['r2_score']:.4f}, MAPE={lgb_test_metrics['mape_pct']:.2f}%, 90% Coverage={cov_lgb:.2f}%")

    # 6. Extract Actual Gain-Based Feature Importance from Winning Model
    feature_importances = []
    if winning_model_name == "XGBoost":
        raw_importances = winning_predictor.model.feature_importances_
        importance_type = "Gain (relative contribution of each feature to the model)"
    else:
        raw_importances = winning_predictor.model.booster_.feature_importance(importance_type="gain")
        raw_importances = raw_importances / np.sum(raw_importances)
        importance_type = "Gain (total gains of splits which use the feature)"

    transformed_feature_names = [f.replace("num__", "").replace("cat__", "") for f in preprocessor.get_feature_names_out()]
    for i, col in enumerate(transformed_feature_names):
        if i < len(raw_importances):
            feature_importances.append({
                "feature": col,
                "importance": round(float(raw_importances[i]), 4)
            })
    feature_importances = sorted(feature_importances, key=lambda x: x["importance"], reverse=True)

    # 7. Save Checkpoint Artifacts
    logger.info("Saving model checkpoints and metadata...")
    joblib.dump(xgb_predictor, os.path.join(SAVED_MODELS_DIR, "xgboost_ames_v1.pkl"))
    joblib.dump(xgb_predictor, os.path.join(SAVED_MODELS_DIR, "xgboost_price.pkl"))
    joblib.dump(lgb_predictor, os.path.join(SAVED_MODELS_DIR, "lightgbm_ames_v1.pkl"))
    joblib.dump(lgb_predictor, os.path.join(SAVED_MODELS_DIR, "lightgbm_price.pkl"))

    winning_test_cov = cov_xgb if winning_model_name == "XGBoost" else cov_lgb

    combined_report = {
        "task": "Ames Housing Price Regression",
        "version": "v1.0-real",
        "target": "SalePrice (USD)",
        "target_transform": "log1p during training, expm1 for prediction",
        "seed": seed,
        "sample_counts": {
            "train": len(y_train),
            "validation": len(y_val),
            "test": len(y_test)
        },
        "features_count": len(training_features),
        "features": training_features,
        "cross_validation_5fold": cv_results,
        "model_selection": {
            "selected_model": winning_model_name,
            "decision_rule": "Lowest Validation RMSE (evaluated strictly on X_val, test set untouched)",
            "rationale": selection_rationale
        },
        "validation_evaluation": {
            "xgboost": xgb_val_metrics,
            "lightgbm": lgb_val_metrics
        },
        "final_heldout_test_evaluation": {
            "xgboost": xgb_test_metrics,
            "lightgbm": lgb_test_metrics
        },
        "uncertainty_quantification": {
            "methodology": "5-Fold Cross-Conformal Prediction (Vovk 2015; Barber et al. 2021)",
            "nonconformity_score": "Relative absolute error |y - yhat| / yhat",
            "calibration_set": "Out-of-fold residuals on X_train (N=1020 properties). Strictly independent of model selection partition X_val (N=219) and held-out test partition X_test (N=219).",
            "target_coverage_pct": 90.0,
            "calibrated_relative_margin_pct": round(winning_predictor.conformal_quantile * 100.0, 2),
            "test_empirical_coverage_pct": round(winning_test_cov, 2),
            "formula": "[yhat * (1 - margin), yhat * (1 + margin)]"
        },
        "feature_importance": {
            "importance_type": importance_type,
            "model": winning_model_name,
            "rankings": feature_importances
        },
        "environment": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "lightgbm": lgb.__version__,
            "sklearn": sklearn.__version__,
            "os": platform.platform()
        },
        "artifacts": {
            "xgboost_checkpoint": "models/saved/xgboost_ames_v1.pkl",
            "lightgbm_checkpoint": "models/saved/lightgbm_ames_v1.pkl",
            "preprocessor": "data/processed/housing_preprocessor.pkl"
        }
    }

    with open(os.path.join(SAVED_MODELS_DIR, "price_metrics.json"), "w") as f:
        json.dump(combined_report, f, indent=2)

    logger.info(f"Price modeling complete. Report saved to {os.path.join(SAVED_MODELS_DIR, 'price_metrics.json')}")
    return combined_report


if __name__ == "__main__":
    train_price_models(seed=42)
