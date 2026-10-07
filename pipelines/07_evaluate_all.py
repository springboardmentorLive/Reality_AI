"""
Pipeline 07: Unified Model Evaluation Suite
Aggregates and formats all metrics across:
1. SpaceNet Satellite Segmentation (IoU, Dice Score)
2. Property Condition Classification (Accuracy, Precision, Recall, F1)
3. Tabular Housing Price Prediction (MAE, RMSE, R2, MAPE)
4. Regional Market Trend Forecasting (MAPE, RMSE)
Produces master evaluation summary for the dashboard.
"""

import os
import sys
import json
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def evaluate_all():
    logger.info("=== Running Unified RealtyAI Evaluation Suite ===")

    master_report = {
        "project_name": "RealtyAI Smart Real Estate Insight Platform",
        "milestones_completed": ["Milestone I", "Milestone II", "Milestone III", "Milestone IV"],
        "modules_implemented": [
            "Data Collection and Cleaning",
            "Image Preprocessing and Satellite Segmentation",
            "Property Condition Classification",
            "Price Prediction Model",
            "TimeSeries Trend Forecasting",
            "Evaluation and Visualization",
            "Interactive Multi-Persona Dashboard"
        ],
        "models": {}
    }

    # 1. Satellite Segmentation
    unet_file = os.path.join(SAVED_MODELS_DIR, "unet_metrics.json")
    if os.path.exists(unet_file):
        with open(unet_file, "r") as f:
            master_report["models"]["segmentation_unet"] = json.load(f)
    else:
        master_report["models"]["segmentation_unet"] = {"status": "pending_or_training"}

    # 2. Property Condition CNN
    cond_file = os.path.join(SAVED_MODELS_DIR, "condition_metrics.json")
    if os.path.exists(cond_file):
        with open(cond_file, "r") as f:
            master_report["models"]["condition_resnet"] = json.load(f)
    else:
        master_report["models"]["condition_resnet"] = {"status": "pending_or_training"}

    # 3. Price Prediction
    price_file = os.path.join(SAVED_MODELS_DIR, "price_metrics.json")
    if os.path.exists(price_file):
        with open(price_file, "r") as f:
            master_report["models"]["price_regression"] = json.load(f)
    else:
        master_report["models"]["price_regression"] = {"status": "pending_or_training"}

    # 4. Trend Forecasting
    fcst_file = os.path.join(SAVED_MODELS_DIR, "forecaster_summary.json")
    if os.path.exists(fcst_file):
        with open(fcst_file, "r") as f:
            master_report["models"]["trend_forecasting"] = json.load(f)
    else:
        master_report["models"]["trend_forecasting"] = {"status": "pending_or_training"}

    # Output master json
    out_path = os.path.join(PROCESSED_DIR, "master_evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(master_report, f, indent=2)

    logger.info(f"Master evaluation metrics saved to {out_path}")
    return master_report


if __name__ == "__main__":
    evaluate_all()
