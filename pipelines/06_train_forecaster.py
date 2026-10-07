"""
Pipeline 06: Train Regional Real Estate Trend Forecaster (Prophet)
Trains time-series models on Zillow ZHVI regional home value data.
Computes MAPE, RMSE, 36-month forecasts, and investor metrics for top metropolitan markets
using strict chronological temporal evaluation without data leakage.
"""

import os
import sys
import json
import platform
import numpy as np
import pandas as pd
import logging
from sklearn.metrics import mean_squared_error

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from models.trend_forecaster import RegionalProphetForecaster, compute_investor_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)


def train_regional_forecasters(top_n_metros=10, forecast_months=36):
    logger.info("=== Starting Time-Series Trend Forecasting Pipeline (Prophet-Only Temporal Evaluation) ===")

    zillow_path = os.path.join(PROCESSED_DIR, "zillow_zhvi_processed.csv")
    if not os.path.exists(zillow_path):
        raise FileNotFoundError(f"Processed Zillow dataset not found at {zillow_path}")

    df = pd.read_csv(zillow_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # Metro Selection Rule: Filter by RegionType == 'msa' and select top N by SizeRank (1 to N)
    msa_df = df[df["RegionType"] == "msa"].copy()
    top_metro_meta = msa_df[["RegionName", "StateName", "SizeRank"]].drop_duplicates().sort_values("SizeRank").head(top_n_metros)
    top_metros = top_metro_meta["RegionName"].tolist()

    logger.info(f"Selected Top {len(top_metros)} MSAs by U.S. Census Population SizeRank (1 to {top_n_metros}): {top_metros}")

    all_forecast_records = []
    metro_evaluations = {}

    val_mapes, val_rmses, val_maes = [], [], []
    test_mapes, test_rmses, test_maes = [], [], []
    test_coverages = []

    train_end = "2021-12-31"
    val_end = "2023-12-31"

    for metro in top_metros:
        rank = int(top_metro_meta[top_metro_meta["RegionName"] == metro]["SizeRank"].iloc[0])
        logger.info(f"Processing MSA [SizeRank {rank}]: {metro}...")
        df_metro = msa_df[msa_df["RegionName"] == metro].sort_values("Date")
        if len(df_metro) < 36:
            logger.warning(f"Skipping {metro}: insufficient observations ({len(df_metro)})")
            continue

        forecaster = RegionalProphetForecaster(changepoint_prior_scale=0.08, interval_width=0.95)

        # 1. Chronological Temporal Evaluation (Train: <= 2021-12-31, Val: 2022-2023, Test: 2024-2026)
        temporal_eval = forecaster.evaluate_temporal(df_metro, train_end=train_end, val_end=val_end)
        metro_evaluations[metro] = {
            "size_rank": rank,
            "temporal_splits": temporal_eval["splits"],
            "validation_metrics": temporal_eval["validation_metrics"],
            "test_metrics": temporal_eval["test_metrics"]
        }

        val_mapes.append(temporal_eval["validation_metrics"]["mape_pct"])
        val_rmses.append(temporal_eval["validation_metrics"]["rmse"])
        val_maes.append(temporal_eval["validation_metrics"]["mae"])

        test_mapes.append(temporal_eval["test_metrics"]["mape_pct"])
        test_rmses.append(temporal_eval["test_metrics"]["rmse"])
        test_maes.append(temporal_eval["test_metrics"]["mae"])
        test_coverages.append(temporal_eval["test_metrics"]["forecast_interval_coverage_95_pct"])

        # 2. Fit Forecaster on full history to generate future 36-month projections
        forecaster.fit(df_metro)
        future_fcst = forecaster.predict_future(months=forecast_months)

        # 3. Investor Financial Metrics
        inv_metrics = compute_investor_metrics(df_metro, future_fcst)
        metro_evaluations[metro]["investor_metrics"] = inv_metrics

        # 4. Save structured forecast records
        future_only = future_fcst[future_fcst["ds"] > df_metro["Date"].iloc[-1]].copy()
        for _, row in future_only.iterrows():
            all_forecast_records.append({
                "RegionName": metro,
                "SizeRank": rank,
                "Date": row["ds"].strftime("%Y-%m-%d"),
                "Forecast_ZHVI": round(float(row["yhat"]), 2),
                "Forecast_Lower_95": round(float(row["yhat_lower"]), 2),
                "Forecast_Upper_95": round(float(row["yhat_upper"]), 2)
            })

    # Overall Summary across all evaluated MSAs
    overall_summary = {
        "task": "Zillow ZHVI Regional Housing Trend Forecasting",
        "version": "v1.0-real",
        "methodology": "Facebook Prophet (Additive Trend + Yearly Seasonality + Bayesian Uncertainty)",
        "model_decision": {
            "selected_model": "Prophet-only",
            "omitted_model": "RealEstateLSTM (Untrained / unverified for production; documented as experimental prototype)",
            "rationale": "Dataset size (264 monthly observations per metro) is optimal for Bayesian structural time-series decomposition. Deep LSTM on short monthly series suffers from high variance and poor generalizability without cross-series pretraining."
        },
        "metro_selection_rule": {
            "filter": "RegionType == 'msa'",
            "ordering": "SizeRank ascending (1 to 10)",
            "justification": "Focuses on the top 10 primary US metropolitan statistical areas representing ~30% of total national residential real estate transaction volume.",
            "selected_metros": top_metros
        },
        "temporal_evaluation_protocol": {
            "leakage_protection": "Strict chronological forward-chaining split; no shuffling; past-to-future projection only",
            "training_period": f"2000-01-31 to {train_end} (264 monthly snapshots)",
            "validation_period": f"2022-01-31 to {val_end} (24 monthly snapshots)",
            "held_out_test_period": f"2024-01-31 to 2026-08-31 (32 monthly snapshots)"
        },
        "forecast_uncertainty": {
            "methodology": "Prophet Bayesian posterior predictive interval",
            "configured_width": 0.95,
            "definition": "95% forecast interval representing parameter and residual variance under Prophet prior specifications; not frequentist asymptotic confidence intervals.",
            "empirical_test_coverage_pct": round(float(np.mean(test_coverages)), 2)
        },
        "validation_aggregate_metrics": {
            "mean_mae": round(float(np.mean(val_maes)), 2),
            "mean_rmse": round(float(np.mean(val_rmses)), 2),
            "mean_mape_pct": round(float(np.mean(val_mapes)), 2)
        },
        "final_heldout_test_aggregate_metrics": {
            "mean_mae": round(float(np.mean(test_maes)), 2),
            "mean_rmse": round(float(np.mean(test_rmses)), 2),
            "mean_mape_pct": round(float(np.mean(test_mapes)), 2),
            "mean_95pct_interval_coverage": round(float(np.mean(test_coverages)), 2)
        },
        "forecast_horizon_months": forecast_months,
        "metros_evaluated_count": len(metro_evaluations),
        "metro_evaluations": metro_evaluations,
        "environment": {
            "python": platform.python_version(),
            "os": platform.platform()
        }
    }

    # Save outputs
    with open(os.path.join(SAVED_MODELS_DIR, "forecaster_summary.json"), "w") as f:
        json.dump(overall_summary, f, indent=2)

    pd.DataFrame(all_forecast_records).to_csv(os.path.join(PROCESSED_DIR, "forecasts_cache.csv"), index=False)
    logger.info(f"Forecasting complete across {len(metro_evaluations)} metros.")
    logger.info(f"  -> Validation: Mean MAPE = {overall_summary['validation_aggregate_metrics']['mean_mape_pct']}%, Mean RMSE = ${overall_summary['validation_aggregate_metrics']['mean_rmse']:,.2f}")
    logger.info(f"  -> Test:       Mean MAPE = {overall_summary['final_heldout_test_aggregate_metrics']['mean_mape_pct']}%, Mean RMSE = ${overall_summary['final_heldout_test_aggregate_metrics']['mean_rmse']:,.2f}, 95% Coverage = {overall_summary['final_heldout_test_aggregate_metrics']['mean_95pct_interval_coverage']}%")
    return overall_summary


if __name__ == "__main__":
    train_regional_forecasters(top_n_metros=10, forecast_months=36)
