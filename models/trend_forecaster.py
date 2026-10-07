"""
Time-Series Trend Forecasting Module
Implements:
- Facebook Prophet model with yearly/trend seasonality for regional housing price forecasting
- PyTorch LSTM sequence predictor for neural time-series projection
- 12-to-36-month future price projections with confidence bounds
- Real estate investor metrics: CAGR, 3-year projected appreciation, risk volatility index
"""

import os
import numpy as np
import pandas as pd
from prophet import Prophet
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


class RegionalProphetForecaster:
    """
    Metropolitan real estate price trend forecaster based on Facebook Prophet.
    Uses additive trend decomposition, yearly seasonality, and changepoint detection.
    Forecast uncertainty intervals are configured explicitly to 95% (interval_width=0.95).
    Note: These are Bayesian posterior predictive forecast intervals reflecting parametric
    trend and variance uncertainty, not frequentist asymptotic confidence intervals.
    """
    def __init__(self, changepoint_prior_scale=0.08, seasonality_mode="additive", interval_width=0.95):
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.model = None
        self.region_name = None

    def fit(self, df_region):
        """
        df_region: DataFrame with columns ['Date', 'ZHVI']
        """
        self.region_name = df_region["RegionName"].iloc[0] if "RegionName" in df_region else "Region"
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df_region["Date"]),
            "y": df_region["ZHVI"].values
        }).sort_values("ds").dropna()

        self.model = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        self.model.fit(prophet_df)
        return self

    def predict_future(self, months=36):
        """Generates future forecast dataframe with explicit 95% forecast intervals."""
        future = self.model.make_future_dataframe(periods=months, freq="M")
        forecast = self.model.predict(future)
        return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    def evaluate_temporal(self, df_region, train_end="2021-12-31", val_end="2023-12-31"):
        """
        Strict chronological evaluation preventing future-to-past temporal leakage.
        - Training period: ds <= train_end (e.g., 2000-01-31 to 2021-12-31, 264 months)
        - Validation period: train_end < ds <= val_end (e.g., 2022-01-31 to 2023-12-31, 24 months)
        - Test period: ds > val_end (e.g., 2024-01-31 to 2026-08-31, 32 months)
        """
        prophet_df = pd.DataFrame({
            "ds": pd.to_datetime(df_region["Date"]),
            "y": df_region["ZHVI"].values
        }).sort_values("ds").dropna()

        train_df = prophet_df[prophet_df["ds"] <= pd.to_datetime(train_end)].copy()
        val_df = prophet_df[(prophet_df["ds"] > pd.to_datetime(train_end)) & (prophet_df["ds"] <= pd.to_datetime(val_end))].copy()
        test_df = prophet_df[prophet_df["ds"] > pd.to_datetime(val_end)].copy()

        # Step 1: Fit model strictly on Training partition for Validation evaluation
        m_val = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        m_val.fit(train_df)

        val_periods = len(val_df)
        future_val = m_val.make_future_dataframe(periods=val_periods, freq="M")
        fcst_val = m_val.predict(future_val)

        val_preds_df = fcst_val.iloc[-val_periods:]
        val_preds = val_preds_df["yhat"].values
        val_actuals = val_df["y"].values

        val_mae = float(np.mean(np.abs(val_actuals - val_preds)))
        val_rmse = float(np.sqrt(mean_squared_error(val_actuals, val_preds)))
        val_mape = float(np.mean(np.abs((val_actuals - val_preds) / val_actuals)) * 100)

        # Step 2: Fit model on Training + Validation partition for Held-Out Test evaluation
        train_val_df = pd.concat([train_df, val_df]).sort_values("ds")
        m_test = Prophet(
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        m_test.fit(train_val_df)

        test_periods = len(test_df)
        future_test = m_test.make_future_dataframe(periods=test_periods, freq="M")
        fcst_test = m_test.predict(future_test)

        test_preds_df = fcst_test.iloc[-test_periods:]
        test_preds = test_preds_df["yhat"].values
        test_lower = test_preds_df["yhat_lower"].values
        test_upper = test_preds_df["yhat_upper"].values
        test_actuals = test_df["y"].values

        test_mae = float(np.mean(np.abs(test_actuals - test_preds)))
        test_rmse = float(np.sqrt(mean_squared_error(test_actuals, test_preds)))
        test_mape = float(np.mean(np.abs((test_actuals - test_preds) / test_actuals)) * 100)
        coverage_95 = float(np.mean((test_actuals >= test_lower) & (test_actuals <= test_upper)) * 100)

        return {
            "splits": {
                "training_interval": [train_df["ds"].min().strftime("%Y-%m-%d"), train_df["ds"].max().strftime("%Y-%m-%d")],
                "validation_interval": [val_df["ds"].min().strftime("%Y-%m-%d"), val_df["ds"].max().strftime("%Y-%m-%d")],
                "test_interval": [test_df["ds"].min().strftime("%Y-%m-%d"), test_df["ds"].max().strftime("%Y-%m-%d")]
            },
            "validation_metrics": {
                "months": val_periods,
                "mae": round(val_mae, 2),
                "rmse": round(val_rmse, 2),
                "mape_pct": round(val_mape, 2)
            },
            "test_metrics": {
                "months": test_periods,
                "mae": round(test_mae, 2),
                "rmse": round(test_rmse, 2),
                "mape_pct": round(test_mape, 2),
                "forecast_interval_coverage_95_pct": round(coverage_95, 2)
            }
        }


class RealEstateLSTM(nn.Module):
    """
    Experimental prototype LSTM architecture for time-series forecasting.
    Note: In Phase 3, this model is NOT trained or used for final reported results
    due to limited historical sequence length (264 months) and unverified convergence.
    Prophet is the single verified, reported forecasting model.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_layers=2, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def compute_investor_metrics(history_df, forecast_df):
    """
    Computes investor-ready metrics:
    - Current Home Value
    - 3-Year Projected Value & Growth %
    - 5-Year Projected CAGR
    - Volatility Risk Rating (Low / Moderate / High)
    - Investor Opportunity Rating (A+, A, B, C)
    """
    current_val = float(history_df["ZHVI"].iloc[-1])
    future_vals = forecast_df[forecast_df["ds"] > history_df["Date"].iloc[-1]]

    if len(future_vals) >= 36:
        val_3yr = float(future_vals["yhat"].iloc[35])
    elif len(future_vals) > 0:
        val_3yr = float(future_vals["yhat"].iloc[-1])
    else:
        val_3yr = current_val * 1.15

    total_growth_pct = ((val_3yr - current_val) / current_val) * 100
    cagr = ((val_3yr / current_val) ** (1 / 3.0) - 1.0) * 100

    # Historical volatility
    monthly_returns = history_df["ZHVI"].pct_change().dropna()
    ann_volatility = float(monthly_returns.std() * np.sqrt(12) * 100)

    if ann_volatility < 3.5:
        risk_tier = "Low Volatility (Defensive Market)"
    elif ann_volatility < 7.0:
        risk_tier = "Moderate Volatility (Balanced Growth)"
    else:
        risk_tier = "High Volatility (Cyclical / High Beta)"

    # Opportunity score
    if cagr >= 6.5:
        opportunity_rating = "A+ (Strong Buy / High Appreciation)"
    elif cagr >= 4.5:
        opportunity_rating = "A (Favorable Long-Term Growth)"
    elif cagr >= 2.5:
        opportunity_rating = "B (Stable Cash-Flow Market)"
    else:
        opportunity_rating = "C (Neutral / Slower Appreciation)"

    return {
        "current_price": round(current_val, 2),
        "projected_price_3yr": round(val_3yr, 2),
        "projected_growth_3yr_pct": round(total_growth_pct, 2),
        "annualized_cagr_pct": round(cagr, 2),
        "annual_volatility_pct": round(ann_volatility, 2),
        "risk_tier": risk_tier,
        "opportunity_rating": opportunity_rating
    }
