"""
Interactive Plotly Charts for RealtyAI Dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


def create_forecast_chart(history_df, forecast_df, metro_name):
    """
    Creates an interactive time-series chart with clearly delineated temporal regimes:
    1. Historical Training (2000-01-31 to 2021-12-31)
    2. Validation Period (2022-01-31 to 2023-12-31)
    3. Held-Out Test Period (2024-01-31 to 2026-08-31)
    4. Future Projection Horizon (2026-09-30 onward)
    with shaded 95% forecast interval (empirical test coverage: 40.31%).
    """
    fig = go.Figure()

    hist_df = history_df.copy()
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    hist_df = hist_df.sort_values("Date")

    # Split historical into temporal evaluation regimes
    train_mask = hist_df["Date"] <= "2021-12-31"
    val_mask = (hist_df["Date"] > "2021-12-31") & (hist_df["Date"] <= "2023-12-31")
    test_mask = (hist_df["Date"] > "2023-12-31") & (hist_df["Date"] <= "2026-08-31")

    df_train = hist_df[train_mask]
    df_val = hist_df[val_mask]
    df_test = hist_df[test_mask]

    # Historical Training trace
    fig.add_trace(go.Scatter(
        x=df_train["Date"],
        y=df_train["ZHVI"],
        mode="lines",
        name="Observed ZHVI (Training: 2000–2021)",
        line=dict(color="#3B82F6", width=2.5)
    ))

    # Observed Validation trace
    if not df_val.empty:
        # Include last point of train for continuous line
        val_plot_df = pd.concat([df_train.tail(1), df_val]) if not df_train.empty else df_val
        fig.add_trace(go.Scatter(
            x=val_plot_df["Date"],
            y=val_plot_df["ZHVI"],
            mode="lines",
            name="Observed ZHVI (Validation: 2022–2023)",
            line=dict(color="#F59E0B", width=2.5)
        ))

    # Observed Held-Out Test trace
    if not df_test.empty:
        test_plot_df = pd.concat([df_val.tail(1), df_test]) if not df_val.empty else df_test
        fig.add_trace(go.Scatter(
            x=test_plot_df["Date"],
            y=test_plot_df["ZHVI"],
            mode="lines",
            name="Observed ZHVI (Held-Out Test: 2024–2026)",
            line=dict(color="#10B981", width=2.5)
        ))

    # Future Projection trace & 95% forecast interval
    if not forecast_df.empty:
        fcst = forecast_df.copy()
        fcst["Date"] = pd.to_datetime(fcst["Date"])
        fcst = fcst.sort_values("Date")
        fcst_dates = fcst["Date"]

        # Upper bound
        fig.add_trace(go.Scatter(
            x=fcst_dates,
            y=fcst["Forecast_Upper_95"] if "Forecast_Upper_95" in fcst.columns else fcst.get("Forecast_Upper"),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        ))

        # Lower bound with fill
        fig.add_trace(go.Scatter(
            x=fcst_dates,
            y=fcst["Forecast_Lower_95"] if "Forecast_Lower_95" in fcst.columns else fcst.get("Forecast_Lower"),
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(168, 85, 247, 0.18)",
            name="95% Forecast Interval (Empirical Cov: 40.31%)",
            hoverinfo="skip"
        ))

        # Future projection line
        fig.add_trace(go.Scatter(
            x=fcst_dates,
            y=fcst["Forecast_ZHVI"],
            mode="lines+markers",
            name="Prophet Future Projection (2026–2029)",
            line=dict(color="#C084FC", width=2.5, dash="dash"),
            marker=dict(size=4)
        ))

    # Vertical demarcation lines separating periods
    fig.add_vline(x="2021-12-31", line_width=1, line_dash="dot", line_color="rgba(245, 158, 11, 0.6)")
    fig.add_vline(x="2023-12-31", line_width=1, line_dash="dot", line_color="rgba(16, 185, 129, 0.6)")
    fig.add_vline(x="2026-08-31", line_width=1, line_dash="dot", line_color="rgba(192, 132, 252, 0.8)")

    fig.update_layout(
        title=f"Regional ZHVI Trajectory & Chronological Regimes: {metro_name}",
        title_font=dict(size=16, color="#F8FAFC", family="Plus Jakarta Sans"),
        paper_bgcolor="rgba(17, 24, 39, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#94A3B8"),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            tickprefix="$",
            showline=True,
            linecolor="rgba(255, 255, 255, 0.1)"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11)
        ),
        hovermode="x unified",
        margin=dict(l=40, r=30, t=60, b=40)
    )
    return fig


def create_feature_importance_chart(feature_list):
    """Horizontal bar chart for pricing model feature importance."""
    df_feat = pd.DataFrame(feature_list).sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df_feat["importance"],
        y=df_feat["feature"],
        orientation="h",
        marker=dict(
            color=df_feat["importance"],
            colorscale=[[0, "#3B82F6"], [1, "#8B5CF6"]],
            line=dict(color="rgba(255, 255, 255, 0.2)", width=1)
        )
    ))

    fig.update_layout(
        title="Key Price Drivers (XGBoost Feature Importance)",
        title_font=dict(size=15, color="#F8FAFC"),
        paper_bgcolor="rgba(17, 24, 39, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#94A3B8"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Normalized Importance Score"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=120, r=30, t=50, b=40),
        height=380
    )
    return fig


def create_confusion_matrix_chart(cm, class_names):
    """Creates an annotated confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        labels=dict(x="Predicted Condition", y="Ground Truth Condition", color="Count"),
        x=class_names,
        y=class_names,
        color_continuous_scale="Viridis",
        text_auto=True
    )

    fig.update_layout(
        title="Condition Classifier Confusion Matrix",
        title_font=dict(size=15, color="#F8FAFC"),
        paper_bgcolor="rgba(17, 24, 39, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#94A3B8"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )
    return fig


def create_training_curves_chart(history_dict, metric_name="IoU"):
    """Visualizes training and validation curves."""
    fig = go.Figure()
    epochs = list(range(1, len(history_dict.get("train_loss", [])) + 1))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=history_dict.get("train_loss", []),
        mode="lines+markers",
        name="Training Loss",
        line=dict(color="#EF4444", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=epochs,
        y=history_dict.get("val_loss", []),
        mode="lines+markers",
        name="Validation Loss",
        line=dict(color="#F59E0B", width=2, dash="dash")
    ))

    metric_key = f"val_{metric_name.lower()}"
    if metric_key in history_dict:
        fig.add_trace(go.Scatter(
            x=epochs,
            y=history_dict.get(metric_key, []),
            mode="lines+markers",
            name=f"Val {metric_name}",
            line=dict(color="#10B981", width=2.5)
        ))

    fig.update_layout(
        title=f"Convergence & Loss Dynamics (U-Net Training)",
        title_font=dict(size=15, color="#F8FAFC"),
        paper_bgcolor="rgba(17, 24, 39, 0.6)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        font=dict(color="#94A3B8"),
        xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Epoch"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
        margin=dict(l=40, r=40, t=50, b=40),
        height=340
    )
    return fig
