"""
Smoke Tests for RealtyAI Streamlit Application
"""

import os
import sys
import pytest

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)


def test_app_imports():
    """Verifies that all views and components can be imported without syntax or import errors."""
    from app.config import CUSTOM_CSS, PAGE_TITLE
    from app.components.header import render_navbar, render_hero
    from app.components.charts import create_forecast_chart, create_feature_importance_chart
    from app.components.map_view import render_metro_market_map
    from app.views.buyer_view import render_buyer_view
    from app.views.investor_view import render_investor_view
    from app.views.urban_planner_view import render_urban_planner_view
    from app.views.model_metrics_view import render_model_metrics_view

    assert "RealtyAI" in PAGE_TITLE
    assert len(CUSTOM_CSS) > 100
