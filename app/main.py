"""
RealtyAI - Smart Real Estate Insight Platform
Main Application Entry Point
"""

import streamlit as st

st.set_page_config(
    page_title="RealtyAI | Smart Real Estate Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from app.config import CUSTOM_CSS
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

from app.components.header import render_navbar
from app.views.buyer_view import render_buyer_view
from app.views.investor_view import render_investor_view
from app.views.urban_planner_view import render_urban_planner_view
from app.views.model_metrics_view import render_model_metrics_view


def main():
    selected_view = render_navbar()

    if selected_view == "Buyer & Property Inspector":
        render_buyer_view()
    elif selected_view == "Real Estate Investor":
        render_investor_view()
    elif selected_view == "Urban Planner & Satellite AI":
        render_urban_planner_view()
    elif selected_view == "Model Performance & AI Hub":
        render_model_metrics_view()

    # Footer
    st.markdown("<br><hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 40px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; color: #64748B; font-size: 0.8rem; padding-bottom: 20px;">
            <div>
                <b>RealtyAI v1.0.0</b> — Built with PyTorch, XGBoost, Prophet & Streamlit
            </div>
            <div>
                Milestones I, II, III & IV Complete | SpaceNet & Zillow Research Powered
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
