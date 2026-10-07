"""
Header and Navigation Components for RealtyAI
"""

import streamlit as st
from streamlit_option_menu import option_menu


def render_navbar():
    """Renders the top navigation bar with personas."""
    with st.container():
        col_logo, col_nav = st.columns([1.2, 3.8])
        with col_logo:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 12px; padding: 6px 0;">
                    <span style="font-size: 2.2rem;">🏢</span>
                    <div>
                        <div style="font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; color: #FFFFFF; line-height: 1.1;">
                            Realty<span style="color: #60A5FA;">AI</span>
                        </div>
                        <div style="font-size: 0.72rem; color: #94A3B8; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;">
                            Smart Real Estate Insight Platform
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with col_nav:
            selected_persona = option_menu(
                menu_title=None,
                options=[
                    "Buyer & Property Inspector",
                    "Real Estate Investor",
                    "Urban Planner & Satellite AI",
                    "Model Performance & AI Hub"
                ],
                icons=["house-door", "graph-up-arrow", "geo-alt", "cpu"],
                menu_icon="cast",
                default_index=0,
                orientation="horizontal",
                styles={
                    "container": {
                        "padding": "0!important",
                        "background-color": "rgba(30, 41, 59, 0.5)",
                        "border-radius": "10px",
                        "border": "1px solid rgba(255, 255, 255, 0.08)"
                    },
                    "icon": {"color": "#60A5FA", "font-size": "15px"},
                    "nav-link": {
                        "font-size": "13px",
                        "text-align": "center",
                        "margin": "0px",
                        "color": "#94A3B8",
                        "font-weight": "600",
                        "--hover-color": "rgba(99, 102, 241, 0.15)"
                    },
                    "nav-link-selected": {
                        "background": "linear-gradient(90deg, #3B82F6 0%, #6366F1 100%)",
                        "color": "#FFFFFF",
                        "border-radius": "8px"
                    },
                }
            )

    st.markdown("<hr style='margin: 14px 0 24px 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.08);'>", unsafe_allow_html=True)
    return selected_persona


def render_hero(title, subtitle, badge_text="Live AI Inference", badge_type="badge-blue"):
    st.markdown(
        f"""
        <div class="hero-banner">
            <span class="badge {badge_type}">{badge_text}</span>
            <div class="hero-title" style="margin-top: 10px;">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
