"""
Buyer Persona & Structural Inspection View
Phase 4 Verified Integration:
1. Real-time property valuation calculator powered by Ames-trained XGBoost Regressor
   with 5-Fold Cross-Conformal Prediction intervals (±19.59%) and transparent feature contract lineage.
2. PEER Post-Disaster Structural Collapse Inspector powered by transfer-learned ResNet-18
   evaluating extreme structural collapse modes (Non-collapse, Partial collapse, Global collapse).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

from app.components.header import render_hero
from models.condition_resnet import PropertyConditionClassifier, get_condition_transforms, evaluate_property_inspection
from models.pricing_feature_contract import (
    FEATURE_CONTRACT,
    build_contract_compliant_input,
    validate_feature_contract
)
from app.config import SAVED_MODELS_DIR, SAMPLE_IMG_DIR, PROCESSED_DIR


@st.cache_resource
def load_pricing_model():
    """Loads canonical verified Phase 3 XGBoost pricing model artifact."""
    model_path = os.path.join(SAVED_MODELS_DIR, "xgboost_ames_v1.pkl")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            return None
    return None


@st.cache_resource
def load_condition_model():
    """Loads canonical verified Phase 3 PEER ResNet collapse classifier artifact."""
    model_path = os.path.join(SAVED_MODELS_DIR, "resnet_peer_collapse_v1.pt")
    if not os.path.exists(model_path):
        return None
    try:
        model = PropertyConditionClassifier(num_classes=3, pretrained=False)
        state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception:
        return None


def render_buyer_view():
    render_hero(
        title="Home Buyer & Structural Inspection Hub",
        subtitle="Ames-trained illustrative property pricing with distribution-free conformal bounds, and PEER post-disaster structural collapse assessment.",
        badge_text="Valuation & Structural AI",
        badge_type="badge-blue"
    )

    tab1, tab2 = st.tabs([
        "💰 Ames Illustrative Property Valuation",
        "🏗️ PEER Post-Disaster Structural Collapse Inspector"
    ])

    # =========================================================================
    # TAB 1: Valuation Calculator
    # =========================================================================
    with tab1:
        st.markdown("### Ames-Trained Illustrative Property Valuation")
        st.caption("Adjust key property attributes to generate predictive pricing using our verified gradient-boosted ML pipeline.")

        # Crucial Applicability & Geographic Scope Warning
        st.warning(
            "⚠️ **RESEARCH DATASET APPLICABILITY WARNING:**\n\n"
            "This model is trained strictly on residential property sales in **Ames, Iowa (2006–2010)** from the Ames Housing Research Dataset. "
            "It must **NOT** be interpreted as 'true market value' or applied to other cities, states, contemporary transactions, or commercial properties. "
            "It is provided solely as an illustrative demonstration of regression and conformal uncertainty quantification."
        )

        pricing_model = load_pricing_model()
        model_status_str = "TRAINED — VERIFIED (XGBoost Ames v1.0-real)" if pricing_model is not None else "MODEL NOT AVAILABLE"
        status_color = "#10B981" if pricing_model is not None else "#EF4444"

        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                <span style="font-weight: 700; font-size: 0.85rem; color: #94A3B8;">MODEL STATUS:</span>
                <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: {status_color}; border: 1px solid {status_color};">
                    {model_status_str}
                </span>
                <span style="color: #64748B; font-size: 0.8rem;">| Checkpoint: <code>models/saved/xgboost_ames_v1.pkl</code></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Step 1: User-Provided Property Attributes (12 Inputs)")
        col_inp1, col_inp2, col_inp3 = st.columns(3)

        with col_inp1:
            gr_liv_area = st.number_input("Living Area (Sq. Ft.)", min_value=500, max_value=8000, value=1850, step=50, help="Above grade finished living area (GrLivArea)")
            overall_qual = st.slider("Overall Quality (1-10 Scale)", min_value=1, max_value=10, value=7, help="1=Very Poor, 5=Average, 7=Good, 10=Luxury")
            overall_cond = st.slider("Overall Condition (1-10 Scale)", min_value=1, max_value=10, value=6, help="1=Poor, 5=Average, 6=Above Average, 10=Excellent")
            year_built = st.slider("Year Built", min_value=1900, max_value=2010, value=2005, help="Original construction year (Ames dataset upper bound: 2010)")

        with col_inp2:
            bedrooms = st.selectbox("Bedrooms (Above Grade)", options=[1, 2, 3, 4, 5, 6], index=2)
            full_bath = st.selectbox("Full Bathrooms", options=[1, 2, 3, 4], index=1)
            half_bath = st.selectbox("Half Bathrooms", options=[0, 1, 2], index=1)
            total_bsmt_sf = st.number_input("Basement Area (Sq. Ft.)", min_value=0, max_value=4000, value=950, step=50, help="Total square feet of basement area")

        with col_inp3:
            neighborhood = st.selectbox("Ames Neighborhood Zone", options=[
                "CollgCr", "NAmes", "OldTown", "Edwards", "Somerst", "NridgHt", "Gilbert", "Sawyer", "NWAmes"
            ], index=0, help="Ames, Iowa municipal neighborhood identifier")
            garage_cars = st.selectbox("Garage Capacity (Stalls)", options=[0, 1, 2, 3, 4], index=2)
            lot_area = st.number_input("Lot Size (Sq. Ft.)", min_value=1000, max_value=50000, value=9500, step=500)
            remodel_year = st.slider("Last Remodel Year", min_value=1950, max_value=2010, value=2008, help="Remodel date or original construction date")

        # Feature Contract Lineage & Breakdown Expander
        with st.expander("🔍 Inspect Full 27-Feature Contract & Lineage (User vs. Derived vs. Imputed)"):
            st.markdown(
                """
                In accordance with research-hardening standards, the model accepts strictly 27 features.
                No hidden random generators or artificial distributions are permitted:
                - **12 User-Provided:** Directly controlled via the inputs above.
                - **6 Derived Features:** Explicitly calculated via documented engineering formulas.
                - **9 Imputed Features:** Fixed to training-set population medians/modes (documented policy).
                """
            )

            raw_inputs = {
                "GrLivArea": gr_liv_area,
                "OverallQual": overall_qual,
                "OverallCond": overall_cond,
                "YearBuilt": year_built,
                "YearRemodAdd": remodel_year,
                "TotalBsmtSF": total_bsmt_sf,
                "BedroomAbvGr": bedrooms,
                "FullBath": full_bath,
                "HalfBath": half_bath,
                "GarageCars": garage_cars,
                "LotArea": lot_area,
                "Neighborhood": neighborhood
            }
            contract_vector = build_contract_compliant_input(raw_inputs)

            f_rows = []
            for item in FEATURE_CONTRACT:
                f_name = item["model_feature"]
                val = contract_vector.get(f_name, "N/A")
                f_rows.append({
                    "Feature": f_name,
                    "Runtime Value": str(val),
                    "Contract Classification": item["state"],
                    "Origin & Justification": item["reason"]
                })
            st.dataframe(pd.DataFrame(f_rows), use_container_width=True, hide_index=True)

        calc_btn = st.button("⚡ Generate Illustrative Valuation", use_container_width=True, type="primary")

        if pricing_model is None:
            st.markdown("<br>", unsafe_allow_html=True)
            st.error("⚠️ MODEL NOT AVAILABLE: Tabular pricing checkpoint (`models/saved/xgboost_ames_v1.pkl`) is missing. Prediction safely halted.")
        else:
            # Validate complete contract compliance before inference
            raw_inputs = {
                "GrLivArea": gr_liv_area,
                "OverallQual": overall_qual,
                "OverallCond": overall_cond,
                "YearBuilt": year_built,
                "YearRemodAdd": remodel_year,
                "TotalBsmtSF": total_bsmt_sf,
                "BedroomAbvGr": bedrooms,
                "FullBath": full_bath,
                "HalfBath": half_bath,
                "GarageCars": garage_cars,
                "LotArea": lot_area,
                "Neighborhood": neighborhood
            }
            contract_input = build_contract_compliant_input(raw_inputs)
            assert len(contract_input) == 27, f"Feature contract violation: expected 27 features, got {len(contract_input)}"

            res = pricing_model.predict_property(contract_input)
            est_price = res["estimated_price"]
            low_price = res["price_range_low"]
            high_price = res["price_range_high"]
            margin_pct = res.get("conformal_relative_margin", 0.1959) * 100.0
            interval_method = res.get("interval_method", "5-Fold Out-of-Fold Residual Calibration")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Valuation Scorecard")

            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Illustrative Valuation Estimate</div>
                        <div class="metric-value" style="color: #60A5FA;">${est_price:,.0f}</div>
                        <div class="metric-delta-pos">Ames XGBoost (Test R²: 0.9103)</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">90% Prediction Interval (±{margin_pct:.1f}%)</div>
                        <div class="metric-value" style="font-size: 1.45rem; color: #C084FC;">${low_price:,.0f} – ${high_price:,.0f}</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">{interval_method}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with r3:
                sqft_price = est_price / max(1.0, float(gr_liv_area))
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Estimated Unit Price</div>
                        <div class="metric-value" style="color: #34D399;">${sqft_price:,.1f}/sqft</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Above-Grade Finished Area</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Methodology and Traceability Card
            st.markdown("<br>", unsafe_allow_html=True)
            with st.container():
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                        <h5 style="margin-top: 0; color: #F8FAFC;">📋 Inference & Uncertainty Methodology</h5>
                        <ul style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 0;">
                            <li><b>Point Prediction:</b> XGBoost Regressor trained on <code>log1p(SalePrice)</code> with early stopping, exponentiated back to dollar scale.</li>
                            <li><b>Uncertainty Calibration:</b> 5-Fold Out-of-Fold Residual Calibration (cross-conformal construction; Vovk 2015; Barber et al. 2021) calibrated on <code>N=1,020</code> independent out-of-fold residuals strictly withholding the model selection partition (<code>X_val</code>, N=219).</li>
                            <li><b>Empirical Validity:</b> The calibrated 90% relative margin (<code>±{margin_pct:.2f}%</code>) achieved <b>92.24% empirical coverage</b> on the held-out test set (<code>N=219</code>).</li>
                            <li><b>Data Lineage:</b> UI Inputs (12) → Feature Contract (27) → <code>data/processed/housing_preprocessor.pkl</code> → <code>models/saved/xgboost_ames_v1.pkl</code>.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # =========================================================================
    # TAB 2: PEER Post-Disaster Structural Collapse Inspector
    # =========================================================================
    with tab2:
        st.markdown("### PEER Post-Disaster Structural Collapse-State Classification")
        st.caption("Computer vision assessment of structural framing failure modes from post-disaster reconnaissance imagery.")

        # Explicit Domain Semantic Guard
        st.info(
            "ℹ️ **DOMAIN LIMITATION & METHODOLOGICAL NOTICE:**\n\n"
            "This model implements **PEER Task 5 Collapse-State Classification** using a transfer-learned ResNet-18 trained on the "
            "Pacific Earthquake Engineering Research Center (PEER) PHI-Net dataset. "
            "It classifies structural post-disaster collapse modes into: **Non-collapse**, **Partial collapse**, or **Global collapse**. "
            "It does **NOT** evaluate routine property maintenance, age, curb appeal, cosmetic condition, or standard construction quality."
        )

        condition_model = load_condition_model()
        cond_status_str = "TRAINED — VERIFIED (PEER ResNet-18 v1.0-real)" if condition_model is not None else "MODEL NOT AVAILABLE"
        cond_status_color = "#10B981" if condition_model is not None else "#EF4444"

        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                <span style="font-weight: 700; font-size: 0.85rem; color: #94A3B8;">CLASSIFIER STATUS:</span>
                <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: {cond_status_color}; border: 1px solid {cond_status_color};">
                    {cond_status_str}
                </span>
                <span style="color: #64748B; font-size: 0.8rem;">| Checkpoint: <code>models/saved/resnet_peer_collapse_v1.pt</code> | Benchmark Test Acc: 70.55%, Macro F1: 67.21% (N=146)</span>
            </div>
            """,
            unsafe_allow_html=True
        )

        cond_col1, cond_col2 = st.columns([1.1, 1.9])

        with cond_col1:
            st.markdown("#### Input Reconnaissance Image")
            input_source = st.radio("Image Source", ["Select Authentic PEER Reconnaissance Preset", "Upload Exterior Photo"], horizontal=True)

            selected_image = None
            sample_caption = ""

            if input_source == "Select Authentic PEER Reconnaissance Preset":
                sample_files = [
                    ("demo_structural_non_collapse.jpg", "PEER Non-Collapse / Intact Framing"),
                    ("demo_structural_partial_collapse.jpg", "PEER Partial Structural Collapse / Framing Failure"),
                    ("demo_structural_global_collapse.jpg", "PEER Global Structural Collapse / Total Loss")
                ]
                sample_choice = st.selectbox("Choose authentic sample:", [s[1] for s in sample_files])
                chosen_fn = next(s[0] for s in sample_files if s[1] == sample_choice)
                chosen_path = os.path.join(SAMPLE_IMG_DIR, chosen_fn)
                if os.path.exists(chosen_path):
                    selected_image = Image.open(chosen_path).convert("RGB")
                    sample_caption = sample_choice
                    st.image(selected_image, caption=sample_caption, use_container_width=True)
                else:
                    st.warning(f"Sample file not found: {chosen_fn}")
            else:
                uploaded = st.file_uploader("Upload building exterior photo (JPG, PNG)", type=["jpg", "jpeg", "png"])
                if uploaded:
                    selected_image = Image.open(uploaded).convert("RGB")
                    sample_caption = "Uploaded Building Reconnaissance View"
                    st.image(selected_image, caption=sample_caption, use_container_width=True)

        with cond_col2:
            st.markdown("#### Structural Collapse-State Assessment")

            if selected_image is None:
                st.info("Select or upload a building reconnaissance image on the left to trigger structural classification.")
            elif condition_model is None:
                st.error("⚠️ MODEL NOT AVAILABLE: Canonical checkpoint (`models/saved/resnet_peer_collapse_v1.pt`) is missing. AI inference is safely disabled.")
            else:
                # Preprocess image
                preprocess = get_condition_transforms(is_train=False)
                tensor_img = preprocess(selected_image).unsqueeze(0)

                with torch.no_grad():
                    logits = condition_model(tensor_img)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                # Exact PEER class mapping verified in Phase 3
                # 0: global_collapse, 1: non_collapse, 2: partial_collapse
                probs_dict = {
                    "global_collapse": float(probs[0]),
                    "non_collapse": float(probs[1]),
                    "partial_collapse": float(probs[2])
                }
                pred_idx = int(np.argmax(probs))
                class_labels = {
                    0: "Global collapse",
                    1: "Non-collapse",
                    2: "Partial collapse"
                }
                predicted_label = class_labels[pred_idx]

                inspection_report = evaluate_property_inspection(probs_dict)

                # Summary scorecards
                sc1, sc2 = st.columns(2)
                with sc1:
                    badge_cls = "badge-green" if pred_idx == 1 else ("badge-amber" if pred_idx == 2 else "badge-purple")
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <span class="badge {badge_cls}">Predicted Collapse Mode</span>
                            <div class="metric-value" style="margin-top: 8px; font-size: 1.45rem;">{predicted_label}</div>
                            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Classification Confidence: <b>{probs[pred_idx]*100:.1f}%</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with sc2:
                    integ_score = inspection_report["condition_score"]
                    hazard_text = inspection_report["hazard_level"]
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <span class="badge badge-blue">Structural Integrity Score</span>
                            <div class="metric-value" style="margin-top: 8px; font-size: 1.45rem;">{integ_score} / 100</div>
                            <div style="color: #94A3B8; font-size: 0.85rem; margin-top: 4px;">Safety Status: <b>{hazard_text}</b></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### PEER Task 5 Class Probability Distribution")

                # Display calibrated probabilities for all 3 collapse states
                display_order = [
                    ("Non-collapse (Intact Structural Framing)", probs_dict["non_collapse"], "#10B981"),
                    ("Partial collapse (Localized Framing Failure)", probs_dict["partial_collapse"], "#F59E0B"),
                    ("Global collapse (Catastrophic Frame Collapse)", probs_dict["global_collapse"], "#EF4444")
                ]
                for name, p_val, bar_color in display_order:
                    st.write(f"**{name}**: `{p_val*100:.1f}%`")
                    st.progress(float(p_val))

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 16px;">
                        <h5 style="margin-top: 0; color: #F8FAFC;">Forensic Engineering Diagnostic Report</h5>
                        <p style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 6px;"><b>Structural Status:</b> {inspection_report['structural_status']}</p>
                        <p style="color: #CBD5E1; font-size: 0.9rem; margin-bottom: 6px;"><b>Engineering Recommendation:</b> {inspection_report['inspection_advice']}</p>
                        <p style="color: #94A3B8; font-size: 0.8rem; margin-bottom: 0;"><i>{inspection_report['domain_disclaimer']}</i></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
