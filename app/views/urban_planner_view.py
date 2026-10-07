"""
Urban Planner & Satellite Image Segmentation View
Phase 4 Verified Integration:
1. SpaceNet 2 Building Footprint Segmentation powered by PyTorch U-Net (canonical checkpoint: unet_spacenet_v1.pt)
2. Rigorous spatial surface analytics: Building Footprint Coverage % and Non-Building Area % (never green space)
3. Transparent documentation: Real SpaceNet Las Vegas AOI development subset (N=24 train, N=6 val; no official public-test labels locally)
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import torch

from app.components.header import render_hero
from models.segmentation_unet import UNet, analyze_satellite_zone, calculate_segmentation_metrics
from app.config import SAVED_MODELS_DIR, RAW_DIR, PROCESSED_DIR, BASE_DIR


@st.cache_resource
def load_unet_model():
    """Loads canonical verified Phase 3 SpaceNet U-Net checkpoint."""
    model_path = os.path.join(SAVED_MODELS_DIR, "unet_spacenet_v1.pt")
    if not os.path.exists(model_path):
        return None
    try:
        model = UNet(in_channels=3, out_channels=1, features=[16, 32, 64, 128])
        state = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    except Exception:
        return None


def render_urban_planner_view():
    render_hero(
        title="Urban Planner & Building Footprint Segmentation",
        subtitle="Detect structural building footprints and analyze built-up surface coverage using PyTorch U-Net on SpaceNet satellite imagery.",
        badge_text="Geospatial AI & Segmentation",
        badge_type="badge-green"
    )

    # Mandatory Evaluation Status & Domain Disclaimer
    st.info(
        "🛰️ **EVALUATION STATUS & GEOSPATIAL SCOPE NOTICE:**\n\n"
        "- **Dataset:** Authentic SpaceNet 2 Las Vegas AOI (Area of Interest) high-resolution 3-band satellite imagery.\n"
        "- **Dataset Scale:** Limited development subset consisting of **24 training chips** and **6 held-out validation chips**.\n"
        "- **Evaluation Status:** Evaluated on the $N=6$ held-out validation chips (Mean IoU: **0.3593 / 35.93%**, Mean Dice: **0.4985 / 49.85%**).\n"
        "- **Benchmark Scope:** **No official public-test ground-truth labels are available locally.** This is a verified development result, not a generalized global benchmark score.\n"
        "- **Semantic Precision:** The complement of building footprint coverage is **Non-Building Area** (roads, open dirt, driveways, yards). It must **NOT** be interpreted as vegetation, parks, or ecological green space."
    )

    unet_model = load_unet_model()
    unet_status_str = "TRAINED — VERIFIED (SpaceNet U-Net v1.0-real)" if unet_model is not None else "MODEL NOT AVAILABLE"
    unet_status_color = "#10B981" if unet_model is not None else "#EF4444"

    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
            <span style="font-weight: 700; font-size: 0.85rem; color: #94A3B8;">MODEL STATUS:</span>
            <span class="badge" style="background: rgba(16, 185, 129, 0.15); color: {unet_status_color}; border: 1px solid {unet_status_color};">
                {unet_status_str}
            </span>
            <span style="color: #64748B; font-size: 0.8rem;">| Checkpoint: <code>models/saved/unet_spacenet_v1.pt</code> | Validation Mean IoU: 35.93%, Dice: 49.85% (N=6 chips)</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    col_ctrl, col_display = st.columns([1.1, 2.3])

    with col_ctrl:
        st.markdown("### 🛰️ Satellite Imagery Source")
        img_source = st.radio("Select Image Source", ["SpaceNet Las Vegas Validation Chips", "Upload Custom Satellite Image"], horizontal=True)

        selected_rgb = None
        selected_gt_mask = None
        chip_identifier = "custom"
        tile_name = "Custom Upload"

        if img_source == "SpaceNet Las Vegas Validation Chips":
            val_options = [
                ("1041", "Las Vegas Chip 1041 (Dense Residential, Val IoU: 31.5%)"),
                ("1042", "Las Vegas Chip 1042 (High Density Residential, Val IoU: 32.6%)"),
                ("1047", "Las Vegas Chip 1047 (Sparse / Complex Terrain, Val IoU: 5.4%)"),
                ("1048", "Las Vegas Chip 1048 (Suburban Sector, Val IoU: 53.7%)"),
                ("1049", "Las Vegas Chip 1049 (Suburban Sector, Val IoU: 64.7%)"),
                ("1051", "Las Vegas Chip 1051 (Residential Cul-de-sac, Val IoU: 27.7%)")
            ]

            choice = st.selectbox("Select Validation Chip:", [t[1] for t in val_options])
            chosen_id = next(t[0] for t in val_options if t[1] == choice)
            chip_identifier = chosen_id
            tile_name = choice

            p_img = os.path.join(RAW_DIR, "spacenet", "research_train", "images", f"spacenet_vegas_chip_{chosen_id}.png")
            p_mask = os.path.join(RAW_DIR, "spacenet", "research_train", "masks", f"spacenet_vegas_chip_{chosen_id}_mask.png")

            if os.path.exists(p_img):
                selected_rgb = Image.open(p_img).convert("RGB")
            if os.path.exists(p_mask):
                selected_gt_mask = Image.open(p_mask).convert("L")
        else:
            uploaded = st.file_uploader("Upload satellite/aerial photo (256x256 recommended)", type=["png", "jpg", "jpeg", "tif"])
            if uploaded:
                selected_rgb = Image.open(uploaded).convert("RGB").resize((256, 256))

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### ⚙️ Segmentation Parameters")
        threshold = st.slider("Footprint Probability Threshold (τ)", min_value=0.1, max_value=0.9, value=0.5, step=0.05, help="Cutoff probability for binary building footprint segmentation.")

    with col_display:
        if selected_rgb is None:
            st.info("Select a SpaceNet satellite chip on the left to run building footprint segmentation.")
        elif unet_model is None:
            st.error("⚠️ MODEL NOT AVAILABLE: Canonical U-Net checkpoint (`models/saved/unet_spacenet_v1.pt`) is missing. Live segmentation inference is safely disabled.")
            col_prev1, col_prev2 = st.columns(2)
            with col_prev1:
                st.image(selected_rgb, caption=f"Selected Chip: {tile_name}", use_container_width=True)
            with col_prev2:
                if selected_gt_mask is not None:
                    st.image(selected_gt_mask, caption="Ground Truth Building Footprint Mask (Vector Rasterized)", use_container_width=True)
        else:
            # Preprocess satellite image for U-Net
            img_np = np.array(selected_rgb.resize((256, 256)), dtype=np.float32) / 255.0
            tensor_in = torch.from_numpy(img_np.transpose((2, 0, 1))).unsqueeze(0)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            tensor_in = (tensor_in - mean) / std

            with torch.no_grad():
                logits = unet_model(tensor_in)
                probs = torch.sigmoid(logits).squeeze().cpu().numpy()

            pred_bin = (probs > threshold).astype(np.uint8)

            # Spatial surface analytics
            zone_info = analyze_satellite_zone(pred_bin)
            bldg_cov = zone_info["building_coverage_pct"]
            non_bldg_cov = zone_info["non_building_area_pct"]

            # Visual overlay: Highlight buildings in cyan/blue overlay: [0, 210, 255]
            overlay_np = np.array(selected_rgb.resize((256, 256))).copy()
            overlay_np[pred_bin == 1] = (overlay_np[pred_bin == 1] * 0.4 + np.array([0, 210, 255]) * 0.6).astype(np.uint8)
            overlay_img = Image.fromarray(overlay_np)

            # Metric Scorecards
            st.markdown(f"#### 📊 Surface Coverage Analytics: {tile_name.split('(')[0].strip()}")
            z1, z2, z3, z4 = st.columns(4)

            with z1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Building Footprint</div>
                        <div class="metric-value" style="color: #60A5FA;">{bldg_cov}%</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Built-Up Ground Area</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with z2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Non-Building Area</div>
                        <div class="metric-value" style="color: #34D399;">{non_bldg_cov}%</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">Open Ground / Roads / Parcels</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with z3:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Built Density Tier</div>
                        <div class="metric-value" style="font-size: 1.15rem; color: #C084FC;">{zone_info['zone_classification'].split(' ')[0]}</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">{zone_info['zone_classification']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with z4:
                if selected_gt_mask is not None:
                    gt_np = (np.array(selected_gt_mask.resize((256, 256), resample=Image.Resampling.NEAREST)) > 128).astype(np.float32)
                    metrics = calculate_segmentation_metrics(torch.from_numpy(probs), torch.from_numpy(gt_np), threshold=threshold)
                    iou_val = metrics["iou"] * 100.0
                    dice_val = metrics["dice"] * 100.0
                    stat_title = f"IoU: {iou_val:.1f}%"
                    stat_sub = f"Dice: {dice_val:.1f}% | Prec: {metrics['precision']*100:.1f}%"
                    sc_color = "#10B981"
                else:
                    stat_title = "Inference Mode"
                    stat_sub = "No ground truth mask"
                    sc_color = "#3B82F6"

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-title">Validation Quality</div>
                        <div class="metric-value" style="font-size: 1.25rem; color: {sc_color};">{stat_title}</div>
                        <div style="color: #94A3B8; font-size: 0.8rem; margin-top: 4px;">{stat_sub}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # Three visual inspection layers
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Visual Segmentation Layers")

            img_col1, img_col2, img_col3 = st.columns(3)

            with img_col1:
                st.image(selected_rgb.resize((256, 256)), caption="1. Authentic SpaceNet Satellite RGB", use_container_width=True)

            with img_col2:
                pred_mask_img = Image.fromarray(pred_bin * 255)
                st.image(pred_mask_img, caption=f"2. U-Net Predicted Footprint Mask (τ={threshold})", use_container_width=True)

            with img_col3:
                st.image(overlay_img, caption="3. Geospatial Footprint Overlay (Cyan)", use_container_width=True)

            # Ground truth comparison & Error Analysis
            if selected_gt_mask is not None:
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("🔍 Inspect Ground Truth Mask & Error Analysis Quad Panel"):
                    gt_col1, gt_col2 = st.columns(2)
                    with gt_col1:
                        st.image(
                            selected_gt_mask.resize((256, 256), resample=Image.Resampling.NEAREST),
                            caption="Ground Truth SpaceNet Mask (Vector Rasterized)",
                            use_container_width=True
                        )
                    with gt_col2:
                        st.image(
                            pred_mask_img,
                            caption=f"U-Net Predicted Mask (IoU: {iou_val:.1f}%, Dice: {dice_val:.1f}%)",
                            use_container_width=True
                        )

                    error_panel_rel = f"models/saved/unet_error_analysis/vegas_{chip_identifier}_error_analysis.png"
                    error_panel_path = os.path.join(BASE_DIR, error_panel_rel)
                    if os.path.exists(error_panel_path):
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("##### Precomputed Error Analysis Quad Panel (RGB / Ground Truth / Predicted / Error Map)")
                        st.image(Image.open(error_panel_path), caption=f"Error Analysis: vegas_{chip_identifier}", use_container_width=True)
