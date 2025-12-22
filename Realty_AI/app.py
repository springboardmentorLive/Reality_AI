import streamlit as st
import numpy as np
import pandas as pd
import os
from PIL import Image

# Import Custom Modules
from config import config
from preprocessing.image_preprocess import load_and_preprocess_image
from preprocessing.tabular_preprocess import preprocess_features
from models.segmentation_model import SegmentationModel
from models.condition_model import ConditionModel
from models.price_model import PriceModel
from models.trend_model import TrendModel
from utils.visualization import plot_segmentation, plot_price_trend

# Page Config
st.set_page_config(
    page_title="RealtyAI Platform",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern UI
st.markdown("""
    <style>
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
        color: #1a1a1a;
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        margin: 0 auto;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #1e3a8a;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    
    /* Cards/Containers */
    .stCard {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #2c3e50;
        font-weight: 700;
    }
    
    /* Reduce Padding for Full Screen feel */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Models Cache (Simulated)
@st.cache_resource
def load_models():
    return {
        'segmentation': SegmentationModel(),
        'condition': ConditionModel(),
        'price': PriceModel(),
        'trend': TrendModel()
    }

models = load_models()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/skyscrapers.png", width=100)
    st.title("RealtyAI 🧠")
    st.markdown("---")
    
    st.write("## Navigation")
    app_mode = st.radio("", 
        ["Home", "Image Segmentation", "Property Condition", "Price Prediction", "Market Trends"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.info("💡 **Tip:** Use the Price Prediction module for rough estimates based on tabular data.")

# --- Home Page ---
if app_mode == "Home":
    st.title("Welcome to RealtyAI")
    st.markdown("#### The Intelligent Real Estate Assistant")
    
    st.markdown("""
    <div style='background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);'>
        <p style='font-size: 1.1rem; color: #555;'>
            RealtyAI combines satellite imagery analysis, property condition assessment, and market trend forecasting to provide a comprehensive view of real estate value.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Method Cards with enhanced visibility
    st.markdown("""
    <style>
    .method-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e0e0e0;
        text-align: center;
        transition: transform 0.2s;
        height: 100%;
    }
    .method-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        border-color: #3498db;
    }
    .method-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    .method-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
        display: block;
    }
    .method-desc {
        color: #666;
        font-size: 0.95rem;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="method-card">
            <div class="method-icon">🌍</div>
            <span class="method-title">Segmentation</span>
            <p class="method-desc">Analyze land use and urban patterns from satellite imagery.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="method-card">
            <div class="method-icon">🏚️</div>
            <span class="method-title">Condition</span>
            <p class="method-desc">Automated assessment of property structural state.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="method-card">
            <div class="method-icon">💰</div>
            <span class="method-title">Valuation</span>
            <p class="method-desc">AI-driven market price prediction and estimation.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="method-card">
            <div class="method-icon">📈</div>
            <span class="method-title">Trends</span>
            <p class="method-desc">Forecast future market direction and values.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Dynamic Image Loading
    image_files = [f for f in os.listdir(config.SAMPLE_IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if image_files:
         sample_img_path = os.path.join(config.SAMPLE_IMAGES_DIR, image_files[0])
         st.image(sample_img_path, use_container_width=True)
    else:
         st.image("https://via.placeholder.com/800x400.png?text=RealtyAI+Demo", caption="RealtyAI Dashboard")

# --- Image Segmentation ---
elif app_mode == "Image Segmentation":
    st.title("🌍 Satellite Segmentation")
    st.markdown("Identify Urban, Vegetation, and Water features in satellite imagery.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        st.markdown("### Settings")
        show_overlay = st.checkbox("Show Overlay", value=True)
    
    with col2:
        if uploaded_file:
            image, image_arr = load_and_preprocess_image(uploaded_file)
            if image_arr is not None:
                st.image(image, caption="Source Image", use_container_width=True)
                
                if st.button("Generate Segmentation Mask", type="primary"):
                    with st.spinner("Processing..."):
                        mask = models['segmentation'].predict(image_arr)
                        fig = plot_segmentation(image_arr, mask, config.SEGMENTATION_CLASSES)
                        st.pyplot(fig)
        else:
            st.info("Please upload a satellite image to begin.")

# --- Property Condition ---
elif app_mode == "Property Condition":
    st.title("🏚️ Property Condition")
    st.markdown("Automated assessment of property exterior condition.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload Property Image", type=["jpg", "png", "jpeg"])
        
    with col2:
        if uploaded_file:
            image, image_arr = load_and_preprocess_image(uploaded_file)
            st.image(image, caption="Property Scan", use_container_width=True)
            
            if st.button("Analyze Condition", type="primary"):
                with st.spinner("Analyzing structural features..."):
                    condition, confidence = models['condition'].predict(image_arr)
                    
                    st.markdown("### Analysis Results")
                    if condition == 'New':
                        st.success(f"**Condition:** {condition}")
                    elif condition == 'Moderate':
                        st.warning(f"**Condition:** {condition}")
                    else:
                        st.error(f"**Condition:** {condition}")
                        
                    st.progress(confidence)
                    st.caption(f"Confidence Score: {confidence:.2%}")

# --- Price Prediction ---
elif app_mode == "Price Prediction":
    st.title("💰 Smart Valuation")
    st.markdown("Estimate property value based on key features.")
    
    with st.form("valuation_form"):
        col1, col2 = st.columns(2)
        with col1:
            sqft = st.number_input("Square Footage", min_value=100, value=2000, step=50)
            bedrooms = st.slider("Bedrooms", 1, 10, 3)
        with col2:
            year_built = st.number_input("Year Built", 1900, 2025, 2010)
            bathrooms = st.slider("Bathrooms", 1, 6, 2)
            
        submitted = st.form_submit_button("Calculated Estimate")
        
    if submitted:
        input_data = pd.DataFrame([{
            'SquareFootage': sqft,
            'Bedrooms': bedrooms,
            'Bathrooms': bathrooms,
            'YearBuilt': year_built
        }])
        
        input_data = preprocess_features(input_data)
        price = models['price'].predict(input_data)
        
        st.markdown("---")
        st.metric(label="Estimated Market Value", value=f"₹{price:,.2f}", delta="2.4% (vs area avg)")

# --- Market Trends ---
elif app_mode == "Market Trends":
    st.title("📈 Market Intelligence")
    st.markdown("Projected market trends for the upcoming period.")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        periods = st.select_slider("Forecast Horizon", options=[30, 90, 180, 365, 730], value=365, format_func=lambda x: f"{x} days")
    
    with col2:
        if st.button("Update Forecast"):
            df_trend = models['trend'].predict(periods=periods)
            fig = plot_price_trend(df_trend)
            st.plotly_chart(fig, use_container_width=True)
