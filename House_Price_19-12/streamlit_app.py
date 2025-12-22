import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(
    page_title="House Rent Predictor",
    page_icon="🏠",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 30px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 20px;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 15px;
        font-size: 1.2rem;
        border-radius: 10px;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
    }
    .info-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_artifacts():
        model = joblib.load('models/rent_predictor_model.pkl')
        location_encoder = joblib.load('models/location_encoder.pkl')
        furnished_encoder = joblib.load('models/furnished_encoder.pkl')
        metadata = joblib.load('models/metadata.pkl')
        return model, location_encoder, furnished_encoder, metadata

def predict_rent(model, location_encoder, furnished_encoder, location, size, bhk, furnished):
    location_encoded = location_encoder.transform([location])[0]
    furnished_encoded = furnished_encoder.transform([furnished])[0]
    
    features = np.array([[location_encoded, size, bhk, furnished_encoded]])

    prediction = model.predict(features)[0]
    return prediction

def main():
    st.markdown('<p class="main-header">🏠 House Rent Predictor</p>', unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center; color: #666; margin-bottom: 30px;'>
        Predict your monthly rent based on location, size, and amenities
    </p>
    """, unsafe_allow_html=True)
    
    model, location_encoder, furnished_encoder, metadata = load_model_and_artifacts()
    
    if model is None:
        st.error("⚠️ Model not found! Please run `python main.py` first to train the model.")
        st.info("Run the following command in your terminal:")
        st.code("python main.py", language="bash")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📍 Location")
        location = st.selectbox(
            "Select Location",
            options=metadata['locations'],
            help="Choose the city where the property is located"
        )
        
        st.subheader("📐 Size (sq ft)")
        size = st.slider(
            "Property Size",
            min_value=300,
            max_value=3000,
            value=1000,
            step=50,
            help="Total area of the property in square feet"
        )
    
    with col2:
        st.subheader("🛏️ BHK")
        bhk = st.slider(
            "Number of BHK",
            min_value=1,
            max_value=5,
            value=2,
            step=1,
            help="Number of Bedrooms, Hall, Kitchen"
        )
        
        st.subheader("🛋️ Furnishing Status")
        furnished = st.selectbox(
            "Select Furnishing",
            options=metadata['furnished_options'],
            help="Current furnishing status of the property"
        )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Monthly Rent", use_container_width=True):
        with st.spinner("Calculating..."):
            prediction = predict_rent(
                model, location_encoder, furnished_encoder,
                location, size, bhk, furnished
            )
        
        # Display prediction with custom styling
        st.markdown(f"""
        <div class="prediction-box">
            <p style='font-size: 1.2rem; margin-bottom: 10px;'>Estimated Monthly Rent</p>
            <p class="prediction-value">₹{prediction:,.0f}</p>
            <p style='font-size: 0.9rem; margin-top: 10px; opacity: 0.8;'>per month</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show input summary
        st.markdown("### 📋 Property Details")
        details_col1, details_col2 = st.columns(2)
        with details_col1:
            st.info(f"**Location:** {location}")
            st.info(f"**Size:** {size} sq ft")
        with details_col2:
            st.info(f"**BHK:** {bhk}")
            st.info(f"**Furnishing:** {furnished}")
    


if __name__ == "__main__":
    main()
