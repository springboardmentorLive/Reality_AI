# RealtyAI – Smart Real Estate Insight Platform

RealtyAI is an AI-powered platform that leverages satellite imagery and tabular data to provide real estate insights.

## Features
- **Image Segmentation**: Segments satellite images into Urban, Vegetation, and Water regions.
- **Property Condition Classification**: Classifies property condition as New, Moderate, or Old.
- **Price Prediction**: Predicts house prices based on tabular features.
- **Market Trends**: Forecasts real estate price trends over time.

## Project Structure
```
RealtyAI/
├── app.py                      # Main Streamlit app
├── config/                     # Configuration settings
├── models/                     # AI/ML Model logic (Mock/Simulated)
├── preprocessing/              # Data preprocessing scripts
├── utils/                      # Visualization and metrics utilities
├── assets/                     # Images and assets
└── requirements.txt            # Python dependencies
```

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
