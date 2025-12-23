# RealtyAI: Smart Real Estate Insight Platform

## Purpose
**RealtyAI** is a unified analytics platform designed to empower real estate stakeholders—buyers, sellers, and agents—with data-driven insights. Its primary purpose is to bridge the gap between raw real estate data and actionable intelligence by combining numerical price analysis with visual property assessment in a single, interactive web application.

## Features
The application is divided into three core modules:

1.  **Market Trends Dashboard**:
    *   Visualizes historical real estate price trends.
    *   **Forecasting Engine**: Utilizes Time Series Analysis (Linear Regression) to predict market prices for the next 5 years (2025-2030), aiding in long-term investment planning.

2.  **Smart Price Predictor**:
    *   An interactive calculator allowing users to input specific property details (Area in sq ft, Number of Rooms).
    *   Uses a Machine Learning model (Linear Regression) trained on historical data to estimate the fair market value of a property instanty.

3.  **Visual Property Analysis**:
    *   **Automated Segmentation**: Uses Computer Vision (OpenCV) to process satellite or aerial imagery, highlighting structural footprints.
    *   **Condition Classification**: Analyzes image features (e.g., brightness/contrast patterns) to automatically classify property condition as "Old", "Average", or "New".
    *   Supports user-uploaded images for custom analysis.

## Libraries & Technologies used
*   **Streamlit**: For building the interactive web frontend.
*   **Pandas**: For data manipulation and CSV handling.
*   **Scikit-learn**: For machine learning algorithms (Linear Regression, SVM).
*   **OpenCV (cv2)**: For image processing, thresholding, and segmentation.
*   **Matplotlib**: For plotting charts and image visualizations.
*   **NumPy**: For numerical array operations.

## Advantages
*   **Unified Interface**: Consolidates multiple AI tools (Forecasting, Prediction, Vision) into one cohesive dashboard.
*   **Interactivity**: Unlike static reports, users can toggle parameters and upload their own data/images.
*   **Speed**: Lightweight models ensure instant feedback without heavy computational delays.
*   **Ease of Deployment**: Built on Streamlit, making it easy to host and share.

## Disadvantages
*   **Model Simplicity**: Currently uses basic Linear Regression and Classification based on simple features (brightness), which may oversimplify complex real-world variables.
*   **Data Limitations**: Relies on a small, static dataset (`real_estate_prices.csv`) rather than a live database.
*   **Segmentation Accuracy**: The threshold-based segmentation is sensitive to lighting conditions and may not perform as well as Deep Learning approaches (like U-Net) on complex backgrounds.

## Future Implementations
*   **Advanced Models**: Upgrade to Random Forest or XGBoost for price prediction to handle non-linear relationships.
*   **Deep Learning Computer Vision**: Implement a pre-trained U-Net or Mask R-CNN model for robust building segmentation.
*   **Live Data Integration**: Connect to real estate APIs (e.g., Zillow, Google Maps) for real-time pricing and imagery.
*   **Geospatial Mapping**: Add an interactive map layer to visualize price heatmaps by location.

## Difficulties Faced
*   **Code Quality Issues**: The initial codebase contained several syntax errors (e.g., typos in function names like `evtColor`, missing variable assignments), which required debugging before integration.
*   **Integration Complexity**: Merging standalone scripts with different execution flows (script-based print statements vs. interactive UI) required significant refactoring to ensure a smooth user experience.
*   **Data Path Management**: Standardizing file paths across different modules to ensure the app runs correctly regardless of the execution directory.
