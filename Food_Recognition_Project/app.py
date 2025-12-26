import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from config import settings
from utils.preprocessing import preprocess
from utils.inference import run_inference
import onnxruntime as ort

def main():
    st.set_page_config(page_title="FoodNet", page_icon="🍕", layout="wide")
    
    st.title("FoodNet 🍕🥩🍣")
    st.markdown("Upload an image to classify it as **Pizza**, **Steak**, or **Sushi**.")
    MODELS = ["lenet64", "tinyvgg", "resnet18"]

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        selected_model = st.selectbox("Select Model", MODELS)
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        analyze_button = st.button("Analyze Image", type="primary")

    # Main Layout
    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        # Display Image in Left Column (or split depending on design, prompt asked for right side results)
        # "In the right side display the prediction results... below it display the confidence rating"
        # I'll put image in col1, results in col2 as implied by "In the right side..."
        
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)

        if analyze_button:
            with st.spinner("Analyzing..."):
                try:
                    # Prepare the file for the request
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                    
                    image_array = preprocess(image, selected_model)
                    session = ort.InferenceSession(
                        str(settings.ONNX_PATH[selected_model]), 
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                    pred_index, prob = run_inference(session, image_array)
                    print(pred_index, prob)

                    
                    if pred_index is not None:
                        prediction = settings.CLASS_NAMES[pred_index]
                        confidence = prob[pred_index]
                        probabilities = prob
                        
                        with col2:
                            st.subheader("Results")
                            
                            # Outcome and Confidence
                            st.markdown(f"### Predicted: **{prediction.title()}**")
                            st.metric(label="Confidence", value=f"{confidence:.2%}")
                            
                            # Convert probabilities to DataFrame for Plotly
                            probs_df = pd.DataFrame({
                                "Class": settings.CLASS_NAMES,
                                "Probability": probabilities
                            })
                            
                        fig = px.pie(
                            probs_df, 
                            values="Probability", 
                            names="Class", 
                            title="Class Probabilities",
                            hole=0.4,
                            color_discrete_sequence=px.colors.sequential.RdBu
                        )
                        # Update layout for dark theme handled by plotly/streamlit mostly, but can force simple
                        fig.update_traces(textposition='inside', textinfo='percent+label')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Error: {pred_index}")
                        
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        with col1:
            st.info("Please upload an image to start.")

if __name__ == "__main__":
    main()