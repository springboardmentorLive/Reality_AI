import streamlit as st
from PIL import Image

st.title("Image Uploader and Viewer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png","pdf"])
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)