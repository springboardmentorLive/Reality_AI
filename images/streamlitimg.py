import streamlit as st
from PIL import Image

st.title("Image Uploader and Viewer")

img = Image.open("IMG-20240129-WA0002.jpg")
st.image(img, caption="Default Image", use_column_width=True)

