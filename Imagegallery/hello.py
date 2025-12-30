import streamlit as st
from PIL import Image

st.title("Image Display Example")

img = Image.open("images/Doraemon.jpg")

col1, col2, col3 = st.columns(3)

with col1:
    st.image(img, caption="Image 1", use_container_width=True)
with col2:
    st.image(img, caption="Image 2", use_container_width=True)
with col3:
    st.image(img, caption="Image 3", use_container_width=True)
