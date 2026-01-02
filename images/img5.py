import streamlit as st
from PIL import Image

col1, col2, col3 = st.columns(3)

with col1:
    img1 = Image.open("IMG-20240129-WA0002.jpg")
    st.image(img1, caption="Image 1", use_column_width=True)
with col2:
    img2 = Image.open("IMG-20240129-WA0002.jpg")
    st.image(img2, caption="Image 2", use_column_width=True)
with col3:
    img3 = Image.open("IMG-20240129-WA0002.jpg")
    st.image(img3, caption="Image 3", use_column_width=True)