import streamlit as st
from PIL import Image
import os

# Set page configuration
st.set_page_config(layout="wide", page_title="Image Gallery")

st.title("My Image Gallery")
st.markdown("---")

image_path = "sample.jpg"

if os.path.exists(image_path):
    image = Image.open(image_path)
    
    st.subheader("Gallery Collection")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.image(image, caption="Sample Image 1", use_container_width=True)
        st.image(image, caption="Sample Image 4", use_container_width=True)
        
    with col2:
        st.image(image, caption="Sample Image 2", use_container_width=True)
        st.image(image, caption="Sample Image 5", use_container_width=True)
        
    with col3:
        st.image(image, caption="Sample Image 3", use_container_width=True)
        st.image(image, caption="Sample Image 6", use_container_width=True)

else:
    st.error("Image 'sample.jpg' not found in the current directory.")
