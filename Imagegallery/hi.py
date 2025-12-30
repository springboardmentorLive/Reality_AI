import streamlit as st
from PIL import Image
import os

st.title("🖼️ Image Gallery")

image_folder = "images"

# Read only image files
images = [img for img in os.listdir(image_folder) if img.lower().endswith(('.jpg', '.png', '.jpeg'))]

cols = st.columns(2)  # 3 images per row

for index, image_name in enumerate(images):
    image_path = os.path.join(image_folder, image_name)
    img = Image.open(image_path)

    with cols[index % 2]:
        st.image(img, caption=image_name, use_container_width=True)
