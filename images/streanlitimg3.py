import streamlit as st
from PIL import Image
import fitz

st.title("Image Uploader and Viewer")

uploaded_file = st.file_uploader("Choose an  PDF...", type=["pdf"])
if uploaded_file is not None:
    st.success("File uploaded successfully!")

    st.write("Displaying PDF content as images:", uploaded_file.name)
    st.write("file size", uploaded_file.size, "bytes")

    text = uploaded_file.read().decode("utf-8")
    st.text(text)




    '''if uploaded_file.type == "application/pdf":
        # Handle PDF file
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            st.image(img, caption=f"Page {page_num + 1}", use_column_width=True)
    else:
        # Handle image file
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)'''