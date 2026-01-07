import streamlit as st
import base64
from PIL import Image

st.title("Upload and Display PDF")

uploaded_pdf = st.file_uploader("Upload a Pdf", type="pdf")

if uploaded_pdf:
    st.success("PDF uploaded successfully")
    st.write("File Name:", uploaded_pdf.name)
    st.write("File Size:", uploaded_pdf.size, "bytes")
    
    with st.spinner('Displaying PDF...'):
        base64_pdf = base64.b64encode(uploaded_pdf.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)