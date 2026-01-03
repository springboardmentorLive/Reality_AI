import streamlit as st
import base64

st.title("Upload and Display PDF")

# Update file_uploader to accept only PDF
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    # Method 1: Using the built-in st.pdf (Streamlit 1.39+ / 2025 Standard)
    # Check if your version supports it; otherwise, use Method 2 below
    try:
        st.pdf(uploaded_file)
    except AttributeError:
        # Method 2: Fallback using HTML iframe (Standard for most environments)
        base64_pdf = base64.b64encode(uploaded_file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)