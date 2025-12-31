import streamlit as st
from fpdf import FPDF
import yagmail
import os

# Page config
st.set_page_config(page_title="Interactive Report Generator", layout="wide")
st.title("Interactive Report Generator")

# Sidebar inputs
st.sidebar.header("User Input")
name = st.sidebar.text_input("Enter your name", "Prasanna")
age = st.sidebar.slider("Select your age", 0, 100, 25)
options = st.sidebar.multiselect(
    "Choose options", ["Option A", "Option B", "Option C"], default=["Option A"]
)
comments = st.sidebar.text_area("Additional comments", "Type here...")

# Preview
st.header("Preview Report")
st.markdown(f"**Name:** {name}")
st.markdown(f"**Age:** {age}")
st.markdown(f"**Selected Options:** {', '.join(options) if options else 'None'}")
st.markdown(f"**Comments:** {comments}")

# Ensure resumes folder exists
os.makedirs("resumes", exist_ok=True)

# PDF Generator
def generate_pdf(name, age, options, comments):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ✅ SAFE page width calculation (FIX)
    page_width = pdf.w - 2 * pdf.l_margin

    pdf.set_font("Arial", "B", 16)
    pdf.cell(page_width, 10, "Interactive Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", size=12)
    pdf.multi_cell(page_width, 10, f"Name: {name}")
    pdf.multi_cell(page_width, 10, f"Age: {age}")
    pdf.multi_cell(
        page_width,
        10,
        f"Selected Options: {', '.join(options) if options else 'None'}"
    )
    pdf.multi_cell(page_width, 10, f"Comments: {comments}")

    file_path = "resumes/PRASANNA(1).pdf"
    pdf.output(file_path)
    return file_path

pdf_file = generate_pdf(name, age, options, comments)

# Download button
with open(pdf_file, "rb") as f:
    st.download_button(
        label="Download PDF",
        data=f,
        file_name="PRASANNA(1).pdf",
        mime="application/pdf"
    )

# Email section
st.header("Send Report via Email")
receiver_email = st.text_input("Enter recipient email")

if st.button("Send Email"):
    try:
        sender_email = "prasannakuchipudi99@gmail.com"
        sender_password = "wcre rciq leci umhu"  # Gmail App Password

        yag = yagmail.SMTP(sender_email, sender_password)
        yag.send(
            to=receiver_email,
            subject="Your Interactive Report",
            contents="Please find attached the report.",
            attachments=pdf_file
        )

        st.success(f"Email sent successfully to {receiver_email}!")

        # Delete after sending
        if os.path.exists(pdf_file):
            os.remove(pdf_file)

    except Exception as e:
        st.error(f"Failed to send email: {e}")
