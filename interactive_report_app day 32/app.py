import streamlit as st
from fpdf import FPDF
import yagmail
import os
import tempfile

# 1. Page configuration
st.set_page_config(page_title="Interactive Report Generator", layout="wide")
st.title("Interactive Report Generator")

# 2. Sidebar Inputs
st.sidebar.header("User Input")
name = st.sidebar.text_input("Enter your name", "John Doe")
age = st.sidebar.slider("Select your age", 0, 100, 25)
options = st.sidebar.multiselect(
    "Choose options",
    ["Option A", "Option B", "Option C"],
    default=["Option A"]
)
comments = st.sidebar.text_area("Additional comments", "Type here...")

# 3. Real-time Preview
st.header("Preview Report")
st.markdown(f"**Name:** {name}")
st.markdown(f"**Age:** {age}")
st.markdown(f"**Selected Options:** {', '.join(options) if options else 'None'}")
st.markdown(f"**Comments:** {comments}")

# 4. Function to Generate PDF (FIXED)
def generate_pdf(name, age, options, comments):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    usable_width = pdf.w - pdf.l_margin - pdf.r_margin

    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(usable_width, 10, "Interactive Report", ln=True, align="C")
    pdf.ln(10)

    # Content
    pdf.set_font("Arial", size=12)

    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width, 10, f"Name: {name}")

    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width, 10, f"Age: {age}")

    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        usable_width,
        10,
        f"Selected Options: {', '.join(options) if options else 'None'}"
    )

    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(usable_width, 10, f"Comments: {comments}")

    # Save to temporary file (Streamlit-safe)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# Generate PDF
pdf_file = generate_pdf(name, age, options, comments)

# 5. Download Button
with open(pdf_file, "rb") as f:
    st.download_button(
        label="📥 Download PDF",
        data=f,
        file_name="Report.pdf",
        mime="application/pdf"
    )

# 6. Send Email
st.header("📧 Send Report via Email")
receiver_email = st.text_input("Enter recipient email")

if st.button("Send Email"):
    try:
        # ⚠️ Configure your credentials
        sender_email = "sivasainath2121@gmail.com"
        sender_password = "rjjn bsha ysis nxnk"  # Use Gmail App Password

        yag = yagmail.SMTP(sender_email, sender_password)
        yag.send(
            to=receiver_email,
            subject="Your Interactive Report",
            contents="Please find attached the report generated from the Streamlit app.",
            attachments=pdf_file
        )
        st.success(f"Email sent successfully to {receiver_email}!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

# Cleanup temp file safely
if os.path.exists(pdf_file):
    os.remove(pdf_file)
