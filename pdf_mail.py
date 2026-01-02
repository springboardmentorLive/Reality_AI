import streamlit as st
from fpdf import FPDF
import yagmail
import os

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="Interactive Report Generator", layout="wide")

st.title("📄 Interactive Report Generator")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("User Input")
name = st.sidebar.text_input("Enter your name", "Anshu Sain")
age = st.sidebar.slider("Select your age", 0, 100, 25)
options = st.sidebar.multiselect(
    "Choose Profession",
    ["Student", "Startup", "MNC"],
    default=["MNC"]
)
comments = st.sidebar.text_area("Additional comments", "Type here...")

# -----------------------------
# Real-time display
# -----------------------------
st.header("Preview Report")
st.markdown(f"**Name:** {name}")
st.markdown(f"**Age:** {age}")
st.markdown(f"**Selected Options:** {', '.join(options) if options else 'None'}")
st.markdown(f"**Comments:** {comments}")

# -----------------------------
# Real-time display
# -----------------------------
def generate_pdf(name, age, options, comments):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", size=12)
    pdf.cell(0,10,"Interactive", ln=True, align="C")
    pdf.ln(10)
    pdf.multi_cell(0, 10, f"Name: {name}", ln=True)
    pdf.multi_cell(0, 10, f"Age: {age}", ln=True)
    pdf.multi_cell(0, 10, f"Selected Options: {', '.join(options) if options else 'None'}", ln=True)
    pdf.multi_cell(0, 10, f"Comments: {comments}", ln=True)
    
    file_path = "anshu.pdf"
    pdf.output(file_path)

    return file_path



pdf_file = generate_pdf("Anshu",22, "Student", comments)


# ----------------------------
# Download PDF
# ----------------------------
with open(pdf_file, "rb") as f:
    st.download_button(
        label="📥 Download PDF",
        data=f,
        file_name="anshu.pdf",
        mime="application/pdf"
    )


# ----------------------------
# Send Email
# ----------------------------
st.header("📧 Send Report via Email")
receiver_email = st.text_input("Enter recipient email")

if st.button("Send Email"):
    try:
        # Configure your email credentials here
        sender_email = "sainanshu40@gmail.com"
        sender_password = "pyyk gyhe sqfh jnkw"

        yag = yagmail.SMTP(sender_email, sender_password)
        yag.send(
            to=receiver_email,
            subject="Your Interactive Report",
            contents="Please find attached the report.",
            attachments=pdf_file
        )

        st.success(f"Email sent successfully to {receiver_email}!")
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")


# ----------------------------
# Clean Generated PDF
# ----------------------------
if os.path.exists(pdf_file):
    os.remove(pdf_file)
