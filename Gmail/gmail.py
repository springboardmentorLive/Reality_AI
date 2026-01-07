import streamlit as st
from fpdf import FPDF
import yagmail
import os

st.set_page_config(page_title="Interactive Report Generator", layout="wide")

st.title("Interactive Report Generator")

st.sidebar.header("User Input")
name = st.sidebar.text_input("Enter your name", "Mowleen")
age = st.sidebar.slider("Select your age", 0, 100, 25)
options = st.sidebar.multiselect(
    "Choose options",
    ["Option A", "Option B", "Option C"],
    default=["Option A"]
)
comments = st.sidebar.text_area("Additional comments", "Type here...")

# Real-Time display
st.header("Preview Report")
st.markdown(f"**Name:** {name}")
st.markdown(f"**Age:** {age}")
st.markdown(f"**SelectedOptions:** {', '.join(options) if options else 'None'}")
st.markdown(f"**Comments:** {comments}")

# Generate PDF
def generate_pdf(name, age, options, comments):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Interactive Report", ln = True, align = 'C')
    pdf.ln(10)
    pdf.set_font("Arial", size = 12)
    pdf.multi_cell(0, 10, txt = f"Name: {name}")
    pdf.multi_cell(0, 10, txt = f"Age: {age}")
    pdf.multi_cell(0, 10, txt = f"Selected Options: {', '.join(options) if options else 'None'}")
    pdf.multi_cell(0, 10, txt = f"Comments: {comments}")
    file_path = "report.pdf"
    pdf.output(file_path)
    return file_path
pdf_file = generate_pdf(name, age, options, comments)

# Download PDF
with open(pdf_file, "rb") as f:
    st.download_button(
        label="Download PDF",
        data=f,
        file_name="Mowleen_final_resume.pdf",
        mime="application/pdf"
    )

# Send Email

st.header("Send Report via Email")
sender_email = st.text_input("Sender Email")
sender_password = st.text_input("Sender App Password", type="password", help="Enter your 16-character Google App Password (not your regular Gmail password)")
receiver_email = st.text_input("Recipient Email")

if st.button("Send Email"):
    if not sender_email or not sender_password or not receiver_email:
        st.warning("Please fill in all email fields (Sender Email, App Password, and Recipient Email).")
    else:
        try:
            with st.spinner("Sending email..."):
                yag = yagmail.SMTP(user=sender_email, password=sender_password)
                yag.send(
                    to=receiver_email, 
                    subject="Your Interactive Report", 
                    contents= "Please find the attached report.",
                    attachments=pdf_file
                )
            st.success(f"Email sent successfully to {receiver_email}!")
        except Exception as e:
            error_message = str(e)
            if "Application-specific password required" in error_message or "Username and Password not accepted" in error_message:
                st.error("Authentication failed. Please ensure you are using a **Google App Password**, not your regular login password. \n\nGo to Google Account > Security > 2-Step Verification > App Passwords to create one.")
                st.markdown("[Learn more about App Passwords](https://support.google.com/accounts/answer/185833)")
            else:
                st.error(f"Failed to send email: {e}")

# Cleanup generated PDF
if os.path.exists(pdf_file):
    try:
        os.remove(pdf_file)
    except PermissionError:
        pass
