import streamlit as st
import os
import pandas as pd
from PIL import Image
import dataset_creator
import trainer
import recognizer
import base64

st.set_page_config(
    page_title="Face Recognition Attendance",
    layout="wide",
    page_icon="📸",
    initial_sidebar_state="expanded"
)

def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode( f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg("assets/bg.jpeg")

st.markdown(
    """
<style>
.block-container{
    background: rgba(255,255,255,0.85)
    padding: 2rem;
    border-radius: 20px;
}
</style>
""",unsafe_allow_html=True
)

st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ["Home", " Register Face", " Train Model",
     "Take Attendance", "Dataset Gallery", "Attendance","Remove Student"]
)

# HOME
if menu == "Home":
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
            <h1 style='font-size:46px;'>Face Recognition Attendance</h1>
            <p style='font-size:18px;'>Smart and simple way to record attendance using face recognition </p>""",unsafe_allow_html=True)
        st.success("Fast | Secure  | Real-Time")
        
    with col2:
        st.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=300)
    st.markdown("---")

# REGISTER
elif menu == " Register Face":
    st.header("Register Student")
    st.caption("Ensure good lighting and clear face visibility")
    st.image("https://tse2.mm.bing.net/th/id/OIP.SlvbXaGSOucTDNJPwonceQHaEw?pid=Api&P=0&h=220",width=200)
    student_id = st.number_input("Enter Student ID", min_value=1)

    if st.button("Capture Face"):
        st.warning("Camera will open in a separate window.Do not close streamlit.")
        dataset_creator.create_dataset(student_id)
        st.success("Face registered successfully")

# TRAIN
elif menu == " Train Model":

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
            <h1 style='font-size:46px;'>Train Model</h1>
            <p style='font-size:18px;'>Update the system after adding new students </p>""",unsafe_allow_html=True)
        
    with col2:
        st.image("https://tse4.mm.bing.net/th/id/OIP.8ODHWDvCgZzciFqT3O3acQHaHa?pid=Api&P=0&h=220", width=300)
    if st.button("Train"):
        trainer.train_model()
        st.success("Model trained successfully")
    st.markdown("---")

# ATTENDANCE
elif menu == "Take Attendance":

    st.markdown("""
    <h1 style='font-size:40px;'>Take Attendance</h1>
    <p style='font-size:18px;'>
    Attendance will be recorded automatically using face recognition.
    </p>
    """, unsafe_allow_html=True)

    st.image(
        "https://tse2.mm.bing.net/th/id/OIP.OVP-3Q4ARCNpV4pXeCaSxQHaHa?pid=Api&P=0&h=220",width=300
    )

    st.markdown("### Instructions")
    st.markdown("""
    - Ensure the camera is properly connected  
    - Students should face the camera clearly  
    - Attendance will be recorded automatically  
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("▶ Start Attendance"):
            st.warning("Camera will open in a separate window")
            recognizer.recognize_student()
            st.success("Attendance session completed")

    with col2:
        st.info("Press **Start Attendance** when everyone is ready")

    st.markdown("---")
    st.caption("Tip: Good lighting improves recognition accuracy")

# GALLERY
elif menu == "Dataset Gallery":
    st.header("Dataset Gallery")

    if not os.path.exists("dataset"):
        st.warning("Dataset folder not found")
    else:
        images = sorted(os.listdir("dataset"))

        student_ids = sorted(set(img.split("_")[0] for img in images))
        selected_id = st.selectbox("Select Student ID", student_ids)
        student_images = [
            img for img in images if img.startswith(f"{selected_id}_")
        ]
        st.caption(f"Showing{len(student_images)} images")

        cols = st.columns(5)

        for i, img in enumerate(student_images[:25]):
            image = Image.open(os.path.join("dataset", img))
            cols[i % 5].image(
                image,
                use_container_width=True
            )

# ATTENDANCE LOG
elif menu == "Attendance":
    st.header("Attendance Records")
    st.image("https://static.vecteezy.com/system/resources/previews/016/422/068/original/attendance-icon-design-free-vector.jpg",
        width=200
    )

    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv", header=None)
        df.columns = ["Student ID", "Time", "Status"]
        st.dataframe(df)

        st.download_button(
            "Download CSV",
            df.to_csv(index=False),
            "attendance.csv",
            "text/csv"
        )
    else:
        st.warning("No attendance records yet")

#dlt
elif menu == "Remove Student":
    st.markdown("<h1> Remove Student</h1>",unsafe_allow_html=True)
    st.write("Remove a registered student from the system")
    student_id = st.text_input("Enter student ID")

    if st.button("Delete Student"):
        if student_id:
            deleted = False
            for img in os.listdir("dataset"):
                if img.startswith(student_id + "_"):
                    os.remove(os.path.join("dataset", img))
                    deleted = True
            if deleted:
                st.success(f"Student {student_id} removed successfully")
                st.info("Please retrain the system after deletion")
            else:
                st.error("Student ID not found")

