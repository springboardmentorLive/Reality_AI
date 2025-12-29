import streamlit as st
from datetime import date

# PAGE CONFIG
st.set_page_config(
    page_title="Sidebar Practice",
    page_icon="📊",
    layout="wide",
)

# SESSION STATE INIT
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# SIDEBAR
with st.sidebar:

    st.markdown("## 🚀 My Streamlit App")
    st.caption("Sidebar Practice")

    st.divider()

    # USER PROFILE
    with st.expander("👤 User Profile", expanded=True):
        st.text_input("Username", value="guest_user")
        st.selectbox("Role", ["Viewer", "Editor", "Admin"])
        st.success("Status: Online")

    st.divider()

    # NAVIGATION
    st.markdown("### 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["Dashboard", "Analytics", "Settings", "About"],
        index=["Dashboard", "Analytics", "Settings", "About"].index(st.session_state.page),
    )
    st.session_state.page = page

    st.divider()

    # FILTERS
    st.markdown("### 🔍 Filters")

    date_range = st.date_input(
        "Date Range",
        value=(date(2024, 1, 1), date.today())
    )

    category = st.multiselect(
        "Select Categories",
        ["Sales", "Marketing", "Finance", "HR", "Tech"],
        default=["Sales", "Tech"]
    )

    score = st.slider(
        "Score Range",
        min_value=0,
        max_value=100,
        value=(30, 80)
    )

    st.divider()

    # FILE UPLOAD 
    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        accept_multiple_files=False
    )

    if uploaded_file:
        st.success("File uploaded successfully!")

    st.divider()

    # SETTINGS
    st.markdown("### ⚙️ App Settings")

    st.session_state.dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
    notifications = st.checkbox("Enable Notifications", value=True)

    st.selectbox(
        "Refresh Rate",
        ["5s", "15s", "30s", "1 min"],
        index=1
    )

    st.divider()

    # ACTION BUTTONS
    st.markdown("### 🛠 Actions")

    if st.button("🔄 Refresh Data"):
        st.toast("Data refreshed!", icon="✅")

    if st.button("🧹 Clear Filters"):
        st.toast("Filters cleared", icon="🧼")

    st.divider()

    # FOOTER
    st.caption("© 2025 My Company")
    st.caption("v1.0.0")

# MAIN CONTENT
st.title(st.session_state.page)

if st.session_state.page == "Dashboard":
    st.write("### 📊 Dashboard Overview")
    st.info("Summary metrics and KPIs appear here.")

elif st.session_state.page == "Analytics":
    st.write("### 📈 Analytics")
    st.write("Charts, graphs, and insights go here.")

elif st.session_state.page == "Settings":
    st.write("### ⚙️ Settings")
    st.write(f"Dark Mode: `{st.session_state.dark_mode}`")

elif st.session_state.page == "About":
    st.write("### ℹ️ About This App")
    st.write("""
    This app demonstrates an **advanced Streamlit sidebar**  
    with navigation, filters, uploads, settings, and state management.
    """)
