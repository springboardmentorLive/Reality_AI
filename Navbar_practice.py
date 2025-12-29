import streamlit as st

# PAGE CONFIG
st.set_page_config(
    page_title="Streamlit Navigation Bar",
    layout="wide",
)

# SESSION STATE
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# CSS FOR NAVBAR
st.markdown("""
<style>
.navbar {
    background-color: #0f172a;
    padding: 0.8rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 10px;
    margin-bottom: 2rem;
}

.nav-left {
    font-size: 22px;
    font-weight: 700;
    color: white;
}

.nav-right button {
    background-color: transparent;
    border: none;
    color: #cbd5f5;
    font-size: 16px;
    margin-left: 15px;
    cursor: pointer;
}

.nav-right button:hover {
    color: white;
}

.active {
    color: #38bdf8 !important;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# NAVBAR
nav_left, nav_right = st.columns([3, 5])

with nav_left:
    st.markdown(
        '<div class="navbar"><div class="nav-left">🚀 My App</div></div>',
        unsafe_allow_html=True
    )

with nav_right:
    c1, c2, c3, c4 = st.columns(4)

    if c1.button("📊 Dashboard"):
        st.session_state.page = "Dashboard"

    if c2.button("📈 Analytics"):
        st.session_state.page = "Analytics"

    if c3.button("⚙️ Settings"):
        st.session_state.page = "Settings"

    if c4.button("ℹ️ About"):
        st.session_state.page = "About"

# PAGE CONTENT 
st.divider()

if st.session_state.page == "Dashboard":
    st.title("📊 Dashboard")
    st.write("This is the dashboard page.")

elif st.session_state.page == "Analytics":
    st.title("📈 Analytics")
    st.write("Analytics and charts go here.")

elif st.session_state.page == "Settings":
    st.title("⚙️ Settings")
    st.write("Application settings page.")

elif st.session_state.page == "About":
    st.title("ℹ️ About")
    st.write("This is a navigation bar learning example.")

# FOOTER
st.markdown("---")
st.caption("Navigation Bar Learning Demo")
