import streamlit as st

st.title("My App with Navigation")

# Sidebar for navigation
page = st.sidebar.selectbox(
    "Go to",
    ["Home", "Dashboard", "Settings"]
)

if page == "Home":
    st.header("Home Page")
    st.write("Welcome to the Home page.")

elif page == "Dashboard":
    st.header("Dashboard")
    st.write("This is the Dashboard page.")

elif page == "Settings":
    st.header("Settings")
    st.write("Change your settings here.")