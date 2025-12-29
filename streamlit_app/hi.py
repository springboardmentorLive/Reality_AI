import streamlit as st

st.title("Vertical Navigation Bar Example")

# Vertical navigation using radio buttons
menu = st.radio(
    "Navigation",
    ["Home", "Dashboard", "Settings"]
)

if menu == "Home":
    st.subheader("Home Page")
    st.write("Welcome!")

elif menu == "Dashboard":
    st.subheader("Dashboard")
    st.line_chart([1, 5, 2, 6, 2, 7])

elif menu == "Settings":
    st.subheader("Settings")
    st.write("Configure your app here.")
