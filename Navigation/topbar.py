import streamlit as st

st.title("Top Navigation Bar Example")

menu = st.radio("Navigation", ["Home", "Dashboard", "Settings"], horizontal=False)

if menu == "Home":
    st.header("Home Page")
    st.write("Home content goes here.")
elif menu == "Dashboard":
    st.header("Dashboard")
    st.write("Dashboard content goes here.")
elif menu == "Settings":
    st.header("Settings")
    st.write("Settings content goes here.")