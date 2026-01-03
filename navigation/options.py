import streamlit as st
from streamlit_option_menu import option_menu

st.title("Top Navigation Menu Example")

selected = option_menu(
    menu_title=None,  
    options=["Home", "Dashboard", "Settings"],
    icons=["house", "bar-chart", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="vertical"
)

if selected == "Home":
    st.subheader("Home Page")
    st.write("Welcome to the Home page")

elif selected == "Dashboard":
    st.subheader("Dashboard")
    st.line_chart([10, 20, 20, 25, 15])

elif selected == "Settings":
    st.subheader("Settings")
    st.write("Configure your app here")