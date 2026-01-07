import streamlit as st
from streamlit_option_menu import option_menu

st.title("Horizontal Navigation")

# Horizontal top menu
selected = option_menu(
    menu_title="Main Menu",
    options=["Home", "Dashboard", "Settings"],
    icons=["house", "bar-chart", "gear"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

if selected == "Home":
    st.header("Home page")
    st.write("Welcome!")
elif selected == "Dashboard":
    st.header("Dashboard page")
    st.line_chart([10,20,25,30,35,15])
elif selected == "Settings":
    st.header("Settings page")
    st.write("App configuration here.")