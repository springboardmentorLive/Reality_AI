import streamlit as st
from streamlit_option_menu import option_menu
st.title("Navigation horizontal Example")

# create top horizontal bar
selected = option_menu(
    menu_title=None,    
    options=["Home", "About", "Dashboard"],
    icons=["house", "info-circle", "bar-chart"],  # optional    
    menu_icon="cast",  # optional
    default_index=0,  # optional
    orientation="horizontal",
)
# Display content based on selection            
if selected == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is the home page content.")
elif selected == "About":
    st.header("About Us")
    st.write("This is the about page content.")
elif selected == "Dashboard":
    st.header("Dashboard")
    st.line_chart({"data": [1, 2, 3, 4, 5]})
    st.write("This is the dashboard content.")