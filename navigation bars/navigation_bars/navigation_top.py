import streamlit as st
st.title("Navigation Side Bar Example")

# create top navigation bar
page = st.radio("Navigate to", ["Home", "About", "Dashboard"], horizontal=True)
# Display content based on selection
if page == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is the home page content.")  
elif page == "About":
    st.header("About Us")
    st.write("This is the about page content.") 
elif page == "Dashboard":
    st.header("Dashboard")
    st.line_chart({"data": [1, 2, 3, 4, 5]})
    st.write("This is the dashboard content.")