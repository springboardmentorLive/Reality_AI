import streamlit as st
st.title("Navigation Side Bar Example")

# Create a sidebar
page = st.sidebar.selectbox("go to", ["Home", "About", "Contact"])
# Display content based on selection
if page == "Home":
    st.header("Welcome to the Home Page")
    st.write("This is the home page content.")
elif page == "About":
    st.header("About Us")
    st.write("This is the about page content.")
elif page == "Contact":
    st.header("Contact Us")
    st.write("This is the contact page content.")
    