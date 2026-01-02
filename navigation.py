################
# Side NavBar #
################
# import streamlit as st

# st.title("My App with Navigation")

# # Sidebar for navigation
# page = st.sidebar.selectbox("Go to", ["Home", "Dashboard", "Settings"])

# # Show content based on selection
# if page == "Home":
#     st.header("Welcome to Home")
#     st.write("This is the home page content.")

# elif page == "Dashboard":
#     st.header("Dashboard")
#     st.write("Charts, graphs, and metrics go here.")

# elif page == "Settings":
#     st.header("Settings")
#     st.write("Change your app preferences here.")









##################################
# Horizontal NavBar Radio Button #
##################################
# import streamlit as st

# st.title("Top Navigation Bar Example")

# # Top navigation using radio buttons
# menu = st.radio(
#     "Navigation",
#     ["Home", "Dashboard", "Settings"],
#     horizontal=True
# )

# if menu == "Home":
#     st.subheader("Home Page")
#     st.write("Welcome!")

# elif menu == "Dashboard":
#     st.subheader("Dashboard")
#     st.line_chart([1, 5, 2, 6, 2, 7])

# elif menu == "Settings":
#     st.subheader("Settings")
#     st.write("Configure your app here.")










################################
# Vertical NavBar Radio Button #
################################
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