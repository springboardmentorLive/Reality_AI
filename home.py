import streamlit as st

st.title(" Home")

st.write("""
This system predicts:
- Student Pass / Fail
- Final Marks

Using Machine Learning.
""")

col1, col2 = st.columns(2)

with col1:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
        width=200
    )
    if st.button("Go to Prediction"):
        st.switch_page("pages/predict.py")

with col2:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/942/942748.png",
        width=200
    )
    if st.button("About Project"):
        st.switch_page("pages/about.py")
