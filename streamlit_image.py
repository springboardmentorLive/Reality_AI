## Date: 26/12/25

#----------------##
## Showing Image ##
#----------------##
# import streamlit as st
# from PIL import Image

# st.title("Streamlit Image Display")

# img = Image.open("C:\\Users\\saina\\Downloads\\figure1_context_timeline.png")
# st.image(img, caption="My Image", use_container_width=True)






#---------------##
## Upload Image ##
#---------------##
# import streamlit as st
# from PIL import Image

# st.title("Streamlit Upload Display")

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

# if uploaded_file:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_container_width=True)






#---------------##
## Upload PDF ##
#---------------##
# import streamlit as st
# from PIL import Image

# st.title("Streamlit Upload Display")

# uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

# if uploaded_file:
#     st.success("PDF Uploaded Successfully")
#     st.pdf(uploaded_file, height="stretch")





#--------------##
## Image Links ##
#--------------##
# import streamlit as st

# st.image("https://plus.unsplash.com/premium_photo-1766746551190-2f186c2f2360?q=80&w=725&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D", caption="Image from Internet")







#--------------##
## Image Gallery ##
#--------------##
import streamlit as st
from PIL import Image

col1,col2,col3 = st.columns(3)
# row1,row2 = st.

with col1:
    st.image(Image.open("C:\\Users\\saina\\Downloads\\figure1_context_timeline.png"))

with col2:
    st.image(Image.open("C:\\Users\\saina\\Downloads\\figure1_context_timeline.png"))

with col3:
    st.image(Image.open("C:\\Users\\saina\\Downloads\\figure1_context_timeline.png"))