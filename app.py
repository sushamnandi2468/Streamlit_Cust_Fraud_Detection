import app1
import app2
import streamlit as st
import base64
from pathlib import Path

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

sidebar_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='200' height='160'>".format(
    img_to_bytes("DRR.PNG")
)
st.sidebar.markdown(
    sidebar_html, unsafe_allow_html=True,
)

PAGES = {
    "App1": app1,
    "App2": app2
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()