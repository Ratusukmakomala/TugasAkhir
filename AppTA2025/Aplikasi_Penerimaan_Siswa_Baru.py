import streamlit as st
import json
from utils import login

st.set_page_config(page_title="Aplikasi Penerimaan Siswa", layout="wide")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    st.switch_page("pages/1_Home.py")
