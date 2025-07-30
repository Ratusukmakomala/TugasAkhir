import streamlit as st
from utils import protect_page, logout

protect_page()

st.title("🔓 Logout")

confirm = st.radio("Apakah Anda Yakin ingin Logout???", ("Tidak", "Ya"))

if confirm == "Ya":
    logout()
else:
    st.info("Anda Tetap Login.")
    if st.button("Kembali ke Beranda"):
        st.switch_page("pages/1_Home.py")
