import streamlit as st
import json
import os

# Fungsi Load User
def load_users():
    if os.path.exists("users.json"):
        with open("users.json", "r") as f:
            return json.load(f)
    return {}

# Fungsi Login
def login():
    st.title("🔐 Login Aplikasi")
    users = load_users()

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success("Login Berhasil!!!")
            st.switch_page("pages/1_Home.py")
        else:
            st.error("Username atau Password Salah!!!")

# Fungsi logout
def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.success("Berhasil Logout!!!.")
    st.switch_page("Aplikasi_Penerimaan_Siswa_Baru.py")

# Fungsi proteksi halaman (hanya bisa diakses jika sudah login)
def protect_page():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        st.warning("⚠️ Silakan Login Terlebih Dahulu!!!")
        st.stop()

# Fungsi untuk konversi nilai ke kategori
def nilai_to_kategori(nilai):
    if nilai >= 80:
        return "Sangat Baik"
    elif nilai >= 70:
        return "Baik"
    elif nilai >= 60:
        return "Cukup"
    elif nilai >= 50:
        return "Kurang"
    else:
        return "Sangat Kurang"
