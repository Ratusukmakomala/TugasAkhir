import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import protect_page

protect_page()

st.title("🏠 Dashboard Home Penerimaan Siswa Baru di SMA")
st.markdown("""
Dashboard ini memungkinkan Anda untuk:
- Mengunggah Data Nilai Siswa
- Melatih model klasifikasi
- Melihat hasil prediksi  Diterima/ Tidak
""")

# === Fungsi bantu untuk load hasil dari JSON jika session kosong ===
def load_hasil_dari_json():
    if os.path.exists("hasil_prediksi.json"):
        try:
            with open("hasil_prediksi.json", "r") as f:
                hasil_summary = json.load(f)

            df_data = []
            for jalur, nilai in hasil_summary.get("per_jalur", {}).items():
                df_data.append({
                    'Jalur': jalur,
                    'Prediksi': jalur,
                    'Diterima/Tidak Diterima': 'Diterima',
                    'Jumlah': nilai['diterima']
                })
                df_data.append({
                    'Jalur': jalur,
                    'Prediksi': jalur,
                    'Diterima/Tidak Diterima': 'Tidak Diterima',
                    'Jumlah': nilai['tidak_diterima']
                })

            df_hasil = pd.DataFrame(df_data).explode('Jumlah')
            df_hasil = df_hasil.loc[df_hasil.index.repeat(df_hasil['Jumlah'])].drop(columns='Jumlah')
            return df_hasil
        except Exception as e:
            st.warning("⚠ Gagal memuat hasil dari file: " + str(e))
    return None

# === Load dari session_state atau file ===
if 'hasil_prediksi' not in st.session_state:
    df_loaded = load_hasil_dari_json()
    if df_loaded is not None:
        st.session_state['hasil_prediksi'] = df_loaded

# === Tampilkan jika ada hasil ===
if 'hasil_prediksi' in st.session_state:
    df_hasil = st.session_state['hasil_prediksi']

    diterima_count = df_hasil[df_hasil['Diterima/Tidak Diterima'] == 'Diterima'].shape[0]
    tidak_diterima_count = df_hasil[df_hasil['Diterima/Tidak Diterima'] == 'Tidak Diterima'].shape[0]

    st.subheader("📈 Ringkasan Hasil Prediksi")
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Diterima"],
        y=[diterima_count],
        name='Diterima',
        marker_color='lightgreen',
        text=[diterima_count],
        textposition='auto'
    ))

    fig.add_trace(go.Bar(
        x=["Tidak Diterima"],
        y=[tidak_diterima_count],
        name='Tidak Diterima',
        marker_color='salmon',
        text=[tidak_diterima_count],
        textposition='auto'
    ))

    fig.update_layout(
        barmode='group',
        title='📊 Jumlah Siswa Diterima dan Tidak Diterima',
        xaxis_title='Status',
        yaxis_title='Jumlah Siswa',
        height=500,
        bargap=0.4
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Grafik Per Jalur ===
    st.subheader("📊 Diterima dan Tidak Diterima per Jalur")
    jalur_list = df_hasil['Prediksi'].unique()

    diterima_per_jalur = df_hasil[df_hasil['Diterima/Tidak Diterima'] == 'Diterima']['Prediksi'].value_counts().to_dict()
    tidak_diterima_per_jalur = df_hasil[df_hasil['Diterima/Tidak Diterima'] == 'Tidak Diterima']['Prediksi'].value_counts().to_dict()

    df_per_jalur = pd.DataFrame([
        {
            'Jalur': jalur,
            'Diterima': diterima_per_jalur.get(jalur, 0),
            'Tidak Diterima': tidak_diterima_per_jalur.get(jalur, 0)
        }
        for jalur in jalur_list
    ])

    fig_jalur = go.Figure()
    fig_jalur.add_trace(go.Bar(
        x=df_per_jalur['Jalur'],
        y=df_per_jalur['Diterima'],
        name='Diterima',
        marker_color='lightgreen',
        text=df_per_jalur['Diterima'],
        textposition='outside'
    ))
    fig_jalur.add_trace(go.Bar(
        x=df_per_jalur['Jalur'],
        y=df_per_jalur['Tidak Diterima'],
        name='Tidak Diterima',
        marker_color='salmon',
        text=df_per_jalur['Tidak Diterima'],
        textposition='outside'
    ))

    fig_jalur.update_layout(
        barmode='group',
        title='📊 Diterima dan Tidak Diterima per Jalur',
        xaxis_title='Jalur',
        yaxis_title='Jumlah Siswa',
        height=500
    )

    st.plotly_chart(fig_jalur, use_container_width=True)

else:
    st.warning("⚠ Belum Ada Hasil Prediksi Penerimaan Siswa.\nSilakan Jalankan Proses Prediksi Terlebih Dahulu di Halaman Analisis Prediksi.")