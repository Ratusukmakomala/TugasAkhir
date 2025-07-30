from datetime import datetime
from PIL import Image
from utils import protect_page

import os
import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

protect_page()

st.title("📂 History Analisis Prediksi")

folder_path = "backup_prediksi"
os.makedirs(folder_path, exist_ok=True)

# === Ambil semua file Excel dan JSON di folder backup ===
backup_files = sorted(os.listdir(folder_path), reverse=True)
backup_sets = {}

for file in backup_files:
    if file.endswith(".xlsx"):
        key = file.replace("hasil_", "").replace(".xlsx", "")
        backup_sets.setdefault(key, {})["excel"] = os.path.join(folder_path, file)
    elif file.startswith("hasil_summary_") and file.endswith(".json"):
        key = file.replace("hasil_summary_", "").replace(".json", "")
        backup_sets.setdefault(key, {})["summary"] = os.path.join(folder_path, file)
    elif file.startswith("model_info_") and file.endswith(".json"):
        key = file.replace("model_info_", "").replace(".json", "")
        backup_sets.setdefault(key, {})["model"] = os.path.join(folder_path, file)
    elif file.startswith("classification_report_") and file.endswith(".json"):
        key = file.replace("classification_report_", "").replace(".json", "")
        backup_sets.setdefault(key, {})["report"] = os.path.join(folder_path, file)
    elif file.startswith("confusion_matrix_") and file.endswith(".png"):
        key = file.replace("confusion_matrix_", "").replace(".png", "")
        backup_sets.setdefault(key, {})["cmatrix"] = os.path.join(folder_path, file)
    elif file.startswith("pie_summary_") and file.endswith(".json"):
        key = file.replace("pie_summary_", "").replace(".json", "")
        backup_sets.setdefault(key, {})["pie"] = os.path.join(folder_path, file)
    elif file.startswith("kuota_vs_prediksi_") and file.endswith(".json"):
        key = file.replace("kuota_vs_prediksi_", "").replace(".json", "")
        backup_sets.setdefault(key, {})["kvp"] = os.path.join(folder_path, file)
    elif file.startswith("kuota_vs_status_") and file.endswith(".json"):
        key = file.replace("kuota_vs_status_", "").replace(".json", "")
        backup_sets.setdefault(key, {})["kvs"] = os.path.join(folder_path, file)
    

# === Pilih riwayat analisis berdasarkan timestamp ===
if backup_sets:
    selected_key = st.selectbox("Pilih Waktu Backup", list(backup_sets.keys()))
    selected = backup_sets[selected_key]

    st.markdown(f"### 🧾 Riwayat Analisis: {selected_key}")

    # === History Hasil Prediksi ===
    if "excel" in selected:
        df = pd.read_excel(selected["excel"])
        st.subheader("📑 Hasil Prediksi")
        st.dataframe(df)

    # === History Classification Report ===
    if "report" in selected:
        try:
            with open(selected["report"], "r") as f:
                report_data = json.load(f)
                st.subheader("📋 Classification Report")
                st.dataframe(pd.DataFrame(report_data).transpose())
        except Exception as e:
            st.error(f"❌ Gagal memuat classification report: {e}")

    # === History Parameter dan Akurasi ===
    if "model" in selected:
        model_info = {}
        try:
            with open(selected["model"], "r") as f:
                content = f.read().strip()
                if content:
                    model_info = json.loads(content)
                else:
                    st.warning("⚠ File model kosong: " + selected["model"])
        except Exception as e:
            st.error(f"❌ Gagal memuat model info: {e}")

        if model_info:
            st.subheader("🏆 Parameter Model dan Akurasi")
            st.json(model_info)
    
    # === History Confusion Matrix ===
    if "cmatrix" in selected:
        try:
            st.subheader("📌 Confusion Matrix")
            cm_image = Image.open(selected["cmatrix"])
            st.image(cm_image, caption="Confusion Matrix", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat confusion matrix: {e}")
    
    # === History Proporsi Diterima dan Tidak Diterima ===
    if "pie" in selected:
        try:
            with open(selected["pie"], "r") as f:
                pie_data_list = json.load(f)
                pie_data = {item["Status"]: item["Jumlah"] for item in pie_data_list}

            st.subheader("🧩 Proporsi Diterima dan Tidak Diterima")
            fig_pie = go.Figure(data=[
                go.Pie(
                    labels=list(pie_data.keys()),
                    values=list(pie_data.values()),
                    hole=0.4,
                    marker=dict(colors=['lightgreen', 'salmon']),
                    textinfo='label+percent',
                    insidetextorientation='radial'
                )
            ])
            fig_pie.update_layout(title="Proporsi Siswa Diterima dan Tidak Diterima")
            st.plotly_chart(fig_pie, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat pie chart: {e}")
            
    # === History Kuota vs Prediksi per Jalur ===
    if "kvp" in selected:
        try:
            df_kvp = pd.read_json(selected["kvp"])
            st.subheader("📊 Kuota vs Prediksi per Jalur")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_kvp['Jalur'], y=df_kvp['Kuota'], name='Kuota', marker_color='goldenrod'))
            fig.add_trace(go.Bar(x=df_kvp['Jalur'], y=df_kvp['Prediksi'], name='Prediksi', marker_color='deepskyblue'))
            fig.update_layout(barmode='group', title='Kuota vs Prediksi per Jalur', xaxis_title='Jalur', yaxis_title='Jumlah')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat grafik kuota vs prediksi: {e}")

    # === History Kuota, Diterima, dan Tidak Diterima per Jalur ===
    if "kvs" in selected:
        try:
            df_kvs = pd.read_json(selected["kvs"])
            st.subheader("📊 Kuota, Diterima, dan Tidak Diterima per Jalur")
            fig_status = go.Figure()
            fig_status.add_trace(go.Bar(x=df_kvs['Jalur'], y=df_kvs['Kuota'], name='Kuota', marker_color='lightseagreen'))
            fig_status.add_trace(go.Bar(x=df_kvs['Jalur'], y=df_kvs['Diterima'], name='Diterima', marker_color='lightcoral'))
            fig_status.add_trace(go.Bar(x=df_kvs['Jalur'], y=df_kvs['Tidak Diterima'], name='Tidak Diterima', marker_color='lightslategray'))
            fig_status.update_layout(barmode='group', title='Kuota vs Status Diterima per Jalur', xaxis_title='Jalur', yaxis_title='Jumlah')
            st.plotly_chart(fig_status, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat grafik kuota vs status: {e}")

    # === History Ringkasan per Jalur ===
    if "summary" in selected:
        try:
            with open(selected["summary"], "r") as f:
                summary = json.load(f)

            st.subheader("📊 Ringkasan Per Jalur")
            jalur_data = summary.get("per_jalur", {})

            df_jalur = pd.DataFrame([
                {
                    "Jalur": jalur,
                    "Diterima": data.get("diterima", 0),
                    "Tidak Diterima": data.get("tidak_diterima", 0)
                }
                for jalur, data in jalur_data.items()
            ])

            fig_jalur = go.Figure()
            fig_jalur.add_trace(go.Bar(x=df_jalur['Jalur'], y=df_jalur['Diterima'], name='Diterima', marker_color='lightgreen'))
            fig_jalur.add_trace(go.Bar(x=df_jalur['Jalur'], y=df_jalur['Tidak Diterima'], name='Tidak Diterima', marker_color='salmon'))
            fig_jalur.update_layout(barmode='group', xaxis_title='Jalur', yaxis_title='Jumlah', title='Jumlah Diterima vs Tidak Diterima per Jalur')
            st.plotly_chart(fig_jalur, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Gagal memuat hasil summary: {e}")

else:
    st.warning("⚠ Belum ada riwayat backup ditemukan. Silakan jalankan prediksi terlebih dahulu di menu Analisis Prediksi.")