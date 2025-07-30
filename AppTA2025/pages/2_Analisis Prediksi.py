from datetime import datetime
from dis import code_info
from utils import protect_page, nilai_to_kategori

import json
import os
import streamlit as st

protect_page()

# === Import Library ===
from io import BytesIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score

st.title("📊 Prediksi Penerimaan Siswa Baru")

# === Uploaad File ===
uploaded_file = st.file_uploader("Upload File Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    # === Konversi kolom nilai ke kategori ===
    map_kolom_nilai = {
        'Agama_Rata-rata': 'Kategori_Agama',
        'BhsIndonesia_Rata-rata': 'Kategori_BhsIndonesia',
        'MTK_Rata-rata': 'Kategori_Matematika',
        'IPA_Rata-rata': 'Kategori_IPA',
        'IPS_Rata-rata': 'Kategori_IPS',
        'BhsInggris_Rata-rata': 'Kategori_BhsInggris'
    }

    for kolom_nilai, kolom_kategori in map_kolom_nilai.items():
        if kolom_nilai in df.columns:
            df[kolom_kategori] = df[kolom_nilai].apply(nilai_to_kategori)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # === Pilih Target dan Fitur ===
    target = st.selectbox("🎯 Pilih Kolom Target (Label Penerimaan)", df.columns)
    fitur = st.multiselect("📌 Pilih Fitur untuk Prediksi", [col for col in df.columns if col != target])

    if fitur:
        num_features = [col for col in fitur if df[col].dtype in ['int64', 'float64']]
        cat_features = [col for col in fitur if df[col].dtype == 'object']

        df.dropna(subset=fitur + [target], inplace=True)

        X = df[fitur]
        y = df[target]

        split_data = st.checkbox("Gunakan Split Data untuk Training dan Testing (70:30)???", value=True)

        if split_data:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=10)
            st.info("✅ Data dibagi menggunakan Train-Test Split (70% Training, 30% Testing).")
        else:
            X_train, X_test = X, X
            y_train, y_test = y, y
            st.warning("⚠ Sedang Menggunakan Seluruh Data untuk Training dan Testing.", icon="⚠")
            
        # === Preprocessing pipeline ===
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

        # === Buat Pipeline ===
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier(random_state=10))
        ])
        
                
        # # === Tentukan Parameter Grid Search Hyperparameter Tuning ===        
        # param_grid = {
        #     'clf__n_estimators': [20, 30],
        #     'clf__max_depth': [5, 10],
        #     'clf__min_samples_split': [2, 3, 5, 10],
        #     'clf__min_samples_leaf': [2, 3, 5, 10],
        #     'clf__max_features': ['sqrt', 'log2'],
        #     'clf__class_weight': [None, 'balanced']
        # }
        
       # === Tentukan Parameter Grid Search Hyperparameter Tuning ===        
        param_grid = {
            'clf__n_estimators': [20, 30],
            'clf__max_depth': [5, 10],
            'clf__min_samples_split': [2, 3, 5, 10],
            'clf__min_samples_leaf': [2, 3, 5, 10],
            'clf__max_features': ['sqrt', 'log2'],
            'clf__class_weight': [None, 'balanced']
        
        }

        with st.spinner("🔍 Sedang Melatih Model dan Mencari Parameter Terbaik..."):
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

        y_pred = best_model.predict(X_test)

        # === Grid Search Results ===
        st.markdown("### 🏆 Parameter Terbaik dan Skor")
        st.write("📌 *Parameter Terbaik:*", grid_search.best_params_)


        # === Classification Report ===
        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose())
        cls_report = classification_report(y_test, y_pred, output_dict=True)
            
            
        # === Output Akurasi ===
        akurasi = accuracy_score(y_test, y_pred)
        st.subheader("🎯 Akurasi Model")
        st.metric(label="Akurasi", value=f"{akurasi * 100:.2f}%")


        # === Output Confusion Matrix ===
        st.subheader("📌 Confusion Matrix")
        fig, ax = plt.subplots(figsize=(15, 10))
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, cmap='Blues')
        st.pyplot(fig)
        

        st.subheader("📑 Tabel Hasil Prediksi")
        df_hasil_test = X_test.copy()

        if 'Nama' in df.columns:
            df_hasil_test.insert(0, 'Nama', df.loc[X_test.index, 'Nama'])

        df_hasil_test['Label_Asli'] = y_test.astype(str)
        df_hasil_test['Prediksi'] = pd.Series(y_pred, index=X_test.index).astype(str)
        df_hasil_test['Diterima/Tidak Diterima'] = np.where(
            df_hasil_test['Label_Asli'] == df_hasil_test['Prediksi'], 'Diterima', 'Tidak Diterima'
        )
        st.dataframe(df_hasil_test)
        
        # === Ringkasan hasil prediksi untuk grafik pie ===
        hasil_count = df_hasil_test['Diterima/Tidak Diterima'].value_counts().reset_index()
        hasil_count.columns = ['Status', 'Jumlah']

        # === Tambahan: Grafik Pie Proporsi Diterima vs Tidak ===
        st.subheader("🧩 Proporsi Diterima dan Tidak Diterima (Pie Chart)")
        fig_pie = go.Figure(data=[
            go.Pie(labels=hasil_count['Status'], values=hasil_count['Jumlah'], hole=0.4,
                marker=dict(colors=['lightgreen', 'salmon']),
                textinfo='label+percent', insidetextorientation='radial')
        ])
        fig_pie.update_layout(title="Proporsi Siswa Diterima dan Tidak Diterima")
        st.plotly_chart(fig_pie, use_container_width=True)


        # === Grafik Kuota vs Semua Prediksi ===
        st.subheader("📊 Perbandingan Kuota dan Hasil Prediksi per Jalur")

        kuota_presentase = {
            'ZONASI': 0.35,
            'AFIRMASI KETM': 0.30,
            'ANAK GURU': 0.05,
            'PRESTASI RAPOT': 0.30,
            'PRESTASI KEJUARAAN': 0.30
        }

        jumlah_total = len(df)
        kuota_jumlah = {jalur: int(jumlah_total * persen) for jalur, persen in kuota_presentase.items()}
        hasil_prediksi = df_hasil_test['Prediksi'].value_counts().to_dict()

        df_kuota_vs_prediksi = pd.DataFrame([
            {
                'Jalur': jalur,
                'Kuota': kuota_jumlah.get(jalur, 0),
                'Prediksi': hasil_prediksi.get(jalur, 0)
            }
            for jalur in kuota_presentase
        ])

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_kuota_vs_prediksi['Jalur'], y=df_kuota_vs_prediksi['Kuota'],
                             name='Kuota', marker_color='goldenrod',
                             text=df_kuota_vs_prediksi['Kuota'], textposition='outside',
                             hovertemplate='<b>%{x}</b><br>Kuota: %{y}<extra></extra>'))
        fig.add_trace(go.Bar(x=df_kuota_vs_prediksi['Jalur'], y=df_kuota_vs_prediksi['Prediksi'],
                             name='Prediksi', marker_color='deepskyblue',
                             text=df_kuota_vs_prediksi['Prediksi'], textposition='outside',
                             hovertemplate='<b>%{x}</b><br>Prediksi: %{y}<extra></extra>'))

        fig.update_layout(barmode='group',
                          title='Perbandingan Kuota dan Hasil Prediksi per Jalur',
                          xaxis_title='Jalur',
                          yaxis_title='Jumlah Siswa',
                          height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # === Peringatan jika jumlah prediksi melebihi kuota
        jalur_melebihi_prediksi = [
            jalur for jalur in kuota_presentase
            if hasil_prediksi.get(jalur, 0) > kuota_jumlah.get(jalur, 0)
        ]

        if jalur_melebihi_prediksi:
            st.error(f"⚠ Jumlah prediksi melebihi kuota untuk jalur: {', '.join(jalur_melebihi_prediksi)}")

        
        # === Grafik Kuota vs Jumlah Diterima dan Tidak Diterima per Jalur ===
        st.subheader("📊 Perbandingan Kuota, Diterima, dan Tidak Diterima per Jalur")

        # Filter data yang diterima dan tidak diterima
        df_diterima = df_hasil_test[df_hasil_test['Diterima/Tidak Diterima'] == 'Diterima']
        df_tidak_diterima = df_hasil_test[df_hasil_test['Diterima/Tidak Diterima'] == 'Tidak Diterima']

        # Hitung jumlah diterima dan tidak diterima per jalur
        diterima_per_jalur = df_diterima['Prediksi'].value_counts().to_dict()
        tidak_diterima_per_jalur = df_tidak_diterima['Prediksi'].value_counts().to_dict()

        # Gabungkan ke dalam dataframe
        df_kuota_vs_status = pd.DataFrame([
            {
                'Jalur': jalur,
                'Kuota': kuota_jumlah.get(jalur, 0),
                'Diterima': diterima_per_jalur.get(jalur, 0),
                'Tidak Diterima': tidak_diterima_per_jalur.get(jalur, 0)
            }
            for jalur in kuota_presentase
        ])

        # Buat grafik
        fig_status = go.Figure()

        # Bar: Kuota
        fig_status.add_trace(go.Bar(
            x=df_kuota_vs_status['Jalur'],
            y=df_kuota_vs_status['Kuota'],
            name='Kuota',
            marker_color='lightseagreen',
            text=df_kuota_vs_status['Kuota'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Kuota: %{y}<extra></extra>'
        ))

        # Bar: Diterima
        fig_status.add_trace(go.Bar(
            x=df_kuota_vs_status['Jalur'],
            y=df_kuota_vs_status['Diterima'],
            name='Diterima',
            marker_color='lightcoral',
            text=df_kuota_vs_status['Diterima'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Diterima: %{y}<extra></extra>'
        ))

        # Bar: Tidak Diterima
        fig_status.add_trace(go.Bar(
            x=df_kuota_vs_status['Jalur'],
            y=df_kuota_vs_status['Tidak Diterima'],
            name='Tidak Diterima',
            marker_color='lightslategray',
            text=df_kuota_vs_status['Tidak Diterima'],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Tidak Diterima: %{y}<extra></extra>'
        ))

        # Layout
        fig_status.update_layout(
            barmode='group',
            title='Perbandingan Kuota, Diterima, dan Tidak Diterima per Jalur',
            xaxis_title='Jalur',
            yaxis_title='Jumlah Siswa',
            height=600
        )

        # Tampilkan di Streamlit
        st.plotly_chart(fig_status, use_container_width=True)
        
        # === Peringatan jika jumlah siswa diterima melebihi kuota
        jalur_melebihi_diterima = [
            jalur for jalur in kuota_presentase
            if diterima_per_jalur.get(jalur, 0) > kuota_jumlah.get(jalur, 0)
        ]

        if jalur_melebihi_diterima:
            st.error(f"⚠ Jumlah siswa DITERIMA melebihi kuota untuk jalur: {', '.join(jalur_melebihi_diterima)}")


        # === Convert dataframe ke Excel ===
        output_excel = BytesIO()
        with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
            df_hasil_test.to_excel(writer, index=False, sheet_name='Prediksi')
        output_excel.seek(0)

        st.download_button(
            label="📥 Download Hasil Prediksi  Algoritma ke Excel (.xlsx)",
            data=output_excel,
            file_name="Hasil Prediksi Algoritma Random Forest.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # === Export File Excel Diterima ===
        df_diterima_excel = df_hasil_test[df_hasil_test['Diterima/Tidak Diterima'] == 'Diterima']
        output_diterima = BytesIO()
        with pd.ExcelWriter(output_diterima, engine='xlsxwriter') as writer:
            df_diterima_excel.to_excel(writer, index=False, sheet_name='Diterima')
        output_diterima.seek(0)

        st.download_button(
            label="📥 Download Siswa Diterima (.xlsx)",
            data=output_diterima,
            file_name="Siswa_Diterima.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # === Export File Excel Tidak Diterima ===
        df_tidak_diterima_excel = df_hasil_test[df_hasil_test['Diterima/Tidak Diterima'] == 'Tidak Diterima']
        output_tidak_diterima = BytesIO()
        with pd.ExcelWriter(output_tidak_diterima, engine='xlsxwriter') as writer:
            df_tidak_diterima_excel.to_excel(writer, index=False, sheet_name='Tidak Diterima')
        output_tidak_diterima.seek(0)

        st.download_button(
            label="📥 Download Siswa Tidak Diterima (.xlsx)",
            data=output_tidak_diterima,
            file_name="Siswa_Tidak_Diterima.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        # === Simpan hasil global ke JSON untuk ditampilkan di Home ===
        hasil_summary = {
            "diterima": len(df_diterima),
            "tidak_diterima": len(df_tidak_diterima),
            "per_jalur": {}  # simpan detail per jalur
        }

        jalur_list = df_hasil_test['Prediksi'].unique()

        for jalur in jalur_list:
            hasil_summary["per_jalur"][jalur] = {
                "diterima": int((df_diterima['Prediksi'] == jalur).sum()),
                "tidak_diterima": int((df_tidak_diterima['Prediksi'] == jalur).sum())
            }

        # Simpan ke file JSON
        with open("hasil_prediksi.json", "w") as f:
            json.dump(hasil_summary, f)


        # Simpan hasil prediksi ke session_state agar bisa dibaca di halaman lain
        st.session_state['hasil_prediksi'] = df_hasil_test.copy()
        
        model_info = {
            "best_params": {k: str(v) for k, v in grid_search.best_params_.items()},
            "accuracy": float(akurasi)
        }

        # === Simpan otomatis ke backup_prediksi/ ===
        os.makedirs("backup_prediksi", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        df_hasil_test.to_excel(f"backup_prediksi/hasil_{timestamp}.xlsx", index=False)
        with open(f"backup_prediksi/hasil_summary_{timestamp}.json", "w") as f:
            json.dump(hasil_summary, f)
        with open(f"backup_prediksi/model_info_{timestamp}.json", "w") as f:
            json.dump(model_info, f)

        st.success("✅ Semua hasil prediksi telah dibackup otomatis ke folder backup_prediksi.")
        
        
        ## ===  Backup File semua proses untuk ditampilkan ke History Analisis ===
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Simpan Excel
        df_hasil_test.to_excel(f"backup_prediksi/hasil_{timestamp}.xlsx", index=False)

        # Simpan ringkasan hasil
        with open(f"backup_prediksi/hasil_summary_{timestamp}.json", "w") as f:
            json.dump(hasil_summary, f)

        # Simpan model info
        with open(f"backup_prediksi/model_info_{timestamp}.json", "w") as f:
            json.dump(model_info, f)

        # Simpan classification report
        with open(f"backup_prediksi/classification_report_{timestamp}.json", "w") as f:
            json.dump(cls_report, f)

        # Simpan confusion matrix PNG
        fig_cm, ax_cm = plt.subplots(figsize=(15, 10))
        ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax_cm, cmap='Blues')
        cm_filename = f"backup_prediksi/confusion_matrix_{timestamp}.png"
        fig_cm.savefig(cm_filename, bbox_inches='tight')
        plt.close(fig_cm)

        # Simpan data pie chart
        hasil_count.to_json(f"backup_prediksi/pie_summary_{timestamp}.json", orient='records')

        # Simpan kuota vs prediksi
        df_kuota_vs_prediksi.to_json(f"backup_prediksi/kuota_vs_prediksi_{timestamp}.json", orient='records')

        # Simpan kuota vs status
        df_kuota_vs_status.to_json(f"backup_prediksi/kuota_vs_status_{timestamp}.json", orient='records')