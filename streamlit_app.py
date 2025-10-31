import streamlit as st
import joblib
import numpy as np
import os

# Sayfa başlığı
st.title("🐟 Balık Raf Ömrü Tahmin Uygulaması")
st.write("Balığın depolama koşullarına göre tahmini kalan raf ömrünü hesaplayabilirsiniz.")

# Model yolları
model_paths = {
    "Random Forest": "models/rf_model_app.joblib",
    "XGBoost": "models/xgb_model_app.joblib"
}

# Girdi alanları
st.header("Depolama Koşulları")

storage_temp = st.number_input("Depolama sıcaklığı (°C)", min_value=-2.0, max_value=20.0, value=4.0, step=0.5)
storage_hours = st.number_input("Depolama süresi (saat)", min_value=1, max_value=240, value=24, step=1)
violation_temp = st.number_input("İhlal sıcaklığı (°C)", min_value=0.0, max_value=40.0, value=10.0, step=0.5)
violation_hours = st.number_input("İhlal süresi (saat)", min_value=0, max_value=48, value=2, step=1)
days_elapsed = st.number_input("Hasattan itibaren geçen süre (gün)", min_value=0, max_value=60, value=5, step=1)

fish_type = st.selectbox("Balık Türü", ["Somon", "Levrek"])

# Model seçimi
model_choice = st.selectbox("Model Seçimi", list(model_paths.keys()))

# Tahmin Et butonu
if st.button("Tahmin Et"):
    model_path = model_paths[model_choice]

    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
    else:
        model = joblib.load(model_path)

        # Balık türünü sayısal değere çevir
        fish_map = {"Somon": 0, "Levrek": 1}
        fish_value = fish_map[fish_type]

        # Girdileri sırayla modele ver
        X = np.array([[storage_temp, storage_hours, violation_temp, violation_hours, days_elapsed, fish_value]])
        prediction = model.predict(X)[0]

        st.success(f"Tahmini kalan raf ömrü: **{prediction:.1f} gün**")
