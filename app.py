import streamlit as st
import joblib
import pandas as pd
import os
import urllib.request

st.set_page_config(page_title="Raf Ömrü Tahmin Uygulaması", layout="centered")
st.title("Raf Ömrü Tahmin Uygulaması")
st.write("Balık türü, sıcaklık ve geçen süre bilgisi girerek kalan raf ömrünü tahmin edebilirsiniz.")

# --- MODELLERİ OTOMATİK YÜKLEME ---
model_urls = {
    "rf_model_app.joblib": "https://github.com/essenalp/fish_shelf_life_app/raw/main/models/rf_model_app.joblib",
    "xgb_model_app.joblib": "https://github.com/essenalp/fish_shelf_life_app/raw/main/models/xgb_model_app.joblib"
}

os.makedirs("models", exist_ok=True)

for fname, url in model_urls.items():
    if not os.path.exists(f"models/{fname}"):
        try:
            with st.spinner(f"{fname} indiriliyor..."):
                urllib.request.urlretrieve(url, f"models/{fname}")
        except Exception as e:
            st.error(f"{fname} indirilemedi: {e}")

# --- KULLANICI GİRİŞİ ---
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
temp = st.number_input("Depolama Sıcaklığı (°C)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
hours = st.number_input("Kaç saat geçti?", min_value=0, max_value=240, value=24, step=1)
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

if st.button("Tahmin Et"):
    try:
        # Modeli yükle
        if model_choice == "Random Forest":
            model = joblib.load("models/rf_model_app.joblib")
        else:
            model = joblib.load("models/xgb_model_app.joblib")

        # Kullanıcı girişini modele uygun DataFrame'e dönüştür
        input_df = pd.DataFrame({
            "hours_0C": [hours if temp == 0 else 0],
            "hours_4C": [hours if temp == 4 else 0],
            "hours_8C": [hours if temp == 8 else 0],
            "hours_12C": [hours if temp == 12 else 0],
            "species_Somon": [1 if species == "Somon" else 0],
            "species_Levrek": [1 if species == "Levrek" else 0]
        })
