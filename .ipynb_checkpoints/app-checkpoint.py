import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import date

# Modeli yükle
model = joblib.load("fish_shelf_life_model.pkl")

st.title("🐟 Balık Raf Ömrü Tahmin Uygulaması")
st.write("Bu uygulama, sıcaklık ve depolama süresi gibi verilere göre balığın tahmini raf ömrünü hesaplar.")

# --- Yeni Sekme: Hasattan itibaren geçen süre ---
st.header("Hasattan itibaren geçen süre")
days_elapsed = st.number_input(
    "Hasattan itibaren geçen süre (gün)",
    min_value=0,
    max_value=30,
    value=0,
    step=1,
    help="Balığın avlanma tarihinden itibaren geçen süreyi gün olarak giriniz."
)

# --- Girdi Alanları ---
st.header("Depolama Koşulları")

temperature = st.number_input(
    "Depolama sıcaklığı (°C)", 
    min_value=-5.0, 
    max_value=30.0, 
    value=4.0, 
    step=0.5
)

storage_days = st.number_input(
    "Depolama süresi (gün)", 
    min_value=0, 
    max_value=30, 
    value=5, 
    step=1
)

# --- Tahmin Butonu ---
if st.button("Tahmini Raf Ömrünü Hesapla"):
    try:
        # Model girişi
        X_input = pd.DataFrame({
            "temperature": [temperature],
            "storage_days": [storage_days],
            "days_elapsed": [days_elapsed]
        })

        # Tahmin
        prediction = model.predict(X_input)
        predicted_life = prediction[0]

        st.success(f"Balığın tahmini raf ömrü: **{predicted_life:.1f} gün**")
        st.caption("Not: Tahmin, girilen sıcaklık, depolama süresi ve hasattan itibaren geçen süreye göre hesaplanmıştır.")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
