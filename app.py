import streamlit as st
import joblib
import pandas as pd
import os

st.title("Raf Ömrü Tahmin Uygulaması")
st.write("Balık türü, sıcaklık ve geçen süre bilgisi girerek kalan raf ömrünü tahmin edebilirsiniz.")

# Kullanıcı girişi
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
temp = st.number_input("Depolama Sıcaklığı (°C)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
hours = st.number_input("Geçen süre (saat)", min_value=0, max_value=240, value=24, step=1)
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

if st.button("Tahmin Et"):
    try:
        # Model dosyası yolunu belirle
        model_file = ""
        if model_choice == "Random Forest":
            model_file = "models/rf_model_app.joblib"
        else:
            model_file = "models/xgb_model_app.joblib"

        # Model dosyası yoksa hata
        if not os.path.exists(model_file):
            st.error(f"Seçilen model dosyası bulunamadı: {model_file}")
        else:
            model = joblib.load(model_file)

            # Simülasyon için input dataframe
            input_df = pd.DataFrame({
                "species_Somon": [1 if species == "Somon" else 0],
                "species_Levrek": [1 if species == "Levrek" else 0],
                "hours_4C": [hours if temp == 4 else 0],
                "hours_8C": [hours if temp == 8 else 0],
                "hours_12C": [hours if temp == 12 else 0]
            })

            # Tahmin
            prediction = model.predict(input_df)[0]
            st.success(f"Tahmini kalan raf ömrü: {prediction:.1f} saat")

    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
