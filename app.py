import streamlit as st
import joblib
import pandas as pd
import os

st.title("Raf Ömrü Tahmin Uygulaması")
st.write("Balık türü, sıcaklık ve geçen süre bilgisi girerek kalan raf ömrünü tahmin edebilirsiniz.")

# Kullanıcı girişi
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
temp = st.number_input("Depolama Sıcaklığı (°C)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
hours = st.number_input("Kaç saat geçti?", min_value=0, max_value=240, value=24, step=1)
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

# Model klasörü yolu
MODEL_DIR = "models"

def load_model(model_name):
    model_path = os.path.join(MODEL_DIR, model_name)
    if not os.path.exists(model_path):
        st.error(f"Model dosyası bulunamadı: {model_path}")
        return None
    return joblib.load(model_path)

if st.button("Tahmin Et"):
    try:
        if model_choice == "Random Forest":
            model = load_model("rf_model_app.joblib")
        else:
            model = load_model("xgb_model_app.joblib")

        if model is not None:
            # Simülasyon verisine uygun encoding
            input_df = pd.DataFrame({
                "hours_0C": [hours if temp == 0 else 0],
                "hours_4C": [hours if temp == 4 else 0],
                "hours_8C": [hours if temp == 8 else 0],
                "hours_12C": [hours if temp == 12 else 0],
                "species_Somon": [1 if species == "Somon" else 0],
                "species_Levrek": [1 if species == "Levrek" else 0],
            })

            prediction = model.predict(input_df)[0]
            st.success(f"Tahmini kalan raf ömrü: {prediction:.1f} saat")

    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
