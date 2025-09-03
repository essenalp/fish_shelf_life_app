import streamlit as st
import joblib
import pandas as pd

st.title("Raf Ömrü Tahmin Uygulaması (Simülasyon Verisi ile)")
st.write("Balık türü ve depolama sıcaklığı girerek kalan raf ömrünü tahmin edebilirsiniz.")

# Kullanıcı girişi
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
temp = st.number_input("Depolama Sıcaklığı (°C)", min_value=0, max_value=12, value=4, step=1)
hours = st.number_input("Kaç saat geçti?", min_value=0, max_value=240, value=24, step=1)
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

if st.button("Tahmin Et"):
    try:
        # Model seçimi
        if model_choice == "Random Forest":
            model = joblib.load("models/rf_model_app.joblib")
        else:
            model = joblib.load("models/xgb_model_app.joblib")

        # Model feature'ları
        model_features = model.feature_names_in_

        # Input DataFrame'i sıfırlarla başlat
        input_dict = {feat: [0] for feat in model_features}

        # Tür encoding
        if species == "Somon":
            if "species_Somon" in input_dict:
                input_dict["species_Somon"][0] = 1
        elif species == "Levrek":
            if "species_Levrek" in input_dict:
                input_dict["species_Levrek"][0] = 1

        # Saat bilgisini ilgili sıcaklık feature'ına ata
        temp_column = f"hours_{temp}C"
        if temp_column in input_dict:
            input_dict[temp_column][0] = hours
        else:
            st.error(f"Modelde bu sıcaklık feature'ı yok: {temp_column}")

        # DataFrame oluştur
        input_df = pd.DataFrame(input_dict)

        # Tahmin
        prediction = model.predict(input_df)[0]
        st.success(f"Tahmini kalan raf ömrü: {prediction:.1f} saat")

    except FileNotFoundError:
        st.error("Seçilen model dosyası bulunamadı. Lütfen modeli 'models' klasörüne kaydedin.")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
