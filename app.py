import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Balık Raf Ömrü Tahmin Uygulaması", layout="centered")
st.title("🐟 Balık Raf Ömrü Tahmin Uygulaması")
st.write("Balığın depolama koşullarına göre tahmini kalan raf ömrünü hesaplayabilirsiniz.")

# ---- Kullanıcı Girdileri ----
st.header("Tahmin İçin Gerekli Bilgiler")

species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
hours = st.number_input("Depolama süresi (saat)", min_value=0, max_value=240, value=24, step=1)
temp = st.selectbox("Depolama sıcaklığı (°C)", [0, 4, 8, 12])
days_elapsed = st.number_input("Hasattan itibaren geçen süre (gün)", min_value=0, max_value=30, value=0, step=1)
avg_temp_post_harvest = st.selectbox("Hasat sonrası ortalama depolama sıcaklığı (°C)", [0, 4, 8, 12])
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

# ---- Model dosya yolunu çöz ----
def resolve_model_path(choice: str) -> str:
    fname = "rf_model_app.joblib" if choice == "Random Forest" else "xgb_model_app.joblib"
    paths = [os.path.join("models", fname), os.path.join("Models", fname)]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

# ---- Tahmin Butonu ----
if st.button("Tahmini Raf Ömrünü Hesapla"):
    try:
        model_path = resolve_model_path(model_choice)
        if not os.path.exists(model_path):
            st.error(f"Model dosyası bulunamadı: {model_path}")
        else:
            model = joblib.load(model_path)

            # Modelin beklediği kolonlar
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = ["total_hours","avg_temp_post_harvest","species_Somon","species_Levrek"]

            # Kullanıcı girdilerini DataFrame formatına çevir
            total_hours = hours + days_elapsed * 24
            row = {col: 0 for col in expected_features}
            row['total_hours'] = total_hours

            # Sıcaklık tercihi: hasat sonrası ortalama sıcaklık öne alınır
            if 'avg_temp_post_harvest' in expected_features:
                row['avg_temp_post_harvest'] = avg_temp_post_harvest if avg_temp_post_harvest is not None else temp

            sp_col = f"species_{species}"
            if sp_col in expected_features:
                row[sp_col] = 1

            X_input = pd.DataFrame([[row[c] for c in expected_features]], columns=expected_features)

            # Tahmin
            pred = model.predict(X_input)[0]
            st.success(f"Tahmini kalan raf ömrü: {pred:.1f} saat")
            st.info(f"Toplam geçen süre: {total_hours:.1f} saat (Depolama + Hasattan geçen günler)")

    except Exception as e:
        st.error(f"Hata oluştu: {e}")
