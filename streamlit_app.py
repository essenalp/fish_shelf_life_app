import streamlit as st
import pandas as pd
import joblib
import os

# Başlık
st.title("🐟 Balık Raf Ömrü Tahmin Uygulaması")
st.write("Balığın depolama koşullarına göre tahmini kalan raf ömrünü hesaplayabilirsiniz. Depolama sıcaklığı ve süresi ihlal verilerinden oluşur.")

# Alt başlık
st.header("Tahmin İçin Gerekli Bilgiler")

# ---- Kullanıcı girişleri ----
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
hours_elapsed = st.number_input("Depolama süresi (saat)", min_value=0, max_value=500, value=24)
current_temp = st.number_input("Depolama sıcaklığı (°C)", min_value=0, max_value=20, value=0)
days_elapsed = st.number_input("Hasattan itibaren geçen süre (gün)", min_value=0, max_value=60, value=0)
post_harvest_temp = st.number_input("Hasat sonrası ortalama depolama sıcaklığı (°C)", min_value=0, max_value=20, value=0)

model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

# ---- Model dosya yolunu çöz ----
def resolve_model_path(choice: str) -> str:
    fname = "rf_model_app.joblib" if choice == "Random Forest" else "xgb_model_app.joblib"
    paths = [os.path.join("models", fname), os.path.join("Models", fname)]
    for p in paths:
        if os.path.exists(p):
            return p
    return paths[0]

# ---- Tahmin butonu ----
if st.button("📈 Tahmin Et"):
    try:
        model_path = resolve_model_path(model_choice)
        if not os.path.exists(model_path):
            st.error(f"Model dosyası bulunamadı: {model_path}")
        else:
            model = joblib.load(model_path)

            # Modelin beklediği özellik adlarını al
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                expected_features = ["hours_8C", "hours_12C", "species_Somon", "species_Levrek"]

            # Kullanıcı girdisini modelin beklediği kolon setine çevir
            row = {col: 0 for col in expected_features}

            # Sıcaklık kolonlarını ayarla (modelin eğitildiği 8C ve 12C varsayılmış)
            hour_cols = [c for c in expected_features if c.startswith("hours_")]
            temp_map = {8: "hours_8C", 12: "hours_12C"}  # eğitimde hangi sıcaklıklar varsa
            if current_temp in temp_map:
                row[temp_map[current_temp]] = hours_elapsed
            # Tür kolonları
            sp_col = f"species_{species}"
            if sp_col in expected_features:
                row[sp_col] = 1

            # DataFrame oluştur
            X_input = pd.DataFrame([[row[c] for c in expected_features]], columns=expected_features)

            pred = model.predict(X_input)[0]
            st.success(f"🧭 Tahmini Raf Ömrü: **{pred:.1f} gün**")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
