import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import re

st.title("Raf Ömrü Tahmin Uygulaması")
st.write("Balık türü, sıcaklık ve geçen süre bilgisi girerek kalan raf ömrünü tahmin edebilirsiniz.")

# ---- Kullanıcı girişleri ----
species = st.selectbox("Balık Türü", ["Somon", "Levrek"])
temp = st.selectbox("Depolama Sıcaklığı (°C)", [0, 4, 8, 12])   # MODELE UYGUN seviyeler
hours = st.number_input("Kaç saat geçti?", min_value=0, max_value=240, value=24, step=1)
model_choice = st.radio("Model Seçimi", ["Random Forest", "XGBoost"])

# ---- Model dosya yolunu çöz ----
def resolve_model_path(choice: str) -> str:
    fname = "rf_model_app.joblib" if choice == "Random Forest" else "xgb_model_app.joblib"
    # Önce 'models/', yoksa 'Models/' dene
    paths = [os.path.join("models", fname), os.path.join("Models", fname)]
    for p in paths:
        if os.path.exists(p):
            return p
    # Hiçbiri yoksa ilkini döndür (hata mesajı için)
    return paths[0]

# ---- Tahmin butonu ----
if st.button("Tahmin Et"):
    try:
        model_path = resolve_model_path(model_choice)
        if not os.path.exists(model_path):
            st.error(f"Model dosyası bulunamadı: {model_path}")
        else:
            model = joblib.load(model_path)

            # Modelin beklediği özellik adlarını tespit et
            if hasattr(model, "feature_names_in_"):
                expected_features = list(model.feature_names_in_)
            else:
                # Yedek plan: en yaygın iki senaryodan birini dene
                candidate1 = ["hours_8C", "hours_12C", "species_Somon", "species_Levrek"]
                candidate2 = ["hours_0C", "hours_4C", "hours_8C", "hours_12C", "species_Somon", "species_Levrek"]
                nfi = getattr(model, "n_features_in_", None)
                if nfi == len(candidate1):
                    expected_features = candidate1
                else:
                    expected_features = candidate2

            # Kullanıcı girdisini tam olarak modelin beklediği kolon setine çevir
            row = {col: 0 for col in expected_features}

            # Saat kolonları: model hangi "hours_*C" kolonlarını bekliyorsa onlara dağıt
            hour_cols = [c for c in expected_features if re.match(r"^hours_\d+C$", c)]
            target_col = f"hours_{temp}C"
            if target_col in hour_cols:
                row[target_col] = hours
            else:
                # Modelin saat kolonları arasında temp yoksa (ör. sadece 8C,12C varken 4C seçildiyse)
                # bu durumda saatleri hiçbirine yazmıyoruz (0 kalır). Bu, modelin eğitim kurgusuna uygundur.
                pass

            # Tür kolonları: varsa onları işaretle
            sp_col = f"species_{species}"
            if sp_col in expected_features:
                row[sp_col] = 1
            # Diğer tür sütunları varsa 0 kalacak.

            # DataFrame'i tam beklenen sırada oluştur
            X_input = pd.DataFrame([[row[c] for c in expected_features]], columns=expected_features)

            # Debug amaçlı (gerekirse açarsın)
            # st.write("Modelin beklediği kolonlar:", expected_features)
            # st.write("Gönderilen satır:", X_input)

            pred = model.predict(X_input)[0]
            st.success(f"Tahmini kalan raf ömrü: {pred:.1f} saat")
    except Exception as e:
        st.error(f"Tahmin sırasında bir hata oluştu: {e}")
