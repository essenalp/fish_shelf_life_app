import joblib
import pandas as pd

# --- Model yolu ---
MODEL_DIR = r"C:\Users\SUMAE\Desktop\ML_Proje\models"
rf_model = joblib.load(f"{MODEL_DIR}/rf_model_app.joblib")
xgb_model = joblib.load(f"{MODEL_DIR}/xgb_model_app.joblib")
print("RF ve XGB modelleri yüklendi.")

# --- Modelin beklediği kolonlar ---
cols = list(rf_model.feature_names_in_)

# --- Kullanıcıdan input al ---
def get_float_input(prompt, default=0.0):
    try:
        val = float(input(f"{prompt} [{default}]: ") or default)
        return val
    except:
        print("Geçersiz değer, varsayılan kullanıldı:", default)
        return default

print("Lütfen tahmin için değerleri girin:")

row = {c:0 for c in cols}  # tüm kolonları önce 0 yap

row['hours'] = get_float_input("Kaç saat geçti?", 24)
row['temp'] = get_float_input("Depolama sıcaklığı (°C)?", 8)
row['days_elapsed'] = get_float_input("Hasattan itibaren geçen gün?", 5)
row['avg_temp_post_harvest'] = get_float_input("Hasattan sonra ortalama depolama sıcaklığı?", 8)
row['tvb_n'] = get_float_input("TVB-N değeri?", 10.5)
row['tba'] = get_float_input("TBA değeri?", 0.8)
row['ph'] = get_float_input("pH değeri?", 6.5)
row['psychrotrophic'] = get_float_input("Psikrotrofik bakteri sayısı?", 4.0)
row['total_mesophilic'] = get_float_input("Toplam mezofilik bakteri sayısı?", 5.0)
row['pseudomonas'] = get_float_input("Pseudomonas sayısı?", 2.5)
row['L'] = get_float_input("Renk L değeri?", 50.0)
row['a'] = get_float_input("Renk a değeri?", 2.0)
row['b'] = get_float_input("Renk b değeri?", 3.0)
row['texture'] = get_float_input("Tekstür skoru?", 7.5)
row['appearance'] = get_float_input("Görünüş skoru?", 8.0)
row['odor'] = get_float_input("Koku skoru?", 7.0)

# Balık türü
species = input("Balık türü (Somon/Levrek) [Somon]: ") or "Somon"
if f"species_{species}" in cols:
    row[f"species_{species}"] = 1

# --- DataFrame ---
X_input = pd.DataFrame([row], columns=cols)

# --- Tahmin ---
rf_pred = rf_model.predict(X_input)[0]
xgb_pred = xgb_model.predict(X_input)[0]

print("\n--- Tahmin Sonuçları ---")
print(f"Random Forest tahmini kalan raf ömrü: {rf_pred:.1f} saat")
print(f"XGBoost tahmini kalan raf ömrü: {xgb_pred:.1f} saat")
