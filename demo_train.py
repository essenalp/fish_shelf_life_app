import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

np.random.seed(42)

# --- Klasör ve dosya yolları ---
BASE_DIR = r"C:\Users\SUMAE\Desktop\ML_Proje"
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)
EXCEL_PATH = os.path.join(BASE_DIR, "demo_fish_shelf_life.xlsx")

# --- Demo veri üretimi ---
n_samples = 5000

species = np.random.choice(['Somon', 'Levrek'], size=n_samples)
hours = np.random.randint(0, 241, size=n_samples)
temp = np.random.choice([0, 4, 8, 12], size=n_samples)
days_elapsed = np.random.randint(0, 61, size=n_samples)  # 0-60 gün
avg_temp_post_harvest = np.random.choice([0, 4, 8, 12], size=n_samples)

tvb_n = np.clip(5 + 0.1*hours + 0.5*days_elapsed + np.random.normal(0, 0.5, n_samples), 0, None)
tba = np.clip(0.2 + 0.05*hours + 0.1*days_elapsed + np.random.normal(0, 0.05, n_samples), 0, None)
ph = np.clip(6 + 0.001*hours + np.random.normal(0, 0.05, n_samples), 5, 9)

psychrotrophic = np.clip(2 + 0.05*hours + 0.2*days_elapsed + np.random.normal(0, 0.5, n_samples), 0, None)
total_mesophilic = np.clip(3 + 0.04*hours + 0.15*days_elapsed + np.random.normal(0, 0.5, n_samples), 0, None)
pseudomonas = np.clip(1 + 0.03*hours + 0.1*days_elapsed + np.random.normal(0, 0.3, n_samples), 0, None)

L = np.clip(50 + np.random.normal(0, 5, n_samples), 0, 100)
a = np.clip(2 + np.random.normal(0, 2, n_samples), -128, 127)
b = np.clip(3 + np.random.normal(0, 2, n_samples), -128, 127)
texture = np.clip(8 - 0.01*hours - 0.02*days_elapsed + np.random.normal(0, 0.5, n_samples), 1, 10)
appearance = np.clip(8 - 0.01*hours - 0.02*days_elapsed + np.random.normal(0, 0.5, n_samples), 1, 10)
odor = np.clip(8 - 0.01*hours - 0.02*days_elapsed + np.random.normal(0, 0.5, n_samples), 1, 10)

remaining_shelf_life = np.clip(48 - hours - days_elapsed*24 - 2*(temp/4) + 0.5*texture, 0, 48)

df = pd.DataFrame({
    'species': species,
    'hours': hours,
    'temp': temp,
    'days_elapsed': days_elapsed,
    'avg_temp_post_harvest': avg_temp_post_harvest,
    'tvb_n': tvb_n,
    'tba': tba,
    'ph': ph,
    'psychrotrophic': psychrotrophic,
    'total_mesophilic': total_mesophilic,
    'pseudomonas': pseudomonas,
    'L': L,
    'a': a,
    'b': b,
    'texture': texture,
    'appearance': appearance,
    'odor': odor,
    'remaining_shelf_life': remaining_shelf_life
})

# --- Excel kaydet ---
df.to_excel(EXCEL_PATH, index=False)
print("Demo veri Excel dosyası kaydedildi:", EXCEL_PATH)

# --- Model eğitimi ---
df_encoded = pd.get_dummies(df, columns=['species'], drop_first=False)

feature_cols = [
    'hours', 'temp', 'days_elapsed', 'avg_temp_post_harvest',
    'tvb_n', 'tba', 'ph', 'psychrotrophic', 'total_mesophilic', 'pseudomonas',
    'L', 'a', 'b', 'texture', 'appearance', 'odor'
] + [c for c in df_encoded.columns if c.startswith('species_')]

X = df_encoded[feature_cols]
y = df_encoded['remaining_shelf_life']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest MAE:", mean_absolute_error(y_test, rf_pred))
joblib.dump(rf, os.path.join(MODEL_DIR, 'rf_model_app.joblib'))
print("Random Forest modeli kaydedildi.")

# XGBoost
xgb = XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
print("XGBoost MAE:", mean_absolute_error(y_test, xgb_pred))
joblib.dump(xgb, os.path.join(MODEL_DIR, 'xgb_model_app.joblib'))
print("XGBoost modeli kaydedildi.")
