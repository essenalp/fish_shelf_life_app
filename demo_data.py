import pandas as pd
import numpy as np

np.random.seed(42)

n_samples = 5000

# Balık türü ve depolama bilgileri
species = np.random.choice(['Somon', 'Levrek'], size=n_samples)
hours = np.random.randint(0, 241, size=n_samples)
temp = np.random.choice([0, 4, 8, 12], size=n_samples)
days_elapsed = np.random.randint(0, 61, size=n_samples)  # 0-60 gün
avg_temp_post_harvest = np.random.choice([0, 4, 8, 12], size=n_samples)

# Kimyasal göstergeler
tvb_n = np.clip(5 + 0.1*hours + 0.5*days_elapsed + np.random.normal(0, 0.5, n_samples), 0, None)
tba = np.clip(0.2 + 0.05*hours + 0.1*days_elapsed + np.random.normal(0, 0.05, n_samples), 0, None)
ph = np.clip(6 + 0.001*hours + np.random.normal(0, 0.05, n_samples), 5, 9)

# Mikrobiyolojik göstergeler (CFU/g)
psychrotrophic = np.clip(2 + 0.05*hours + 0.2*days_elapsed + np.random.normal(0, 0.5, n_samples), 0, None)
total_mesophilic = np.clip(3 + 0.04*hours + 0.15*days_elapsed + np.random.normal(0, 0.5, n_samples), 0, None)
pseudomonas = np.clip(1 + 0.03*hours + 0.1*days_elapsed + np.random.normal(0, 0.3, n_samples), 0, None)

# Görsel ve duyusal parametreler
L = np.clip(50 + np.random.normal(0, 5, n_samples), 0, 100)
a = np.clip(2 + np.random.normal(0, 2, n_samples), -128, 127)
b = np.clip(3 + np.random.normal(0, 2, n_samples), -128, 127)
texture = np.clip(8 - 0.01*hours - 0.02*days_elapsed + np.random.normal(0, 0.5, n_samples), 1, 10)
appearance = np.clip(8 - 0.01*hours - 0.02*days_elapsed + np.random.normal(0, 0.5, n_samples), 1, 10)
odor = np.clip(8 - 0.01*hours - 0.02*days_elapsed + np.random.normal(0, 0.5, n_samples), 1, 10)

# Kalan raf ömrü (demo formül)
remaining_shelf_life = np.clip(48 - hours - days_elapsed*24 - 2*(temp/4) + 0.5*texture, 0, 48)

# DataFrame oluştur
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

# Excel’e kaydet
df.to_excel(r"C:\Users\SUMAE\Desktop\ML_Proje\demo_fish_shelf_life.xlsx", index=False)
print("Demo veri Excel dosyası 'demo_fish_shelf_life.xlsx' olarak kaydedildi.")
