import pandas as pd
import zipfile
import os

# ==================================================
# 1. PATHS
# ==================================================

BASE_DIR = os.getcwd()

hc251_zip = os.path.join(BASE_DIR, "h251xlsx.zip")
hc252_zip = os.path.join(BASE_DIR, "h252xlsx.zip")

hc251_dir = os.path.join(BASE_DIR, "hc251")
hc252_dir = os.path.join(BASE_DIR, "hc252")

os.makedirs(hc251_dir, exist_ok=True)
os.makedirs(hc252_dir, exist_ok=True)

# ==================================================
# 2. EXTRACT ZIP FILES
# ==================================================

with zipfile.ZipFile(hc251_zip, "r") as z:
    z.extractall(hc251_dir)

with zipfile.ZipFile(hc252_zip, "r") as z:
    z.extractall(hc252_dir)

# ==================================================
# 3. FIND EXCEL FILES
# ==================================================

def find_xlsx(folder):
    for f in os.listdir(folder):
        if f.lower().endswith(".xlsx"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No .xlsx found in {folder}")

hc251_path = find_xlsx(hc251_dir)
hc252_path = find_xlsx(hc252_dir)

# ==================================================
# 4. LOAD HC-251 (MODEL FEATURES)
# ==================================================

hc251_features = [
    # ID
    "DUPERSID",

    # Demographics
    "AGE","SEX","RACEV1X","EDUCYR","MARRY","POVCAT","EMPST","FAMSZE",

    # Insurance baseline
    "INSURCOV","PRIVINS","PUBINS",

    # Cost & access stress
    "UNABLE_PAY","DELAY_CARE","FORGO_CARE","RXCOST","MEDBILLS",

    # Utilization
    "OBTOTV","ERTOT","IPDIS","RXEXP",

    # Health burden
    "CHRONIC","DIABDX","HIBPDX","ASTHDX"
]

hc251 = pd.read_excel(
    hc251_path,
    usecols=lambda c: c in hc251_features
)

# ==================================================
# 5. LOAD HC-252 (MONTHLY INSURANCE STATUS)
# ==================================================

hc252 = pd.read_excel(hc252_path)

# REAL MEPS monthly insurance columns (Year 1)
month_cols = [
    "INSJAY1X","INSFEY1X","INSMAY1X","INSAPY1X","INSMYY1X","INSJUY1X",
    "INSJLY1X","INSAUY1X","INSSEY1X","INSOCY1X","INSNOY1X","INSDEY1X"
]

hc252 = hc252[["DUPERSID"] + month_cols]

# ==================================================
# 6. DERIVE CHURN (VECTORIZED, FAST)
# ==================================================
# MEPS coding:
# 1 = Private
# 2 = Public
# 3 = Uninsured

# Count uninsured months
hc252["UNINS_MONTHS"] = (hc252[month_cols] == 3).sum(axis=1)

# Count insurance transitions
hc252["TRANSITIONS"] = (
    hc252[month_cols].ne(hc252[month_cols].shift(axis=1))
).sum(axis=1) - 1

# Churn definition
hc252["CHURN"] = (
    (hc252["UNINS_MONTHS"] >= 2) |
    (hc252["TRANSITIONS"] >= 2)
).astype(int)

hc252_churn = hc252[["DUPERSID", "CHURN"]]

# ==================================================
# 7. FINAL MODELING DATASET
# ==================================================

final_df = hc251.merge(hc252_churn, on="DUPERSID", how="inner")

print("Final dataset shape:", final_df.shape)
print("Churn rate:", final_df["CHURN"].mean().round(3))
print(final_df.head())
