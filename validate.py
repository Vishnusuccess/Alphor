import pandas as pd
from pandas.api.types import is_numeric_dtype

# =========================
# 1. Load data
# =========================
file_path = "/Users/vishnu/Downloads/Dataset of health insurance portfolio/Dataset of health insurance portfolio.xlsx"
df = pd.read_excel(file_path)

print("Original shape:", df.shape)

# =========================
# 2. Inspect lapse (sanity check)
# =========================
print("\nLapse distribution:")
print(df["lapse"].value_counts(dropna=False))

# =========================
# 3. Create churn column (CORRECT)
# =========================
# churn = 1 if lapse == 1, else 0
df["churn"] = (df["lapse"] == 1).astype(int)

# =========================
# 4. Print churn rate
# =========================
churn_rate = df["churn"].mean()
print(f"\nChurn rate: {churn_rate:.2%}")

# =========================
# 5. Keep only churn-related rows (optional)
# =========================
# If you want ONLY valid status rows
df = df[df["lapse"].isin([1, 2])]

print("After filtering valid lapse codes:", df.shape)

# =========================
# 6. Drop leakage columns
# =========================
leakage_cols = [
    "lapse",
    "date_lapse_insured",
    "year_lapse_insured",
    "date_lapse_policy",
    "year_lapse_policy"
]

df.drop(columns=[c for c in leakage_cols if c in df.columns], inplace=True)

# =========================
# 7. Drop identifiers (optional but recommended)
# =========================
id_cols = ["insured_id", "policy_id"]
df.drop(columns=[c for c in id_cols if c in df.columns], inplace=True)

# =========================
# 8. Save churn dataset
# =========================
output_path = "new_churn_dataset.csv"
df.to_csv(output_path, index=False)

print("\nChurn dataset saved to:", output_path)
print("Final shape:", df.shape)

# =========================
# 9. Final target check
# =========================
print("\nFinal churn stats:")
print(df.dtypes)

print("\nColumn summary stats (min, max, mean, median, mode):")
print("cols =", df.columns.tolist())
for col in df.columns:
    if col == "ID":
        continue
    print(f"\n{col}:")
    col_ser = df[col]
    try:
        if is_numeric_dtype(col_ser):
            print("  min:", col_ser.min())
            print("  max:", col_ser.max())
            print("  mean:", col_ser.mean())
            print("  median:", col_ser.median())
        else:
            print("  min:", col_ser.min())
            print("  max:", col_ser.max())

        modes = col_ser.mode(dropna=True)
        if modes.empty:
            print("  mode: <none>")
        else:
            modes_list = modes.tolist()
            if len(modes_list) == 1:
                print("  mode:", modes_list[0])
            else:
                print("  mode:", modes_list)
    except Exception as e:
        print("  error computing stats:", e)

