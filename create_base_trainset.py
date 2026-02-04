import numpy as np
import pandas as pd

np.random.seed(42)

N = 50_000

# -------------------------
# Core identifiers
# -------------------------
ID_policy = np.random.randint(1, 45_163, N)
ID_insured = np.random.randint(1, 1_887, N)
# robust string concat for numpy unicode arrays
ID = np.char.add(np.char.add(ID_policy.astype(str), "_"), ID_insured.astype(str))

period = np.random.choice([2017, 2018, 2019], size=N, p=[0.33, 0.34, 0.33])

# -------------------------
# Demographics
# -------------------------
age = np.clip(np.random.normal(45.5, 18, N), 0, 99).astype(int)
gender = np.random.choice(["F", "M"], size=N, p=[0.54, 0.46])

# -------------------------
# Seniority & exposure
# -------------------------
seniority_insured = np.clip(np.random.exponential(scale=13, size=N), 0, 87).astype(int)
seniority_policy = np.clip(seniority_insured + np.random.randint(-2, 3, N), 0, 87)

exposure_time = np.where(
    np.random.rand(N) < 0.93, 1.0, np.round(np.random.uniform(0.1, 0.9, N), 2)
)

# -------------------------
# Product / channel
# -------------------------
type_policy = np.random.choice(["I", "C"], size=N, p=[0.72, 0.28])
type_product = np.random.choice(["S", "D"], size=N, p=[0.75, 0.25])
reimbursement = np.random.choice(["Yes", "No"], size=N, p=[0.35, 0.65])
new_business = np.random.choice(["Yes", "No"], size=N, p=[0.18, 0.82])

distribution_channel = np.random.choice(
    ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
    size=N,
    p=[0.05, 0.05, 0.06, 0.08, 0.07, 0.1, 0.12, 0.15, 0.32]
)

# -------------------------
# Premium (right-skewed)
# -------------------------
premium = np.round(
    np.random.lognormal(mean=np.log(770), sigma=0.6, size=N), 2
)
premium = np.clip(premium, 33.33, 19_860)

# -------------------------
# Medical usage & claims
# -------------------------
n_medical_services = np.random.poisson(lam=6, size=N)
n_medical_services[np.random.rand(N) < 0.30] = 0  # zero inflation

claims = np.where(
    n_medical_services == 0,
    0,
    np.random.lognormal(mean=np.log(250), sigma=1.1, size=N)
)

claims = np.clip(claims, 0, 78_035)
claims = np.round(claims, 2)

# -------------------------
# Family / geographic size
# -------------------------
n_insured_pc = np.random.randint(1, 2_519, N)
n_insured_mun = np.random.randint(1, 19_316, N)
n_insured_prov = np.random.randint(4, 29_466, N)

# -------------------------
# Churn generation (REAL SIGNAL)
# -------------------------
loss_ratio = claims / (premium + 1)

logit = (
    -2.2
    + 0.9 * (loss_ratio > 1.2)
    + 0.6 * (new_business == "Yes").astype(int)
    + 0.4 * (n_medical_services == 0)
    - 0.03 * seniority_insured
    + 0.015 * (age < 30)
    + 0.25 * (premium > np.percentile(premium, 75))
)

churn_prob = 1 / (1 + np.exp(-logit))
churn = (np.random.rand(N) < churn_prob).astype(int)

print("Churn rate:", churn.mean())

# -------------------------
# Final dataset
# -------------------------
df = pd.DataFrame({
    "ID": ID,
    "ID_policy": ID_policy,
    "ID_insured": ID_insured,
    "period": period,
    "age": age,
    "gender": gender,
    "seniority_insured": seniority_insured,
    "seniority_policy": seniority_policy,
    "exposure_time": exposure_time,
    "type_policy": type_policy,
    "type_product": type_product,
    "reimbursement": reimbursement,
    "new_business": new_business,
    "distribution_channel": distribution_channel,
    "premium": premium,
    "cost_claims_year": claims,
    "n_medical_services": n_medical_services,
    "n_insured_pc": n_insured_pc,
    "n_insured_mun": n_insured_mun,
    "n_insured_prov": n_insured_prov,
    "churn": churn
})

output_path = "base_trainset.csv"
df.to_csv(output_path, index=False)

# -------------------------
# Simple churn prediction model
# -------------------------
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import json

print("\nPreparing data for modeling...")

# features / target
if "churn" not in df.columns:
    raise RuntimeError("Target column 'churn' not found in dataframe")

X = df.drop(columns=["churn"])
y = df["churn"].astype(int)

# Drop identifier columns to avoid one-hot expansion (IDs are high-cardinality)
id_cols = ["ID", "ID_policy", "ID_insured"]
present_ids = [c for c in id_cols if c in X.columns]
if present_ids:
    X.drop(columns=present_ids, inplace=True)
    print("Dropped identifier columns before encoding:", present_ids)

# encode categoricals with one-hot (drop_first to reduce collinearity)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# fill any remaining NaNs
X_enc.fillna(0, inplace=True)

print("Features shape after encoding:", X_enc.shape)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, stratify=y, random_state=42
)

print("Preparing resampling pipeline (undersample majority -> SMOTE)...")
# decide undersample target: reduce majority to at most 5x the minority to limit memory
n_pos = int(y_train.sum())
n_neg = int(y_train.shape[0] - n_pos)
desired_neg = min(int(max(1, n_pos) * 5), n_neg)
if desired_neg < n_neg:
    sampling_strategy = {0: desired_neg, 1: n_pos}
else:
    sampling_strategy = "auto"

rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
sm = SMOTE(random_state=42)
pipeline = ImbPipeline(steps=[("rus", rus), ("smote", sm)])

try:
    X_res, y_res = pipeline.fit_resample(X_train, y_train)
    print("Resampled training shape:", X_res.shape, "->", int(y_res.sum()), "positives")
except MemoryError:
    print("MemoryError during resampling pipeline; falling back to undersampling only")
    X_res, y_res = rus.fit_resample(X_train, y_train)
    print("Resampled (undersample only) shape:", X_res.shape, "->", int(y_res.sum()), "positives")

print("Running RandomizedSearchCV for RandomForest hyperparameters (fewer iters, single-job)...")
param_dist = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 8, 10, 12, 16, None],
    "max_features": ["sqrt", "log2", 0.2, 0.5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

base_rf = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
search = RandomizedSearchCV(
    base_rf,
    param_distributions=param_dist,
    n_iter=12,
    scoring="f1",
    n_jobs=1,
    cv=cv,
    verbose=1,
    random_state=42,
)

search.fit(X_res, y_res)
rf_best = search.best_estimator_

print("Best params:", search.best_params_)

print("Evaluating best model on test set...")
y_pred = rf_best.predict(X_test)
y_proba = rf_best.predict_proba(X_test)[:, 1] if hasattr(rf_best, "predict_proba") else None

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1: {f1:.4f}")
print(f"ROC AUC: {auc:.4f}")

print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

# save model and feature list for later inference
model_path = "churn_rf_model.joblib"
features_path = "churn_features.json"
joblib.dump(rf_best, model_path)
with open(features_path, "w") as fh:
    json.dump(X_enc.columns.tolist(), fh)

print("Saved model to:", model_path)
print("Saved feature list to:", features_path)

# -------------------------
# Try XGBoost and inspect top churn-probability cases
# -------------------------
try:
    import xgboost as xgb
except Exception:
    xgb = None

if xgb is None:
    print("xgboost not installed — skip XGBoost training. Install with 'pip install xgboost'.")
else:
    print("\nTraining XGBoost classifier on resampled data...")
    xgb_clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=1,
        random_state=42,
        verbosity=0,
    )
    xgb_clf.fit(X_res, y_res)

    print("Evaluating XGBoost on test set...")
    y_pred_xgb = xgb_clf.predict(X_test)
    y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]

    acc_x = accuracy_score(y_test, y_pred_xgb)
    prec_x = precision_score(y_test, y_pred_xgb, zero_division=0)
    rec_x = recall_score(y_test, y_pred_xgb, zero_division=0)
    f1_x = f1_score(y_test, y_pred_xgb, zero_division=0)
    auc_x = roc_auc_score(y_test, y_proba_xgb)

    print(f"XGB Accuracy: {acc_x:.4f}")
    print(f"XGB Precision: {prec_x:.4f}")
    print(f"XGB Recall: {rec_x:.4f}")
    print(f"XGB F1: {f1_x:.4f}")
    print(f"XGB ROC AUC: {auc_x:.4f}")

    print("\nTop churn-probability cases (top 20) and whether they are true positives:")
    results = pd.DataFrame(index=X_test.index)
    results["prob"] = y_proba_xgb
    results["pred"] = y_pred_xgb
    results["true"] = y_test.values
    top_n = 20
    top = results.sort_values("prob", ascending=False).head(top_n)

    # attach IDs if present in original df
    id_cols = [c for c in ["ID", "ID_policy", "ID_insured"] if c in df.columns]
    if id_cols:
        top = top.join(df[id_cols])

    print(top[[*id_cols, "prob", "pred", "true"]] if id_cols else top[["prob", "pred", "true"]])

    tp_count = int((top["true"] == 1).sum())
    print(f"\nOf top {top_n} predicted churn cases, {tp_count} are true positives.")

    # save xgb model
    try:
        joblib.dump(xgb_clf, "churn_xgb_model.joblib")
        print("Saved XGBoost model to: churn_xgb_model.joblib")
    except Exception as e:
        print("Could not save XGBoost model:", e)