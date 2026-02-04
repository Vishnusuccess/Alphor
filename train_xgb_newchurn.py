import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

try:
    import xgboost as xgb
except Exception:
    xgb = None


def main(csv_path="new_churn_dataset.csv", top_n=20):
    print("Loading:", csv_path)
    df = pd.read_csv(csv_path)
    print("Original shape:", df.shape)

    if "churn" not in df.columns:
        raise RuntimeError("CSV must contain 'churn' column as target")

    # keep ID columns for final inspection if present
    id_cols = [c for c in ["ID", "ID_policy", "ID_insured"] if c in df.columns]
    ids = df[id_cols].copy() if id_cols else None

    # drop leakage/id cols before encoding
    leakage_cols = [
        "lapse",
        "date_lapse_insured",
        "year_lapse_insured",
        "date_lapse_policy",
        "year_lapse_policy",
        'date_effect_insured', 
        'date_effect_policy',
        'exposure_time'
    ]
    drop_cols = [c for c in leakage_cols + id_cols if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print("Dropped columns before encoding:", drop_cols)

    X = df.drop(columns=["churn"]) 
    y = df["churn"].astype(int)

    # identify categoricals
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("Categorical columns:", cat_cols)

    # one-hot encode; fallback to ordinal if dummies explosion
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    print("Features after get_dummies:", X_enc.shape)

    if X_enc.shape[1] > 5000:
        print("High dimensionality after one-hot: falling back to OrdinalEncoder")
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_cat = X[cat_cols].astype(str).fillna("__nan__") if cat_cols else pd.DataFrame()
        if not X_cat.empty:
            X_cat_enc = pd.DataFrame(enc.fit_transform(X_cat), index=X.index, columns=cat_cols)
            X_num = X.drop(columns=cat_cols)
            X_enc = pd.concat([X_num.reset_index(drop=True), X_cat_enc.reset_index(drop=True)], axis=1)
        print("Features after ordinal encoding:", X_enc.shape)

    # fill NaNs
    X_enc.fillna(0, inplace=True)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.2, stratify=y, random_state=42
    )

    # Train on the natural distribution — use class weighting for XGBoost
    n_pos = int(y_train.sum())
    n_neg = int(y_train.shape[0] - n_pos)
    scale_pos_weight = n_neg / max(1, n_pos)
    print(
        f"Training on natural distribution: positives={n_pos}, negatives={n_neg}, scale_pos_weight={scale_pos_weight:.2f}"
    )

    if xgb is None:
        print("xgboost not installed — install with: pip install xgboost")
        return

    print("Training XGBoost classifier with class weighting...")
    clf = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=1,
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_proba = clf.predict_proba(X_test)[:, 1]
    # apply probability cutoff
    cutoff = 0.8
    y_pred = (y_proba >= cutoff).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")

    print("\nClassification report:\n", classification_report(y_test, y_pred, zero_division=0))

    # inspect top-N predicted churn probabilities
    results = pd.DataFrame(index=X_test.index)
    results["prob"] = y_proba
    results["pred"] = y_pred
    results["true"] = y_test.values
    top = results.sort_values("prob", ascending=False).head(top_n)

    if ids is not None:
        top = top.join(ids.loc[top.index])

    display_cols = ([*id_cols] if id_cols else []) + ["prob", "pred", "true"]
    print(f"\nTop {top_n} churn-probability cases:")
    print(top[display_cols])

    tp_count = int((top["true"] == 1).sum())
    print(f"\nOf top {top_n} predicted churn cases, {tp_count} are true positives.")

    # -------------------------
    # Feature importance + SHAP explanations for top cases
    # -------------------------
    try:
        # global feature importance
        fi = pd.Series(clf.feature_importances_, index=X_enc.columns).sort_values(ascending=False)
        print("\nTop all features by model importance:")
        print(fi.head(50))

        # SHAP explanations (compute only for the top rows)
        try:
            import shap
        except Exception as e:
            shap = None
            print("shap not installed or failed to import:", e)

        if shap is not None:
            print("\nComputing SHAP values for top cases (this may take a moment)...")
            X_top = X_test.loc[top.index]
            try:
                explainer = shap.TreeExplainer(clf)
                shap_res = explainer(X_top)
                # shap_res may be an Explanation object or array/list
                try:
                    shap_vals = shap_res.values
                except Exception:
                    shap_vals = shap_res

                # handle multiclass/list output: pick class-1 contributions
                if isinstance(shap_vals, (list, tuple)):
                    shap_vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]

                for i, ridx in enumerate(X_top.index):
                    print(f"\nTop case index {ridx} — true={int(top.loc[ridx,'true'])} prob={top.loc[ridx,'prob']:.4f}")
                    # attach IDs if available
                    if ids is not None:
                        idinfo = ids.loc[ridx].to_dict()
                        print(" IDs:", idinfo)

                    row_shap = shap_vals[i]
                    abs_idx = np.argsort(np.abs(row_shap))[::-1][:10]
                    print(" Top SHAP contributors:")
                    for fi_idx in abs_idx:
                        feat = X_top.columns[fi_idx]
                        feat_val = X_top.iloc[i, fi_idx]
                        contrib = row_shap[fi_idx]
                        print(f"  {feat}: value={feat_val}  shap={contrib:.4f}")
            except Exception as e:
                print("Failed to compute SHAP explanations:", e)
        else:
            print("SHAP not available — install with 'pip install shap' to get per-case explanations")
    except Exception as e:
        print("Failed to compute feature importance or SHAP:", e)

    # save model and feature list
    model_path = "xgb_newchurn_model.joblib"
    features_path = "xgb_newchurn_features.json"
    joblib.dump(clf, model_path)
    with open(features_path, "w") as fh:
        json.dump(X_enc.columns.tolist(), fh)
    print("Saved XGBoost model to:", model_path)
    print("Saved feature list to:", features_path)


if __name__ == "__main__":
    main()
