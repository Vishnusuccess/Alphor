# phase0.py
from __future__ import annotations
import io, pickle, logging
from datetime import datetime
from uuid import uuid4
import math
import pandas as pd
import numpy as np
import lightgbm as lgb

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from db.mongo import (
    data_col,
    model_col,
    predictions_col,
    feedback_col,
    metrics_col,
    system_col,
    client_data_col
)

# ---------------- LOGGER ----------------
logger = logging.getLogger("phase_0")

# ---------------- ROUTER ----------------
router = APIRouter()

# ---------------- FEATURES ----------------
FEATURES = [
    "period", "age", "gender", "seniority_insured", "seniority_policy",
    "exposure_time", "type_policy", "type_product", "reimbursement",
    "new_business", "distribution_channel", "premium", "cost_claims_year",
    "n_medical_services", "n_insured_pc", "n_insured_mun", "n_insured_prov",
]

PHASE_1_ACCURACY_THRESHOLD = 0.70


def clean_row(row):
    clean = {}
    for k, v in row.items():
        # Replace NaN or Inf with 0 for numeric columns
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = 0.0
        else:
            clean[k] = v
    return clean

# ---------------- SCHEMA ----------------
class FeedbackItem(BaseModel):
    ID: str
    actual_churn: int

# ---------------- UTIL FUNCTIONS ----------------
def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for f in FEATURES:
        if f not in df:
            df[f] = 0
    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df[FEATURES].astype(float)

def load_best_model():
    """Return CLIENT_PHASE_0 if exists, otherwise GENERAL"""
    client = model_col.find_one({"model_type":"CLIENT_PHASE_0"})
    if client:
        return pickle.loads(client["model"]), pickle.loads(client["scaler"]), "CLIENT_PHASE_0"
    general = model_col.find_one({"model_type":"GENERAL"})
    if not general:
        raise RuntimeError("No model available (GENERAL missing)")
    return pickle.loads(general["model"]), pickle.loads(general["scaler"]), "GENERAL"

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj

def risk_label(prob: float) -> str:
    if prob >= 0.6:
        return "High"
    elif prob >= 0.3:
        return "Medium"
    return "Low"

# ---------------- PREDICT ----------------
# @router.post("/predict")
# async def predict_phase_0(file: UploadFile = File(...)):
#     model, scaler, model_type = load_best_model()

#     df = pd.read_csv(io.BytesIO(await file.read()))
#     if "ID" not in df.columns:
#         raise HTTPException(400, "CSV must contain stable ID column")

#     # Store client data
#     for _, row in df.iterrows():
#         client_data_col.update_one(
#             {"ID": row["ID"]},
#             {"$set": row.to_dict() | {"ingested_at": datetime.utcnow()}},
#             upsert=True
#         )

#     X = ensure_features(df)
#     probs = model.predict(scaler.transform(X))
#     preds = (probs >= 0.5).astype(int)

#     df["churn_prob"] = probs
#     df["prediction"] = preds

#     for _, row in df.iterrows():
#         predictions_col.update_one(
#             {"ID": row["ID"]},
#             {"$set": {
#                 "ID": row["ID"],
#                 "prediction": int(row["prediction"]),
#                 "churn_prob": float(row["churn_prob"]),
#                 "model_type": model_type,
#                 "timestamp": datetime.utcnow()
#             }},
#             upsert=True
#         )

#     return JSONResponse({
#         "phase": "PHASE_0",
#         "rows_scored": len(df),
#         "model_used": model_type
#     })

# ---------------- MANUAL FEEDBACK ----------------
@router.post("/feedback")
def submit_feedback_phase_0(items: list[FeedbackItem]):
    feedback_col.insert_many([
        {
            "ID": item.ID,
            "actual_churn": item.actual_churn,
            "timestamp": datetime.utcnow()
        }
        for item in items
    ])
    return {"feedback_received": len(items)}

@router.post("/predict")
async def predict_phase_0(file: UploadFile = File(...)):
    model, scaler, model_type = load_best_model()

    df = pd.read_csv(io.BytesIO(await file.read()))
    if "ID" not in df.columns:
        raise HTTPException(400, "CSV must contain stable ID column")

    # Store client data safely
    for _, row in df.iterrows():
        safe_row = clean_row(row.to_dict())
        safe_row["ingested_at"] = datetime.utcnow()
        client_data_col.update_one(
            {"ID": row["ID"]},
            {"$set": safe_row},
            upsert=True
        )

    X = ensure_features(df)
    probs = model.predict(scaler.transform(X))
    preds = (probs >= 0.5).astype(int)

    df["churn_prob"] = probs
    df["prediction"] = preds

    # Store predictions safely
    for _, row in df.iterrows():
        client_id = row["ID"]
        churn_prob = float(row["churn_prob"]) if not math.isnan(row["churn_prob"]) else 0.0
        predictions_col.update_one(
            {"ID": client_id},
            {"$set": {
                "ID": client_id,
                "prediction": int(row["prediction"]),
                "churn_prob": churn_prob,
                "model_type": model_type,
                "timestamp": datetime.utcnow()
            }},
            upsert=True
        )

    return JSONResponse({
    "phase": "PHASE_0",
    "rows_scored": len(df),
    "model_used": model_type,
    "message": f"Processed {len(df)} rows with {model_type} model"
})

# ---------------- BULK FEEDBACK ----------------
@router.post("/feedback/bulk")
async def submit_bulk_feedback(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files supported")

    df = pd.read_csv(io.BytesIO(await file.read()))
    if not {"ID", "churn"}.issubset(df.columns):
        raise HTTPException(400, "CSV must contain ID and churn")

    df["churn"] = pd.to_numeric(df["churn"], errors="coerce")
    df = df[df["churn"].isin([0, 1])]

    known_ids = set(p["ID"] for p in predictions_col.find({}, {"ID": 1}))
    df = df[df["ID"].isin(known_ids)]

    if df.empty:
        raise HTTPException(400, "No matching predictions found")

    feedback_col.insert_many([
        {
            "ID": row["ID"],
            "actual_churn": int(row["churn"]),
            "timestamp": datetime.utcnow()
        }
        for _, row in df.iterrows()
    ])

    return {"feedback_received": len(df)}

# ---------------- RETRAIN ----------------
@router.post("/retrain")
def retrain_phase_0():
    feedback = list(feedback_col.find({}))
    if len(feedback) < 10:
        raise HTTPException(400, "Not enough feedback to retrain")

    fb_df = pd.DataFrame(feedback)
    preds_df = pd.DataFrame(list(predictions_col.find({})))
    raw_df = pd.DataFrame(list(client_data_col.find({})))

    # Merge predictions, feedback, and raw data
    df = (
        preds_df
        .merge(fb_df[["ID", "actual_churn"]], on="ID")
        .merge(raw_df, on="ID", how="left")
    )
    df["churn"] = df["actual_churn"]

    # Train model
    acc, model, scaler = train_client_phase_0_model(df)

    model_col.delete_many({"model_type": "CLIENT_PHASE_0"})
    model_col.insert_one({
        "model_type": "CLIENT_PHASE_0",
        "model": pickle.dumps(model),
        "scaler": pickle.dumps(scaler),
        "trained_at": datetime.utcnow(),
        "accuracy": acc
    })

    metrics_col.insert_one({
        "phase": "PHASE_0",
        "accuracy": acc,
        "rows": len(df),
        "trained_at": datetime.utcnow()
    })

    # Update system phase if ready
    if acc >= PHASE_1_ACCURACY_THRESHOLD:
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {
                "phase": "PHASE_1",
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )

    return {
        "retrained": True,
        "accuracy": acc,
        "phase_1_ready": acc >= PHASE_1_ACCURACY_THRESHOLD
    }

# ---------------- TRAINING FUNCTION ----------------
def train_client_phase_0_model(feedback_df: pd.DataFrame, general_sample_fraction: float = 0.05):
    if "churn" not in feedback_df.columns:
        raise ValueError("Feedback dataframe must include 'churn' column")

    fb_train, fb_val = train_test_split(feedback_df, test_size=0.1, random_state=42)
    general_doc = model_col.find_one({"model_type": "GENERAL"})
    if not general_doc:
        raise RuntimeError("GENERAL model not found in MongoDB")

    general_model = pickle.loads(general_doc["model"])
    scaler = pickle.loads(general_doc["scaler"])

    general_data = list(data_col.find({}))
    if general_data and general_sample_fraction > 0:
        gen_df = pd.DataFrame(general_data).drop(columns=["_id"], errors="ignore")
        anchor_size = max(100, int(len(fb_train) * general_sample_fraction))
        gen_sample = gen_df.sample(min(anchor_size, len(gen_df)), random_state=42)
        train_df = pd.concat([fb_train, gen_sample], ignore_index=True)
    else:
        train_df = fb_train

    X_train = ensure_features(train_df)
    y_train = train_df["churn"].astype(int)
    X_val = ensure_features(fb_val)
    y_val = fb_val["churn"].astype(int)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_ds = lgb.Dataset(X_train_scaled, label=y_train)
    val_ds = lgb.Dataset(X_val_scaled, label=y_val)

    model = lgb.train(
        params={"objective": "binary", "metric": "binary_logloss", "learning_rate": 0.03, "num_leaves": 64},
        train_set=train_ds,
        valid_sets=[val_ds],
        num_boost_round=30,
        init_model=general_model
    )

    preds = (model.predict(X_val_scaled) >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_val, preds))

    # Backup old GENERAL
    model_col.insert_one({
        "model_type": "GENERAL_BACKUP",
        "model": general_doc["model"],
        "scaler": general_doc["scaler"],
        "backed_up_at": datetime.utcnow()
    })

    # Save updated GENERAL
    model_col.update_one(
        {"model_type": "GENERAL"},
        {"$set": {
            "model": pickle.dumps(model),
            "scaler": general_doc["scaler"],
            "accuracy": accuracy,
            "trained_at": datetime.utcnow()
        }},
        upsert=True
    )

    return accuracy, model, scaler