from __future__ import annotations

import io
import pickle
import logging
from datetime import datetime
from uuid import uuid4
from bson import ObjectId
import numpy as np
import pandas as pd
import lightgbm as lgb

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from db.mongo import (
    data_col,
    model_col,
    metrics_col,
    system_col
)
from dotenv import load_dotenv
load_dotenv()  # this reads the .env file

# ---------------- LOGGER ----------------
logger = logging.getLogger("pre_phase_0")

# ---------------- ROUTER ----------------
router = APIRouter()

# ---------------- FEATURES ----------------
FEATURES = [
    "period",
    "age",
    "gender",
    "seniority_insured",
    "seniority_policy",
    "exposure_time",
    "type_policy",
    "type_product",
    "reimbursement",
    "new_business",
    "distribution_channel",
    "premium",
    "cost_claims_year",
    "n_medical_services",
    "n_insured_pc",
    "n_insured_mun",
    "n_insured_prov",
]

# ---------------- SCHEMA ----------------
class TrainRequest(BaseModel):
    sample_rows: int = 50000

# ---------------- UTIL ----------------
def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for f in FEATURES:
        if f not in df:
            df[f] = 0

    for col in FEATURES:
        if df[col].dtype == "object":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df[FEATURES].astype(float)

def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    else:
        return obj


# ---------------- INGEST ----------------
@router.post("/churn/ingest")
async def ingest(file: UploadFile = File(...)):
    df = pd.read_csv(io.BytesIO(await file.read()))

    if "customer_id" not in df:
        df["customer_id"] = [str(uuid4()) for _ in range(len(df))]

    data_col.insert_many(df.to_dict("records"))

    return {"rows": len(df)}

# ---------------- MODEL TRAINING ----------------
def train_model(df: pd.DataFrame, old_model=None):
    logger.info("🚀 Training GENERAL model (PRE_PHASE_0)")

    # Ensure churn exists (bootstrap)
    if "churn" not in df.columns:
        df["churn"] = np.random.randint(0, 2, size=len(df))

    customer_ids = df.get("customer_id", pd.Series([None] * len(df))).tolist()

    # remove customer_id if present, keep for return
    df_train = df.drop(columns=["customer_id"], errors="ignore")

    if "churn" not in df_train.columns:
        df_train["churn"] = np.random.randint(0, 2, size=len(df_train))

    y = df_train["churn"].astype(int)
    # keep ID columns for possible inspection
    id_cols = [c for c in ["customer_id"] if c in df.columns]

    # drop leakage / id / high-cardinality columns before encoding
    leakage_cols = [
        "lapse",
        "date_lapse_insured",
        "year_lapse_insured",
        "date_lapse_policy",
        "year_lapse_policy",
        "date_effect_insured",
        "date_effect_policy",
        "exposure_time",
        "ID",
        "ID_insured",
        "ID_policy",
    ]
    drop_cols = [c for c in leakage_cols if c in df_train.columns]
    if drop_cols:
        df_train = df_train.drop(columns=drop_cols)
        logger.info(f"Dropped columns before encoding: {drop_cols}")

    X = df_train.drop(columns=["churn"]) 

    # use ensure_features() to apply consistent label encoding (matches phase_0.py)
    X_enc = ensure_features(X)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_enc)

    Xtr, Xva, ytr, yva = train_test_split(
        Xs, y, test_size=0.1, random_state=42
    )

    train_ds = lgb.Dataset(Xtr, label=ytr)
    val_ds = lgb.Dataset(Xva, label=yva)

    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.05,
            "num_leaves": 64
        },
        train_ds,
        valid_sets=[val_ds],
        num_boost_round=200 if not old_model else 50
    )

    preds = (model.predict(Xs) >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "recall": float(recall_score(y, preds, zero_division=0)),
        "rows": len(df),
        "model_type": "GENERAL",
        "trained_at": datetime.utcnow()
    }

    # 🔑 Store GENERAL model
    model_col.delete_many({"model_type": "GENERAL"})
    model_col.insert_one({
        "model_type": "GENERAL",
        "model": pickle.dumps(model),
        "scaler": pickle.dumps(scaler),
        "trained_at": metrics["trained_at"]
    })

    metrics_col.insert_one(metrics)

    logger.info(
        f"✅ GENERAL model trained | Acc={metrics['accuracy']:.3f} | Rows={metrics['rows']}"
    )

    return metrics, customer_ids, model, scaler

# ---------------- TRAIN ENDPOINT ----------------
@router.post("/churn/train")
def train(req: TrainRequest):
    rows = list(data_col.find({}).limit(req.sample_rows))
    if not rows:
        raise HTTPException(status_code=400, detail="No data to train on")

    df = pd.DataFrame(rows).drop(columns=["_id"], errors="ignore")
    metrics, _, _, _ = train_model(df)

    # 🔑 Transition system into PHASE_0
    system_col.update_one(
        {"key": "autonomy_phase"},
        {"$set": {
            "phase": "PHASE_0",
            "phase_started_at": datetime.utcnow(),
            "processing": False
        }},
        upsert=True
    )

    logger.info("🚦 GENERAL model ready → PHASE_0 unlocked")

    return JSONResponse(
        content=convert_to_native({
            "metrics": metrics,
            "phase": "PHASE_0",
            "message": "GENERAL model trained. PHASE_0 can begin."
        })
    )
