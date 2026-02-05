# phase1.py  (FULL UPDATED FILE - FAST BATCH LOOP, SAME LOGIC / SAME ENDPOINTS)

import io
import gzip
import pickle
from uuid import uuid4
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb

from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from fastapi_utils.tasks import repeat_every
from pymongo import UpdateOne

from db.mongo import (
    db,
    client_data_col,
    model_col,
    predictions_col,
    system_col,
    metrics_col,
    phase1_feedback_col,
    actions_col,
)

from routers.pre_phase_0 import ensure_features
from routers.phase_0 import load_best_model  # phase 0 function

router = APIRouter()

# ---------------- FEATURES (must match Phase 0 FEATURES) ----------------
FEATURES = [
    "period", "age", "gender", "seniority_insured", "seniority_policy",
    "exposure_time", "type_policy", "type_product", "reimbursement",
    "new_business", "distribution_channel", "premium", "cost_claims_year",
    "n_medical_services", "n_insured_pc", "n_insured_mun", "n_insured_prov",
]

# ---------------- THRESHOLDS ----------------
EXECUTION_THRESHOLD = 0.65
HUMAN_REVIEW_THRESHOLD = 0.85
FEEDBACK_THRESHOLD = 5                 # trigger retraining after this many feedback entries
AUTONOMOUS_LOOP_SECONDS = 10
PHASE_1_ACCURACY_THRESHOLD = 0.45

# ---------------- PERFORMANCE TUNING ----------------
BATCH_SIZE = 2000                      # how many customers per 10s tick
BULK_BATCH_SIZE = 10000                # bulk write chunking if needed (usually fine)

# ---------------- ACTIONS ----------------
def calculate_discount(prob: float, ltv: float = 1000):
    if prob >= 0.85:
        return min(200, 0.2 * ltv)
    if prob >= 0.65:
        return min(100, 0.1 * ltv)
    return 0


def generate_demo_brief(customer: dict) -> dict:
    prob = float(customer.get("churn_prob", 0))

    risk = "NO_RISK"
    decision = "NONE"
    actions = []
    discount = 0
    email_draft = ""
    call_script = ""

    if prob < 0.40:
        risk = "NO_RISK"
        decision = "NONE"
        brief = "Customer has no churn risk. No action required."

    elif prob < 0.50:
        risk = "LOW"
        decision = "EXECUTE"
        actions = ["send_email"]
        email_draft = (
            f"Subject: Checking in, {customer['ID']}\n\n"
            "Dear Customer,\nWe hope you're doing well. Let us know if you need anything.\n"
        )
        brief = "Low churn risk. Email will be automatically sent."

    elif prob < 0.70:
        risk = "MODERATE"
        decision = "RECOMMEND"
        actions = ["send_email", "offer_discount"]
        discount = calculate_discount(prob)
        email_draft = (
            f"Subject: A Special Offer for You, {customer['ID']}\n\n"
            "We value your relationship with us. A discount is recommended.\n"
        )
        brief = "Moderate churn risk detected.\n⚠ Human review required before execution."

    else:
        risk = "HIGH"
        decision = "ESCALATE"
        actions = ["schedule_call", "send_email", "offer_discount"]
        discount = calculate_discount(prob)
        email_draft = (
            f"Subject: Important – Please Contact Us, {customer['ID']}\n\n"
            "Your account requires immediate attention.\n"
        )
        call_script = (
            f"Hello {customer['ID']}, this is your insurance representative.\n"
            "We’d like to discuss your account and available options."
        )
        brief = "High churn risk.\n🚨 Escalation required. Human approval mandatory."

    return {
        "risk": risk,
        "decision": decision,
        "actions": actions,
        "discount": discount,
        "brief": brief,
        "email_draft": email_draft,
        "call_script": call_script
    }


def execute_actions(customer: dict, demo: dict):
    """
    ORIGINAL LOGIC (kept).
    NOTE: This is slow if called 200k times because it insert_one per row.
    The fast loop below does the same write, but in bulk (execute_actions_bulk_docs).
    """
    risk = demo["risk"]

    executed = risk == "LOW"
    human_review_required = risk in ["MODERATE", "HIGH"]

    if executed:
        for a in demo["actions"]:
            if a == "send_email":
                logger.info(f"[AUTO EMAIL] {customer['ID']}")
            elif a == "offer_discount":
                logger.info(f"[AUTO DISCOUNT] {customer['ID']}")

    actions_col.insert_one({
        "ID": customer["ID"],
        "phase": "PHASE_1",
        "risk": demo["risk"],
        "decision": demo["decision"],
        "actions": demo["actions"],
        "discount": float(demo["discount"]),
        "churn_prob": float(customer["churn_prob"]),
        "executed": executed,
        "human_review_required": human_review_required,
        "brief": demo["brief"],
        "email_draft": demo["email_draft"],
        "call_script": demo["call_script"],
        "timestamp": datetime.utcnow()
    })


def execute_actions_bulk_docs(customers: List[dict], demos: List[dict], timestamp: datetime) -> List[dict]:
    """
    SAME DATA/LOGIC as execute_actions(), but returns documents for insert_many (FAST).
    """
    docs = []
    for customer, demo in zip(customers, demos):
        risk = demo["risk"]
        executed = risk == "LOW"
        human_review_required = risk in ["MODERATE", "HIGH"]

        # keep your logging behavior for LOW auto execution (optional, but kept)
        if executed:
            for a in demo["actions"]:
                if a == "send_email":
                    logger.info(f"[AUTO EMAIL] {customer['ID']}")
                elif a == "offer_discount":
                    logger.info(f"[AUTO DISCOUNT] {customer['ID']}")

        docs.append({
            "ID": customer["ID"],
            "phase": "PHASE_1",
            "risk": demo["risk"],
            "decision": demo["decision"],
            "actions": demo["actions"],
            "discount": float(demo["discount"]),
            "churn_prob": float(customer.get("churn_prob", 0.0)),
            "executed": executed,
            "human_review_required": human_review_required,
            "brief": demo["brief"],
            "email_draft": demo["email_draft"],
            "call_script": demo["call_script"],
            "timestamp": timestamp
        })
    return docs


def complete_phase0_retrain(accuracy: float):
    logger.info(f"Phase 0 retrain finished | Accuracy={accuracy:.2f}")
    if accuracy >= PHASE_1_ACCURACY_THRESHOLD:
        logger.info("✅ Phase 1 ready → updating system state to start Phase 1 automatically")
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {
                "phase": "PHASE_2",
                "processing": False,
                "processing_phase2": False,
                "awaiting_feedback": False,
                "updated_at": datetime.utcnow()
            }},
            upsert=True
        )

# ---------------- PREDICT + EXECUTE (FAST BULK WRITES) ----------------
@router.post("/predict/phase1")
async def predict_phase_1(file: UploadFile = File(...)):
    """
    Same endpoint/response/logic, but fast:
    - batch compute demos
    - actions insert_many
    - predictions insert_many
    """
    model, scaler, model_type = load_best_model()  # PHASE_0 or GENERAL

    raw = await file.read()
    if file.filename and file.filename.endswith(".gz"):
        try:
            raw = gzip.decompress(raw)
        except OSError:
            raise HTTPException(400, "Invalid .gz file")

    df = pd.read_csv(io.BytesIO(raw))

    # Keep your behavior
    if "ID" not in df.columns:
        df["ID"] = [str(uuid4()) for _ in range(len(df))]

    X = ensure_features(df)
    probs = model.predict(scaler.transform(X))
    probs = np.where(np.isfinite(probs), probs, 0.0)
    preds = (probs >= 0.5).astype(int)

    df["churn_prob"] = probs
    df["prediction"] = preds

    now = datetime.utcnow()

    # Build customers + demos in memory
    customers = df[["ID", "churn_prob", "prediction"]].to_dict("records")
    demos = [generate_demo_brief(c) for c in customers]

    # Bulk actions (same content as execute_actions)
    action_docs = execute_actions_bulk_docs(customers, demos, now)
    if action_docs:
        actions_col.insert_many(action_docs, ordered=False)

    # Bulk predictions (your original logic kept; insert_many)
    pred_docs = (
        df[["ID", "prediction", "churn_prob"]]
        .assign(model_type=model_type, timestamp=now)
        .to_dict("records")
    )
    if pred_docs:
        predictions_col.insert_many(pred_docs, ordered=False)

    return JSONResponse({
        "phase": "PHASE_1",
        "model_used": model_type,
        "rows_scored": len(df),
        "message": "Predictions + actions executed"
    })


# ---------------- AUTONOMOUS PHASE 1 LOOP (FAST BATCH, UPDATES STATE) ----------------
@router.on_event("startup")
@repeat_every(seconds=AUTONOMOUS_LOOP_SECONDS, wait_first=True)
def autonomous_loop_phase1():
    """
    Same high-level logic, but FAST and incremental:
    - processes BATCH_SIZE every 10 seconds
    - does NOT load all client data into pandas
    - does NOT compute processed_ids list
    - marks customers as phase1_processed True
    - updates system_col progress so UI can show updates via /system/state
    """
    # 1) Load system state
    state = system_col.find_one({"key": "autonomy_phase"})
    if not state or state.get("phase") != "PHASE_1":
        return

    # 2) Acquire processing lock (fast + safe)
    lock = system_col.find_one_and_update(
        {"key": "autonomy_phase", "processing": {"$ne": True}},
        {"$set": {"processing": True, "last_tick_at": datetime.utcnow()}},
        return_document=True
    )
    if not lock:
        return

    started = datetime.utcnow()
    try:
        total = client_data_col.count_documents({})
        done = client_data_col.count_documents({"phase1_processed": True})

        # If nothing to do
        if total == 0:
            system_col.update_one(
                {"key": "autonomy_phase"},
                {"$set": {
                    "processing": False,
                    "phase1_progress_total": 0,
                    "phase1_progress_done": 0,
                    "phase1_complete": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            return

        # If complete
        if done >= total:
            system_col.update_one(
                {"key": "autonomy_phase"},
                {"$set": {
                    "processing": False,
                    "phase1_progress_total": int(total),
                    "phase1_progress_done": int(done),
                    "phase1_complete": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            return

        # 3) Feedback gate (your original concept kept)
        feedback_count = phase1_feedback_col.count_documents({})
        INITIAL_FEEDBACK_THRESHOLD = 5

        # 4) Decide model to use (kept)
        model_doc = model_col.find_one({"model_type": "CLIENT_PHASE_1"})
        if not model_doc:
            model_doc = model_col.find_one({"model_type": "CLIENT_PHASE_0"})
            if model_doc:
                logger.info("Using latest Phase 0 model for initial Phase 1 predictions")
        if not model_doc:
            model_doc = model_col.find_one({"model_type": "GENERAL"})

        if not model_doc:
            raise RuntimeError("No model available for Phase 1")

        model = pickle.loads(model_doc["model"])
        scaler = pickle.loads(model_doc["scaler"])
        model_type = model_doc["model_type"]

        # 5) Fetch ONLY unprocessed customers (next batch)
        projection = {f: 1 for f in FEATURES}
        projection["ID"] = 1
        projection["_id"] = 0

        # First run: you wanted "predict + act on all customers" originally.
        # Doing "all" in one tick is what made it slow.
        # This processes them in batches every 10 seconds until all are completed (same end result).
        cursor = client_data_col.find(
            {"phase1_processed": {"$ne": True}},
            projection
        ).limit(BATCH_SIZE)

        batch = list(cursor)
        if not batch:
            system_col.update_one(
                {"key": "autonomy_phase"},
                {"$set": {
                    "processing": False,
                    "phase1_progress_total": int(total),
                    "phase1_progress_done": int(done),
                    "phase1_complete": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            return

        df = pd.DataFrame(batch)
        if df.empty:
            return

        # 6) Predict
        X = ensure_features(df)
        probs = model.predict(scaler.transform(X))
        probs = np.where(np.isfinite(probs), probs, 0.0)
        preds = (probs >= 0.5).astype(int)

        now = datetime.utcnow()
        ids = df["ID"].astype(str).to_numpy()

        # 7) Build customer dicts for decision logic (same logic)
        customers = []
        for cid, prob, pred in zip(ids, probs, preds):
            customers.append({
                "ID": cid,
                "churn_prob": float(prob),
                "prediction": int(pred),
            })

        demos = [generate_demo_brief(c) for c in customers]

        # 8) Bulk write actions + predictions (FAST)
        action_docs = execute_actions_bulk_docs(customers, demos, now)
        if action_docs:
            actions_col.insert_many(action_docs, ordered=False)

        pred_docs = []
        for cid, prob, pred in zip(ids, probs, preds):
            pred_docs.append({
                "ID": cid,
                "prediction": int(pred),
                "churn_prob": float(prob),
                "model_type": model_type,
                "timestamp": now
            })
        if pred_docs:
            predictions_col.insert_many(pred_docs, ordered=False)

        # 9) Mark processed in one query (FAST)
        client_data_col.update_many(
            {"ID": {"$in": ids.tolist()}},
            {"$set": {"phase1_processed": True, "phase1_processed_at": now}}
        )

        # 10) Update UI progress in system state (your UI already calls /system/state)
        done2 = done + len(ids)
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {
                "processing": False,
                "phase1_progress_total": int(total),
                "phase1_progress_done": int(done2),
                "phase1_last_batch_size": int(len(ids)),
                "phase1_last_batch_model": model_type,
                "phase1_last_run_at": now,
                "phase1_seconds_last_batch": (datetime.utcnow() - started).total_seconds(),
                "updated_at": now
            }},
            upsert=True
        )

        # 11) Retraining trigger (kept: after minimum feedback)
        if feedback_count >= INITIAL_FEEDBACK_THRESHOLD:
            # retrain after feedback is sufficient
            logger.info(f"Retraining Phase 1 model using {feedback_count} feedback entries")
            retrain_phase1_model()
        else:
            logger.info(
                f"Pausing retrain until initial feedback threshold "
                f"({feedback_count}/{INITIAL_FEEDBACK_THRESHOLD}) is reached"
            )

    except Exception as e:
        logger.exception(e)
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"processing": False, "phase1_error": str(e), "updated_at": datetime.utcnow()}},
            upsert=True
        )


# ---------------- PHASE_1 FEEDBACK ITEM ----------------
class Phase1FeedbackItem(BaseModel):
    ID: str
    churn: Optional[int] = None
    executed_action: List[str] = []
    decision_correct: int = 0
    action_appropriate: int = 0
    human_override: int = 0
    phase: str = "PHASE_1"


# ---------------- PHASE_1 FEEDBACK ENDPOINT ----------------
@router.post("/feedback")
def submit_feedback_phase1(items: List[Phase1FeedbackItem], background_tasks: BackgroundTasks):
    retraining_triggered = False
    now = datetime.utcnow()

    # Bulk insert feedback (FAST)
    docs = []
    for item in items:
        d = item.dict()
        d["timestamp"] = now
        d["phase"] = "PHASE_1"
        docs.append(d)

        logger.info(
            f"💬 PHASE_1 Feedback received | Customer={item.ID} | "
            f"Churn={item.churn} | ExecutedActions={item.executed_action} | "
            f"DecisionCorrect={item.decision_correct} | ActionAppropriate={item.action_appropriate} | "
            f"HumanOverride={item.human_override}"
        )

    if docs:
        phase1_feedback_col.insert_many(docs, ordered=False)

    labeled_count = phase1_feedback_col.count_documents({})
    logger.info(f"Phase 1 feedback total: {labeled_count}")

    if labeled_count >= FEEDBACK_THRESHOLD:
        retraining_triggered = True
        background_tasks.add_task(retrain_phase1_model)

        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"awaiting_feedback": False}}
        )
        logger.info("🔁 Phase 1 retraining triggered!")

    return {"feedback_received": True, "retraining_triggered": retraining_triggered}


# ---------------- PHASE_1 RETRAIN FUNCTION ----------------
def retrain_phase1_model():
    feedback = list(phase1_feedback_col.find({}))
    if len(feedback) < 5:
        logger.warning("Not enough Phase 1 feedback to retrain (min 5 required)")
        return

    fb_df = pd.DataFrame(feedback).drop(columns=["_id"], errors="ignore")

    if "churn" not in fb_df.columns:
        logger.error("Feedback missing 'churn' field, cannot retrain")
        return

    fb_df["churn"] = pd.to_numeric(fb_df["churn"], errors="coerce")
    fb_df["churn"] = fb_df["churn"].replace([np.inf, -np.inf], np.nan)
    fb_df = fb_df.dropna(subset=["churn"])
    fb_df["churn"] = fb_df["churn"].astype(int)

    client_data = list(client_data_col.find({}))
    if not client_data:
        logger.error("No client data found; cannot align feedback labels with features")
        return

    client_df = pd.DataFrame(client_data).drop(columns=["_id"], errors="ignore")
    if "ID" not in client_df.columns:
        logger.error("Client data missing 'ID' column; cannot join feedback")
        return

    merged = fb_df.merge(client_df, on="ID", how="inner", suffixes=("_fb", "_client"))
    if merged.empty:
        logger.error("No matching IDs between feedback and client data; cannot retrain")
        return

    if "churn_fb" in merged.columns and "churn" not in merged.columns:
        merged = merged.rename(columns={"churn_fb": "churn"})
    merged = merged.drop(columns=["churn_client"], errors="ignore")

    train_df, val_df = train_test_split(merged, test_size=0.1, random_state=42)

    X_train = ensure_features(train_df)
    y_train = train_df["churn"].astype(int)

    X_val = ensure_features(val_df)
    y_val = val_df["churn"].astype(int)

    phase0_model_doc = model_col.find_one({"model_type": "CLIENT_PHASE_0"})
    if not phase0_model_doc:
        phase0_model_doc = model_col.find_one({"model_type": "GENERAL"})
        logger.warning("CLIENT_PHASE_0 not found, using GENERAL model for Phase 1 retraining")

    base_model = pickle.loads(phase0_model_doc["model"])
    scaler = pickle.loads(phase0_model_doc["scaler"])

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    train_ds = lgb.Dataset(X_train_scaled, label=y_train)
    val_ds = lgb.Dataset(X_val_scaled, label=y_val)

    updated_model = lgb.train(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 64
        },
        train_set=train_ds,
        valid_sets=[val_ds],
        num_boost_round=30,
        init_model=base_model
    )

    val_preds = (updated_model.predict(X_val_scaled) >= 0.5).astype(int)
    acc = float(accuracy_score(y_val, val_preds))
    logger.info(f"Phase 1 validation accuracy: {acc:.2f}")

    existing_phase1_model = model_col.find_one({"model_type": "CLIENT_PHASE_1"})
    if existing_phase1_model:
        model_col.insert_one({
            "model_type": "CLIENT_PHASE_1_BACKUP",
            "model": existing_phase1_model["model"],
            "scaler": existing_phase1_model["scaler"],
            "backed_up_at": datetime.utcnow()
        })

    model_col.update_one(
        {"model_type": "CLIENT_PHASE_1"},
        {"$set": {
            "model": pickle.dumps(updated_model),
            "scaler": pickle.dumps(scaler),
            "accuracy": acc,
            "trained_at": datetime.utcnow()
        }},
        upsert=True
    )

    if "human_override" in train_df.columns:
        human_override_count = pd.to_numeric(train_df["human_override"], errors="coerce").fillna(0).sum()
        total_feedback_count = len(train_df)
        human_override_rate = float(human_override_count) / float(total_feedback_count) if total_feedback_count > 0 else 0.0
    else:
        human_override_rate = 0.0

    metrics_col.insert_one({
        "phase": "PHASE_1",
        "accuracy": acc,
        "rows": int(len(train_df)),
        "trained_at": datetime.utcnow(),
        "human_override_rate": human_override_rate
    })

    if acc >= PHASE_1_ACCURACY_THRESHOLD:
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {
                "phase": "PHASE_2",
                "processing": False,
                "processing_phase2": False,
                "awaiting_feedback": False,
                "updated_at": datetime.utcnow()
            }}
        )
        logger.info(f"✅ Accuracy >= {PHASE_1_ACCURACY_THRESHOLD*100:.0f}% → Phase 2 ready")

    logger.info(f"✅ Phase 1 retraining complete | Accuracy={acc:.2f} | Training rows={len(train_df)}")


# ---------------- GET SYSTEM STATE ----------------
@router.get("/system/state")
def get_system_state():
    state = system_col.find_one({"key": "autonomy_phase"}, {"_id": 0})
    if not state:
        return {"phase": "PHASE_1"}
    return state
