import io
import pickle
from datetime import datetime
from typing import List
from uuid import uuid4
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import lightgbm as lgb
from fastapi import APIRouter, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from sklearn.preprocessing import StandardScaler
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from loguru import logger
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
from typing import List
from fastapi import BackgroundTasks, HTTPException
from db.mongo import (
    db,
    client_data_col,
    model_col,
    predictions_col,
    system_col,
    metrics_col,
    phase1_feedback_col
)
from datetime import datetime
from routers.pre_phase_0 import ensure_features
from routers.phase_0 import PHASE_1_ACCURACY_THRESHOLD, load_best_model  # phase 0 functions
from fastapi_utils.tasks import repeat_every
from sklearn.model_selection import train_test_split
router = APIRouter()

# ---------------- COLLECTIONS ----------------
from db.mongo import (
    db,
    client_data_col,
    model_col,
    predictions_col,
    system_col,
    metrics_col,
    phase1_feedback_col,
    actions_col,   # ✅ add this
)



# ---------------- THRESHOLDS ----------------
EXECUTION_THRESHOLD = 0.65
HUMAN_REVIEW_THRESHOLD = 0.85
FEEDBACK_THRESHOLD = 5  # trigger retraining after this many feedback entries
AUTONOMOUS_LOOP_SECONDS = 10 
PHASE_1_ACCURACY_THRESHOLD = 0.45
# ---------------- ACTIONS ----------------
def calculate_discount(prob: float, ltv: float = 1000):
    if prob >= 0.85: return min(200, 0.2*ltv)
    if prob >= 0.65: return min(100, 0.1*ltv)
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
    risk = demo["risk"]

    # Execution rules
    executed = risk == "LOW"
    human_review_required = risk in ["MODERATE", "HIGH"]

    # Only LOW risk executes automatically
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



def complete_phase0_retrain(accuracy: float):
    """
    Call this after Phase 0 retraining finishes
    """
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

# ---------------- PREDICT + EXECUTE ----------------
@router.post("/predict/phase1")
async def predict_phase_1(file: UploadFile = UploadFile(...)):
    model, scaler, model_type = load_best_model()  # can be PHASE_0 or GENERAL

    df = pd.read_csv(io.BytesIO(await file.read()))
    if "ID" not in df:
        df["ID"] = [str(uuid4()) for _ in range(len(df))]

    X = ensure_features(df)
    probs = model.predict(scaler.transform(X))
    preds = (probs >= 0.5).astype(int)

    df["churn_prob"] = probs
    df["prediction"] = preds

    # execute actions based on demo brief
    for _, r in df.iterrows():
        demo = generate_demo_brief(r.to_dict())
        execute_actions(r.to_dict(), demo)

    # store predictions
    predictions_col.insert_many(
        df[["ID", "prediction", "churn_prob"]]
        .assign(model_type=model_type, timestamp=datetime.utcnow())
        .to_dict("records")
    )

    return JSONResponse({
        "phase": "PHASE_1",
        "model_used": model_type,
        "rows_scored": len(df),
        "message": "Predictions + actions executed"
    })


# ---------------- AUTONOMOUS PHASE 1 LOOP ----------------
@router.on_event("startup")
@repeat_every(seconds=AUTONOMOUS_LOOP_SECONDS, wait_first=True)
def autonomous_loop_phase1():
    """
    Robust Phase 1 autonomous loop:
    1. First run: predict + act on all customers using latest Phase 0 model.
    2. Pause until initial feedback threshold is reached.
    3. Retrain Phase 1 model with feedback.
    4. Predict + act on next batch (e.g., 40 customers).
    """
    INITIAL_FEEDBACK_THRESHOLD = 5
    BATCH_SIZE = 40

    # 1️⃣ Load system state
    state = system_col.find_one({"key": "autonomy_phase"})
    if not state or state.get("phase") != "PHASE_1":
        return

    # 2️⃣ Acquire processing lock
    lock = system_col.find_one_and_update(
        {"key": "autonomy_phase", "processing": False},
        {"$set": {"processing": True}},
        return_document=True
    )
    if not lock:
        return

    try:
        # 3️⃣ Load client customer data (NOT training data)
        df = pd.DataFrame(list(client_data_col.find({})))
        if df.empty:
            logger.info("No customer data found")
            return

        # 4️⃣ Determine unprocessed customers
        processed_ids = [
            r["ID"]
            for r in actions_col.find({"phase": "PHASE_1"})
        ]

        unprocessed_df = df[~df["ID"].isin(processed_ids)]
        if unprocessed_df.empty:
            logger.info("All customers already processed")
            return

        # 5️⃣ Count feedback received
        feedback_count = phase1_feedback_col.count_documents({})

        # 6️⃣ Decide model to use
        model_doc = model_col.find_one({"model_type": "CLIENT_PHASE_1"})
        if not model_doc:
            model_doc = model_col.find_one({"model_type": "CLIENT_PHASE_0"})
            logger.info("Using latest Phase 0 model for initial Phase 1 predictions")

        model = pickle.loads(model_doc["model"])
        scaler = pickle.loads(model_doc["scaler"])

        # 7️⃣ Determine batch (COPY to avoid pandas warnings)
        if feedback_count < INITIAL_FEEDBACK_THRESHOLD:
            batch_df = unprocessed_df.copy()
            logger.info(f"First Phase 1 run: predicting for {len(batch_df)} customers")
        else:
            batch_df = unprocessed_df.head(BATCH_SIZE).copy()
            logger.info(f"Processing next batch of {len(batch_df)} customers")

        # 8️⃣ Predict
        X = ensure_features(batch_df)
        probs = model.predict(scaler.transform(X))

        batch_df.loc[:, "churn_prob"] = probs
        batch_df.loc[:, "prediction"] = (probs >= 0.5).astype(int)

        # 9️⃣ Generate decisions & execute (ONLY HERE)
        for _, r in batch_df.iterrows():
            customer_dict = r.to_dict()
            demo_brief = generate_demo_brief(customer_dict)

            # Single source of truth
            execute_actions(customer_dict, demo_brief)

        # 🔟 Pause until minimum feedback collected
        if feedback_count < INITIAL_FEEDBACK_THRESHOLD:
            logger.info(
                f"Pausing loop until initial feedback threshold "
                f"({feedback_count}/{INITIAL_FEEDBACK_THRESHOLD}) is reached"
            )
            return

        # 1️⃣1️⃣ Retrain Phase 1 model
        logger.info(f"Retraining Phase 1 model using {feedback_count} feedback entries")
        retrain_phase1_model()

    finally:
        # 1️⃣2️⃣ Release processing lock
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"processing": False}}
        )


# ---------------- PHASE_1 FEEDBACK ITEM ----------------
class Phase1FeedbackItem(BaseModel):
    ID: str
    churn: int              # actual outcome → mandatory
    executed_action: list   # what system executed → mandatory
    decision_correct: int = 0  # optional
    action_appropriate: int = 0 # optional
    human_override: int = 0     # optional
    phase: str = "PHASE_1"      # default included


# ---------------- PHASE_1 FEEDBACK ENDPOINT ----------------

@router.post("/feedback/phase1")
def submit_feedback_phase1(
    items: List[Phase1FeedbackItem], background_tasks: BackgroundTasks
):
    """
    Store PHASE_1 human feedback (single or batch) including actual churn and executed actions.
    Triggers retraining once FEEDBACK_THRESHOLD is reached.
    """
    retraining_triggered = False

    for item in items:
        feedback_doc = item.dict()
        feedback_doc["timestamp"] = datetime.utcnow()

        # Ensure phase is always "PHASE_1"
        feedback_doc["phase"] = "PHASE_1"

        # Insert into MongoDB
        phase1_feedback_col.insert_one(feedback_doc)

        logger.info(
            f"💬 PHASE_1 Feedback received | Customer={item.ID} | "
            f"Churn={item.churn} | ExecutedActions={item.executed_action} | "
            f"DecisionCorrect={item.decision_correct} | ActionAppropriate={item.action_appropriate} | "
            f"HumanOverride={item.human_override}"
        )

    # ✅ Count all Phase 1 feedback documents
    labeled_count = phase1_feedback_col.count_documents({})  # count all docs
    logger.info(f"Phase 1 feedback total: {labeled_count}")

    if labeled_count >= FEEDBACK_THRESHOLD:
        retraining_triggered = True
        background_tasks.add_task(retrain_phase1_model)

        # Allow autonomous loop to continue
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"awaiting_feedback": False}}
        )
        logger.info("🔁 Phase 1 retraining triggered!")

    return {"feedback_received": True, "retraining_triggered": retraining_triggered}


# ---------------- PHASE_1 RETRAIN FUNCTION ----------------
def retrain_phase1_model():
    """
    Retrains Phase 1 model using Phase 1 feedback labels + ORIGINAL client features
    so the Phase 0 scaler feature shape matches (no transform crash).
    Saves updated model as CLIENT_PHASE_1 and writes metrics.
    """
    # 1) Load all Phase 1 feedback
    feedback = list(phase1_feedback_col.find({}))
    if len(feedback) < 5:
        logger.warning("Not enough Phase 1 feedback to retrain (min 5 required)")
        return

    fb_df = pd.DataFrame(feedback).drop(columns=["_id"], errors="ignore")

    # 2) Clean churn labels
    if "churn" not in fb_df.columns:
        logger.error("Feedback missing 'churn' field, cannot retrain")
        return

    fb_df["churn"] = pd.to_numeric(fb_df["churn"], errors="coerce")
    fb_df["churn"] = fb_df["churn"].replace([np.inf, -np.inf], np.nan)
    fb_df = fb_df.dropna(subset=["churn"])
    fb_df["churn"] = fb_df["churn"].astype(int)

    # 3) Load original client feature rows and join labels by ID
    client_data = list(client_data_col.find({}))
    if not client_data:
        logger.error("No client data found; cannot align feedback labels with features")
        return

    client_df = pd.DataFrame(client_data).drop(columns=["_id"], errors="ignore")

    if "ID" not in client_df.columns:
        logger.error("Client data missing 'ID' column; cannot join feedback")
        return

    # Keep only feedback IDs that exist in client data
    merged = fb_df.merge(client_df, on="ID", how="inner", suffixes=("_fb", "_client"))
    if merged.empty:
        logger.error("No matching IDs between feedback and client data; cannot retrain")
        return

    # ✅ FIX: feedback churn becomes churn_fb after merge; rename back to churn
    if "churn_fb" in merged.columns and "churn" not in merged.columns:
        merged = merged.rename(columns={"churn_fb": "churn"})

    # ✅ drop client churn label if present
    merged = merged.drop(columns=["churn_client"], errors="ignore")


    # 4) Train/val split (only on merged labeled rows)
    train_df, val_df = train_test_split(merged, test_size=0.1, random_state=42)

    # 5) Build features using the SAME pipeline as inference
    # ensure_features() returns the model feature set used in Phase 0/1 inference
    X_train = ensure_features(train_df)
    y_train = train_df["churn"].astype(int)

    X_val = ensure_features(val_df)
    y_val = val_df["churn"].astype(int)

    # 6) Load latest Phase 0 model + scaler (scaler MUST match ensure_features output)
    phase0_model_doc = model_col.find_one({"model_type": "CLIENT_PHASE_0"})
    if not phase0_model_doc:
        phase0_model_doc = model_col.find_one({"model_type": "GENERAL"})
        logger.warning("CLIENT_PHASE_0 not found, using GENERAL model for Phase 1 retraining")

    base_model = pickle.loads(phase0_model_doc["model"])
    scaler = pickle.loads(phase0_model_doc["scaler"])  # do not refit

    # 7) Scale features (NOW it matches, so no crash)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 8) Fine-tune LightGBM
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

    # 9) Evaluate
    val_preds = (updated_model.predict(X_val_scaled) >= 0.5).astype(int)
    acc = float(accuracy_score(y_val, val_preds))
    logger.info(f"Phase 1 validation accuracy: {acc:.2f}")

    # 10) Backup previous CLIENT_PHASE_1 if exists
    existing_phase1_model = model_col.find_one({"model_type": "CLIENT_PHASE_1"})
    if existing_phase1_model:
        model_col.insert_one({
            "model_type": "CLIENT_PHASE_1_BACKUP",
            "model": existing_phase1_model["model"],
            "scaler": existing_phase1_model["scaler"],
            "backed_up_at": datetime.utcnow()
        })

    # 11) Save updated CLIENT_PHASE_1 model
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

    # 12) Metrics (compute override rate from feedback docs if present)
    # Note: merged contains human_override from feedback if it existed.
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

    # 13) Phase advance to PHASE_2 if threshold met
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
    state = system_col.find_one({"key": "autonomy_phase"})
    if not state:
        return {"phase": "PHASE_1"}
    return state
