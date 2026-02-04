import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from datetime import datetime
from typing import List
from pydantic import BaseModel
from loguru import logger
from fastapi import BackgroundTasks
from fastapi import APIRouter
from fastapi_utils.tasks import repeat_every

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from db.mongo import (
    db,
    client_data_col,
    model_col,
    phase2_actions_col,
    phase2_feedback_col,
    system_col,
    metrics_col,
    predictions_col
)

from routers.pre_phase_0 import ensure_features

router = APIRouter()

# ==============================
# CONFIG
# ==============================
AUTONOMOUS_LOOP_SECONDS = 10
PHASE_2_ACCURACY_THRESHOLD = 0.70

# ==============================
# DISCOUNT LOGIC
# ==============================
def calculate_discount_phase2(prob: float, ltv: float = 1000) -> float:
    if prob >= 0.85:
        return min(150, 0.15 * ltv)
    if prob >= 0.70:
        return min(80, 0.08 * ltv)
    if prob >= 0.50:
        return min(40, 0.04 * ltv)
    return 0.0


# ==============================
# FEEDBACK SCHEMA
# ==============================
class Phase2FeedbackItem(BaseModel):
    ID: str
    churn: int

    risk_level: str
    auto_executed: int

    decision_correct: int
    action_effective: int
    escalation_needed: int = 0
    human_override: int = 0

    phase: str = "PHASE_2"


# ==============================
# DECISION ENGINE
# ==============================
def generate_demo_brief_phase2(customer: dict) -> dict:
    prob = float(customer["churn_prob"])

    if prob < 0.40:
        return {"risk": "NO_RISK", "decision": "NONE", "actions": [], "discount": 0.0}

    if prob < 0.50:
        return {
            "risk": "LOW",
            "decision": "EXECUTE",
            "actions": ["send_email"],
            "discount": calculate_discount_phase2(prob),
        }

    if prob < 0.70:
        return {
            "risk": "MODERATE",
            "decision": "EXECUTE",   # 🚀 autonomous in Phase 2
            "actions": ["send_email", "offer_discount"],
            "discount": calculate_discount_phase2(prob),
        }

    return {
        "risk": "HIGH",
        "decision": "ESCALATE",
        "actions": ["schedule_call", "send_email", "offer_discount"],
        "discount": calculate_discount_phase2(prob),
    }

# ==============================
# DRAFT GENERATORS — PHASE 2
# ==============================

def generate_email_draft_phase2(customer: dict, demo: dict) -> str:
    name = customer.get("name", "Valued Customer")
    prob = customer.get("churn_prob", 0)

    if demo["risk"] == "LOW":
        return (
            f"Hi {name},\n\n"
            "We noticed you’ve been actively engaging with our services. "
            "Just checking in to see if there’s anything we can do better.\n\n"
            "Best regards,\nCustomer Success Team"
        )

    if demo["risk"] == "MODERATE":
        return (
            f"Hi {name},\n\n"
            "We value your partnership and wanted to share a special offer "
            "as a thank-you for being with us.\n\n"
            f"We’ve applied a ${demo['discount']} loyalty credit to your account.\n\n"
            "Let us know if you’d like to explore more options.\n\n"
            "Best,\nCustomer Success Team"
        )

    return (
        f"Hi {name},\n\n"
        "We want to ensure you’re getting the most value from our services. "
        "A member of our team will be reaching out shortly.\n\n"
        "Sincerely,\nCustomer Experience Team"
    )


def generate_call_script_phase2(customer: dict, demo: dict) -> str:
    return (
        f"Customer ID: {customer['ID']}\n"
        f"Risk Level: {demo['risk']}\n"
        f"Churn Probability: {customer['churn_prob']:.2f}\n\n"
        "Call Objective:\n"
        "- Understand dissatisfaction drivers\n"
        "- Reinforce value proposition\n"
        "- Offer tailored incentive if needed\n\n"
        f"Suggested Discount: ${demo['discount']}\n"
    )


def generate_demo_brief_text_phase2(customer: dict, demo: dict) -> str:
    return (
        f"PHASE 2 AUTONOMOUS DECISION\n"
        f"Customer: {customer['ID']}\n"
        f"Risk: {demo['risk']}\n"
        f"Decision: {demo['decision']}\n"
        f"Actions: {', '.join(demo['actions'])}\n"
        f"Discount: ${demo['discount']}\n"
        f"Churn Probability: {customer['churn_prob']:.2f}\n"
    )

# ==============================
# ACTION EXECUTION
# ==============================
def execute_actions_phase2(customer: dict, demo: dict):
    """
    Executes Phase 2 actions and stores them in MongoDB with all relevant artifacts:
    - demo_brief
    - email_draft
    - call_script
    - confidence
    """

    risk = demo.get("risk", "NO_RISK")
    auto_exec = risk in ["LOW", "MODERATE"]

    # Generate artifacts
    try:
        demo_brief = generate_demo_brief_text_phase2(customer, demo)
    except Exception as e:
        demo_brief = f"Error generating demo_brief: {e}"
        logger.error(demo_brief)

    try:
        email_draft = generate_email_draft_phase2(customer, demo)
    except Exception as e:
        email_draft = f"Error generating email_draft: {e}"
        logger.error(email_draft)

    try:
        call_script = generate_call_script_phase2(customer, demo) if risk == "HIGH" else None
    except Exception as e:
        call_script = f"Error generating call_script: {e}"
        logger.error(call_script)

    confidence = round(1 - abs(float(customer.get("churn_prob", 0)) - 0.5), 2)

    # Log before insertion
    logger.info(
        f"Inserting Phase 2 action → Customer: {customer.get('ID')} | "
        f"Risk: {risk} | Decision: {demo.get('decision')} | "
        f"AutoExec: {auto_exec} | Confidence: {confidence}"
    )

    # Insert into MongoDB
    try:
        phase2_actions_col.insert_one({
            "ID": customer.get("ID", "UNKNOWN"),
            "phase": "PHASE_2",

            "risk": risk,
            "decision": demo.get("decision", "NONE"),
            "actions": demo.get("actions", []),
            "discount": float(demo.get("discount", 0.0)),
            "churn_prob": float(customer.get("churn_prob", 0)),

            "auto_executed": auto_exec,
            "human_review_required": not auto_exec,

            # 🔥 Artifacts
            "demo_brief": demo_brief,
            "email_draft": email_draft,
            "call_script": call_script,

            "confidence": confidence,

            "timestamp": datetime.utcnow()
        })
        logger.info(f"✅ Phase 2 action stored for {customer.get('ID')}")
    except Exception as e:
        logger.error(f"❌ Failed to insert Phase 2 action for {customer.get('ID')}: {e}")



# ==============================
# AUTONOMOUS LOOP — PHASE 2
# ==============================
@router.on_event("startup")
@repeat_every(seconds=AUTONOMOUS_LOOP_SECONDS, wait_first=True)
def autonomous_loop_phase2():
    # 1️⃣ Check system state
    state = system_col.find_one({"key": "autonomy_phase"})
    if not state or state.get("phase") != "PHASE_2":
        return

    # 2️⃣ Acquire PHASE-2–specific lock (SAFE)
    lock = system_col.find_one_and_update(
        {
            "key": "autonomy_phase",
            "phase": "PHASE_2",
            "processing_phase2": {"$ne": True}
        },
        {"$set": {"processing_phase2": True}},
    )
    if not lock:
        return

    try:
        # 3️⃣ Load customer data
        df = pd.DataFrame(list(client_data_col.find({})))
        if df.empty:
            logger.info("PHASE 2 | No customer data found")
            return

        # 4️⃣ Find unprocessed customers
        processed_ids = {
            r["ID"]
            for r in phase2_actions_col.find({}, {"ID": 1})
        }

        batch = df[~df["ID"].isin(processed_ids)].head(40).copy()
        if batch.empty:
            logger.info("PHASE 2 | All customers already processed")
            return

        # 5️⃣ Load best available model (Phase 2 → Phase 1 → Phase 0)
        model_doc = (
            model_col.find_one({"model_type": "CLIENT_PHASE_2"})
            or model_col.find_one({"model_type": "CLIENT_PHASE_1"})
            or model_col.find_one({"model_type": "CLIENT_PHASE_0"})
        )
        logger.info(f"PHASE 2 | Using model_type={model_doc.get('model_type')}")

        if not model_doc:
            logger.error("PHASE 2 | No model found")
            return

        model = pickle.loads(model_doc["model"])
        scaler = pickle.loads(model_doc["scaler"])

        # 6️⃣ Predict churn probabilities
        X = ensure_features(batch)
        probs = model.predict(scaler.transform(X))
        batch.loc[:, "churn_prob"] = probs
        # 🔥 Store Phase 2 predictions for audit / dashboard
        predictions_col.insert_many(
            [
                {
                    "ID": row["ID"],
                    "prediction": int(p >= 0.5),
                    "churn_prob": float(p),
                    "phase": "PHASE_2",
                    "model_type": model_doc.get("model_type"),
                    "timestamp": datetime.utcnow(),
                }
                for (_, row), p in zip(batch.iterrows(), probs)
            ]
        )


        # 7️⃣ Generate decisions + execute actions
        for _, row in batch.iterrows():
            customer = row.to_dict()
            demo = generate_demo_brief_phase2(customer)
            execute_actions_phase2(customer, demo)

        logger.info(f"PHASE 2 | Processed {len(batch)} customers")

    finally:
        # 8️⃣ Release PHASE-2 lock
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"processing_phase2": False}}
        )


# ==============================
# RETRAIN PHASE 2 — FULL FIX
# ==============================
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import lightgbm as lgb

def retrain_phase2_model():
    # 1️⃣ Load Phase 2 feedback
    feedback = list(phase2_feedback_col.find({}))
    if len(feedback) < 5:
        logger.warning("Not enough Phase 2 feedback to retrain (min 5 required)")
        return

    fb_df = pd.DataFrame(feedback).drop(columns=["_id"], errors="ignore")
    fb_df["churn"] = pd.to_numeric(fb_df["churn"], errors="coerce")
    fb_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    fb_df = fb_df.dropna(subset=["churn"])
    fb_df["churn"] = fb_df["churn"].astype(int)

    if fb_df.empty:
        logger.error("Phase 2 feedback empty after cleaning")
        return

    # 2️⃣ Merge with anchor client data for stability
    client_data = list(client_data_col.find({}))
    if client_data:
        client_df = pd.DataFrame(client_data).drop(columns=["_id"], errors="ignore")
        if "churn" in client_df.columns:
            client_df["churn"] = pd.to_numeric(client_df["churn"], errors="coerce").dropna().astype(int)
            anchor_size = min(10, len(client_df))  # small number of anchor rows
            client_sample = client_df.sample(n=anchor_size, random_state=42)
            train_df = pd.concat([fb_df, client_sample], ignore_index=True)
        else:
            train_df = fb_df
    else:
        train_df = fb_df

    # 3️⃣ Ensure both classes exist
    if len(train_df["churn"].unique()) < 2:
        logger.warning("Only one class in Phase 2 feedback; adding dummy example for stability")
        opposite_class = 1 if train_df["churn"].iloc[0] == 0 else 0
        train_df = pd.concat([train_df, pd.DataFrame([{"churn": opposite_class}])], ignore_index=True)

    # 4️⃣ Oversample minority class if needed
    counts = train_df["churn"].value_counts()
    if len(counts) > 1 and counts.min() != counts.max():
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        df_majority = train_df[train_df["churn"] == majority_class]
        df_minority = train_df[train_df["churn"] == minority_class]
        df_minority_upsampled = resample(df_minority,
                                         replace=True,
                                         n_samples=len(df_majority),
                                         random_state=42)
        train_df = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)

    # 5️⃣ Split into train/validation (use all data if very small)
    if len(train_df) > 20:
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df["churn"])
        X_train = ensure_features(train_df)
        y_train = train_df["churn"]
        X_val = ensure_features(val_df)
        y_val = val_df["churn"]
    else:
        X_train = ensure_features(train_df)
        y_train = train_df["churn"]
        X_val = X_train
        y_val = y_train

    # 6️⃣ Load Phase 1 model as base
    base_doc = model_col.find_one({"model_type": "CLIENT_PHASE_1"})
    if not base_doc:
        logger.error("CLIENT_PHASE_1 model not found for Phase 2 retraining")
        return

    base_model = pickle.loads(base_doc["model"])
    scaler = pickle.loads(base_doc["scaler"])  # DO NOT refit

    # 7️⃣ Scale features
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 8️⃣ Train/fine-tune LightGBM model
    num_boost_round = max(30, len(train_df)//2)
    train_ds = lgb.Dataset(X_train_s, label=y_train)
    val_ds = lgb.Dataset(X_val_s, label=y_val)

    updated_model = lgb.train(
        params={
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.03,
            "num_leaves": 64,
            "min_data_in_leaf": 1
        },
        train_set=train_ds,
        valid_sets=[val_ds],
        num_boost_round=num_boost_round,
        init_model=base_model
    )

    # 9️⃣ Evaluate validation accuracy
    val_preds = (updated_model.predict(X_val_s) >= 0.5).astype(int)
    acc = float(accuracy_score(y_val, val_preds))
    logger.info(f"Phase 2 validation accuracy: {acc:.2f}")
    logger.info(f"Phase 2 churn distribution in training set:\n{train_df['churn'].value_counts()}")

    # 🔟 Save updated Phase 2 model
    model_col.update_one(
        {"model_type": "CLIENT_PHASE_2"},
        {"$set": {
            "model": pickle.dumps(updated_model),
            "scaler": pickle.dumps(scaler),
            "accuracy": acc,
            "trained_at": datetime.utcnow()
        }},
        upsert=True
    )

    metrics_col.insert_one({
        "phase": "PHASE_2",
        "accuracy": acc,
        "rows": len(train_df),
        "trained_at": datetime.utcnow()
    })

    # 1️⃣1️⃣ Switch to Phase 3 if accuracy threshold is met
    if acc >= PHASE_2_ACCURACY_THRESHOLD:
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"phase": "PHASE_3", "updated_at": datetime.utcnow()}}
        )

    logger.info(f"✅ Phase 2 retraining complete | Accuracy={acc:.2f} | Training rows={len(train_df)}")

# ==============================
# FEEDBACK ENDPOINT
# ==============================

PHASE_2_FEEDBACK_THRESHOLD = 5

@router.post("/feedback/phase2")
def submit_feedback_phase2(
    items: List[Phase2FeedbackItem],
    background_tasks: BackgroundTasks
):
    """
    Store PHASE_2 feedback and trigger retraining when threshold is met.
    """
    retraining_triggered = False

    for item in items:
        doc = item.dict()
        doc["timestamp"] = datetime.utcnow()
        doc["phase"] = "PHASE_2"

        db["phase2_feedback"].insert_one(doc)

        logger.info(
            f"💬 PHASE_2 Feedback | "
            f"Customer={item.ID} | "
            f"Churn={item.churn} | "
            f"Risk={item.risk_level} | "
            f"AutoExec={item.auto_executed} | "
            f"DecisionCorrect={item.decision_correct} | "
            f"ActionEffective={item.action_effective} | "
            f"EscalationNeeded={item.escalation_needed}"
        )

    total = db["phase2_feedback"].count_documents({})
    logger.info(f"Phase 2 feedback total: {total}")

    if total >= PHASE_2_FEEDBACK_THRESHOLD:
        retraining_triggered = True
        background_tasks.add_task(retrain_phase2_model)

        # allow loop to continue
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"awaiting_feedback": False}}
        )

        logger.info("🔁 Phase 2 retraining triggered")

    return {
        "feedback_received": True,
        "retraining_triggered": retraining_triggered
    }
