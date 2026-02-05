# phase2.py  (UPDATED - SAME LOGIC, FIXED SLA + ONE DOC PER CUSTOMER)

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

from datetime import datetime
from typing import List
from pydantic import BaseModel
from loguru import logger
from fastapi import BackgroundTasks, APIRouter
from fastapi_utils.tasks import repeat_every

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

from pymongo import UpdateOne  # ✅ NEW

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

# Performance tuning
BATCH_SIZE = 2000  # customers per tick (10s). Tune 500..10000 depending on Mongo/server.
FEATURES = [
    "period", "age", "gender", "seniority_insured", "seniority_policy",
    "exposure_time", "type_policy", "type_product", "reimbursement",
    "new_business", "distribution_channel", "premium", "cost_claims_year",
    "n_medical_services", "n_insured_pc", "n_insured_mun", "n_insured_prov",
]

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
            "decision": "EXECUTE",   # autonomous in Phase 2
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
# ACTION DOC BUILDER (same content, unchanged)
# ==============================
def build_phase2_action_docs(customers: List[dict], demos: List[dict], ts: datetime) -> List[dict]:
    docs = []
    for customer, demo in zip(customers, demos):
        risk = demo.get("risk", "NO_RISK")
        auto_exec = risk in ["LOW", "MODERATE"]

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

        docs.append({
            "ID": str(customer.get("ID", "UNKNOWN")),
            "phase": "PHASE_2",

            "risk": risk,
            "decision": demo.get("decision", "NONE"),
            "actions": demo.get("actions", []),
            "discount": float(demo.get("discount", 0.0)),
            "churn_prob": float(customer.get("churn_prob", 0.0)),

            "auto_executed": auto_exec,
            "human_review_required": not auto_exec,

            "demo_brief": demo_brief,
            "email_draft": email_draft,
            "call_script": call_script,

            "confidence": float(confidence),

            # ✅ timestamps
            "timestamp": ts,

            # NOTE: we will set sla_started_at properly in the loop (transition-aware)
            "sla_started_at": ts if risk == "HIGH" else None,
        })
    return docs


# ==============================
# AUTONOMOUS LOOP — PHASE 2 (FAST)
# ==============================
@router.on_event("startup")
@repeat_every(seconds=AUTONOMOUS_LOOP_SECONDS, wait_first=True)
def autonomous_loop_phase2():
    # 1) Check system state
    state = system_col.find_one({"key": "autonomy_phase"})
    if not state or state.get("phase") != "PHASE_2":
        return

    # 2) Acquire PHASE-2 lock
    lock = system_col.find_one_and_update(
        {
            "key": "autonomy_phase",
            "phase": "PHASE_2",
            "processing_phase2": {"$ne": True}
        },
        {"$set": {"processing_phase2": True, "phase2_last_tick_at": datetime.utcnow()}},
    )
    if not lock:
        return

    started = datetime.utcnow()
    try:
        # Progress counts for UI (fast)
        total = client_data_col.count_documents({})
        done = client_data_col.count_documents({"phase2_processed": True})

        # If nothing
        if total == 0:
            system_col.update_one(
                {"key": "autonomy_phase"},
                {"$set": {
                    "processing_phase2": False,
                    "phase2_progress_total": 0,
                    "phase2_progress_done": 0,
                    "phase2_complete": True,
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
                    "processing_phase2": False,
                    "phase2_progress_total": int(total),
                    "phase2_progress_done": int(done),
                    "phase2_complete": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            return

        # 3) Fetch ONLY unprocessed customers (next batch)
        projection = {f: 1 for f in FEATURES}
        projection["ID"] = 1
        projection["name"] = 1
        projection["_id"] = 0

        batch = list(
            client_data_col.find({"phase2_processed": {"$ne": True}}, projection).limit(BATCH_SIZE)
        )
        if not batch:
            system_col.update_one(
                {"key": "autonomy_phase"},
                {"$set": {
                    "processing_phase2": False,
                    "phase2_progress_total": int(total),
                    "phase2_progress_done": int(done),
                    "phase2_complete": True,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
            return

        df = pd.DataFrame(batch)
        if df.empty:
            return

        # 4) Load best available model (Phase 2 → Phase 1 → Phase 0)
        model_doc = (
            model_col.find_one({"model_type": "CLIENT_PHASE_2"})
            or model_col.find_one({"model_type": "CLIENT_PHASE_1"})
            or model_col.find_one({"model_type": "CLIENT_PHASE_0"})
        )
        if not model_doc:
            logger.error("PHASE 2 | No model found")
            return

        model_type = model_doc.get("model_type")
        logger.info(f"PHASE 2 | Using model_type={model_type}")

        model = pickle.loads(model_doc["model"])
        scaler = pickle.loads(model_doc["scaler"])

        # 5) Predict churn probabilities
        X = ensure_features(df)
        probs = model.predict(scaler.transform(X))
        probs = np.where(np.isfinite(probs), probs, 0.0)
        preds = (probs >= 0.5).astype(int)

        now = datetime.utcnow()
        ids = df["ID"].astype(str).to_numpy()
        ids_list = ids.tolist()

        # 6) Bulk store predictions (FAST)
        pred_docs = []
        for cid, p, pr in zip(ids, probs, preds):
            pred_docs.append({
                "ID": str(cid),
                "prediction": int(pr),
                "churn_prob": float(p),
                "phase": "PHASE_2",
                "model_type": model_type,
                "timestamp": now,
            })
        if pred_docs:
            predictions_col.insert_many(pred_docs, ordered=False)

        # 7) Decisions + actions (build)
        customers = []
        for cid, p in zip(ids, probs):
            if "name" in df.columns:
                # safer: compare as str to str
                name_val = df.loc[df["ID"].astype(str) == str(cid), "name"].iloc[0]
            else:
                name_val = None

            customers.append({
                "ID": str(cid),
                "churn_prob": float(p),
                "name": name_val
            })

        demos = [generate_demo_brief_phase2(c) for c in customers]
        action_docs = build_phase2_action_docs(customers, demos, now)

        # ✅ NEW: store sla_started_at at prediction time (HIGH transition), UPSERT one doc per customer
        existing = list(
            phase2_actions_col.find(
                {"phase": "PHASE_2", "ID": {"$in": ids_list}},
                {"_id": 0, "ID": 1, "risk": 1, "sla_started_at": 1},
            )
        )
        existing_map = {str(d["ID"]): d for d in existing}

        ops = []
        for doc in action_docs:
            cid = str(doc["ID"])
            prev = existing_map.get(cid)

            new_risk = doc.get("risk", "NO_RISK")
            prev_risk = prev.get("risk") if prev else None
            prev_sla = prev.get("sla_started_at") if prev else None

            # ✅ SLA starts only when it becomes HIGH (or if missing)
            if new_risk == "HIGH":
                if (prev_risk != "HIGH") or (prev_sla is None):
                    doc["sla_started_at"] = now  # stored at prediction time
                else:
                    doc["sla_started_at"] = prev_sla  # preserve original start
            else:
                doc["sla_started_at"] = None

            ops.append(
                UpdateOne(
                    {"phase": "PHASE_2", "ID": cid},
                    {"$set": doc},
                    upsert=True
                )
            )

        if ops:
            phase2_actions_col.bulk_write(ops, ordered=False)

        # 8) Mark processed in ONE query (FAST)
        client_data_col.update_many(
            {"ID": {"$in": ids_list}},
            {"$set": {"phase2_processed": True, "phase2_processed_at": now}}
        )

        # 9) Update UI progress
        done2 = done + len(ids_list)
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {
                "processing_phase2": False,
                "phase2_progress_total": int(total),
                "phase2_progress_done": int(done2),
                "phase2_last_batch_size": int(len(ids_list)),
                "phase2_last_batch_model": model_type,
                "phase2_last_run_at": now,
                "phase2_seconds_last_batch": (datetime.utcnow() - started).total_seconds(),
                "updated_at": now
            }},
            upsert=True
        )

        logger.info(f"PHASE 2 | Processed {len(df)} customers")

    except Exception as e:
        logger.exception(e)
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"processing_phase2": False, "phase2_error": str(e), "updated_at": datetime.utcnow()}},
            upsert=True
        )

    finally:
        # Ensure lock released
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"processing_phase2": False}}
        )


# ==============================
# RETRAIN PHASE 2 — SAME LOGIC (unchanged)
# ==============================
def retrain_phase2_model():
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

    client_data = list(client_data_col.find({}))
    if client_data:
        client_df = pd.DataFrame(client_data).drop(columns=["_id"], errors="ignore")
        if "churn" in client_df.columns:
            client_df["churn"] = pd.to_numeric(client_df["churn"], errors="coerce")
            client_df = client_df.dropna(subset=["churn"])
            client_df["churn"] = client_df["churn"].astype(int)

            anchor_size = min(10, len(client_df))
            if anchor_size > 0:
                client_sample = client_df.sample(n=anchor_size, random_state=42)
                train_df = pd.concat([fb_df, client_sample], ignore_index=True)
            else:
                train_df = fb_df
        else:
            train_df = fb_df
    else:
        train_df = fb_df

    if len(train_df["churn"].unique()) < 2:
        logger.warning("Only one class in Phase 2 feedback; adding dummy example for stability")
        opposite_class = 1 if train_df["churn"].iloc[0] == 0 else 0
        train_df = pd.concat([train_df, pd.DataFrame([{"churn": opposite_class}])], ignore_index=True)

    counts = train_df["churn"].value_counts()
    if len(counts) > 1 and counts.min() != counts.max():
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        df_majority = train_df[train_df["churn"] == majority_class]
        df_minority = train_df[train_df["churn"] == minority_class]
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        train_df = pd.concat([df_majority, df_minority_upsampled], ignore_index=True)

    if len(train_df) > 20 and len(train_df["churn"].unique()) > 1:
        train_df, val_df = train_test_split(
            train_df, test_size=0.1, random_state=42, stratify=train_df["churn"]
        )
        X_train = ensure_features(train_df)
        y_train = train_df["churn"]
        X_val = ensure_features(val_df)
        y_val = val_df["churn"]
    else:
        X_train = ensure_features(train_df)
        y_train = train_df["churn"]
        X_val = X_train
        y_val = y_train

    base_doc = model_col.find_one({"model_type": "CLIENT_PHASE_1"})
    if not base_doc:
        logger.error("CLIENT_PHASE_1 model not found for Phase 2 retraining")
        return

    base_model = pickle.loads(base_doc["model"])
    scaler = pickle.loads(base_doc["scaler"])  # DO NOT refit

    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)

    num_boost_round = max(30, len(train_df) // 2)
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

    val_preds = (updated_model.predict(X_val_s) >= 0.5).astype(int)
    acc = float(accuracy_score(y_val, val_preds))
    logger.info(f"Phase 2 validation accuracy: {acc:.2f}")
    logger.info(f"Phase 2 churn distribution in training set:\n{train_df['churn'].value_counts()}")

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
        "rows": int(len(train_df)),
        "trained_at": datetime.utcnow()
    })

    if acc >= PHASE_2_ACCURACY_THRESHOLD:
        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"phase": "PHASE_3", "updated_at": datetime.utcnow()}}
        )

    logger.info(f"✅ Phase 2 retraining complete | Accuracy={acc:.2f} | Training rows={len(train_df)}")


# ==============================
# FEEDBACK ENDPOINT (unchanged)
# ==============================
PHASE_2_FEEDBACK_THRESHOLD = 5

@router.post("/feedback/phase2")
def submit_feedback_phase2(items: List[Phase2FeedbackItem], background_tasks: BackgroundTasks):
    retraining_triggered = False
    now = datetime.utcnow()

    docs = []
    for item in items:
        doc = item.dict()
        doc["timestamp"] = now
        doc["phase"] = "PHASE_2"
        docs.append(doc)

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

    if docs:
        db["phase2_feedback"].insert_many(docs, ordered=False)

    total = db["phase2_feedback"].count_documents({})
    logger.info(f"Phase 2 feedback total: {total}")

    if total >= PHASE_2_FEEDBACK_THRESHOLD:
        retraining_triggered = True
        background_tasks.add_task(retrain_phase2_model)

        system_col.update_one(
            {"key": "autonomy_phase"},
            {"$set": {"awaiting_feedback": False}}
        )
        logger.info("🔁 Phase 2 retraining triggered")

    return {"feedback_received": True, "retraining_triggered": retraining_triggered}


# ==============================
# OPTIONAL: system state (unchanged)
# ==============================
@router.get("/system/state")
def get_system_state():
    state = system_col.find_one({"key": "autonomy_phase"}, {"_id": 0})
    if not state:
        return {"phase": "PHASE_2"}
    return state


@router.delete("/system/reset_database")
def reset_database():
    try:
        collections_to_drop = [
            "ChurnModel",
            "ModelMetrics",
            "Decisions",
            "Actions",
            "Outcomes",
            "PolicyStatus",
            "Payments",
            "Feedback",
            "SystemState",
            "Phase1Feedback",
            "feedback",
            "models",
            "predictions",
            "system_state",
            "client_data",
            "metrics",
            "phase1_actions",
            "phase1_feedback",
            "phase2_actions",
            "phase2_feedback",
            "phase2_predictions",
        ]

        for col_name in collections_to_drop:
            db.drop_collection(col_name)
            logger.info(f"✅ Dropped collection: {col_name}")

        return {"status": "success", "message": "Database reset complete."}

    except Exception as e:
        logger.error(f"❌ Failed to reset database: {e}")
        raise
