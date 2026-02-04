# dashboard/dashboard.py

from fastapi import APIRouter, Body, logger
from datetime import datetime
from bson import ObjectId
import math
from db.mongo import (
    client_data_col,
    predictions_col,
    feedback_col,
    metrics_col,
    system_col,
    actions_col,
    phase1_feedback_col
)

router = APIRouter()

# ---------------- UTILS ----------------
def safe_float(x, default=0.0):
    """Convert value to float safely; replace NaN/Inf with default"""
    try:
        x = float(x)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default

def serialize(doc: dict) -> dict:
    """
    Convert MongoDB ObjectId and datetime to string for JSON serialization.
    Cleans string fields by stripping quotes and whitespace.
    Does NOT modify 'ID' field.
    """
    if not doc:
        return {}

    out = {}
    for k, v in doc.items():
        if k == "ID":
            out[k] = v  # preserve ID, we will force it later
        elif isinstance(v, ObjectId):
            out[k] = str(v)
        elif isinstance(v, datetime):
            out[k] = v.isoformat()
        elif isinstance(v, str):
            out[k] = v.replace('"', '').strip()
        else:
            out[k] = v
    return out

def risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "HIGH"
    elif prob >= 0.4:
        return "MODERATE"
    elif prob >= 0.2:
        return "LOW"
    return "NO_RISK"


@router.get("/summary")
def dashboard_summary():
    """
    Returns all customers with predictions for the dashboard.
    Merges predictions with client data, ensuring clean IDs.
    """
    # Current system phase
    state = system_col.find_one({"key": "autonomy_phase"}) or {}
    phase = state.get("phase", "PHASE_0")
    phase_1_ready = phase == "PHASE_1"

    # Build clients dict with cleaned keys
    clients = {
        str(c.get("ID", "")).replace('"', '').strip(): c
        for c in client_data_col.find({})
    }

    customers = []

    for p in predictions_col.find({}).sort("timestamp", -1):
        # Get prediction ID safely
        pred_id = str(p.get("ID") or p.get("_id") or "").replace('"', '').strip()
        if not pred_id:
            continue  # skip invalid predictions

        # Get client safely
        client = clients.get(pred_id)

        # Serialize client data
        base = serialize(client) if client else {}

        # Merge prediction info
        prob = safe_float(p.get("churn_prob"), 0.0)
        base.update({
            "churn_prob": prob,
            "prediction": int(p.get("prediction", 0)),
            "risk_level": risk_label(prob),
            "policy_tenure": safe_float(client.get("seniority_policy") if client else None, None),
            "missed_payments": safe_float(client.get("n_medical_services") if client else None, None),
        })

        # Force correct top-level ID (always last)
        base["ID"] = pred_id

        customers.append(base)

    return {
        "phase": phase,
        "phase_1_ready": phase_1_ready,
        "customers": customers
    }


# =========================================================
# METRICS (MODEL PERFORMANCE)
# =========================================================
@router.get("/metrics")
def dashboard_metrics():
    """
    Returns training metrics (accuracy, timestamp) for dashboard charts.
    """
    metrics = list(metrics_col.find({}).sort("trained_at", 1))
    return [serialize(m) for m in metrics]

# =========================================================
# FEEDBACK SUBMISSION
# =========================================================
@router.post("/feedback")
def submit_feedback(items: list[dict] = Body(...)):
    """
    Submit feedback for customer churn.
    Each item should have: { ID, actual_churn }
    """
    if not items:
        return {"feedback_received": 0, "status": "No feedback submitted"}

    feedback_docs = [
        {
            "ID": item["ID"],
            "actual_churn": int(item["actual_churn"]),
            "timestamp": datetime.utcnow()
        }
        for item in items
    ]
    feedback_col.insert_many(feedback_docs)
    return {"feedback_received": len(feedback_docs), "status": "OK"}

# =========================================================
# PHASE 1 ACTIONS (READ-ONLY FOR UI)
# =========================================================
@router.get("/actions")
def get_phase1_actions():
    actions = list(actions_col.find({}).sort("timestamp", -1))
    result = []

    for a in actions:
        doc = serialize(a)

        # 🔑 CANONICAL ID (single source of truth)
        doc["ID"] = str(
            doc.get("ID")
            or doc.get("customer_id")
            or doc.get("_id")
            or "-"
        )

        # 🔑 CANONICAL FIELDS
        doc["risk"] = doc.get("risk") or risk_label(safe_float(doc.get("churn_prob"), 0.0))
        doc["churn_prob"] = safe_float(doc.get("churn_prob"), 0.0)
        doc["decision"] = doc.get("decision", "NONE")
        doc["actions"] = doc.get("actions", [])
        doc["discount"] = safe_float(doc.get("discount"), 0.0)
        doc["human_review_required"] = bool(doc.get("human_review_required", False))

        # UI details
        doc["brief"] = doc.get("brief", "-")
        doc["email_draft"] = doc.get("email_draft", "")
        doc["call_script"] = doc.get("call_script", "")
        doc["confidence"] = safe_float(doc.get("confidence"), None)

        result.append(doc)

    return result


# =========================================================
# PHASE 2 SYSTEM CONTROL
# =========================================================
@router.get("/system_phase")
def get_system_phase():
    """
    Returns the current system phase and whether Phase 2 is awaiting feedback.
    """
    state = system_col.find_one({"key": "autonomy_phase"})
    if not state:
        return {"phase": "PHASE_1", "awaiting_feedback": True}
    return {
        "phase": state.get("phase", "PHASE_1"),
        "awaiting_feedback": state.get("awaiting_feedback", True)
    }

@router.post("/system_phase")
def set_system_phase(payload: dict = Body(...)):
    """
    Allows client to manually activate Phase 2 or Phase 3 via UI button.
    Expected payload: { "phase": "PHASE_2" or "PHASE_3" }
    """
    phase = payload.get("phase")
    if phase not in ["PHASE_2", "PHASE_3"]:
        return {"status": "error", "message": "Invalid phase"}
    system_col.update_one(
        {"key": "autonomy_phase"},
        {"$set": {"phase": phase, "processing_phase2": False, "updated_at": datetime.utcnow()}},
        upsert=True
    )
    return {"status": "ok"}
