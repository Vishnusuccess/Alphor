# dashboard/dashboard.py

from fastapi import APIRouter, Body, logger
from datetime import datetime
from fastapi import APIRouter, Body, logger, Query
from bson import ObjectId
import math
from db.mongo import (
    db,
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

# =========================================================
# PHASE 0 CUSTOMERS SUMMARY
# =========================================================

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
# PHASE 0 CUSTOMERS (PAGINATED FOR INFINITE SCROLL)
# =========================================================
@router.get("/phase0/customers")
def get_phase0_customers_paginated(
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
):
    clients = {
        str(c.get("ID", "")).replace('"', '').strip(): c
        for c in client_data_col.find({})
    }

    pipeline = [
        # newest predictions first
        {"$sort": {"timestamp": -1}},
        # ensure a stable ID field to group on
        {"$addFields": {"_cid": {"$ifNull": ["$ID", {"$toString": "$_id"}]}}},
        # one doc per customer: take latest prediction
        {"$group": {"_id": "$_cid", "doc": {"$first": "$$ROOT"}}},
        # keep sorting stable after grouping (optional but helpful)
        {"$sort": {"doc.timestamp": -1}},
        # paginate unique customers
        {"$skip": skip},
        {"$limit": limit},
    ]

    grouped = list(predictions_col.aggregate(pipeline))

    customers = []
    for g in grouped:
        p = g["doc"]
        pred_id = str(p.get("ID") or p.get("_id") or "").replace('"', '').strip()
        if not pred_id:
            continue

        client = clients.get(pred_id)
        base = serialize(client) if client else {}

        prob = safe_float(p.get("churn_prob"), 0.0)
        base.update({
            "churn_prob": prob,
            "prediction": int(p.get("prediction", 0)),
            "risk_level": risk_label(prob),
            "policy_tenure": safe_float(client.get("seniority_policy") if client else None, None),
            "missed_payments": safe_float(client.get("n_medical_services") if client else None, None),
        })
        base["ID"] = pred_id
        customers.append(base)

    return customers


# =========================================================
# PHASE 0 SUMMARY (KPIs показать ALL customers - not paginated)
# =========================================================
@router.get("/phase0/summary")
def get_phase0_summary():
    state = system_col.find_one({"key": "autonomy_phase"}) or {}
    phase = state.get("phase", "PHASE_0")
    phase_1_ready = (phase == "PHASE_1")

    pipeline = [
        {"$sort": {"timestamp": -1}},
        {"$addFields": {"_cid": {"$ifNull": ["$ID", {"$toString": "$_id"}]}}},
        {"$group": {"_id": "$_cid", "doc": {"$first": "$$ROOT"}}},
        {"$project": {"churn_prob": {"$ifNull": ["$doc.churn_prob", 0]}}},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "high": {"$sum": {"$cond": [{"$gte": ["$churn_prob", 0.6]}, 1, 0]}},
            "medium": {"$sum": {"$cond": [{"$and": [{"$gte": ["$churn_prob", 0.3]}, {"$lt": ["$churn_prob", 0.6]}]}, 1, 0]}},
            "low": {"$sum": {"$cond": [{"$lt": ["$churn_prob", 0.3]}, 1, 0]}},
        }}
    ]

    out = list(predictions_col.aggregate(pipeline))
    if not out:
        return {"total": 0, "high": 0, "medium": 0, "low": 0, "phase_1_ready": phase_1_ready}

    s = out[0]
    return {
        "total": int(s.get("total", 0)),
        "high": int(s.get("high", 0)),
        "medium": int(s.get("medium", 0)),
        "low": int(s.get("low", 0)),
        "phase_1_ready": phase_1_ready,
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


@router.delete("/system/reset_database")
def reset_database():
    try:
        # List all collections you want to drop
        collections_to_drop = [
            "ChurnModel",
            "ModelMetrics",
            "Decisions",
            "Actions",
            "Outcomes",
            "PolicyStatus",
            "Payments",
            "Feedback",
            "SystemState"
            "Phase1Feedback",
            "SystemState",
            "feedback",
            "models",
            "predictions",
            "system_state",
            "client_data",
            "feedback",
            "metrics",
            "models",
            "phase1_actions"
            "SystemState",
            "phase1_feedback",
            "phase2_actions",
            "phase2_feedback",
            "phase2_predictions",
            "phase1_actions"
        ]
        
        for col_name in collections_to_drop:
            db.drop_collection(col_name)
            logger.info(f"✅ Dropped collection: {col_name}")
        
        return {"status": "success", "message": "Database reset complete."}
    
    except Exception as e:
        logger.error(f"❌ Failed to reset database: {e}")

# =========================================================
# PHASE 1 ACTIONS (PAGINATED FOR INFINITE SCROLL)
# =========================================================
@router.get("/phase1/actions")
def get_phase1_actions_paginated(
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
):
    """
    Paginated Phase 1 actions feed for infinite scroll.
    Frontend calls: /dashboard/phase1/actions?skip=0&limit=200
    """
    actions = list(
        actions_col.find({"phase": "PHASE_1"}).sort("timestamp", -1).skip(skip).limit(limit)
    )

    result = []
    for a in actions:
        doc = serialize(a)

        # Canonical ID
        doc["ID"] = str(doc.get("ID") or doc.get("customer_id") or doc.get("_id") or "-")
        doc["customer_id"] = doc["ID"]

        # Canonical fields
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

    return result  # ARRAY


# =========================================================
# PHASE 1 SUMMARY (KPIs FOR ALL ACTIONS - NOT PAGINATED)
# =========================================================
@router.get("/phase1/summary")
def get_phase1_summary():
    """
    KPI totals for Phase 1 based on ALL PHASE_1 actions in DB.
    This keeps KPIs correct even while table is paginated.
    """
    pipeline = [
        {"$match": {"phase": "PHASE_1"}},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "human_review": {"$sum": {"$cond": ["$human_review_required", 1, 0]}},
            "total_discount": {"$sum": {"$ifNull": ["$discount", 0]}},
            "auto_emails": {"$sum": {"$cond": [{"$in": ["send_email", {"$ifNull": ["$actions", []]}]}, 1, 0]}},
            "discounts_offered": {"$sum": {"$cond": [{"$in": ["offer_discount", {"$ifNull": ["$actions", []]}]}, 1, 0]}},
        }}
    ]

    out = list(actions_col.aggregate(pipeline))
    if not out:
        return {
            "total": 0,
            "human_review": 0,
            "total_discount": 0.0,
            "auto_emails": 0,
            "discounts_offered": 0,
            "feedback_total": int(phase1_feedback_col.count_documents({})),
        }

    s = out[0]
    return {
        "total": int(s.get("total", 0)),
        "human_review": int(s.get("human_review", 0)),
        "total_discount": float(s.get("total_discount", 0.0)),
        "auto_emails": int(s.get("auto_emails", 0)),
        "discounts_offered": int(s.get("discounts_offered", 0)),
        "feedback_total": int(phase1_feedback_col.count_documents({})),
    }
