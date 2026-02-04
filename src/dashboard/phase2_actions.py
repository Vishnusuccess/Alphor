from fastapi import APIRouter, Body
from db.mongo import phase2_actions_col, metrics_col
from datetime import datetime
router = APIRouter()

# GET all Phase 2 actions for Executive & Support dashboards
@router.get("/phase2/actions")
def get_phase2_actions():
    return list(phase2_actions_col.find({}, {"_id": 0}))

# GET Phase 2 metrics
@router.get("/phase2/metrics")
def get_phase2_metrics():
    """
    Returns full model accuracy history across all phases
    """
    metrics = list(
        metrics_col.find(
            {"phase": {"$in": ["PHASE_0", "PHASE_1", "PHASE_2"]}},
            {"_id": 0}
        ).sort("trained_at", 1)
    )

    # Safety: ensure trained_at exists
    for m in metrics:
        if "trained_at" not in m:
            m["trained_at"] = datetime.utcnow()

    return metrics

# POST notes from support team sidebar
@router.post("/phase2/notes")
def save_notes(payload: dict = Body(...)):
    customer_id = payload.get("customer_id")
    notes = payload.get("notes", "")
    if not customer_id:
        return {"status": "error", "message": "customer_id required"}

    phase2_actions_col.update_one(
        {"customer_id": customer_id},
        {"$set": {"agent_notes": notes}}
    )
    return {"status": "ok"}

