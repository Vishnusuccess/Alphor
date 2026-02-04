from fastapi import APIRouter
from db.mongo import phase2_feedback_col, model_col

router = APIRouter()

@router.get("/dashboard/model/retrain_status")
def retrain_status():
    pending = phase2_feedback_col.count_documents({"processed": {"$ne": True}})
    last_model = model_col.find_one(sort=[("trained_at", -1)])

    return {
        "pending_feedback": pending,
        "last_trained_at": last_model.get("trained_at") if last_model else None,
        "retrain_required": pending >= 20
    }
