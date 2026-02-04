from fastapi import APIRouter, Body
from datetime import datetime
from db.mongo import system_col

router = APIRouter()

@router.post("/dashboard/system/set_phase")
def set_system_phase(payload: dict = Body(...)):
    phase = payload.get("phase")
    if phase not in ["PHASE_1", "PHASE_2"]:
        return {"status": "error", "message": "Invalid phase"}

    system_col.update_one(
        {"key": "autonomy_phase"},
        {"$set": {"phase": phase, "updated_at": datetime.utcnow()}},
        upsert=True
    )
    return {"status": "ok", "phase": phase}
