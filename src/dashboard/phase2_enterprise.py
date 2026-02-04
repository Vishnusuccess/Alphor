from fastapi import APIRouter, Body
from datetime import datetime, timedelta, timezone
from db.mongo import phase2_actions_col

router = APIRouter()

SLA_HOURS = 2


def _parse_dt(value):
    """Accept datetime or ISO string, return datetime (UTC-naive) or None."""
    if value is None:
        return None

    if isinstance(value, datetime):
        # Convert aware -> naive UTC, keep naive as-is
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    if isinstance(value, str):
        try:
            # handle Z suffix
            s = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    return None


@router.get("/dashboard/phase2/support")
def get_support_view():
    now = datetime.utcnow()
    rows = []

    # keep excluding _id if you want, but we will guarantee customer_id ourselves
    for a in phase2_actions_col.find({}, {"_id": 0}):
        # ✅ Guarantee a stable identifier for UI
        a["customer_id"] = (
            a.get("customer_id")
            or a.get("ID")
            or a.get("customerId")
            or a.get("client_id")
            or "-"
        )

        created = _parse_dt(a.get("timestamp") or a.get("created_at") or a.get("created"))
        deadline = created + timedelta(hours=SLA_HOURS) if created else None

        if deadline:
            remaining = int((deadline - now).total_seconds())
            a["sla_remaining_sec"] = max(remaining, 0)
            a["sla_breached"] = remaining <= 0
        else:
            a["sla_remaining_sec"] = None
            a["sla_breached"] = None

        rows.append(a)

    return rows


@router.post("/dashboard/phase2/assign_agent")
def assign_agent(payload: dict = Body(...)):
    customer_id = payload.get("customer_id")
    agent = payload.get("agent")

    if not customer_id or not agent:
        return {"status": "error", "message": "customer_id and agent are required"}

    phase2_actions_col.update_one(
        {"customer_id": customer_id},
        {"$set": {
            "assigned_agent": agent,
            "assigned_at": datetime.utcnow()
        }},
        upsert=False
    )
    return {"status": "assigned"}
