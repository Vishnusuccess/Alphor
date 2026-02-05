from fastapi import APIRouter, Body, HTTPException, Query
from datetime import datetime, timedelta, timezone
from db.mongo import phase2_actions_col

router = APIRouter()

SLA_HOURS = 2


def _parse_dt(value):
    """Accept datetime or ISO string, return datetime (UTC-naive) or None."""
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    if isinstance(value, str):
        try:
            s = value.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is not None:
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except Exception:
            return None

    return None


def _iso(dt):
    try:
        return dt.isoformat() if hasattr(dt, "isoformat") else dt
    except Exception:
        return dt


def _canon_id(doc: dict) -> str:
    return str(
        doc.get("ID")
        or doc.get("customer_id")
        or doc.get("customerId")
        or doc.get("client_id")
        or "-"
    )


def _sla_fields(doc: dict, now: datetime) -> dict:
    """
    SLA should start when a case becomes HIGH (escalated), not from an old timestamp.
    We use sla_started_at (preferred). If missing for HIGH, we backfill once.
    """

    # ✅ Prefer SLA start time if present
    created = _parse_dt(
        doc.get("sla_started_at")
        or doc.get("timestamp")
        or doc.get("created_at")
        or doc.get("created")
    )

    # ✅ Backfill sla_started_at for existing HIGH docs that don't have it
    # This prevents instant 0m due to old timestamps.
    if doc.get("risk") == "HIGH" and not doc.get("sla_started_at"):
        # choose best fallback: timestamp if valid else "now"
        fallback = _parse_dt(doc.get("timestamp")) or now



        # write it back (best-effort; ignore any errors)
        try:
            phase2_actions_col.update_one(
                {"ID": str(doc.get("ID"))},
                {"$set": {"sla_started_at": fallback}},
                upsert=False
            )
        except Exception:
            pass

        created = fallback

    deadline = created + timedelta(hours=SLA_HOURS) if created else None

    if deadline:
        remaining = int((deadline - now).total_seconds())
        doc["sla_remaining_sec"] = max(remaining, 0)
        doc["sla_breached"] = remaining <= 0
        doc["sla_deadline"] = _iso(deadline)
    else:
        doc["sla_remaining_sec"] = None
        doc["sla_breached"] = None
        doc["sla_deadline"] = None

    # serialize datetimes for JSON
    if "timestamp" in doc and isinstance(doc["timestamp"], datetime):
        doc["timestamp"] = _iso(doc["timestamp"])
    if "assigned_at" in doc and isinstance(doc["assigned_at"], datetime):
        doc["assigned_at"] = _iso(doc["assigned_at"])
    if "sla_started_at" in doc and isinstance(doc["sla_started_at"], datetime):
        doc["sla_started_at"] = _iso(doc["sla_started_at"])

    return doc


# =========================================================
# SUPPORT TABLE (PAGINATED) — infinite scroll
# =========================================================
@router.get("/dashboard/phase2/support")
def get_support_view(
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
):
    now = datetime.utcnow()

    projection = {
        "_id": 0,
        "ID": 1,
        "phase": 1,
        "risk": 1,
        "decision": 1,
        "actions": 1,
        "discount": 1,
        "churn_prob": 1,
        "confidence": 1,
        "timestamp": 1,
        "sla_started_at": 1,   # ✅ include it
        "agent_notes": 1,
        "assigned_agent": 1,
        "assigned_at": 1,
        "demo_brief": 1,
        "email_draft": 1,
        "call_script": 1,
        "human_review_required": 1,
    }

    docs = list(
        phase2_actions_col.find({"phase": "PHASE_2"}, projection)
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )

    out = []
    for d in docs:
        d["customer_id"] = _canon_id(d)
        d.setdefault("actions", [])
        d.setdefault("risk", "NO_RISK")
        d.setdefault("churn_prob", 0.0)
        d.setdefault("confidence", 0.0)
        d.setdefault("assigned_agent", None)

        d = _sla_fields(d, now)
        out.append(d)

    return out  # ARRAY


# =========================================================
# SUPPORT SUMMARY (KPIs for ALL customers)
# =========================================================
@router.get("/dashboard/phase2/support/summary")
def get_support_summary():
    pipeline = [
        {"$match": {"phase": "PHASE_2"}},
        {"$project": {
            "risk": {"$ifNull": ["$risk", "NO_RISK"]},
            "actions": {"$ifNull": ["$actions", []]},
            "confidence": {"$ifNull": ["$confidence", 0]},
        }},
        {"$group": {
            "_id": None,
            "total": {"$sum": 1},
            "scheduled_calls": {"$sum": {"$cond": [{"$in": ["schedule_call", "$actions"]}, 1, 0]}},
            "high_risk": {"$sum": {"$cond": [{"$eq": ["$risk", "HIGH"]}, 1, 0]}},
            "avg_conf": {"$avg": "$confidence"},
        }},
    ]

    agg = list(phase2_actions_col.aggregate(pipeline))
    if not agg:
        return {"total": 0, "scheduled_calls": 0, "high_risk": 0, "avg_confidence": 0.0}

    s = agg[0]
    return {
        "total": int(s.get("total", 0)),
        "scheduled_calls": int(s.get("scheduled_calls", 0)),
        "high_risk": int(s.get("high_risk", 0)),
        "avg_confidence": float(s.get("avg_conf", 0.0) or 0.0),
    }


# =========================================================
# ASSIGN AGENT (match by ID)
# =========================================================
@router.post("/dashboard/phase2/assign_agent")
def assign_agent(payload: dict = Body(...)):
    customer_id = payload.get("customer_id") or payload.get("ID")
    agent = payload.get("agent")

    if not customer_id or not agent:
        raise HTTPException(status_code=400, detail="customer_id and agent are required")

    res = phase2_actions_col.update_one(
    {"phase": "PHASE_2", "ID": str(customer_id)},
    {"$set": {"assigned_agent": agent, "assigned_at": datetime.utcnow()}},
    upsert=False
)

    return {"status": "assigned", "matched": res.matched_count, "modified": res.modified_count}
