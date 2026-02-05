from fastapi import APIRouter, Body, HTTPException, Query
from db.mongo import phase2_actions_col, metrics_col
from datetime import datetime

router = APIRouter()

def _iso(dt):
    try:
        return dt.isoformat() if hasattr(dt, "isoformat") else dt
    except Exception:
        return dt

@router.get("/phase2/actions")
def get_phase2_actions(
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=1000),
):
    """
    Pagination for infinite scroll:
    - Returns ARRAY (frontend expects array)
    - Supports ?skip=0&limit=200
    """
    projection = {
        "_id": 0,
        "ID": 1,
        "risk": 1,
        "decision": 1,
        "actions": 1,
        "discount": 1,
        "churn_prob": 1,
        "auto_executed": 1,
        "human_review_required": 1,
        "confidence": 1,
        "timestamp": 1,
        "agent_notes": 1,
        "demo_brief": 1,
        "email_draft": 1,
        "call_script": 1,
    }

    docs = list(
        phase2_actions_col.find({}, projection)
        .sort("timestamp", -1)
        .skip(skip)
        .limit(limit)
    )

    for d in docs:
        d.setdefault("ID", "UNKNOWN")
        d.setdefault("risk", "NO_RISK")
        d.setdefault("decision", "NONE")
        d.setdefault("actions", [])
        d.setdefault("discount", 0.0)
        d.setdefault("churn_prob", 0.0)
        d.setdefault("auto_executed", False)
        d.setdefault("human_review_required", True)
        d.setdefault("confidence", 0.0)
        d.setdefault("agent_notes", "")
        if "timestamp" in d:
            d["timestamp"] = _iso(d["timestamp"])
        d["customer_id"] = d.get("ID")  # UI uses customer_id || ID

    return docs  # ✅ ARRAY


# ✅ GET Phase 2 metrics (not used by your current HTML for the graph)
@router.get("/phase2/metrics")
def get_phase2_metrics():
    metrics = list(
        metrics_col.find(
            {"phase": {"$in": ["PHASE_0", "PHASE_1", "PHASE_2"]}},
            {"_id": 0}
        ).sort("trained_at", 1)
    )

    for m in metrics:
        if "trained_at" in m:
            m["trained_at"] = _iso(m["trained_at"])
        else:
            m["trained_at"] = datetime.utcnow().isoformat()

    return metrics  # ARRAY is fine here too

# ✅ POST notes from support team sidebar
@router.post("/phase2/notes")
def save_notes(payload: dict = Body(...)):
    customer_id = payload.get("customer_id") or payload.get("ID")
    notes = payload.get("notes", "")

    if not customer_id:
        raise HTTPException(status_code=400, detail="customer_id required")

    res = phase2_actions_col.update_one(
        {"ID": str(customer_id)},
        {"$set": {"agent_notes": notes, "notes_updated_at": datetime.utcnow()}}
    )
    return {"status": "ok", "matched": res.matched_count, "modified": res.modified_count}

@router.get("/phase2/summary")
def get_phase2_summary():
    """
    Fast KPI summary for ALL Phase 2 actions (not just the current page).
    Uses Mongo aggregation so UI stays fast even with many rows.
    """
    pipeline = [
        {
            "$group": {
                "_id": None,
                "total": {"$sum": 1},
                "auto_actions": {
                    "$sum": {"$cond": [{"$eq": ["$auto_executed", True]}, 1, 0]}
                },
                "escalations": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$decision", "ESCALATE"]},
                            1,
                            0
                        ]
                    }
                },
                "retention_value": {"$sum": {"$ifNull": ["$discount", 0]}},
                "avg_confidence": {"$avg": {"$ifNull": ["$confidence", 0]}},
            }
        }
    ]

    out = list(phase2_actions_col.aggregate(pipeline))
    if not out:
        return {
            "total": 0,
            "auto_actions": 0,
            "escalations": 0,
            "retention_value": 0.0,
            "avg_confidence": 0.0
        }

    r = out[0]
    return {
        "total": int(r.get("total", 0)),
        "auto_actions": int(r.get("auto_actions", 0)),
        "escalations": int(r.get("escalations", 0)),
        "retention_value": float(r.get("retention_value", 0.0)),
        "avg_confidence": float(r.get("avg_confidence", 0.0)),
    }

@router.get("/phase2/risk_summary")
def get_phase2_risk_summary():
    """
    Returns ALL-customer risk distribution for Phase 2 pie chart.
    Uses Mongo aggregation so it's fast and not dependent on pagination.
    """
    pipeline = [
        {
            "$group": {
                "_id": "$risk",
                "count": {"$sum": 1}
            }
        }
    ]

    rows = list(phase2_actions_col.aggregate(pipeline))

    # Default buckets (so frontend always has 4 keys)
    out = {"NO_RISK": 0, "LOW": 0, "MODERATE": 0, "HIGH": 0}

    for r in rows:
        k = r.get("_id") or "NO_RISK"
        if k not in out:
            out[k] = 0
        out[k] += int(r.get("count", 0))

    return out
