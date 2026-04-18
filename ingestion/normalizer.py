def normalize_record(rec: dict) -> dict:
    return {
        "employee": (rec.get("employee") or "").strip().title(),
        "project": (rec.get("project") or "").strip(),
        "month": rec.get("month"),
        "actual_hours": rec.get("actual_hours") or 0,
        "billable_hours": rec.get("billable_hours") or 0,
        "working_days": rec.get("working_days") or 0,
        "vacation_days": rec.get("vacation_days") or 0,
        "billing_rate": rec.get("billing_rate"),
        "cost_rate": rec.get("cost_rate"),
        "revenue": rec.get("revenue"),
        "cost": rec.get("cost"),
        "profit": rec.get("profit"),
        "margin_pct": rec.get("margin_pct"),
        "utilisation_pct": rec.get("utilisation_pct"),
        "validation_flags": rec.get("validation_flags", []),
        "is_profitable": rec.get("is_profitable"),
    }