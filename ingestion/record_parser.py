"""
record_parser.py — type-cast and clean each mapped record.
Handles: date parsing (many formats), numeric coercion, days→hours conversion,
         missing value defaults, per-record validation flags.
"""

import re
from datetime import date, datetime
from typing import Optional

from .field_config import DATE_FORMATS, OPTIONAL_FIELDS, REQUIRED_FIELDS


# ── Date parsing ──────────────────────────────────────────────────────────────

def parse_date(raw: str) -> Optional[date]:
    if not raw or raw.strip() == "":
        return None
    raw = raw.strip()

    # Try standard formats
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue

    # Try pandas as fallback (handles many ambiguous formats)
    try:
        import pandas as pd
        return pd.to_datetime(raw, dayfirst=False).date()
    except Exception:
        pass

    return None


# ── Numeric parsing ───────────────────────────────────────────────────────────

def parse_float(raw: str) -> Optional[float]:
    if not raw or raw.strip() == "":
        return None
    # Strip currency symbols, commas, percent signs
    cleaned = re.sub(r"[£$€¥,% ]", "", str(raw).strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_hours(raw: str, field_name: str = "hours") -> Optional[float]:
    """Parse hours — if original column looked like 'days', convert ×8."""
    val = parse_float(raw)
    if val is None:
        return None
    # If column name suggested days (captured in extra_ prefix earlier),
    # days flag is passed in via field_name hint
    if "day" in field_name.lower():
        val = val * 8.0
    return val


# ── Per-record validation flags ───────────────────────────────────────────────

def validate_record(record: dict) -> list[str]:
    flags = []

    hours = record.get("hours")
    if hours is None:
        flags.append("MISSING_HOURS")
    elif hours > 24:
        flags.append("INVALID_HOURS")
    elif hours <= 0:
        flags.append("ZERO_OR_NEGATIVE_HOURS")

    if record.get("billing_rate") is None:
        flags.append("MISSING_BILLING")

    if record.get("cost_rate") is None:
        flags.append("MISSING_COST")

    if record.get("date") is None:
        flags.append("MISSING_DATE")

    if not record.get("employee"):
        flags.append("MISSING_EMPLOYEE")

    if not record.get("project"):
        flags.append("MISSING_PROJECT")

    return flags


# ── Metrics computation ───────────────────────────────────────────────────────

def compute_metrics(record: dict) -> dict:
    hours = record.get("hours") or 0.0
    br = record.get("billing_rate")
    cr = record.get("cost_rate")

    revenue = round(hours * br, 2) if br is not None else None
    cost = round(hours * cr, 2) if cr is not None else None
    profit = round(revenue - cost, 2) if (revenue is not None and cost is not None) else None
    margin = round((profit / revenue) * 100, 2) if (profit is not None and revenue and revenue > 0) else None

    return {
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "margin_pct": margin,
    }


# ── Time features ─────────────────────────────────────────────────────────────

def time_features(d: Optional[date]) -> dict:
    if d is None:
        return {"month": None, "quarter": None, "year": None,
                "month_label": None, "quarter_label": None}
    q = (d.month - 1) // 3 + 1
    return {
        "month": d.month,
        "year": d.year,
        "quarter": q,
        "month_label": d.strftime("%Y-%m"),
        "quarter_label": f"{d.year}-Q{q}",
    }


# ── Main record parser ────────────────────────────────────────────────────────

def parse_record(raw_record: dict) -> dict:
    """
    Take a single mapped row dict and return a fully typed, validated,
    metrics-enriched record dict.
    """
    # Parse each canonical field
    employee = (raw_record.get("employee") or "").strip() or None
    project = (raw_record.get("project") or "").strip() or None
    date_val = parse_date(raw_record.get("date") or "")
    # Find the original raw column name that was mapped to hours (stored as extra hint)
    hours_raw_key = raw_record.get("_hours_original_col", "hours")
    hours = parse_hours(raw_record.get("hours") or "", hours_raw_key)
    billing_rate = parse_float(raw_record.get("billing_rate") or "")
    cost_rate = parse_float(raw_record.get("cost_rate") or "")

    # Collect any extra_ columns
    extras = {k: v for k, v in raw_record.items() if k.startswith("extra_")}

    base = {
        "employee": employee,
        "project": project,
        "date": date_val.isoformat() if date_val else None,
        "hours": hours,
        "billing_rate": billing_rate,
        "cost_rate": cost_rate,
    }

    # Validate
    flags = validate_record({**base, "date": date_val})

    # Metrics
    metrics = compute_metrics(base)

    # Time features
    tf = time_features(date_val)

    return {
        **base,
        **metrics,
        **tf,
        "validation_flags": flags,
        "is_valid": len([f for f in flags if "MISSING_BILLING" not in f and "MISSING_COST" not in f]) == 0,
        **extras,
    }
