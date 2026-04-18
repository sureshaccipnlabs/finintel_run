"""
record_parser.py — Parses a single mapped row into canonical format.
"""

from typing import Dict, Any, List
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def to_float(v):
    try:
        return float(v) if v is not None else None
    except:
        return None


def normalize_employee(name: str) -> str:
    if not name:
        return None
    return " ".join(str(name).strip().title().split())


def normalize_project(name: str) -> str:
    if not name:
        return None
    return str(name).replace("-", "").replace(" ", "").lower()


def normalize_month(m):
    if not m:
        return None
    try:
        return datetime.strptime(m, "%B %Y").strftime("%Y-%m")
    except:
        return str(m)


# ─────────────────────────────────────────────────────────────────────────────
# Main Parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_record(row: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []

    # Extract fields
    employee = normalize_employee(row.get("employee"))
    project = normalize_project(row.get("project"))
    month = normalize_month(row.get("month"))

    hours = to_float(row.get("hours"))
    billing_rate = to_float(row.get("billing_rate"))
    cost_rate = to_float(row.get("cost_rate"))

    # Defaults
    actual_hours = hours or 0
    billable_hours = hours or 0
    working_days = round(actual_hours / 8, 2) if actual_hours else 0
    vacation_days = 0

    # ─────────────────────────────────────────
    # Financial Calculations
    # ─────────────────────────────────────────

    # Revenue
    if billing_rate:
        revenue = round(billable_hours * billing_rate, 2)
    else:
        revenue = None
        flags.append("MISSING_BILLING_RATE")

    # Cost
    if cost_rate and cost_rate != 0:
        cost = round(actual_hours * cost_rate, 2)
    else:
        cost = None
        flags.append("MISSING_COST_RATE")

    # Profit
    if revenue is not None and cost is not None:
        profit = round(revenue - cost, 2)
    else:
        profit = None

    # Margin
    if revenue and profit is not None and revenue > 0:
        margin_pct = round((profit / revenue) * 100, 2)
    else:
        margin_pct = None

    # Utilisation
    utilisation_pct = round((actual_hours / (actual_hours or 1)) * 100, 2) if actual_hours else 0

    # Flags
    if margin_pct is not None and margin_pct < 10:
        flags.append("LOW_MARGIN")
    if margin_pct is not None and margin_pct < 0:
        flags.append("LOSS_MAKING")

    # ─────────────────────────────────────────
    # Final Record
    # ─────────────────────────────────────────

    return {
        "employee": employee,
        "project": project,
        "month": month,
        "actual_hours": actual_hours,
        "billable_hours": billable_hours,
        "working_days": working_days,
        "vacation_days": vacation_days,
        "billing_rate": billing_rate,
        "cost_rate": cost_rate,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "margin_pct": margin_pct,
        "utilisation_pct": utilisation_pct,
        "validation_flags": flags,
        "is_profitable": profit > 0 if profit is not None else None,
    }