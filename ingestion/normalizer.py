import re
from datetime import datetime


def normalize_month_label(raw) -> str:
    """Normalize any month string to canonical 'March 2026' format."""
    if not raw:
        return ""
    s = str(raw).strip()

    # Already canonical: "March 2026"
    for fmt in ("%B %Y", "%B %y", "%b %Y", "%b %y"):
        try:
            return datetime.strptime(s, fmt).strftime("%B %Y")
        except ValueError:
            continue

    # ISO-ish: "2026-03"
    m = re.match(r"^(\d{4})-(\d{1,2})$", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), 1).strftime("%B %Y")
        except ValueError:
            pass

    # LLM-style: "MARCH'26", "March-2026", "MARCH 26", "Mar_2026"
    m = re.match(r"([A-Za-z]+)\D*(\d{2,4})", s)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        if len(year_str) == 2:
            year_str = "20" + year_str
        for fmt in ("%B", "%b"):
            try:
                month_num = datetime.strptime(month_name, fmt).month
                return datetime(int(year_str), month_num, 1).strftime("%B %Y")
            except ValueError:
                continue

    return s  # fallback: return as-is


def normalize_employee_name(name: str) -> str:
    """Normalize employee name: title case, collapse whitespace."""
    if not name:
        return ""
    return " ".join(str(name).strip().title().split())


def normalize_record(rec: dict) -> dict:
    return {
        "employee": normalize_employee_name(rec.get("employee")),
        "project": (rec.get("project") or "").strip(),
        "month": normalize_month_label(rec.get("month")),
        "actual_hours": rec.get("actual_hours") or 0,
        "billable_hours": rec.get("billable_hours") or 0,
        "expected_hours": rec.get("expected_hours") or 0,
        "working_days": rec.get("working_days") or 0,
        "vacation_days": rec.get("vacation_days") or 0,
        "effective_working_days": rec.get("effective_working_days") or 0,
        "leave_hours": rec.get("leave_hours") or 0,
        "billing_rate": rec.get("billing_rate"),
        "cost_rate": rec.get("cost_rate"),
        "revenue": rec.get("revenue"),
        "cost": rec.get("cost"),
        "profit": rec.get("profit"),
        "margin_pct": rec.get("margin_pct"),
        "utilisation_pct": rec.get("utilisation_pct"),
        "validation_flags": rec.get("validation_flags", []),
        "loss_reasons": rec.get("loss_reasons", []),
        "is_profitable": rec.get("is_profitable"),
        "is_valid": rec.get("is_valid", True),
        "_source": rec.get("_source", "timesheet_parser"),
    }