"""
dataset.py — Global in-memory dataset and analytics transforms.
"""

import datetime as dt
import re
from typing import Optional, List

GLOBAL_DATASET: List[dict] = []

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_FILES_PROCESSED: List[str] = []


def clear():
    GLOBAL_DATASET.clear()
    _FILES_PROCESSED.clear()


def append_records(records: List[dict], filename: str = ""):
    if filename and filename not in _FILES_PROCESSED:
        _FILES_PROCESSED.append(filename)
    skipped = 0
    for r in records:
        emp  = r.get("employee") or ""
        proj = r.get("project") or ""
        mon  = r.get("month") or ""

        # Skip records with no identifiable employee
        if not emp or emp.lower() in ("unknown", "none", ""):
            skipped += 1
            continue

        # Dedup: skip if identical employee+month+project already exists
        if not any(
            (existing.get("employee") or "") == emp
            and (existing.get("month") or "") == mon
            and (existing.get("project") or "") == proj
            for existing in GLOBAL_DATASET
        ):
            GLOBAL_DATASET.append(r)

    if skipped:
        print(f"[dataset] Skipped {skipped} records with missing employee name")


def _parse_month_date(month_str: str) -> Optional[dt.date]:
    for fmt in ("%B %Y", "%B %y", "%b %Y", "%b %y"):
        try:
            return dt.datetime.strptime(month_str, fmt).date()
        except ValueError:
            continue
    m = re.match(r"([A-Za-z]+)\D*(\d{2,4})", month_str)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        if len(year_str) == 2:
            year_str = "20" + year_str
        for fmt in ("%B", "%b"):
            try:
                month_num = dt.datetime.strptime(month_name, fmt).month
                return dt.date(int(year_str), month_num, 1)
            except ValueError:
                continue
    return None


def filter_by_range(records: List[dict], time_range: Optional[str] = None) -> List[dict]:
    if not time_range or time_range.upper() == "ALL":
        return records

    days_map = {"1M": 30, "3M": 90, "6M": 180, "12M": 365}
    cutoff_days = days_map.get(time_range.upper())
    if not cutoff_days:
        return records

    dated = []
    for r in records:
        d = _parse_month_date(r.get("month", ""))
        if d:
            dated.append((r, d))

    if not dated:
        return records

    max_date = max(d for _, d in dated)
    cutoff = max_date - dt.timedelta(days=cutoff_days)

    return [r for r, d in dated if d >= cutoff]


def get_months_available(records: List[dict] = None) -> List[str]:
    recs = records if records is not None else GLOBAL_DATASET
    months_set = set()
    for r in recs:
        m = r.get("month")
        if m and isinstance(m, str):
            months_set.add(m)

    def _sort_key(m):
        try:
            d = _parse_month_date(m)
            return d if d else dt.date.min
        except Exception:
            return dt.date.min

    return sorted(months_set, key=_sort_key)


def build_projects(records: List[dict]) -> dict:
    projects = {}
    for r in records:
        proj = r.get("project") or "Unknown"
        if proj not in projects:
            projects[proj] = {
                "revenue": 0, "cost": 0, "profit": 0,
                "employees": set(),
            }
        projects[proj]["revenue"] += r.get("revenue", 0)
        projects[proj]["cost"] += r.get("cost", 0)
        projects[proj]["profit"] += r.get("profit", 0)
        projects[proj]["employees"].add(r.get("employee", ""))

    for p in projects.values():
        p["revenue"] = round(p["revenue"], 2)
        p["cost"] = round(p["cost"], 2)
        p["profit"] = round(p["profit"], 2)
        p["employees"] = len(p["employees"])

    return projects


def build_monthly(records: List[dict]) -> dict:
    monthly = {}
    for r in records:
        m = r.get("month", "Unknown")
        if m not in monthly:
            monthly[m] = {
                "total_revenue": 0, "total_cost": 0, "total_profit": 0,
                "employees": set(),
            }
        monthly[m]["total_revenue"] += r.get("revenue", 0)
        monthly[m]["total_cost"] += r.get("cost", 0)
        monthly[m]["total_profit"] += r.get("profit", 0)
        monthly[m]["employees"].add(r.get("employee", ""))

    for v in monthly.values():
        v["total_revenue"] = round(v["total_revenue"], 2)
        v["total_cost"] = round(v["total_cost"], 2)
        v["total_profit"] = round(v["total_profit"], 2)
        rev = v["total_revenue"]
        pft = v["total_profit"]
        v["avg_margin_pct"] = round((pft / rev) * 100, 2) if rev > 0 else 0
        v["employees"] = len(v["employees"])

    return monthly


def build_overall_summary(records: List[dict]) -> dict:
    total_rev = round(sum(r.get("revenue", 0) for r in records), 2)
    total_cost = round(sum(r.get("cost", 0) for r in records), 2)
    total_profit = round(sum(r.get("profit", 0) for r in records), 2)
    unique_employees = len({r.get("employee") for r in records if r.get("employee")})
    return {
        "total_revenue": total_rev,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "avg_margin_pct": round((total_profit / total_rev) * 100, 2) if total_rev > 0 else 0,
        "total_employees": unique_employees,
    }


def build_top_performers(records: List[dict], limit: int = 5) -> List[dict]:
    emp_profit = {}
    for r in records:
        name = r.get("employee", "")
        emp_profit[name] = emp_profit.get(name, 0) + r.get("profit", 0)
    ranked = sorted(emp_profit.items(), key=lambda x: x[1], reverse=True)
    return [{"employee": name, "total_profit": round(pft, 2)} for name, pft in ranked[:limit]]


def build_risks(records: List[dict]) -> List[dict]:
    risks = []
    for r in records:
        if r.get("profit", 0) < 0:
            risks.append({"employee": r["employee"], "month": r.get("month"), "issue": "LOSS_MAKING", "profit": r["profit"]})
        if r.get("vacation_days", 0) >= 3:
            risks.append({"employee": r["employee"], "month": r.get("month"), "issue": "HIGH_LEAVE", "vacation_days": r["vacation_days"]})
        if r.get("utilisation_pct", 100) < 80:
            risks.append({"employee": r["employee"], "month": r.get("month"), "issue": "LOW_UTILISATION", "utilisation_pct": r["utilisation_pct"]})
    return risks


def _clean_record_for_api(r: dict) -> dict:
    return {
        "employee": r.get("employee", ""),
        "project": r.get("project", ""),
        "month": r.get("month", ""),
        "actual_hours": r.get("actual_hours", 0),
        "billable_hours": r.get("billable_hours", 0),
        "working_days": r.get("working_days", 0),
        "vacation_days": r.get("vacation_days", 0),
        "billing_rate": r.get("billing_rate", 0),
        "cost_rate": r.get("cost_rate", 0),
        "revenue": r.get("revenue", 0),
        "cost": r.get("cost", 0),
        "profit": r.get("profit", 0),
        "margin_pct": r.get("margin_pct", 0),
        "utilisation_pct": r.get("utilisation_pct", 0),
        "is_profitable": r.get("is_profitable", True),
    }


def transform_to_api_format(time_range: Optional[str] = None) -> dict:
    filtered = filter_by_range(GLOBAL_DATASET, time_range)
    employees = [_clean_record_for_api(r) for r in filtered]

    return {
        "metadata": {
            "files_processed": len(_FILES_PROCESSED),
            "months_available": get_months_available(filtered),
            "time_range": time_range or "ALL",
            "total_records": len(employees),
        },
        "employees": employees,
        "projects": build_projects(filtered),
        "monthly": build_monthly(filtered),
        "overall_summary": build_overall_summary(filtered),
        "top_performers": build_top_performers(filtered),
        "risks": build_risks(filtered),
    }
