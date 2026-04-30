"""
dataset.py — Global in-memory dataset and analytics transforms.
"""

import datetime as dt
import re
import threading
import math
from typing import Optional, List

_LOCK = threading.Lock()
GLOBAL_DATASET: List[dict] = []

# Callback for cache invalidation (set by qa_engine to avoid circular import)
_ON_DATASET_CHANGE_CALLBACK = None

def set_on_dataset_change_callback(callback):
    """Register a callback to be called when dataset changes."""
    global _ON_DATASET_CHANGE_CALLBACK
    _ON_DATASET_CHANGE_CALLBACK = callback

def _notify_dataset_change():
    """Notify listeners that dataset has changed."""
    if _ON_DATASET_CHANGE_CALLBACK:
        try:
            _ON_DATASET_CHANGE_CALLBACK()
        except Exception:
            pass

MONTH_ORDER = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

_FILES_PROCESSED: List[str] = []
_DEDUP_KEYS: set = set()


def _normalize_key(s: str) -> str:
    """Normalize a string for dedup comparison: lowercase, strip all non-alphanumeric."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _to_num(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if text == "":
            return 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _calc_margin(total_revenue: float, total_cost: float) -> float:
    if total_revenue <= 0:
        return 0.0
    return round(((total_revenue - total_cost) / total_revenue) * 100, 2)


def _month_sort_key(month_label: str) -> dt.date:
    parsed = _parse_month_date(month_label)
    return parsed if parsed else dt.date.min


def _trend_from_values(values: List[float]) -> str:
    if len(values) < 2:
        return "Stable"
    compare = values[-3:] if len(values) >= 3 else values[-2:]
    start = compare[0]
    end = compare[-1]
    delta = end - start
    baseline = max(abs(start), 1.0)
    if abs(delta) <= baseline * 0.03:
        return "Stable"
    return "Up" if delta > 0 else "Down"


def clear():
    with _LOCK:
        GLOBAL_DATASET.clear()
        _FILES_PROCESSED.clear()
        _DEDUP_KEYS.clear()
    _notify_dataset_change()


def append_records(records: List[dict], filename: str = ""):
    from .normalizer import normalize_month_label
    with _LOCK:
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

            # Normalize month on ingest for consistency
            norm_month = normalize_month_label(mon)
            if norm_month and norm_month != mon:
                r["month"] = norm_month
                mon = norm_month

            # Tag record with source file for audit trail
            if filename and "_source_file" not in r:
                r["_source_file"] = filename

            # O(1) dedup with fuzzy key: "Proof point" == "proofpoint", case-insensitive names
            key = (_normalize_key(emp), _normalize_key(mon), _normalize_key(proj))
            if key not in _DEDUP_KEYS:
                _DEDUP_KEYS.add(key)
                GLOBAL_DATASET.append(r)

        if skipped:
            print(f"[dataset] Skipped {skipped} records with missing employee name")
    _notify_dataset_change()


def remove_by_file(filename: str) -> int:
    """Remove all records from a specific source file. Returns count removed."""
    removed = 0
    with _LOCK:
        before = len(GLOBAL_DATASET)
        remaining = [r for r in GLOBAL_DATASET if r.get("_source_file") != filename]
        removed = before - len(remaining)
        GLOBAL_DATASET.clear()
        GLOBAL_DATASET.extend(remaining)
        # Rebuild dedup keys
        _DEDUP_KEYS.clear()
        for r in GLOBAL_DATASET:
            emp = r.get("employee") or ""
            mon = r.get("month") or ""
            proj = r.get("project") or ""
            _DEDUP_KEYS.add((_normalize_key(emp), _normalize_key(mon), _normalize_key(proj)))
        if filename in _FILES_PROCESSED:
            _FILES_PROCESSED.remove(filename)
    _notify_dataset_change()
    return removed


def get_files_processed() -> List[str]:
    return list(_FILES_PROCESSED)


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

    key = time_range.upper().strip()
    if key.endswith("M"):
        key = key[:-1]

    if not key.isdigit():
        return records

    months_requested = int(key)
    if months_requested <= 0:
        return records

    month_dates = sorted({_parse_month_date(r.get("month", "")) for r in records if _parse_month_date(r.get("month", ""))})
    if not month_dates:
        return records

    allowed_months = set(month_dates[-months_requested:])

    return [
        r for r in records
        if _parse_month_date(r.get("month", "")) in allowed_months
    ]


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
                "hours": 0.0, "approved_hours": 0.0,
                "employees": set(),
            }
        projects[proj]["revenue"] += r.get("revenue") or 0
        projects[proj]["cost"] += r.get("cost") or 0
        projects[proj]["profit"] += r.get("profit") or 0
        projects[proj]["hours"] += _to_num(r.get("actual_hours"))
        projects[proj]["approved_hours"] += _to_num(r.get("expected_hours") if r.get("expected_hours") is not None else r.get("max_hours"))
        projects[proj]["employees"].add(r.get("employee", ""))

    for p in projects.values():
        p["revenue"] = round(p["revenue"], 2)
        p["cost"] = round(p["cost"], 2)
        p["profit"] = round(p["profit"], 2)
        if p["approved_hours"] > 0:
            p["avg_utilisation"] = round((p["hours"] / p["approved_hours"]) * 100, 2)
        else:
            p["avg_utilisation"] = 0
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
                "leave_days": 0.0, "vacation_days": 0.0, "holiday_days": 0.0,
                "working_days": 0.0, "total_hours": 0.0,
            }
        monthly[m]["total_revenue"] += r.get("revenue") or 0
        monthly[m]["total_cost"] += r.get("cost") or 0
        monthly[m]["total_profit"] += r.get("profit") or 0
        monthly[m]["employees"].add(r.get("employee", ""))
        monthly[m]["leave_days"] += (r.get("leave_days") or 0)
        monthly[m]["vacation_days"] += (r.get("vacation_days") or 0)
        monthly[m]["holiday_days"] += (r.get("holiday_days") or 0)
        monthly[m]["working_days"] += (r.get("working_days") or 0)
        monthly[m]["total_hours"] += (r.get("actual_hours") or 0)

    for v in monthly.values():
        v["total_revenue"] = round(v["total_revenue"], 2)
        v["total_cost"] = round(v["total_cost"], 2)
        v["total_profit"] = round(v["total_profit"], 2)
        rev = v["total_revenue"]
        pft = v["total_profit"]
        v["avg_margin_pct"] = round((pft / rev) * 100, 2) if rev > 0 else 0
        v["employees"] = len(v["employees"])
        v["leave_days"] = round(v["leave_days"], 2)
        v["vacation_days"] = round(v["vacation_days"], 2)
        v["holiday_days"] = round(v["holiday_days"], 2)
        v["working_days"] = round(v["working_days"], 2)
        v["total_hours"] = round(v["total_hours"], 2)

    return monthly


def build_overall_summary(records: List[dict]) -> dict:
    total_rev = 0.0
    total_cost = 0.0
    total_hours = 0.0
    employees = set()

    for r in records:
        total_rev += _to_num(r.get("revenue"))
        total_cost += _to_num(r.get("cost"))
        total_hours += _to_num(r.get("actual_hours"))
        name = r.get("employee")
        if name:
            employees.add(name)

    total_rev = round(total_rev, 2)
    total_cost = round(total_cost, 2)
    total_profit = round(total_rev - total_cost, 2)
    return {
        "total_revenue": total_rev,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "avg_margin_pct": _calc_margin(total_rev, total_cost),
        "total_employees": len(employees),
        "total_hours": round(total_hours, 2),
    }


def build_project_summaries(records: List[dict]) -> List[dict]:
    projects = {}

    for r in records:
        project_name = r.get("project") or "Unknown"
        month = r.get("month") or "Unknown"
        revenue = _to_num(r.get("revenue"))
        cost = _to_num(r.get("cost"))

        if project_name not in projects:
            projects[project_name] = {
                "revenue": 0.0,
                "cost": 0.0,
                "monthly": {},
                "employees": set(),
            }

        proj = projects[project_name]
        proj["revenue"] += revenue
        proj["cost"] += cost
        employee_name = r.get("employee")
        if employee_name:
            proj["employees"].add(employee_name)

        if month not in proj["monthly"]:
            proj["monthly"][month] = {"revenue": 0.0, "cost": 0.0}
        proj["monthly"][month]["revenue"] += revenue
        proj["monthly"][month]["cost"] += cost

    output = []
    for project_name, data in projects.items():
        total_revenue = round(data["revenue"], 2)
        total_cost = round(data["cost"], 2)
        total_profit = round(total_revenue - total_cost, 2)
        gross_margin_pct = _calc_margin(total_revenue, total_cost)

        ordered_months = sorted(data["monthly"].keys(), key=_month_sort_key)
        revenue_series = [round(data["monthly"][m]["revenue"], 2) for m in ordered_months]
        cost_series = [round(data["monthly"][m]["cost"], 2) for m in ordered_months]
        profit_series = [round(rev - cst, 2) for rev, cst in zip(revenue_series, cost_series)]
        margin_series = [_calc_margin(rev, cst) for rev, cst in zip(revenue_series, cost_series)]

        if gross_margin_pct > 40:
            status = "Healthy"
        elif gross_margin_pct >= 30:
            status = "Optimal"
        else:
            status = "At Risk"

        output.append({
            "project_name": project_name,
            "total_revenue": total_revenue,
            "total_cost": total_cost,
            "total_profit": total_profit,
            "gross_margin_pct": gross_margin_pct,
            "employees": len(data["employees"]),
            "status": status,
            "trends": {
                "revenue_trend": _trend_from_values(revenue_series),
                "cost_trend": _trend_from_values(cost_series),
                "profit_trend": _trend_from_values(profit_series),
                "margin_trend": _trend_from_values(margin_series),
            },
        })

    output.sort(key=lambda x: x["total_profit"], reverse=True)
    return output


def build_employee_summaries(records: List[dict]) -> List[dict]:
    employees = {}

    for r in records:
        employee_name = r.get("employee") or "Unknown"
        project_name = r.get("project") or "Unknown"

        hours = _to_num(r.get("actual_hours"))
        revenue = _to_num(r.get("revenue"))
        cost = _to_num(r.get("cost"))
        profit = revenue - cost
        approved_hours = _to_num(r.get("expected_hours") if r.get("expected_hours") is not None else r.get("max_hours"))

        vacation_days = _to_num(r.get("vacation_days"))
        leave_days = _to_num(r.get("leave_days"))
        working_days = _to_num(r.get("working_days"))

        if employee_name not in employees:
            employees[employee_name] = {
                "hours": 0.0,
                "revenue": 0.0,
                "profit": 0.0,
                "approved_hours": 0.0,
                "vacation_days": 0.0,
                "leave_days": 0.0,
                "working_days": 0.0,
                "projects": {},
            }

        emp = employees[employee_name]
        emp["hours"] += hours
        emp["revenue"] += revenue
        emp["profit"] += profit
        emp["approved_hours"] += approved_hours
        emp["vacation_days"] += vacation_days
        emp["leave_days"] += leave_days
        emp["working_days"] += working_days

        if project_name not in emp["projects"]:
            emp["projects"][project_name] = {
                "hours": 0.0,
                "revenue": 0.0,
                "profit": 0.0,
            }

        proj = emp["projects"][project_name]
        proj["hours"] += hours
        proj["revenue"] += revenue
        proj["profit"] += profit

    total_employees = len(employees)
    contribution_slice = max(1, math.ceil(total_employees * 0.25)) if total_employees else 0
    ranked = sorted(employees.items(), key=lambda item: item[1]["profit"], reverse=True)
    high_contributors = {name for name, _ in ranked[:contribution_slice]}
    low_contributors = {name for name, _ in ranked[-contribution_slice:]} if contribution_slice else set()

    output = []
    for employee_name, data in employees.items():
        approved_total = data["approved_hours"]
        utilization = round((data["hours"] / approved_total) * 100, 2) if approved_total > 0 else None
        margin_pct = round((data["profit"] / data["revenue"]) * 100, 2) if data["revenue"] > 0 else 0.0
        
        # Attendance calculation: match risk_engine formula
        # Uses only vacation_days (same as risk_engine.py line 237-243)
        vacation = data["vacation_days"]
        working = data["working_days"]
        leave_pct = round((vacation / working) * 100, 2) if working > 0 else 0.0
        attendance_pct = round(100.0 - leave_pct, 2)

        projects = []
        for project_name, p in data["projects"].items():
            projects.append({
                "project_name": project_name,
                "revenue": round(p["revenue"], 2),
                "profit": round(p["profit"], 2),
                "hours": round(p["hours"], 2),
            })
        projects.sort(key=lambda x: x["profit"], reverse=True)

        if employee_name in high_contributors:
            contribution_status = "High"
        elif employee_name in low_contributors:
            contribution_status = "Low"
        else:
            contribution_status = "Optimal"

        output.append({
            "employee_name": employee_name,
            "total_hours": round(data["hours"], 2),
            "total_revenue": round(data["revenue"], 2),
            "total_profit": round(data["profit"], 2),
            "margin_pct": margin_pct,
            "utilization_pct": utilization,
            "attendance_pct": attendance_pct,
            "vacation_days": round(data["vacation_days"], 1),
            "leave_days": round(data["leave_days"], 1),
            "working_days": round(data["working_days"], 1),
            "projects": projects,
            "contribution_status": contribution_status,
        })

    output.sort(key=lambda x: x["total_profit"], reverse=True)
    return output


def build_top_performers(records: List[dict], limit: int = 5) -> List[dict]:
    emp_profit = {}
    for r in records:
        name = r.get("employee", "")
        emp_profit[name] = emp_profit.get(name, 0) + (r.get("profit") or 0)
    ranked = sorted(emp_profit.items(), key=lambda x: x[1], reverse=True)
    return [{"employee": name, "total_profit": round(pft, 2)} for name, pft in ranked[:limit]]


def build_risks(records: List[dict]) -> List[dict]:
    risks = []
    for r in records:
        project = r.get("project", "Unknown")
        if (r.get("profit") or 0) < 0:
            risks.append({"employee": r["employee"], "project": project, "month": r.get("month"), "issue": "LOSS_MAKING", "profit": r.get("profit") or 0})
        if (r.get("vacation_days") or 0) >= 3:
            risks.append({"employee": r["employee"], "project": project, "month": r.get("month"), "issue": "HIGH_LEAVE", "vacation_days": r.get("vacation_days") or 0})
        if (r.get("utilisation_pct") or 0) < 80:
            risks.append({"employee": r["employee"], "project": project, "month": r.get("month"), "issue": "LOW_UTILISATION", "utilisation_pct": r.get("utilisation_pct") or 0})
    return risks


def _clean_record_for_api(r: dict) -> dict:
    return {
        "employee": r.get("employee", ""),
        "project": r.get("project", ""),
        "month": r.get("month", ""),
        "actual_hours": r.get("actual_hours") or 0,
        "billable_hours": r.get("billable_hours") or 0,
        "working_days": r.get("working_days") or 0,
        "leave_days": r.get("leave_days") or r.get("vacation_days") or 0,
        "holiday_days": r.get("holiday_days") or 0,
        "vacation_days": r.get("vacation_days") or 0,
        "billing_rate": r.get("billing_rate"),
        "cost_rate": r.get("cost_rate"),
        "revenue": r.get("revenue"),
        "cost": r.get("cost"),
        "profit": r.get("profit"),
        "margin_pct": r.get("margin_pct"),
        "utilisation_pct": r.get("utilisation_pct"),
        "is_profitable": r.get("is_profitable"),
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
