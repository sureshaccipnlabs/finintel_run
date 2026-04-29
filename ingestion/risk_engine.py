"""
risk_engine.py — HR-grade risk analysis and recommendations for FinIntel AI.

Production-ready for HR performance reviews. Covers:

  FINANCIAL RISKS
  ───────────────
  LOSS_MAKING_EMPLOYEE      Employee generating net loss (revenue < cost)
  LOSS_MAKING_PROJECT       Project generating net loss
  LOW_MARGIN_EMPLOYEE       Margin < 15% — fragile, one bad month = loss
  LOW_MARGIN_PROJECT        Project margin < 20%
  RATE_GAP_RISK             billing_rate / cost_rate < 1.15 — no buffer at all
  MISSING_RATES             No billing or cost data — can't compute P&L

  WORKFORCE / HR RISKS
  ────────────────────
  BURNOUT_RISK              Utilisation > 105% for 2+ consecutive months
  FLIGHT_RISK               Leave spike AND declining billable hours — pre-resignation pattern
  DISENGAGEMENT_RISK        Utilisation declining 3+ consecutive months
  HIGH_LEAVE                Leave > 25% of working days in a month
  LOW_UTILISATION           Utilisation < 75% (below productive threshold)
  BENCH_RISK                Utilisation < 50% for 2+ months — allocation failure
  ONBOARDING_DRAIN          New joiner (<3 months) with utilisation < 60%
  BILLING_CEILING_HIT       Billable = approved hours for 3+ months — headcount needed

  POSITIVE SIGNALS (for retention and recognition)
  ─────────────────────────────────────────────────
  STAR_PERFORMER            Margin > 40%, utilisation 85–105%, leave < 15%
  CONSISTENT_PERFORMER      Solid metrics sustained across 3+ months

  PROJECT HEALTH
  ──────────────
  KEY_PERSON_DEPENDENCY     Project revenue > $10k with only 1 employee
  DECLINING_PROJECT         Project margin declining 3+ consecutive months

  TREND SIGNALS
  ─────────────
  DECLINING_MARGIN_TREND    Portfolio margin fell 2+ pts over recent period
  RISING_COST_TREND         Total cost up 10%+ month-on-month
  FALLING_UTILISATION_TREND Avg utilisation dropped 5+ pts

AI layer (optional):
  Uses Ollama (llama3) when available for strategic insights.
  Falls back to rule-based engine output when Ollama is unavailable.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from .dataset import (
    GLOBAL_DATASET,
    build_employee_summaries,
    build_monthly,
    build_overall_summary,
    build_projects,
    filter_by_range,
    get_months_available,
)
from .ai_mapper import _llm_generate, is_llm_available

# ── Constants ─────────────────────────────────────────────────────────────────

HOURS_PER_DAY     = 8.0
DEFAULT_WORK_DAYS = 22          # fallback when working_days not in record

# Thresholds (tweak here — no code changes needed elsewhere)
THRESHOLDS = {
    "margin_healthy":          40.0,   # above → Healthy
    "margin_warning":          20.0,   # above → Warning; below → Risk
    "margin_employee_low":     15.0,   # employee-level low-margin flag
    "rate_gap_min":             1.15,  # billing/cost ratio min before RATE_GAP_RISK
    "utilisation_overload":   105.0,   # above → burnout territory
    "utilisation_optimal_hi": 100.0,
    "utilisation_optimal_lo":  75.0,
    "utilisation_low":         75.0,   # below → LOW_UTILISATION
    "utilisation_bench":       50.0,   # below → BENCH_RISK
    "leave_pct_high":          25.0,   # leave > 25% of working days → HIGH_LEAVE
    "leave_pct_flight":        20.0,   # used in combination with billable drop
    "leave_pct_star":          15.0,   # below → good for STAR
    "billable_drop_pct":       15.0,   # billable hours drop by this % → flight signal
    "onboarding_months":        3,     # months considered "new joiner"
    "onboarding_util_min":     60.0,
    "star_margin_min":         40.0,
    "star_util_min":           85.0,
    "star_util_max":          105.0,
    "revenue_kpd_threshold": 10000.0,  # key-person-dependency revenue floor
    "declining_project_pts":    3.0,   # margin pts decline per month to flag
    "trend_margin_drop":        2.0,   # portfolio margin pts drop to flag
    "trend_cost_rise_pct":     10.0,   # cost % rise to flag
    "trend_util_drop":          5.0,   # utilisation pts drop to flag
}

SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "positive": 4}
PRIORITY_RANK = {"IMMEDIATE": 0, "SHORT_TERM": 1, "LONG_TERM": 2}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _num(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_month_date(month_str: str) -> Optional[date]:
    """Parse any month label into a date object for sorting."""
    import datetime as dt
    if not month_str:
        return None
    for fmt in ("%B %Y", "%B %y", "%b %Y", "%b %y"):
        try:
            return dt.datetime.strptime(str(month_str), fmt).date()
        except ValueError:
            continue
    m = re.match(r"([A-Za-z]+)\D*(\d{2,4})", str(month_str))
    if m:
        name, year = m.group(1), m.group(2)
        if len(year) == 2:
            year = "20" + year
        for fmt in ("%B", "%b"):
            try:
                mon = dt.datetime.strptime(name, fmt).month
                return dt.date(int(year), mon, 1)
            except ValueError:
                continue
    return None


def _sort_months(months: list[str]) -> list[str]:
    return sorted(months, key=lambda m: _parse_month_date(m) or date.min)


def _consecutive_condition(values: list[bool]) -> int:
    """Return length of trailing consecutive True run."""
    count = 0
    for v in reversed(values):
        if v:
            count += 1
        else:
            break
    return count


def _trend_direction(values: list[float], min_delta: float = 0.0) -> str:
    """'improving' | 'declining' | 'stable' based on last 3 values."""
    if len(values) < 2:
        return "stable"
    window = values[-3:] if len(values) >= 3 else values
    delta = window[-1] - window[0]
    if delta > min_delta:
        return "improving"
    if delta < -min_delta:
        return "declining"
    return "stable"


def _money(v: float) -> str:
    return f"${abs(v):,.2f}"


def _pct(v: float) -> str:
    return f"{v:.1f}%"


def _severity_from_score(score: float) -> str:
    if score >= 9:
        return "critical"
    if score >= 7:
        return "high"
    if score >= 4:
        return "medium"
    return "low"


def _priority_from_score(score: float) -> str:
    if score >= 8:
        return "IMMEDIATE"
    if score >= 5:
        return "SHORT_TERM"
    return "LONG_TERM"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Employee timeline builder
# ═════════════════════════════════════════════════════════════════════════════

def _build_employee_timelines(records: list[dict]) -> dict[str, list[dict]]:
    """
    Group and chronologically sort records by employee.
    Returns {employee_name: [record_month1, record_month2, ...]}
    """
    timelines: dict[str, list[dict]] = {}
    for r in records:
        name = r.get("employee") or ""
        if not name or name.lower() in ("unknown", "none"):
            continue
        timelines.setdefault(name, []).append(r)

    for name in timelines:
        timelines[name].sort(
            key=lambda x: _parse_month_date(x.get("month") or "") or date.min
        )
    return timelines


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Composite performance score
# ═════════════════════════════════════════════════════════════════════════════

def employee_performance_score(emp_records: list[dict]) -> dict:
    """
    Compute a 0–100 composite performance score for HR reviews.

    Weights:
      Margin contribution  40%  — how profitable is this person
      Utilisation          30%  — are they working at the right level
      Attendance           30%  — low leave = consistent availability

    Bands:
      80–100 → Star
      60–79  → Solid
      40–59  → Watch
       0–39  → At Risk
    """
    total_rev    = sum(_num(r.get("revenue"))        for r in emp_records)
    total_cost   = sum(_num(r.get("cost"))           for r in emp_records)
    total_hours  = sum(_num(r.get("actual_hours"))   for r in emp_records)
    appr_hours   = sum(_num(r.get("approved_hours") or r.get("max_hours") or r.get("expected_hours")) for r in emp_records)
    leave_days   = sum(_num(r.get("vacation_days"))  for r in emp_records)
    work_days    = sum(_num(r.get("working_days") or DEFAULT_WORK_DAYS) for r in emp_records)

    margin       = ((total_rev - total_cost) / total_rev * 100) if total_rev > 0 else 0.0
    utilisation  = (total_hours / appr_hours * 100) if appr_hours > 0 else (
                   100.0 if total_hours > 0 else 0.0)
    leave_pct    = (leave_days / work_days * 100) if work_days > 0 else 0.0

    # Scores (each 0–100)
    # Margin: 50%+ margin = perfect 100. Negative = 0.
    margin_score = max(0.0, min(margin / 50.0 * 100.0, 100.0))

    # Utilisation: 90–100% = perfect. Below 75 = declining. Above 110 = over-stretched.
    if utilisation >= 90 and utilisation <= 105:
        util_score = 100.0
    elif utilisation > 105:
        util_score = max(0.0, 100.0 - (utilisation - 105) * 3)   # penalise overload
    elif utilisation >= 75:
        util_score = 60.0 + (utilisation - 75) / 15.0 * 40.0
    else:
        util_score = max(0.0, utilisation / 75.0 * 60.0)

    # Attendance: 0% leave = 100. 25%+ leave = 0.
    leave_score = max(0.0, 100.0 - leave_pct * 4.0)

    composite = round(
        margin_score * 0.40 + util_score * 0.30 + leave_score * 0.30, 1
    )

    if composite >= 80:
        band = "Star"
    elif composite >= 60:
        band = "Solid"
    elif composite >= 40:
        band = "Watch"
    else:
        band = "At Risk"

    return {
        "score":     composite,
        "band":      band,
        "breakdown": {
            "margin_score":      round(margin_score, 1),
            "utilisation_score": round(util_score, 1),
            "attendance_score":  round(leave_score, 1),
        },
        "inputs": {
            "margin_pct":     round(margin, 2),
            "utilisation_pct": round(utilisation, 2),
            "leave_pct":      round(leave_pct, 2),
            "months_covered": len(emp_records),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Per-employee risk detectors
# ═════════════════════════════════════════════════════════════════════════════

def _detect_employee_risks(name: str, monthly_records: list[dict], overall_revenue: float) -> list[dict]:
    """
    Run all per-employee risk checks across their full history.
    Returns list of risk dicts, each fully enriched with specifics.
    """
    risks: list[dict] = []

    # Aggregate totals
    total_rev   = sum(_num(r.get("revenue"))      for r in monthly_records)
    total_cost  = sum(_num(r.get("cost"))         for r in monthly_records)
    total_hours = sum(_num(r.get("actual_hours")) for r in monthly_records)
    total_leave = sum(_num(r.get("vacation_days"))for r in monthly_records)
    appr_hours  = sum(_num(r.get("approved_hours") or r.get("max_hours") or r.get("expected_hours")) for r in monthly_records)
    total_profit= total_rev - total_cost
    avg_margin  = (total_profit / total_rev * 100) if total_rev > 0 else 0.0
    avg_util    = (total_hours / appr_hours * 100) if appr_hours > 0 else (
                  100.0 if total_hours > 0 else 0.0)

    # Most recent record for context
    latest      = monthly_records[-1]
    project     = latest.get("project") or "unknown"
    billing_rate= _num(latest.get("billing_rate"))
    cost_rate   = _num(latest.get("cost_rate"))
    revenue_pct = round((total_rev / max(overall_revenue, 1)) * 100, 2)

    # Month-by-month series
    utils       = [_num(r.get("utilisation_pct"))  for r in monthly_records]
    margins     = [_num(r.get("margin_pct"))        for r in monthly_records]
    billables   = [_num(r.get("billable_hours"))    for r in monthly_records]
    leave_pcts  = [
        _num(r.get("vacation_days")) / max(_num(r.get("working_days") or DEFAULT_WORK_DAYS), 1) * 100
        for r in monthly_records
    ]

    months_str  = [r.get("month") or "" for r in monthly_records]
    latest_month= months_str[-1] if months_str else ""
    n           = len(monthly_records)

    # ── 1. LOSS_MAKING_EMPLOYEE ───────────────────────────────────────────
    if total_profit < 0:
        avg_billable = sum(_num(r.get("billable_hours")) for r in monthly_records) / max(n, 1)
        avg_actual   = total_hours / max(n, 1)
        breakeven_rate = round(cost_rate * avg_actual / max(avg_billable, 1), 2) if cost_rate and avg_billable > 0 else None
        risks.append({
            "type":     "LOSS_MAKING_EMPLOYEE",
            "category": "financial",
            "severity": "critical",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} is loss-making: revenue {_money(total_rev)}, "
                f"cost {_money(total_cost)}, net loss {_money(total_profit)}"
            ),
            "recommendation": (
                f"Raise billing rate from ${billing_rate:.0f}/h to at least "
                f"${breakeven_rate:.0f}/h to break even, or transition {name} "
                f"to a higher-margin project."
                if breakeven_rate else
                f"Review {name}'s cost structure on {project} — currently losing {_money(total_profit)}."
            ),
            "owner":    "Finance Controller + Project Manager",
            "deadline": "Before next billing cycle",
            "metrics": {
                "total_revenue":  round(total_rev, 2),
                "total_cost":     round(total_cost, 2),
                "total_profit":   round(total_profit, 2),
                "billing_rate":   billing_rate,
                "cost_rate":      cost_rate,
                "months_covered": n,
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 2. LOW_MARGIN_EMPLOYEE ────────────────────────────────────────────
    elif avg_margin < THRESHOLDS["margin_employee_low"] and total_rev > 0:
        gap = THRESHOLDS["margin_employee_low"] - avg_margin
        risks.append({
            "type":     "LOW_MARGIN_EMPLOYEE",
            "category": "financial",
            "severity": "high" if avg_margin < 10 else "medium",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} margin is {_pct(avg_margin)} (target >{_pct(THRESHOLDS['margin_employee_low'])}) "
                f"on {project}"
            ),
            "recommendation": (
                f"Increase {name}'s billing rate by ~{_pct(gap + 5)} "
                f"(currently ${billing_rate:.0f}/h, cost rate ${cost_rate:.0f}/h) "
                f"to reach a healthy margin. Expected profit recovery: "
                f"~{_money(total_rev * (gap / 100))} over the same period."
            ),
            "owner":    "Account Manager + Finance",
            "deadline": "Next contract renewal or rate review",
            "metrics": {
                "avg_margin_pct": round(avg_margin, 2),
                "target_margin":  THRESHOLDS["margin_employee_low"],
                "gap_pct":        round(gap, 2),
                "billing_rate":   billing_rate,
                "cost_rate":      cost_rate,
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 3. RATE_GAP_RISK ─────────────────────────────────────────────────
    if billing_rate > 0 and cost_rate > 0:
        rate_ratio = billing_rate / cost_rate
        if rate_ratio < THRESHOLDS["rate_gap_min"] and total_profit >= 0:  # not yet a loss
            buffer_pct = (rate_ratio - 1) * 100
            risks.append({
                "type":     "RATE_GAP_RISK",
                "category": "financial",
                "severity": "medium",
                "entity":   name,
                "project":  project,
                "description": (
                    f"{name} has only {_pct(buffer_pct)} rate buffer "
                    f"(billing ${billing_rate:.0f}/h vs cost ${cost_rate:.0f}/h) — "
                    f"one sick day or small rate change causes a loss"
                ),
                "recommendation": (
                    f"Negotiate billing rate for {name} from ${billing_rate:.0f}/h to "
                    f"at least ${cost_rate * THRESHOLDS['rate_gap_min']:.0f}/h "
                    f"to maintain a minimum {_pct((THRESHOLDS['rate_gap_min']-1)*100)} buffer."
                ),
                "owner":    "Account Manager",
                "deadline": "Next rate review",
                "metrics": {
                    "billing_rate": billing_rate,
                    "cost_rate":    cost_rate,
                    "rate_ratio":   round(rate_ratio, 3),
                    "buffer_pct":   round(buffer_pct, 2),
                },
                "linked_employees": [name],
                "revenue_contribution_pct": revenue_pct,
            })

    # ── 4. MISSING_RATES ─────────────────────────────────────────────────
    if billing_rate == 0 or cost_rate == 0:
        missing = []
        if billing_rate == 0: missing.append("billing rate")
        if cost_rate == 0:    missing.append("cost rate")
        risks.append({
            "type":     "MISSING_RATES",
            "category": "financial",
            "severity": "medium",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} is missing {' and '.join(missing)} — P&L cannot be computed"
            ),
            "recommendation": (
                f"Add {' and '.join(missing)} for {name} in the timesheet "
                f"to enable profit/loss tracking. Without this, financial risk "
                f"for {project} is invisible."
            ),
            "owner":    "HR / Finance data owner",
            "deadline": "Before end of current reporting period",
            "metrics": {"missing_fields": missing},
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 5. BURNOUT_RISK ───────────────────────────────────────────────────
    overload_flags = [u > THRESHOLDS["utilisation_overload"] for u in utils]
    consecutive_overload = _consecutive_condition(overload_flags)
    if consecutive_overload >= 2:
        avg_overload = sum(u for u in utils[-consecutive_overload:]) / consecutive_overload
        risks.append({
            "type":     "BURNOUT_RISK",
            "category": "workforce",
            "severity": "high" if consecutive_overload >= 3 else "medium",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} has been over {_pct(THRESHOLDS['utilisation_overload'])} "
                f"utilised for {consecutive_overload} consecutive months "
                f"(avg {_pct(avg_overload)})"
            ),
            "recommendation": (
                f"Schedule a workload review for {name} with their manager immediately. "
                f"Consider redistributing tasks or adding a resource to {project}. "
                f"Overloaded employees at this level are 3× more likely to resign within 90 days."
            ),
            "owner":    "Direct Manager + HR Business Partner",
            "deadline": "Within 2 weeks",
            "metrics": {
                "consecutive_overload_months": consecutive_overload,
                "avg_utilisation_pct": round(avg_overload, 1),
                "months": months_str[-consecutive_overload:],
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 6. FLIGHT_RISK ────────────────────────────────────────────────────
    if n >= 2:
        recent_leave_pct  = leave_pcts[-1] if leave_pcts else 0
        prev_billable     = billables[-2] if len(billables) >= 2 else 0
        curr_billable     = billables[-1] if billables else 0
        billable_drop     = ((prev_billable - curr_billable) / max(prev_billable, 1) * 100
                             if prev_billable > 0 else 0)

        if (recent_leave_pct > THRESHOLDS["leave_pct_flight"]
                and billable_drop > THRESHOLDS["billable_drop_pct"]):
            risks.append({
                "type":     "FLIGHT_RISK",
                "category": "workforce",
                "severity": "high",
                "entity":   name,
                "project":  project,
                "description": (
                    f"{name} shows flight risk pattern: leave spiked to "
                    f"{_pct(recent_leave_pct)} in {latest_month} while billable hours "
                    f"dropped {_pct(billable_drop)} vs prior month"
                ),
                "recommendation": (
                    f"HR Business Partner to schedule a confidential 1:1 with {name} "
                    f"within 7 days. Check for engagement, compensation, or workload issues. "
                    f"Replacing {name} would cost an estimated "
                    f"{_money(total_rev / max(n, 1) * 3)} (3 months revenue equivalent)."
                ),
                "owner":    "HR Business Partner",
                "deadline": "Within 7 days — do not delay",
                "metrics": {
                    "leave_pct_latest":   round(recent_leave_pct, 1),
                    "billable_drop_pct":  round(billable_drop, 1),
                    "latest_month":       latest_month,
                    "estimated_replacement_cost": round(total_rev / max(n, 1) * 3, 2),
                },
                "linked_employees": [name],
                "revenue_contribution_pct": revenue_pct,
            })

    # ── 7. DISENGAGEMENT_RISK ─────────────────────────────────────────────
    if n >= 3:
        util_trend = _trend_direction(utils[-3:], min_delta=3.0)
        if util_trend == "declining":
            drop = utils[-3] - utils[-1]
            risks.append({
                "type":     "DISENGAGEMENT_RISK",
                "category": "workforce",
                "severity": "medium",
                "entity":   name,
                "project":  project,
                "description": (
                    f"{name} utilisation has declined {_pct(drop)} over the last 3 months "
                    f"({_pct(utils[-3])} → {_pct(utils[-1])}) — progressive disengagement pattern"
                ),
                "recommendation": (
                    f"Manager to have a structured career conversation with {name}. "
                    f"Review project fit, growth opportunities, and compensation benchmarking. "
                    f"Current trend: -{_pct(drop)} utilisation over 3 months."
                ),
                "owner":    "Direct Manager",
                "deadline": "Within 30 days",
                "metrics": {
                    "util_series":    [round(u, 1) for u in utils[-3:]],
                    "total_drop_pct": round(drop, 1),
                    "months":         months_str[-3:],
                },
                "linked_employees": [name],
                "revenue_contribution_pct": revenue_pct,
            })

    # ── 8. HIGH_LEAVE (contextual %) ──────────────────────────────────────
    for i, r in enumerate(monthly_records):
        leave_d   = _num(r.get("vacation_days"))
        work_d    = _num(r.get("working_days") or DEFAULT_WORK_DAYS)
        month_lbl = r.get("month") or ""
        lv_pct    = (leave_d / work_d * 100) if work_d > 0 else 0
        if lv_pct > THRESHOLDS["leave_pct_high"]:
            rev_lost = leave_d * HOURS_PER_DAY * billing_rate if billing_rate else None
            risks.append({
                "type":     "HIGH_LEAVE",
                "category": "workforce",
                "severity": "high" if lv_pct > 40 else "medium",
                "entity":   name,
                "project":  project,
                "description": (
                    f"{name} took {leave_d:.0f} leave days in {month_lbl} "
                    f"({_pct(lv_pct)} of {work_d:.0f} working days)"
                ),
                "recommendation": (
                    f"Review whether {name}'s {leave_d:.0f} days leave in {month_lbl} "
                    f"was planned/approved. "
                    + (f"Estimated billing impact: {_money(rev_lost)}. " if rev_lost else "")
                    + "If unplanned, check for wellbeing concerns — this may indicate FLIGHT_RISK."
                ),
                "owner":    "Direct Manager + HR",
                "deadline": "Review in next 1:1",
                "metrics": {
                    "leave_days":   leave_d,
                    "working_days": work_d,
                    "leave_pct":    round(lv_pct, 1),
                    "month":        month_lbl,
                    "estimated_revenue_impact": round(rev_lost, 2) if rev_lost else None,
                },
                "linked_employees": [name],
                "revenue_contribution_pct": revenue_pct,
            })

    # ── 9. LOW_UTILISATION ────────────────────────────────────────────────
    if avg_util < THRESHOLDS["utilisation_low"] and avg_util > 0:
        hours_gap  = max(0, appr_hours * (THRESHOLDS["utilisation_low"] / 100) - total_hours)
        rev_gap    = hours_gap * billing_rate if billing_rate else 0
        risks.append({
            "type":     "LOW_UTILISATION",
            "category": "operational",
            "severity": "high" if avg_util < THRESHOLDS["utilisation_bench"] else "medium",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} averaged {_pct(avg_util)} utilisation on {project} "
                f"(target ≥{_pct(THRESHOLDS['utilisation_low'])})"
            ),
            "recommendation": (
                f"Reallocate {name} from {project} (current utilisation {_pct(avg_util)}) "
                f"to projects with open approved headcount. "
                + (f"Potential revenue recovery: ~{_money(rev_gap)} per period." if rev_gap else "")
            ),
            "owner":    "Resource Manager",
            "deadline": "By next sprint planning",
            "metrics": {
                "avg_utilisation_pct": round(avg_util, 1),
                "target_pct":          THRESHOLDS["utilisation_low"],
                "hours_gap":           round(hours_gap, 1),
                "estimated_rev_gap":   round(rev_gap, 2),
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 10. BENCH_RISK ────────────────────────────────────────────────────
    bench_flags = [u < THRESHOLDS["utilisation_bench"] for u in utils]
    consecutive_bench = _consecutive_condition(bench_flags)
    if consecutive_bench >= 2:
        bench_cost = sum(
            _num(r.get("cost")) for r in monthly_records[-consecutive_bench:]
        )
        risks.append({
            "type":     "BENCH_RISK",
            "category": "operational",
            "severity": "high",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} has been under {_pct(THRESHOLDS['utilisation_bench'])} "
                f"utilised for {consecutive_bench} months — "
                f"paying full cost with minimal billing ({_money(bench_cost)} cost with low output)"
            ),
            "recommendation": (
                f"Urgently find billable allocation for {name} — "
                f"{consecutive_bench} months on bench has cost {_money(bench_cost)} "
                f"with minimal revenue return. Options: (1) assign to open project, "
                f"(2) upskill for higher-demand role, (3) review headcount need."
            ),
            "owner":    "Resource Manager + Finance",
            "deadline": "IMMEDIATE — next week",
            "metrics": {
                "consecutive_bench_months": consecutive_bench,
                "avg_utilisation_pct": round(avg_util, 1),
                "bench_cost": round(bench_cost, 2),
                "months": months_str[-consecutive_bench:],
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 11. BILLING_CEILING_HIT ───────────────────────────────────────────
    if n >= 3:
        ceiling_flags = [
            abs(_num(r.get("billable_hours")) - _num(r.get("approved_hours") or r.get("max_hours") or r.get("expected_hours"))) < 2
            and _num(r.get("approved_hours") or r.get("max_hours") or r.get("expected_hours")) > 0
            for r in monthly_records
        ]
        consecutive_ceiling = _consecutive_condition(ceiling_flags)
        if consecutive_ceiling >= 3:
            risks.append({
                "type":     "BILLING_CEILING_HIT",
                "category": "operational",
                "severity": "medium",
                "entity":   name,
                "project":  project,
                "description": (
                    f"{name} has hit their approved billing ceiling every month for "
                    f"{consecutive_ceiling} months — demand exceeds current allocation"
                ),
                "recommendation": (
                    f"Increase {name}'s approved hours on {project} or add a supporting resource. "
                    f"Consistently capping at approved hours for {consecutive_ceiling} months "
                    f"suggests unmet demand that may be going to competitors or causing delivery risk."
                ),
                "owner":    "Project Manager + HR",
                "deadline": "Next headcount review",
                "metrics": {
                    "consecutive_ceiling_months": consecutive_ceiling,
                    "approved_hours_per_month": _num(latest.get("approved_hours") or latest.get("max_hours") or latest.get("expected_hours")),
                },
                "linked_employees": [name],
                "revenue_contribution_pct": revenue_pct,
            })

    # ── 12. ONBOARDING_DRAIN ──────────────────────────────────────────────
    onboard_raw = latest.get("onboard_date")
    if onboard_raw and n <= THRESHOLDS["onboarding_months"]:
        if avg_util < THRESHOLDS["onboarding_util_min"]:
            risks.append({
                "type":     "ONBOARDING_DRAIN",
                "category": "workforce",
                "severity": "low",
                "entity":   name,
                "project":  project,
                "description": (
                    f"{name} is a new joiner ({n} month(s) on record) with "
                    f"only {_pct(avg_util)} utilisation — below expected onboarding ramp"
                ),
                "recommendation": (
                    f"Check {name}'s onboarding progress — {_pct(avg_util)} utilisation "
                    f"at {n} month(s) is below the {_pct(THRESHOLDS['onboarding_util_min'])} "
                    f"target. Assign a buddy and review project readiness checklist."
                ),
                "owner":    "HR + Onboarding Manager",
                "deadline": "Within 2 weeks",
                "metrics": {
                    "months_on_record": n,
                    "avg_utilisation_pct": round(avg_util, 1),
                    "target_pct": THRESHOLDS["onboarding_util_min"],
                },
                "linked_employees": [name],
                "revenue_contribution_pct": revenue_pct,
            })

    # ── 13. STAR_PERFORMER (positive signal) ──────────────────────────────
    if (avg_margin >= THRESHOLDS["star_margin_min"]
            and THRESHOLDS["star_util_min"] <= avg_util <= THRESHOLDS["star_util_max"]
            and all(lp < THRESHOLDS["leave_pct_star"] for lp in leave_pcts)):
        perf = employee_performance_score(monthly_records)
        risks.append({
            "type":     "STAR_PERFORMER",
            "category": "workforce",
            "severity": "positive",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} is a star performer: {_pct(avg_margin)} margin, "
                f"{_pct(avg_util)} utilisation, consistent attendance — "
                f"performance score {perf['score']}/100"
            ),
            "recommendation": (
                f"Prioritise retention for {name}: review compensation benchmarking, "
                f"career progression, and recognition. Their contribution generates "
                f"~{_money(total_profit)} profit over {n} month(s). "
                f"At-risk replacement cost: ~{_money(total_rev / max(n,1) * 4)}."
            ),
            "owner":    "HR Business Partner + Senior Management",
            "deadline": "Include in next quarterly review",
            "metrics": {
                "avg_margin_pct":     round(avg_margin, 2),
                "avg_utilisation_pct": round(avg_util, 2),
                "total_profit":       round(total_profit, 2),
                "performance_score":  perf["score"],
                "performance_band":   perf["band"],
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    # ── 14. CONSISTENT_PERFORMER (positive) ───────────────────────────────
    elif (n >= 3
          and avg_margin >= THRESHOLDS["margin_warning"]
          and avg_util >= THRESHOLDS["utilisation_optimal_lo"]
          and "STAR_PERFORMER" not in {r["type"] for r in risks}):
        risks.append({
            "type":     "CONSISTENT_PERFORMER",
            "category": "workforce",
            "severity": "positive",
            "entity":   name,
            "project":  project,
            "description": (
                f"{name} has been consistently solid for {n} months: "
                f"{_pct(avg_margin)} margin, {_pct(avg_util)} utilisation"
            ),
            "recommendation": (
                f"Acknowledge {name}'s consistent performance in their next review. "
                f"Consider for mentoring role or project lead responsibilities."
            ),
            "owner":    "Direct Manager",
            "deadline": "Next quarterly review",
            "metrics": {
                "avg_margin_pct":      round(avg_margin, 2),
                "avg_utilisation_pct": round(avg_util, 2),
                "months_covered":      n,
            },
            "linked_employees": [name],
            "revenue_contribution_pct": revenue_pct,
        })

    return risks


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Project-level risk detectors
# ═════════════════════════════════════════════════════════════════════════════

def _detect_project_risks(records: list[dict]) -> list[dict]:
    """Detect project-level financial and operational risks."""
    risks: list[dict] = []
    projects_data = build_projects(records)
    overall_rev   = sum(p["revenue"] for p in projects_data.values())

    # Build per-project monthly series for trend detection
    proj_monthly: dict[str, dict[str, dict]] = {}
    for r in records:
        proj  = r.get("project") or "Unknown"
        month = r.get("month")   or "Unknown"
        proj_monthly.setdefault(proj, {}).setdefault(month, {"rev": 0.0, "cost": 0.0, "emps": set()})
        proj_monthly[proj][month]["rev"]  += _num(r.get("revenue"))
        proj_monthly[proj][month]["cost"] += _num(r.get("cost"))
        proj_monthly[proj][month]["emps"].add(r.get("employee") or "")

    for proj_name, pdata in projects_data.items():
        rev    = pdata["revenue"]
        cost   = pdata["cost"]
        profit = pdata["profit"]
        n_emps = pdata["employees"]
        margin = (profit / rev * 100) if rev > 0 else 0.0
        rev_pct= round(rev / max(overall_rev, 1) * 100, 2)

        # Monthly series for this project (sorted)
        monthly_sorted = _sort_months(list(proj_monthly.get(proj_name, {}).keys()))
        margin_series  = [
            ((proj_monthly[proj_name][m]["rev"] - proj_monthly[proj_name][m]["cost"])
             / max(proj_monthly[proj_name][m]["rev"], 1) * 100)
            for m in monthly_sorted
            if proj_monthly[proj_name][m]["rev"] > 0
        ]

        # LOSS_MAKING_PROJECT
        if profit < 0:
            risks.append({
                "type":     "LOSS_MAKING_PROJECT",
                "category": "financial",
                "severity": "critical",
                "entity":   proj_name,
                "project":  proj_name,
                "description": (
                    f"Project {proj_name} is loss-making: "
                    f"revenue {_money(rev)}, cost {_money(cost)}, loss {_money(profit)}"
                ),
                "recommendation": (
                    f"Immediate profitability review for {proj_name}: "
                    f"(1) audit staffing mix and reduce high-cost resources, "
                    f"(2) renegotiate client billing rates, "
                    f"(3) review scope — consider pausing or renegotiating if loss exceeds {_money(abs(profit))}."
                ),
                "owner":    "Project Manager + Finance Controller",
                "deadline": "Within 5 business days",
                "metrics":  {"revenue": round(rev,2), "cost": round(cost,2),
                             "profit": round(profit,2), "employees": n_emps},
                "linked_employees": list(proj_monthly.get(proj_name, {}).get(monthly_sorted[-1] if monthly_sorted else "", {}).get("emps", [])),
                "revenue_contribution_pct": rev_pct,
            })

        # LOW_MARGIN_PROJECT
        elif 0 < margin < THRESHOLDS["margin_warning"]:
            risks.append({
                "type":     "LOW_MARGIN_PROJECT",
                "category": "financial",
                "severity": "high" if margin < 10 else "medium",
                "entity":   proj_name,
                "project":  proj_name,
                "description": (
                    f"Project {proj_name} margin is {_pct(margin)} "
                    f"(target ≥{_pct(THRESHOLDS['margin_warning'])})"
                ),
                "recommendation": (
                    f"Review {proj_name} staffing rates and client pricing. "
                    f"A {_pct(5)} rate increase across {n_emps} employees would recover "
                    f"~{_money(rev * 0.05)} in revenue."
                ),
                "owner":    "Account Manager + Finance",
                "deadline": "Next contract review",
                "metrics":  {"margin_pct": round(margin,2), "revenue": round(rev,2),
                             "profit": round(profit,2), "employees": n_emps},
                "linked_employees": [],
                "revenue_contribution_pct": rev_pct,
            })

        # DECLINING_PROJECT
        if len(margin_series) >= 3:
            proj_trend = _trend_direction(margin_series, min_delta=THRESHOLDS["declining_project_pts"])
            if proj_trend == "declining":
                drop = margin_series[-3] - margin_series[-1]
                risks.append({
                    "type":     "DECLINING_PROJECT",
                    "category": "financial",
                    "severity": "high",
                    "entity":   proj_name,
                    "project":  proj_name,
                    "description": (
                        f"Project {proj_name} margin declining: "
                        f"{_pct(margin_series[-3])} → {_pct(margin_series[-1])} "
                        f"over last 3 months (fell {_pct(drop)})"
                    ),
                    "recommendation": (
                        f"Investigate root cause of {proj_name}'s declining margin. "
                        f"Check for: scope creep, rate erosion, increased headcount costs. "
                        f"At current trend, project could be loss-making within "
                        f"{max(1, int(margin_series[-1] / max(drop/3, 0.1))):.0f} months."
                    ),
                    "owner":    "Project Manager",
                    "deadline": "Review in next steering committee",
                    "metrics":  {"margin_series": [round(m,1) for m in margin_series[-3:]],
                                 "months": monthly_sorted[-3:],
                                 "drop_pct": round(drop,1)},
                    "linked_employees": [],
                    "revenue_contribution_pct": rev_pct,
                })

        # KEY_PERSON_DEPENDENCY
        if n_emps == 1 and rev > THRESHOLDS["revenue_kpd_threshold"]:
            risks.append({
                "type":     "KEY_PERSON_DEPENDENCY",
                "category": "operational",
                "severity": "medium",
                "entity":   proj_name,
                "project":  proj_name,
                "description": (
                    f"Project {proj_name} has {_money(rev)} revenue "
                    f"dependent on a single employee"
                ),
                "recommendation": (
                    f"Add a backup or shadow resource to {proj_name} to reduce "
                    f"single-point-of-failure risk on {_money(rev)} revenue. "
                    f"Document key knowledge and client relationships."
                ),
                "owner":    "Project Manager + HR",
                "deadline": "Next headcount planning cycle",
                "metrics":  {"revenue": round(rev,2), "employee_count": n_emps},
                "linked_employees": [],
                "revenue_contribution_pct": rev_pct,
            })

    return risks


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Portfolio trend signals
# ═════════════════════════════════════════════════════════════════════════════

def _detect_trend_risks(records: list[dict]) -> list[dict]:
    """Detect portfolio-level trends across months."""
    risks: list[dict] = []
    monthly      = build_monthly(records)
    months       = _sort_months(list(monthly.keys()))
    if len(months) < 2:
        return risks

    recent = months[-3:] if len(months) >= 3 else months

    # Margin trend
    margins = [monthly[m]["avg_margin_pct"] for m in recent]
    if _trend_direction(margins, THRESHOLDS["trend_margin_drop"]) == "declining":
        drop = margins[0] - margins[-1]
        risks.append({
            "type":     "DECLINING_MARGIN_TREND",
            "category": "financial",
            "severity": "high",
            "entity":   "Portfolio",
            "project":  "All projects",
            "description": (
                f"Portfolio margin declined {_pct(drop)} over {len(recent)} months "
                f"({_pct(margins[0])} → {_pct(margins[-1])})"
            ),
            "recommendation": (
                f"Portfolio-wide margin compression requires a pricing and cost review. "
                f"Priority: identify projects driving the {_pct(drop)} drop and renegotiate rates."
            ),
            "owner":    "Finance Director",
            "deadline": "Next monthly business review",
            "metrics":  {"margin_series": [round(m,1) for m in margins], "months": recent},
            "linked_employees": [],
            "revenue_contribution_pct": 100.0,
        })

    # Cost trend
    costs = [monthly[m]["total_cost"] for m in recent]
    if len(costs) >= 2 and costs[0] > 0:
        cost_growth = (costs[-1] - costs[0]) / costs[0] * 100
        if cost_growth >= THRESHOLDS["trend_cost_rise_pct"]:
            risks.append({
                "type":     "RISING_COST_TREND",
                "category": "financial",
                "severity": "medium",
                "entity":   "Portfolio",
                "project":  "All projects",
                "description": (
                    f"Total cost rose {_pct(cost_growth)} over {len(recent)} months "
                    f"({_money(costs[0])} → {_money(costs[-1])})"
                ),
                "recommendation": (
                    f"Audit cost drivers — a {_pct(cost_growth)} cost rise over {len(recent)} months "
                    f"needs explanation. Check for: new hires, rate increases, or overhead allocation changes."
                ),
                "owner":    "Finance Controller",
                "deadline": "By end of month",
                "metrics":  {"cost_series": [round(c,2) for c in costs],
                             "cost_growth_pct": round(cost_growth,1), "months": recent},
                "linked_employees": [],
                "revenue_contribution_pct": 100.0,
            })

    # Utilisation trend
    util_series = []
    for m in recent:
        recs_m = [r for r in records if r.get("month") == m]
        utils  = [_num(r.get("utilisation_pct")) for r in recs_m if r.get("utilisation_pct")]
        if utils:
            util_series.append(sum(utils) / len(utils))

    if len(util_series) >= 2:
        util_drop = util_series[0] - util_series[-1]
        if util_drop >= THRESHOLDS["trend_util_drop"]:
            risks.append({
                "type":     "FALLING_UTILISATION_TREND",
                "category": "operational",
                "severity": "medium",
                "entity":   "Portfolio",
                "project":  "All projects",
                "description": (
                    f"Average utilisation fell {_pct(util_drop)} over {len(recent)} months "
                    f"({_pct(util_series[0])} → {_pct(util_series[-1])})"
                ),
                "recommendation": (
                    f"Investigate allocation gaps across the portfolio. "
                    f"A {_pct(util_drop)} utilisation drop means growing bench cost. "
                    f"Review pipeline and accelerate resourcing on upcoming projects."
                ),
                "owner":    "Resource Manager",
                "deadline": "Next resource planning meeting",
                "metrics":  {"util_series": [round(u,1) for u in util_series],
                             "drop_pct": round(util_drop,1), "months": recent},
                "linked_employees": [],
                "revenue_contribution_pct": 100.0,
            })

    return risks


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Risk scoring and sorting
# ═════════════════════════════════════════════════════════════════════════════

def _score_risk(risk: dict) -> float:
    """Compute a 0–10 priority score for sorting."""
    severity = (risk.get("severity") or "medium").lower()
    base     = {"critical": 9.5, "high": 7.5, "medium": 5.0, "low": 2.5, "positive": 0.0}.get(severity, 5.0)
    t        = risk.get("type", "")
    m        = risk.get("metrics") or {}

    bonus = 0.0
    # Revenue weight bonus
    rev_pct = _num(risk.get("revenue_contribution_pct"))
    if rev_pct >= 20:
        bonus += 1.0
    elif rev_pct >= 10:
        bonus += 0.5

    # Type-specific bonuses
    if t == "BURNOUT_RISK":
        months = _num(m.get("consecutive_overload_months"))
        bonus += min(months * 0.5, 1.5)
    if t in ("LOW_MARGIN_EMPLOYEE", "LOW_MARGIN_PROJECT"):
        margin = _num(m.get("avg_margin_pct") or m.get("margin_pct"), 20)
        if margin < 5:
            bonus += 1.5
        elif margin < 10:
            bonus += 0.75
    if t == "LOSS_MAKING_EMPLOYEE":
        bonus += 1.0
    if t == "FLIGHT_RISK":
        bonus += 1.0   # retention cost is high
    if t == "BENCH_RISK":
        bench_cost = _num(m.get("bench_cost"))
        if bench_cost > 20000:
            bonus += 1.0

    return min(10.0, round(base + bonus, 2))


def _sort_risks(risks: list[dict]) -> list[dict]:
    """Sort: positive signals last, then by score desc."""
    def key(r):
        is_positive = r.get("severity") == "positive"
        return (1 if is_positive else 0, -r.get("_score", 0))
    return sorted(risks, key=key)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Build summary structures for API response
# ═════════════════════════════════════════════════════════════════════════════

def _build_risk_overview(risks: list[dict]) -> dict:
    by_sev = {"critical": 0, "high": 0, "medium": 0, "low": 0, "positive": 0}
    by_cat = {"financial": 0, "workforce": 0, "operational": 0, "general": 0}
    for r in risks:
        sev = r.get("severity", "medium")
        cat = r.get("category", "general")
        by_sev[sev] = by_sev.get(sev, 0) + 1
        by_cat[cat] = by_cat.get(cat, 0) + 1
    action_needed = by_sev["critical"] + by_sev["high"]
    return {
        "total_risks":        len(risks),
        "action_needed":      action_needed,
        "by_severity":        by_sev,
        "by_category":        by_cat,
        "highest_priority":   risks[0]["type"] if risks else None,
    }


def _build_employee_scorecards(timelines: dict[str, list[dict]]) -> list[dict]:
    """Return a performance scorecard for every employee."""
    cards = []
    for name, records in timelines.items():
        perf = employee_performance_score(records)
        latest = records[-1]
        cards.append({
            "employee":        name,
            "project":         latest.get("project") or "Unknown",
            "months_covered":  len(records),
            "latest_month":    latest.get("month") or "",
            "performance":     perf,
            "total_revenue":   round(sum(_num(r.get("revenue")) for r in records), 2),
            "total_profit":    round(sum(_num(r.get("revenue")) - _num(r.get("cost")) for r in records), 2),
            "avg_utilisation": round(sum(_num(r.get("utilisation_pct")) for r in records) / max(len(records),1), 1),
        })
    cards.sort(key=lambda c: c["performance"]["score"], reverse=True)
    return cards


def _risk_impact_value(risk: dict) -> float:
    """Best-effort monetary impact estimate from risk metrics."""
    t = risk.get("type", "")
    m = risk.get("metrics") or {}

    if t == "LOSS_MAKING_PROJECT":
        return abs(_num(m.get("profit")))
    if t == "LOSS_MAKING_EMPLOYEE":
        return abs(_num(m.get("total_profit") or m.get("profit")))
    if t == "BENCH_RISK":
        return abs(_num(m.get("bench_cost")))
    if t == "LOW_UTILISATION":
        return abs(_num(m.get("estimated_rev_gap")))
    if t == "HIGH_LEAVE":
        return abs(_num(m.get("estimated_revenue_impact")))
    return 0.0


def _build_financial_exposure(risks: list[dict]) -> dict:
    """Aggregate monetary exposure estimates for executive reporting."""
    components = {
        "loss_making_projects": 0.0,
        "loss_making_employees": 0.0,
        "bench_cost": 0.0,
        "low_utilisation_gap": 0.0,
        "high_leave_impact": 0.0,
    }

    for r in risks:
        t = r.get("type", "")
        m = r.get("metrics") or {}
        if t == "LOSS_MAKING_PROJECT":
            components["loss_making_projects"] += abs(_num(m.get("profit")))
        elif t == "LOSS_MAKING_EMPLOYEE":
            components["loss_making_employees"] += abs(_num(m.get("total_profit") or m.get("profit")))
        elif t == "BENCH_RISK":
            components["bench_cost"] += abs(_num(m.get("bench_cost")))
        elif t == "LOW_UTILISATION":
            components["low_utilisation_gap"] += abs(_num(m.get("estimated_rev_gap")))
        elif t == "HIGH_LEAVE":
            components["high_leave_impact"] += abs(_num(m.get("estimated_revenue_impact")))

    total = round(sum(components.values()), 2)
    return {
        "estimated_total": total,
        "components": {k: round(v, 2) for k, v in components.items()},
    }


def _build_project_risk_heatmap(risks: list[dict]) -> list[dict]:
    """Risk concentration by project for HR/leadership triage."""
    buckets: dict[str, dict] = {}
    for r in risks:
        if r.get("severity") == "positive":
            continue
        project = (r.get("project") or "Unknown").strip() or "Unknown"
        sev = r.get("severity", "medium")
        if project not in buckets:
            buckets[project] = {
                "project": project,
                "total_risks": 0,
                "critical": 0,
                "high": 0,
                "medium": 0,
                "low": 0,
            }
        buckets[project]["total_risks"] += 1
        if sev in ("critical", "high", "medium", "low"):
            buckets[project][sev] += 1

    rows = list(buckets.values())
    rows.sort(key=lambda x: (x["critical"], x["high"], x["total_risks"]), reverse=True)
    return rows


def _build_data_quality_summary(records: list[dict]) -> dict:
    """Compute data completeness and confidence for risk interpretation."""
    if not records:
        return {
            "confidence": "LOW",
            "completeness_pct": 0.0,
            "missing_fields": {},
            "notes": "No records available for risk analysis.",
        }

    tracked_fields = [
        "employee", "project", "month", "actual_hours",
        "billing_rate", "cost_rate", "revenue", "cost", "profit", "utilisation_pct",
    ]

    missing: dict[str, int] = {f: 0 for f in tracked_fields}
    for r in records:
        for f in tracked_fields:
            v = r.get(f)
            if v is None or v == "":
                missing[f] += 1

    total_cells = len(records) * len(tracked_fields)
    missing_cells = sum(missing.values())
    completeness = (1.0 - (missing_cells / max(total_cells, 1))) * 100

    if completeness >= 95:
        confidence = "HIGH"
    elif completeness >= 85:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    top_missing = {k: v for k, v in sorted(missing.items(), key=lambda kv: kv[1], reverse=True) if v > 0}

    return {
        "confidence": confidence,
        "completeness_pct": round(completeness, 1),
        "missing_fields": top_missing,
        "notes": (
            "Risk confidence is derived from completeness of critical financial and utilisation fields."
        ),
    }


def _build_action_tracker(recommendations: list[dict]) -> list[dict]:
    """Transform recommendations into trackable execution items."""
    target_days = {"IMMEDIATE": 7, "SHORT_TERM": 30, "LONG_TERM": 90}
    tracker = []
    for i, rec in enumerate(recommendations, start=1):
        pr = rec.get("priority", "SHORT_TERM")
        tracker.append({
            "id": f"ACT-{i:03d}",
            "action": rec.get("action", ""),
            "owner": rec.get("owner", "TBD"),
            "category": rec.get("category", "general"),
            "priority": pr,
            "status": "OPEN",
            "target_days": target_days.get(pr, 30),
            "related_risk_type": rec.get("related_risk_type", ""),
        })
    return tracker


def _build_explainability_sources(risks: list[dict], limit: int = 12) -> list[dict]:
    """Provide compact evidence snippets for top risks."""
    sources = []
    for r in risks[:limit]:
        m = r.get("metrics") or {}
        evidence = {}
        for k in (
            "total_profit", "profit", "avg_margin_pct", "margin_pct", "avg_utilisation_pct",
            "consecutive_overload_months", "consecutive_bench_months", "leave_pct_latest",
            "billable_drop_pct", "bench_cost", "estimated_rev_gap", "months",
        ):
            if k in m and m.get(k) is not None:
                evidence[k] = m.get(k)

        sources.append({
            "risk_type": r.get("type", ""),
            "entity": r.get("entity", ""),
            "project": r.get("project", ""),
            "severity": r.get("severity", ""),
            "evidence": evidence,
        })
    return sources


def _build_executive_summary(risks: list[dict], recommendations: list[dict]) -> dict:
    """Build C-level summary: highest risks, exposure, and execution pressure."""
    non_positive = [r for r in risks if r.get("severity") != "positive"]
    critical_high = [r for r in non_positive if r.get("severity") in ("critical", "high")]

    top_items = []
    for r in critical_high[:5]:
        top_items.append({
            "type": r.get("type", ""),
            "entity": r.get("entity", ""),
            "project": r.get("project", ""),
            "severity": r.get("severity", ""),
            "priority": r.get("priority", ""),
            "owner": r.get("owner", ""),
            "deadline": r.get("deadline", ""),
            "estimated_impact": round(_risk_impact_value(r), 2),
        })

    repeated = {}
    for r in non_positive:
        key = f"{r.get('type','')}::{r.get('entity','')}"
        repeated[key] = repeated.get(key, 0) + 1
    recurring_count = sum(1 for _, c in repeated.items() if c > 1)

    return {
        "top_critical_actions": top_items,
        "financial_exposure": _build_financial_exposure(non_positive),
        "execution_load": {
            "total_recommendations": len(recommendations),
            "immediate_actions": sum(1 for r in recommendations if r.get("priority") == "IMMEDIATE"),
            "short_term_actions": sum(1 for r in recommendations if r.get("priority") == "SHORT_TERM"),
            "recurring_signals": recurring_count,
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9 — AI strategic insights layer (Ollama)
# ═════════════════════════════════════════════════════════════════════════════


_INSIGHT_PROMPT = """\
You are FinIntel AI, a senior HR and financial analytics advisor.

Dataset summary:
{summary}

Top 10 risks detected:
{risk_text}

Employee scorecards:
{scorecard_text}

Write exactly 3 strategic management insights in plain English.
Each insight must:
- Be 1–2 sentences
- Include a specific number from the data
- Tell management what to DO, not just what is happening
- Be addressed to a senior HR or Finance leader
No bullet numbering. Separate insights with a blank line.
"""


def _get_ai_insights(records: list[dict], risks: list[dict], scorecards: list[dict]) -> str:
    """Use configured LLM provider for strategic insights. Returns empty string on failure."""
    overall   = build_overall_summary(records)
    months    = get_months_available(records)
    projects  = build_projects(records)

    summary_lines = [
        f"Period: {', '.join(months)}",
        f"Revenue: {_money(overall['total_revenue'])}, "
        f"Cost: {_money(overall['total_cost'])}, "
        f"Profit: {_money(overall['total_profit'])}, "
        f"Margin: {_pct(overall['avg_margin_pct'])}",
        f"Employees: {overall['total_employees']}",
    ]
    for p, d in projects.items():
        m = (d["profit"] / d["revenue"] * 100) if d["revenue"] > 0 else 0
        summary_lines.append(f"  {p}: Rev={_money(d['revenue'])}, Margin={_pct(m)}, HC={d['employees']}")

    risk_lines = [
        f"- [{r['severity'].upper()}] {r['description']}"
        for r in risks[:10]
        if r.get("severity") != "positive"
    ]
    sc_lines = [
        f"- {s['employee']}: score {s['performance']['score']}/100 ({s['performance']['band']}), "
        f"util {s['avg_utilisation']}%, profit {_money(s['total_profit'])}"
        for s in scorecards[:8]
    ]

    prompt = _INSIGHT_PROMPT.format(
        summary="\n".join(summary_lines),
        risk_text="\n".join(risk_lines) or "None",
        scorecard_text="\n".join(sc_lines) or "None",
    )

    if is_llm_available():
        try:
            return _llm_generate(prompt, timeout=90).strip()
        except Exception as e:
            print(f"[risk_engine] LLM insight failed: {e}")

    return ""


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Main public function
# ═════════════════════════════════════════════════════════════════════════════

def get_risks_and_recommendations(
    time_range: Optional[str] = None,
    max_items: int = 8,
    include_positive: bool = True,
    include_ai_insights: bool = True,
) -> dict:
    """
    Main entry point. Returns a complete risk and recommendation report.

    Args:
        time_range:         "1M" | "3M" | "6M" | "1Y" | None (all)
        max_items:          max risks / recommendations in summary lists
        include_positive:   whether to include STAR/CONSISTENT signals
        include_ai_insights: whether to call Ollama for strategic insights

    Returns:
        {
          overview:            {total_risks, action_needed, by_severity, by_category}
          risks:               [all risks, sorted by priority — positives last]
          financial_risks:     [filtered]
          workforce_risks:     [filtered]
          operational_risks:   [filtered]
          positive_signals:    [STAR/CONSISTENT performers]
          trend_insights:      [portfolio-level trend signals]
          recommendations:     [top N specific actionable recs]
          employee_scorecards: [all employees with performance score]
          ai_insights:         str — strategic management commentary
          summary:             overall financial summary
        }
    """
    records = GLOBAL_DATASET
    if not records:
        return _empty_response()

    if time_range:
        records = filter_by_range(records, time_range)
    if not records:
        return _empty_response()

    limit     = max(3, min(int(max_items or 8), 30))
    overall   = build_overall_summary(records)
    total_rev = overall.get("total_revenue") or 1.0

    # ── Build employee timelines ──────────────────────────────────────────
    timelines = _build_employee_timelines(records)

    # ── Detect all risks ──────────────────────────────────────────────────
    all_risks: list[dict] = []

    for name, emp_records in timelines.items():
        emp_risks = _detect_employee_risks(name, emp_records, total_rev)
        all_risks.extend(emp_risks)

    all_risks.extend(_detect_project_risks(records))
    all_risks.extend(_detect_trend_risks(records))

    # Score and sort
    for r in all_risks:
        r["_score"]   = _score_risk(r)
        r["priority"] = _priority_from_score(r["_score"])

    all_risks = _sort_risks(all_risks)

    # Filter by include_positive
    if not include_positive:
        all_risks = [r for r in all_risks if r.get("severity") != "positive"]

    # ── Categorise ────────────────────────────────────────────────────────
    financial_risks   = [r for r in all_risks if r["category"] == "financial"    and r.get("severity") != "positive"]
    workforce_risks   = [r for r in all_risks if r["category"] == "workforce"    and r.get("severity") != "positive"]
    operational_risks = [r for r in all_risks if r["category"] == "operational"  and r.get("severity") != "positive"]
    positive_signals  = [r for r in all_risks if r.get("severity") == "positive"]

    # ── Trend insights (already detected above as risk items) ─────────────
    trend_types   = {"DECLINING_MARGIN_TREND", "RISING_COST_TREND", "FALLING_UTILISATION_TREND"}
    trend_risks   = [r for r in all_risks if r["type"] in trend_types]
    action_risks  = [r for r in all_risks if r["type"] not in trend_types and r.get("severity") != "positive"]

    # ── Recommendations = top N risks with their specific recommendation ──
    recommendations = [
        {
            "action":           r["recommendation"],
            "owner":            r.get("owner", ""),
            "deadline":         r.get("deadline", ""),
            "priority":         r.get("priority", "SHORT_TERM"),
            "priority_score":   r.get("_score", 5.0),
            "category":         r.get("category", "general"),
            "linked_employees": r.get("linked_employees", []),
            "related_risk_type": r.get("type", ""),
        }
        for r in action_risks[:limit]
    ]

    # ── Employee scorecards ───────────────────────────────────────────────
    scorecards = _build_employee_scorecards(timelines)

    # ── AI strategic insights ─────────────────────────────────────────────
    ai_insights = ""
    if include_ai_insights and action_risks:
        ai_insights = _get_ai_insights(records, action_risks, scorecards)

    # ── Clean _score from output (internal field) ─────────────────────────
    for r in all_risks:
        r.pop("_score", None)

    action_tracker = _build_action_tracker(recommendations)
    executive_summary = _build_executive_summary(all_risks, recommendations)
    data_quality = _build_data_quality_summary(records)
    project_risk_heatmap = _build_project_risk_heatmap(all_risks)
    explainability_sources = _build_explainability_sources(all_risks)

    return {
        "overview":            _build_risk_overview(all_risks),
        "executive_summary":   executive_summary,
        "risks":               all_risks,
        "financial_risks":     financial_risks[:limit],
        "workforce_risks":     workforce_risks[:limit],
        "operational_risks":   operational_risks[:limit],
        "positive_signals":    positive_signals,
        "trend_insights":      trend_risks,
        "recommendations":     recommendations,
        "action_tracker":      action_tracker,
        "project_risk_heatmap": project_risk_heatmap,
        "data_quality":        data_quality,
        "sources":             explainability_sources,
        "employee_scorecards": scorecards,
        "ai_insights":         ai_insights,
        "summary":             overall,
        "meta": {
            "time_range":        time_range or "ALL",
            "total_employees":   len(timelines),
            "total_risks":       len(all_risks),
            "action_needed":     sum(1 for r in all_risks if r.get("severity") in ("critical","high")),
            "records_analysed":  len(records),
        },
    }


def _empty_response() -> dict:
    return {
        "overview":            {"total_risks": 0, "action_needed": 0, "by_severity": {}, "by_category": {}},
        "executive_summary":   {
            "top_critical_actions": [],
            "financial_exposure": {"estimated_total": 0.0, "components": {}},
            "execution_load": {"total_recommendations": 0, "immediate_actions": 0,
                                "short_term_actions": 0, "recurring_signals": 0},
        },
        "risks":               [],
        "financial_risks":     [],
        "workforce_risks":     [],
        "operational_risks":   [],
        "positive_signals":    [],
        "trend_insights":      [],
        "recommendations":     [],
        "action_tracker":      [],
        "project_risk_heatmap": [],
        "data_quality":        {"confidence": "LOW", "completeness_pct": 0.0,
                                 "missing_fields": {}, "notes": "No records available for risk analysis."},
        "sources":             [],
        "employee_scorecards": [],
        "ai_insights":         "",
        "summary":             {},
        "meta":                {"time_range": "ALL", "total_employees": 0, "total_risks": 0,
                                "action_needed": 0, "records_analysed": 0},
    }