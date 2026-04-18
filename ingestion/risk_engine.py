"""
risk_engine.py — Risk analysis and AI-powered recommendations.

Detects financial, operational, and workforce risks from the dataset,
then uses Ollama to generate actionable recommendations.
"""

from .dataset import (
    GLOBAL_DATASET, build_projects, build_overall_summary,
    build_top_performers, get_months_available, filter_by_range,
)
from .ai_mapper import _ollama_generate, is_ollama_available


# ── Risk detection ───────────────────────────────────────────────────────────

def _detect_risks(records):
    """Analyze records and return categorized risks with severity."""
    risks = []

    # ── Employee-level risks ─────────────────────────────────────────────
    emp_data = {}
    for r in records:
        name = r.get("employee", "Unknown")
        if name not in emp_data:
            emp_data[name] = {
                "project": r.get("project", ""),
                "total_profit": 0, "total_revenue": 0, "total_cost": 0,
                "months": set(), "vacation_days": 0,
                "utilisation_pcts": [], "margin_pcts": [],
            }
        emp_data[name]["total_profit"] += r.get("profit", 0)
        emp_data[name]["total_revenue"] += r.get("revenue", 0)
        emp_data[name]["total_cost"] += r.get("cost", 0)
        emp_data[name]["months"].add(r.get("month", ""))
        emp_data[name]["vacation_days"] += r.get("vacation_days", 0)
        if r.get("utilisation_pct"):
            emp_data[name]["utilisation_pcts"].append(r["utilisation_pct"])
        if r.get("margin_pct") is not None:
            emp_data[name]["margin_pcts"].append(r["margin_pct"])

    for name, data in emp_data.items():
        # Loss-making employee
        if data["total_profit"] < 0:
            risks.append({
                "category": "financial",
                "severity": "high",
                "type": "LOSS_MAKING_EMPLOYEE",
                "entity": name,
                "project": data["project"],
                "detail": f"Net loss of ${abs(data['total_profit']):,.2f}",
                "metrics": {
                    "profit": round(data["total_profit"], 2),
                    "revenue": round(data["total_revenue"], 2),
                    "cost": round(data["total_cost"], 2),
                },
            })

        # Low margin
        avg_margin = (
            sum(data["margin_pcts"]) / len(data["margin_pcts"])
            if data["margin_pcts"] else None
        )
        if avg_margin is not None and 0 <= avg_margin < 15:
            risks.append({
                "category": "financial",
                "severity": "medium",
                "type": "LOW_MARGIN_EMPLOYEE",
                "entity": name,
                "project": data["project"],
                "detail": f"Average margin only {avg_margin:.1f}%",
                "metrics": {"avg_margin_pct": round(avg_margin, 2)},
            })

        # High leave
        if data["vacation_days"] >= 5:
            risks.append({
                "category": "workforce",
                "severity": "medium",
                "type": "HIGH_LEAVE",
                "entity": name,
                "project": data["project"],
                "detail": f"{data['vacation_days']} vacation days taken",
                "metrics": {"vacation_days": data["vacation_days"]},
            })

        # Low utilisation
        avg_util = (
            sum(data["utilisation_pcts"]) / len(data["utilisation_pcts"])
            if data["utilisation_pcts"] else None
        )
        if avg_util is not None and avg_util < 75:
            risks.append({
                "category": "operational",
                "severity": "medium" if avg_util >= 50 else "high",
                "type": "LOW_UTILISATION",
                "entity": name,
                "project": data["project"],
                "detail": f"Utilisation at {avg_util:.1f}%",
                "metrics": {"utilisation_pct": round(avg_util, 2)},
            })

    # ── Project-level risks ──────────────────────────────────────────────
    projects = build_projects(records)
    overall = build_overall_summary(records)

    for proj_name, pdata in projects.items():
        if pdata["profit"] < 0:
            risks.append({
                "category": "financial",
                "severity": "critical",
                "type": "LOSS_MAKING_PROJECT",
                "entity": proj_name,
                "project": proj_name,
                "detail": f"Project net loss of ${abs(pdata['profit']):,.2f}",
                "metrics": pdata,
            })
        elif pdata["revenue"] > 0:
            proj_margin = (pdata["profit"] / pdata["revenue"]) * 100
            if proj_margin < 20:
                risks.append({
                    "category": "financial",
                    "severity": "medium",
                    "type": "LOW_MARGIN_PROJECT",
                    "entity": proj_name,
                    "project": proj_name,
                    "detail": f"Project margin only {proj_margin:.1f}%",
                    "metrics": {**pdata, "margin_pct": round(proj_margin, 2)},
                })

        # Single-person dependency
        if pdata["employees"] == 1 and pdata["revenue"] > 10000:
            risks.append({
                "category": "operational",
                "severity": "medium",
                "type": "KEY_PERSON_DEPENDENCY",
                "entity": proj_name,
                "project": proj_name,
                "detail": f"Only 1 employee generating ${pdata['revenue']:,.2f} revenue",
                "metrics": pdata,
            })

    # Sort: critical > high > medium > low
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    risks.sort(key=lambda r: severity_order.get(r["severity"], 99))

    return risks


# ── Human-readable risk descriptions ────────────────────────────────────────

def _risk_description(risk):
    """Generate a UI-friendly one-liner for a risk card."""
    t = risk["type"]
    entity = risk["entity"]
    m = risk.get("metrics", {})

    if t == "LOSS_MAKING_PROJECT":
        return f"{entity}: loss-making project (${abs(m.get('profit', 0)):,.2f} loss)"
    if t == "LOW_MARGIN_PROJECT":
        return f"{entity}: low margin project ({m.get('margin_pct', 0)}%)"
    if t == "LOSS_MAKING_EMPLOYEE":
        return f"{entity}: generating net loss (${abs(m.get('profit', 0)):,.2f})"
    if t == "LOW_MARGIN_EMPLOYEE":
        return f"{entity}: low margin ({m.get('avg_margin_pct', 0)}%)"
    if t == "HIGH_LEAVE":
        return f"{entity}: high leave ({m.get('vacation_days', 0)} days)"
    if t == "LOW_UTILISATION":
        return f"{entity}: low utilisation ({m.get('utilisation_pct', 0)}%)"
    if t == "KEY_PERSON_DEPENDENCY":
        return f"{entity}: single-person dependency (${m.get('revenue', 0):,.2f} revenue)"
    return f"{entity}: {risk['detail']}"


# ── Per-risk recommendation generator ───────────────────────────────────────

def _generate_recommendation(risk):
    """Return a structured recommendation for a single risk."""
    t = risk["type"]
    entity = risk["entity"]
    project = risk.get("project", entity)
    m = risk.get("metrics", {})

    if t == "LOSS_MAKING_PROJECT":
        return {
            "action": f"Increase billing rates or reduce costs for {entity} to eliminate ${abs(m.get('profit', 0)):,.2f} loss",
            "priority": "IMMEDIATE",
            "category": "financial",
        }
    if t == "LOW_MARGIN_PROJECT":
        return {
            "action": f"Increase billing rate or reduce costs for {entity} (margin {m.get('margin_pct', 0)}%)",
            "priority": "SHORT_TERM",
            "category": "financial",
        }
    if t == "LOSS_MAKING_EMPLOYEE":
        return {
            "action": f"Review {entity}'s billing rate and cost structure on {project}",
            "priority": "IMMEDIATE",
            "category": "financial",
        }
    if t == "LOW_MARGIN_EMPLOYEE":
        return {
            "action": f"Renegotiate billing rate for {entity} (current margin {m.get('avg_margin_pct', 0)}%)",
            "priority": "SHORT_TERM",
            "category": "financial",
        }
    if t == "HIGH_LEAVE":
        return {
            "action": f"Review workload and engagement for {entity} ({m.get('vacation_days', 0)} leave days)",
            "priority": "SHORT_TERM",
            "category": "workforce",
        }
    if t == "LOW_UTILISATION":
        util = m.get("utilisation_pct", 0)
        return {
            "action": f"Reallocate {entity} to higher-demand projects (utilisation {util}%)",
            "priority": "IMMEDIATE" if util < 50 else "SHORT_TERM",
            "category": "operational",
        }
    if t == "KEY_PERSON_DEPENDENCY":
        return {
            "action": f"Add backup resource to {entity} to reduce single-person risk",
            "priority": "LONG_TERM",
            "category": "operational",
        }
    return {
        "action": f"Investigate {entity}: {risk['detail']}",
        "priority": "SHORT_TERM",
        "category": "general",
    }


# ── AI strategic insights (optional Ollama layer) ───────────────────────────

_INSIGHT_PROMPT = """\
You are FinIntel AI, a financial analytics advisor.

Data:
{summary}

Risks:
{risks_text}

Give 2-3 high-level strategic insights for management in plain English.
Keep each insight to 1 sentence. Use actual numbers. No bullet numbering.
"""


def _build_summary_text(records):
    overall = build_overall_summary(records)
    projects = build_projects(records)
    months = get_months_available(records)
    lines = [
        f"Period: {', '.join(months)}",
        f"Revenue: ${overall['total_revenue']:,.2f}, Cost: ${overall['total_cost']:,.2f}, "
        f"Profit: ${overall['total_profit']:,.2f}, Margin: {overall['avg_margin_pct']}%",
    ]
    for name, data in projects.items():
        margin = round((data["profit"] / data["revenue"]) * 100, 1) if data["revenue"] > 0 else 0
        lines.append(f"  {name}: Rev=${data['revenue']:,.2f}, Profit=${data['profit']:,.2f}, Margin={margin}%, HC={data['employees']}")
    return "\n".join(lines)


# ── Main function ───────────────────────────────────────────────────────────

def get_risks_and_recommendations(time_range=None):
    """
    Detect risks and generate structured recommendations.
    Returns data shaped for the Risks & Recs UI tab.
    """
    records = GLOBAL_DATASET
    if not records:
        return {
            "risks": [],
            "recommendations": [],
            "ai_insights": "",
            "risk_count": {"critical": 0, "high": 0, "medium": 0},
            "summary": {},
        }

    if time_range:
        records = filter_by_range(records, time_range)

    raw_risks = _detect_risks(records)
    overall = build_overall_summary(records)

    # Build structured risks with UI descriptions
    risks = []
    for r in raw_risks:
        risks.append({
            "severity": r["severity"],
            "category": r["category"],
            "type": r["type"],
            "entity": r["entity"],
            "project": r.get("project", ""),
            "description": _risk_description(r),
            "metrics": r.get("metrics", {}),
        })

    # Build structured recommendations (one per risk, deduplicated)
    recommendations = []
    seen_actions = set()
    for r in raw_risks:
        rec = _generate_recommendation(r)
        if rec["action"] not in seen_actions:
            seen_actions.add(rec["action"])
            recommendations.append(rec)

    # Optional: AI strategic insights via Ollama
    ai_insights = ""
    if raw_risks and is_ollama_available():
        try:
            summary_text = _build_summary_text(records)
            risk_lines = [f"- {_risk_description(r)}" for r in raw_risks]
            prompt = _INSIGHT_PROMPT.format(
                summary=summary_text, risks_text="\n".join(risk_lines)
            )
            ai_insights = _ollama_generate(prompt, timeout=120).strip()
        except Exception:
            ai_insights = ""

    return {
        "risks": risks,
        "recommendations": recommendations,
        "ai_insights": ai_insights,
        "risk_count": {
            "critical": len([r for r in risks if r["severity"] == "critical"]),
            "high": len([r for r in risks if r["severity"] == "high"]),
            "medium": len([r for r in risks if r["severity"] == "medium"]),
        },
        "summary": overall,
    }
