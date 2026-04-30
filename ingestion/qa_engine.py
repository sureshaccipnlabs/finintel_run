"""
qa_engine.py — LLM-powered Q&A over the ingested financial dataset.

Builds a compact text summary of GLOBAL_DATASET and sends it as context
to Ollama so the LLM can answer any natural-language question about the data.
"""

import json
import re
import hashlib
import time

from .dataset import (
    GLOBAL_DATASET, build_projects, build_monthly, build_overall_summary,
    build_top_performers, build_risks, get_months_available, filter_by_range,
    build_employee_summaries, set_on_dataset_change_callback
)
from .ai_mapper import _ollama_generate, is_ollama_available
from .forecast_ import try_answer_forecast, is_likely_forecast


# ── QA-specific JSON extraction ───────────────────────────────────────────────

def _clean_qa_json(raw: str) -> str:
    """Clean LLM JSON output: strip comments, trailing commas, fix number formatting."""
    # Strip JS-style comments
    raw = re.sub(r'//[^\n]*', '', raw)
    # Remove trailing commas before } or ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    # Fix numbers with commas like 18,816.0 -> 18816.0
    def fix_number_commas(match):
        return match.group(0).replace(',', '')
    raw = re.sub(r'(?<=[:\s\[])(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', fix_number_commas, raw)
    return raw.strip()


def _extract_qa_json(text: str):
    """Extract JSON from Q&A LLM response. Independent of ai_mapper._extract_json."""
    if not text:
        return None
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try with cleaning
    try:
        return json.loads(_clean_qa_json(text))
    except json.JSONDecodeError:
        pass
    # Try extracting from markdown code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(_clean_qa_json(m.group(1)))
        except json.JSONDecodeError:
            pass
    # Greedy: find outermost { ... }
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(_clean_qa_json(text[start:i + 1]))
                except json.JSONDecodeError:
                    start = None
    return None


# ── Context Caching ──────────────────────────────────────────────────────────

_CONTEXT_CACHE = {
    "hash": None,
    "context": None,
    "timestamp": 0,
}
_CACHE_TTL_SECONDS = 300  # 5 minutes TTL as backup


def _compute_dataset_hash(records) -> str:
    """Compute a hash of the dataset for cache invalidation."""
    if not records:
        return "empty"
    # Hash based on: record count + first record + last record
    parts = [
        str(len(records)),
        str(records[0]) if records else "",
        str(records[-1]) if records else "",
    ]
    return hashlib.md5("||".join(parts).encode()).hexdigest()


def _get_cached_context(records=None, question: str = None):
    """Get context from cache if valid, otherwise rebuild and cache.
    
    Note: When question is provided, we build a targeted context (not cached).
    Full context is cached for fallback/general use.
    """
    recs = records if records is not None else GLOBAL_DATASET
    
    # If question provided, build targeted context (don't cache - question-specific)
    if question:
        return _build_dataset_context(recs, question=question)
    
    # For general context (no question), use cache
    current_hash = _compute_dataset_hash(recs)
    now = time.time()
    
    # Check if cache is valid (hash matches AND within TTL)
    if (_CONTEXT_CACHE["hash"] == current_hash 
        and _CONTEXT_CACHE["context"] is not None
        and (now - _CONTEXT_CACHE["timestamp"]) < _CACHE_TTL_SECONDS):
        return _CONTEXT_CACHE["context"]
    
    # Rebuild full context
    context = _build_dataset_context(recs)
    
    # Update cache
    _CONTEXT_CACHE["hash"] = current_hash
    _CONTEXT_CACHE["context"] = context
    _CONTEXT_CACHE["timestamp"] = now
    
    return context


def invalidate_context_cache():
    """Manually invalidate the context cache (call after dataset changes)."""
    _CONTEXT_CACHE["hash"] = None
    _CONTEXT_CACHE["context"] = None
    _CONTEXT_CACHE["timestamp"] = 0


# Register cache invalidation callback with dataset module
set_on_dataset_change_callback(invalidate_context_cache)


# ── Context builder ──────────────────────────────────────────────────────────

def _detect_question_scope(question: str) -> dict:
    """Analyze question to determine what context sections are needed."""
    q = (question or "").lower()
    
    scope = {
        "needs_overall": True,  # Always include
        "needs_projects": False,
        "needs_monthly": False,
        "needs_employee_summary": False,
        "needs_employee_details": False,
        "needs_risks": False,
        "specific_employee": None,
        "specific_project": None,
        "specific_month": None,
    }
    
    # Project-related keywords (expanded)
    project_keywords = [
        "project", "projects", "client", "clients", "projecty", "projectx",
        "which project", "project health", "project performance"
    ]
    if any(w in q for w in project_keywords):
        scope["needs_projects"] = True
    
    # Monthly/trend-related (expanded)
    monthly_keywords = [
        "month", "monthly", "trend", "over time", "by month", "each month",
        "quarter", "q1", "q2", "q3", "q4", "growth", "change", "from", "to",
        "ytd", "year to date", "last quarter", "this quarter", "previous month",
        "last month", "this month", "fiscal", "fy", "weekly", "week"
    ]
    if any(w in q for w in monthly_keywords):
        scope["needs_monthly"] = True
    
    # Specific month mentioned
    month_pattern = r"(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    month_match = re.search(month_pattern, q)
    if month_match:
        scope["needs_monthly"] = True
        scope["needs_employee_details"] = True
        scope["specific_month"] = month_match.group(1)
    
    # Employee-related keywords (expanded for natural language)
    employee_keywords = [
        "employee", "employees", "person", "people", "who", "top", "bottom", 
        "best", "worst", "performer", "performers", "resource", "resources",
        "team", "staff", "idle", "sitting", "working", "contributor", "contributors",
        "underutilized", "overloaded", "free", "busy", "risky", "productive"
    ]
    if any(w in q for w in employee_keywords):
        scope["needs_employee_summary"] = True
    
    # Specific employee name mentioned (capitalized word that's not a common word)
    common_words = {
        "give", "me", "show", "list", "what", "how", "the", "for", "and", "with", "summary", 
        "detail", "details", "total", "revenue", "profit", "cost", "margin", "hours", 
        "attendance", "utilization", "which", "who", "where", "when", "why", "does", "did",
        "employees", "employee", "project", "projects", "highest", "lowest", "top", "bottom",
        "best", "worst", "most", "least", "generated", "has", "have", "had", "are", "is", "was",
        "show", "compare", "trend", "performance", "health", "summary", "insights", "issues",
        "risks", "anomalies", "identify", "detect", "predict", "suggest", "analyze"
    }
    name_match = re.findall(r"\b([A-Z][a-z]+)\b", question)
    for name in name_match:
        if name.lower() not in common_words and name.lower() not in month_pattern:
            scope["needs_employee_summary"] = True
            scope["needs_employee_details"] = True
            scope["specific_employee"] = name
            break
    
    # Specific metrics that need employee details (expanded)
    metric_keywords = [
        "hours", "actual hours", "billable", "billable hours", "vacation", "leave", 
        "working days", "utilization", "utilisation", "util", "billing rate", 
        "cost rate", "zero hours", "no hours", "logged", "per hour", "per day",
        "per employee", "per project", "ratio", "average", "avg", "mean",
        "below", "above", "less than", "more than", "greater than", "threshold"
    ]
    if any(w in q for w in metric_keywords):
        scope["needs_employee_summary"] = True
        if any(w in q for w in ["his", "her", "their", "'s"]) or re.search(r"\b[A-Z][a-z]+\b", question):
            scope["needs_employee_details"] = True
    
    # Risk-related keywords (expanded for natural language)
    risk_keywords = [
        "risk", "risks", "risky", "issue", "issues", "problem", "problems", 
        "concern", "flag", "flags", "warning", "warnings", "low utilization",
        "underperform", "underutilized", "idle", "zero", "not generating",
        "not working", "not billing", "not billable", "not profitable",
        "anomaly", "anomalies", "inconsistent", "validation", "declining",
        "losing", "leakage", "attention", "health"
    ]
    if any(w in q for w in risk_keywords):
        scope["needs_risks"] = True
        scope["needs_employee_summary"] = True
    
    # Comparison queries need multiple sections
    comparison_keywords = ["compare", "comparison", "vs", "versus", "difference", "between"]
    if any(w in q for w in comparison_keywords):
        scope["needs_projects"] = True
        scope["needs_employee_summary"] = True
        scope["needs_monthly"] = True
    
    # Insight/summary queries (executive level)
    insight_keywords = [
        "insight", "insights", "summary", "overview", "highlight", "observations",
        "key", "important", "performance summary", "health summary", "improve"
    ]
    if any(w in q for w in insight_keywords):
        scope["needs_projects"] = True
        scope["needs_employee_summary"] = True
        scope["needs_risks"] = True
    
    # Aggregation queries (totals, counts) - overall is enough
    if re.search(r"\b(total|sum|count|how many|how much|overall|all)\b", q) and not scope["needs_employee_details"]:
        pass  # Overall summary is sufficient
    
    # For short questions (<=3 words), infer from keywords
    if len(q.split()) <= 3:
        if any(w in q for w in ["revenue", "profit", "cost", "margin", "total", "overall", "rev", "pft"]):
            pass  # Overall summary already included
        elif any(w in q for w in ["employee", "employees", "who", "person", "emp", "top", "bottom"]):
            scope["needs_employee_summary"] = True
        elif any(w in q for w in ["project", "projects"]):
            scope["needs_projects"] = True
        elif any(w in q for w in ["month", "monthly", "trend"]):
            scope["needs_monthly"] = True
        elif any(w in q for w in ["risk", "risks", "issues", "problems"]):
            scope["needs_risks"] = True
            scope["needs_employee_summary"] = True
        else:
            scope["needs_employee_summary"] = True
    
    # Explicit "show all" type requests
    if any(w in q for w in ["everything", "all data", "full summary", "complete"]):
        scope["needs_projects"] = True
        scope["needs_monthly"] = True
        scope["needs_employee_summary"] = True
        scope["needs_risks"] = True
    
    return scope


def _build_dataset_context(records=None, question: str = None):
    """Build a compact text summary of the dataset for the LLM prompt."""
    recs = records if records is not None else GLOBAL_DATASET
    if not recs:
        return "No data has been loaded yet."

    # Determine what sections are needed based on question
    scope = _detect_question_scope(question) if question else None
    include_all = scope is None  # If no question, include everything
    
    overall = build_overall_summary(recs)
    months = get_months_available(recs)

    lines = []

    # Overall - always include (compact)
    lines.append("=== OVERALL SUMMARY ===")
    lines.append(f"Total Revenue: ${overall['total_revenue']:,.2f}, Cost: ${overall['total_cost']:,.2f}, Profit: ${overall['total_profit']:,.2f}")
    lines.append(f"Hours: {overall.get('total_hours', 0)}, Margin: {overall['avg_margin_pct']}%, Employees: {overall['total_employees']}")
    lines.append(f"Months: {', '.join(months)}")
    if months:
        lines.append(f"LAST MONTH (most recent data available): {months[-1]}")
        lines.append(f"NOTE: If user asks about 'last month', return data for {months[-1]} and clarify this is the most recent data available.")
    lines.append("")

    # Projects - include if needed
    if include_all or scope.get("needs_projects"):
        projects = build_projects(recs)
        lines.append("=== PROJECTS ===")
        for name, data in sorted(projects.items(), key=lambda x: x[1]["revenue"], reverse=True):
            proj_revenue = float(data.get("revenue") or 0)
            proj_cost = float(data.get("cost") or 0)
            proj_margin = round(((proj_revenue - proj_cost) / proj_revenue) * 100, 2) if proj_revenue > 0 else 0
            lines.append(
                f"- {name}: Revenue=${data['revenue']:,.2f}, Cost=${data.get('cost', 0):,.2f}, "
                f"Profit=${data['profit']:,.2f}, GrossMargin={proj_margin}%, "
                f"Utilisation={data.get('avg_utilisation', 0)}%, Employees={data['employees']}"
            )
        lines.append("")

    # Monthly - include if needed (sorted chronologically)
    if include_all or scope.get("needs_monthly"):
        monthly = build_monthly(recs)
        lines.append("=== MONTHLY ===")
        # Sort months chronologically using the same order as get_months_available
        sorted_months = [m for m in months if m in monthly]
        for m in sorted_months:
            data = monthly[m]
            is_last = (m == months[-1]) if months else False
            marker = " [LAST MONTH]" if is_last else ""
            lines.append(
                f"- {m}{marker}: Revenue=${data['total_revenue']:,.2f}, Profit=${data['total_profit']:,.2f}, "
                f"Margin={data['avg_margin_pct']}%, Employees={data['employees']}"
            )
        lines.append("")

    # Employee Summaries - include if needed
    if include_all or scope.get("needs_employee_summary"):
        employee_summaries = build_employee_summaries(recs)
        # Filter to specific employee if mentioned
        if scope and scope.get("specific_employee"):
            emp_name = scope["specific_employee"].lower()
            employee_summaries = [e for e in employee_summaries if emp_name in e["employee_name"].lower()]
        lines.append("=== EMPLOYEES ===")
        for emp in employee_summaries:
            lines.append(
                f"- {emp['employee_name']}: Revenue=${emp['total_revenue']:,.2f}, "
                f"Profit=${emp['total_profit']:,.2f}, Margin={emp.get('gross_margin_pct') or emp.get('margin_pct') or 0}%, "
                f"Hours={emp['total_hours']}, Utilisation={emp['utilization_pct'] or 0}%, "
                f"Attendance={emp.get('attendance_pct', 100)}%, VacationDays={emp.get('vacation_days', 0)}"
            )
        if not employee_summaries:
            lines.append("- No matching employee found")
        lines.append("")

    # Employee Details - include if needed (limit to relevant records)
    if include_all or scope.get("needs_employee_details"):
        lines.append("=== EMPLOYEE DETAILS ===")
        # Filter records if specific month/employee mentioned
        filtered_recs = recs
        if scope and scope.get("specific_employee"):
            emp_name = scope["specific_employee"].lower()
            filtered_recs = [r for r in recs if emp_name in (r.get("employee") or "").lower()]
        if scope and scope.get("specific_month"):
            month_key = scope["specific_month"].lower()
            filtered_recs = [r for r in filtered_recs if month_key in (r.get("month") or "").lower()]
        
        # Limit to 30 records max to keep context manageable
        for r in filtered_recs[:30]:
            rev = r.get('revenue') or 0
            pft = r.get('profit') or 0
            lines.append(
                f"- {r.get('employee', '?')} | {r.get('project', '?')} | {r.get('month', '?')} | "
                f"Hours={r.get('actual_hours', 0)}, Vacation={r.get('vacation_days', 0)}, "
                f"Revenue=${rev:,.2f}, Profit=${pft:,.2f}, "
                f"Utilisation={r.get('utilisation_pct', 0)}%"
            )
        if len(filtered_recs) > 30:
            lines.append(f"... and {len(filtered_recs) - 30} more records")
        lines.append("")

    # Risks - include if needed
    if include_all or scope.get("needs_risks"):
        risks = build_risks(recs)
        if risks:
            lines.append("=== RISKS ===")
            for r in risks:
                lines.append(f"- {r['employee']} ({r.get('month', '')}): {r['issue']}")
            lines.append("")

    return "\n".join(lines)


# ── Prompt template ──────────────────────────────────────────────────────────

_QA_PROMPT = """\
You are FinIntel AI. Answer from this data:

{context}
---
Question: {question}

Return ONLY valid JSON. No text before or after the JSON.

EXAMPLES:

Q: "What is total revenue?"
{{"summary": "Total revenue is $88,960.", "visual_type": "metric", "columns": [], "data": [{{"label": "Total Revenue", "value": 88960}}]}}

Q: "Give me overall summary"
{{"summary": "Overall financial summary.", "visual_type": "metric", "columns": [], "data": [{{"label": "Total Revenue", "value": 88960}}, {{"label": "Total Cost", "value": 77216}}, {{"label": "Total Profit", "value": 11744}}, {{"label": "Margin", "value": 13.2}}]}}

Q: "Top 3 employees by profit"
{{"summary": "Top 3 employees by profit.", "visual_type": "table", "columns": ["employee", "profit"], "data": [{{"employee": "John", "profit": 50000}}, {{"employee": "Jane", "profit": 45000}}, {{"employee": "Bob", "profit": 40000}}]}}

Q: "Revenue by project"
{{"summary": "Revenue by project.", "visual_type": "table", "columns": ["project", "revenue"], "data": [{{"project": "Alpha", "revenue": 200000}}, {{"project": "Beta", "revenue": 150000}}]}}

Q: "How many employees?"
{{"summary": "There are 7 employees.", "visual_type": "metric", "columns": [], "data": [{{"label": "Employee Count", "value": 7}}]}}

Q: "Monthly revenue trend"
{{"summary": "Monthly revenue trend.", "visual_type": "table", "columns": ["month", "revenue"], "data": [{{"month": "January 2026", "revenue": 30000}}, {{"month": "February 2026", "revenue": 32000}}]}}

Q: "Employees with highest attendance"
{{"summary": "Employees sorted by highest attendance.", "visual_type": "table", "columns": ["employee", "attendance"], "data": [{{"employee": "Jane", "attendance": 100}}, {{"employee": "John", "attendance": 95}}, {{"employee": "Bob", "attendance": 90}}]}}

Q: "Which employees are not generating revenue"
{{"summary": "Employees with zero or negative revenue.", "visual_type": "table", "columns": ["employee", "revenue"], "data": [{{"employee": "Bob", "revenue": 0}}, {{"employee": "Jane", "revenue": -500}}]}}

Q: "Employees with margin below 30%"
{{"summary": "Employees with margin below 30%.", "visual_type": "table", "columns": ["employee", "margin"], "data": [{{"employee": "Bob", "margin": 15}}, {{"employee": "Jane", "margin": 22}}]}}

Q: "Show employees with revenue above $10,000"
{{"summary": "Employees with revenue above $10,000.", "visual_type": "table", "columns": ["employee", "revenue"], "data": [{{"employee": "John", "revenue": 25000}}, {{"employee": "Jane", "revenue": 18000}}]}}

Q: "Revenue per employee"
{{"summary": "Average revenue per employee.", "visual_type": "metric", "columns": [], "data": [{{"label": "Revenue per Employee", "value": 12500}}]}}

Q: "Compare Project Alpha vs Project Beta"
{{"summary": "Comparison of Project Alpha and Project Beta.", "visual_type": "table", "columns": ["metric", "Project Alpha", "Project Beta"], "data": [{{"metric": "Revenue", "Project Alpha": 50000, "Project Beta": 45000}}, {{"metric": "Profit", "Project Alpha": 12000, "Project Beta": 10000}}, {{"metric": "Margin", "Project Alpha": 24, "Project Beta": 22}}]}}

Q: "Top 3 employees in Project Alpha"
{{"summary": "Top 3 employees in Project Alpha by profit.", "visual_type": "table", "columns": ["employee", "project", "profit"], "data": [{{"employee": "John", "project": "Alpha", "profit": 8000}}, {{"employee": "Jane", "project": "Alpha", "profit": 6000}}, {{"employee": "Bob", "project": "Alpha", "profit": 4000}}]}}

RULES:
1. "metric" for totals, counts, averages, summaries - use label/value pairs
2. "table" for lists, rankings, breakdowns with multiple columns
3. "text" only if no data found
4. Use lowercase keys: employee, project, month, revenue, cost, profit, utilization, hours, attendance
5. Numbers must be raw numbers (not strings with $ or %)
6. summary must be a specific answer, not "Overall summary"
7. Return ONLY the JSON object, nothing else
8. "last month" or "recent month" = the month marked as "LAST MONTH (most recent)" in the data
9. For "highest", "top", "best" questions: sort data DESCENDING by the relevant metric
10. For "lowest", "bottom", "worst" questions: sort data ASCENDING by the relevant metric
11. "not generating", "zero", "no revenue/profit" = filter for values that are 0 or negative
12. "below X%", "less than X", "under X" = ONLY include values < X; "above X%", "more than X", "over X" = ONLY include values > X. NEVER include values that don't meet the threshold!
13. "per employee", "per project", "per hour" = calculate average or divide total by count
14. "compare A vs B" = show side-by-side comparison table with metrics as rows
15. "YTD", "year to date" = sum from January to the last available month
16. "Q1" = Jan-Mar, "Q2" = Apr-Jun, "Q3" = Jul-Sep, "Q4" = Oct-Dec
17. For "top N in Project X" = filter by project first, then rank
"""


def _clean_answer(text: str) -> str:
    answer = (text or "")
    for _ in range(3):
        answer = answer.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    answer = answer.replace("\r\n", "\n").replace("\r", "\n")
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    answer = re.sub(r"\*(.*?)\*", r"\1", answer)
    answer = re.sub(r"^\s*(based on the (provided )?(dataset|data)[,:]?|from the (provided )?(dataset|data)[,:]?)\s*", "", answer, flags=re.IGNORECASE)
    # Remove echoed question lines like 'User question: ...' or a repeated question at the start
    answer = re.sub(r"(?im)^\s*(user question|question)\s*:\s*.*?$", "", answer)
    answer = re.sub(r"(?im)^\s*(which|who|what|how many|how much|count|show|list)\b.*?[?:]?\s*$", "", answer)
    answer = re.sub(r"(?m)^\s*\d+[.)]\s*", "- ", answer)
    answer = re.sub(r"(?m)^\s*[•*]\s*", "- ", answer)
    answer = re.sub(r"(?m);\s+(?=[^-].*\|)", "\n- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\s{2,}", " ", answer)
    return answer.strip()


def _to_float_num(s: str):
    try:
        txt = (s or "").strip()
        # Remove common decorations, keep signs and decimals
        txt = txt.replace("$", "").replace(",", "").replace("%", "").strip()
        if not txt:
            return None
        # Extract the first numeric token (handles leading text like "... (±2.0pp)")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None


def _extract_rows(text: str):
    rows = []
    if not text:
        return rows
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        row = {}
        for idx, part in enumerate(parts):
            if ":" in part:
                key, val = part.split(":", 1)
                k = key.strip().lower()
                v = val.strip()
                if k == "month":
                    row["month"] = v
                elif k.replace(" ", "") in ("monthno", "month_no"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["month_no"] = int(round(fv))
                elif k == "quarter":
                    row["quarter"] = v
                elif k.replace(" ", "") in ("quarterno", "quarter_no", "qno"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["quarter_no"] = int(round(fv))
                elif k == "project":
                    row["project"] = v
                elif k == "revenue":
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["revenue"] = fv
                elif k == "cost":
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["cost"] = fv
                elif k == "profit":
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["profit"] = fv
                elif k == "headcount":
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["headcount"] = int(round(fv))
                elif k in ("utilization", "utilisation"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["utilization_pct"] = fv
                elif k in ("hours", "hrs"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["hours"] = fv
                elif k in ("leaves", "leave", "pto", "time off", "timeoff"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["leave_days"] = fv
                elif k in ("vacation", "vacations"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["vacation_days"] = fv
                elif k in ("holidays", "holiday"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["holiday_days"] = fv
                elif k.replace(" ", "") in ("workingdays", "workdays", "workdayscount"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["working_days"] = fv
                elif k.startswith("margin") or "margin" in k.replace(" ", ""):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["margin_pct"] = fv
                elif k.replace(" ", "") in ("billrate", "billingrate"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["billing_rate"] = fv
                elif k.replace(" ", "") in ("costrate", "costperhour"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["cost_rate"] = fv
                else:
                    row[k] = v
            else:
                if idx == 0 and "employee" not in row:
                    row["employee"] = part
        if any(k in row for k in ("employee", "month", "project", "revenue", "cost", "profit", "utilization_pct", "hours", "margin_pct")):
            rows.append(row)
    return rows


def _format_column_name(col: str) -> str:
    """Format column name: remove special chars, capitalize properly."""
    if not col:
        return col
    
    # Column name mappings for common fields
    col_map = {
        "employee": "Employee",
        "project": "Project",
        "month": "Month",
        "quarter": "Quarter",
        "revenue": "Revenue",
        "cost": "Cost",
        "profit": "Profit",
        "headcount": "Headcount",
        "utilization_pct": "Utilization",
        "utilisation_pct": "Utilization",
        "utilization": "Utilization",
        "utilisation": "Utilization",
        "hours": "Hours",
        "total_hours": "Hours",
        "leave_days": "Leaves",
        "vacation_days": "Vacation",
        "holiday_days": "Holidays",
        "working_days": "Working Days",
        "margin_pct": "Margin",
        "margin": "Margin",
        "gross_margin_pct": "Gross Margin",
        "avg_margin_pct": "Avg Margin",
        "gross_margin": "Gross Margin",
        "avg_margin": "Avg Margin",
        "billing_rate": "Bill Rate",
        "cost_rate": "Cost Rate",
        "actual_hours": "Actual Hours",
        "billable_hours": "Billable Hours",
    }
    
    col_lower = col.lower().strip()
    if col_lower in col_map:
        return col_map[col_lower]
    
    # Generic cleanup: replace underscores/special chars with spaces, title case
    cleaned = re.sub(r"[_\-]+", " ", col)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
    cleaned = cleaned.strip()
    
    # Title case each word
    if cleaned:
        return " ".join(word.capitalize() for word in cleaned.split())
    
    return col


def _format_columns(columns: list) -> list:
    """Format all column names in a list."""
    return [_format_column_name(c) for c in (columns or [])]


def _format_value(key: str, value) -> str:
    """Format a value based on its column type (add $, %, etc.)."""
    if value is None:
        return ""
    
    key_lower = key.lower().replace(" ", "_")
    
    # Currency fields
    if key_lower in ("revenue", "cost", "profit", "billing_rate", "bill_rate", "cost_rate"):
        try:
            num = float(value) if not isinstance(value, (int, float)) else value
            return f"${num:,.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Percentage fields
    if key_lower in (
        "utilization", "utilization_pct", "utilisation", "utilisation_pct",
        "margin", "margin_pct", "gross_margin", "gross_margin_pct", "avg_margin", "avg_margin_pct",
    ) or ("margin" in key_lower) or key_lower.endswith("_pct"):
        try:
            if isinstance(value, (int, float)):
                num = float(value)
            else:
                parsed = _to_float_num(str(value))
                if parsed is None:
                    return str(value)
                num = parsed
            return f"{num:.1f}%"
        except (ValueError, TypeError):
            return str(value)
    
    # Integer fields
    if key_lower in ("headcount", "hours", "total_hours", "leaves", "leave_days", "vacation", "vacation_days", 
                     "holidays", "holiday_days", "working_days", "workdays", "actual_hours", "billable_hours"):
        try:
            num = float(value) if not isinstance(value, (int, float)) else value
            return f"{int(round(num)):,}"
        except (ValueError, TypeError):
            return str(value)
    
    # Default: return as-is
    return value


def _pivot_table_if_needed(data: list, columns: list) -> tuple:
    """
    Pivot table if it has repeating entity+month combinations.
    
    Handles various column counts:
    - 2 cols: Month, Revenue -> Single row with months as columns
    - 3 cols: Month, Employee, Leaves -> Employee, May 2026, June 2026
    - 4+ cols: Month, Employee, Leaves, Vacation -> Employee, May 2026 Leaves, May 2026 Vacation, ...
    
    Returns: (pivoted_data, pivoted_columns) or (data, columns) if no pivot needed.
    """
    if not data or len(data) < 2 or len(columns) < 2:
        return data, columns
    
    cols_lower = [c.lower() for c in columns]
    
    # Detect month and entity columns
    month_col = None
    entity_col = None
    metric_cols = []
    
    for i, c in enumerate(cols_lower):
        if c in ("month", "quarter") and month_col is None:
            month_col = columns[i]
        elif c in ("employee", "project") and entity_col is None:
            entity_col = columns[i]
        else:
            metric_cols.append(columns[i])
    
    # Must have month column and at least one metric
    if not month_col or not metric_cols:
        return data, columns
    
    # Collect unique months
    months_seen = []
    for row in data:
        m = row.get(month_col)
        if m and m not in months_seen:
            months_seen.append(m)
    
    if len(months_seen) < 2:
        return data, columns
    
    # Case 1: 2-column table (Month + Metric) - no entity column
    # Convert vertical to horizontal: Month | Revenue -> Jan 2026 | Feb 2026 | Mar 2026
    if len(columns) == 2 and not entity_col:
        metric_col = metric_cols[0]
        pivoted_columns = months_seen.copy()
        new_row = {}
        for row in data:
            m = row.get(month_col)
            v = row.get(metric_col)
            if m and v is not None:
                new_row[m] = v
        # Only pivot if we have values for all months
        if len(new_row) == len(months_seen):
            return [new_row], pivoted_columns
        else:
            # Not all months have values, keep original format
            return data, columns
    
    # Case 2: 3+ column table with entity - need entity column
    if not entity_col:
        return data, columns
    
    # Check if there are repeating entities (same entity appears multiple times)
    entities = [row.get(entity_col) for row in data]
    if len(entities) == len(set(entities)):
        # No repetition, no need to pivot
        return data, columns
    
    # Collect unique entities (preserve order)
    entities_seen = []
    for row in data:
        e = row.get(entity_col)
        if e and e not in entities_seen:
            entities_seen.append(e)
    
    # Build pivoted data: entity as row, months+metrics as columns
    pivot_map = {}  # {entity: {month: {metric: value}}}
    for row in data:
        e = row.get(entity_col)
        m = row.get(month_col)
        if e not in pivot_map:
            pivot_map[e] = {}
        if m not in pivot_map[e]:
            pivot_map[e][m] = {}
        for metric in metric_cols:
            pivot_map[e][m][metric] = row.get(metric, "")
    
    # Create pivoted rows and columns
    if len(metric_cols) == 1:
        # Single metric: columns are just months
        # Employee | May 2026 | June 2026
        pivoted_columns = [entity_col] + months_seen
        pivoted_data = []
        for entity in entities_seen:
            new_row = {entity_col: entity}
            for month in months_seen:
                metric = metric_cols[0]
                new_row[month] = pivot_map.get(entity, {}).get(month, {}).get(metric, "")
            pivoted_data.append(new_row)
    else:
        # Multiple metrics: columns are month+metric combinations
        # Employee | May 2026 Leaves | May 2026 Vacation | June 2026 Leaves | June 2026 Vacation
        pivoted_columns = [entity_col]
        for month in months_seen:
            for metric in metric_cols:
                col_name = f"{month} {_format_column_name(metric)}"
                pivoted_columns.append(col_name)
        
        pivoted_data = []
        for entity in entities_seen:
            new_row = {entity_col: entity}
            for month in months_seen:
                for metric in metric_cols:
                    col_name = f"{month} {_format_column_name(metric)}"
                    new_row[col_name] = pivot_map.get(entity, {}).get(month, {}).get(metric, "")
            pivoted_data.append(new_row)
    
    return pivoted_data, pivoted_columns


def _format_data_keys(data: list, columns: list) -> list:
    """Rename keys in data rows to match formatted column names and format values."""
    if not data or not columns:
        return data
    
    # Build mapping from original to formatted
    key_map = {}
    for col in columns:
        formatted = _format_column_name(col)
        if formatted != col:
            key_map[col] = formatted
    
    # Rename keys and format values in each row
    formatted_data = []
    for row in data:
        new_row = {}
        for k, v in row.items():
            new_key = key_map.get(k, k)
            # Format the value based on the original key type
            formatted_value = _format_value(k, v)
            new_row[new_key] = formatted_value
        formatted_data.append(new_row)
    
    return formatted_data


def _forecast_table_name(header: str) -> str:
    """Convert forecast headers to clean table names like 'Q3 2026 Forecast' or 'May-Jul 2026 Forecast'."""
    if not header:
        return "Forecast"
    
    # Extract the period from headers like "Estimate — Q3 2026 —" or "Estimate — May 2026 to July 2026 —"
    m = re.search(r"(?i)estimate\s+—\s*(.*?)\s*—", header)
    if m:
        period = m.group(1).strip()
        # Clean up period: "May 2026 to July 2026" -> "May-Jul 2026"
        range_match = re.match(r"(\w+)\s+(\d{4})\s+to\s+(\w+)\s+(\d{4})", period)
        if range_match:
            m1, y1, m2, y2 = range_match.groups()
            if y1 == y2:
                period = f"{m1[:3]}-{m2[:3]} {y1}"
            else:
                period = f"{m1[:3]} {y1}-{m2[:3]} {y2}"
        return f"{period} Forecast"
    
    # Try to extract quarter or month pattern
    q_match = re.search(r"(Q[1-4]\s*\d{4})", header, re.IGNORECASE)
    if q_match:
        return f"{q_match.group(1)} Forecast"
    
    m_match = re.search(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})", header, re.IGNORECASE)
    if m_match:
        return f"{m_match.group(1)} Forecast"
    
    # Fallback: remove "Estimate" prefix
    header2 = re.sub(r"(?i)^estimate\s*[-—:]\s*", "", header).strip()
    header2 = re.sub(r"\s*[-—]\s*$", "", header2).strip()
    return f"{header2} Forecast" if header2 else "Forecast"


def _concise_table_name_from_summary(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return "Results"
    # Limit length for a compact table title
    if len(s) > 80:
        s = s[:80].rstrip()
    if not re.match(r"(?i)^(forecast|results)\s+—\s*", s):
        s = f"Results — {s}"
    return s


def _llm_table_name(summary: str, columns, data) -> str:
    """Generate a clean, relevant table name from columns and summary."""
    cols = [str(c or "").strip() for c in (columns or [])]
    cols_l = [c.lower() for c in cols]
    s = (summary or "").strip()
    
    # Detect entity dimension
    entity_map = {
        "employee": "Employees",
        "project": "Projects",
        "month": "Monthly",
        "quarter": "Quarterly",
        "department": "Departments",
        "client": "Clients",
        "team": "Teams",
    }
    entity = None
    entity_key = None
    for k, v in entity_map.items():
        if k in cols_l:
            entity = v
            entity_key = k
            break
    
    # Detect metrics (can have multiple)
    metric_aliases = [
        ("utilization", "Utilization"), ("utilisation", "Utilization"), ("utilization_pct", "Utilization"),
        ("profit", "Profit"), ("revenue", "Revenue"), ("cost", "Cost"),
        ("hours", "Hours"), ("hrs", "Hours"), ("total_hours", "Hours"),
        ("leave", "Leaves"), ("leaves", "Leaves"), ("leave_days", "Leaves"), ("pto", "Leaves"),
        ("vacation", "Vacation"), ("vacation_days", "Vacation"),
        ("holidays", "Holidays"), ("holiday", "Holidays"), ("holiday_days", "Holidays"),
        ("working_days", "Working Days"), ("workdays", "Working Days"),
        ("margin", "Margin"), ("margin_pct", "Margin"),
        ("gross_margin", "Gross Margin"), ("gross_margin_pct", "Gross Margin"),
        ("avg_margin", "Avg Margin"), ("avg_margin_pct", "Avg Margin"),
        ("headcount", "Headcount"),
    ]
    metrics = []
    for key, label in metric_aliases:
        if key in cols_l and label not in metrics:
            metrics.append(label)
    
    # Ranking detection from summary
    rank_match = re.search(r"(?i)\b(top|highest|bottom|lowest)\b\s*(\d+)?", s)
    rank_word = None
    n = None
    if rank_match:
        w = rank_match.group(1).lower()
        rank_word = "Top" if w in ("top", "highest") else "Bottom"
        if rank_match.group(2):
            try:
                n = int(rank_match.group(2))
            except Exception:
                pass
    
    # Infer count from data if not in summary
    if n is None and isinstance(data, list) and 1 <= len(data) <= 50:
        if rank_word:
            n = len(data)
    
    # Detect comparison/trend keywords
    is_comparison = bool(re.search(r"(?i)\b(compar|vs|versus|between|difference)\b", s))
    is_breakdown = bool(re.search(r"(?i)\b(breakdown|by month|by project|by employee|distribution)\b", s))
    is_summary = bool(re.search(r"(?i)\b(summary|overview|total|all)\b", s))
    
    # Build clean title
    metric_str = " & ".join(metrics[:2]) if metrics else None  # Limit to 2 metrics
    
    # Ranked results: "Top 5 Employees by Profit"
    if rank_word and n:
        if entity and metric_str:
            return f"{rank_word} {n} {entity} by {metric_str}"
        elif entity:
            return f"{rank_word} {n} {entity}"
        elif metric_str:
            return f"{rank_word} {n} by {metric_str}"
    
    # Breakdown: "Revenue by Project" or "Monthly Profit"
    if is_breakdown or (entity and metric_str):
        if entity_key in ("month", "quarter"):
            return f"{entity} {metric_str}" if metric_str else f"{entity} Data"
        elif entity and metric_str:
            return f"{metric_str} by {entity}"
        elif entity:
            return f"{entity} Data"
    
    # Comparison
    if is_comparison:
        if entity and metric_str:
            return f"{entity} {metric_str} Comparison"
        elif metric_str:
            return f"{metric_str} Comparison"
    
    # Summary/Overview
    if is_summary:
        if metric_str:
            return f"{metric_str} Summary"
        elif entity:
            return f"{entity} Summary"
    
    # Generic fallback based on columns
    if entity and metric_str:
        return f"{entity} {metric_str}"
    elif entity:
        return f"{entity} Data"
    elif metric_str:
        return metric_str
    
    # Last resort: clean up the summary
    if s:
        # Remove common prefixes
        s = re.sub(r"(?i)^(results?\s*[-—:]?\s*|the\s+|here\s+(is|are)\s+(the\s+)?)", "", s)
        s = re.sub(r"(?i)\s+(are|is)\s+listed\s+below\.?$", "", s)
        s = re.sub(r"(?i)\s+based\s+on\s+.*$", "", s)
        s = s.strip(" .,:-—")
        if s and len(s) <= 60:
            return s.title() if len(s) < 30 else s[:60]
    
    return "Data"

# ── Main Q&A function ───────────────────────────────────────────────────────

def ask(question: str, time_range: str = None) -> dict:
    """
    Answer a natural-language question about the dataset.

    Returns: {"summary": str, "visual_type": str, "columns": list, "data": list, "sources": dict}
    """
    _start_time = time.time()
    
    records = GLOBAL_DATASET
    if not records:
        return {
            "summary": "No data loaded. Please upload timesheets first via POST /ingest.",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    # Apply time range filter if specified
    if time_range:
        records = filter_by_range(records, time_range)

    # Forecast intent: use AI forecaster (forecast_.py) only
    # Note: try_answer_forecast now uses fast keyword check internally
    _fc_start = time.time()
    fc_answer = try_answer_forecast(question, records)
    print(f"[qa_engine] Forecast check took {time.time() - _fc_start:.2f}s (likely_forecast={is_likely_forecast(question)})")
    if fc_answer is not None:
        rows = _extract_rows(fc_answer)
        if rows:
            preferred_cols = [
                "month", "quarter",
                "project", "employee",
                "revenue", "cost", "profit", "headcount", "utilization_pct",
                "leave_days", "vacation_days", "holiday_days", "working_days", "hours",
            ]
            present = []
            for c in preferred_cols:
                if any(c in r for r in rows):
                    present.append(c)
            data = [{k: r.get(k) for k in present} for r in rows]
            header = fc_answer.splitlines()[0].strip()
            table_name = _forecast_table_name(header)
            
            # Pivot table if it has repeating entity+month combinations
            data, present = _pivot_table_if_needed(data, present)
            
            # Format column names and data keys
            formatted_cols = _format_columns(present)
            formatted_data = _format_data_keys(data, present)
            return {
                "summary": table_name,
                "visual_type": "table",
                "columns": formatted_cols,
                "data": formatted_data,
                "sources": {
                    "total_records": len(records),
                    "months": get_months_available(records),
                    "time_range": time_range or "ALL",
                    "forecast": True,
                },
            }
        else:
            return {
                "summary": fc_answer,
                "visual_type": "text",
                "columns": [],
                "data": [],
                "sources": {
                    "total_records": len(records),
                    "months": get_months_available(records),
                    "time_range": time_range or "ALL",
                    "forecast": True,
                },
            }

    # Check Ollama availability
    if not is_ollama_available():
        return {
            "summary": "LLM service (Ollama) is not running. Please start it with: ollama serve",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    # Build targeted context based on question (smart context selection)
    _ctx_start = time.time()
    context = _get_cached_context(records, question=question)
    print(f"[qa_engine] Context build took {time.time() - _ctx_start:.2f}s")
    
    prompt = _QA_PROMPT.format(context=context, question=question)

    try:
        _llm_start = time.time()
        raw_response = _ollama_generate(prompt, timeout=120)
        print(f"[qa_engine] LLM call took {time.time() - _llm_start:.2f}s")
    except Exception as e:
        return {
            "summary": f"LLM query failed: {str(e)}",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    print(f"[qa_engine] Total time: {time.time() - _start_time:.2f}s")
    # Parse the JSON from the LLM
    parsed_json = _extract_qa_json(raw_response)
    
    # Validate and fix LLM response structure
    if not parsed_json or not isinstance(parsed_json, dict):
        parsed_json = {
            "summary": _clean_answer(raw_response),
            "visual_type": "text",
            "data": [],
            "columns": []
        }
    
    # Ensure required fields exist with correct types
    if "summary" not in parsed_json or not isinstance(parsed_json.get("summary"), str):
        parsed_json["summary"] = _clean_answer(raw_response) if raw_response else "Unable to process response."
    
    if "visual_type" not in parsed_json or parsed_json.get("visual_type") not in ("table", "metric", "text", "bar_chart"):
        # Infer visual_type from data structure
        data = parsed_json.get("data", [])
        if isinstance(data, list) and len(data) == 1 and "label" in data[0] and "value" in data[0]:
            parsed_json["visual_type"] = "metric"
        elif isinstance(data, list) and len(data) > 1:
            parsed_json["visual_type"] = "table"
        else:
            parsed_json["visual_type"] = "text"
    
    if "columns" not in parsed_json or not isinstance(parsed_json.get("columns"), list):
        parsed_json["columns"] = []
    
    if "data" not in parsed_json or not isinstance(parsed_json.get("data"), list):
        parsed_json["data"] = []
    
    # For table type, ensure columns match data keys
    visual_type = parsed_json.get("visual_type", "text")
    raw_data = parsed_json.get("data", [])
    raw_columns = parsed_json.get("columns", [])
    
    if visual_type == "table" and raw_data and not raw_columns:
        # Extract columns from first data row
        if isinstance(raw_data[0], dict):
            raw_columns = list(raw_data[0].keys())
            parsed_json["columns"] = raw_columns
    
    # Convert "metric/value" table to proper metric format
    if visual_type == "table" and raw_data:
        cols_lower = [c.lower() for c in raw_columns]
        if cols_lower == ["metric", "value"] or cols_lower == ["label", "value"]:
            # This is actually a metric, not a table
            visual_type = "metric"
            parsed_json["visual_type"] = "metric"
            # Convert to label/value format
            new_data = []
            for row in raw_data:
                label = row.get("metric") or row.get("label") or row.get("Metric") or row.get("Label") or ""
                value = row.get("value") or row.get("Value") or ""
                # Skip if value is a list (invalid)
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                new_data.append({"label": str(label), "value": value})
            raw_data = new_data
            raw_columns = []

    summary_text = parsed_json.get("summary", "")
    
    if visual_type == "table":
        summary_text = _llm_table_name(summary_text, raw_columns, raw_data)
        
        # Pivot table if it has repeating entity+month combinations
        raw_data, raw_columns = _pivot_table_if_needed(raw_data, raw_columns)
        
        # Format column names and data keys
        formatted_cols = _format_columns(raw_columns)
        formatted_data = _format_data_keys(raw_data, raw_columns)
    elif visual_type == "metric":
        # Format metric value if it's a number
        formatted_cols = raw_columns
        formatted_data = []
        for item in raw_data:
            if isinstance(item, dict) and "label" in item and "value" in item:
                label = item["label"]
                value = item["value"]
                # Try to format numeric values
                if isinstance(value, (int, float)):
                    label_lower = label.lower()
                    if any(w in label_lower for w in ["revenue", "cost", "profit", "salary", "budget", "amount"]):
                        value = f"${value:,.2f}"
                    elif any(w in label_lower for w in ["percent", "rate", "utilization", "margin"]):
                        value = f"{value:.1f}%"
                    elif any(w in label_lower for w in ["count", "number", "total", "employee", "headcount"]):
                        value = f"{int(value):,}"
                    else:
                        value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                formatted_data.append({"label": label, "value": str(value)})
            else:
                formatted_data.append(item)
    else:
        formatted_cols = raw_columns
        formatted_data = raw_data
    
    # Construct the final, UI-friendly payload
    return {
        "summary": summary_text,
        "visual_type": visual_type,
        "columns": formatted_cols,
        "data": formatted_data,
        "sources": {
            "total_records": len(records),
            "months": get_months_available(records),
            "time_range": time_range or "ALL",
            "forecast": False,
        }
    }

