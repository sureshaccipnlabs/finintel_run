"""
qa_engine.py — LLM-powered Q&A over the ingested financial dataset.

Builds a compact text summary of GLOBAL_DATASET and sends it as context
to Ollama so the LLM can answer any natural-language question about the data.
"""

import re

from .dataset import (
    GLOBAL_DATASET, build_projects, build_monthly, build_overall_summary,
    build_top_performers, build_risks, get_months_available, filter_by_range,
    build_employee_summaries
)
from .ai_mapper import _ollama_generate, is_ollama_available, _extract_json
from .forecast_ import try_answer_forecast


# ── Context builder ──────────────────────────────────────────────────────────

def _build_dataset_context(records=None):
    """Build a compact text summary of the dataset for the LLM prompt."""
    recs = records if records is not None else GLOBAL_DATASET
    if not recs:
        return "No data has been loaded yet."

    overall = build_overall_summary(recs)
    projects = build_projects(recs)
    months = get_months_available(recs)
    monthly = build_monthly(recs)
    top = build_top_performers(recs, limit=5)
    risks = build_risks(recs)
    employee_summaries = build_employee_summaries(recs)

    lines = []

    # Overall
    lines.append("=== OVERALL SUMMARY ===")
    lines.append(f"Total Revenue: ${overall['total_revenue']:,.2f}")
    lines.append(f"Total Cost: ${overall['total_cost']:,.2f}")
    lines.append(f"Total Profit: ${overall['total_profit']:,.2f}")
    lines.append(f"Total Hours Worked: {overall.get('total_hours', 0)}")
    lines.append(f"Avg Margin: {overall['avg_margin_pct']}%")
    lines.append(f"Total Employees: {overall['total_employees']}")
    lines.append(f"Months: {', '.join(months)}")
    lines.append("")

    # Projects
    lines.append("=== PROJECTS ===")
    for name, data in sorted(projects.items(), key=lambda x: x[1]["revenue"], reverse=True):
        lines.append(
            f"- {name}: Revenue=${data['revenue']:,.2f}, "
            f"Cost=${data['cost']:,.2f}, Profit=${data['profit']:,.2f}, "
            f"Avg Utilisation={data.get('avg_utilisation', 0)}%, "
            f"Employees={data['employees']}"
        )
    lines.append("")

    # Monthly
    lines.append("=== MONTHLY BREAKDOWN ===")
    for m, data in monthly.items():
        lines.append(
            f"- {m}: Revenue=${data['total_revenue']:,.2f}, "
            f"Cost=${data['total_cost']:,.2f}, Profit=${data['total_profit']:,.2f}, "
            f"Margin={data['avg_margin_pct']}%, Employees={data['employees']}"
        )
    lines.append("")

    # Employee Totals
    lines.append("=== EMPLOYEE ALL-TIME TOTALS ===")
    for emp in employee_summaries:
        lines.append(
            f"- {emp['employee_name']}: Total Revenue=${emp['total_revenue']:,.2f}, "
            f"Total Profit=${emp['total_profit']:,.2f}, Total Hours={emp['total_hours']}, "
            f"Utilisation={emp['utilization_pct'] or 0}%"
        )
    lines.append("")

    # Per-employee detail
    lines.append("=== EMPLOYEE MONTHLY DETAILS ===")
    for r in recs:
        lines.append(
            f"- {r.get('employee', '?')} | Proj: {r.get('project', '?')} | "
            f"Month: {r.get('month', '?')} | "
            f"ActualHrs: {r.get('actual_hours', 0)} | BillableHrs: {r.get('billable_hours', 0)} | "
            f"WorkDays: {r.get('working_days', 0)} | Vacation: {r.get('vacation_days', 0)} | "
            f"BillRate: {r.get('billing_rate', 0)} | CostRate: {r.get('cost_rate', 0)} | "
            f"Revenue: {r.get('revenue', 0)} | Cost: {r.get('cost', 0)} | "
            f"Profit: {r.get('profit', 0)} | Margin: {r.get('margin_pct', 0)}% | "
            f"Utilisation: {r.get('utilisation_pct', 0)}%"
        )
    lines.append("")

    # Top performers
    lines.append("=== TOP PERFORMERS (by profit) ===")
    for t in top:
        lines.append(f"- {t['employee']}: ${t['total_profit']:,.2f}")
    lines.append("")

    # Risks
    if risks:
        lines.append("=== RISK FLAGS ===")
        for r in risks:
            lines.append(f"- {r['employee']} ({r.get('month', '')}): {r['issue']}")
    else:
        lines.append("=== RISK FLAGS ===")
        lines.append("No risks detected.")

    return "\n".join(lines)


# ── Prompt template ──────────────────────────────────────────────────────────

_QA_PROMPT = """\
You are FinIntel AI, a financial data analyst assistant. Answer questions based ONLY on the provided dataset context.

Here is the current dataset:
{context}

---
User question: {question}

INSTRUCTIONS:
You MUST return your answer as a valid, strictly formatted JSON object. Do not include any markdown formatting like ```json or outside text.

JSON SCHEMA:
{{
  "summary": "A clear, conversational 1-2 sentence answer to the user's question.",
  "visual_type": "table", // Choose exactly one: "table", "metric", "bar_chart", or "text"
  "columns": ["col1", "col2"], // Provide column keys if visual_type is 'table' or 'bar_chart'. Use lowercase keys.
  "data": [
      // If visual_type is "table" or "bar_chart": output an array of objects matching the columns.
      // If visual_type is "metric": output exactly one object like {{"label": "Total Revenue", "value": "$100,000"}}
      // If visual_type is "text": output an empty array []
  ]
}}

RULES:
1. "visual_type" must be "metric" for single numbers/counts (e.g., Total Revenue, Employee count).
2. "visual_type" must be "table" for lists, top/bottom performers, or grouped data.
3. For "table", ensure the keys in the "data" objects exactly match the strings in "columns".
4. Never make up data. If no data matches, set visual_type to "text" and explain in "summary".
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
                elif k.replace(" ", "") in ("quarterno", "quarterno", "quarter_no", "qno"):
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
                elif k.startswith("margin"):
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


def _forecast_table_name(header: str) -> str:
    # Convert headers like "Estimate — Q3 2026 —" to a concise table name: "Forecast — Q3 2026"
    if not header:
        return "Forecast"
    m = re.search(r"(?i)estimate\s+—\s*(.*?)\s*—", header)
    if m:
        core = m.group(1).strip()
        return f"Forecast — {core}"
    # Fallback: replace leading "Estimate" if present
    header2 = re.sub(r"(?i)^estimate\s+—\s*", "Forecast — ", header).strip()
    return header2 or "Forecast"


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
    cols = [str(c or "").strip() for c in (columns or [])]
    cols_l = [c.lower() for c in cols]
    # Detect entity dimension
    entity_map = {
        "employee": "Employees",
        "project": "Projects",
        "month": "Months",
        "quarter": "Quarters",
        "department": "Departments",
        "client": "Clients",
        "team": "Teams",
    }
    entity = None
    for k, v in entity_map.items():
        if k in cols_l:
            entity = v
            break
    # Detect metric
    metric_aliases = [
        ("utilization", "Utilization"), ("utilisation", "Utilization"),
        ("profit", "Profit"), ("revenue", "Revenue"), ("cost", "Cost"),
        ("hours", "Hours"), ("hrs", "Hours"),
        ("leave", "Leaves"), ("leaves", "Leaves"), ("pto", "Leaves"),
        ("vacation", "Vacation"), ("holidays", "Holidays"), ("holiday", "Holidays"),
        ("working days", "Working Days"), ("working_days", "Working Days"), ("workdays", "Working Days"),
        ("margin", "Margin"), ("margin_pct", "Margin"),
    ]
    metric = None
    for key, label in metric_aliases:
        if key in cols_l:
            metric = label
            break
    # Ranking detection
    s = (summary or "").strip()
    m = re.search(r"(?i)\b(top|highest|bottom|lowest)\b\s*(\d+)?", s)
    rank_word = None
    n = None
    if m:
        w = m.group(1).lower()
        if w in ("top", "highest"):
            rank_word = "Top"
        elif w in ("bottom", "lowest"):
            rank_word = "Bottom"
        if m.group(2):
            try:
                n = int(m.group(2))
            except Exception:
                n = None
    if n is None and isinstance(data, list) and 1 <= len(data) <= 50:
        # Heuristic: if a small list is returned and rank was implied, use its size
        if re.search(r"(?i)\b(top|highest|bottom|lowest)\b", s):
            n = len(data)
    # Build title
    if rank_word and n and (entity or metric):
        if entity and metric:
            return f"{rank_word} {n} — {entity} by {metric}"
        return f"{rank_word} {n} — {entity or metric}"
    if entity and metric:
        return f"Results — {metric} by {entity}"
    if entity:
        return f"Results — {entity}"
    if metric:
        return f"Results — {metric}"
    return _concise_table_name_from_summary(summary)

# ── Main Q&A function ───────────────────────────────────────────────────────

def ask(question: str, time_range: str = None) -> dict:
    """
    Answer a natural-language question about the dataset.

    Returns: {"summary": str, "visual_type": str, "columns": list, "data": list, "sources": dict}
    """
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
    fc_answer = try_answer_forecast(question, records)
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
            return {
                "summary": table_name,
                "visual_type": "table",
                "columns": present,
                "data": data,
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

    # Build context and prompt
    context = _build_dataset_context(records)
    prompt = _QA_PROMPT.format(context=context, question=question)

    try:
        raw_response = _ollama_generate(prompt, timeout=120)
    except Exception as e:
        return {
            "summary": f"LLM query failed: {str(e)}",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    # Parse the JSON from the LLM
    parsed_json = _extract_json(raw_response)
    
    # Fallback if LLM fails to return valid JSON
    if not parsed_json:
        parsed_json = {
            "summary": _clean_answer(raw_response),
            "visual_type": "text",
            "data": [],
            "columns": []
        }

    visual_type = parsed_json.get("visual_type", "text")
    summary_text = parsed_json.get("summary", "")
    if visual_type == "table":
        summary_text = _llm_table_name(summary_text, parsed_json.get("columns", []), parsed_json.get("data", []))
    # Construct the final, UI-friendly payload
    return {
        "summary": summary_text,
        "visual_type": visual_type,
        "columns": parsed_json.get("columns", []),
        "data": parsed_json.get("data", []),
        "sources": {
            "total_records": len(records),
            "months": get_months_available(records),
            "time_range": time_range or "ALL",
            "forecast": False,
        }
    }

