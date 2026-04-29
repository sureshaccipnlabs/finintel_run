"""
qa_engine.py — LLM-powered Q&A over the ingested financial dataset.

Builds a compact text summary of GLOBAL_DATASET and sends it as context
to Ollama so the LLM can answer any natural-language question about the data.
"""

import re

from .dataset import (
    GLOBAL_DATASET, build_projects, build_monthly, build_overall_summary,
    build_top_performers, build_risks, get_months_available, filter_by_range,
)
from .ai_mapper import _llm_generate, is_llm_available


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

    lines = []

    # Overall
    lines.append("=== OVERALL SUMMARY ===")
    lines.append(f"Total Revenue: ${overall['total_revenue']:,.2f}")
    lines.append(f"Total Cost: ${overall['total_cost']:,.2f}")
    lines.append(f"Total Profit: ${overall['total_profit']:,.2f}")
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

    # Per-employee detail
    lines.append("=== EMPLOYEE DETAILS ===")
    for r in recs:
        lines.append(
            f"- {r.get('employee', '?')} | Project: {r.get('project', '?')} | "
            f"Month: {r.get('month', '?')} | Hours: {r.get('actual_hours', 0)} | "
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
You are FinIntel AI, a financial data analyst assistant. You answer questions \
about uploaded timesheet and financial data.

Here is the current dataset:

{context}

---

User question: {question}

Instructions:
- Answer using ONLY the dataset above.
- Do NOT infer, generalize, summarize, or combine records unless the user explicitly asks for a summary or aggregation.
- Do NOT use phrases like "multiple months", "appears", "indicates", "suggests", or any similar interpretive wording unless that exact conclusion is directly supported by the data and requested by the user.
- Do not repeat the user’s question.
- Do not restate or paraphrase the question.
- Start directly with the answer.
- For superlative questions like highest, lowest, max, minimum, return only the record(s) tied for the extreme value.
- If the question asks who, which employee, which month, or similar, answer with the exact matching rows from the dataset.
- If an employee appears in more than one month, list each employee-month separately unless the user explicitly asks for grouped results.
- If the question asks "how many", "count", "number of", or "total number of", return a count first, not a row dump.
- For count questions about employees, count unique employee names unless the user explicitly asks for employee-month records, rows, entries, or occurrences.
- For count questions about projects, count unique project names unless the user explicitly asks for rows, entries, or occurrences.
- If the question is ambiguous, prefer the business entity count rather than the row count.
- Do not mix unique counts with row counts in the same answer unless the user asks for both.
- If helpful, after the count you may list the unique names included in that count on the same line or in a short plain-text continuation.
- If there are no matching rows, say that no matching records were found in the provided data.
- If the data does not contain enough information to answer, say so plainly.
- Be concise and precise. Use exact employee names, project names, months, numbers, currency, and percentages from the data.
- Format currency as $X,XXX.XX and percentages as X.X%.
- When comparing values, show both values side by side.
- Do not explain what the numbers mean unless the user explicitly asks for analysis.
- Prefer bullet points for lists.
- Do not start the answer with phrases like "Based on the dataset", "Based on the provided data", "From the data", or similar lead-ins.
- Do not output escaped newline characters like \n or \t.
- Do not use markdown emphasis such as **bold**, *italic*, headings, or code fences.
- Return plain text only.

Output rules:
- For aggregated questions, show the calculation result only if it can be read or computed directly from the provided data.
- For count questions, prefer formats like:
  <N> employees are underutilized.
  <N> unique employees are underutilized: <Name1>, <Name2>, <Name3>.
"""


_FORECAST_PROMPT = """\
You are FinIntel AI, a financial forecasting assistant.

Here is the current dataset:

{context}

---

User question: {question}

Forecasting mode instructions:
- The user is explicitly asking for future projection. Provide a forecast answer.
- Use ONLY the data provided above and the assumptions below.
- Assumption 1: Future monthly actual hours remain equal to current observed level per employee.
- Assumption 2: Billing rate and cost rate remain unchanged from current observed level per employee.
- Assumption 3: If multiple records exist for an employee, use a simple average of observed values.
- Compute monthly forecast per employee using:
  revenue = actual_hours * billing_rate
  cost = actual_hours * cost_rate
  profit = revenue - cost
- Forecast for next 3 months with these assumptions and provide concise results.
- Clearly include one short disclaimer: assumptions-based forecast from historical data only.
- Return plain text only (no markdown emphasis, no code blocks).
"""


def _is_forecast_question(question: str) -> bool:
    q = (question or "").lower()
    keywords = (
        "forecast", "predict", "projection", "projected", "next month", "next 3",
        "next three", "upcoming", "future", "how will", "will be performance",
    )
    return any(k in q for k in keywords)


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
    answer = answer
    answer = answer
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = answer
    answer = re.sub(r"\s{2,}", " ", answer)
    return answer.strip()


# ── Main Q&A function ───────────────────────────────────────────────────────

def ask(question: str, time_range: str = None) -> dict:
    """
    Answer a natural-language question about the dataset.

    Returns: {"answer": str, "sources": dict}
    """
    records = GLOBAL_DATASET
    if not records:
        return {
            "answer": "No data loaded. Please upload timesheets first via POST /ingest.",
            "sources": [],
        }

    # Apply time range filter if specified
    if time_range:
        records = filter_by_range(records, time_range)

    # Check LLM availability based on configured provider preference
    if not is_llm_available():
        return {
            "answer": "No configured LLM provider is available. Set AI_PROVIDER and related env vars (Ollama/OpenAI).",
            "sources": [],
        }

    # Build context and prompt
    context = _build_dataset_context(records)
    prompt_template = _FORECAST_PROMPT if _is_forecast_question(question) else _QA_PROMPT
    prompt = prompt_template.format(context=context, question=question)

    try:
        answer = _llm_generate(prompt, timeout=120)
    except Exception as e:
        return {
            "answer": f"LLM query failed: {str(e)}",
            "sources": [],
        }

    # Return answer with metadata
    return {
        "answer": _clean_answer(answer),
        "sources": {
            "total_records": len(records),
            "months": get_months_available(records),
            "time_range": time_range or "ALL",
        },
    }
