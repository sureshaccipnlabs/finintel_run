"""
qa_engine.py — LLM-powered Q&A over the ingested financial dataset.

Builds a compact text summary of GLOBAL_DATASET and sends it as context
to Ollama so the LLM can answer any natural-language question about the data.
"""

from .dataset import (
    GLOBAL_DATASET, build_projects, build_monthly, build_overall_summary,
    build_top_performers, build_risks, get_months_available, filter_by_range,
)
from .ai_mapper import _ollama_generate, is_ollama_available


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
about timesheet and financial data that has been uploaded.

Here is the current dataset:

{context}

---

User question: {question}

Instructions:
- Answer based ONLY on the data above. Do not make up numbers.
- Be concise and specific. Use actual numbers from the data.
- If the data doesn't contain enough information to answer, say so.
- Format currency as $X,XXX.XX and percentages as X.X%.
- When comparing, show both values side by side.
"""


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

    # Check Ollama availability
    if not is_ollama_available():
        return {
            "answer": "LLM service (Ollama) is not running. Please start it with: ollama serve",
            "sources": [],
        }

    # Build context and prompt
    context = _build_dataset_context(records)
    prompt = _QA_PROMPT.format(context=context, question=question)

    try:
        answer = _ollama_generate(prompt, timeout=120)
    except Exception as e:
        return {
            "answer": f"LLM query failed: {str(e)}",
            "sources": [],
        }

    # Return answer with metadata
    return {
        "answer": answer.strip(),
        "sources": {
            "total_records": len(records),
            "months": get_months_available(records),
            "time_range": time_range or "ALL",
        },
    }
