from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from ingestion.ingest import ingest_file
from ingestion.dataset import (
    GLOBAL_DATASET, transform_to_api_format, filter_by_range,
    build_projects, build_monthly, build_overall_summary,
    build_top_performers, build_risks, get_months_available,
    clear as clear_dataset,
)
import tempfile
import os

app = FastAPI(title="FinIntel AI", description="AI-powered Financial Document Intelligence System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /ingest — Upload and process file(s) ──────────────────────────
@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    results = []

    for file in files:
        ext = os.path.splitext(file.filename)[1]
        if not ext:
            ext = ".csv"

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = ingest_file(tmp_path, original_filename=file.filename)

        results.append({
            "filename": file.filename,
            "result": result,
        })

    return {
        "status": "success",
        "files_processed": len(results),
        "total_records_in_dataset": len(GLOBAL_DATASET),
        "months_available": get_months_available(),
        "results": results,
    }


# ── POST /analyze — Alias for /ingest (backward compat) ────────────────
@app.post("/analyze")
async def analyze(files: List[UploadFile] = File(...)):
    return await ingest(files)


# ── GET /dataset — Full unified dataset ─────────────────────────────────
@app.get("/dataset")
def get_dataset(range: Optional[str] = Query(None, description="1M, 3M, 6M, 12M, or ALL")):
    return transform_to_api_format(time_range=range)


# ── GET /metrics — Overall summary metrics ──────────────────────────────
@app.get("/metrics")
def get_metrics(range: Optional[str] = Query(None)):
    filtered = filter_by_range(GLOBAL_DATASET, range)
    return {
        "time_range": range or "ALL",
        "overall_summary": build_overall_summary(filtered),
        "monthly": build_monthly(filtered),
    }


# ── GET /projects — Project-level breakdown ─────────────────────────────
@app.get("/projects")
def get_projects(range: Optional[str] = Query(None)):
    filtered = filter_by_range(GLOBAL_DATASET, range)
    return {
        "time_range": range or "ALL",
        "projects": build_projects(filtered),
    }


# ── GET /employees — Employee-level records ─────────────────────────────
@app.get("/employees")
def get_employees(range: Optional[str] = Query(None), project: Optional[str] = Query(None)):
    filtered = filter_by_range(GLOBAL_DATASET, range)
    if project:
        filtered = [r for r in filtered if r.get("project", "").lower() == project.lower()]
    employees = []
    for r in filtered:
        employees.append({
            "employee": r.get("employee"),
            "project": r.get("project"),
            "month": r.get("month"),
            "actual_hours": r.get("actual_hours", 0),
            "billable_hours": r.get("billable_hours", 0),
            "working_days": r.get("working_days", 0),
            "vacation_days": r.get("vacation_days", 0),
            "revenue": r.get("revenue", 0),
            "cost": r.get("cost", 0),
            "profit": r.get("profit", 0),
            "margin_pct": r.get("margin_pct", 0),
            "utilisation_pct": r.get("utilisation_pct", 0),
            "is_profitable": r.get("is_profitable", True),
        })
    return {
        "time_range": range or "ALL",
        "count": len(employees),
        "employees": employees,
    }


# ── GET /risks — All risk flags ─────────────────────────────────────────
@app.get("/risks")
def get_risks():
    return {
        "total_records": len(GLOBAL_DATASET),
        "risks": build_risks(GLOBAL_DATASET),
    }


# ── POST /ask — Natural language Q&A ────────────────────────────────────
class AskRequest(BaseModel):
    query: str


@app.post("/ask")
def ask_question(req: AskRequest):
    answer, sources = _handle_qa(req.query)
    return {"answer": answer, "sources": sources}


# ── DELETE /dataset — Reset ─────────────────────────────────────────────
@app.delete("/dataset")
def reset_dataset():
    clear_dataset()
    return {"status": "cleared", "records": 0}


# ── Q&A Engine (deterministic keyword-based) ────────────────────────────
def _handle_qa(query: str):
    q = query.lower()
    records = GLOBAL_DATASET

    if not records:
        return "No data loaded. Please upload timesheets first via POST /ingest.", []

    if "risk" in q or "at risk" in q:
        risks = build_risks(records)
        if not risks:
            return "No risks detected. All employees are profitable with healthy metrics.", []
        summary_lines = []
        for r in risks[:10]:
            summary_lines.append(f"- {r['employee']} ({r.get('month', '')}): {r['issue']}")
        return f"Found {len(risks)} risk(s):\n" + "\n".join(summary_lines), risks[:10]

    if "underperform" in q or "low perform" in q or "worst" in q:
        emp_profit = {}
        for r in records:
            name = r.get("employee", "")
            emp_profit[name] = emp_profit.get(name, 0) + r.get("profit", 0)
        ranked = sorted(emp_profit.items(), key=lambda x: x[1])
        bottom = ranked[:5]
        lines = [f"- {name}: ${pft:,.2f} profit" for name, pft in bottom]
        return "Lowest performing employees by profit:\n" + "\n".join(lines), [{"employee": n, "profit": round(p, 2)} for n, p in bottom]

    if "top perform" in q or "best" in q:
        top = build_top_performers(records, limit=5)
        lines = [f"- {t['employee']}: ${t['total_profit']:,.2f} profit" for t in top]
        return "Top performing employees:\n" + "\n".join(lines), top

    if "project" in q:
        projects = build_projects(records)
        lines = []
        for name, data in sorted(projects.items(), key=lambda x: x[1]["total_profit"], reverse=True):
            lines.append(f"- {name}: Revenue ${data['total_revenue']:,.2f}, Profit ${data['total_profit']:,.2f}, Margin {data['avg_margin_pct']}%")
        return "Project summary:\n" + "\n".join(lines), projects

    if any(w in q for w in ["last 3 month", "3m", "3 month", "quarter"]):
        filtered = filter_by_range(records, "3M")
        summary = build_overall_summary(filtered)
        months = get_months_available(filtered)
        return (
            f"Last 3 months ({', '.join(months)}):\n"
            f"- Revenue: ${summary['total_revenue']:,.2f}\n"
            f"- Cost: ${summary['total_cost']:,.2f}\n"
            f"- Profit: ${summary['total_profit']:,.2f}\n"
            f"- Margin: {summary['avg_margin_pct']}%\n"
            f"- Employees: {summary['total_employees']}"
        ), summary

    if any(w in q for w in ["revenue", "cost", "profit", "margin", "summary", "overall"]):
        summary = build_overall_summary(records)
        months = get_months_available(records)
        return (
            f"Overall summary ({len(months)} months):\n"
            f"- Revenue: ${summary['total_revenue']:,.2f}\n"
            f"- Cost: ${summary['total_cost']:,.2f}\n"
            f"- Profit: ${summary['total_profit']:,.2f}\n"
            f"- Margin: {summary['avg_margin_pct']}%\n"
            f"- Employees: {summary['total_employees']}"
        ), summary

    if "employee" in q or "who" in q:
        unique = sorted({r.get("employee", "") for r in records})
        return f"Employees in dataset ({len(unique)}):\n" + "\n".join(f"- {e}" for e in unique), unique

    if "month" in q:
        months = get_months_available(records)
        return f"Available months ({len(months)}):\n" + "\n".join(f"- {m}" for m in months), months

    return (
        "I can answer questions about:\n"
        "- 'Which project is at risk?'\n"
        "- 'Who is underperforming?'\n"
        "- 'Show top performers'\n"
        "- 'Show last 3 months performance'\n"
        "- 'What is the overall revenue?'\n"
        "- 'List all employees'\n"
        "- 'Show project summary'\n"
        "\nPlease rephrase your question."
    ), []
