from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from ingestion.ingest import ingest_file
from ingestion.dataset import (
    GLOBAL_DATASET, transform_to_api_format, filter_by_range,
    build_projects, build_monthly, build_overall_summary,
    build_top_performers, get_months_available,
    clear as clear_dataset,
)
from ingestion.qa_engine import ask as qa_ask
from ingestion.risk_engine import get_risks_and_recommendations
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
        result.pop("overall_summary", None)

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
        "overall_summary": build_overall_summary(GLOBAL_DATASET),
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



# ── GET /risks-recommendations — Risk analysis + AI recommendations ──────
@app.get("/risks-recommendations")
def risks_and_recommendations(range: Optional[str] = None):
    return get_risks_and_recommendations(time_range=range)


# ── POST /ask — Natural language Q&A (LLM-powered) ──────────────────────
class AskRequest(BaseModel):
    query: str
    time_range: Optional[str] = None


@app.post("/ask")
def ask_question(req: AskRequest):
    return qa_ask(req.query, time_range=req.time_range)


# ── DELETE /dataset — Reset ─────────────────────────────────────────────
@app.delete("/dataset")
def reset_dataset():
    clear_dataset()
    return {"status": "cleared", "records": 0}


