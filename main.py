from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from ingestion.ingest import ingest_file
from ingestion.dataset import (
    GLOBAL_DATASET, transform_to_api_format, filter_by_range,
    build_projects, build_monthly, build_overall_summary,
    build_top_performers, get_months_available,
    clear as clear_dataset, remove_by_file, get_files_processed,
    build_project_summaries, build_employee_summaries,
)
from ingestion.qa_engine import ask as qa_ask
from ingestion.risk_engine import get_risks_and_recommendations
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

_ingest_pool = ThreadPoolExecutor(max_workers=4)

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
    ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv", ".tsv", ".txt", ".pdf"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

    # Do not maintain upload history for now; each ingest replaces dataset state.
    clear_dataset()

    # Phase 1: Validate all files and write to temp (fast, sequential)
    staged = []
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if not ext:
            ext = ".csv"
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported file type '{ext}' for '{file.filename}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File '{file.filename}' is too large ({len(content) / 1024 / 1024:.1f} MB). Max: 50 MB.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(content)
            staged.append({"filename": file.filename, "tmp_path": tmp.name})

    # Phase 2: Process all files in parallel (heavy, concurrent)
    def _process_one(item):
        result = ingest_file(item["tmp_path"], original_filename=item["filename"])
        try:
            os.unlink(item["tmp_path"])
        except OSError:
            pass
        result.pop("overall_summary", None)
        return {"filename": item["filename"], "result": result}

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(_ingest_pool, _process_one, item) for item in staged]
    results = await asyncio.gather(*tasks)

    return {
        "status": "success",
        "files_processed": len(results),
        "total_records_in_dataset": len(GLOBAL_DATASET),
        "months_available": get_months_available(),
        "data_warnings": _build_data_warnings(GLOBAL_DATASET),
        "results": results,
        "overall_summary": build_overall_summary(GLOBAL_DATASET),
    }


# ── Data quality warnings (shared helper) ─────────────────────────────
def _build_data_warnings(records: list) -> list:
    """Scan records and return UI-friendly warnings for missing/incomplete data."""
    if not records:
        return []

    warnings = []
    total = len(records)

    # Count missing fields
    missing_billing = [r for r in records if not r.get("billing_rate")]
    missing_cost    = [r for r in records if not r.get("cost_rate")]
    missing_revenue = [r for r in records if r.get("revenue") is None]
    missing_profit  = [r for r in records if r.get("profit") is None]
    missing_hours   = [r for r in records if not r.get("actual_hours")]
    missing_month   = [r for r in records if not r.get("month")]

    if missing_billing:
        names = sorted({r.get("employee", "?") for r in missing_billing})[:5]
        warnings.append({
            "level": "error",
            "code": "MISSING_BILLING_RATE",
            "message": f"{len(missing_billing)} of {total} records have no billing rate — revenue cannot be calculated",
            "affected_employees": names,
            "count": len(missing_billing),
        })

    if missing_cost:
        names = sorted({r.get("employee", "?") for r in missing_cost})[:5]
        warnings.append({
            "level": "error",
            "code": "MISSING_COST_RATE",
            "message": f"{len(missing_cost)} of {total} records have no cost rate — profit cannot be calculated",
            "affected_employees": names,
            "count": len(missing_cost),
        })

    if missing_hours:
        names = sorted({r.get("employee", "?") for r in missing_hours})[:5]
        warnings.append({
            "level": "error",
            "code": "MISSING_HOURS",
            "message": f"{len(missing_hours)} of {total} records have no hours data — all financials will be zero",
            "affected_employees": names,
            "count": len(missing_hours),
        })

    if missing_month:
        warnings.append({
            "level": "warning",
            "code": "MISSING_MONTH",
            "message": f"{len(missing_month)} of {total} records have no month — range filtering will not work for these",
            "count": len(missing_month),
        })

    if missing_revenue:
        pct = round(len(missing_revenue) / total * 100)
        warnings.append({
            "level": "warning",
            "code": "INCOMPLETE_FINANCIALS",
            "message": f"{pct}% of records ({len(missing_revenue)}/{total}) have incomplete financial data (revenue/profit = null)",
            "count": len(missing_revenue),
        })

    # Flag records with validation issues
    flagged = [r for r in records if r.get("validation_flags")]
    if flagged:
        # Aggregate flag counts
        flag_counts = {}
        for r in flagged:
            for f in r.get("validation_flags", []):
                flag_counts[f] = flag_counts.get(f, 0) + 1
        warnings.append({
            "level": "info",
            "code": "VALIDATION_FLAGS",
            "message": f"{len(flagged)} of {total} records have validation flags",
            "flag_counts": flag_counts,
        })

    return warnings


# ── GET /data-quality — Data quality report ────────────────────────────
@app.get("/data-quality")
def data_quality(time_range: Optional[str] = Query(None, alias="range")):
    filtered = filter_by_range(GLOBAL_DATASET, time_range)
    total = len(filtered)
    valid = sum(1 for r in filtered if r.get("is_valid", True))
    return {
        "time_range": time_range or "ALL",
        "total_records": total,
        "valid_records": valid,
        "incomplete_records": total - valid,
        "completeness_pct": round(valid / total * 100, 1) if total else 0,
        "warnings": _build_data_warnings(filtered),
        "files": get_files_processed(),
    }


# ── GET /dataset — Full unified dataset ─────────────────────────────────
@app.get("/dataset")
def get_dataset(time_range: Optional[str] = Query(None, alias="range", description="1M, 3M, 6M, 12M, or ALL")):
    return transform_to_api_format(time_range=time_range)


# ── GET /metrics — Overall summary metrics ──────────────────────────────
@app.get("/metrics")
def get_metrics(time_range: Optional[str] = Query(None, alias="range")):
    filtered = filter_by_range(GLOBAL_DATASET, time_range)
    overall_summary = build_overall_summary(filtered)
    return {
        "time_range": time_range or "ALL",
        "total_hours": overall_summary.get("total_hours", 0),
        "overall_summary": overall_summary,
        "monthly": build_monthly(filtered),
    }


# ── GET /projects — Project-level breakdown ─────────────────────────────
@app.get("/projects")
def get_projects(time_range: Optional[str] = Query(None, alias="range")):
    filtered = filter_by_range(GLOBAL_DATASET, time_range)
    return {
        "time_range": time_range or "ALL",
        "projects": build_project_summaries(filtered),
    }


# ── GET /employees — Employee-level records ─────────────────────────────
@app.get("/employees")
def get_employees(time_range: Optional[str] = Query(None, alias="range"), project: Optional[str] = Query(None)):
    filtered = filter_by_range(GLOBAL_DATASET, time_range)
    if project:
        filtered = [r for r in filtered if r.get("project", "").lower() == project.lower()]
    employees = build_employee_summaries(filtered)
    return {
        "time_range": time_range or "ALL",
        "count": len(employees),
        "employees": employees,
    }



# ── GET /risks-recommendations — Risk analysis + AI recommendations ──────
@app.get("/risks-recommendations")
def risks_and_recommendations(time_range: Optional[str] = Query(None, alias="range")):
    return get_risks_and_recommendations(time_range=time_range)


# ── POST /ask — Natural language Q&A (LLM-powered) ──────────────────────
class AskRequest(BaseModel):
    query: str
    time_range: Optional[str] = None


@app.post("/ask")
def ask_question(req: AskRequest):
    return qa_ask(req.query, time_range=req.time_range)


# ── GET /files — List uploaded files ────────────────────────────────────
@app.get("/files")
def list_files():
    files = get_files_processed()
    return {"files": files, "count": len(files)}


# ── DELETE /dataset — Reset all ────────────────────────────────────────
@app.delete("/dataset")
def reset_dataset():
    clear_dataset()
    return {"status": "cleared", "records": 0}


# ── DELETE /dataset/{filename} — Remove one file's data ────────────────
@app.delete("/dataset/{filename}")
def remove_file_data(filename: str):
    removed = remove_by_file(filename)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No records found for file '{filename}'")
    return {
        "status": "removed",
        "filename": filename,
        "records_removed": removed,
        "records_remaining": len(GLOBAL_DATASET),
    }


