from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
from ingestion.ingest import (
    ingest_file,
    extract_cost_rates_from_excel,
    apply_cost_rates_to_global_dataset,
)
from ingestion.dataset import (
    GLOBAL_DATASET, transform_to_api_format, filter_by_range,
    build_projects, build_monthly, build_overall_summary,
    build_top_performers, get_months_available,
    clear as clear_dataset, remove_by_file, get_files_processed,
    build_project_summaries, build_employee_summaries,
)
from ingestion.qa_engine import ask as qa_ask
from ingestion.risk_engine import get_risks_and_recommendations
from ingestion import ai_mapper
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os
import re

_ingest_pool = ThreadPoolExecutor(max_workers=4)


def _looks_like_cost_reference(filename: str) -> bool:
    name = (filename or "").lower()
    tokens = ("cost", "ctc", "rate card", "ratecard", "salary", "resource cost")
    return any(t in name for t in tokens)


def _extract_month_scope_from_filename(filename: str):
    name = (filename or "")
    months = re.findall(
        r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)",
        name,
        flags=re.IGNORECASE,
    )
    years = re.findall(r"\b(\d{4}|\d{2})\b", name)
    month_set = {m.upper()[:3] for m in months}
    year_set = {y[-2:] for y in years}
    return month_set, year_set


def _looks_like_timesheet_file(filename: str) -> bool:
    name = (filename or "").lower()
    return ("timesheet" in name) and not _looks_like_cost_reference(name)

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

        cost_map = None
        ext = os.path.splitext(item["filename"])[1].lower()
        if ext in (".xlsx", ".xls") and _looks_like_cost_reference(item["filename"]):
            try:
                cost_map = extract_cost_rates_from_excel(item["tmp_path"])
            except Exception as e:
                cost_map = {"error": str(e), "employee_rates": {}, "employee_project_rates": {}}

        try:
            os.unlink(item["tmp_path"])
        except OSError:
            pass
        result.pop("overall_summary", None)
        return {"filename": item["filename"], "result": result, "cost_map": cost_map}

    loop = asyncio.get_event_loop()
    tasks = [loop.run_in_executor(_ingest_pool, _process_one, item) for item in staged]
    results = await asyncio.gather(*tasks)

    timesheet_month_scope = set()
    timesheet_year_scope = set()
    for row in results:
        fname = row.get("filename") or ""
        if not _looks_like_timesheet_file(fname):
            continue
        mset, yset = _extract_month_scope_from_filename(fname)
        timesheet_month_scope.update(mset)
        timesheet_year_scope.update(yset)

    merged_employee_rates = {}
    merged_employee_project_rates = {}
    cost_sources = []
    api_results = []
    total_updated_records = 0
    total_updated_employee_only = 0
    total_untouched_records = 0
    for row in results:
        cm = row.get("cost_map")
        api_row = {
            "filename": row.get("filename"),
            "result": row.get("result"),
        }

        if cm:
            api_row["cost_map_summary"] = {
                "rows_scanned": cm.get("rows_scanned", 0),
                "mapped_rows": cm.get("mapped_rows", 0),
                "employee_rate_mappings": len(cm.get("employee_rates") or {}),
                "employee_project_rate_mappings": len(cm.get("employee_project_rates") or {}),
                "error": cm.get("error"),
            }
        api_results.append(api_row)

        if not cm or cm.get("error"):
            continue

        employee_rates = cm.get("employee_rates") or {}
        employee_project_rates = cm.get("employee_project_rates") or {}
        merged_employee_rates.update(employee_rates)
        merged_employee_project_rates.update(employee_project_rates)

        allowed_months, allowed_years = _extract_month_scope_from_filename(row.get("filename") or "")

        effective_months = set(allowed_months) if allowed_months else set()
        effective_years = set(allowed_years) if allowed_years else set()

        if timesheet_month_scope and effective_months:
            effective_months = effective_months.intersection(timesheet_month_scope)
        if timesheet_year_scope and effective_years:
            effective_years = effective_years.intersection(timesheet_year_scope)

        if allowed_months and not effective_months:
            merge_stats = {"updated_records": 0, "updated_employee_only": 0, "untouched_records": 0}
            cost_sources.append({
                "filename": row.get("filename"),
                "rows_scanned": cm.get("rows_scanned", 0),
                "mapped_rows": cm.get("mapped_rows", 0),
                "applied_month_scope": [],
                "applied_year_scope": sorted(effective_years) if effective_years else None,
                "merge_stats": merge_stats,
                "skipped_reason": "COST_MONTH_MISMATCH_WITH_TIMESHEET",
            })
            continue

        merge_stats = apply_cost_rates_to_global_dataset(
            employee_rates,
            employee_project_rates,
            allowed_months=effective_months or None,
            allowed_years=effective_years or None,
        )
        total_updated_records += int(merge_stats.get("updated_records", 0) or 0)
        total_updated_employee_only += int(merge_stats.get("updated_employee_only", 0) or 0)
        total_untouched_records += int(merge_stats.get("untouched_records", 0) or 0)

        cost_sources.append({
            "filename": row.get("filename"),
            "rows_scanned": cm.get("rows_scanned", 0),
            "mapped_rows": cm.get("mapped_rows", 0),
            "applied_month_scope": sorted(effective_months) if effective_months else None,
            "applied_year_scope": sorted(effective_years) if effective_years else None,
            "merge_stats": merge_stats,
        })

    cost_merge_summary = None
    if merged_employee_rates or merged_employee_project_rates:
        cost_merge_summary = {
            "cost_files_used": len(cost_sources),
            "employee_rate_mappings": len(merged_employee_rates),
            "employee_project_rate_mappings": len(merged_employee_project_rates),
            "merge_stats": {
                "updated_records": total_updated_records,
                "updated_employee_only": total_updated_employee_only,
                "untouched_records": total_untouched_records,
            },
            "sources": cost_sources,
        }

    return {
        "status": "success",
        "files_processed": len(results),
        "total_records_in_dataset": len(GLOBAL_DATASET),
        "months_available": get_months_available(),
        "data_warnings": _build_data_warnings(GLOBAL_DATASET),
        "results": api_results,
        "cost_rate_merge": cost_merge_summary,
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
def get_projects(
    time_range: Optional[str] = Query(None, alias="range"),
    project: Optional[str] = Query(None),
):
    filtered = filter_by_range(GLOBAL_DATASET, time_range)
    if project:
        project_lc = project.strip().lower()
        filtered = [r for r in filtered if (r.get("project") or "").strip().lower() == project_lc]
    return {
        "time_range": time_range or "ALL",
        "months_considered": get_months_available(filtered),
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
def risks_and_recommendations(
    time_range: Optional[str] = Query(None, alias="range"),
    project: Optional[str] = Query(None),
    max_items: int = Query(8, ge=3, le=20),
):
    return get_risks_and_recommendations(time_range=time_range, project=project, max_items=max_items)


# ── POST /ask — Natural language Q&A (LLM-powered) ──────────────────────
class AskRequest(BaseModel):
    query: str
    time_range: Optional[str] = None


class AIProviderConfigRequest(BaseModel):
    provider: Optional[str] = None
    provider_order: Optional[str] = None


@app.post("/ask")
def ask_question(req: AskRequest):
    return qa_ask(req.query, time_range=req.time_range)


# ── GET /files — List uploaded files ────────────────────────────────────
@app.get("/files")
def list_files():
    files = get_files_processed()
    return {"files": files, "count": len(files)}


# ── GET /ai-provider — Active AI provider config and health ────────────────
@app.get("/ai-provider")
def get_ai_provider_status():
    return ai_mapper.get_ai_provider_status()


# ── POST /ai-provider — Update provider config at runtime ────────────────
@app.post("/ai-provider")
def set_ai_provider_status(req: AIProviderConfigRequest):
    try:
        return ai_mapper.configure_ai_provider(
            provider=req.provider,
            provider_order=req.provider_order,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


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


