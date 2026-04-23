"""
ingest.py — main entry point for FinIntel AI file ingestion.

Supported formats:
    .xlsx, .xls  → timesheet_parser (smart sheet analysis + AI column mapping)
    .csv, .tsv   → file_reader → field_mapper (fuzzy + AI fallback) → record_parser
    .txt         → auto-detect: delimited → CSV flow, freeform → LLM extraction
    .pdf         → pdfplumber table extraction → field_mapper → record_parser
"""

import json
import re
from pathlib import Path
from typing import Optional

from .file_reader import read_file
from .field_mapper import build_column_mapping, mapping_report, apply_mapping
from .record_parser import parse_record
from .dataset import append_records
from .text_parser import parse_freeform_text, is_freeform_text, looks_like_timesheet_text


def _extract_filename_target_months(filename: str):
    """Return target month tokens from filenames like 'MAY-JAN-2026'.
    Example output: ["MAY'26", "JAN'26"]. Returns [] if not a multi-month hint.
    """
    base = Path(filename).stem
    month_hits = re.findall(
        r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)",
        base,
        flags=re.IGNORECASE,
    )
    if len(month_hits) < 2:
        return []

    year_hits = re.findall(r"(\d{4}|\d{2})", base)
    if not year_hits:
        return []
    year_2 = year_hits[-1][-2:]

    out = []
    for m in month_hits:
        token = f"{m.upper()[:3]}'{year_2}"
        if token not in out:
            out.append(token)
    return out


def _month_key_from_text(text) -> Optional[str]:
    if not text:
        return None
    m = re.search(
        r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)\D*(\d{2,4})",
        str(text),
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return f"{m.group(1).upper()[:3]}'{m.group(2)[-2:]}"


def _sheet_month_key(sheet_name: str, sheet_data: dict) -> Optional[str]:
    by_name = _month_key_from_text(sheet_name)
    if by_name:
        return by_name
    for emp in sheet_data.get("employees", []):
        by_emp = _month_key_from_text(emp.get("month"))
        if by_emp:
            return by_emp
    return None


def _sheet_quality_score(sheet_data: dict):
    emps = sheet_data.get("employees", [])
    total = len(emps)
    with_both_rates = sum(1 for e in emps if e.get("billing_rate") and e.get("cost_rate"))
    total_revenue = sheet_data.get("summary", {}).get("total_revenue", 0) or 0
    return (with_both_rates, total, total_revenue)


def _rebuild_overall_summary_from_sheets(sheets: dict) -> dict:
    total_revenue = round(sum(s.get("summary", {}).get("total_revenue", 0) for s in sheets.values()), 2)
    total_cost = round(sum(s.get("summary", {}).get("total_cost", 0) for s in sheets.values()), 2)
    total_profit = round(sum(s.get("summary", {}).get("total_profit", 0) for s in sheets.values()), 2)
    avg_margin = round((total_profit / total_revenue) * 100, 2) if total_revenue > 0 else 0
    return {
        "total_revenue": total_revenue,
        "total_cost": total_cost,
        "total_profit": total_profit,
        "avg_margin_pct": avg_margin,
    }

def ingest_file(filepath: str, time_range: Optional[str] = None, original_filename: Optional[str] = None) -> dict:
    errors = []

    # ── STEP 0: Freeform text detection (.txt that isn't delimited) ──────
    if filepath.lower().endswith(".txt") and is_freeform_text(filepath):
        return _ingest_freeform(filepath, time_range, original_filename)

    # ── STEP 1: Excel Timesheet (.xlsx, .xls) ───────────────────────────
    if filepath.lower().endswith((".xlsx", ".xls")):
        try:
            from .timesheet_parser import parse_timesheet

            # For multi-month filenames like MAY-JAN-2026, process only those months.
            # Otherwise, process all timesheet sheets in the workbook.
            target_months = _extract_filename_target_months(original_filename or filepath)
            ts_result = parse_timesheet(filepath)

            if target_months:
                target_set = set(target_months)
                # Keep only the best sheet per target month key.
                best_by_month = {}
                for sheet_name, sheet_data in ts_result.get("sheets", {}).items():
                    key = _sheet_month_key(sheet_name, sheet_data)
                    if key not in target_set:
                        continue
                    current = best_by_month.get(key)
                    if not current:
                        best_by_month[key] = (sheet_name, sheet_data)
                        continue
                    _, current_data = current
                    if _sheet_quality_score(sheet_data) > _sheet_quality_score(current_data):
                        best_by_month[key] = (sheet_name, sheet_data)

                filtered_sheets = {name: data for name, data in best_by_month.values()}
                if filtered_sheets:
                    ts_result["sheets"] = filtered_sheets
                    ts_result["overall_summary"] = _rebuild_overall_summary_from_sheets(filtered_sheets)

            all_employees = []
            for sheet in ts_result.get("sheets", {}).values():
                all_employees.extend(sheet.get("employees", []))
            append_records(all_employees, filename=original_filename or filepath)

            return ts_result

        except Exception as e:
            return {
                "file": filepath,
                "sheets": {},
                "overall_summary": {},
                "errors": [str(e)],
            }

    # ── STEP 2: Delimited files (CSV, TSV, tabular TXT, PDF) ───────────
    try:
        raw_rows, file_meta = read_file(filepath)
    except Exception as e:
        return {"errors": [str(e)], "records": [], "file_meta": {}, "column_mapping": {}, "summary": {}}

    if not raw_rows:
        return {"errors": ["File is empty or no data rows found."], "records": [], "file_meta": file_meta, "column_mapping": {}, "summary": {}}

    raw_columns = list(raw_rows[0].keys())
    mapping = build_column_mapping(raw_columns)
    report = mapping_report(raw_columns, mapping)

    if report["missing_required"]:
        errors.append({
            "type": "MISSING_REQUIRED_COLUMNS",
            "details": f"Could not map: {report['missing_required']}",
            "suggestion": f"Columns found: {raw_columns}",
        })

    mapped_rows = apply_mapping(raw_rows, mapping)

    records = []
    parse_errors = 0
    for i, row in enumerate(mapped_rows):
        try:
            parsed = parse_record(row)
            parsed["_row_index"] = i + 2
            records.append(parsed)
        except Exception as e:
            parse_errors += 1
            errors.append({"type": "PARSE_ERROR", "row": i + 2, "detail": str(e)})

    # Report records missing required fields
    invalid_count = sum(1 for r in records if not r.get("is_valid", True))
    missing_emp = sum(1 for r in records if "MISSING_EMPLOYEE" in r.get("validation_flags", []))
    missing_hrs = sum(1 for r in records if "MISSING_HOURS" in r.get("validation_flags", []))
    if missing_emp:
        errors.append({
            "type": "MISSING_DATA_WARNING",
            "details": f"{missing_emp} records have no employee name — these will be skipped",
        })
    if missing_hrs:
        errors.append({
            "type": "MISSING_DATA_WARNING",
            "details": f"{missing_hrs} records have no hours data — financial calculations will be zero",
        })

    if time_range and records:
        records = _apply_time_filter(records, time_range)

    append_records(records, filename=original_filename or filepath)

    summary = _build_summary(records, file_meta, report, parse_errors, time_range)

    return {
        "file_meta": file_meta,
        "column_mapping": report,
        "records": records,
        "summary": summary,
        "errors": errors,
    }


def _ingest_freeform(filepath: str, time_range: Optional[str], original_filename: Optional[str]) -> dict:
    """Parse freeform/unstructured text via LLM extraction."""
    try:
        with open(filepath, "r", errors="replace") as f:
            raw_text = f.read()
    except Exception as e:
        return {"errors": [f"Could not read file: {e}"], "records": [], "file_meta": {}, "summary": {}}

    # Pre-validation: reject non-timesheet text before wasting an LLM call
    if not looks_like_timesheet_text(raw_text):
        return {
            "errors": [
                "This file does not appear to contain timesheet or financial data. "
                "Expected keywords like employee names, hours, rates, billing, etc. "
                "Please upload a file with timesheet/financial content."
            ],
            "records": [],
            "file_meta": {
                "filename": original_filename or filepath,
                "file_type": "REJECTED_TEXT",
                "char_count": len(raw_text),
            },
            "summary": {},
        }

    records = parse_freeform_text(raw_text)

    if not records:
        return {
            "errors": ["Text looks like timesheet data but no employee records could be extracted. "
                       "Check that the text contains specific employee names, hours, and rates."],
            "records": [],
            "file_meta": {
                "filename": original_filename or filepath,
                "file_type": "FREEFORM_TEXT",
                "char_count": len(raw_text),
            },
            "summary": {},
        }

    if time_range:
        records = _apply_time_filter(records, time_range)

    append_records(records, filename=original_filename or filepath)

    file_meta = {
        "filename": original_filename or filepath,
        "file_type": "FREEFORM_TEXT",
        "char_count": len(raw_text),
        "row_count": len(records),
    }

    revenues = [r["revenue"] for r in records if r.get("revenue")]
    costs = [r["cost"] for r in records if r.get("cost")]
    profits = [r["profit"] for r in records if r.get("profit")]
    total_rev = round(sum(revenues), 2) if revenues else 0
    total_cost = round(sum(costs), 2) if costs else 0
    total_profit = round(sum(profits), 2) if profits else 0

    return {
        "file_meta": file_meta,
        "column_mapping": {"method": "LLM_EXTRACTION"},
        "records": records,
        "summary": {
            "total_records": len(records),
            "projects": sorted({r["project"] for r in records if r.get("project")}),
            "employees": sorted({r["employee"] for r in records if r.get("employee")}),
            "total_revenue": total_rev,
            "total_cost": total_cost,
            "total_profit": total_profit,
            "extraction_method": "freeform_llm",
        },
        "errors": [],
    }


def _apply_time_filter(records, time_range):
    from .dataset import filter_by_range
    return filter_by_range(records, time_range)


def _build_summary(records, file_meta, map_report, parse_errors, time_range):
    total = len(records)
    valid = sum(1 for r in records if r.get("is_valid"))
    revenues = [r["revenue"] for r in records if r.get("revenue") is not None]
    costs    = [r["cost"]    for r in records if r.get("cost")    is not None]
    profits  = [r["profit"]  for r in records if r.get("profit")  is not None]
    total_rev    = round(sum(revenues), 2) if revenues else None
    total_cost   = round(sum(costs), 2)    if costs    else None
    total_profit = round(sum(profits), 2)  if profits  else None
    avg_margin   = round((total_profit / total_rev * 100), 2) if (total_profit is not None and total_rev and total_rev > 0) else None
    all_flags: dict = {}
    for r in records:
        for f in r.get("validation_flags", []):
            all_flags[f] = all_flags.get(f, 0) + 1
    return {
        "time_range_applied": time_range or "ALL",
        "total_records": total,
        "valid_records": valid,
        "flagged_records": total - valid,
        "parse_errors": parse_errors,
        "projects":  sorted({r["project"]  for r in records if r.get("project")}),
        "employees": sorted({r["employee"] for r in records if r.get("employee")}),
        "total_revenue": total_rev,
        "total_cost":    total_cost,
        "total_profit":  total_profit,
        "avg_margin_pct": avg_margin,
        "validation_flag_counts": all_flags,
        "mapping_confidence": map_report.get("confidence"),
        "missing_required_fields": map_report.get("missing_required", []),
        "missing_optional_fields": map_report.get("missing_optional", []),
    }


def ingest_to_json(filepath: str, output_path: Optional[str] = None,
                   time_range: Optional[str] = None, pretty: bool = True) -> str:
    result = ingest_file(filepath, time_range=time_range)
    json_str = json.dumps(result, indent=2 if pretty else None, default=str)
    if output_path:
        Path(output_path).write_text(json_str, encoding="utf-8")
    return json_str
