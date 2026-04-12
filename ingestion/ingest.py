"""
ingest.py — main entry point for FinIntel AI file ingestion.
"""

import json
from pathlib import Path
from typing import Optional

from .file_reader import read_file
from .field_mapper import build_column_mapping, mapping_report, apply_mapping
from .record_parser import parse_record


def ingest_file(filepath: str, time_range: Optional[str] = None) -> dict:
    errors = []

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

    if time_range and records:
        records = _apply_time_filter(records, time_range)

    summary = _build_summary(records, file_meta, report, parse_errors, time_range)

    return {
        "file_meta": file_meta,
        "column_mapping": report,
        "records": records,
        "summary": summary,
        "errors": errors,
    }


def _apply_time_filter(records, time_range):
    import datetime as dt
    dated = [r for r in records if r.get("date")]
    if not dated:
        return records
    max_date = max(dt.date.fromisoformat(r["date"]) for r in dated)
    days_map = {"1M": 30, "3M": 90, "6M": 180, "1Y": 365}
    cutoff_days = days_map.get(time_range.upper())
    if not cutoff_days:
        return records
    cutoff = max_date - dt.timedelta(days=cutoff_days)
    return [r for r in records if r.get("date") and dt.date.fromisoformat(r["date"]) >= cutoff]


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
