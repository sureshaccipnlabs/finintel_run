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
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import pandas as pd

from .file_reader import read_file
from .field_mapper import build_column_mapping, mapping_report, apply_mapping
from .record_parser import parse_record
from .dataset import append_records, GLOBAL_DATASET
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


def _month_parts_from_text(text) -> tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    m = re.search(
        r"(JAN(?:UARY)?|FEB(?:RUARY)?|MAR(?:CH)?|APR(?:IL)?|MAY|JUN(?:E)?|JUL(?:Y)?|AUG(?:UST)?|SEP(?:TEMBER)?|OCT(?:OBER)?|NOV(?:EMBER)?|DEC(?:EMBER)?)\D*(\d{2,4})?",
        str(text),
        flags=re.IGNORECASE,
    )
    if not m:
        return None, None
    month = m.group(1).upper()[:3]
    year = m.group(2)[-2:] if m.group(2) else None
    return month, year


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


def _norm_emp(v: str) -> str:
    return " ".join(str(v or "").strip().title().split())


def _norm_proj(v: str) -> str:
    return " ".join(str(v or "").strip().lower().split())


def _to_rate(v) -> Optional[float]:
    if v is None:
        return None
    text = str(v).strip().replace(",", "")
    text = text.replace("$", "")
    if text == "":
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _extract_cost_rates_from_dataframe(df: pd.DataFrame) -> tuple[dict, dict, int, int]:
    employee_rates: dict[str, float] = {}
    employee_project_rates: dict[tuple[str, str], float] = {}
    rows_scanned = 0
    mapped_rows = 0

    employee_keys = ("employee", "emp", "name", "resource", "consultant", "staff")
    cost_keys = ("cost rate", "cost_rate", "ctc", "pay rate", "internal rate", "cost/hr", "resource cost", "cost")
    project_keys = ("project", "client", "account", "engagement")

    raw_cols = [str(c).strip() for c in df.columns]
    cols_norm = {c: str(c).strip().lower() for c in raw_cols}

    emp_col = next((c for c in raw_cols if any(k in cols_norm[c] for k in employee_keys)), None)
    cost_col = next((c for c in raw_cols if any(k in cols_norm[c] for k in cost_keys)), None)
    proj_col = next((c for c in raw_cols if any(k in cols_norm[c] for k in project_keys)), None)

    if not emp_col or not cost_col:
        return employee_rates, employee_project_rates, rows_scanned, mapped_rows

    for _, row in df.iterrows():
        rows_scanned += 1
        emp = _norm_emp(row.get(emp_col))
        rate = _to_rate(row.get(cost_col))
        if not emp or rate is None or rate <= 0:
            continue
        employee_rates[emp] = rate
        mapped_rows += 1

        if proj_col:
            proj = _norm_proj(row.get(proj_col))
            if proj:
                employee_project_rates[(emp, proj)] = rate

    return employee_rates, employee_project_rates, rows_scanned, mapped_rows


def _is_ods_file(filepath: str) -> bool:
    try:
        with zipfile.ZipFile(filepath, "r") as zf:
            if "mimetype" not in zf.namelist():
                return False
            mt = zf.read("mimetype").decode("utf-8", errors="ignore").strip()
            return mt == "application/vnd.oasis.opendocument.spreadsheet"
    except Exception:
        return False


def _extract_cost_rates_from_ods(filepath: str) -> dict:
    table_ns = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
    text_ns = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

    with zipfile.ZipFile(filepath, "r") as zf:
        content_xml = zf.read("content.xml")
    root = ET.fromstring(content_xml)

    employee_rates: dict[str, float] = {}
    employee_project_rates: dict[tuple[str, str], float] = {}
    rows_scanned = 0
    mapped_rows = 0
    sheet_count = 0

    employee_keys = ("employee", "emp", "name", "resource", "consultant", "staff")
    cost_keys = ("cost rate", "cost_rate", "ctc", "pay rate", "internal rate", "cost/hr", "resource cost", "cost")
    project_keys = ("project", "client", "account", "engagement")

    for table in root.findall(f".//{{{table_ns}}}table"):
        sheet_count += 1
        rows = []
        for row_el in table.findall(f"{{{table_ns}}}table-row"):
            row_vals = []
            for cell in list(row_el):
                tag = cell.tag
                if not tag.endswith("table-cell"):
                    continue
                repeat = int(cell.attrib.get(f"{{{table_ns}}}number-columns-repeated", "1"))
                txt = " ".join(
                    (p.text or "").strip()
                    for p in cell.findall(f"{{{text_ns}}}p")
                    if (p.text or "").strip()
                ).strip()
                if txt == "":
                    txt = cell.attrib.get(f"{{urn:oasis:names:tc:opendocument:xmlns:office:1.0}}value", "")
                for _ in range(max(repeat, 1)):
                    row_vals.append(txt)
            if any(str(v).strip() for v in row_vals):
                rows.append(row_vals)

        if not rows:
            continue

        header_idx = None
        for i, r in enumerate(rows[:12]):
            low = [str(c).strip().lower() for c in r]
            has_emp = any(any(k in c for k in employee_keys) for c in low)
            has_cost = any(any(k in c for k in cost_keys) for c in low)
            if has_emp and has_cost:
                header_idx = i
                break
        if header_idx is None:
            continue

        headers = [str(c).strip() or f"col_{j}" for j, c in enumerate(rows[header_idx])]
        data_rows = rows[header_idx + 1:]
        if not data_rows:
            continue

        max_len = max(len(headers), max(len(r) for r in data_rows))
        headers = (headers + [f"col_{i}" for i in range(len(headers), max_len)])[:max_len]
        normalized_rows = []
        for r in data_rows:
            vals = list(r) + [""] * (max_len - len(r))
            normalized_rows.append({headers[i]: vals[i] for i in range(max_len)})

        df = pd.DataFrame(normalized_rows)
        er, epr, rs, mr = _extract_cost_rates_from_dataframe(df)
        employee_rates.update(er)
        employee_project_rates.update(epr)
        rows_scanned += rs
        mapped_rows += mr

    return {
        "employee_rates": employee_rates,
        "employee_project_rates": employee_project_rates,
        "rows_scanned": rows_scanned,
        "mapped_rows": mapped_rows,
        "sheet_count": sheet_count,
    }


def _xml_local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _xlsx_col_to_index(cell_ref: str) -> int:
    letters = ""
    for ch in str(cell_ref or ""):
        if ch.isalpha():
            letters += ch.upper()
        else:
            break
    if not letters:
        return 0
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(0, idx - 1)


def _extract_cost_rates_from_xlsx_zip(filepath: str) -> dict:
    employee_rates: dict[str, float] = {}
    employee_project_rates: dict[tuple[str, str], float] = {}
    rows_scanned = 0
    mapped_rows = 0

    with zipfile.ZipFile(filepath, "r") as zf:
        names = zf.namelist()
        sheet_files = sorted(n for n in names if n.startswith("xl/worksheets/") and n.endswith(".xml"))
        if not sheet_files:
            return {
                "employee_rates": employee_rates,
                "employee_project_rates": employee_project_rates,
                "rows_scanned": rows_scanned,
                "mapped_rows": mapped_rows,
                "sheet_count": 0,
            }

        shared_strings = []
        if "xl/sharedStrings.xml" in names:
            ss_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in ss_root.iter():
                if _xml_local_name(si.tag) != "si":
                    continue
                chunks = []
                for node in si.iter():
                    if _xml_local_name(node.tag) == "t" and node.text:
                        chunks.append(node.text)
                shared_strings.append("".join(chunks).strip())

        for sheet_file in sheet_files:
            root = ET.fromstring(zf.read(sheet_file))
            rows = []
            for row_el in root.iter():
                if _xml_local_name(row_el.tag) != "row":
                    continue
                row_vals = []
                for c in list(row_el):
                    if _xml_local_name(c.tag) != "c":
                        continue
                    ref = c.attrib.get("r", "")
                    col_idx = _xlsx_col_to_index(ref)
                    while len(row_vals) <= col_idx:
                        row_vals.append("")

                    cell_type = c.attrib.get("t", "")
                    val = ""
                    if cell_type == "inlineStr":
                        parts = []
                        for n in c.iter():
                            if _xml_local_name(n.tag) == "t" and n.text:
                                parts.append(n.text)
                        val = "".join(parts).strip()
                    else:
                        v_el = next((n for n in list(c) if _xml_local_name(n.tag) == "v"), None)
                        raw = (v_el.text or "").strip() if v_el is not None else ""
                        if cell_type == "s" and raw.isdigit():
                            idx = int(raw)
                            if 0 <= idx < len(shared_strings):
                                val = shared_strings[idx]
                            else:
                                val = raw
                        else:
                            val = raw

                    row_vals[col_idx] = val

                if any(str(v).strip() for v in row_vals):
                    rows.append(row_vals)

            if not rows:
                continue

            header_idx = None
            employee_keys = ("employee", "emp", "name", "resource", "consultant", "staff")
            cost_keys = ("cost rate", "cost_rate", "ctc", "pay rate", "internal rate", "cost/hr", "resource cost", "cost")
            for i, r in enumerate(rows[:12]):
                low = [str(c).strip().lower() for c in r]
                has_emp = any(any(k in c for k in employee_keys) for c in low)
                has_cost = any(any(k in c for k in cost_keys) for c in low)
                if has_emp and has_cost:
                    header_idx = i
                    break

            if header_idx is None:
                continue

            headers = [str(c).strip() or f"col_{j}" for j, c in enumerate(rows[header_idx])]
            data_rows = rows[header_idx + 1:]
            if not data_rows:
                continue

            max_len = max(len(headers), max(len(r) for r in data_rows))
            headers = (headers + [f"col_{i}" for i in range(len(headers), max_len)])[:max_len]
            normalized_rows = []
            for r in data_rows:
                vals = list(r) + [""] * (max_len - len(r))
                normalized_rows.append({headers[i]: vals[i] for i in range(max_len)})

            df = pd.DataFrame(normalized_rows)
            er, epr, rs, mr = _extract_cost_rates_from_dataframe(df)
            employee_rates.update(er)
            employee_project_rates.update(epr)
            rows_scanned += rs
            mapped_rows += mr

    return {
        "employee_rates": employee_rates,
        "employee_project_rates": employee_project_rates,
        "rows_scanned": rows_scanned,
        "mapped_rows": mapped_rows,
        "sheet_count": len(sheet_files),
    }


def extract_cost_rates_from_excel(filepath: str) -> dict:
    """Extract employee cost-rate mappings from standalone cost workbook."""
    try:
        xl = pd.ExcelFile(filepath)
        employee_rates: dict[str, float] = {}
        employee_project_rates: dict[tuple[str, str], float] = {}
        rows_scanned = 0
        mapped_rows = 0

        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(filepath, sheet_name=sheet, dtype=str)
            except Exception:
                continue
            if df is None or df.empty:
                continue
            er, epr, rs, mr = _extract_cost_rates_from_dataframe(df)
            employee_rates.update(er)
            employee_project_rates.update(epr)
            rows_scanned += rs
            mapped_rows += mr

        result = {
            "employee_rates": employee_rates,
            "employee_project_rates": employee_project_rates,
            "rows_scanned": rows_scanned,
            "mapped_rows": mapped_rows,
            "sheet_count": len(xl.sheet_names),
        }
        if result["mapped_rows"] == 0:
            return _extract_cost_rates_from_xlsx_zip(filepath)
        return result
    except Exception:
        if _is_ods_file(filepath):
            return _extract_cost_rates_from_ods(filepath)
        return _extract_cost_rates_from_xlsx_zip(filepath)


def apply_cost_rates_to_global_dataset(
    employee_rates: dict,
    employee_project_rates: dict,
    allowed_months: Optional[set[str]] = None,
    allowed_years: Optional[set[str]] = None,
) -> dict:
    """Fill missing cost_rate in GLOBAL_DATASET and recalculate financial fields."""
    updated = 0
    updated_employee_only = 0
    untouched = 0
    employees_with_project_rates = {emp for emp, _ in employee_project_rates.keys()}

    for rec in GLOBAL_DATASET:
        current_rate = rec.get("cost_rate")
        if current_rate not in (None, "", 0, 0.0):
            untouched += 1
            continue

        if allowed_months:
            rec_month, rec_year = _month_parts_from_text(rec.get("month"))
            if not rec_month or rec_month not in allowed_months:
                continue
            if allowed_years and (not rec_year or rec_year not in allowed_years):
                continue

        emp = _norm_emp(rec.get("employee"))
        proj = _norm_proj(rec.get("project"))
        if not emp:
            continue

        new_rate = employee_project_rates.get((emp, proj)) if proj else None
        if new_rate is None:
            if emp not in employees_with_project_rates:
                new_rate = employee_rates.get(emp)
                if new_rate is not None:
                    updated_employee_only += 1
        if new_rate is None:
            continue

        rec["cost_rate"] = float(new_rate)

        actual_hours = float(rec.get("actual_hours") or 0)
        leave_hours = float(rec.get("leave_hours") or 0)
        hours_for_cost = actual_hours + leave_hours if leave_hours > 0 else actual_hours
        rec["cost"] = round(hours_for_cost * float(new_rate), 2)

        billing_rate = rec.get("billing_rate")
        if billing_rate not in (None, "", 0, 0.0):
            billable_hours = float(rec.get("billable_hours") or actual_hours)
            rec["revenue"] = round(billable_hours * float(billing_rate), 2)

        revenue = rec.get("revenue")
        cost = rec.get("cost")
        if revenue is not None and cost is not None:
            rec["profit"] = round(float(revenue) - float(cost), 2)
            rec["margin_pct"] = round((rec["profit"] / float(revenue)) * 100, 2) if float(revenue) > 0 else 0

        flags = rec.get("validation_flags") or []
        if "MISSING_COST_RATE" in flags:
            rec["validation_flags"] = [f for f in flags if f != "MISSING_COST_RATE"]

        updated += 1

    return {
        "updated_records": updated,
        "updated_employee_only": updated_employee_only,
        "untouched_records": untouched,
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
