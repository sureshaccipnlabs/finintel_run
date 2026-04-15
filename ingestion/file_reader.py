"""
file_reader.py — reads any supported file type into a raw list of dicts.
Supports: CSV, TSV, XLSX, XLS, PDF (table extraction), TXT
"""

import csv
import io
import re
from pathlib import Path
from typing import Optional
from .timesheet_parser import parse_timesheet

import chardet
import pandas as pd
import pdfplumber


SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".pdf"}


def detect_encoding(raw_bytes: bytes) -> str:
    result = chardet.detect(raw_bytes)
    return result.get("encoding") or "utf-8"


def read_file(filepath: str) -> tuple[list[dict], dict]:
    """
    Read any supported file into a list of raw row dicts.

    Returns:
        rows      — list of dicts (raw, unprocessed column names)
        file_meta — info about the file (name, type, row count, sheet, etc.)
    """
    path = Path(filepath)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext in (".csv", ".tsv", ".txt"):
        return _read_delimited(path, ext)
    elif ext in (".xlsx", ".xls"):
        return _read_timesheet(path)
    elif ext == ".pdf":
        return _read_pdf(path)


def _read_timesheet(path):
    records = parse_timesheet(str(path))  # already list of dicts

    return records, {
        "type": "timesheet_excel",
        "sheets": ["parsed"]  # or keep actual if needed
    }

def _read_delimited(path: Path, ext: str) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    encoding = detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")

    # Auto-detect delimiter
    sample = text[:4096]
    delimiter = _detect_delimiter(sample, ext)

    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    rows = [dict(row) for row in reader]

    # Strip whitespace from all keys and values
    rows = _clean_rows(rows)

    return rows, {
        "filename": path.name,
        "file_type": ext.lstrip(".").upper(),
        "encoding": encoding,
        "delimiter": repr(delimiter),
        "raw_columns": list(rows[0].keys()) if rows else [],
        "row_count": len(rows),
    }


def _detect_delimiter(sample: str, ext: str) -> str:
    if ext == ".tsv":
        return "\t"
    counts = {d: sample.count(d) for d in [",", "\t", ";", "|"]}
    return max(counts, key=counts.get)


def _read_excel(path: Path) -> tuple[list[dict], dict]:
    xl = pd.ExcelFile(path)
    sheet_name = xl.sheet_names[0]  # use first sheet by default

    # Try to find the sheet with the most data columns if multiple sheets
    best_sheet = sheet_name
    best_cols = 0
    for s in xl.sheet_names:
        df_peek = pd.read_excel(path, sheet_name=s, nrows=2)
        if len(df_peek.columns) > best_cols:
            best_cols = len(df_peek.columns)
            best_sheet = s

    df = pd.read_excel(path, sheet_name=best_sheet, dtype=str)
    df = df.dropna(how="all")  # drop fully empty rows
    df.columns = [str(c).strip() for c in df.columns]

    rows = df.to_dict(orient="records")
    rows = _clean_rows(rows)

    return rows, {
        "filename": path.name,
        "file_type": "EXCEL",
        "sheet": best_sheet,
        "all_sheets": xl.sheet_names,
        "raw_columns": list(df.columns),
        "row_count": len(rows),
    }


def _read_pdf(path: Path) -> tuple[list[dict], dict]:
    all_rows = []
    pages_with_tables = 0

    with pdfplumber.open(path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if not table or len(table) < 2:
                    continue
                pages_with_tables += 1
                header = [str(c).strip() if c else f"col_{i}" for i, c in enumerate(table[0])]
                for row in table[1:]:
                    if any(cell for cell in row):  # skip blank rows
                        row_dict = {header[i]: (str(row[i]).strip() if row[i] else "") for i in range(len(header))}
                        all_rows.append(row_dict)

    if not all_rows:
        raise ValueError("No tables found in PDF. Make sure the PDF contains tabular data (not scanned images).")

    all_rows = _clean_rows(all_rows)

    return all_rows, {
        "filename": path.name,
        "file_type": "PDF",
        "total_pages": total_pages,
        "pages_with_tables": pages_with_tables,
        "raw_columns": list(all_rows[0].keys()) if all_rows else [],
        "row_count": len(all_rows),
    }


def _clean_rows(rows: list[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        clean = {}
        for k, v in row.items():
            key = str(k).strip() if k is not None else ""
            val = str(v).strip() if v is not None else ""
            if val.lower() in ("none", "nan", "n/a", "na", "-", ""):
                val = ""
            if key:
                clean[key] = val
        cleaned.append(clean)
    return cleaned

if __name__ == "__main__":
    import json
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parent.parent
    filepath = BASE_DIR / "sample_data/sample_2_messy_realworld.csv"

    rows, meta = read_file(filepath)

    result = {
        "file_meta": meta,
        "rows": rows[:5]
    }

    print(json.dumps(result, indent=2))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    out_file = out_dir / "file_reader_output.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nSaved to: {out_file}")
