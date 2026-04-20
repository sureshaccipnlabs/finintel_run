"""
text_parser.py — LLM-powered parser for freeform / unstructured text files.

Sends raw text to Ollama and asks it to extract structured employee records.
Falls back gracefully if Ollama is unavailable.
"""

import json
import re

try:
    from .ai_mapper import _ollama_generate, is_ollama_available, _extract_json
    _AI_OK = True
except ImportError:
    _AI_OK = False
    def _ollama_generate(*a, **kw): return ""
    def is_ollama_available(): return False
    def _extract_json(t): return None


_EXTRACT_PROMPT = """\
You are a data extraction assistant. Extract ALL employee timesheet/financial \
records from the text below.

TEXT:
{text}

For EACH employee entry found, extract these fields:
- employee: person's name
- project: project or client name
- month: month and year (e.g. "MARCH'26" or "March 2026")
- actual_hours: hours worked (number)
- billing_rate: billing rate per hour (number)
- cost_rate: cost rate per hour (number)
- vacation_days: leave/vacation days (number, 0 if not mentioned)

Return ONLY a JSON array of objects. Example:
[
  {{"employee": "John Smith", "project": "Alpha", "month": "MARCH'26", \
"actual_hours": 160, "billing_rate": 150, "cost_rate": 85, "vacation_days": 0}},
  {{"employee": "Jane Doe", "project": "Beta", "month": "MARCH'26", \
"actual_hours": 140, "billing_rate": 120, "cost_rate": 70, "vacation_days": 2}}
]

If no employee records can be extracted, return an empty array: []
Extract ALL records you can find. Do NOT make up data that isn't in the text.
"""


def parse_freeform_text(text: str) -> list[dict]:
    """
    Use LLM to extract structured records from freeform text.

    Returns: list of dicts with employee record fields.
    """
    if not text or not text.strip():
        return []

    if not _AI_OK or not is_ollama_available():
        print("[text_parser] Ollama unavailable — cannot parse freeform text")
        return []

    # Truncate very large text to fit context window
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(truncated)"

    prompt = _EXTRACT_PROMPT.format(text=text)

    try:
        raw_response = _ollama_generate(prompt, timeout=120)
    except Exception as e:
        print(f"[text_parser] LLM extraction failed: {e}")
        return []

    # Try to parse JSON array from response
    records = _parse_json_array(raw_response)
    if not records:
        print(f"[text_parser] Could not parse LLM response as JSON array")
        return []

    # Normalize extracted records
    cleaned = []
    for r in records:
        if not isinstance(r, dict):
            continue
        if not r.get("employee"):
            continue
        cleaned.append(_normalize_extracted(r))

    print(f"[text_parser] Extracted {len(cleaned)} records from freeform text")
    return cleaned


def _parse_json_array(text: str) -> list:
    """Extract a JSON array from LLM response text."""
    text = text.strip()

    # Direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Look for ```json ... ```
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Find outermost [ ... ]
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        try:
            # Clean trailing commas
            snippet = text[start:end + 1]
            snippet = re.sub(r',\s*([}\]])', r'\1', snippet)
            return json.loads(snippet)
        except json.JSONDecodeError:
            pass

    return []


def _normalize_extracted(r: dict) -> dict:
    """Normalize an LLM-extracted record into canonical format."""
    def _float(v):
        try:
            return float(v) if v is not None else 0
        except (ValueError, TypeError):
            return 0

    employee = str(r.get("employee", "")).strip().title()
    project = str(r.get("project", "Unknown")).strip()
    month = str(r.get("month", "Unknown")).strip()
    actual_hours = _float(r.get("actual_hours"))
    billing_rate = _float(r.get("billing_rate"))
    cost_rate = _float(r.get("cost_rate"))
    vacation_days = _float(r.get("vacation_days"))

    revenue = round(actual_hours * billing_rate, 2) if billing_rate else 0
    cost = round(actual_hours * cost_rate, 2) if cost_rate else 0
    profit = round(revenue - cost, 2)
    margin_pct = round((profit / revenue) * 100, 2) if revenue > 0 else 0
    billable_hours = actual_hours
    working_days = round(actual_hours / 8, 2) if actual_hours else 0
    utilisation_pct = 100.0 if actual_hours > 0 else 0

    flags = []
    if not billing_rate:
        flags.append("MISSING_BILLING_RATE")
    if not cost_rate:
        flags.append("MISSING_COST_RATE")
    if margin_pct < 10:
        flags.append("LOW_MARGIN")
    if profit < 0:
        flags.append("LOSS_MAKING")

    return {
        "employee": employee,
        "project": project,
        "month": month,
        "actual_hours": actual_hours,
        "billable_hours": billable_hours,
        "working_days": working_days,
        "vacation_days": vacation_days,
        "billing_rate": billing_rate,
        "cost_rate": cost_rate,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "margin_pct": margin_pct,
        "utilisation_pct": utilisation_pct,
        "validation_flags": flags,
        "is_profitable": profit > 0,
        "is_valid": len(flags) == 0 or flags == ["LOW_MARGIN"],
        "_source": "text_parser",
    }


def is_freeform_text(filepath: str) -> bool:
    """
    Check if a text file is freeform (not delimited tabular data).
    Returns True if the file doesn't look like CSV/TSV.
    """
    try:
        with open(filepath, "r", errors="replace") as f:
            sample = f.read(4096)
    except Exception:
        return False

    if not sample.strip():
        return False

    lines = sample.strip().split("\n")
    if len(lines) < 2:
        return True  # single line or very short — likely freeform

    # Check if it looks like delimited data (consistent delimiter counts)
    for delim in [",", "\t", "|", ";"]:
        counts = [line.count(delim) for line in lines[:10] if line.strip()]
        if counts and min(counts) >= 2 and max(counts) - min(counts) <= 2:
            return False  # looks like delimited tabular data

    return True
