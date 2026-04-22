"""
field_mapper.py — maps raw column names (whatever the user uploaded)
to canonical FinIntel field names using alias lookup + fuzzy matching.

Falls back to Ollama AI mapping when required fields are still unmapped.
"""

import re
from difflib import SequenceMatcher
from typing import Optional

from .field_config import FIELD_ALIASES, OPTIONAL_FIELDS, REQUIRED_FIELDS

try:
    from .ai_mapper import ai_map_columns, is_ollama_available
    _AI_AVAILABLE = True
except ImportError:
    _AI_AVAILABLE = False
    def ai_map_columns(*a, **kw): return {}
    def is_ollama_available(): return False


def normalise(text: str) -> str:
    """Lowercase, strip, collapse spaces, remove special chars."""
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower().strip())


def alias_score(raw: str, aliases: list[str]) -> float:
    """Return best match score between raw column name and a list of aliases."""
    raw_n = normalise(raw)
    best = 0.0
    for alias in aliases:
        alias_n = normalise(alias)
        # Exact match
        if raw_n == alias_n:
            return 1.0
        # Contains match (e.g. "total hours worked" contains "hours")
        if alias_n in raw_n or raw_n in alias_n:
            score = 0.85
        else:
            score = SequenceMatcher(None, raw_n, alias_n).ratio()
        best = max(best, score)
    return best


def build_column_mapping(raw_columns: list[str], threshold: float = 0.65) -> dict:
    """
    Given a list of raw column names from the uploaded file,
    return a mapping: {raw_column_name → canonical_field_name}.

    Columns that can't be matched above threshold are left unmapped.
    Each canonical field can only be claimed by one raw column (best score wins).
    """
    # Score every raw column against every canonical field
    scores = {}
    for col in raw_columns:
        scores[col] = {}
        for field, aliases in FIELD_ALIASES.items():
            scores[col][field] = alias_score(col, aliases)

    # Greedy assignment: highest score first, each field claimed once
    mapping = {}          # raw_col → canonical_field
    claimed_fields = set()

    # Flatten all (raw_col, field, score) and sort by score desc
    all_candidates = [
        (col, field, scores[col][field])
        for col in raw_columns
        for field in FIELD_ALIASES
        if scores[col][field] >= threshold
    ]
    all_candidates.sort(key=lambda x: x[2], reverse=True)

    for col, field, score in all_candidates:
        if col not in mapping and field not in claimed_fields:
            mapping[col] = field
            claimed_fields.add(field)

    # AI fallback: if required fields still unmapped, ask Ollama
    missing = [f for f in REQUIRED_FIELDS if f not in claimed_fields]
    if missing and _AI_AVAILABLE and is_ollama_available():
        try:
            ai_result = ai_map_columns(raw_columns, already_mapped=mapping)
            # ai_result maps field_name → column_index
            for field_name, col_idx in ai_result.items():
                if field_name in claimed_fields:
                    continue
                if isinstance(col_idx, (int, float)) and 0 <= int(col_idx) < len(raw_columns):
                    col_name = raw_columns[int(col_idx)]
                    if col_name not in mapping:
                        mapping[col_name] = field_name
                        claimed_fields.add(field_name)
                        print(f"[field_mapper] AI mapped: '{col_name}' → '{field_name}'")
        except Exception as e:
            print(f"[field_mapper] AI fallback failed: {e}")

    return mapping


def mapping_report(raw_columns: list[str], mapping: dict) -> dict:
    """
    Produce a human-readable mapping report for debugging / display.
    """
    mapped = {v: k for k, v in mapping.items()}   # canonical → raw
    missing_required = [f for f in REQUIRED_FIELDS if f not in mapped]
    missing_optional = [f for f in OPTIONAL_FIELDS if f not in mapped]
    unmapped_cols = [c for c in raw_columns if c not in mapping]

    return {
        "mapped_fields": {canon: raw for raw, canon in mapping.items()},
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "unmapped_columns": unmapped_cols,
        "confidence": "HIGH" if not missing_required else (
            "PARTIAL" if len(missing_required) < len(REQUIRED_FIELDS) else "LOW"
        ),
    }


def apply_mapping(rows: list[dict], mapping: dict) -> list[dict]:
    """
    Rename columns in each row from raw names → canonical names.
    Columns not in mapping are kept as-is under their original name (as extra_* fields).
    """
    result = []
    for row in rows:
        new_row = {}
        for raw_col, val in row.items():
            canonical = mapping.get(raw_col)
            if canonical:
                new_row[canonical] = val
                if canonical == "hours":
                    new_row["_hours_original_col"] = raw_col  # preserve for days→hours
            else:
                new_row[f"extra_{raw_col}"] = val
        result.append(new_row)
    return result
