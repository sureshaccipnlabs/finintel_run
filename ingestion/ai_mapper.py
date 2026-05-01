"""
ai_mapper.py — LLM-powered sheet analyser (Ollama/OpenAI).

Two levels of AI assistance:
  1. ai_map_columns  — given a header row, map columns to semantic fields.
  2. ai_analyze_sheet — given the first N rows, detect full sheet structure
     (header row, column mapping, project name, data range).

Providers:
    - Ollama (local): set AI_PROVIDER=ollama (or auto)
    - OpenAI (API): set AI_PROVIDER=openai (or auto) and OPENAI_API_KEY

Degrades gracefully — if configured provider is unreachable, returns empty results.
"""

import json
import os
import re
import urllib.request
import urllib.error

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")

OPENAI_URL = os.environ.get("OPENAI_URL", "https://api.openai.com/v1/chat/completions")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

AI_PROVIDER = os.environ.get("AI_PROVIDER", "openai").strip().lower()
AI_PROVIDER_ORDER = os.environ.get("AI_PROVIDER_ORDER", "openai,ollama")

TARGET_FIELDS = [
    "name", "project", "billing_rate", "cost_rate",
    "actual_hours", "billable_hours", "max_hours",
    "leaves", "working_days",
]

# ── Shared helpers ───────────────────────────────────────────────────────

def _ollama_generate(prompt, timeout=60):
    """Send a prompt to Ollama and return the raw response text."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
        return body.get("response", "")


def _openai_generate(prompt, timeout=60):
    """Send a prompt to OpenAI Chat Completions and return response text."""
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    payload = json.dumps({
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()

    req = urllib.request.Request(
        OPENAI_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}",
        },
    )

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode())
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        return message.get("content", "")


def _normalized_provider_order() -> list[str]:
    if AI_PROVIDER in ("ollama", "openai"):
        return [AI_PROVIDER]

    if AI_PROVIDER != "auto":
        return ["ollama", "openai"]

    order = []
    for token in (AI_PROVIDER_ORDER or "").split(","):
        p = token.strip().lower()
        if p in ("ollama", "openai") and p not in order:
            order.append(p)
    if not order:
        order = ["ollama", "openai"]
    return order


def _llm_generate(prompt, timeout=60):
    """Generate text using preferred provider selection logic."""
    errors = []
    for provider in _normalized_provider_order():
        try:
            if provider == "ollama":
                return _ollama_generate(prompt, timeout=timeout)
            if provider == "openai":
                return _openai_generate(prompt, timeout=timeout)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
            if AI_PROVIDER in ("ollama", "openai"):
                break
            continue

    raise RuntimeError("All configured LLM providers failed: " + " | ".join(errors))


def _clean_json_text(raw):
    """Strip JS-style comments and trailing commas that LLMs love to add."""
    raw = re.sub(r'//[^\n]*', '', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return raw.strip()


def _extract_json(text):
    """Pull the first JSON object out of LLM response text."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        cleaned = _clean_json_text(m.group(1))
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
    # Greedy: find outermost { ... }, then clean and parse
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                cleaned = _clean_json_text(text[start : i + 1])
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    start = None
    return None


_ollama_cache = {"available": None, "checked_at": 0}

def is_ollama_available():
    """Quick health check with 30s cache — returns True if Ollama is reachable."""
    import time
    now = time.time()
    if _ollama_cache["available"] is not None and (now - _ollama_cache["checked_at"]) < 30:
        return _ollama_cache["available"]
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=3) as resp:
            _ollama_cache["available"] = resp.status == 200
    except Exception:
        _ollama_cache["available"] = False
    _ollama_cache["checked_at"] = now
    return _ollama_cache["available"]


def is_openai_available():
    """OpenAI considered available when API key is configured."""
    return bool(OPENAI_API_KEY)


def is_llm_available():
    """Availability check based on configured provider preference."""
    for provider in _normalized_provider_order():
        if provider == "ollama" and is_ollama_available():
            return True
        if provider == "openai" and is_openai_available():
            return True
    return False


def get_active_provider() -> str | None:
    """Return the first currently available provider in preference order."""
    for provider in _normalized_provider_order():
        if provider == "ollama" and is_ollama_available():
            return "ollama"
        if provider == "openai" and is_openai_available():
            return "openai"
    return None


def get_ai_provider_status() -> dict:
    """Return runtime provider configuration and availability status."""
    return {
        "configured_provider": AI_PROVIDER,
        "provider_order": _normalized_provider_order(),
        "active_provider": get_active_provider(),
        "active_available": is_llm_available(),
        "providers": {
            "ollama": {
                "available": is_ollama_available(),
                "url": OLLAMA_URL,
                "model": OLLAMA_MODEL,
            },
            "openai": {
                "available": is_openai_available(),
                "url": OPENAI_URL,
                "model": OPENAI_MODEL,
                "api_key_configured": bool(OPENAI_API_KEY),
            },
        },
    }


def configure_ai_provider(
    provider: str | None = None,
    provider_order: str | None = None,
    openai_api_key: str | None = None,
    openai_model: str | None = None,
    openai_url: str | None = None,
    ollama_model: str | None = None,
    ollama_url: str | None = None,
) -> dict:
    """Update provider runtime configuration and return latest status."""
    global AI_PROVIDER, AI_PROVIDER_ORDER
    global OPENAI_API_KEY, OPENAI_MODEL, OPENAI_URL
    global OLLAMA_MODEL, OLLAMA_URL

    if provider is not None:
        p = provider.strip().lower()
        if p not in ("ollama", "openai", "auto"):
            raise ValueError("provider must be one of: ollama, openai, auto")
        AI_PROVIDER = p

    if provider_order is not None:
        AI_PROVIDER_ORDER = provider_order.strip()

    if openai_api_key is not None:
        OPENAI_API_KEY = openai_api_key.strip()
    if openai_model is not None:
        OPENAI_MODEL = openai_model.strip() or OPENAI_MODEL
    if openai_url is not None:
        OPENAI_URL = openai_url.strip() or OPENAI_URL

    if ollama_model is not None:
        OLLAMA_MODEL = ollama_model.strip() or OLLAMA_MODEL
    if ollama_url is not None:
        OLLAMA_URL = ollama_url.strip() or OLLAMA_URL
        _ollama_cache["available"] = None
        _ollama_cache["checked_at"] = 0

    return get_ai_provider_status()


# ── Level 1: Column mapping (given a known header row) ───────────────────

_COL_PROMPT = """\
You are a data analyst. Given column headers from a timesheet Excel file, \
map each to a semantic field.

Column headers (index → value):
{header_list}

Fields:
- name: employee/person name
- project: project, client, account, engagement
- billing_rate: billing rate per hour (NOT a dollar total/amount)
- cost_rate: cost/CTC rate per hour
- actual_hours: actual hours worked
- billable_hours: final billable hours (NOT the same as max/approved hours)
- max_hours: maximum/approved/expected/budgeted hours
- leaves: leave days, vacation, PTO
- working_days: working days

Return ONLY a JSON: {{"name": <idx>, "project": <idx>, ...}}
Use null if a field has no matching column.
"""


def ai_map_columns(header_row, already_mapped=None):
    """Map header columns to semantic fields via LLM."""
    already_mapped = already_mapped or {}
    unmapped = [f for f in TARGET_FIELDS if f not in already_mapped]
    if not unmapped:
        return {}
    parts = []
    for i, v in enumerate(header_row):
        parts.append(f"  {i}: {str(v).strip() if v else '(empty)'}")
    prompt = _COL_PROMPT.format(header_list="\n".join(parts))
    try:
        raw = _llm_generate(prompt)
    except Exception as exc:
        print(f"[ai_mapper] column mapping failed ({exc})")
        return {}
    mapping = _extract_json(raw)
    if not mapping:
        print(f"[ai_mapper] could not parse column response: {raw[:200]}")
        return {}
    result = {}
    max_col = len(header_row) - 1
    for field in unmapped:
        idx = mapping.get(field)
        if idx is not None and isinstance(idx, (int, float)) and 0 <= int(idx) <= max_col:
            result[field] = int(idx)
    if result:
        print(f"[ai_mapper] AI column mapping: {result}")
    return result


# ── Level 2: Full sheet structure analysis ───────────────────────────────

_SHEET_PROMPT = """\
You are a data analyst. Analyze the following rows from a timesheet Excel sheet \
and determine its structure.

Sheet name: "{sheet_name}"

Rows (row_index: [cell values]):
{rows_text}

Determine:
1. header_row: Which row index contains the main data column headers (the row \
with labels like Name, Hours, Rate, etc.)
2. columns: Map each semantic field to its 0-based column index in that header row:
   - name: employee/person name column
   - project: project/client/account column (null if not a column)
   - billing_rate: billing rate per hour (NOT a dollar total)
   - cost_rate: cost/CTC rate per hour
   - actual_hours: actual hours worked
   - billable_hours: final billable hours
   - max_hours: max/approved/expected hours
   - leaves: leave/vacation/PTO days
   - working_days: working days
3. project_name: If "project" column is null, what is the project name from \
the sheet metadata/header area? (null if unknown)
4. data_start: Row index where employee data rows begin (first row after header)
5. data_end: Row index of the last employee data row (before totals/blank rows)

Return ONLY a JSON object:
{{"header_row": <int>, "columns": {{"name": <int>, "project": <int|null>, ...}}, \
"project_name": <string|null>, "data_start": <int>, "data_end": <int>}}
"""


def ai_analyze_sheet(rows, sheet_name="Sheet1", max_rows=35):
    """
    Ask LLM to detect the full structure of a sheet.

    Returns dict with keys: header_row, columns, project_name, data_start, data_end
    or None on failure.
    """
    # Build text representation of rows
    lines = []
    for i, row in enumerate(rows[:max_rows]):
        cells = []
        for v in row:
            if v is None:
                cells.append("")
            else:
                cells.append(str(v).strip()[:40])
        lines.append(f"  {i}: {cells}")
    rows_text = "\n".join(lines)
    prompt = _SHEET_PROMPT.format(sheet_name=sheet_name, rows_text=rows_text)

    try:
        raw = _llm_generate(prompt, timeout=90)
    except Exception as exc:
        print(f"[ai_mapper] sheet analysis failed ({exc})")
        return None

    result = _extract_json(raw)
    if not result or "header_row" not in result or "columns" not in result:
        print(f"[ai_mapper] could not parse sheet analysis: {raw[:300]}")
        return None

    # Validate types
    try:
        result["header_row"] = int(result["header_row"])
        result["data_start"] = int(result.get("data_start", result["header_row"] + 1))
        result["data_end"]   = int(result.get("data_end", min(result["data_start"] + 30, len(rows) - 1)))
    except (ValueError, TypeError):
        return None

    cols = result.get("columns", {})
    max_col = max(len(r) for r in rows[:max_rows]) - 1 if rows else 0
    clean_cols = {}
    for field in TARGET_FIELDS:
        idx = cols.get(field)
        if idx is not None:
            try:
                idx = int(idx)
                if 0 <= idx <= max_col:
                    clean_cols[field] = idx
            except (ValueError, TypeError):
                pass
    result["columns"] = clean_cols

    proj = result.get("project_name")
    if proj and isinstance(proj, str) and proj.strip():
        result["project_name"] = proj.strip()
    else:
        result["project_name"] = None

    print(f"[ai_mapper] sheet analysis: header={result['header_row']}, "
          f"cols={result['columns']}, project={result['project_name']}, "
          f"data={result['data_start']}-{result['data_end']}")
    return result
