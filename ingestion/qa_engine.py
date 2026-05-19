"""
qa_engine.py — LLM-powered Q&A over the ingested financial dataset.

Builds a compact text summary of GLOBAL_DATASET and sends it as context
to Ollama so the LLM can answer any natural-language question about the data.
"""

import json
import re
import hashlib
import time
import os
import logging
from datetime import date as _date

from .dataset import (
    GLOBAL_DATASET, build_projects, build_monthly, build_overall_summary,
    build_top_performers, build_risks, get_months_available, filter_by_range,
    build_employee_summaries, set_on_dataset_change_callback, _trend_from_values,
    _calc_margin, match_entities_by_word_boundary
)
from .ai_mapper import _llm_generate, is_llm_available

# ── Debug Logging Configuration ───────────────────────────────────────────────
DEBUG = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")

_logger = logging.getLogger("qa_engine")
if not _logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(asctime)s] [qa_engine] %(message)s", datefmt="%H:%M:%S"))
    _logger.addHandler(_handler)
    _logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

def _log(msg: str, level: str = "info"):
    """Structured logger for qa_engine (respects DEBUG env var for debug level)."""
    if level == "debug" and not DEBUG:
        return
    getattr(_logger, level if hasattr(_logger, level) else "info")(msg)

# Use legacy forecast_ module (set USE_NEW_FORECAST=False to use old implementation)
USE_NEW_FORECAST = False  # Set to True to use new OOP module

if USE_NEW_FORECAST:
    try:
        from .forecast import try_answer_forecast, is_likely_forecast
        _USING_NEW_FORECAST = True
        print("[qa_engine] Using NEW OOP forecast module")
    except ImportError as e:
        from .forecast_ import try_answer_forecast, is_likely_forecast
        _USING_NEW_FORECAST = False
        print(f"[qa_engine] Using LEGACY forecast_ module (import error: {e})")
else:
    from .forecast_ import try_answer_forecast, is_likely_forecast
    _USING_NEW_FORECAST = False
    print("[qa_engine] Using LEGACY forecast_ module (configured)")


# ── Threshold comparison utility ─────────────────────────────────────────────

def _apply_filter(value, op: str, threshold) -> bool:
    """Return True if value satisfies the comparison (op, threshold). Used in tests."""
    if value is None:
        return False
    try:
        v, t = float(value), float(threshold)
    except (TypeError, ValueError):
        return False
    return {"<": v < t, "<=": v <= t, ">": v > t, ">=": v >= t, "==": v == t}.get(op, False)


# ── QA-specific JSON extraction ───────────────────────────────────────────────

def _clean_qa_json(raw: str) -> str:
    """Clean LLM JSON output: strip comments, trailing commas, fix number formatting."""
    # Strip JS-style comments
    raw = re.sub(r'//[^\n]*', '', raw)
    # Remove trailing commas before } or ]
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    # Fix numbers with commas like 18,816.0 -> 18816.0
    def fix_number_commas(match):
        return match.group(0).replace(',', '')
    raw = re.sub(r'(?<=[:\s\[])(\d{1,3}(?:,\d{3})+(?:\.\d+)?)', fix_number_commas, raw)
    return raw.strip()


def _extract_qa_json(text: str):
    """Extract JSON from Q&A LLM response. Independent of ai_mapper._extract_json."""
    if not text:
        return None
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try with cleaning
    try:
        return json.loads(_clean_qa_json(text))
    except json.JSONDecodeError:
        pass
    # Try extracting from ALL markdown code blocks — prefer the richest one
    candidates = []
    for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
        try:
            obj = json.loads(_clean_qa_json(m.group(1)))
            candidates.append(obj)
        except json.JSONDecodeError:
            pass
    if candidates:
        # Prefer block that has visual_type + data; otherwise pick by key count
        def _score(o):
            return (bool(o.get("visual_type")) + bool(o.get("data"))) * 100 + len(o)
        return max(candidates, key=_score)
    # Greedy: find outermost { ... }
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(_clean_qa_json(text[start:i + 1]))
                except json.JSONDecodeError:
                    start = None
    return None


# ── Context Caching ──────────────────────────────────────────────────────────

_CONTEXT_CACHE = {
    "hash": None,
    "context": None,
    "timestamp": 0,
}
_CACHE_TTL_SECONDS = 300  # 5 minutes TTL as backup


def _compute_dataset_hash(records) -> str:
    """Compute a hash of the dataset for cache invalidation."""
    if not records:
        return "empty"
    # Hash based on: record count + first record + last record
    parts = [
        str(len(records)),
        str(records[0]) if records else "",
        str(records[-1]) if records else "",
    ]
    return hashlib.md5("||".join(parts).encode()).hexdigest()


def _get_cached_context(records=None, question: str = None):
    """Get context from cache if valid, otherwise rebuild and cache.
    
    Note: When question is provided, we build a targeted context (not cached).
    Full context is cached for fallback/general use.
    """
    recs = records if records is not None else GLOBAL_DATASET
    
    # If question provided, build targeted context (don't cache - question-specific)
    if question:
        return _build_dataset_context(recs, question=question)
    
    # For general context (no question), use cache
    current_hash = _compute_dataset_hash(recs)
    now = time.time()
    
    # Check if cache is valid (hash matches AND within TTL)
    if (_CONTEXT_CACHE["hash"] == current_hash 
        and _CONTEXT_CACHE["context"] is not None
        and (now - _CONTEXT_CACHE["timestamp"]) < _CACHE_TTL_SECONDS):
        return _CONTEXT_CACHE["context"]
    
    # Rebuild full context
    context = _build_dataset_context(recs)
    
    # Update cache
    _CONTEXT_CACHE["hash"] = current_hash
    _CONTEXT_CACHE["context"] = context
    _CONTEXT_CACHE["timestamp"] = now
    
    return context


def invalidate_context_cache():
    """Manually invalidate the context cache (call after dataset changes)."""
    _CONTEXT_CACHE["hash"] = None
    _CONTEXT_CACHE["context"] = None
    _CONTEXT_CACHE["timestamp"] = 0


# ── Response Caching ──────────────────────────────────────────────────────────

_RESPONSE_CACHE = {}
_RESPONSE_CACHE_TTL = 300  # 5 minutes
_RESPONSE_CACHE_MAX_SIZE = 100


def _get_response_cache_key(question: str, data_hash: str) -> str:
    """Generate cache key from question + data hash."""
    normalized = question.lower().strip()
    return hashlib.md5(f"{normalized}|{data_hash}".encode()).hexdigest()


def _get_cached_response(question: str, data_hash: str):
    """Return cached response if valid, else None."""
    key = _get_response_cache_key(question, data_hash)
    entry = _RESPONSE_CACHE.get(key)
    if entry and (time.time() - entry["timestamp"]) < _RESPONSE_CACHE_TTL:
        print(f"[qa_engine] Cache HIT for: {question[:50]}...")
        return entry["response"]
    return None


def _cache_response(question: str, data_hash: str, response: dict):
    """Store response in cache."""
    if len(_RESPONSE_CACHE) >= _RESPONSE_CACHE_MAX_SIZE:
        # Remove oldest entry
        oldest = min(_RESPONSE_CACHE, key=lambda k: _RESPONSE_CACHE[k]["timestamp"])
        del _RESPONSE_CACHE[oldest]
    
    key = _get_response_cache_key(question, data_hash)
    _RESPONSE_CACHE[key] = {"response": response, "timestamp": time.time()}
    print(f"[qa_engine] Cached response for: {question[:50]}...")


def invalidate_response_cache():
    """Clear response cache when data changes."""
    _RESPONSE_CACHE.clear()
    print("[qa_engine] Response cache invalidated")


# Register cache invalidation callback with dataset module
def _invalidate_all_caches():
    """Invalidate both context and response caches."""
    invalidate_context_cache()
    invalidate_response_cache()

set_on_dataset_change_callback(_invalidate_all_caches)


def _fuzzy_match_suggestion(typo: str, known_names: list, threshold: float = 0.6) -> str:
    """Find closest match for a typo using simple similarity ratio.
    
    Returns the best match if similarity >= threshold, else None.
    """
    if not typo or not known_names:
        return None
    typo_lower = typo.lower()
    best_match = None
    best_score = 0
    
    for name in known_names:
        name_lower = name.lower()
        # Simple similarity: common characters / max length
        common = sum(1 for c in typo_lower if c in name_lower)
        score = common / max(len(typo_lower), len(name_lower))
        # Bonus for same starting letter
        if typo_lower and name_lower and typo_lower[0] == name_lower[0]:
            score += 0.2
        # Bonus for substring match
        if typo_lower in name_lower or name_lower in typo_lower:
            score += 0.3
        if score > best_score:
            best_score = score
            best_match = name
    
    return best_match if best_score >= threshold else None


# ── Context builder ──────────────────────────────────────────────────────────

def _detect_question_scope(question: str, records: list = None) -> dict:
    """Analyze question to determine what context sections are needed.
    
    Args:
        question: The user's question
        records: Optional dataset records to validate entity names against actual data
    """
    q = (question or "").lower()
    
    # Fix 2: Build lookup sets from actual data (case-insensitive)
    actual_employees = set()
    actual_employees_original = {}  # lowercase -> original case
    actual_projects = set()
    actual_projects_original = {}  # lowercase -> original case
    if records:
        for r in records:
            emp = r.get("employee", "")
            proj = r.get("project", "")
            if emp:
                emp_lower = emp.lower()
                actual_employees.add(emp_lower)
                actual_employees_original[emp_lower] = emp
                # Also add first name for partial matching
                first_name = emp.split()[0].lower() if emp else ""
                if first_name:
                    actual_employees.add(first_name)
                    actual_employees_original[first_name] = emp.split()[0]
            if proj:
                proj_lower = proj.lower()
                actual_projects.add(proj_lower)
                actual_projects_original[proj_lower] = proj
    
    scope = {
        "needs_overall": True,  # Always include
        "needs_projects": False,
        "needs_monthly": False,
        "needs_employee_summary": False,
        "needs_employee_details": False,
        "needs_risks": False,
        "specific_employee": None,
        "specific_project": None,
        "specific_month": None,
        "specific_year": None,
        "mentioned_projects": [],
        "missing_entities": [],  # Fix 2: Track entities not found in data
    }
    
    # Project-related keywords (expanded)
    project_keywords = [
        "project", "projects", "client", "clients", "projecty", "projectx",
        "which project", "project health", "project performance"
    ]
    if any(w in q for w in project_keywords):
        scope["needs_projects"] = True
    
    # Monthly/trend-related (expanded)
    monthly_keywords = [
        "month", "monthly", "trend", "over time", "by month", "each month",
        "quarter", "q1", "q2", "q3", "q4", "growth", "change", "from", "to",
        "ytd", "year to date", "last quarter", "this quarter", "previous month",
        "last month", "this month", "fiscal", "fy", "weekly", "week"
    ]
    if any(w in q for w in monthly_keywords):
        scope["needs_monthly"] = True
    
    # Specific month mentioned (with optional year)
    # \b word-boundary anchors prevent matching 'mar' inside 'margin', 'apr' inside 'april', etc.
    month_pattern = r"\b(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b[\s,\-]*(\d{4})?"
    month_match = re.search(month_pattern, q)
    if month_match:
        scope["needs_monthly"] = True
        scope["needs_employee_details"] = True
        scope["specific_month"] = month_match.group(1)
        scope["specific_year"] = month_match.group(2)  # e.g. "2024" or None

    # Year-only mention (no month): e.g. "in 2024", "for 2024", "2024 revenue"
    if not scope["specific_year"]:
        year_only = re.search(r"\b(20\d{2})\b", q)
        if year_only:
            scope["specific_year"] = year_only.group(1)
    
    # Employee-related keywords (expanded for natural language + designation roles)
    employee_keywords = [
        "employee", "employees", "person", "people", "who", "top", "bottom", 
        "best", "worst", "performer", "performers", "resource", "resources",
        "team", "staff", "idle", "sitting", "working", "contributor", "contributors",
        "underutilized", "overloaded", "free", "busy", "risky", "productive",
        # Designation / role-based keywords
        "senior", "junior", "lead", "developer", "developers", "architect", "architects",
        "consultant", "consultants", "manager", "managers", "analyst", "analysts",
        "engineer", "engineers", "designer", "director", "executive", "head",
        "designation", "role", "title", "seniority"
    ]
    if any(w in q for w in employee_keywords):
        scope["needs_employee_summary"] = True
    
    # Specific employee name mentioned (capitalized word that's not a common word)
    common_words = {
        "give", "me", "show", "list", "what", "how", "the", "for", "and", "with", "summary",
        "detail", "details", "total", "revenue", "profit", "cost", "margin", "hours",
        "attendance", "utilization", "which", "who", "where", "when", "why", "does", "did",
        # Contraction stems: "What's" → "Whats", "Who's" → "Whos", etc.
        "whats", "whos", "wheres", "hows", "thats", "theres", "heres", "lets",
        # Other common question/sentence starters
        "please", "can", "could", "would", "should", "do", "tell", "give", "need",
        "want", "find", "fetch", "pull", "return", "provide", "any", "some", "no",
        "overall", "aggregate", "blended", "combined", "all", "average", "avg",
        "employees", "employee", "project", "projects", "highest", "lowest", "top", "bottom",
        "best", "worst", "most", "least", "generated", "has", "have", "had", "are", "is", "was",
        "show", "compare", "trend", "performance", "health", "summary", "insights", "issues",
        "risks", "anomalies", "identify", "detect", "predict", "suggest", "analyze",
        # Action words (Fix 1: prevent these from being detected as names)
        "compute", "calculate", "find", "get", "display", "tell", "about",
        # Time period words
        "monthly", "weekly", "daily", "yearly", "annual", "quarterly",
        "forecast", "projection", "budget", "estimate", "expected", "actual",
        # Common adjectives/descriptors that may appear capitalized in questions
        "high", "low", "medium", "good", "bad", "poor", "strong", "weak",
        "positive", "negative", "active", "inactive", "current", "previous",
        "new", "old", "recent", "latest", "last", "first", "second", "third",
        "maximum", "minimum", "max", "min", "single", "multiple", "each",
        "only", "also", "other", "another", "same", "different", "similar",
        "total", "net", "gross", "base", "full", "partial", "specific",
        "top", "bottom", "above", "below", "under", "over", "between",
        # Business/financial adjectives
        "billable", "non", "unbillable", "offshore", "onshore", "senior",
        "junior", "lead", "associate", "principal", "key", "critical",
        # Common gerunds / performance descriptors used in questions
        "performing", "contributing", "generating", "earning", "billing",
        "underperforming", "overperforming", "utilizing", "working", "running",
    }
    
    # Fix 2: Data-based entity detection (works with any case)
    # Short words to skip (prepositions, articles, etc.)
    skip_words = {"in", "on", "at", "to", "of", "by", "vs", "or", "an", "a", "the", "is", "it"}
    
    if records and (actual_employees or actual_projects):
        # Check each word in question against actual data
        words = re.findall(r"[a-zA-Z]+(?:'s)?", question)
        for word in words:
            # Clean the word: remove possessive 's and convert to lowercase
            word_clean = word.lower()
            if word_clean.endswith("'s"):
                word_clean = word_clean[:-2]
            
            # Skip common words, short words, and month names (exact match only)
            if word_clean in common_words or word_clean in skip_words or re.fullmatch(month_pattern, word_clean):
                continue
            
            # Check if it's an actual employee (exact match only for short words)
            if len(word_clean) <= 2:
                is_employee = word_clean in actual_employees
                is_project = word_clean in actual_projects
            else:
                is_employee = word_clean in actual_employees or any(word_clean in e for e in actual_employees)
                is_project = word_clean in actual_projects or any(word_clean in p for p in actual_projects)
            
            if is_project and not is_employee:
                # Definitely a project
                scope["needs_projects"] = True
                scope["specific_project"] = actual_projects_original.get(word_clean, word.title())
                _log(f"Detected project: '{word_clean}' -> '{scope['specific_project']}'", "debug")
            elif is_employee and not is_project:
                # Definitely an employee
                scope["needs_employee_summary"] = True
                scope["needs_employee_details"] = True
                scope["specific_employee"] = actual_employees_original.get(word_clean, word.title())
            elif is_employee and is_project:
                # Fix 4: Ambiguous - use context clues to disambiguate
                project_context = any(kw in q for kw in ["project", "client", "project's", "of project"])
                employee_context = any(kw in q for kw in ["employee", "person", "employee's", "'s performance", "'s margin", "'s utilization"])
                
                if project_context and not employee_context:
                    scope["needs_projects"] = True
                    scope["specific_project"] = actual_projects_original.get(word_clean, word.title())
                elif employee_context and not project_context:
                    scope["needs_employee_summary"] = True
                    scope["needs_employee_details"] = True
                    scope["specific_employee"] = actual_employees_original.get(word_clean, word.title())
                else:
                    # Still ambiguous - include both
                    scope["needs_employee_summary"] = True
                    scope["needs_employee_details"] = True
                    scope["specific_employee"] = actual_employees_original.get(word_clean, word.title())
                    scope["needs_projects"] = True
                    scope["specific_project"] = actual_projects_original.get(word_clean, word.title())
            elif len(word_clean) > 2 and word[0].isupper():
                # Capitalized word not in data - track as missing and try fuzzy match
                missing_word = word.replace("'s", "")
                scope["missing_entities"].append(missing_word)
                # Try fuzzy match for typo correction
                all_known = list(actual_projects_original.values()) + list(actual_employees_original.values())
                suggestion = _fuzzy_match_suggestion(missing_word, all_known)
                if suggestion:
                    if "suggestions" not in scope:
                        scope["suggestions"] = []
                    scope["suggestions"].append(f"'{missing_word}' -> Did you mean '{suggestion}'?")
    else:
        # Fallback: Old behavior when no records provided (capitalized words)
        name_match = re.findall(r"\b([A-Z][a-z]+)\b", question)
        for name in name_match:
            if name.lower() not in common_words and not re.match(month_pattern, name.lower()):
                scope["needs_employee_summary"] = True
                scope["needs_employee_details"] = True
                scope["specific_employee"] = name
                break
    
    # Apply shared entity matcher to reinforce specific project/employee detection
    try:
        projects_list = list(actual_projects_original.values()) if actual_projects_original else []
        employees_list = list(actual_employees_original.values()) if actual_employees_original else []
        matched_projects = match_entities_by_word_boundary([question], projects_list) if question else []
        matched_employees = match_entities_by_word_boundary([question], employees_list) if question else []
        if matched_projects and not scope.get("specific_project"):
            scope["needs_projects"] = True
            if len(matched_projects) == 1:
                scope["specific_project"] = matched_projects[0]
        if matched_employees:
            scope["needs_employee_summary"] = True
            scope["needs_employee_details"] = True
            if len(matched_employees) == 1 and not scope.get("specific_employee"):
                scope["specific_employee"] = matched_employees[0]
            elif len(matched_employees) >= 2:
                # Multiple employees detected — clear single-employee filter so all are shown
                scope["specific_employee"] = None
                scope["mentioned_employees"] = matched_employees
    except Exception:
        pass

    # For project utilization questions, ONLY need PROJECTS section (not employees)
    is_project_util_question = ("project" in q) and any(w in q for w in ["utilization", "utilisation", "util"])
    
    # Specific metrics that need employee details (expanded)
    metric_keywords = [
        "hours", "actual hours", "billable", "billable hours", "vacation", "leave", 
        "working days", "billing rate", 
        "cost rate", "zero hours", "no hours", "logged", "per hour", "per day",
        "per employee", "ratio", "average", "avg", "mean",
        "below", "above", "less than", "more than", "greater than", "threshold"
    ]
    if any(w in q for w in metric_keywords) and not is_project_util_question:
        scope["needs_employee_summary"] = True
        if any(w in q for w in ["his", "her", "their", "'s"]) or re.search(r"\b[A-Z][a-z]+\b", question):
            scope["needs_employee_details"] = True
    # Employee utilization questions need employee summary
    if any(w in q for w in ["utilization", "utilisation", "util"]) and not is_project_util_question:
        scope["needs_employee_summary"] = True
    
    # Risk-related keywords (expanded for natural language)
    risk_keywords = [
        "risk", "risks", "risky", "issue", "issues", "problem", "problems", 
        "concern", "flag", "flags", "warning", "warnings", "low utilization",
        "underperform", "underutilized", "idle", "zero", "not generating",
        "not working", "not billing", "not billable", "not profitable",
        "anomaly", "anomalies", "inconsistent", "validation", "declining",
        "losing", "leakage", "attention", "health"
    ]
    if any(w in q for w in risk_keywords):
        scope["needs_risks"] = True
        scope["needs_employee_summary"] = True
    
    # Comparison queries need multiple sections
    comparison_keywords = ["compare", "comparison", "vs", "versus", "difference", "between"]
    if any(w in q for w in comparison_keywords):
        scope["needs_projects"] = True
        scope["needs_employee_summary"] = True
        scope["needs_monthly"] = True
        # Resolve canonical project names for compare — prevents single-project filtering
        if actual_projects_original:
            mentioned = match_entities_by_word_boundary([question], list(actual_projects_original.values()))
            scope["mentioned_projects"] = mentioned
            if len(mentioned) >= 2:
                scope["specific_project"] = None  # Show ALL mentioned projects, not just one

    # Also detect multi-project references without explicit compare keyword
    # e.g. "BARCLAYS and CRYSTAL performance" — no 'compare', but two project names
    if actual_projects_original and not scope.get("mentioned_projects"):
        all_proj_matched = match_entities_by_word_boundary([question], list(actual_projects_original.values()))
        if len(all_proj_matched) >= 2:
            scope["mentioned_projects"] = all_proj_matched
            scope["specific_project"] = None
            scope["needs_projects"] = True
    
    # Insight/summary queries (executive level)
    insight_keywords = [
        "insight", "insights", "summary", "overview", "highlight", "observations",
        "key", "important", "performance summary", "health summary", "improve"
    ]
    if any(w in q for w in insight_keywords):
        scope["needs_projects"] = True
        scope["needs_employee_summary"] = True
        scope["needs_risks"] = True
    
    # Aggregation queries (totals, counts) - overall is enough
    if re.search(r"\b(total|sum|count|how many|how much|overall|all)\b", q) and not scope["needs_employee_details"]:
        pass  # Overall summary is sufficient
    
    # For short questions (<=3 words), infer from keywords
    if len(q.split()) <= 3:
        if any(w in q for w in ["revenue", "profit", "cost", "margin", "total", "overall", "rev", "pft"]):
            pass  # Overall summary already included
        elif any(w in q for w in ["employee", "employees", "who", "person", "emp", "top", "bottom"]):
            scope["needs_employee_summary"] = True
        elif any(w in q for w in ["project", "projects"]):
            scope["needs_projects"] = True
        elif any(w in q for w in ["month", "monthly", "trend"]):
            scope["needs_monthly"] = True
        elif any(w in q for w in ["risk", "risks", "issues", "problems"]):
            scope["needs_risks"] = True
            scope["needs_employee_summary"] = True
        else:
            scope["needs_employee_summary"] = True
    
    # Explicit "show all" type requests
    if any(w in q for w in ["everything", "all data", "full summary", "complete"]):
        scope["needs_projects"] = True
        scope["needs_monthly"] = True
        scope["needs_employee_summary"] = True
        scope["needs_risks"] = True
    
    return scope


def _build_dataset_context(records=None, question: str = None):
    """Build a compact text summary of the dataset for the LLM prompt."""
    recs = records if records is not None else GLOBAL_DATASET
    if not recs:
        return "No data has been loaded yet."

    # Determine what sections are needed based on question (Fix 2: pass records for validation)
    scope = _detect_question_scope(question, recs) if question else None
    include_all = scope is None  # If no question, include everything

    # Year-aware context narrowing: when exactly 1 year is mentioned, restrict context to that year.
    # Applied locally here only — does NOT touch the records used by Forecast in ask().
    specific_year = scope.get("specific_year") if scope else None
    # Detect "this year" / "current year" / "YTD" phrases when no explicit 4-digit year found
    _THIS_YEAR_RE = re.compile(r"\b(this year|current year|this fiscal year|ytd|year to date)\b", re.IGNORECASE)
    if not specific_year and _THIS_YEAR_RE.search(question or ""):
        specific_year = str(_date.today().year)
    if specific_year:
        year_matches = re.findall(r"\b(20\d{2})\b", question or "")
        if len(year_matches) <= 1:  # Filter when 0 or 1 explicit year (not multi-year compare)
            recs = [r for r in recs if specific_year in (r.get("month") or "")]

    # Resolve the canonical project name for employee-scoped filtering.
    # Must be computed early (before PROJECTS block) so EMPLOYEE DETAILS can be scoped.
    _specific_proj_scope = scope.get("specific_project") if scope else None
    _specific_emp_scope  = scope.get("specific_employee") if scope else None
    _matched_proj_for_emp: str | None = None
    if _specific_proj_scope:
        for _p in sorted({r.get("project") for r in recs if r.get("project")}):
            if _p.lower() == _specific_proj_scope.lower():
                # Prefer the uppercase/canonical form found in data
                _matched_proj_for_emp = _p.upper() if _p.upper() in {r.get("project") for r in recs} else _p
                break

    # emp_recs controls what EMPLOYEES summary sees:
    #   • project specified (alone OR with employee) → project-scoped
    #     If the employee works on that project they'll be found with correct margin;
    #     if they don't, "no data" is the correct answer — not blended totals.
    #   • employee-only or broad question → full recs
    _proj_lower = _matched_proj_for_emp.lower() if _matched_proj_for_emp else None
    emp_recs = (
        [r for r in recs if (r.get("project") or "").lower() == _proj_lower]
        if _proj_lower else recs
    )

    overall = build_overall_summary(recs)
    months = get_months_available(recs)

    lines = []

    # Overall - always include (compact)
    lines.append("=== OVERALL SUMMARY ===")
    lines.append(f"Total Revenue: ${overall['total_revenue']:,.2f}, Cost: ${overall['total_cost']:,.2f}, Profit: ${overall['total_profit']:,.2f}")
    lines.append(f"Hours: {overall.get('total_hours', 0)}, Margin: {overall['avg_margin_pct']}%, Employees: {overall['total_employees']}")
    lines.append(f"Months: {', '.join(months)}")
    if months:
        lines.append(f"LAST MONTH (most recent data available): {months[-1]}")
        lines.append(f"NOTE: If user asks about 'last month', return data for {months[-1]} and clarify this is the most recent data available.")
    
    # Add typo suggestions if any
    if scope and scope.get("suggestions"):
        lines.append("")
        lines.append("=== POSSIBLE TYPOS DETECTED ===")
        for suggestion in scope["suggestions"]:
            lines.append(f"WARNING: {suggestion}")
        lines.append("If the user meant one of the suggested names, use that data. Otherwise, respond with 'No data found'.")
    
    # Add missing entities warning
    if scope and scope.get("missing_entities"):
        lines.append("NOTE: No data found.")
    
    lines.append("")

    # Projects - include if needed
    if include_all or scope.get("needs_projects"):
        projects = build_projects(recs)
        
        # Build per-project monthly series for trends and avg utilisation
        proj_monthly = {}
        proj_billable = {}
        proj_util_values = {}  # Collect utilisation_pct per project for averaging
        for r in recs:
            proj = r.get("project") or "Unknown"
            month = r.get("month") or "Unknown"
            if proj not in proj_monthly:
                proj_monthly[proj] = {}
                proj_billable[proj] = 0.0
                proj_util_values[proj] = []
            if month not in proj_monthly[proj]:
                proj_monthly[proj][month] = {"rev": 0.0, "cost": 0.0, "util_vals": []}
            proj_monthly[proj][month]["rev"] += float(r.get("revenue") or 0)
            proj_monthly[proj][month]["cost"] += float(r.get("cost") or 0)
            # Track utilization per month for monthly trends
            util_pct_month = r.get("utilisation_pct")
            if util_pct_month is not None and util_pct_month > 0:
                proj_monthly[proj][month]["util_vals"].append(float(util_pct_month))
            proj_billable[proj] += float(r.get("billable_hours") or 0)
            # Collect utilisation_pct for averaging
            util_pct = r.get("utilisation_pct")
            if util_pct is not None and util_pct > 0:
                proj_util_values[proj].append(float(util_pct))
        
        # Pre-calculate utilization for sorting
        proj_utils = {}
        for name in projects.keys():
            util_values = proj_util_values.get(name, [])
            proj_utils[name] = round(sum(util_values) / len(util_values), 2) if util_values else 0
        
        # Sort by various metrics for ranking hints
        sorted_by_util = sorted(proj_utils.items(), key=lambda x: x[1], reverse=True)
        sorted_by_cost = sorted(projects.items(), key=lambda x: x[1].get("cost", 0))
        sorted_by_revenue = sorted(projects.items(), key=lambda x: x[1].get("revenue", 0), reverse=True)
        sorted_by_profit = sorted(projects.items(), key=lambda x: x[1].get("profit", 0), reverse=True)
        
        lines.append("=== PROJECTS ===")
        lines.append(f"AVAILABLE PROJECTS (use EXACT names): {', '.join(sorted(projects.keys()))}")

        # Inject canonical names for compare queries so LLM uses exact casing
        mentioned_projs = scope.get("mentioned_projects", []) if scope else []
        if len(mentioned_projs) >= 2:
            lines.append(
                f"COMPARE INSTRUCTION: User is comparing {' vs '.join(mentioned_projs)}. "
                f"Use EXACTLY these names as column headers: {', '.join(mentioned_projs)}"
            )

        # Filter to specific project if requested
        specific_proj = scope.get("specific_project") if scope else None
        _log(f"Building context: specific_project={specific_proj}, all_projects={list(projects.keys())}", "debug")
        if specific_proj:
            # Case-insensitive match to find the actual project name
            specific_proj_lower = specific_proj.lower()
            matched_proj = None
            for p in projects.keys():
                if p.lower() == specific_proj_lower:
                    matched_proj = p
                    break
            if matched_proj:
                _log(f"Filtering to project: {matched_proj}", "debug")
                lines.append(f">>> IMPORTANT: User is asking about PROJECT '{matched_proj}' ONLY. Use ONLY the data below for '{matched_proj}'. <<<")
                lines.append(f"FILTERED TO PROJECT: {matched_proj}")
                projects_to_show = {matched_proj: projects[matched_proj]}
            else:
                _log(f"Project '{specific_proj}' not found in {list(projects.keys())}", "debug")
                lines.append(f"WARNING: Requested project '{specific_proj}' not found. Showing all projects.")
                projects_to_show = projects
        else:
            # For compare queries with 2+ projects, restrict to mentioned projects only
            if len(mentioned_projs) >= 2:
                projects_to_show = {p: projects[p] for p in mentioned_projs if p in projects}
                if not projects_to_show:  # fallback if none matched
                    projects_to_show = projects
            else:
                projects_to_show = projects

        if not specific_proj:
            lines.append(f"RANKING HINTS: HighestUtilization={sorted_by_util[0][0]}, LowestUtilization={sorted_by_util[-1][0]}, "
                         f"HighestRevenue={sorted_by_revenue[0][0]}, LowestRevenue={sorted_by_revenue[-1][0]}, "
                         f"HighestCost={sorted_by_cost[-1][0]}, LowestCost={sorted_by_cost[0][0]}, "
                         f"HighestProfit={sorted_by_profit[0][0]}, LowestProfit={sorted_by_profit[-1][0]}")
        for name, data in sorted(projects_to_show.items(), key=lambda x: x[1]["revenue"], reverse=True):
            proj_revenue = float(data.get("revenue") or 0)
            proj_cost = float(data.get("cost") or 0)
            proj_margin = round(((proj_revenue - proj_cost) / proj_revenue) * 100, 2) if proj_revenue > 0 else 0
            
            # Determine project status based on margin (same logic as /projects API)
            if proj_margin > 40:
                proj_status = "Healthy"
            elif proj_margin >= 30:
                proj_status = "Optimal"
            else:
                proj_status = "At Risk"
            
            # Calculate trends from monthly series
            monthly_data = proj_monthly.get(name, {})
            sorted_months = sorted(monthly_data.keys(), key=lambda m: get_months_available([{"month": m}]))
            revenue_series = [monthly_data[m]["rev"] for m in sorted_months]
            cost_series = [monthly_data[m]["cost"] for m in sorted_months]
            profit_series = [monthly_data[m]["rev"] - monthly_data[m]["cost"] for m in sorted_months]
            margin_series = [_calc_margin(monthly_data[m]["rev"], monthly_data[m]["cost"]) for m in sorted_months]
            # Calculate per-month utilization averages
            util_series = []
            for m in sorted_months:
                uvals = monthly_data[m].get("util_vals", [])
                util_series.append(round(sum(uvals) / len(uvals), 2) if uvals else 0)
            
            rev_trend = _trend_from_values(revenue_series)
            cost_trend = _trend_from_values(cost_series)
            profit_trend = _trend_from_values(profit_series)
            margin_trend = _trend_from_values(margin_series)
            util_trend = _trend_from_values(util_series)
            billable_hrs = round(proj_billable.get(name, 0), 2)
            
            # Calculate AvgUtilisation from employee utilisation_pct values
            util_values = proj_util_values.get(name, [])
            avg_util = round(sum(util_values) / len(util_values), 2) if util_values else 0
            
            # Calculate best/worst month for this project
            proj_months = proj_monthly.get(name, {})
            if proj_months:
                best_rev_month = max(proj_months.items(), key=lambda x: x[1]["rev"])[0]
                worst_rev_month = min(proj_months.items(), key=lambda x: x[1]["rev"])[0]
            else:
                best_rev_month = worst_rev_month = "N/A"
            
            lines.append(
                f"- {name}: Revenue=${data['revenue']:,.2f}, Cost=${data.get('cost', 0):,.2f}, "
                f"Profit=${data['profit']:,.2f}, GrossMargin={proj_margin}%, "
                f"AvgUtilisation={avg_util}%, Employees={data['employees']}, "
                f"BillableHours={billable_hrs}, Status={proj_status}, "
                f"RevenueTrend={rev_trend}, CostTrend={cost_trend}, ProfitTrend={profit_trend}, MarginTrend={margin_trend}, UtilTrend={util_trend}, "
                f"BestRevenueMonth={best_rev_month}, WorstRevenueMonth={worst_rev_month}, "
                f"MonthlyBreakdown(last {min(6, len(sorted_months))} months)=[{', '.join(f'{m}:Rev=${monthly_data[m]["rev"]:,.0f}/Cost=${monthly_data[m]["cost"]:,.0f}/Util={round(sum(monthly_data[m].get("util_vals", [0])) / max(len(monthly_data[m].get("util_vals", [1])), 1), 1)}%' for m in sorted_months[-6:])}]"
            )
        lines.append("")

    # Month-specific data section - ADD when specific month is mentioned
    # This provides accurate month-specific metrics without breaking trend/comparison queries
    if scope and scope.get("specific_month"):
        month_key = scope["specific_month"].lower()
        # Find full month name for display
        month_names = {
            "jan": "January", "feb": "February", "mar": "March", "apr": "April",
            "may": "May", "jun": "June", "jul": "July", "aug": "August",
            "sep": "September", "oct": "October", "nov": "November", "dec": "December",
            "january": "January", "february": "February", "march": "March", "april": "April",
            "june": "June", "july": "July", "august": "August",
            "september": "September", "october": "October", "november": "November", "december": "December"
        }
        display_month = month_names.get(month_key, month_key.capitalize())
        
        # Filter records for this specific month (year-aware)
        specific_year = scope.get("specific_year") if scope else None
        month_filtered_recs = [
            r for r in recs
            if month_key in (r.get("month") or "").lower()
            and (not specific_year or specific_year in (r.get("month") or ""))
        ]
        
        if month_filtered_recs:
            lines.append(f"=== DATA FOR {display_month.upper()} ONLY ===")
            lines.append(f"NOTE: Use this section when answering questions about {display_month} specifically.")
            
            # Build month-specific project data
            month_projects = build_projects(month_filtered_recs)
            if month_projects:
                lines.append("Projects:")
                for name, data in sorted(month_projects.items(), key=lambda x: x[1]["revenue"], reverse=True):
                    proj_revenue = float(data.get("revenue") or 0)
                    proj_cost = float(data.get("cost") or 0)
                    proj_profit = proj_revenue - proj_cost
                    proj_margin = round((proj_profit / proj_revenue) * 100, 2) if proj_revenue > 0 else 0
                    lines.append(
                        f"- {name}: Revenue=${proj_revenue:,.2f}, Cost=${proj_cost:,.2f}, "
                        f"Profit=${proj_profit:,.2f}, Margin={proj_margin}%"
                    )
            
            # Build month-specific employee data
            month_employees = build_employee_summaries(month_filtered_recs)
            if month_employees:
                lines.append("Employees:")
                for emp in sorted(month_employees, key=lambda x: x.get('total_revenue') or 0, reverse=True)[:10]:
                    lines.append(
                        f"- {emp['employee_name']}: Revenue=${emp['total_revenue']:,.2f}, "
                        f"Profit=${emp['total_profit']:,.2f}, Hours={emp['total_hours']}, "
                        f"Utilization={emp.get('utilization_pct') or 0}%"
                    )
            
            # Month totals
            month_overall = build_overall_summary(month_filtered_recs)
            lines.append(
                f"Month Totals: Revenue=${month_overall['total_revenue']:,.2f}, "
                f"Cost=${month_overall['total_cost']:,.2f}, Profit=${month_overall['total_profit']:,.2f}, "
                f"Margin={month_overall['avg_margin_pct']}%"
            )
            lines.append("")

    # Monthly - include if needed (sorted chronologically)
    if include_all or scope.get("needs_monthly"):
        monthly = build_monthly(recs)
        lines.append("=== MONTHLY ===")
        # Sort months chronologically using the same order as get_months_available
        sorted_months = [m for m in months if m in monthly]
        
        # Add ranking hints for months
        if sorted_months:
            sorted_by_revenue = sorted(sorted_months, key=lambda m: monthly[m].get('total_revenue', 0), reverse=True)
            sorted_by_profit = sorted(sorted_months, key=lambda m: monthly[m].get('total_profit', 0), reverse=True)
            lines.append(f"RANKING HINTS: HighestRevenue={sorted_by_revenue[0]}, LowestRevenue={sorted_by_revenue[-1]}, "
                         f"HighestProfit={sorted_by_profit[0]}, LowestProfit={sorted_by_profit[-1]}")
        
        for m in sorted_months:
            data = monthly[m]
            is_last = (m == months[-1]) if months else False
            marker = " [LAST MONTH]" if is_last else ""
            lines.append(
                f"- {m}{marker}: Revenue=${data['total_revenue']:,.2f}, Profit=${data['total_profit']:,.2f}, "
                f"Margin={data['avg_margin_pct']}%, Employees={data['employees']}"
            )
        lines.append("")

    # Employee Summaries - include if needed
    if include_all or scope.get("needs_employee_summary"):
        employee_summaries = build_employee_summaries(emp_recs)
        # Filter to specific employee if mentioned
        if scope and scope.get("specific_employee"):
            emp_name = scope["specific_employee"].lower()
            employee_summaries = [e for e in employee_summaries if emp_name in e["employee_name"].lower()]
        
        # Add ranking hints for employees
        if employee_summaries:
            sorted_by_util = sorted(employee_summaries, key=lambda x: x.get('utilization_pct') or 0, reverse=True)
            sorted_by_revenue = sorted(employee_summaries, key=lambda x: x.get('total_revenue') or 0, reverse=True)
            sorted_by_profit = sorted(employee_summaries, key=lambda x: x.get('total_profit') or 0, reverse=True)
            lines.append("=== EMPLOYEES ===")
            lines.append(f"RANKING HINTS: HighestUtilization={sorted_by_util[0]['employee_name']}, LowestUtilization={sorted_by_util[-1]['employee_name']}, "
                         f"HighestRevenue={sorted_by_revenue[0]['employee_name']}, LowestRevenue={sorted_by_revenue[-1]['employee_name']}, "
                         f"HighestProfit={sorted_by_profit[0]['employee_name']}, LowestProfit={sorted_by_profit[-1]['employee_name']}")
        else:
            lines.append("=== EMPLOYEES ===")
        # Label margin scope so LLM knows whether it's project-specific or blended
        _margin_scope_label = f"{_matched_proj_for_emp} only" if _matched_proj_for_emp else "overall"

        # Cap broad listings; no cap when a specific employee is requested
        _MAX_EMP_LIST = 20
        _is_specific = bool(scope and scope.get("specific_employee"))
        _display_emps = employee_summaries
        if not _is_specific and len(employee_summaries) > _MAX_EMP_LIST:
            # Sort by revenue desc so top contributors appear first
            _display_emps = sorted(
                employee_summaries, key=lambda x: x.get("total_revenue") or 0, reverse=True
            )[:_MAX_EMP_LIST]

        for emp in _display_emps:
            desig = emp.get('designation') or ''
            desig_str = f", Designation={desig}" if desig else ""
            lines.append(
                f"- {emp['employee_name']}{desig_str}: Revenue=${emp['total_revenue']:,.2f}, "
                f"Cost=${emp['total_cost']:,.2f}, Profit=${emp['total_profit']:,.2f}, "
                f"Margin={emp.get('gross_margin_pct') or emp.get('margin_pct') or 0}%({_margin_scope_label}), "
                f"Hours={emp['total_hours']}, IndividualUtilisation={emp['utilization_pct'] or 0}%, "
                f"Attendance={emp.get('attendance_pct', 100)}%, VacationDays={emp.get('vacation_days', 0)}"
            )
        if not _display_emps:
            lines.append("- No matching employee found")
        _hidden = len(employee_summaries) - len(_display_emps)
        if _hidden > 0:
            lines.append(f"NOTE: Showing top {_MAX_EMP_LIST} of {len(employee_summaries)} employees by revenue. Ask about a specific employee for full details.")
        lines.append("")

    # Employee Details — only for specific-employee drilldowns.
    # For broad "list employees" queries the EMPLOYEES summary above is sufficient
    # and avoids raw-record noise in the context.
    _specific_emp_for_details = scope.get("specific_employee") if scope else None
    if _specific_emp_for_details and (include_all or scope.get("needs_employee_details")):
        lines.append("=== EMPLOYEE DETAILS ===")
        filtered_recs = emp_recs  # already project-scoped when a project is specified
        emp_name = _specific_emp_for_details.lower()
        filtered_recs = [r for r in filtered_recs if emp_name in (r.get("employee") or "").lower()]
        if scope and scope.get("specific_month"):
            month_key = scope["specific_month"].lower()
            filtered_recs = [r for r in filtered_recs if month_key in (r.get("month") or "").lower()]

        # No row cap — a single employee has at most ~24 month×project records
        for r in filtered_recs:
            rev = r.get('revenue') or 0
            cst = r.get('cost') or 0
            pft = r.get('profit') or 0
            mgn = r.get('margin_pct') or (round((pft / rev) * 100, 2) if rev > 0 else 0)
            lines.append(
                f"- {r.get('employee', '?')} | {r.get('project', '?')} | {r.get('month', '?')} | "
                f"Hours={r.get('actual_hours', 0)}, Vacation={r.get('vacation_days', 0)}, "
                f"Revenue=${rev:,.2f}, Cost=${cst:,.2f}, Profit=${pft:,.2f}, Margin={mgn}%, "
                f"Utilisation={r.get('utilisation_pct', 0)}%"
            )
        lines.append("")

    # Risks - include if needed
    if include_all or scope.get("needs_risks"):
        risks = build_risks(recs)
        if risks:
            lines.append("=== RISKS ===")
            for r in risks:
                lines.append(f"- {r['employee']} | Project: {r.get('project', 'Unknown')} | Month: {r.get('month', '')} | Issue: {r['issue']}")
            lines.append("")

    return "\n".join(lines)


# ── Prompt template ──────────────────────────────────────────────────────────

_QA_PROMPT = """\
You are FinIntel AI. Answer from this data:

{context}
---
Question: {question}

Return ONLY valid JSON. No text before or after the JSON.

EXAMPLES:

Q: "What is total revenue?"
{{"summary": "Total revenue is $88,960.", "visual_type": "metric", "columns": [], "data": [{{"label": "Total Revenue", "value": 88960}}]}}

Q: "Give me overall summary"
{{"summary": "Total Revenue: $88,960, Cost: $77,216, Profit: $11,744, Margin: 13.2%", "visual_type": "metric", "columns": [], "data": [{{"label": "Total Revenue", "value": 88960}}, {{"label": "Total Cost", "value": 77216}}, {{"label": "Total Profit", "value": 11744}}, {{"label": "Margin", "value": 13.2}}]}}

Q: "Top 3 employees by profit"
{{"summary": "Top 3 employees by profit.", "visual_type": "table", "columns": ["employee", "profit"], "data": [{{"employee": "John", "profit": 50000}}, {{"employee": "Jane", "profit": 45000}}, {{"employee": "Bob", "profit": 40000}}]}}

Q: "Revenue by project"
{{"summary": "Revenue by project.", "visual_type": "table", "columns": ["project", "revenue"], "data": [{{"project": "Alpha", "revenue": 200000}}, {{"project": "Beta", "revenue": 150000}}]}}

Q: "How many employees?"
{{"summary": "There are 7 employees.", "visual_type": "metric", "columns": [], "data": [{{"label": "Employee Count", "value": 7}}]}}

Q: "Monthly revenue trend"
{{"summary": "Monthly revenue trend.", "visual_type": "table", "columns": ["month", "revenue"], "data": [{{"month": "January 2026", "revenue": 30000}}, {{"month": "February 2026", "revenue": 32000}}]}}

Q: "Employees with highest attendance"
{{"summary": "Employees sorted by highest attendance.", "visual_type": "table", "columns": ["employee", "attendance"], "data": [{{"employee": "Jane", "attendance": 100}}, {{"employee": "John", "attendance": 95}}, {{"employee": "Bob", "attendance": 90}}]}}

Q: "Which employees are not generating revenue"
{{"summary": "Employees with zero or negative revenue.", "visual_type": "table", "columns": ["employee", "revenue"], "data": [{{"employee": "Bob", "revenue": 0}}, {{"employee": "Jane", "revenue": -500}}]}}

Q: "Employees with margin below 30%"
{{"summary": "Employees with margin below 30%.", "visual_type": "table", "columns": ["employee", "margin"], "data": [{{"employee": "Bob", "margin": 15}}, {{"employee": "Jane", "margin": 22}}]}}

Q: "Show employees with revenue above $10,000"
{{"summary": "Employees with revenue above $10,000.", "visual_type": "table", "columns": ["employee", "revenue"], "data": [{{"employee": "John", "revenue": 25000}}, {{"employee": "Jane", "revenue": 18000}}]}}

Q: "Revenue per employee"
{{"summary": "Average revenue per employee.", "visual_type": "metric", "columns": [], "data": [{{"label": "Revenue per Employee", "value": 12500}}]}}

Q: "Compare Project Alpha vs Project Beta" or "compare projects"
{{"summary": "Comparison of ALPHA vs BETA.", "visual_type": "table", "columns": ["metric", "ALPHA", "BETA"], "data": [{{"metric": "Revenue", "ALPHA": 50000, "BETA": 45000}}, {{"metric": "Cost", "ALPHA": 38000, "BETA": 35000}}, {{"metric": "Profit", "ALPHA": 12000, "BETA": 10000}}, {{"metric": "Margin %", "ALPHA": 24, "BETA": 22}}, {{"metric": "Utilization %", "ALPHA": 85, "BETA": 78}}, {{"metric": "Employees", "ALPHA": 5, "BETA": 4}}]}}

Q: "Top 3 employees in Project Alpha"
{{"summary": "Top 3 employees in Project Alpha by profit.", "visual_type": "table", "columns": ["employee", "project", "profit"], "data": [{{"employee": "John", "project": "Alpha", "profit": 8000}}, {{"employee": "Jane", "project": "Alpha", "profit": 6000}}, {{"employee": "Bob", "project": "Alpha", "profit": 4000}}]}}

Q: "Barclays project details" or "Project X details"
{{"summary": "BARCLAYS project details: Revenue=$50,000, Cost=$30,000, Profit=$20,000, Margin=40%, Utilization=85%, Employees=5, Status=Healthy", "visual_type": "metric", "columns": [], "data": [{{"label": "Revenue", "value": 50000}}, {{"label": "Cost", "value": 30000}}, {{"label": "Profit", "value": 20000}}, {{"label": "Margin %", "value": 40}}, {{"label": "Utilization %", "value": 85}}, {{"label": "Employees", "value": 5}}, {{"label": "Status", "value": "Healthy"}}]}}

RULES:
1. "metric" for totals, counts, averages, summaries - use label/value pairs
2. "table" for lists, rankings, breakdowns with multiple columns
3. "text" only if no data found
4. Use lowercase keys: employee, project, month, revenue, cost, profit, utilization, hours, attendance
5. Numbers must be raw numbers (not strings with $ or %)
6. summary must be a specific answer, not "Overall summary"
7. Return ONLY the JSON object, nothing else
8. "last month" or "recent month" = the month marked as "LAST MONTH (most recent)" in the data
9. For "highest", "top", "best" questions: sort data DESCENDING by the relevant metric
10. For "lowest", "bottom", "worst" questions: sort data ASCENDING by the relevant metric
11. "not generating", "zero", "no revenue/profit" = filter for values that are 0 or negative
12. "below X%", "less than X", "under X" = ONLY include values < X; "above X%", "more than X", "over X" = ONLY include values > X. NEVER include values that don't meet the threshold!
13. "per employee", "per project", "per hour" = calculate average or divide total by count
14. "compare A vs B" or "compare projects" = show side-by-side comparison table with metrics as rows. Column headers MUST be the EXACT project name as listed in "AVAILABLE PROJECTS" (e.g., "BARCLAYS" not "Project Barclays"). Include all key metrics: Revenue, Cost, Profit, Margin %, Utilization %, Employees. Use only numeric values — NEVER text like "No data available".
15. "YTD", "year to date" = sum from January to the last available month
16. "Q1" = Jan-Mar, "Q2" = Apr-Jun, "Q3" = Jul-Sep, "Q4" = Oct-Dec
17. For "top N in Project X" = filter by project first, then rank
18. IMPORTANT: For "project utilization" or "which project has highest/lowest utilization", ONLY use "AvgUtilisation" from PROJECTS section. NEVER use IndividualUtilisation from EMPLOYEES section for project-level questions.
19. "which project/employee has highest/lowest X" OR "which contributes most/least to X" = return ONLY the single answer as visual_type="metric" with one label/value pair naming the winner and its value. Use RANKING HINTS directly. Do NOT return a full table of all entities — the user asked for one answer.
20. When comparing numbers, 96.6 > 96.0 > 95.0. Always compare the actual numeric values, not just the first digits.

ANTI-HALLUCINATION (CRITICAL):
21. ONLY use data explicitly provided in the context above. NEVER invent, estimate, or assume values.
22. If a project/employee name is completely absent from the data (zero records), respond as a WHOLE with: {{"summary": "No data found.", "visual_type": "text", "columns": [], "data": []}}. NEVER write "No data found" or any error text as a cell value inside a table — use 0 instead.
23. Project names are CASE-SENSITIVE and must match EXACTLY as shown in PROJECTS section (e.g., "BARCLAYS" not "Barclays", "CRYSTAL" not "Crystal").
24. If the question mentions a name similar to but not exactly matching a project/employee, clarify: "Did you mean [exact name from data]?"
25. NEVER mix data from different projects. Each project's metrics are independent.
26. For monthly breakdowns, ONLY use months explicitly listed in MonthlyBreakdown for that specific project.
27. Employee margin: "Margin(overall)" in EMPLOYEES = blended across ALL their projects. For "X's margin in Project Y", use Cost and Revenue from EMPLOYEE DETAILS rows for that project: margin = (Revenue-Cost)/Revenue*100. NEVER use per-project margin for general threshold/ranking questions.
28. For designation-based questions ("senior developers", "architects", "consultants"): filter employees by their Designation field. If Designation is empty for an employee, do NOT include them in designation-filtered results.
29. For "Q1"/"Q2"/"Q3"/"Q4" without an explicit year: use the most recent year present in the data. Never sum Q-data across multiple years unless the question explicitly requests multi-year comparison.
30. For "this year", "current year", or "YTD" questions: ONLY use months from the current calendar year as filtered in the context. Do not include months from prior years.
"""


def _clean_answer(text: str) -> str:
    answer = (text or "")
    for _ in range(3):
        answer = answer.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", " ")
    answer = answer.replace("\r\n", "\n").replace("\r", "\n")
    answer = re.sub(r"\*\*(.*?)\*\*", r"\1", answer)
    answer = re.sub(r"\*(.*?)\*", r"\1", answer)
    answer = re.sub(r"^\s*(based on the (provided )?(dataset|data)[,:]?|from the (provided )?(dataset|data)[,:]?)\s*", "", answer, flags=re.IGNORECASE)
    # Remove echoed question lines like 'User question: ...' or a repeated question at the start
    answer = re.sub(r"(?im)^\s*(user question|question)\s*:\s*.*?$", "", answer)
    answer = re.sub(r"(?im)^\s*(which|who|what|how many|how much|count|show|list)\b.*?[?:]?\s*$", "", answer)
    answer = re.sub(r"(?m)^\s*\d+[.)]\s*", "- ", answer)
    answer = re.sub(r"(?m)^\s*[•*]\s*", "- ", answer)
    answer = re.sub(r"(?m);\s+(?=[^-].*\|)", "\n- ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer)
    answer = re.sub(r"[ \t]+\n", "\n", answer)
    answer = re.sub(r"\s{2,}", " ", answer)
    return answer.strip()


def _to_float_num(s: str):
    try:
        txt = (s or "").strip()
        # Remove common decorations, keep signs and decimals
        txt = txt.replace("$", "").replace(",", "").replace("%", "").strip()
        if not txt:
            return None
        # Extract the first numeric token (handles leading text like "... (±2.0pp)")
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", txt)
        if not m:
            return None
        return float(m.group(0))
    except Exception:
        return None


def _extract_value_and_band(s: str):
    """Extract numeric value and ± band from strings like '$108,727.92 (±5.2%)'
    
    Returns: (value, band) where band is the percentage as a string like '±5.2%' or None
    """
    try:
        txt = (s or "").strip()
        # Extract the ± band if present
        band = None
        band_match = re.search(r'\(([±+-]?\d+\.?\d*%?p?p?)\)', txt)
        if band_match:
            band = band_match.group(1)
            if not band.startswith('±'):
                band = '±' + band
        
        # Extract the main numeric value
        value = _to_float_num(txt)
        return value, band
    except Exception:
        return None, None


def _extract_rows(text: str):
    rows = []
    if not text:
        return rows
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("- "):
            line = line[2:].strip()
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        row = {}
        for idx, part in enumerate(parts):
            if ":" in part:
                key, val = part.split(":", 1)
                k = key.strip().lower()
                v = val.strip()
                if k == "month":
                    row["month"] = v
                elif k.replace(" ", "") in ("monthno", "month_no"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["month_no"] = int(round(fv))
                elif k == "quarter":
                    row["quarter"] = v
                elif k.replace(" ", "") in ("quarterno", "quarter_no", "qno"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["quarter_no"] = int(round(fv))
                elif k == "project":
                    row["project"] = v
                elif k == "revenue":
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["revenue"] = fv
                        if band is not None:
                            row["revenue_variance"] = band
                elif k == "cost":
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["cost"] = fv
                elif k == "profit":
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["profit"] = fv
                        if band is not None:
                            row["profit_variance"] = band
                elif k == "headcount":
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["headcount"] = int(round(fv))
                        if band is not None:
                            row["headcount_variance"] = band
                elif k in ("utilization", "utilisation"):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["utilization_pct"] = fv
                        if band is not None:
                            row["utilization_variance"] = band
                elif k in ("hours", "hrs"):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["hours"] = fv
                        if band is not None:
                            row["hours_variance"] = band
                elif k in ("leaves", "leave", "pto", "time off", "timeoff"):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["leave_days"] = fv
                        if band is not None:
                            row["leave_days_variance"] = band
                elif k in ("vacation", "vacations"):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["vacation_days"] = fv
                        if band is not None:
                            row["vacation_days_variance"] = band
                elif k in ("holidays", "holiday"):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["holiday_days"] = fv
                        if band is not None:
                            row["holiday_days_variance"] = band
                elif k.replace(" ", "") in ("workingdays", "workdays", "workdayscount"):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["working_days"] = fv
                        if band is not None:
                            row["working_days_variance"] = band
                elif k.startswith("margin") or "margin" in k.replace(" ", ""):
                    fv, band = _extract_value_and_band(v)
                    if fv is not None:
                        row["margin_pct"] = fv
                        if band is not None:
                            row["margin_variance"] = band
                elif k.replace(" ", "") in ("billrate", "billingrate"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["billing_rate"] = fv
                elif k.replace(" ", "") in ("costrate", "costperhour"):
                    fv = _to_float_num(v)
                    if fv is not None:
                        row["cost_rate"] = fv
                else:
                    row[k] = v
            else:
                if idx == 0 and "employee" not in row:
                    row["employee"] = part
        if any(k in row for k in ("employee", "month", "project", "revenue", "cost", "profit", "utilization_pct", "hours", "margin_pct")):
            rows.append(row)
    return rows


def _format_column_name(col: str) -> str:
    """Format column name: remove special chars, capitalize properly."""
    if not col:
        return col
    
    # Column name mappings for common fields
    col_map = {
        "employee": "Employee",
        "project": "Project",
        "month": "Month",
        "quarter": "Quarter",
        "revenue": "Revenue",
        "cost": "Cost",
        "profit": "Profit",
        "headcount": "Headcount",
        "utilization_pct": "Utilization",
        "utilisation_pct": "Utilization",
        "utilization": "Utilization",
        "utilisation": "Utilization",
        "hours": "Hours",
        "total_hours": "Hours",
        "leave_days": "Leaves",
        "vacation_days": "Vacation",
        "holiday_days": "Holidays",
        "working_days": "Working Days",
        "margin_pct": "Margin",
        "margin": "Margin",
        "gross_margin_pct": "Gross Margin",
        "avg_margin_pct": "Avg Margin",
        "gross_margin": "Gross Margin",
        "avg_margin": "Avg Margin",
        "billing_rate": "Bill Rate",
        "cost_rate": "Cost Rate",
        "actual_hours": "Actual Hours",
        "billable_hours": "Billable Hours",
    }
    
    col_lower = col.lower().strip()
    if col_lower in col_map:
        return col_map[col_lower]
    
    # Generic cleanup: replace underscores/special chars with spaces, title case
    cleaned = re.sub(r"[_\-]+", " ", col)
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", cleaned)
    cleaned = cleaned.strip()
    
    # Title case each word
    if cleaned:
        return " ".join(word.capitalize() for word in cleaned.split())
    
    return col


def _format_columns(columns: list) -> list:
    """Format all column names in a list."""
    return [_format_column_name(c) for c in (columns or [])]


def _format_value(key: str, value) -> str:
    """Format a value based on its column type (add $, %, etc.)."""
    if value is None:
        return ""
    
    key_lower = key.lower().replace(" ", "_")
    
    # Currency fields - also match month-based columns like "February 2026 Revenue" or just "February 2026" (when it's revenue data)
    currency_keywords = ("revenue", "cost", "profit", "billing_rate", "bill_rate", "cost_rate")
    is_currency = key_lower in currency_keywords or any(kw in key_lower for kw in currency_keywords)
    
    # Also treat month columns as currency if value looks like a large number (likely revenue/cost)
    # But exclude if key contains non-currency keywords like "hours", "days", "count"
    if not is_currency and isinstance(value, (int, float)) and value > 1000:
        non_currency_keywords = ["hours", "hour", "days", "day", "count", "headcount", "employees", "utilization", "utilisation", "margin", "attendance"]
        if not any(nc in key_lower for nc in non_currency_keywords):
            # Check if key looks like a month (e.g., "February 2026", "March 2026")
            month_patterns = ["january", "february", "march", "april", "may", "june", 
                             "july", "august", "september", "october", "november", "december"]
            if any(m in key_lower for m in month_patterns):
                is_currency = True
    
    if is_currency:
        try:
            num = float(value) if not isinstance(value, (int, float)) else value
            return f"${num:,.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Percentage fields
    if key_lower in (
        "utilization", "utilization_pct", "utilisation", "utilisation_pct",
        "margin", "margin_pct", "gross_margin", "gross_margin_pct", "avg_margin", "avg_margin_pct",
    ) or ("margin" in key_lower) or key_lower.endswith("_pct"):
        try:
            if isinstance(value, (int, float)):
                num = float(value)
            else:
                parsed = _to_float_num(str(value))
                if parsed is None:
                    return str(value)
                num = parsed
            return f"{num:.1f}%"
        except (ValueError, TypeError):
            return str(value)
    
    # Integer fields
    if key_lower in ("headcount", "hours", "total_hours", "leaves", "leave_days", "vacation", "vacation_days", 
                     "holidays", "holiday_days", "working_days", "workdays", "actual_hours", "billable_hours"):
        try:
            num = float(value) if not isinstance(value, (int, float)) else value
            return f"{int(round(num)):,}"
        except (ValueError, TypeError):
            return str(value)
    
    # Default: return as-is
    return value


def _pivot_table_if_needed(data: list, columns: list) -> tuple:
    """
    Pivot table if it has repeating entity+month combinations.
    
    Handles various column counts:
    - 2 cols: Month, Revenue -> Single row with months as columns
    - 3 cols: Month, Employee, Leaves -> Employee, May 2026, June 2026
    - 4+ cols: Month, Employee, Leaves, Vacation -> Employee, May 2026 Leaves, May 2026 Vacation, ...
    
    Returns: (pivoted_data, pivoted_columns) or (data, columns) if no pivot needed.
    """
    if not data or len(data) < 2 or len(columns) < 2:
        return data, columns
    
    cols_lower = [c.lower() for c in columns]
    
    # Detect month and entity columns
    month_col = None
    entity_col = None
    metric_cols = []
    
    for i, c in enumerate(cols_lower):
        if c in ("month", "quarter") and month_col is None:
            month_col = columns[i]
        elif c in ("employee", "project") and entity_col is None:
            entity_col = columns[i]
        else:
            metric_cols.append(columns[i])
    
    # Must have month column and at least one metric
    if not month_col or not metric_cols:
        return data, columns
    
    # Collect unique months
    months_seen = []
    for row in data:
        m = row.get(month_col)
        if m and m not in months_seen:
            months_seen.append(m)
    
    if len(months_seen) < 2:
        return data, columns
    
    # Case 1: 2-column table (Month + Metric) - no entity column
    # Convert vertical to horizontal: Month | Revenue -> Jan 2026 | Feb 2026 | Mar 2026
    if len(columns) == 2 and not entity_col:
        metric_col = metric_cols[0]
        pivoted_columns = months_seen.copy()
        new_row = {}
        for row in data:
            m = row.get(month_col)
            v = row.get(metric_col)
            if m and v is not None:
                new_row[m] = v
        # Only pivot if we have values for all months
        if len(new_row) == len(months_seen):
            return [new_row], pivoted_columns
        else:
            # Not all months have values, keep original format
            return data, columns
    
    # Case 2: 3+ column table with entity - need entity column
    if not entity_col:
        return data, columns
    
    # Check if there are repeating entities (same entity appears multiple times)
    entities = [row.get(entity_col) for row in data]
    if len(entities) == len(set(entities)):
        # No repetition, no need to pivot
        return data, columns
    
    # Collect unique entities (preserve order)
    entities_seen = []
    for row in data:
        e = row.get(entity_col)
        if e and e not in entities_seen:
            entities_seen.append(e)
    
    # Build pivoted data: entity as row, months+metrics as columns
    pivot_map = {}  # {entity: {month: {metric: value}}}
    for row in data:
        e = row.get(entity_col)
        m = row.get(month_col)
        if e not in pivot_map:
            pivot_map[e] = {}
        if m not in pivot_map[e]:
            pivot_map[e][m] = {}
        for metric in metric_cols:
            pivot_map[e][m][metric] = row.get(metric, "")
    
    # Create pivoted rows and columns
    if len(metric_cols) == 1:
        # Single metric: columns are just months
        # Employee | May 2026 | June 2026
        pivoted_columns = [entity_col] + months_seen
        pivoted_data = []
        for entity in entities_seen:
            new_row = {entity_col: entity}
            for month in months_seen:
                metric = metric_cols[0]
                new_row[month] = pivot_map.get(entity, {}).get(month, {}).get(metric, "")
            pivoted_data.append(new_row)
    else:
        # Multiple metrics: columns are month+metric combinations
        # Employee | May 2026 Leaves | May 2026 Vacation | June 2026 Leaves | June 2026 Vacation
        pivoted_columns = [entity_col]
        for month in months_seen:
            for metric in metric_cols:
                col_name = f"{month} {_format_column_name(metric)}"
                pivoted_columns.append(col_name)
        
        pivoted_data = []
        for entity in entities_seen:
            new_row = {entity_col: entity}
            for month in months_seen:
                for metric in metric_cols:
                    col_name = f"{month} {_format_column_name(metric)}"
                    new_row[col_name] = pivot_map.get(entity, {}).get(month, {}).get(metric, "")
            pivoted_data.append(new_row)
    
    return pivoted_data, pivoted_columns


def _format_data_keys(data: list, columns: list) -> list:
    """Rename keys in data rows to match formatted column names and format values."""
    if not data or not columns:
        return data
    
    # Build mapping from original to formatted
    key_map = {}
    for col in columns:
        formatted = _format_column_name(col)
        if formatted != col:
            key_map[col] = formatted
    
    # Rename keys and format values in each row
    formatted_data = []
    for row in data:
        new_row = {}
        for k, v in row.items():
            new_key = key_map.get(k, k)
            # Format the value based on the original key type
            formatted_value = _format_value(k, v)
            new_row[new_key] = formatted_value
        formatted_data.append(new_row)
    
    return formatted_data


def _forecast_table_name(header: str) -> str:
    """Convert forecast headers to clean table names like 'Q3 2026 Forecast' or 'May-Jul 2026 Forecast'."""
    if not header:
        return "Forecast"
    
    # Extract the period from headers like "Estimate — Q3 2026 —" or "Estimate — May 2026 to July 2026 —"
    m = re.search(r"(?i)estimate\s+—\s*(.*?)\s*—", header)
    if m:
        period = m.group(1).strip()
        # Clean up period: "May 2026 to July 2026" -> "May-Jul 2026"
        range_match = re.match(r"(\w+)\s+(\d{4})\s+to\s+(\w+)\s+(\d{4})", period)
        if range_match:
            m1, y1, m2, y2 = range_match.groups()
            if y1 == y2:
                period = f"{m1[:3]}-{m2[:3]} {y1}"
            else:
                period = f"{m1[:3]} {y1}-{m2[:3]} {y2}"
        return f"{period} Forecast"
    
    # Try to extract quarter or month pattern
    q_match = re.search(r"(Q[1-4]\s*\d{4})", header, re.IGNORECASE)
    if q_match:
        return f"{q_match.group(1)} Forecast"
    
    m_match = re.search(r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})", header, re.IGNORECASE)
    if m_match:
        return f"{m_match.group(1)} Forecast"
    
    # Fallback: remove "Estimate" prefix
    header2 = re.sub(r"(?i)^estimate\s*[-—:]\s*", "", header).strip()
    header2 = re.sub(r"\s*[-—]\s*$", "", header2).strip()
    return f"{header2} Forecast" if header2 else "Forecast"


def _concise_table_name_from_summary(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    if not s:
        return "Results"
    # Limit length for a compact table title
    if len(s) > 80:
        s = s[:80].rstrip()
    if not re.match(r"(?i)^(forecast|results)\s+—\s*", s):
        s = f"Results — {s}"
    return s


def _llm_table_name(summary: str, columns, data) -> str:
    """Generate a clean, relevant table name from columns and summary."""
    cols = [str(c or "").strip() for c in (columns or [])]
    cols_l = [c.lower() for c in cols]
    s = (summary or "").strip()
    
    # Detect entity dimension
    entity_map = {
        "employee": "Employees",
        "project": "Projects",
        "month": "Monthly",
        "quarter": "Quarterly",
        "department": "Departments",
        "client": "Clients",
        "team": "Teams",
    }
    entity = None
    entity_key = None
    for k, v in entity_map.items():
        if k in cols_l:
            entity = v
            entity_key = k
            break
    
    # Detect metrics (can have multiple)
    metric_aliases = [
        ("utilization", "Utilization"), ("utilisation", "Utilization"), ("utilization_pct", "Utilization"),
        ("profit", "Profit"), ("revenue", "Revenue"), ("cost", "Cost"),
        ("hours", "Hours"), ("hrs", "Hours"), ("total_hours", "Hours"),
        ("leave", "Leaves"), ("leaves", "Leaves"), ("leave_days", "Leaves"), ("pto", "Leaves"),
        ("vacation", "Vacation"), ("vacation_days", "Vacation"),
        ("holidays", "Holidays"), ("holiday", "Holidays"), ("holiday_days", "Holidays"),
        ("working_days", "Working Days"), ("workdays", "Working Days"),
        ("margin", "Margin"), ("margin_pct", "Margin"),
        ("gross_margin", "Gross Margin"), ("gross_margin_pct", "Gross Margin"),
        ("avg_margin", "Avg Margin"), ("avg_margin_pct", "Avg Margin"),
        ("headcount", "Headcount"),
    ]
    metrics = []
    for key, label in metric_aliases:
        if key in cols_l and label not in metrics:
            metrics.append(label)
    
    # Ranking detection from summary
    rank_match = re.search(r"(?i)\b(top|highest|bottom|lowest)\b\s*(\d+)?", s)
    rank_word = None
    n = None
    if rank_match:
        w = rank_match.group(1).lower()
        rank_word = "Top" if w in ("top", "highest") else "Bottom"
        if rank_match.group(2):
            try:
                n = int(rank_match.group(2))
            except Exception:
                pass
    
    # Infer count from data if not in summary
    if n is None and isinstance(data, list) and 1 <= len(data) <= 50:
        if rank_word:
            n = len(data)
    
    # Detect comparison/trend keywords
    is_comparison = bool(re.search(r"(?i)\b(compar|vs|versus|between|difference)\b", s))
    is_breakdown = bool(re.search(r"(?i)\b(breakdown|by month|by project|by employee|distribution)\b", s))
    is_summary = bool(re.search(r"(?i)\b(summary|overview|total|all)\b", s))
    
    # Build clean title
    metric_str = " & ".join(metrics[:2]) if metrics else None  # Limit to 2 metrics
    
    # Ranked results: "Top 5 Employees by Profit"
    if rank_word and n:
        if entity and metric_str:
            return f"{rank_word} {n} {entity} by {metric_str}"
        elif entity:
            return f"{rank_word} {n} {entity}"
        elif metric_str:
            return f"{rank_word} {n} by {metric_str}"
    
    # Breakdown: "Revenue by Project" or "Monthly Profit"
    if is_breakdown or (entity and metric_str):
        if entity_key in ("month", "quarter"):
            return f"{entity} {metric_str}" if metric_str else f"{entity} Data"
        elif entity and metric_str:
            return f"{metric_str} by {entity}"
        elif entity:
            return f"{entity} Data"
    
    # Comparison
    if is_comparison:
        if entity and metric_str:
            return f"{entity} {metric_str} Comparison"
        elif metric_str:
            return f"{metric_str} Comparison"
    
    # Summary/Overview
    if is_summary:
        if metric_str:
            return f"{metric_str} Summary"
        elif entity:
            return f"{entity} Summary"
    
    # Generic fallback based on columns
    if entity and metric_str:
        return f"{entity} {metric_str}"
    elif entity:
        return f"{entity} Data"
    elif metric_str:
        return metric_str
    
    # Last resort: clean up the summary
    if s:
        # Remove common prefixes
        s = re.sub(r"(?i)^(results?\s*[-—:]?\s*|the\s+|here\s+(is|are)\s+(the\s+)?)", "", s)
        s = re.sub(r"(?i)\s+(are|is)\s+listed\s+below\.?$", "", s)
        s = re.sub(r"(?i)\s+based\s+on\s+.*$", "", s)
        s = s.strip(" .,:-—")
        if s and len(s) <= 60:
            return s.title() if len(s) < 30 else s[:60]
    
    return "Data"

# ── Main Q&A function ───────────────────────────────────────────────────────

def ask(question: str, time_range: str = None) -> dict:
    """
    Answer a natural-language question about the dataset.

    Returns: {"summary": str, "visual_type": str, "columns": list, "data": list, "sources": dict}
    """
    _start_time = time.time()
    
    records = GLOBAL_DATASET
    if not records:
        return {
            "summary": "No data loaded. Please upload timesheets first via POST /ingest.",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    # Apply time range filter if specified
    if time_range:
        records = filter_by_range(records, time_range)

    # ── Check response cache ──
    data_hash = _compute_dataset_hash(records)
    cached_response = _get_cached_response(question, data_hash)
    if cached_response:
        _log(f"Total time: {time.time() - _start_time:.2f}s (cached)")
        return cached_response

    # Forecast intent: use AI forecaster
    # Note: try_answer_forecast now uses fast keyword check internally
    _fc_start = time.time()
    _forecast_module = "forecast (OOP)" if _USING_NEW_FORECAST else "forecast_ (legacy)"
    fc_answer = try_answer_forecast(question, records)
    _log(f"Forecast check took {time.time() - _fc_start:.2f}s (module={_forecast_module}, likely_forecast={is_likely_forecast(question)})", level="debug")
    if fc_answer is not None:
        rows = _extract_rows(fc_answer)
        if rows:
            preferred_cols = [
                "month", "quarter",
                "project", "employee",
                "revenue", "revenue_variance", "cost", "profit", "profit_variance",
                "margin_pct", "margin_variance",
                "headcount", "headcount_variance",
                "utilization_pct", "utilization_variance",
                "hours", "hours_variance",
                "leave_days", "leave_days_variance",
                "vacation_days", "vacation_days_variance",
                "holiday_days", "holiday_days_variance",
                "working_days", "working_days_variance",
            ]
            present = []
            for c in preferred_cols:
                if any(c in r for r in rows):
                    present.append(c)
            data = [{k: r.get(k) for k in present} for r in rows]
            header = fc_answer.splitlines()[0].strip()
            table_name = _forecast_table_name(header)
            
            # Pivot table if it has repeating entity+month combinations
            data, present = _pivot_table_if_needed(data, present)
            
            # Format column names and data keys
            formatted_cols = _format_columns(present)
            formatted_data = _format_data_keys(data, present)
            return {
                "summary": table_name,
                "visual_type": "table",
                "columns": formatted_cols,
                "data": formatted_data,
                "sources": {
                    "total_records": len(records),
                    "months": get_months_available(records),
                    "time_range": time_range or "ALL",
                    "forecast": True,
                },
            }
        else:
            return {
                "summary": fc_answer,
                "visual_type": "text",
                "columns": [],
                "data": [],
                "sources": {
                    "total_records": len(records),
                    "months": get_months_available(records),
                    "time_range": time_range or "ALL",
                    "forecast": True,
                },
            }

    # Check configured provider availability
    if not is_llm_available():
        return {
            "summary": "No configured LLM provider is available. Set AI_PROVIDER and related env vars (Ollama/OpenAI).",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    # Early-return if scope detects entities not in the dataset (prevents hallucination)
    _scope_check = _detect_question_scope(question, records)
    _missing = _scope_check.get("missing_entities", [])
    if _missing:
        _msg = "No data found."
        _elapsed = time.time() - _start_time
        _log(f"Total time: {_elapsed:.2f}s (validation_error)")
        return {
            "summary": _msg,
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {
                "total_records": len(records),
                "months": get_months_available(records),
                "time_range": time_range or "ALL",
                "forecast": False,
                "validation_error": True,
                "missing_entities": _missing,
            },
        }

    # Build targeted context based on question (smart context selection)
    _ctx_start = time.time()
    context = _get_cached_context(records, question=question)
    _log(f"Context build took {time.time() - _ctx_start:.2f}s")
    
    # Debug: log context details
    context_lines = context.split('\n')
    context_sections = [l for l in context_lines if l.startswith('===')]
    has_filter = any('FILTERED TO PROJECT' in l or 'IMPORTANT:' in l for l in context_lines)
    _log(f"Context sections: {context_sections}", "debug")
    _log(f"Context length: {len(context)} chars, {len(context_lines)} lines", "debug")
    _log(f"Project filter active: {has_filter}", "debug")
    if DEBUG:
        # Log full context in debug mode
        _log(f"--- FULL CONTEXT START ---\n{context}\n--- FULL CONTEXT END ---", "debug")
    
    prompt = _QA_PROMPT.format(context=context, question=question)

    try:
        _llm_start = time.time()
        raw_response = _llm_generate(prompt, timeout=120)
        print(f"[qa_engine] LLM call took {time.time() - _llm_start:.2f}s")
    except Exception as e:
        return {
            "summary": f"LLM query failed: {str(e)}",
            "visual_type": "text",
            "columns": [],
            "data": [],
            "sources": {},
        }

    _log(f"Total time: {time.time() - _start_time:.2f}s")
    
    # Detect echo/hallucination: LLM just repeating the question back
    raw_lower = (raw_response or "").lower().strip()
    q_lower = (question or "").lower().strip()
    
    # Pre-compute scope once for fallback use (lightweight - just string matching)
    _fallback_scope = None
    _fallback_project_data = None
    
    def _get_fallback_data():
        """Lazy-load fallback data only when needed (avoids iterating 1000s of records unless necessary)."""
        nonlocal _fallback_scope, _fallback_project_data
        if _fallback_scope is None:
            _fallback_scope = _detect_question_scope(question, records)
            if _fallback_scope.get("specific_project"):
                proj_name = _fallback_scope["specific_project"]
                # Filter records for this project FIRST, then aggregate (much faster)
                proj_records = [r for r in records if r.get("project") == proj_name]
                if proj_records:
                    # Simple aggregation for single project
                    _fallback_project_data = {
                        "name": proj_name,
                        "revenue": sum(float(r.get("revenue") or 0) for r in proj_records),
                        "cost": sum(float(r.get("cost") or 0) for r in proj_records),
                        "profit": sum(float(r.get("profit") or 0) for r in proj_records),
                        "employees": len(set(r.get("employee") for r in proj_records if r.get("employee"))),
                    }
        return _fallback_scope, _fallback_project_data
    
    # Check if response is just echoing the question (hallucination indicator)
    # Only check for echo if response is NOT valid JSON (valid JSON means LLM processed it)
    is_likely_json = raw_lower.strip().startswith('{') or '{"summary"' in raw_lower
    if raw_lower and q_lower and not is_likely_json:
        # Remove common prefixes/suffixes for comparison
        raw_clean = re.sub(r'^(project\s+|details\s+(of|for)\s+|show\s+|get\s+)', '', raw_lower)
        q_clean = re.sub(r'^(project\s+|details\s+(of|for)\s+|show\s+|get\s+)', '', q_lower)
        # If response is very similar to question (echo), it's likely hallucination
        # Must be high similarity (>80% match) to avoid false positives
        if raw_clean.replace(' ', '') == q_clean.replace(' ', ''):
            _log(f"WARNING: LLM echoed question back - likely hallucination", "debug")
            # Return structured response with actual data (lazy-loaded)
            scope, proj_data = _get_fallback_data()
            if proj_data:
                return {
                    "summary": f"{proj_data['name']}: Revenue=${proj_data['revenue']:,.2f}, Cost=${proj_data['cost']:,.2f}, Profit=${proj_data['profit']:,.2f}, Employees={proj_data['employees']}",
                    "visual_type": "metric",
                    "columns": [],
                    "data": [
                        {"label": "Revenue", "value": proj_data['revenue']},
                        {"label": "Cost", "value": proj_data['cost']},
                        {"label": "Profit", "value": proj_data['profit']},
                        {"label": "Employees", "value": proj_data['employees']},
                    ],
                    "sources": {"total_records": len(records), "fallback": "echo_detection"},
                }
    
    # Parse the JSON from the LLM
    parsed_json = _extract_qa_json(raw_response)
    
    # Validate and fix LLM response structure
    if not parsed_json or not isinstance(parsed_json, dict):
        _log(f"LLM returned non-JSON response, using fallback", "debug")
        parsed_json = {
            "summary": _clean_answer(raw_response),
            "visual_type": "text",
            "data": [],
            "columns": []
        }
    
    # Check for hallucination: summary is too short or generic
    # Only trigger if summary looks like echoed question (not a valid short answer)
    summary = parsed_json.get("summary", "")
    summary_lower = summary.lower().strip()
    looks_like_echo = summary_lower and (
        summary_lower in q_lower or 
        q_lower.rstrip('?').rstrip('.') in summary_lower or
        summary_lower.endswith('details') or
        summary_lower.endswith('details.')
    )
    if summary and len(summary) < 30 and not parsed_json.get("data") and looks_like_echo:
        # Very short summary that looks like echoed question - likely hallucination
        _log(f"WARNING: Short echo-like summary with no data - possible hallucination: '{summary}'", "debug")
        # Try to provide actual data (reuse lazy-loaded fallback)
        scope, proj_data = _get_fallback_data()
        if proj_data:
            parsed_json = {
                "summary": f"{proj_data['name']}: Revenue=${proj_data['revenue']:,.2f}, Cost=${proj_data['cost']:,.2f}, Profit=${proj_data['profit']:,.2f}, Employees={proj_data['employees']}",
                "visual_type": "metric",
                "columns": [],
                "data": [
                    {"label": "Revenue", "value": proj_data['revenue']},
                    {"label": "Cost", "value": proj_data['cost']},
                    {"label": "Profit", "value": proj_data['profit']},
                    {"label": "Employees", "value": proj_data['employees']},
                ],
            }
    
    # Ensure required fields exist with correct types
    if "summary" not in parsed_json or not isinstance(parsed_json.get("summary"), str):
        parsed_json["summary"] = _clean_answer(raw_response) if raw_response else "Unable to process response."
    
    if "visual_type" not in parsed_json or parsed_json.get("visual_type") not in ("table", "metric", "text", "bar_chart"):
        # Infer visual_type from data structure
        data = parsed_json.get("data", [])
        if isinstance(data, list) and len(data) == 1 and "label" in data[0] and "value" in data[0]:
            parsed_json["visual_type"] = "metric"
        elif isinstance(data, list) and len(data) > 1:
            parsed_json["visual_type"] = "table"
        else:
            parsed_json["visual_type"] = "text"
    
    if "columns" not in parsed_json or not isinstance(parsed_json.get("columns"), list):
        parsed_json["columns"] = []
    
    if "data" not in parsed_json or not isinstance(parsed_json.get("data"), list):
        parsed_json["data"] = []
    
    # For table type, ensure columns match data keys
    visual_type = parsed_json.get("visual_type", "text")
    raw_data = parsed_json.get("data", [])
    raw_columns = parsed_json.get("columns", [])

    # Scrub LLM placeholder strings from cell values (catches "No data found for X.", "N/A", etc.)
    _NO_DATA_RE = re.compile(r"no\s+data\s+(found|available)(?: for [^\"]+)?\.*", re.IGNORECASE)
    if raw_data and isinstance(raw_data, list):
        for row in raw_data:
            if isinstance(row, dict):
                for k, v in list(row.items()):
                    if isinstance(v, str) and (_NO_DATA_RE.search(v.strip()) or v.strip().lower() in {"n/a", "na", "not available"}):
                        row[k] = None
    
    if visual_type == "table" and raw_data and not raw_columns:
        # Extract columns from first data row
        if isinstance(raw_data[0], dict):
            raw_columns = list(raw_data[0].keys())
            parsed_json["columns"] = raw_columns
    
    # Convert "metric/value" table to proper metric format
    if visual_type == "table" and raw_data:
        cols_lower = [c.lower() for c in raw_columns]
        if cols_lower == ["metric", "value"] or cols_lower == ["label", "value"]:
            # This is actually a metric, not a table
            visual_type = "metric"
            parsed_json["visual_type"] = "metric"
            # Convert to label/value format
            new_data = []
            for row in raw_data:
                label = row.get("metric") or row.get("label") or row.get("Metric") or row.get("Label") or ""
                value = row.get("value") or row.get("Value") or ""
                # Skip if value is a list (invalid)
                if isinstance(value, list):
                    value = ", ".join(str(v) for v in value)
                new_data.append({"label": str(label), "value": value})
            raw_data = new_data
            raw_columns = []

    summary_text = parsed_json.get("summary", "")
    
    if visual_type == "table":
        summary_text = _llm_table_name(summary_text, raw_columns, raw_data)
        
        # Remove duplicate rows (LLM sometimes repeats data)
        if raw_data:
            seen = set()
            unique_data = []
            for row in raw_data:
                # Create a hashable key from row values
                row_key = tuple(sorted((k, str(v)) for k, v in row.items()))
                if row_key not in seen:
                    seen.add(row_key)
                    unique_data.append(row)
            raw_data = unique_data
        
        # Pivot table if it has repeating entity+month combinations
        raw_data, raw_columns = _pivot_table_if_needed(raw_data, raw_columns)
        
        # Format column names and data keys
        formatted_cols = _format_columns(raw_columns)
        formatted_data = _format_data_keys(raw_data, raw_columns)
    elif visual_type == "metric":
        # Format metric value if it's a number
        formatted_cols = raw_columns
        formatted_data = []
        for item in raw_data:
            if isinstance(item, dict) and "label" in item and "value" in item:
                label = item["label"]
                value = item["value"]
                # Try to format numeric values
                if isinstance(value, (int, float)):
                    label_lower = label.lower()
                    if any(w in label_lower for w in ["revenue", "cost", "profit", "salary", "budget", "amount"]):
                        value = f"${value:,.2f}"
                    elif any(w in label_lower for w in ["percent", "rate", "utilization", "margin"]):
                        value = f"{value:.1f}%"
                    elif any(w in label_lower for w in ["count", "number", "total", "employee", "headcount"]):
                        value = f"{int(value):,}"
                    else:
                        value = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
                formatted_data.append({"label": label, "value": str(value)})
            else:
                formatted_data.append(item)
    else:
        formatted_cols = raw_columns
        formatted_data = raw_data
    
    # Construct the final, UI-friendly payload
    result = {
        "summary": summary_text,
        "visual_type": visual_type,
        "columns": formatted_cols,
        "data": formatted_data,
        "sources": {
            "total_records": len(records),
            "months": get_months_available(records),
            "time_range": time_range or "ALL",
            "forecast": False,
        }
    }
    
    # ── Cache the response ──
    _cache_response(question, data_hash, result)
    
    return result