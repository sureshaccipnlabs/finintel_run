import math
import re
import datetime as dt
from typing import List, Dict, Optional, Tuple

from .dataset import (
    build_monthly,
    get_months_available,
    _parse_month_date,
    GLOBAL_DATASET,
)
from .ai_mapper import _ollama_generate, is_ollama_available, _extract_json


def _fmt_money(v: float) -> str:
    return f"${v:,.2f}"


def _fmt_int(v: float) -> str:
    try:
        return str(int(round(v)))
    except Exception:
        return "0"


def _fmt_pct(v: float) -> str:
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    return f"{v:.1f}%"


def _clamp_nonneg(v: float) -> float:
    try:
        return max(0.0, float(v))
    except Exception:
        return 0.0


def _clamp_pct(v: float) -> float:
    if isinstance(v, float) and math.isnan(v):
        return v
    try:
        vv = float(v)
    except Exception:
        return float("nan")
    return min(100.0, max(0.0, vv))


def _add_months(d: dt.date, n: int) -> dt.date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return dt.date(y, m, 1)


def _quarter_label_for(d: dt.date) -> str:
    qn = (d.month - 1) // 3 + 1
    return f"Q{qn} {d.year}"


def _month_label_for(d: dt.date) -> str:
    return d.strftime("%B %Y")


def _build_overall_series(records: List[dict]) -> Dict[str, List[float]]:
    return _series_from_monthly_overall(records)


def _last_n_snapshot(months: List[str], series: Dict[str, List[float]], n: int = 6) -> Dict[str, List[float]]:
    out = {}
    take = min(n, len(months))
    idx0 = len(months) - take
    for k, arr in series.items():
        if k == "months":
            out[k] = months[idx0:]
        else:
            out[k] = arr[idx0:]
    return out


def _extract_requested_employees(question: str) -> List[str]:
    if not question:
        return []
    out: List[str] = []
    # Capture phrases like 'employee John Doe', 'employee: Jane', 'for employee Mark'
    for m in re.finditer(r"(?i)\bemployee\s*[:\-]?\s*([a-z0-9 .&/_-]{1,60})", question):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        stop_words = [
            " for ", " in ", " by ", " with ", " on ", " during ", " next ",
            " revenue", " cost", " profit", " utilisation", " utilization",
            " headcount", " leaves", " vacation", " holidays", " hours", " working",
            ",", "|", "\n"
        ]
        low = " " + raw.lower() + " "
        cuts = [low.find(tok) for tok in stop_words if low.find(tok) >= 0]
        if cuts:
            cut = min(cuts)
            cleaned = low[1:cut].strip(" .,:;|-")
        else:
            cleaned = raw.strip(" .,:;|-")
        if cleaned:
            out.append(cleaned)
    return out


def _extract_freeform_targets(question: str) -> List[str]:
    if not question:
        return []
    out: List[str] = []
    # Phrases like 'for Maersk', 'for John Doe'
    for m in re.finditer(r"(?i)\bfor\s+([a-z0-9 .&/_-]{1,60})", question):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        stop_words = [
            " for ", " in ", " by ", " with ", " on ", " during ", " next ",
            ",", "|", "\n"
        ]
        low = " " + raw.lower() + " "
        cuts = [low.find(tok) for tok in stop_words if low.find(tok) >= 0]
        if cuts:
            cut = min(cuts)
            cleaned = low[1:cut].strip(" .,:;|-")
        else:
            cleaned = raw.strip(" .,:;|-")
        # Filter out obvious time/metric words
        discard = False
        kws = [
            "next", "month", "months", "quarter", "quarters", "q1", "q2", "q3", "q4",
            "revenue", "profit", "cost", "utilisation", "utilization", "hours", "leaves", "vacation", "holidays",
        ]
        # Month names
        kws += [
            "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
            "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
            "oct", "october", "nov", "november", "dec", "december"
        ]
        if cleaned and len(cleaned) >= 2:
            for kw in kws:
                if kw in cleaned.lower():
                    discard = True
                    break
            if not discard:
                out.append(cleaned)
    return out

def _llm_parse_question(question: str, known_projects: List[str], known_employees: List[str]) -> Optional[dict]:
    """
    Use LLM to parse a forecast question into structured parameters.
    Returns dict with: is_forecast, metrics, scope, targets, horizon_type, horizon_value, explicit_periods
    """
    if not is_ollama_available():
        return None
    prompt = f"""You are a financial data assistant. Analyze this question and extract structured parameters for forecasting.

Question: "{question}"

Known projects in dataset: {', '.join(known_projects[:20]) if known_projects else 'None'}
Known employees in dataset: {', '.join(known_employees[:20]) if known_employees else 'None'}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "is_forecast": true or false,
  "metrics": ["revenue", "profit", "leaves", "hours", "utilization", "cost", "headcount", "vacation", "holidays", "working_days"],
  "scope": "overall" or "project" or "employee",
  "targets": ["Project or Employee Name", ...],
  "horizon_type": "months" or "quarters" or "explicit_months" or "explicit_quarter",
  "horizon_value": 3,
  "explicit_periods": ["July 2026", "Q3 2026", ...]
}}

Rules:
1. is_forecast=true if the question asks about future predictions, forecasts, estimates, projections, or expected values.
2. metrics: list only the metrics mentioned or implied. Use lowercase keys.
3. scope: "project" if asking about specific project(s), "employee" if about specific employee(s), else "overall".
4. targets: list the project or employee names mentioned. Match to known names if possible.
5. horizon_type: "months" for "next N months", "quarters" for "next N quarters", "explicit_months" for specific months like "July 2026", "explicit_quarter" for "Q3 2026".
6. horizon_value: the number of months or quarters requested (e.g., 3 for "next 3 months").
7. explicit_periods: list specific periods mentioned (e.g., ["July 2026", "August 2026"] or ["Q3 2026"]).
8. If no metrics specified, return empty list for metrics.
9. If not a forecast question, set is_forecast=false and leave other fields empty/default.
"""
    try:
        raw = _ollama_generate(prompt, timeout=30)
        parsed = _extract_json(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception as exc:
        print(f"[forecast_] LLM question parsing failed: {exc}")
    return None


def _llm_advise_settings(question: str, months: List[str], series: Dict[str, List[float]]) -> Optional[dict]:
    if not is_ollama_available():
        return None
    snap = _last_n_snapshot(months, series, n=6)
    lines = []
    lines.append("You are a forecasting assistant. Based on the last months, advise the method and adjustments.")
    lines.append("Return ONLY JSON with keys: method, params, damping, adjustments.")
    lines.append("- method: { metric: 'lin'|'holt'|'sma' }")
    lines.append("- params: { metric: {alpha?: number, beta?: number, k?: number} }")
    lines.append("- damping: { metric: number in [0,1] }  // 0=no damping, 0.3=30% damp of month-over-month change")
    lines.append("- adjustments: [ {label: 'Month YYYY'|'Qn YYYY', metric: string, type: 'percent'|'absolute', value: number} ]")
    lines.append("")
    lines.append("Recent monthly snapshot (last up to 6):")
    for i, m in enumerate(snap.get("months", [])):
        parts = []
        for mk in ["revenue","cost","profit","headcount","utilization","leave_days","vacation_days","holiday_days","working_days","total_hours"]:
            arr = snap.get(mk)
            if arr and i < len(arr):
                val = arr[i]
                if mk in ("revenue","cost","profit"):
                    parts.append(f"{mk}={val:.2f}")
                elif mk in ("utilization",):
                    parts.append(f"{mk}={val:.1f}")
                else:
                    parts.append(f"{mk}={val}")
        lines.append(f"- {m}: " + ", ".join(parts))
    lines.append("")
    lines.append("User question: " + (question or "").strip())
    lines.append("Now respond with strict JSON only.")
    prompt = "\n".join(lines)
    try:
        raw = _ollama_generate(prompt, timeout=60)
        cfg = _extract_json(raw)
        if isinstance(cfg, dict):
            return cfg
    except Exception as exc:
        print(f"[forecast_] LLM advice failed: {exc}")
    return None


def _guided_method_for(metric: str, cfg: Optional[dict]) -> Tuple[str, dict, float]:
    if not cfg:
        return ("auto", {}, 0.0)
    method = (cfg.get("method", {}) or {}).get(metric)
    params = (cfg.get("params", {}) or {}).get(metric) or {}
    damping = (cfg.get("damping", {}) or {}).get(metric) or 0.0
    try:
        damping = float(damping)
    except Exception:
        damping = 0.0
    return (method or "auto", params if isinstance(params, dict) else {}, max(0.0, min(1.0, damping)))


def _forecast_values_with_guidance(values: List[float], steps: int, metric: str, cfg: Optional[dict]) -> List[float]:
    method, params, damping = _guided_method_for(metric, cfg)
    if method == "holt":
        arr = _holt_linear_forecast(values, steps, float(params.get("alpha", 0.5)), float(params.get("beta", 0.3)))
    elif method == "sma":
        arr = _sma_forecast(values, steps, int(params.get("k", 3)))
    elif method == "lin":
        arr = _lin_forecast(values, steps)
    else:
        arr = _forecast_values(values, steps)
    # Apply damping to month-over-month changes
    if damping > 0 and arr:
        out = [arr[0]]
        for i in range(1, len(arr)):
            delta = arr[i] - arr[i-1]
            out.append(arr[i-1] + delta * (1.0 - damping))
        arr = out
    return arr


def _build_adjustment_map(cfg: Optional[dict]) -> Dict[str, Dict[str, dict]]:
    amap: Dict[str, Dict[str, dict]] = {}
    if not cfg:
        return amap
    adjs = cfg.get("adjustments") or []
    if not isinstance(adjs, list):
        return amap
    for it in adjs:
        try:
            label = str(it.get("label") or "").strip()
            metric = str(it.get("metric") or "").strip().lower()
            atype = str(it.get("type") or "percent").strip().lower()
            val = float(it.get("value"))
            if not label or not metric:
                continue
            amap.setdefault(label, {})[metric] = {"type": atype, "value": val}
        except Exception:
            continue
    return amap


def _apply_adjustment(value: float, metric: str, date_label: str, adj_map: Dict[str, Dict[str, dict]]) -> float:
    v = float(value)
    # month-level
    for lab in (date_label,):
        m = adj_map.get(lab, {})
        if metric in m:
            a = m[metric]
            if a.get("type") == "absolute":
                v = v + float(a.get("value", 0.0))
            else:  # percent
                v = v * (1.0 + float(a.get("value", 0.0)) / 100.0)
    return v


def _select_metrics(question: str) -> List[str]:
    q = (question or "").lower()
    sel: List[str] = []
    if any(w in q for w in ["revenue", "turnover", "sales"]):
        sel.append("revenue")
    if any(w in q for w in ["cost", "costs", "expense", "expenses"]):
        sel.append("cost")
    if "profit" in q or "margin" in q:
        sel.append("profit")
    if any(w in q for w in ["headcount", "head count", "employees", "employee count"]):
        sel.append("headcount")
    if any(w in q for w in ["utilization", "utilisation", "util"]):
        sel.append("utilization")
    if any(w in q for w in ["leave", "leaves", "pto", "time off", "timeoff"]):
        sel.append("leave_days")
    if any(w in q for w in ["vacation", "vacations"]):
        sel.append("vacation_days")
    if any(w in q for w in ["holiday", "holidays"]):
        sel.append("holiday_days")
    if any(w in q for w in ["working days", "workingdays", "workdays", "work days"]):
        sel.append("working_days")
    if any(w in q for w in ["hours", "work hours", "total hours", "actual hours"]):
        sel.append("total_hours")
    return sel


def _month_num(name: str) -> Optional[int]:
    try:
        return dt.datetime.strptime(name, "%B").month
    except ValueError:
        try:
            return dt.datetime.strptime(name, "%b").month
        except ValueError:
            return None


def _find_target_months(question: str, base: dt.date) -> List[dt.date]:
    if not question:
        return []
    pattern = r"(?i)\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b(?:\s*[-/ ]?\s*(\d{2,4}))?"
    results: List[dt.date] = []
    year_cursor = base.year
    prev_month = base.month
    for mo in re.finditer(pattern, question):
        mon_name = mo.group(1)
        yr = mo.group(2)
        mnum = _month_num(mon_name)
        if mnum is None:
            continue
        if yr:
            y = int(("20" + yr) if len(yr) == 2 else yr)
            year_cursor = y
        else:
            if not results:
                y = year_cursor + (1 if mnum <= base.month else 0)
                year_cursor = y
            else:
                if mnum < prev_month:
                    year_cursor += 1
                y = year_cursor
        prev_month = mnum
        try:
            results.append(dt.date(int(y), int(mnum), 1))
        except Exception:
            continue
    return results


def _select_months_count(question: str) -> Optional[int]:
    if not question:
        return None
    m = re.search(r"(?i)next\s+(\d{1,2})\s*months?\b", question)
    if m:
        try:
            n = int(m.group(1))
            return n if n >= 1 else None
        except Exception:
            return None
    m2 = re.search(r"(?i)next\s+(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s*months?\b", question)
    if m2:
        word = m2.group(1).lower()
        mapping = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
        }
        return mapping.get(word)
    return None


def _select_quarters_count(question: str) -> Optional[int]:
    if not question:
        return None
    # Singular form: "next quarter"
    if re.search(r"(?i)\bnext\s+quarter\b", question):
        return 1
    # Numeric form: next 2 quarters
    m = re.search(r"(?i)next\s+(\d{1,2})\s*(qtrs?|quarters?)\b", question)
    if m:
        try:
            n = int(m.group(1))
            return n if n >= 1 else None
        except Exception:
            return None
    # Word form: next one|two|three|four quarters
    m2 = re.search(r"(?i)next\s+(one|two|three|four)\s*(qtrs?|quarters?)\b", question)
    if m2:
        mapping = {"one": 1, "two": 2, "three": 3, "four": 4}
        return mapping.get(m2.group(1).lower())
    return None


def _find_target_quarter(question: str) -> Optional[Tuple[int, int, dt.date, str]]:
    if not question:
        return None
    m = re.search(r"(?i)\bq([1-4])\s*[- ]?\s*(\d{2,4})\b", question)
    if not m:
        return None
    qn = int(m.group(1))
    y = m.group(2)
    if len(y) == 2:
        y = "20" + y
    year = int(y)
    month = (qn - 1) * 3 + 1
    start = dt.date(year, month, 1)
    label = f"Q{qn} {year}"
    return (qn, year, start, label)


def _distinct_values(records: List[dict], key: str) -> List[str]:
    vals = set()
    for r in records:
        v = (r.get(key) or "").strip()
        if v:
            vals.add(v)
    return sorted(vals)


def _detect_scope(question: str, records: List[dict]) -> Tuple[str, List[str]]:
    q = (question or "").lower()
    projects = _distinct_values(records, "project")
    employees = _distinct_values(records, "employee")

    target_projects: List[str] = []
    target_employees: List[str] = []

    ask_proj = any(w in q for w in ["per project", "by project", "each project", "all projects", "projects"]) or "project:" in q
    ask_emp = any(w in q for w in ["per employee", "by employee", "each employee", "all employees", "employees", "staff", "resources"]) or "employee:" in q

    for p in projects:
        if p and p.lower() in q:
            target_projects.append(p)
    for e in employees:
        if e and e.lower() in q:
            target_employees.append(e)

    if ask_proj or target_projects:
        if not target_projects and ask_proj:
            return ("project", [])
        return ("project", sorted(set(target_projects)))

    if ask_emp or target_employees:
        if not target_employees and ask_emp:
            return ("employee", [])
        return ("employee", sorted(set(target_employees)))

    return ("overall", [])


def _norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _extract_requested_projects(question: str) -> List[str]:
    if not question:
        return []
    out: List[str] = []
    # Capture phrases like 'project Maersk', 'project: Astral', 'for project Optium'
    for m in re.finditer(r"(?i)\bproject\s*[:\-]?\s*([a-z0-9 .&/_-]{1,60})", question):
        raw = (m.group(1) or "").strip()
        if not raw:
            continue
        # Trim trailing qualifiers
        stop_words = [
            " for ", " in ", " by ", " with ", " on ", " during ", " next ",
            " revenue", " cost", " profit", " utilisation", " utilization",
            " headcount", " leaves", " vacation", " holidays", " hours", " working",
            ",", "|", "\n"
        ]
        low = " " + raw.lower() + " "
        cuts = [low.find(tok) for tok in stop_words if low.find(tok) >= 0]
        if cuts:
            cut = min(cuts)
            # remove the leading added space and cut position
            cleaned = low[1:cut].strip(" .,:;|-")
        else:
            cleaned = raw.strip(" .,:;|-")
        if cleaned:
            out.append(cleaned)
    return out


def _lin_forecast(values: List[float], steps: int) -> List[float]:
    clean_idx = [i for i, v in enumerate(values) if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean_idx:
        return []
    y = [values[i] for i in clean_idx]
    x = list(range(len(y)))
    n = len(y)
    if n == 1:
        return [y[-1]] * steps
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    den = sum((xi - mean_x) ** 2 for xi in x)
    b = (num / den) if den > 0 else 0.0
    a = mean_y - b * mean_x
    out = []
    for k in range(steps):
        out.append(a + b * (len(clean_idx) + k))
    return out


def _sma_forecast(values: List[float], steps: int, k: int = 3) -> List[float]:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return []
    if len(clean) < k:
        avg = sum(clean) / len(clean)
        return [avg] * steps
    avg = sum(clean[-k:]) / k
    return [avg] * steps


def _holt_linear_forecast(values: List[float], steps: int, alpha: float = 0.5, beta: float = 0.3) -> List[float]:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    if not clean:
        return []
    if len(clean) == 1:
        return [clean[-1]] * steps
    l = clean[0]
    b = clean[1] - clean[0]
    for t in range(1, len(clean)):
        y = clean[t]
        lt = alpha * y + (1 - alpha) * (l + b)
        bt = beta * (lt - l) + (1 - beta) * b
        l, b = lt, bt
    return [l + (i + 1) * b for i in range(steps)]


def _rmse(actual: List[float], pred: List[float]) -> float:
    pairs = [(a, p) for a, p in zip(actual, pred) if not (math.isnan(a) or math.isnan(p))]
    if not pairs:
        return float("inf")
    se = [(a - p) ** 2 for a, p in pairs]
    return math.sqrt(sum(se) / len(se))


def _choose_model(values: List[float]) -> Tuple[str, dict]:
    clean = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(clean)
    if n <= 2:
        return ("sma", {"k": max(1, n)})
    # Linear fit
    try:
        x = list(range(n))
        mean_x = sum(x) / n
        mean_y = sum(clean) / n
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, clean))
        den = sum((xi - mean_x) ** 2 for xi in x)
        b = (num / den) if den > 0 else 0.0
        a = mean_y - b * mean_x
        lin_fit = [a + b * xi for xi in x]
        lin_rmse = _rmse(clean, lin_fit)
    except Exception:
        lin_rmse = float("inf")
    # SMA k=3
    if n >= 3:
        sma_fit = [clean[0], clean[1]] + [sum(clean[i-3:i]) / 3 for i in range(3, n)]
        sma_rmse = _rmse(clean[2:], sma_fit[2:])
    else:
        sma_rmse = float("inf")
    # Holt grid
    holt_best = (float("inf"), 0.5, 0.3)
    for a in (0.2, 0.5, 0.8):
        for b_ in (0.2, 0.5, 0.8):
            l = clean[0]
            bt = clean[1] - clean[0]
            preds = [clean[0]]
            for t in range(1, n):
                preds.append(l + bt)
                y = clean[t]
                lt = a * y + (1 - a) * (l + bt)
                bt = b_ * (lt - l) + (1 - b_) * bt
                l = lt
            rmse = _rmse(clean[1:], preds[1:])
            if rmse < holt_best[0]:
                holt_best = (rmse, a, b_)
    holt_rmse, a_best, b_best = holt_best
    best = min(("lin", lin_rmse), ("sma", sma_rmse), ("holt", holt_rmse), key=lambda x: x[1])
    if best[0] == "holt":
        return ("holt", {"alpha": a_best, "beta": b_best})
    if best[0] == "sma":
        return ("sma", {"k": 3})
    return ("lin", {})


def _forecast_values(values: List[float], steps: int) -> List[float]:
    method, params = _choose_model(values)
    if method == "holt":
        return _holt_linear_forecast(values, steps, params.get("alpha", 0.5), params.get("beta", 0.3))
    if method == "sma":
        return _sma_forecast(values, steps, params.get("k", 3))
    return _lin_forecast(values, steps)


def _series_from_monthly_overall(records: List[dict]) -> Dict[str, List[float]]:
    monthly = build_monthly(records)
    months = get_months_available(records)
    rev = []
    cst = []
    pft = []
    hc = []
    lv = []
    vac = []
    hol = []
    work = []
    hrs = []
    util_map: Dict[str, Dict[str, float]] = {}
    for r in records:
        m = r.get("month")
        e = r.get("employee")
        u = r.get("utilisation_pct")
        if not m or not e or u is None:
            continue
        util_map.setdefault(m, {})[e] = float(u)
    for m in months:
        v = monthly.get(m) or {}
        rev.append(float(v.get("total_revenue", 0) or 0))
        cst.append(float(v.get("total_cost", 0) or 0))
        pft.append(float(v.get("total_profit", 0) or 0))
        hc.append(float(v.get("employees", 0) or 0))
        lv.append(float(v.get("leave_days", 0) or 0))
        vac.append(float(v.get("vacation_days", 0) or 0))
        hol.append(float(v.get("holiday_days", 0) or 0))
        work.append(float(v.get("working_days", 0) or 0))
        hrs.append(float(v.get("total_hours", 0) or 0))
    utl = []
    for m in months:
        per_emp = util_map.get(m) or {}
        if per_emp:
            vals = list(per_emp.values())
            utl.append(sum(vals) / len(vals))
        else:
            utl.append(float("nan"))
    return {
        "months": months,
        "revenue": rev,
        "cost": cst,
        "profit": pft,
        "headcount": hc,
        "utilization": utl,
        "leave_days": lv,
        "vacation_days": vac,
        "holiday_days": hol,
        "working_days": work,
        "total_hours": hrs,
    }


def _series_by_group(records: List[dict], group_key: str) -> Dict[str, Dict[str, List[float]]]:
    assert group_key in ("project", "employee")
    months = get_months_available(records)
    agg: Dict[str, Dict[str, dict]] = {}
    util_acc: Dict[str, Dict[str, Dict[str, float]]] = {}
    for r in records:
        g = (r.get(group_key) or "Unknown").strip() or "Unknown"
        m = r.get("month") or "Unknown"
        a = agg.setdefault(g, {}).setdefault(m, {
            "revenue": 0.0, "cost": 0.0, "profit": 0.0,
            "emp_set": set(),
            "leave_days": 0.0, "vacation_days": 0.0, "holiday_days": 0.0,
            "working_days": 0.0, "total_hours": 0.0,
        })
        a["revenue"] += r.get("revenue") or 0.0
        a["cost"] += r.get("cost") or 0.0
        a["profit"] += r.get("profit") or 0.0
        a["total_hours"] += r.get("actual_hours") or 0.0
        a["working_days"] += r.get("working_days") or 0.0
        a["leave_days"] += r.get("leave_days") or 0.0
        a["vacation_days"] += r.get("vacation_days") or 0.0
        a["holiday_days"] += r.get("holiday_days") or 0.0
        emp = (r.get("employee") or "").strip()
        if emp:
            a["emp_set"].add(emp)
        u = r.get("utilisation_pct")
        if u is not None:
            util_acc.setdefault(g, {}).setdefault(m, {})[emp or f"_r{len(util_acc)}"] = float(u)
    out: Dict[str, Dict[str, List[float]]] = {}
    for g, per_month in agg.items():
        rev = []
        cst = []
        pft = []
        hc = []
        lv = []
        vac = []
        hol = []
        work = []
        hrs = []
        utl = []
        for m in months:
            v = per_month.get(m) or {"revenue": 0.0, "cost": 0.0, "profit": 0.0, "emp_set": set(),
                                     "leave_days": 0.0, "vacation_days": 0.0, "holiday_days": 0.0,
                                     "working_days": 0.0, "total_hours": 0.0}
            rev.append(float(v["revenue"]))
            cst.append(float(v["cost"]))
            pft.append(float(v["profit"]))
            hc.append(float(len(v["emp_set"]) if group_key == "project" else (1 if len(v["emp_set"]) > 0 else 0)))
            lv.append(float(v["leave_days"]))
            vac.append(float(v["vacation_days"]))
            hol.append(float(v["holiday_days"]))
            work.append(float(v["working_days"]))
            hrs.append(float(v["total_hours"]))
            util_map = util_acc.get(g, {}).get(m, {})
            if util_map:
                vals = list(util_map.values())
                utl.append(sum(vals) / len(vals))
            else:
                utl.append(float("nan"))
        out[g] = {
            "months": months,
            "revenue": rev,
            "cost": cst,
            "profit": pft,
            "headcount": hc,
            "utilization": utl,
            "leave_days": lv,
            "vacation_days": vac,
            "holiday_days": hol,
            "working_days": work,
            "total_hours": hrs,
        }
    return out


def _std_dev(vals: List[float]) -> float:
    clean = [float(v) for v in vals if isinstance(v, (int, float)) and not math.isnan(v)]
    n = len(clean)
    if n < 2:
        return 0.0
    m = sum(clean) / n
    return math.sqrt(sum((x - m) ** 2 for x in clean) / (n - 1))


def _estimate_util_band_pp(series: Dict[str, List[float]], take: int = 6) -> float:
    arr = series.get("utilization") or []
    hist = [v for v in arr if isinstance(v, (int, float)) and not math.isnan(v)]
    if not hist:
        return 2.0
    hist = hist[-take:]
    sd = _std_dev(hist)
    try:
        return max(0.5, min(5.0, round(sd, 1)))
    except Exception:
        return 2.0


def _estimate_sum_band_pct(series: Dict[str, List[float]], key: str, take: int = 6, horizon_months: int = 3) -> float:
    arr = series.get(key) or []
    hist = [v for v in arr if isinstance(v, (int, float)) and not math.isnan(v)]
    if not hist:
        return 5.0
    hist = hist[-take:]
    sd_m = _std_dev(hist)
    mean_m = (sum(hist) / len(hist)) if hist else 0.0
    if mean_m <= 1e-6:
        return 10.0
    # CV adjustment for sum over horizon: sd_sum/(mean_sum) = (sqrt(h)*sd_m)/(h*mean_m) = (sd_m/(mean_m*sqrt(h)))
    h = max(1, int(horizon_months))
    band = (sd_m / (mean_m * math.sqrt(h))) * 100.0
    try:
        return max(1.0, min(30.0, round(band, 1)))
    except Exception:
        return 10.0


def _format_month_line(idx: int, label: str, metrics: List[str], fmap: Dict[str, List[float]], step_idx: int,
                       group_type: Optional[str] = None, group_name: Optional[str] = None,
                       adj_map: Optional[Dict[str, Dict[str, dict]]] = None) -> str:
    parts: List[str] = []
    adj_map = adj_map or {}
    if "revenue" in metrics and fmap.get("revenue"):
        v = _apply_adjustment(fmap['revenue'][step_idx], 'revenue', label, adj_map)
        parts.append(f"Revenue: {_fmt_money(_clamp_nonneg(v))}")
    if "cost" in metrics and fmap.get("cost"):
        v = _apply_adjustment(fmap['cost'][step_idx], 'cost', label, adj_map)
        parts.append(f"Cost: {_fmt_money(_clamp_nonneg(v))}")
    if "profit" in metrics and fmap.get("profit"):
        v = _apply_adjustment(fmap['profit'][step_idx], 'profit', label, adj_map)
        parts.append(f"Profit: {_fmt_money(_clamp_nonneg(v))}")
    if "headcount" in metrics and fmap.get("headcount"):
        v = _apply_adjustment(fmap['headcount'][step_idx], 'headcount', label, adj_map)
        parts.append(f"Headcount: {_fmt_int(_clamp_nonneg(v))}")
    if "utilization" in metrics and fmap.get("utilization"):
        v = _apply_adjustment(fmap['utilization'][step_idx], 'utilization', label, adj_map)
        parts.append(f"Utilization: {_fmt_pct(_clamp_pct(v))}")
    if "leave_days" in metrics and fmap.get("leave_days"):
        v = _apply_adjustment(fmap['leave_days'][step_idx], 'leave_days', label, adj_map)
        parts.append(f"Leaves: {_fmt_int(_clamp_nonneg(v))}")
    if "vacation_days" in metrics and fmap.get("vacation_days"):
        v = _apply_adjustment(fmap['vacation_days'][step_idx], 'vacation_days', label, adj_map)
        parts.append(f"Vacation: {_fmt_int(_clamp_nonneg(v))}")
    if "holiday_days" in metrics and fmap.get("holiday_days"):
        v = _apply_adjustment(fmap['holiday_days'][step_idx], 'holiday_days', label, adj_map)
        parts.append(f"Holidays: {_fmt_int(_clamp_nonneg(v))}")
    if "working_days" in metrics and fmap.get("working_days"):
        v = _apply_adjustment(fmap['working_days'][step_idx], 'working_days', label, adj_map)
        parts.append(f"WorkingDays: {_fmt_int(_clamp_nonneg(v))}")
    if "total_hours" in metrics and fmap.get("total_hours"):
        v = _apply_adjustment(fmap['total_hours'][step_idx], 'total_hours', label, adj_map)
        parts.append(f"Hours: {_fmt_int(_clamp_nonneg(v))}")
    lead = f"- Month: {label}"
    if group_type and group_name:
        lead += f" | {group_type.capitalize()}: {group_name}"
    return lead + " | " + " | ".join(parts)


def _format_quarter_line(idx: int, qlabel: str, metrics: List[str], qagg: Dict[str, float],
                         group_type: Optional[str] = None, group_name: Optional[str] = None,
                         util_band_pp: Optional[float] = None,
                         rev_band_pct: Optional[float] = None,
                         profit_band_pct: Optional[float] = None,
                         hours_band_pct: Optional[float] = None) -> str:
    parts: List[str] = []
    if "revenue" in metrics and "revenue" in qagg:
        rtxt = f"Revenue: {_fmt_money(_clamp_nonneg(qagg['revenue']))}"
        if rev_band_pct is not None:
            rtxt += f" (±{rev_band_pct:.1f}%)"
        parts.append(rtxt)
    if "cost" in metrics and "cost" in qagg:
        parts.append(f"Cost: {_fmt_money(_clamp_nonneg(qagg['cost']))}")
    if "profit" in metrics and "profit" in qagg:
        ptxt = f"Profit: {_fmt_money(_clamp_nonneg(qagg['profit']))}"
        if profit_band_pct is not None:
            ptxt += f" (±{profit_band_pct:.1f}%)"
        parts.append(ptxt)
    if "headcount" in metrics and "headcount" in qagg:
        parts.append(f"Headcount: {_fmt_int(_clamp_nonneg(qagg['headcount']))} (avg)")
    if "utilization" in metrics and "utilization" in qagg:
        if util_band_pp is not None:
            parts.append(f"Utilization: {_fmt_pct(_clamp_pct(qagg['utilization']))} (avg, ±{util_band_pp:.1f}pp)")
        else:
            parts.append(f"Utilization: {_fmt_pct(_clamp_pct(qagg['utilization']))} (avg)")
    if "leave_days" in metrics and "leave_days" in qagg:
        parts.append(f"Leaves: {_fmt_int(_clamp_nonneg(qagg['leave_days']))}")
    if "vacation_days" in metrics and "vacation_days" in qagg:
        parts.append(f"Vacation: {_fmt_int(_clamp_nonneg(qagg['vacation_days']))}")
    if "holiday_days" in metrics and "holiday_days" in qagg:
        parts.append(f"Holidays: {_fmt_int(_clamp_nonneg(qagg['holiday_days']))}")
    if "working_days" in metrics and "working_days" in qagg:
        parts.append(f"WorkingDays: {_fmt_int(_clamp_nonneg(qagg['working_days']))}")
    if "total_hours" in metrics and "total_hours" in qagg:
        htxt = f"Hours: {_fmt_int(_clamp_nonneg(qagg['total_hours']))}"
        if hours_band_pct is not None:
            htxt += f" (±{hours_band_pct:.1f}%)"
        parts.append(htxt)
    lead = f"- Quarter: {qlabel}"
    if group_type and group_name:
        lead += f" | {group_type.capitalize()}: {group_name}"
    return lead + " | " + " | ".join(parts)


def _map_llm_metrics(llm_metrics: List[str]) -> List[str]:
    """Map LLM metric names to internal metric keys."""
    mapping = {
        "revenue": "revenue", "profit": "profit", "cost": "cost",
        "leaves": "leave_days", "leave": "leave_days", "leave_days": "leave_days",
        "vacation": "vacation_days", "vacations": "vacation_days", "vacation_days": "vacation_days",
        "holidays": "holiday_days", "holiday": "holiday_days", "holiday_days": "holiday_days",
        "hours": "total_hours", "total_hours": "total_hours", "work_hours": "total_hours",
        "utilization": "utilization", "utilisation": "utilization",
        "headcount": "headcount", "head_count": "headcount",
        "working_days": "working_days", "workdays": "working_days",
    }
    out = []
    for m in llm_metrics:
        key = mapping.get(m.lower().strip())
        if key and key not in out:
            out.append(key)
    return out


def _parse_llm_explicit_periods(periods: List[str], base: dt.date) -> Tuple[List[dt.date], Optional[Tuple]]:
    """Parse explicit periods from LLM output into month dates and/or quarter tuple."""
    explicit_months: List[dt.date] = []
    explicit_quarter = None
    for p in periods:
        p = (p or "").strip()
        if not p:
            continue
        # Try quarter format: Q1 2026, Q2 2026, etc.
        qm = re.match(r"(?i)q([1-4])\s*(\d{4})", p)
        if qm:
            qn = int(qm.group(1))
            yr = int(qm.group(2))
            start_month = dt.date(yr, (qn - 1) * 3 + 1, 1)
            explicit_quarter = (qn, yr, start_month, f"Q{qn} {yr}")
            continue
        # Try month format: July 2026, Jul 2026, etc.
        for fmt in ("%B %Y", "%b %Y", "%B-%Y", "%b-%Y"):
            try:
                d = dt.datetime.strptime(p, fmt).date().replace(day=1)
                if d not in explicit_months:
                    explicit_months.append(d)
                break
            except ValueError:
                continue
    return explicit_months, explicit_quarter


def try_answer_forecast(question: str, records: List[dict] = None) -> Optional[str]:
    q = (question or "").lower()
    recs = records if records is not None else GLOBAL_DATASET
    if not recs:
        return None
    months = get_months_available(recs)
    if not months:
        return "Insufficient historical data to estimate a forecast."
    last_label = months[-1]
    base = _parse_month_date(last_label) or dt.date.today().replace(day=1)

    # Collect known entities for LLM context
    known_projects = _distinct_values(recs, "project")
    known_emps = _distinct_values(recs, "employee")

    # --- LLM-based question parsing (primary) ---
    llm_parsed = _llm_parse_question(question, known_projects, known_emps)
    use_llm = llm_parsed and llm_parsed.get("is_forecast") is True

    if use_llm:
        # Extract parameters from LLM response
        llm_metrics = _map_llm_metrics(llm_parsed.get("metrics") or [])
        llm_scope = (llm_parsed.get("scope") or "overall").lower()
        llm_targets = llm_parsed.get("targets") or []
        llm_horizon_type = (llm_parsed.get("horizon_type") or "").lower()
        llm_horizon_value = llm_parsed.get("horizon_value")
        llm_explicit_periods = llm_parsed.get("explicit_periods") or []

        # Parse explicit periods
        explicit_months, explicit_quarter = _parse_llm_explicit_periods(llm_explicit_periods, base)

        # Determine months_n / quarters_n from LLM
        months_n = None
        quarters_n = None
        if llm_horizon_type == "months" and llm_horizon_value:
            try:
                months_n = int(llm_horizon_value)
            except Exception:
                months_n = 1
        elif llm_horizon_type == "quarters" and llm_horizon_value:
            try:
                quarters_n = int(llm_horizon_value)
            except Exception:
                quarters_n = 1
        elif llm_horizon_type == "explicit_months":
            pass  # explicit_months already set
        elif llm_horizon_type == "explicit_quarter":
            pass  # explicit_quarter already set

        # Default horizon if none specified
        if not months_n and not quarters_n and not explicit_months and not explicit_quarter:
            months_n = 1  # default to next month

        # Metrics
        metrics = llm_metrics if llm_metrics else ["revenue", "profit"]

        # Scope and targets
        scope = llm_scope if llm_scope in ("overall", "project", "employee") else "overall"
        targets: List[str] = []

        # Validate and match targets
        if scope == "project" and llm_targets:
            kp_norm = {_norm_key(p): p for p in known_projects}
            matched: List[str] = []
            for t in llm_targets:
                nn = _norm_key(t)
                if nn in kp_norm:
                    matched.append(kp_norm[nn])
                else:
                    for kn, orig in kp_norm.items():
                        if nn and (nn in kn or kn in nn):
                            matched.append(orig)
            matched = sorted(set(matched))
            if not matched:
                return f"Project not found\n- No matching project for '{', '.join(llm_targets)}'"
            targets = matched
        elif scope == "employee" and llm_targets:
            ke_norm = {_norm_key(e): e for e in known_emps}
            matched_e: List[str] = []
            for t in llm_targets:
                nn = _norm_key(t)
                if nn in ke_norm:
                    matched_e.append(ke_norm[nn])
                else:
                    for kn, orig in ke_norm.items():
                        if nn and (nn in kn or kn in nn):
                            matched_e.append(orig)
            matched_e = sorted(set(matched_e))
            if not matched_e:
                return f"Employee not found\n- No matching employee for '{', '.join(llm_targets)}'"
            targets = matched_e

    else:
        # --- Regex-based fallback (when LLM unavailable or says not a forecast) ---
        months_n = _select_months_count(q)
        quarters_n = _select_quarters_count(q)
        explicit_quarter = _find_target_quarter(q)
        explicit_months = _find_target_months(q, base)

        intent = bool(months_n or quarters_n or explicit_quarter or explicit_months) \
            or any(w in q for w in ["forecast", "predict", "projection", "estimate"]) \
            or any(w in q for w in ["expected", "projected", "future", "futuristic"]) \
            or any(p in q for p in ["next month", "next quarter", "next months", "upcoming months", "coming months"]) \
            or bool(re.search(r"(?i)next\s+\d+\s*(months?|qtrs?|quarters?)\b", q))
        if not intent:
            return None

        metrics = _select_metrics(question)
        if not metrics:
            metrics = ["revenue", "profit", "headcount", "utilization"]

        scope, targets = _detect_scope(q, recs)

        # Extra: if a concrete 'project X' was asked, try to map to known projects; if no match, stop early
        req_projects = _extract_requested_projects(question)
        if req_projects:
            kp_norm = {_norm_key(p): p for p in known_projects}
            matched: List[str] = []
            for rp in req_projects:
                nn = _norm_key(rp)
                if nn in kp_norm:
                    matched.append(kp_norm[nn])
                else:
                    for kn, orig in kp_norm.items():
                        if nn and (nn in kn or kn in nn):
                            matched.append(orig)
            matched = sorted(set(matched))
            if not matched:
                return f"Project not found\n- No matching project for '{', '.join(req_projects)}'"
            scope = "project"
            targets = matched

        # Extra: explicit employee guard
        req_employees = _extract_requested_employees(question)
        if req_employees:
            ke_norm = {_norm_key(e): e for e in known_emps}
            matched_e: List[str] = []
            for re_emp in req_employees:
                nn = _norm_key(re_emp)
                if nn in ke_norm:
                    matched_e.append(ke_norm[nn])
                else:
                    for kn, orig in ke_norm.items():
                        if nn and (nn in kn or kn in nn):
                            matched_e.append(orig)
            matched_e = sorted(set(matched_e))
            if not matched_e:
                return f"Employee not found\n- No matching employee for '{', '.join(req_employees)}'"
            scope = "employee"
            targets = matched_e

        # Extra: freeform 'for <name>' guard
        free_targets = _extract_freeform_targets(question)
        if len(free_targets) == 1 and not (req_projects or req_employees):
            token = free_targets[0]
            kp_norm = {_norm_key(p): p for p in known_projects}
            ke_norm = {_norm_key(e): e for e in known_emps}
            nn = _norm_key(token)
            exists = nn in kp_norm or nn in ke_norm
            if not exists:
                exists = any(nn in k for k in kp_norm.keys()) or any(nn in k for k in ke_norm.keys())
            if not exists:
                return f"Target not found\n- '{token}' does not match any project or employee in the dataset."

    # LLM guidance for forecasting method (overall snapshot)
    overall_series = _build_overall_series(recs)
    cfg = _llm_advise_settings(question, months, overall_series)
    adj_map = _build_adjustment_map(cfg)

    def _forecast_for_series(series: Dict[str, List[float]], steps: int) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for key in ["revenue", "cost", "profit", "headcount", "utilization", "leave_days", "vacation_days", "holiday_days", "working_days", "total_hours"]:
            vals = series.get(key, [])
            if not vals:
                continue
            out[key] = _forecast_values_with_guidance(vals, steps, key, cfg)
        return out

    if scope == "overall":
        series_map = {"__overall__": _series_from_monthly_overall(recs)}
        group_type = None
    elif scope == "project":
        all_series = _series_by_group(recs, "project")
        if targets:
            series_map = {k: v for k, v in all_series.items() if k in set(targets)}
            if not series_map:
                return "Project not found\n- No matching project in dataset for the requested query."
        else:
            series_map = all_series
        group_type = "project"
    else:
        all_series = _series_by_group(recs, "employee")
        if targets:
            series_map = {k: v for k, v in all_series.items() if k in set(targets)}
            if not series_map:
                return "Employee not found\n- No matching employee in dataset for the requested query."
        else:
            series_map = all_series
        group_type = "employee"

    lines: List[str] = []

    if explicit_months:
        deltas = [max(1, (d.year - base.year) * 12 + (d.month - base.month)) for d in explicit_months]
        max_steps = max(deltas)
        for gname, ser in series_map.items():
            fmap = _forecast_for_series(ser, max_steps)
            for idx, d in enumerate(explicit_months, start=1):
                step = max(1, (d.year - base.year) * 12 + (d.month - base.month))
                label = d.strftime("%B %Y")
                lines.append(_format_month_line(idx, label, metrics, fmap, step - 1, None if scope == "overall" else group_type, None if scope == "overall" else gname, adj_map))
        header = ", ".join([d.strftime("%B %Y") for d in explicit_months])
        return f"Estimate — {header} —\n" + "\n".join(lines)

    if explicit_quarter is not None:
        qn, year, start_month, qlabel = explicit_quarter
        delta = (start_month.year - base.year) * 12 + (start_month.month - base.month)
        steps = max(1, delta + 3)
        for gname, ser in series_map.items():
            fmap = _forecast_for_series(ser, steps)
            # Build explicit quarter months and aggregate with adjustments
            months_in_q = [start_month, _add_months(start_month, 1), _add_months(start_month, 2)]
            idxs = [(m.year - base.year) * 12 + (m.month - base.month) - 1 for m in months_in_q]
            q_agg: Dict[str, float] = {}
            for key in ["revenue", "cost", "profit", "leave_days", "vacation_days", "holiday_days", "working_days", "total_hours"]:
                arr = fmap.get(key) or []
                total = 0.0
                for i, mdate in zip(idxs, months_in_q):
                    if 0 <= i < len(arr):
                        mlabel = _month_label_for(mdate)
                        v = _apply_adjustment(arr[i], key, mlabel, adj_map)
                        # Also allow quarter-level label adjustments
                        v = _apply_adjustment(v, key, qlabel, adj_map)
                        total += _clamp_nonneg(v)
                if total != 0.0:
                    q_agg[key] = total
            if fmap.get("headcount"):
                vals = []
                for i, mdate in zip(idxs, months_in_q):
                    if 0 <= i < len(fmap["headcount"]):
                        mlabel = _month_label_for(mdate)
                        v = _apply_adjustment(fmap["headcount"][i], "headcount", mlabel, adj_map)
                        v = _apply_adjustment(v, "headcount", qlabel, adj_map)
                        vals.append(max(0.0, v))
                if vals:
                    q_agg["headcount"] = sum(vals) / len(vals)
            if fmap.get("utilization"):
                vals = []
                for i, mdate in zip(idxs, months_in_q):
                    if 0 <= i < len(fmap["utilization"]):
                        mlabel = _month_label_for(mdate)
                        v = _apply_adjustment(fmap["utilization"][i], "utilization", mlabel, adj_map)
                        v = _apply_adjustment(v, "utilization", qlabel, adj_map)
                        if not math.isnan(v):
                            vals.append(v)
                q_agg["utilization"] = (sum(vals) / len(vals)) if vals else float("nan")
            util_band = _estimate_util_band_pp(ser)
            util_band_q = round(util_band / math.sqrt(3), 1)
            rev_band = _estimate_sum_band_pct(ser, "revenue", horizon_months=3)
            prof_band = _estimate_sum_band_pct(ser, "profit", horizon_months=3)
            hrs_band = _estimate_sum_band_pct(ser, "total_hours", horizon_months=3)
            lines.append(_format_quarter_line(1, qlabel, metrics, q_agg,
                                              None if scope == "overall" else group_type,
                                              None if scope == "overall" else gname,
                                              util_band_q, rev_band, prof_band, hrs_band))
        return f"Estimate — {qlabel} —\n" + "\n".join(lines)

    if quarters_n is not None and quarters_n >= 1:
        total_months = 3 * quarters_n
        for gname, ser in series_map.items():
            fmap = _forecast_for_series(ser, total_months)
            start_next = _add_months(base, 1)
            curr = start_next
            for qi in range(quarters_n):
                months_in_q = [curr, _add_months(curr, 1), _add_months(curr, 2)]
                idxs = [(m.year - base.year) * 12 + (m.month - base.month) - 1 for m in months_in_q]
                q_agg: Dict[str, float] = {}
                for key in ["revenue", "cost", "profit", "leave_days", "vacation_days", "holiday_days", "working_days", "total_hours"]:
                    arr = fmap.get(key) or []
                    total = 0.0
                    for i, mdate in zip(idxs, months_in_q):
                        if 0 <= i < len(arr):
                            mlabel = _month_label_for(mdate)
                            v = _apply_adjustment(arr[i], key, mlabel, adj_map)
                            v = _apply_adjustment(v, key, _quarter_label_for(curr), adj_map)
                            total += _clamp_nonneg(v)
                    if total != 0.0:
                        q_agg[key] = total
                if fmap.get("headcount"):
                    vals = []
                    for i, mdate in zip(idxs, months_in_q):
                        if 0 <= i < len(fmap["headcount"]):
                            mlabel = _month_label_for(mdate)
                            v = _apply_adjustment(fmap["headcount"][i], "headcount", mlabel, adj_map)
                            v = _apply_adjustment(v, "headcount", _quarter_label_for(curr), adj_map)
                            vals.append(max(0.0, v))
                    q_agg["headcount"] = sum(vals) / len(vals) if vals else 0.0
                if fmap.get("utilization"):
                    vals = []
                    for i, mdate in zip(idxs, months_in_q):
                        if 0 <= i < len(fmap["utilization"]):
                            mlabel = _month_label_for(mdate)
                            v = _apply_adjustment(fmap["utilization"][i], "utilization", mlabel, adj_map)
                            v = _apply_adjustment(v, "utilization", _quarter_label_for(curr), adj_map)
                            if not math.isnan(v):
                                vals.append(v)
                    q_agg["utilization"] = sum(vals) / len(vals) if vals else float("nan")
                qn2 = (curr.month - 1) // 3 + 1
                qlabel = f"Q{qn2} {curr.year}"
                util_band = _estimate_util_band_pp(ser)
                util_band_q = round(util_band / math.sqrt(3), 1)
                rev_band = _estimate_sum_band_pct(ser, "revenue", horizon_months=3)
                prof_band = _estimate_sum_band_pct(ser, "profit", horizon_months=3)
                hrs_band = _estimate_sum_band_pct(ser, "total_hours", horizon_months=3)
                lines.append(_format_quarter_line(qi + 1, qlabel, metrics, q_agg,
                                                  None if scope == "overall" else group_type,
                                                  None if scope == "overall" else gname,
                                                  util_band_q, rev_band, prof_band, hrs_band))
                curr = _add_months(curr, 3)
        header_txt = "Next quarter" if quarters_n == 1 else f"Next {quarters_n} quarters"
        return f"Estimate — {header_txt} —\n" + "\n".join(lines)

    steps = months_n if months_n is not None else 1
    for gname, ser in series_map.items():
        fmap = _forecast_for_series(ser, steps)
        for i in range(steps):
            label = _add_months(base, i + 1).strftime("%B %Y")
            lines.append(_format_month_line(i + 1, label, metrics, fmap, i, None if scope == "overall" else group_type, None if scope == "overall" else gname, adj_map))
    if steps == 1:
        return "Estimate — Next month —\n" + "\n".join(lines)
    return f"Estimate — Next {steps} months —\n" + "\n".join(lines)
